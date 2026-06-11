"""Bangla-Aware One-Hot Word Swap Attack — refactored from BanglaClassificationAugment."""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from destror.attacks.base import Attack, VictimFn
from destror.utils.seeding import set_seed

# Mask-fill models to try (in order of preference for Bangla)
_MASKER_MODELS = {
    "banglabert": "csebuetnlp/banglabert",
    "muril":      "google/muril-base-cased",
    "xlm-r":      "xlm-roberta-base",  # fallback / ablation
}

_TOP_K = 10          # mask-fill candidates per position
_MAX_BUDGET = 0.20   # 20 % token perturbation cap (revision.md §2.3)


class BanglaOneHotSwapAttack(Attack):
    """Word-swap attack with BanglaBERT/MuRIL mask filling and importance ranking.

    importance_mode="loo"      — leave-one-out (black-box, default)
    importance_mode="gradient" — gradient saliency (white-box, requires gradient_model)
    """

    name = "one_hot_swap"

    def __init__(
        self,
        victim: VictimFn,
        seed: int = 42,
        masker: str = "banglabert",
        top_k: int = _TOP_K,
        perturbation_budget: float = _MAX_BUDGET,
        similarity_threshold: float = 0.7,
        importance_mode: str = "loo",
        gradient_model=None,
        gradient_tokenizer=None,
    ):
        super().__init__(victim, seed)
        if masker not in _MASKER_MODELS:
            raise ValueError(f"masker must be one of {list(_MASKER_MODELS)}")
        self.masker = masker
        self.top_k = top_k
        self.perturbation_budget = perturbation_budget
        self.similarity_threshold = similarity_threshold
        self.importance_mode = importance_mode
        self.gradient_model = gradient_model
        self.gradient_tokenizer = gradient_tokenizer
        self._fill_pipeline = None

    def _load_masker(self):
        if self._fill_pipeline is not None:
            return
        from transformers import pipeline
        model_name = _MASKER_MODELS[self.masker]
        self._fill_pipeline = pipeline(
            "fill-mask",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            top_k=self.top_k,
        )

    def _tokenize(self, text: str) -> list[str]:
        try:
            from bnlp import BasicTokenizer
            return BasicTokenizer()(text)
        except Exception:
            return text.split()

    def _importance_scores(
        self, tokens: list[str], orig_pred: int | str, orig_conf: float
    ) -> list[tuple[int, float]]:
        """Rank token importance. Uses LOO (black-box) or gradient saliency (white-box)."""
        if self.importance_mode == "gradient" and self.gradient_model is not None:
            return self._gradient_importance(tokens, orig_pred)
        return self._loo_importance(tokens, orig_pred, orig_conf)

    def _loo_importance(
        self, tokens: list[str], orig_pred: int | str, orig_conf: float
    ) -> list[tuple[int, float]]:
        """Leave-one-out: score = confidence drop when token is removed."""
        scores: list[tuple[int, float]] = []
        for i in range(len(tokens)):
            masked = tokens[:i] + tokens[i + 1:]
            if not masked:
                continue
            pred, conf = self.victim(" ".join(masked))
            if pred != orig_pred:
                scores.append((i, float("inf")))
            else:
                scores.append((i, orig_conf - conf))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _gradient_importance(
        self, tokens: list[str], orig_pred: int | str
    ) -> list[tuple[int, float]]:
        """Gradient saliency: ||∂L/∂e_i|| × ||e_i|| (L1-norm of integrated gradient × embedding)."""
        model = self.gradient_model
        tokenizer = self.gradient_tokenizer
        if model is None or tokenizer is None:
            return [(i, 1.0) for i in range(len(tokens))]

        text = " ".join(tokens)
        label_id = model.config.label2id.get(str(orig_pred), 0)

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if next(model.parameters()).is_cuda:
            enc = {k: v.cuda() for k, v in enc.items()}

        embeddings = model.get_input_embeddings()(enc["input_ids"])
        embeddings = embeddings.detach().requires_grad_(True)

        # Forward pass with embedding hook
        outputs = model(inputs_embeds=embeddings, attention_mask=enc.get("attention_mask"))
        logits = outputs.logits
        loss = -logits[0, label_id]  # maximise confidence drop
        loss.backward()

        # Token saliency = L2 norm of gradient × embedding (input × gradient)
        grad = embeddings.grad[0]  # (seq_len, hidden)
        saliency = (grad * embeddings[0].detach()).norm(dim=-1)  # (seq_len,)

        # Map subword tokens back to whitespace tokens (approximate)
        subword_ids = enc["input_ids"][0].tolist()
        word_tokens = tokenizer.convert_ids_to_tokens(subword_ids)
        scores: list[tuple[int, float]] = []
        tok_idx, sal_buf = 0, []
        word_idx = 0
        for i, wt in enumerate(word_tokens):
            if wt in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
                continue
            sal_buf.append(saliency[i].item())
            # Heuristic: new word starts when token doesn't begin with continuation marker
            is_new = not (wt.startswith("##") or wt.startswith("▁") is False and i > 1)
            if is_new and sal_buf and word_idx < len(tokens):
                scores.append((word_idx, float(np.mean(sal_buf))))
                word_idx += 1
                sal_buf = [saliency[i].item()]

        if sal_buf and word_idx < len(tokens):
            scores.append((word_idx, float(np.mean(sal_buf))))

        # Pad remaining tokens with mean saliency
        mean_sal = float(np.mean([s for _, s in scores])) if scores else 0.0
        covered = {idx for idx, _ in scores}
        for i in range(len(tokens)):
            if i not in covered:
                scores.append((i, mean_sal))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def attack_one(
        self,
        text: str,
        orig_pred: int | str,
        orig_conf: float,
    ) -> tuple[str, int | str, float, int]:
        from destror.metrics.core import semantic_similarity

        set_seed(self.seed)
        self._load_masker()

        tokens = self._tokenize(text)
        if not tokens:
            return text, orig_pred, orig_conf, 1

        n_queries = 1  # already used for original prediction
        max_swaps = max(1, int(len(tokens) * self.perturbation_budget))

        ranked = self._importance_scores(tokens, orig_pred, orig_conf)
        if self.importance_mode == "loo":
            n_queries += len(tokens)  # leave-one-out queries to victim

        current_tokens = list(tokens)
        swaps_done = 0

        for idx, importance in ranked:
            if swaps_done >= max_swaps:
                break
            if idx >= len(current_tokens):
                continue

            # Build masked sentence for fill-mask
            masked_sent = " ".join(
                current_tokens[:idx] + ["<mask>"] + current_tokens[idx + 1:]
            )
            try:
                fills = self._fill_pipeline(masked_sent)
            except Exception:
                continue

            # Try each fill candidate
            for fill in fills:
                candidate_tokens = (
                    current_tokens[:idx] + [fill["token_str"].strip()] + current_tokens[idx + 1:]
                )
                candidate_text = " ".join(candidate_tokens)

                sim = semantic_similarity(text, candidate_text)
                if sim < self.similarity_threshold:
                    continue

                pred, conf = self.victim(candidate_text)
                n_queries += 1

                if pred != orig_pred:
                    return candidate_text, pred, conf, n_queries

                # Accept swap if it reduces confidence even without flipping
                if conf < orig_conf:
                    current_tokens[idx] = fill["token_str"].strip()
                    swaps_done += 1
                    orig_conf = conf
                    break

        final_text = " ".join(current_tokens)
        final_pred, final_conf = self.victim(final_text)
        n_queries += 1
        return final_text, final_pred, final_conf, n_queries
