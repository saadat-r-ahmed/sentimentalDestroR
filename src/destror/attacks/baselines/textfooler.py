"""TextFooler (Jin et al. 2020) — Bangla adaptation.

Key differences from original:
- USE → LaBSE for semantic constraint
- Counter-fitting vectors → BanglaBERT token embeddings for candidate similarity
- Importance: leave-one-out (same as original)
"""
from __future__ import annotations

import torch
import numpy as np
from destror.attacks.base import Attack, VictimFn
from destror.utils.seeding import set_seed

_TOP_K = 50
_SIM_THRESHOLD = 0.7
_BUDGET = 0.20


class BanglaTextFoolerAttack(Attack):
    """LOO importance + BanglaBERT embedding nearest-neighbours for substitution."""

    name = "textfooler"

    def __init__(
        self,
        victim: VictimFn,
        seed: int = 42,
        similarity_threshold: float = _SIM_THRESHOLD,
        perturbation_budget: float = _BUDGET,
        top_k: int = _TOP_K,
        embed_model: str = "csebuetnlp/banglabert",
    ):
        super().__init__(victim, seed)
        self.similarity_threshold = similarity_threshold
        self.perturbation_budget = perturbation_budget
        self.top_k = top_k
        self.embed_model = embed_model
        self._tokenizer = None
        self._model = None

    def _load_embed_model(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModel
        self._tokenizer = AutoTokenizer.from_pretrained(self.embed_model)
        self._model = AutoModel.from_pretrained(self.embed_model)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pool last hidden state for each text."""
        self._load_embed_model()
        enc = self._tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=128, padding=True
        )
        if next(self._model.parameters()).is_cuda:
            enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return emb.cpu().numpy()

    def _word_candidates(self, word: str, context_tokens: list[str], idx: int) -> list[str]:
        """Use fill-mask via BanglaBERT to get substitution candidates, then rank by embedding similarity."""
        # Build masked sentence
        masked = " ".join(context_tokens[:idx] + [self._tokenizer.mask_token] + context_tokens[idx + 1:])
        try:
            enc = self._tokenizer(masked, return_tensors="pt", truncation=True, max_length=256)
            if next(self._model.parameters()).is_cuda:
                enc = {k: v.cuda() for k, v in enc.items()}
            mask_pos = (enc["input_ids"][0] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_pos) == 0:
                return []
            with torch.no_grad():
                logits = self._model(**enc).last_hidden_state[0, mask_pos[0]]
            top_ids = logits.topk(self.top_k).indices.tolist()
            cands = [self._tokenizer.decode([i]).strip() for i in top_ids]
            cands = [c for c in cands if c and c != word and not c.startswith("##")]
        except Exception:
            return []

        # Re-rank by cosine similarity to original word embedding
        if not cands:
            return []
        try:
            orig_emb = self._embed([word])
            cand_embs = self._embed(cands[:20])
            sims = (cand_embs @ orig_emb.T).squeeze() / (
                np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(orig_emb) + 1e-8
            )
            order = np.argsort(-sims)
            return [cands[i] for i in order]
        except Exception:
            return cands

    def attack_one(self, text, orig_pred, orig_conf):
        from destror.metrics.core import semantic_similarity
        set_seed(self.seed)

        tokens = text.split()
        if not tokens:
            return text, orig_pred, orig_conf, 1

        max_swaps = max(1, int(len(tokens) * self.perturbation_budget))
        n_queries = 1

        # LOO importance ranking
        scores = []
        for i, tok in enumerate(tokens):
            masked = tokens[:i] + tokens[i + 1:]
            pred, conf = self.victim(" ".join(masked)) if masked else (orig_pred, orig_conf)
            n_queries += 1
            scores.append((i, orig_conf - conf if pred == orig_pred else float("inf")))
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        current = list(tokens)
        swaps = 0

        for idx, _ in ranked:
            if swaps >= max_swaps:
                break
            cands = self._word_candidates(current[idx], current, idx)
            for cand in cands:
                new_tokens = current[:idx] + [cand] + current[idx + 1:]
                candidate_text = " ".join(new_tokens)
                sim = semantic_similarity(text, candidate_text)
                if sim < self.similarity_threshold:
                    continue
                pred, conf = self.victim(candidate_text)
                n_queries += 1
                if pred != orig_pred:
                    return candidate_text, pred, conf, n_queries
                if conf < orig_conf:
                    current[idx] = cand
                    swaps += 1
                    orig_conf = conf
                    break

        final = " ".join(current)
        pred, conf = self.victim(final)
        n_queries += 1
        return final, pred, conf, n_queries
