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
    """Word-swap attack with BanglaBERT/MuRIL mask filling and leave-one-out importance.

    Changes from original:
    - Mask-fill model is configurable (default: BanglaBERT, ablation: XLM-R)
    - Importance ranking uses leave-one-out (black-box) — gradient saliency in future work
    - Hard budget cap: ≤20% of tokens perturbed
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
    ):
        super().__init__(victim, seed)
        if masker not in _MASKER_MODELS:
            raise ValueError(f"masker must be one of {list(_MASKER_MODELS)}")
        self.masker = masker
        self.top_k = top_k
        self.perturbation_budget = perturbation_budget
        self.similarity_threshold = similarity_threshold
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
        """Leave-one-out importance: score = drop in confidence when token removed."""
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
        n_queries += len(tokens)  # leave-one-out queries

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
