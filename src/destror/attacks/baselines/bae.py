"""BAE (Garg & Ramakrishnan 2020) — Bangla adaptation.

Key distinction from one_hot_swap:
- Generates candidates for ALL positions simultaneously (not greedily one at a time)
- Greedy substitution: at each round picks the (position, substitution) pair that
  most reduces model confidence on the CURRENT text (re-evaluates after each swap)

BERT → BanglaBERT; USE similarity → LaBSE.
"""
from __future__ import annotations

import torch
from destror.attacks.base import Attack, VictimFn
from destror.utils.seeding import set_seed

_TOP_K = 10
_BUDGET = 0.20


class BanglaBAEAttack(Attack):
    """BAE-R: greedy confidence-minimising substitution with BanglaBERT fill-mask."""

    name = "bae"

    def __init__(
        self,
        victim: VictimFn,
        seed: int = 42,
        perturbation_budget: float = _BUDGET,
        similarity_threshold: float = 0.7,
        masker: str = "csebuetnlp/banglabert",
        top_k: int = _TOP_K,
    ):
        super().__init__(victim, seed)
        self.perturbation_budget = perturbation_budget
        self.similarity_threshold = similarity_threshold
        self.masker = masker
        self.top_k = top_k
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline
        self._pipeline = pipeline(
            "fill-mask", model=self.masker,
            device=0 if torch.cuda.is_available() else -1,
            top_k=self.top_k,
        )

    def _all_candidates(self, tokens: list[str]) -> dict[int, list[str]]:
        """For each position, get fill-mask candidates."""
        self._load_pipeline()
        mask_tok = self._pipeline.tokenizer.mask_token
        result: dict[int, list[str]] = {}
        for i, tok in enumerate(tokens):
            masked = " ".join(tokens[:i] + [mask_tok] + tokens[i + 1:])
            try:
                fills = self._pipeline(masked)
                cands = [f["token_str"].strip() for f in fills
                         if f["token_str"].strip() and f["token_str"].strip() != tok]
                result[i] = cands
            except Exception:
                result[i] = []
        return result

    def attack_one(self, text, orig_pred, orig_conf):
        from destror.metrics.core import semantic_similarity
        set_seed(self.seed)

        tokens = text.split()
        if not tokens:
            return text, orig_pred, orig_conf, 1

        max_swaps = max(1, int(len(tokens) * self.perturbation_budget))
        n_queries = 1

        # Pre-generate candidates for all positions
        all_cands = self._all_candidates(tokens)
        current = list(tokens)
        swaps = 0
        cur_pred, cur_conf = orig_pred, orig_conf

        for _ in range(max_swaps):
            best_conf = cur_conf
            best_tokens = None
            best_pred = cur_pred

            # Evaluate all (position, candidate) pairs on CURRENT text
            for idx, cands in all_cands.items():
                if idx >= len(current):
                    continue
                for cand in cands[:5]:   # limit per-position candidates for speed
                    new_tokens = current[:idx] + [cand] + current[idx + 1:]
                    candidate_text = " ".join(new_tokens)
                    sim = semantic_similarity(text, candidate_text)
                    if sim < self.similarity_threshold:
                        continue
                    pred, conf = self.victim(candidate_text)
                    n_queries += 1
                    if pred != orig_pred:
                        return candidate_text, pred, conf, n_queries
                    if conf < best_conf:
                        best_conf = conf
                        best_tokens = new_tokens
                        best_pred = pred

            if best_tokens is None:
                break   # no improvement found

            current = best_tokens
            cur_conf = best_conf
            cur_pred = best_pred
            swaps += 1

        final = " ".join(current)
        if final != text:
            final_pred, final_conf = self.victim(final)
            n_queries += 1
            return final, final_pred, final_conf, n_queries
        return final, cur_pred, cur_conf, n_queries
