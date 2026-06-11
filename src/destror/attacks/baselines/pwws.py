"""PWWS (Ren et al. 2019) — Bangla adaptation.

Key distinction:
- Saliency = softmax-normalised LOO confidence drop × inverse token frequency
  (approximates the word importance × NE weight from the original paper)
- Substitution: BanglaBERT fill-mask (WordNet → fill-mask since BanglaNet is unavailable)
- Selection: highest-saliency word first, accept first flip
"""
from __future__ import annotations

import math
import torch
from collections import Counter
from destror.attacks.base import Attack, VictimFn
from destror.utils.seeding import set_seed

_TOP_K = 10
_BUDGET = 0.20


class BanglaPWWSAttack(Attack):
    """PWWS with saliency × IDF importance and BanglaBERT substitution."""

    name = "pwws"

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

    def _pwws_importance(
        self, tokens: list[str], orig_pred, orig_conf
    ) -> list[tuple[int, float]]:
        """Saliency(i) = softmax(H(i)) × IDF(i) where H(i) = LOO confidence drop."""
        n_queries = 0
        loo_scores = []
        for i in range(len(tokens)):
            masked = tokens[:i] + tokens[i + 1:]
            pred, conf = self.victim(" ".join(masked)) if masked else (orig_pred, orig_conf)
            n_queries += 1
            h = (orig_conf - conf) if pred == orig_pred else orig_conf + 1.0
            loo_scores.append(h)

        # Softmax normalise
        max_h = max(loo_scores) if loo_scores else 1.0
        exp_h = [math.exp(h - max_h) for h in loo_scores]
        denom = sum(exp_h) + 1e-9
        softmax_h = [e / denom for e in exp_h]

        # IDF proxy: inverse frequency within this sentence
        freq = Counter(tokens)
        idf = {w: math.log(len(tokens) / (freq[w] + 1)) + 1.0 for w in tokens}

        scored = [
            (i, softmax_h[i] * idf[tokens[i]])
            for i in range(len(tokens))
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True), n_queries

    def attack_one(self, text, orig_pred, orig_conf):
        from destror.metrics.core import semantic_similarity
        set_seed(self.seed)
        self._load_pipeline()

        tokens = text.split()
        if not tokens:
            return text, orig_pred, orig_conf, 1

        max_swaps = max(1, int(len(tokens) * self.perturbation_budget))
        n_queries = 1

        ranked, loo_queries = self._pwws_importance(tokens, orig_pred, orig_conf)
        n_queries += loo_queries

        mask_tok = self._pipeline.tokenizer.mask_token
        current = list(tokens)
        swaps = 0

        for idx, _ in ranked:
            if swaps >= max_swaps:
                break
            if idx >= len(current):
                continue

            masked = " ".join(current[:idx] + [mask_tok] + current[idx + 1:])
            try:
                fills = self._pipeline(masked)
                cands = [f["token_str"].strip() for f in fills
                         if f["token_str"].strip() and f["token_str"].strip() != current[idx]]
            except Exception:
                continue

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
