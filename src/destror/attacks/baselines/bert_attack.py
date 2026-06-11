"""BERT-Attack (Li et al. 2020) — Bangla adaptation.

Key distinction from BAE:
- Importance: masks each token and measures the model's prediction change
  (mask-then-predict saliency, not LOO on the victim)
- Substitution: accepts first candidate that flips the label (greedy)
- BanglaBERT for fill-mask
"""
from __future__ import annotations

import torch
from destror.attacks.base import Attack, VictimFn
from destror.utils.seeding import set_seed

_TOP_K = 48
_BUDGET = 0.20


class BanglaBERTAttackAttack(Attack):
    """BERT-Attack with BanglaBERT importance scoring and substitution."""

    name = "bert_attack"

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
        self._mlm_model = None
        self._mlm_tokenizer = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "fill-mask", model=self.masker, device=device, top_k=self.top_k,
        )
        # Also load MLM model directly for importance scoring
        self._mlm_tokenizer = self._pipeline.tokenizer
        self._mlm_model = self._pipeline.model

    def _mask_importance(self, tokens: list[str]) -> list[tuple[int, float]]:
        """Importance = entropy increase when token is masked (white-box MLM signal)."""
        self._load_pipeline()
        mask_tok = self._mlm_tokenizer.mask_token
        scores = []
        for i in range(len(tokens)):
            masked = " ".join(tokens[:i] + [mask_tok] + tokens[i + 1:])
            try:
                enc = self._mlm_tokenizer(masked, return_tensors="pt", truncation=True, max_length=256)
                mask_pos = (enc["input_ids"][0] == self._mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    scores.append((i, 0.0))
                    continue
                if next(self._mlm_model.parameters()).is_cuda:
                    enc = {k: v.cuda() for k, v in enc.items()}
                with torch.no_grad():
                    logits = self._mlm_model(**enc).logits[0, mask_pos[0]]
                probs = torch.softmax(logits, dim=-1)
                # Entropy of the masked distribution — higher means the token was important
                entropy = -(probs * (probs + 1e-9).log()).sum().item()
                scores.append((i, entropy))
            except Exception:
                scores.append((i, 0.0))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def attack_one(self, text, orig_pred, orig_conf):
        from destror.metrics.core import semantic_similarity
        set_seed(self.seed)
        self._load_pipeline()

        tokens = text.split()
        if not tokens:
            return text, orig_pred, orig_conf, 1

        max_swaps = max(1, int(len(tokens) * self.perturbation_budget))
        n_queries = 1
        mask_tok = self._mlm_tokenizer.mask_token

        ranked = self._mask_importance(tokens)  # no victim queries
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
        final_pred, final_conf = self.victim(final)
        n_queries += 1
        return final, final_pred, final_conf, n_queries
