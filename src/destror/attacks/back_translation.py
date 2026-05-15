"""Bangla Pivot Back-Translation Attack — refactored from BanglaClassificationAugment."""
from __future__ import annotations

import torch
from typing import Optional

from destror.attacks.base import Attack, VictimFn
from destror.utils.seeding import set_seed


# Pivot languages: BN→pivot→BN. EN is the original; HI adds Indic diversity.
PIVOT_CONFIGS = {
    "en": {
        "bn_to_pivot": "csebuetnlp/banglat5_nmt_bn_en",
        "pivot_to_bn": "csebuetnlp/banglat5_nmt_en_bn",
    },
    # Hindi pivot via Helsinki-NLP models (multilingual)
    "hi": {
        "bn_to_pivot": "Helsinki-NLP/opus-mt-bn-hi",
        "pivot_to_bn":  "Helsinki-NLP/opus-mt-hi-bn",
    },
}

_TEMPERATURES = [1.0, 0.9, 1.1, 1.2]


class BanglaBackTranslationAttack(Attack):
    """Back-translation attack with pivot language sweep and candidate selection.

    For each pivot language and temperature, generates a back-translated candidate
    then picks the one that maximises attack_score × LaBSE_similarity (Step 2.2).
    """

    name = "back_translation"

    def __init__(
        self,
        victim: VictimFn,
        seed: int = 42,
        pivots: list[str] | None = None,
        similarity_threshold: float = 0.7,
    ):
        super().__init__(victim, seed)
        self.pivots = pivots or ["en", "hi"]
        self.similarity_threshold = similarity_threshold
        self._models: dict[str, tuple] = {}  # pivot_lang → (fwd_model, fwd_tok, bwd_model, bwd_tok)

    def _load_pivot(self, pivot: str):
        if pivot in self._models:
            return
        if pivot not in PIVOT_CONFIGS:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        cfg = PIVOT_CONFIGS[pivot]

        fwd_tok = AutoTokenizer.from_pretrained(cfg["bn_to_pivot"], use_fast=False)
        fwd_mdl = AutoModelForSeq2SeqLM.from_pretrained(cfg["bn_to_pivot"])
        bwd_tok = AutoTokenizer.from_pretrained(cfg["pivot_to_bn"], use_fast=False)
        bwd_mdl = AutoModelForSeq2SeqLM.from_pretrained(cfg["pivot_to_bn"])

        for m in (fwd_mdl, bwd_mdl):
            m.eval()
            if torch.cuda.is_available():
                m.cuda()

        self._models[pivot] = (fwd_mdl, fwd_tok, bwd_mdl, bwd_tok)

    def _translate(self, text: str, model, tokenizer, temperature: float = 1.0) -> str:
        try:
            from normalizer import normalize
            text = normalize(text)
        except Exception:
            pass
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        do_sample = temperature != 1.0
        out = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            max_new_tokens=256,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _generate_candidates(self, text: str) -> list[str]:
        set_seed(self.seed)
        candidates: list[str] = []
        seen: set[str] = set()

        for pivot in self.pivots:
            try:
                self._load_pivot(pivot)
            except Exception:
                continue
            if pivot not in self._models:
                continue
            fwd_mdl, fwd_tok, bwd_mdl, bwd_tok = self._models[pivot]

            for temp in _TEMPERATURES:
                try:
                    pivot_text = self._translate(text, fwd_mdl, fwd_tok, temperature=temp)
                    back = self._translate(pivot_text, bwd_mdl, bwd_tok, temperature=temp)
                    back = back.strip()
                    if back and back != text and back not in seen:
                        seen.add(back)
                        candidates.append(back)
                except Exception:
                    continue

        return candidates

    def attack_one(
        self,
        text: str,
        orig_pred: int | str,
        orig_conf: float,
    ) -> tuple[str, int | str, float, int]:
        from destror.metrics.core import semantic_similarity

        candidates = self._generate_candidates(text)
        n_queries = 1

        best_adv = candidates[0] if candidates else text
        best_pred, best_conf = orig_pred, orig_conf
        best_score = -1.0
        found = False

        for cand in candidates:
            sim = semantic_similarity(text, cand)
            if sim < self.similarity_threshold:
                continue
            pred, conf = self.victim(cand)
            n_queries += 1

            # Combined score: attack probability × similarity
            flip_score = (1.0 - conf) if pred != orig_pred else 0.0
            combined = flip_score * sim

            if combined > best_score:
                best_score = combined
                best_adv, best_pred, best_conf = cand, pred, conf
                if pred != orig_pred:
                    found = True

        if not found and candidates:
            # fallback: return highest-similarity candidate
            best_adv = max(candidates, key=lambda c: semantic_similarity(text, c))
            best_pred, best_conf = self.victim(best_adv)
            n_queries += 1

        return best_adv, best_pred, best_conf, n_queries
