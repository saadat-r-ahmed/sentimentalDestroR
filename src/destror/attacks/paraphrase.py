"""Bangla Paraphrase Attack — refactored from BanglaClassificationAugment."""
from __future__ import annotations

from typing import Optional
import torch

from destror.attacks.base import Attack, VictimFn
from destror.utils.seeding import set_seed


class BanglaParaphraseAttack(Attack):
    """Paraphrase attack using csebuetnlp/banglat5_banglaparaphrase.

    Sweeps beam-search and temperature settings (Step 2.1) and picks
    the candidate that maximises attack score, falling back to the first
    candidate if none succeed.
    """

    name = "paraphrase"

    # (num_beams, temperature) pairs — sweep as in Step 2.1
    _SETTINGS = [
        (4, 1.0),
        (6, 0.9),
        (8, 0.8),
        (4, 1.2),
        (6, 1.1),
    ]

    def __init__(
        self,
        victim: VictimFn,
        seed: int = 42,
        similarity_threshold: float = 0.7,
        model_name: str = "csebuetnlp/banglat5_banglaparaphrase",
    ):
        super().__init__(victim, seed)
        self.similarity_threshold = similarity_threshold
        self._model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=False)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def _generate_candidates(self, text: str) -> list[str]:
        from normalizer import normalize
        self._load_model()
        set_seed(self.seed)

        try:
            normalized = normalize(text)
        except Exception:
            normalized = text

        inputs = self._tokenizer(normalized, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        candidates: list[str] = []
        seen: set[str] = set()

        for num_beams, temp in self._SETTINGS:
            do_sample = temp != 1.0
            try:
                out = self._model.generate(
                    **inputs,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    temperature=temp if do_sample else 1.0,
                    num_return_sequences=min(num_beams, 3),
                    max_new_tokens=256,
                )
                decoded = self._tokenizer.batch_decode(out, skip_special_tokens=True)
                for d in decoded:
                    d = d.strip()
                    if d and d != text and d not in seen:
                        seen.add(d)
                        candidates.append(d)
            except Exception:
                continue

        return candidates

    def attack_one(
        self,
        text: str,
        orig_pred: int | str,
        orig_conf: float,
    ) -> tuple[str, int | str, float, int]:
        candidates = self._generate_candidates(text)
        n_queries = 1  # victim query for the original

        best_adv = candidates[0] if candidates else text
        best_pred, best_conf = orig_pred, orig_conf
        found = False

        for cand in candidates:
            pred, conf = self.victim(cand)
            n_queries += 1
            if pred != orig_pred and not found:
                best_adv, best_pred, best_conf = cand, pred, conf
                found = True
                break  # take first successful candidate (lowest perturbation)

        if not found and candidates:
            # return the candidate that most reduces confidence even without flipping
            best_adv = candidates[0]
            best_pred, best_conf = self.victim(best_adv)
            n_queries += 1

        return best_adv, best_pred, best_conf, n_queries
