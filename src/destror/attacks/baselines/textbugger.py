"""TextBugger (Li et al. 2019) — Bangla adaptation.

Two-stage strategy (unique among baselines):
1. Character-level: swap adjacent Bangla chars, delete, substitute visually similar
2. Word-level fallback: BanglaBERT fill-mask

Character ops work on Bangla Unicode (U+0980–U+09FF) without any model.
"""
from __future__ import annotations

import random
import torch
from destror.attacks.base import Attack, VictimFn
from destror.utils.seeding import set_seed

# Bangla vowel diacritics that can substitute each other
_BANGLA_VOWEL_MARKS = ["া", "ি", "ী", "ু", "ূ", "ে", "ৈ"]
_BUDGET = 0.20
_TOP_K = 10


class BanglaTextBuggerAttack(Attack):
    """Character-level Bangla perturbations + word-level fill-mask fallback."""

    name = "textbugger"

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

    # ---- character-level ops ----

    def _char_swap(self, word: str) -> str:
        """Swap two adjacent non-space characters."""
        if len(word) < 2:
            return word
        i = random.randint(0, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]

    def _char_delete(self, word: str) -> str:
        if len(word) <= 1:
            return word
        i = random.randint(0, len(word) - 1)
        return word[:i] + word[i + 1:]

    def _char_substitute(self, word: str) -> str:
        """Replace a Bangla vowel mark with another (visually confusable)."""
        chars = list(word)
        marks_in_word = [(i, c) for i, c in enumerate(chars) if c in _BANGLA_VOWEL_MARKS]
        if not marks_in_word:
            return self._char_swap(word)
        idx, original = random.choice(marks_in_word)
        replacement = random.choice([m for m in _BANGLA_VOWEL_MARKS if m != original])
        chars[idx] = replacement
        return "".join(chars)

    def _char_insert(self, word: str) -> str:
        """Insert a zero-width non-joiner (invisible but changes byte sequence)."""
        i = random.randint(0, len(word))
        return word[:i] + "‌" + word[i:]  # ZWNJ

    def _char_perturb_word(self, word: str, rng: random.Random) -> list[str]:
        ops = [self._char_swap, self._char_delete, self._char_substitute, self._char_insert]
        results = []
        for op in ops:
            try:
                p = op(word)
                if p != word:
                    results.append(p)
            except Exception:
                pass
        return results

    # ---- word-level op ----

    def _word_candidates(self, tokens: list[str], idx: int, mask_token: str) -> list[str]:
        self._load_pipeline()
        masked = " ".join(tokens[:idx] + [mask_token] + tokens[idx + 1:])
        try:
            fills = self._pipeline(masked)
            return [f["token_str"].strip() for f in fills if f["token_str"].strip() != tokens[idx]]
        except Exception:
            return []

    def attack_one(self, text, orig_pred, orig_conf):
        from destror.metrics.core import semantic_similarity
        from transformers import AutoTokenizer
        set_seed(self.seed)
        rng = random.Random(self.seed)

        tokens = text.split()
        if not tokens:
            return text, orig_pred, orig_conf, 1

        max_swaps = max(1, int(len(tokens) * self.perturbation_budget))
        n_queries = 1

        # LOO importance
        ranked = []
        for i in range(len(tokens)):
            masked = tokens[:i] + tokens[i + 1:]
            pred, conf = self.victim(" ".join(masked)) if masked else (orig_pred, 0.0)
            n_queries += 1
            ranked.append((i, orig_conf - conf if pred == orig_pred else float("inf")))
        ranked.sort(key=lambda x: x[1], reverse=True)

        current = list(tokens)
        swaps = 0

        for idx, _ in ranked:
            if swaps >= max_swaps:
                break

            # Stage 1: character-level
            char_cands = self._char_perturb_word(current[idx], rng)
            # Stage 2: word-level fill-mask
            self._load_pipeline()
            mask_tok = self._pipeline.tokenizer.mask_token
            word_cands = self._word_candidates(current, idx, mask_tok)

            for cand in char_cands + word_cands:
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
