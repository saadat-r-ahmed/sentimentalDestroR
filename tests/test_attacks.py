"""Unit tests for all attack contracts (revision.md §2.4)."""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_victim(pred="Positive", conf=0.9):
    """Fake victim that always returns the same prediction."""
    return MagicMock(return_value=(pred, conf))


def _sample_records(n=5):
    return [
        {"id": str(i), "text": f"এটি একটি পরীক্ষামূলক বাক্য {i}", "label": "Positive"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Paraphrase attack
# ---------------------------------------------------------------------------

class TestBanglaParaphraseAttack:
    def _make_attack(self, victim):
        from destror.attacks.paraphrase import BanglaParaphraseAttack
        atk = BanglaParaphraseAttack(victim=victim, seed=42)
        # Patch model loading so tests don't need GPU
        atk._generate_candidates = MagicMock(return_value=["ভিন্ন একটি বাক্য"])
        return atk

    def test_label_field_preserved(self):
        victim = _make_victim("Positive", 0.9)
        atk = self._make_attack(victim)
        results = atk.attack_dataset(_sample_records(3), "test_ds", "test_model")
        for r in results:
            assert r.orig_label is not None

    def test_perturbation_budget(self):
        victim = _make_victim("Positive", 0.9)
        atk = self._make_attack(victim)
        results = atk.attack_dataset(_sample_records(3), "test_ds", "test_model")
        for r in results:
            assert r.perturbation_pct >= 0

    def test_deterministic_with_seed(self):
        victim1 = _make_victim("Positive", 0.9)
        victim2 = _make_victim("Positive", 0.9)
        from destror.attacks.paraphrase import BanglaParaphraseAttack
        atk1 = BanglaParaphraseAttack(victim=victim1, seed=42)
        atk2 = BanglaParaphraseAttack(victim=victim2, seed=42)
        atk1._generate_candidates = MagicMock(return_value=["একটি বাক্য"])
        atk2._generate_candidates = MagicMock(return_value=["একটি বাক্য"])
        r1 = atk1.attack_dataset(_sample_records(2), "ds", "mdl")
        r2 = atk2.attack_dataset(_sample_records(2), "ds", "mdl")
        assert [r.adversarial for r in r1] == [r.adversarial for r in r2]

    def test_required_fields_logged(self):
        victim = _make_victim("Positive", 0.9)
        atk = self._make_attack(victim)
        results = atk.attack_dataset(_sample_records(2), "ds", "mdl")
        for r in results:
            assert r.id is not None
            assert r.dataset == "ds"
            assert r.model == "mdl"
            assert r.attack == "paraphrase"
            assert r.n_queries >= 1
            assert r.seed == 42


# ---------------------------------------------------------------------------
# One-hot swap attack
# ---------------------------------------------------------------------------

class TestBanglaOneHotSwapAttack:
    def _make_attack(self, victim):
        from destror.attacks.one_hot_swap import BanglaOneHotSwapAttack
        atk = BanglaOneHotSwapAttack(victim=victim, seed=42, masker="xlm-r")
        atk._load_masker = MagicMock()
        atk._fill_pipeline = MagicMock(return_value=[
            {"token_str": "ভালো", "score": 0.9},
            {"token_str": "খারাপ", "score": 0.8},
        ])
        return atk

    def test_respects_perturbation_budget(self):
        victim = _make_victim("Positive", 0.9)
        atk = self._make_attack(victim)
        text = "এটি একটি পরীক্ষা বাক্য যার অনেক শব্দ আছে"
        adv, _, _, _ = atk.attack_one(text, "Positive", 0.9)
        orig_toks = text.split()
        adv_toks = adv.split()
        changed = sum(a != b for a, b in zip(orig_toks, adv_toks))
        assert changed / len(orig_toks) <= 0.20 + 1e-6

    def test_required_fields_logged(self):
        victim = _make_victim("Positive", 0.9)
        atk = self._make_attack(victim)
        results = atk.attack_dataset(_sample_records(2), "ds", "mdl")
        for r in results:
            assert r.n_queries >= 1
            assert r.seed == 42


# ---------------------------------------------------------------------------
# Back-translation attack
# ---------------------------------------------------------------------------

class TestBanglaBackTranslationAttack:
    def _make_attack(self, victim):
        from destror.attacks.back_translation import BanglaBackTranslationAttack
        atk = BanglaBackTranslationAttack(victim=victim, seed=42, pivots=["en"])
        # Bypass model loading entirely — mock at the candidate level
        atk._generate_candidates = MagicMock(return_value=["ভিন্ন একটি বাক্য", "আরেকটি বাক্য"])
        return atk

    def test_required_fields_logged(self):
        victim = _make_victim("Positive", 0.9)
        atk = self._make_attack(victim)
        results = atk.attack_dataset(_sample_records(2), "ds", "mdl")
        for r in results:
            assert r.attack == "back_translation"
            assert r.n_queries >= 1
            assert r.seed == 42

    def test_picks_best_candidate(self):
        """When one candidate flips the label the result should be success=True."""
        # victim returns Negative for second candidate (flip), Positive for first
        call_count = [0]
        def victim(text):
            call_count[0] += 1
            if "আরেকটি" in text:
                return ("Negative", 0.85)
            return ("Positive", 0.9)

        from destror.attacks.back_translation import BanglaBackTranslationAttack
        atk = BanglaBackTranslationAttack(victim=victim, seed=42, similarity_threshold=0.0)
        atk._generate_candidates = MagicMock(return_value=["ভিন্ন একটি বাক্য", "আরেকটি বাক্য"])
        results = atk.attack_dataset(_sample_records(1), "ds", "mdl")
        assert results[0].success is True

    def test_falls_back_on_no_flip(self):
        """With no label flip, success should be False and adversarial still set."""
        victim = _make_victim("Positive", 0.9)
        from destror.attacks.back_translation import BanglaBackTranslationAttack
        atk = BanglaBackTranslationAttack(victim=victim, seed=42, similarity_threshold=0.0)
        atk._generate_candidates = MagicMock(return_value=["ভিন্ন একটি বাক্য"])
        results = atk.attack_dataset(_sample_records(1), "ds", "mdl")
        assert results[0].success is False
        assert results[0].adversarial != ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perturbation_rate_zero_for_identical(self):
        from destror.metrics.core import perturbation_rate
        assert perturbation_rate("hello world", "hello world") == 0.0

    def test_perturbation_rate_100_all_different(self):
        from destror.metrics.core import perturbation_rate
        assert perturbation_rate("a b", "c d") == 100.0

    def test_asr_zero_no_success(self):
        from destror.metrics.core import attack_success_rate
        from dataclasses import dataclass

        @dataclass
        class _R:
            success: bool
            meta: dict

        results = [_R(success=False, meta={}) for _ in range(5)]
        assert attack_success_rate(results) == 0.0

    def test_asr_full_success(self):
        from destror.metrics.core import attack_success_rate
        from dataclasses import dataclass

        @dataclass
        class _R:
            success: bool
            meta: dict

        results = [_R(success=True, meta={}) for _ in range(5)]
        assert attack_success_rate(results) == 1.0
