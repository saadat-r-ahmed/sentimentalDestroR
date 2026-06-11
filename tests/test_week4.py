"""Tests for Week 4 adversarial training helpers."""
import json
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# adv_train helpers
# ---------------------------------------------------------------------------

class TestRegimeTag:
    def test_empty_is_clean(self):
        from adv_train import _regime_tag
        assert _regime_tag([]) == "clean"

    def test_single_attack(self):
        from adv_train import _regime_tag
        assert _regime_tag(["paraphrase"]) == "adv_paraphrase"

    def test_two_attacks_sorted(self):
        from adv_train import _regime_tag
        assert _regime_tag(["one_hot_swap", "back_translation"]) == "adv_back_translation_one_hot_swap"

    def test_three_or_more_is_adv_all(self):
        from adv_train import _regime_tag
        assert _regime_tag(["paraphrase", "back_translation", "one_hot_swap"]) == "adv_all"


class TestLoadAdvRecords:
    def _make_jsonl(self, tmp_path: Path, rows: list[dict]) -> Path:
        p = tmp_path / "adv.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p

    def test_only_successful_non_skipped(self, tmp_path):
        from adv_train import _load_adv_records
        p = self._make_jsonl(tmp_path, [
            {"success": True,  "meta": {},                 "adversarial": "আক্রমণ",   "orig_label": "Positive"},
            {"success": False, "meta": {},                 "adversarial": "ব্যর্থ",    "orig_label": "Negative"},
            {"success": True,  "meta": {"skipped": True},  "adversarial": "এড়িয়ে যাওয়া", "orig_label": "Positive"},
        ])
        # _load_adv_records(adv_dir, attack, model_key, dataset, seed) looks for
        # {attack}_{model_key}_{dataset}_seed{seed}.jsonl, so rename the file
        named = tmp_path / "paraphrase_banglabert_blp23_seed42.jsonl"
        p.rename(named)

        records = _load_adv_records(tmp_path, "paraphrase", "banglabert", "blp23", 42)
        assert len(records) == 1
        assert records[0]["text"] == "আক্রমণ"
        assert records[0]["label"] == "Positive"

    def test_missing_file_returns_empty(self, tmp_path):
        from adv_train import _load_adv_records
        records = _load_adv_records(tmp_path, "paraphrase", "banglabert", "blp23", 42)
        assert records == []


# ---------------------------------------------------------------------------
# eval_robustness helpers
# ---------------------------------------------------------------------------

class TestAsrFromJsonl:
    def _write(self, tmp_path, rows):
        p = tmp_path / "run.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p

    def test_asr_computation(self, tmp_path):
        from eval_robustness import _asr_from_jsonl
        p = self._write(tmp_path, [
            {"success": True,  "meta": {},                "n_queries": 10, "labse_sim": 0.9},
            {"success": False, "meta": {},                "n_queries": 5,  "labse_sim": 0.8},
            {"success": False, "meta": {"skipped": True}, "n_queries": 1,  "labse_sim": 0.0},
        ])
        stats = _asr_from_jsonl(p)
        # skipped row is excluded from eligible → n=2, 1 success → ASR=0.5
        assert abs(stats["asr"] - 0.5) < 1e-6
        assert stats["n"] == 2

    def test_all_skipped(self, tmp_path):
        from eval_robustness import _asr_from_jsonl
        p = self._write(tmp_path, [
            {"success": False, "meta": {"skipped": True}, "n_queries": 1, "labse_sim": 0.0},
        ])
        stats = _asr_from_jsonl(p)
        assert stats["n"] == 0 or stats["asr"] == 0.0
