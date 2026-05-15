"""Sanity checks for seeding and JSON serialization."""
import json
import pytest


def test_set_seed_no_crash():
    from destror.utils.seeding import set_seed
    set_seed(42)
    set_seed(0)


def test_attack_result_json_roundtrip():
    from destror.attacks.base import AttackResult
    r = AttackResult(
        id="1", dataset="blp23", model="banglabert", attack="paraphrase",
        original="মূল বাক্য", adversarial="পরিবর্তিত বাক্য",
        orig_label="Positive", orig_pred="Positive", adv_pred="Negative",
        orig_conf=0.91, adv_conf=0.62,
        n_queries=5, perturbation_pct=12.5,
        labse_sim=0.87, perplexity_delta=14.3,
        success=True, seed=42, git_sha="abc1234",
    )
    dumped = r.to_json()
    loaded = json.loads(dumped)
    assert loaded["id"] == "1"
    assert loaded["success"] is True
    assert loaded["labse_sim"] == 0.87
    assert loaded["original"] == "মূল বাক্য"
