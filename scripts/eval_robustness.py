"""Build the robustness matrix (Table 4 in the paper).

For each adv-trained checkpoint × attack × dataset, this script:
  1. Loads the checkpoint as a victim
  2. Runs the specified attack on the test split
  3. Records ASR, mean_queries, mean_sim
  4. Accumulates into results/robustness_matrix.json

Attack results for clean-trained models (from results/raw_runs/) are folded in
automatically by reading existing JSONL files.

Usage:
  # Evaluate all adv-trained models against all attacks:
  uv run python scripts/eval_robustness.py

  # Evaluate only one attack:
  uv run python scripts/eval_robustness.py attack.name=paraphrase

  # Dry-run (list what would be evaluated, no GPU):
  uv run python scripts/eval_robustness.py dry_run=true
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import defaultdict

import hydra
from omegaconf import DictConfig

from destror.utils.logging import get_logger

log = get_logger(__name__)

_RAW_RUNS_DIR  = Path("results/raw_runs")
_ADV_MODELS_DIR = Path("results/adv_trained_models")
_MATRIX_PATH   = Path("results/robustness_matrix.json")

# Attacks that appear as test columns in the matrix
_TEST_ATTACKS = ["paraphrase", "back_translation", "one_hot_swap", "textfooler", "bae"]


def _asr_from_jsonl(path: Path) -> dict:
    """Compute aggregate metrics from an existing attack JSONL."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    eligible = [r for r in results if not r.get("meta", {}).get("skipped")]
    n = max(1, len(eligible))
    n_success = sum(r.get("success", False) for r in eligible)
    return {
        "asr": n_success / n,
        "n": n,
        "mean_queries": sum(r.get("n_queries", 0) for r in eligible) / n,
        "mean_sim": sum(r.get("labse_sim", 0.0) for r in eligible) / n,
    }


def _find_checkpoint(model_key: str, dataset_name: str, regime: str, seed: int) -> Path | None:
    """Locate the best-model checkpoint directory for an adv-trained run.

    adv_train.py saves to: results/adv_trained_models/{regime}/{model}_{dataset}_seed{seed}/
    The Trainer creates checkpoint-XXXX subdirs; trainer_state.json records the best one.
    """
    # Primary layout: regime subdir (current convention)
    run_name = f"{model_key}_{dataset_name}_seed{seed}"
    parent = _ADV_MODELS_DIR / regime / run_name
    if parent.exists():
        try:
            import json as _json
            ckpts = sorted(parent.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if ckpts:
                state = _json.loads((ckpts[-1] / "trainer_state.json").read_text())
                best = state.get("best_model_checkpoint")
                if best and Path(best).exists():
                    return Path(best)
                return ckpts[-1]
        except Exception:
            pass
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="attack/paraphrase")
def main(cfg: DictConfig) -> None:
    from destror.attacks.paraphrase import BanglaParaphraseAttack
    from destror.attacks.back_translation import BanglaBackTranslationAttack
    from destror.attacks.one_hot_swap import BanglaOneHotSwapAttack
    from destror.attacks.baselines import (
        BanglaTextFoolerAttack, BanglaBAEAttack,
    )
    from destror.datasets.loaders import load_dataset
    from destror.models.loaders import load_victim

    attack_map = {
        "paraphrase":       BanglaParaphraseAttack,
        "back_translation": BanglaBackTranslationAttack,
        "one_hot_swap":     BanglaOneHotSwapAttack,
        "textfooler":       BanglaTextFoolerAttack,
        "bae":              BanglaBAEAttack,
    }

    dry_run = bool(getattr(cfg, "dry_run", False))
    seeds = list(cfg.run.seeds)
    datasets = list(cfg.dataset.names)
    victims = list(cfg.model.victims)

    # Discover available training regimes — each regime is a subdir of adv_trained_models/
    regimes_found: set[str] = {"clean"}
    if _ADV_MODELS_DIR.exists():
        for d in _ADV_MODELS_DIR.iterdir():
            if d.is_dir() and d.name.startswith("adv_"):
                regimes_found.add(d.name)

    log.info(f"Regimes found: {sorted(regimes_found)}")

    # Load existing matrix
    matrix: dict = json.loads(_MATRIX_PATH.read_text()) if _MATRIX_PATH.exists() else {}

    # 1 ── Fold in existing raw_runs (clean-trained model test results)
    if _RAW_RUNS_DIR.exists():
        for jsonl in _RAW_RUNS_DIR.glob("*.jsonl"):
            stem = jsonl.stem  # {attack}_{model}_{dataset}_seed{seed}
            parts = stem.split("_")
            if len(parts) < 4:
                continue
            # attack might be multi-word (back_translation) so parse from filename pattern
            for atk in attack_map:
                if stem.startswith(atk + "_"):
                    remainder = stem[len(atk) + 1:]  # model_dataset_seed{seed}
                    for seed in seeds:
                        suffix = f"_seed{seed}"
                        if remainder.endswith(suffix):
                            rest = remainder[:-len(suffix)]
                            for dataset in datasets:
                                if rest.endswith("_" + dataset):
                                    model_key = rest[: -(len(dataset) + 1)]
                                    key = f"clean|{model_key}|{dataset}|seed{seed}"
                                    col = atk
                                    if key not in matrix:
                                        matrix[key] = {}
                                    if col not in matrix[key]:
                                        matrix[key][col] = _asr_from_jsonl(jsonl)
                                        log.info(f"  folded: {key} vs {col}")

    # 2 ── Evaluate adv-trained models against each test attack
    eval_dir = Path("results/adv_eval_runs")
    eval_dir.mkdir(parents=True, exist_ok=True)

    for regime in sorted(regimes_found - {"clean"}):
        for seed in seeds:
            for model_key in victims:
                for dataset_name in datasets:
                    ckpt = _find_checkpoint(model_key, dataset_name, regime, seed)
                    if ckpt is None:
                        log.debug(f"checkpoint not found: {model_key}/{dataset_name}/{regime}/seed{seed}")
                        continue

                    matrix_key = f"{regime}|{model_key}|{dataset_name}|seed{seed}"

                    for attack_name, attack_cls in attack_map.items():
                        if attack_name not in _TEST_ATTACKS:
                            continue
                        if matrix_key in matrix and attack_name in matrix[matrix_key]:
                            log.info(f"skip (exists): {matrix_key} vs {attack_name}")
                            continue

                        out_file = eval_dir / f"{attack_name}_{model_key}_{dataset_name}_{regime}_seed{seed}.jsonl"

                        if dry_run:
                            log.info(f"[DRY RUN] would eval: {matrix_key} vs {attack_name}")
                            continue

                        log.info(f"Evaluating: {matrix_key} vs {attack_name}")
                        try:
                            victim = load_victim(str(ckpt))
                            attack = attack_cls(
                                victim=victim,
                                seed=seed,
                                **{k: v for k, v in cfg.attack.items() if k not in ("name",)},
                            )
                            records = load_dataset(dataset_name, split="test", seed=seed)
                            results = attack.attack_dataset(
                                records, dataset_name=dataset_name, model_name=f"{model_key}_{regime}"
                            )
                            with open(out_file, "w", encoding="utf-8") as f:
                                for r in results:
                                    f.write(r.to_json() + "\n")

                            if matrix_key not in matrix:
                                matrix[matrix_key] = {}
                            matrix[matrix_key][attack_name] = _asr_from_jsonl(out_file)
                            log.info(f"  ASR={matrix[matrix_key][attack_name]['asr']:.3f}")

                        except Exception as e:
                            log.error(f"  FAILED: {e}")

    _MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MATRIX_PATH.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Matrix saved → {_MATRIX_PATH}  ({len(matrix)} rows)")

    # Print human-readable summary
    _print_matrix(matrix, datasets, victims, seeds)


def _print_matrix(matrix: dict, datasets, victims, seeds):
    """Print a condensed ASCII table averaged over datasets, victims, seeds."""
    regimes = sorted({k.split("|")[0] for k in matrix})
    print(f"\n{'Regime':<28}", end="")
    for atk in _TEST_ATTACKS:
        print(f"  {atk[:10]:<12}", end="")
    print()
    print("-" * (28 + 14 * len(_TEST_ATTACKS)))

    for regime in regimes:
        asrs = defaultdict(list)
        for key, cols in matrix.items():
            if key.startswith(regime + "|"):
                for atk, stats in cols.items():
                    asrs[atk].append(stats["asr"])

        print(f"{regime:<28}", end="")
        for atk in _TEST_ATTACKS:
            if asrs[atk]:
                mean_asr = sum(asrs[atk]) / len(asrs[atk])
                print(f"  {mean_asr:.3f}       ", end="")
            else:
                print(f"  {'—':<12}", end="")
        print()


if __name__ == "__main__":
    main()
