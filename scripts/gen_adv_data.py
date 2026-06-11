"""Generate adversarial examples from the train split for adversarial training.

Uses the same attack configs as run_attack.py but targets split=train and
limits examples to augment_ratio × len(train) (capped by max_adv_samples).

Victim model: the locally finetuned checkpoint for each (model, dataset, seed)
triple is preferred over the base model, so attacks are on the model we will
later make robust rather than the pretrained weights.

Usage:
  uv run python scripts/gen_adv_data.py --config-name=attack/paraphrase
  uv run python scripts/gen_adv_data.py --config-name=attack/back_translation
  uv run python scripts/gen_adv_data.py --config-name=attack/one_hot_swap

Optional overrides:
  +augment_ratio=0.5     fraction of train to attack (default 0.5)
  +max_adv_samples=300   hard cap per (model, dataset) combo
  +checkpoint_dir=results/finetuned_models
  model.victims=[banglabert,banglishbert,xlm-r,muril,indicbertv2]
"""
import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from destror.utils.seeding import set_seed
from destror.utils.logging import get_logger

log = get_logger(__name__)

_ADV_DATA_DIR      = Path("results/adv_train_data")
_DEFAULT_CKPT_DIR  = Path("results/finetuned_models")
_BASE_VICTIMS      = ["banglabert", "banglishbert", "xlm-r", "muril", "indicbertv2"]


def _best_local_ckpt(model_key: str, dataset_name: str, seed: int, ckpt_root: Path) -> Path | None:
    """Return the best saved checkpoint for (model, dataset, seed), or None."""
    run_dir = ckpt_root / f"{model_key}_{dataset_name}_seed{seed}"
    if not run_dir.exists():
        return None
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        return None
    last = ckpts[-1]
    try:
        state = json.loads((last / "trainer_state.json").read_text())
        best = state.get("best_model_checkpoint")
        if best and Path(best).exists():
            return Path(best)
    except Exception:
        pass
    return last


@hydra.main(version_base=None, config_path="../configs", config_name="attack/paraphrase")
def main(cfg: DictConfig) -> None:
    from destror.attacks.paraphrase import BanglaParaphraseAttack
    from destror.attacks.back_translation import BanglaBackTranslationAttack
    from destror.attacks.one_hot_swap import BanglaOneHotSwapAttack
    from destror.attacks.baselines import (
        BanglaTextFoolerAttack, BanglaTextBuggerAttack,
        BanglaBAEAttack, BanglaBERTAttackAttack, BanglaPWWSAttack,
    )
    from destror.datasets.loaders import load_dataset
    from destror.models.loaders import load_victim

    attack_map = {
        "paraphrase":       BanglaParaphraseAttack,
        "back_translation": BanglaBackTranslationAttack,
        "one_hot_swap":     BanglaOneHotSwapAttack,
        "textfooler":       BanglaTextFoolerAttack,
        "textbugger":       BanglaTextBuggerAttack,
        "bae":              BanglaBAEAttack,
        "bert_attack":      BanglaBERTAttackAttack,
        "pwws":             BanglaPWWSAttack,
    }

    augment_ratio  = float(getattr(cfg, "augment_ratio", 0.5))
    max_adv        = int(getattr(cfg, "max_adv_samples", 300))
    ckpt_root      = Path(getattr(cfg, "checkpoint_dir", str(_DEFAULT_CKPT_DIR)))
    attack_cls     = attack_map[cfg.attack.name]

    # Use override victims list if provided; otherwise fall back to base 5 models
    victims = list(cfg.model.victims) if cfg.model.get("victims") else _BASE_VICTIMS
    # Exclude titulm-1b — QLoRA victim is too slow to attack thousands of examples
    victims = [v for v in victims if v != "titulm-1b"]

    out_dir = _ADV_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in cfg.run.seeds:
        set_seed(seed)

        for model_key in victims:
            for dataset_name in cfg.dataset.names:
                out_file = out_dir / f"{cfg.attack.name}_{model_key}_{dataset_name}_seed{seed}.jsonl"
                if out_file.exists():
                    log.info(f"skip (exists): {out_file.name}")
                    continue

                # Load the finetuned checkpoint for this (model, dataset) pair
                ckpt = _best_local_ckpt(model_key, dataset_name, seed, ckpt_root)
                if ckpt:
                    log.info(f"victim: local checkpoint {ckpt}")
                    victim = load_victim(str(ckpt))
                else:
                    log.warning(f"no local checkpoint for {model_key}/{dataset_name}/seed{seed} — using base model")
                    victim = load_victim(model_key)

                attack = attack_cls(
                    victim=victim,
                    seed=seed,
                    **{k: v for k, v in cfg.attack.items() if k not in ("name",)},
                )

                records = load_dataset(dataset_name, split="train", seed=seed)
                n = min(max_adv, max(1, int(len(records) * augment_ratio)))
                records = records[:n]

                log.info(
                    f"{cfg.attack.name}@{model_key}/{dataset_name}/seed{seed} — "
                    f"attacking {n} train examples (ratio={augment_ratio}, cap={max_adv})"
                )

                results = attack.attack_dataset(
                    records, dataset_name=dataset_name, model_name=model_key
                )

                with open(out_file, "w", encoding="utf-8") as f:
                    for r in results:
                        f.write(r.to_json() + "\n")

                n_success = sum(r.success for r in results)
                asr = n_success / max(1, sum(not r.meta.get("skipped") for r in results))
                log.info(f"  ASR={asr:.3f}  n_success={n_success}  saved → {out_file.name}")


if __name__ == "__main__":
    main()
