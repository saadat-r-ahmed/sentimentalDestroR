"""Generate adversarial examples from the train split for adversarial training.

Uses the same attack configs as run_attack.py but targets split=train and
limits examples to augment_ratio × len(train).

Usage:
  uv run python scripts/gen_adv_data.py --config-name=attack/paraphrase +augment_ratio=0.5
  uv run python scripts/gen_adv_data.py --config-name=attack/back_translation +augment_ratio=0.5
  uv run python scripts/gen_adv_data.py --config-name=attack/one_hot_swap +augment_ratio=0.5
"""
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from destror.utils.seeding import set_seed
from destror.utils.logging import get_logger

log = get_logger(__name__)

_ADV_DATA_DIR = Path("results/adv_train_data")


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

    augment_ratio = float(getattr(cfg, "augment_ratio", 0.5))
    out_dir = _ADV_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_cls = attack_map[cfg.attack.name]

    for seed in cfg.run.seeds:
        set_seed(seed)

        for model_key in cfg.model.victims:
            victim = load_victim(model_key)
            attack = attack_cls(
                victim=victim,
                seed=seed,
                **{k: v for k, v in cfg.attack.items() if k not in ("name",)},
            )

            for dataset_name in cfg.dataset.names:
                out_file = out_dir / f"{cfg.attack.name}_{model_key}_{dataset_name}_seed{seed}.jsonl"
                if out_file.exists():
                    log.info(f"skip (exists): {out_file.name}")
                    continue

                records = load_dataset(dataset_name, split="train", seed=seed)
                n = max(1, int(len(records) * augment_ratio))
                records = records[:n]

                log.info(
                    f"{cfg.attack.name}@{model_key}/{dataset_name} — attacking "
                    f"{n}/{int(n/augment_ratio)} train examples (ratio={augment_ratio})"
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
