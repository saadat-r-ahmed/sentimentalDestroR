"""Entry point: run a named attack across all models and datasets."""
import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from destror.utils.seeding import set_seed
from destror.utils.logging import get_logger, get_git_sha

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="attack/paraphrase")
def main(cfg: DictConfig) -> None:
    import wandb
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
        "paraphrase":      BanglaParaphraseAttack,
        "back_translation": BanglaBackTranslationAttack,
        "one_hot_swap":    BanglaOneHotSwapAttack,
        "textfooler":      BanglaTextFoolerAttack,
        "textbugger":      BanglaTextBuggerAttack,
        "bae":             BanglaBAEAttack,
        "bert_attack":     BanglaBERTAttackAttack,
        "pwws":            BanglaPWWSAttack,
    }

    attack_cls = attack_map[cfg.attack.name]
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in cfg.run.seeds:
        set_seed(seed)

        for model_key in cfg.model.victims:
            victim = load_victim(model_key)

            attack = attack_cls(victim=victim, seed=seed, **{
                k: v for k, v in cfg.attack.items()
                if k not in ("name",)
            })

            for dataset_name in cfg.dataset.names:
                run_id = f"{cfg.attack.name}_{model_key}_{dataset_name}_seed{seed}"
                log.info(f"Starting run: {run_id}")

                if cfg.run.wandb:
                    wandb.init(
                        project=os.getenv("WANDB_PROJECT", "destror"),
                        name=run_id,
                        config={
                            "attack": cfg.attack.name,
                            "model": model_key,
                            "dataset": dataset_name,
                            "seed": seed,
                            "git_sha": get_git_sha(),
                        },
                        reinit=True,
                    )

                try:
                    records = load_dataset(dataset_name, split=cfg.dataset.split, seed=seed)
                    if cfg.dataset.max_samples:
                        records = records[:cfg.dataset.max_samples]

                    results = attack.attack_dataset(
                        records,
                        dataset_name=dataset_name,
                        model_name=model_key,
                    )

                    # Save per-run JSON
                    run_file = out_dir / f"{run_id}.jsonl"
                    with open(run_file, "w", encoding="utf-8") as f:
                        for r in results:
                            f.write(r.to_json() + "\n")

                    asr = sum(r.success for r in results) / max(1, sum(
                        not r.meta.get("skipped") for r in results
                    ))
                    log.info(f"  ASR={asr:.3f}  saved → {run_file}")

                    if cfg.run.wandb:
                        wandb.log({"asr": asr, "n_samples": len(results)})

                except Exception as e:
                    log.error(f"  FAILED: {e}")
                finally:
                    if cfg.run.wandb:
                        wandb.finish()


if __name__ == "__main__":
    main()
