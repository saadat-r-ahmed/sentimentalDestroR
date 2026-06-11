"""Adversarial training: finetune victim models on clean + adversarial mixed data.

Reads pre-generated JSONL from gen_adv_data.py. Only successful adversarial
examples (success=true, skipped not set) are added as augmentation.

Training regime tag is derived from cfg.adv_data.attacks:
  [paraphrase]                         → adv_paraphrase
  [back_translation]                   → adv_back_translation
  [paraphrase, back_translation, ...]  → adv_all

Usage:
  uv run python scripts/adv_train.py
  uv run python scripts/adv_train.py adv_data.attacks=[back_translation]
  uv run python scripts/adv_train.py adv_data.attacks=[paraphrase,back_translation,one_hot_swap]
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from destror.utils.seeding import set_seed
from destror.utils.logging import get_logger

log = get_logger(__name__)


def _regime_tag(attacks: list[str]) -> str:
    if len(attacks) == 0:
        return "clean"
    if len(attacks) >= 3:
        return "adv_all"
    return "adv_" + "_".join(sorted(attacks))


def _load_adv_records(adv_dir: Path, attack: str, model_key: str, dataset_name: str, seed: int) -> list[dict]:
    """Load successful adversarial examples from JSONL as {text, label} dicts."""
    path = adv_dir / f"{attack}_{model_key}_{dataset_name}_seed{seed}.jsonl"
    if not path.exists():
        log.warning(f"adv data not found: {path} — skipping {attack} augmentation")
        return []

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("success") and not r.get("meta", {}).get("skipped"):
                records.append({"text": r["adversarial"], "label": r["orig_label"]})
    return records


@hydra.main(version_base=None, config_path="../configs", config_name="training/adv_train")
def main(cfg: DictConfig) -> None:
    import os
    import wandb
    from destror.datasets.loaders import load_dataset
    from destror.models.finetune import finetune

    attacks = list(cfg.adv_data.attacks)
    regime = _regime_tag(attacks)
    adv_dir = Path(cfg.adv_data.dir)
    out_root = Path(cfg.run.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    lora_cfg = OmegaConf.to_container(cfg.lora, resolve=True)
    overrides = OmegaConf.to_container(cfg.get("model_overrides", {}), resolve=True)

    summary_path = out_root / f"robustness_results_{regime}.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    existing_keys = {(m["model"], m["dataset"], m["seed"], m["regime"]) for m in existing}
    all_metrics: list[dict] = []

    for seed in cfg.run.seeds:
        set_seed(seed)

        for model_key in cfg.model.victims:
            mo = overrides.get(model_key, {})
            batch_size = mo.get("batch_size", cfg.training.batch_size)
            grad_accum = mo.get("gradient_accumulation_steps", cfg.training.gradient_accumulation_steps)
            max_seq = mo.get("max_seq_length", cfg.training.max_seq_length)
            fp16 = mo.get("fp16", cfg.training.fp16)
            bf16 = mo.get("bf16", cfg.training.bf16)
            use_lora = model_key == "titulm-1b"

            for dataset_name in cfg.dataset.names:
                run_key = (model_key, dataset_name, seed, regime)
                if run_key in existing_keys:
                    log.info(f"skip (exists): {model_key}/{dataset_name}/{regime}/seed{seed}")
                    continue

                train_records = load_dataset(dataset_name, split="train", seed=cfg.dataset.split_seed)
                dev_records   = load_dataset(dataset_name, split="dev",   seed=cfg.dataset.split_seed)
                test_records  = load_dataset(dataset_name, split="test",  seed=cfg.dataset.split_seed)

                # Gather adversarial augmentation from each requested attack
                adv_pool: list[dict] = []
                for atk in attacks:
                    adv_pool.extend(_load_adv_records(adv_dir, atk, model_key, dataset_name, seed))

                if adv_pool:
                    random.seed(seed)
                    n_aug = max(1, int(len(train_records) * cfg.adv_data.augment_ratio))
                    adv_sample = random.sample(adv_pool, min(n_aug, len(adv_pool)))
                    mixed_train = train_records + adv_sample
                    random.shuffle(mixed_train)
                    log.info(
                        f"{model_key}/{dataset_name}/{regime}/seed{seed} — "
                        f"train={len(train_records)}  aug={len(adv_sample)}  total={len(mixed_train)}"
                    )
                else:
                    mixed_train = train_records
                    log.info(
                        f"{model_key}/{dataset_name}/{regime}/seed{seed} — "
                        f"no adv data found; training on clean only"
                    )

                run_name = f"{model_key}_{dataset_name}_{regime}_seed{seed}"
                ckpt_dir = out_root / run_name

                if cfg.run.wandb:
                    wandb.init(
                        project=os.getenv("WANDB_PROJECT", "destror"),
                        name=run_name,
                        config={
                            "model": model_key,
                            "dataset": dataset_name,
                            "regime": regime,
                            "seed": seed,
                            "n_adv": len(adv_pool),
                        },
                        reinit=True,
                    )

                try:
                    metrics = finetune(
                        model_key=model_key,
                        dataset_name=dataset_name,
                        train_records=mixed_train,
                        eval_records=dev_records,
                        test_records=test_records,
                        output_dir=ckpt_dir.parent,
                        num_epochs=cfg.training.num_epochs,
                        batch_size=batch_size,
                        gradient_accumulation_steps=grad_accum,
                        learning_rate=cfg.training.learning_rate,
                        warmup_ratio=cfg.training.warmup_ratio,
                        weight_decay=cfg.training.weight_decay,
                        max_seq_length=max_seq,
                        fp16=fp16,
                        bf16=bf16,
                        seed=seed,
                        use_lora=use_lora,
                        lora_config=lora_cfg if use_lora else None,
                    )
                    metrics["regime"] = regime
                    all_metrics.append(metrics)

                    if cfg.run.wandb:
                        wandb.log({
                            "clean_f1_macro": metrics["clean_f1_macro"],
                            "regime": regime,
                        })
                except Exception as e:
                    log.error(f"  FAILED: {e}")
                finally:
                    if cfg.run.wandb:
                        wandb.finish()

    merged = existing + [m for m in all_metrics
                         if (m["model"], m["dataset"], m["seed"], m["regime"]) not in existing_keys]
    summary_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Summary written → {summary_path}")


if __name__ == "__main__":
    main()
