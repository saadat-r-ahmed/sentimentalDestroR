"""Finetune all victim models on all datasets. Week 1 entry point."""
import os
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from destror.utils.seeding import set_seed
from destror.utils.logging import get_logger

log = get_logger(__name__)

RESULTS_DIR = Path("results/finetuned_models")


@hydra.main(version_base=None, config_path="../configs", config_name="training/finetune")
def main(cfg: DictConfig) -> None:
    import wandb
    from destror.datasets.loaders import load_dataset
    from destror.models.finetune import finetune

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    # titulm-1b uses LoRA; all others are full finetune
    lora_models = {"titulm-1b"}

    for seed in cfg.run.seeds:
        set_seed(seed)

        for model_key in cfg.model.keys:
            use_lora = model_key in lora_models

            # Apply per-model hyperparameter overrides (e.g. smaller batch for LLMs)
            overrides = dict(cfg.model_overrides.get(model_key, {})) if hasattr(cfg, "model_overrides") else {}
            batch_size  = overrides.get("batch_size",  cfg.training.batch_size)
            grad_accum  = overrides.get("gradient_accumulation_steps", cfg.training.gradient_accumulation_steps)
            max_seq_len = overrides.get("max_seq_length", cfg.training.max_seq_length)

            for dataset_name in cfg.dataset.names:
                run_id = f"{model_key}_{dataset_name}_seed{seed}"
                log.info(f"==> {run_id}")

                if cfg.run.wandb:
                    wandb.init(
                        project=os.getenv("WANDB_PROJECT", "destror"),
                        name=f"finetune_{run_id}",
                        config={
                            "model": model_key,
                            "dataset": dataset_name,
                            "seed": seed,
                            "phase": "finetune",
                        },
                        reinit=True,
                    )

                try:
                    train = load_dataset(dataset_name, split="train", seed=seed)
                    dev   = load_dataset(dataset_name, split="dev",   seed=seed)
                    test  = load_dataset(dataset_name, split="test",  seed=seed)

                    hub_id = None
                    if cfg.run.push_to_hub:
                        hub_id = f"{cfg.run.hub_org}/destror-{model_key}-{dataset_name}"

                    metrics = finetune(
                        model_key=model_key,
                        dataset_name=dataset_name,
                        train_records=train,
                        eval_records=dev,
                        test_records=test,
                        output_dir=RESULTS_DIR,
                        num_epochs=cfg.training.num_epochs,
                        batch_size=batch_size,
                        gradient_accumulation_steps=grad_accum,
                        learning_rate=cfg.training.learning_rate,
                        warmup_ratio=cfg.training.warmup_ratio,
                        weight_decay=cfg.training.weight_decay,
                        max_seq_length=max_seq_len,
                        fp16=cfg.training.fp16,
                        seed=seed,
                        push_to_hub=cfg.run.push_to_hub,
                        hub_model_id=hub_id,
                        use_lora=use_lora,
                        lora_config=dict(cfg.lora) if use_lora else None,
                    )

                    all_metrics.append(metrics)

                    if cfg.run.wandb:
                        wandb.log({
                            "clean_f1_macro": metrics["clean_f1_macro"],
                            "clean_f1_weighted": metrics["clean_f1_weighted"],
                        })

                except Exception as e:
                    log.error(f"  FAILED {run_id}: {e}")
                    import traceback; traceback.print_exc()
                finally:
                    if cfg.run.wandb:
                        wandb.finish()

    # Write summary table
    summary_path = RESULTS_DIR / "clean_f1_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Summary written → {summary_path}")

    # Print clean-F1 table
    print("\n=== Clean F1 (macro) ===")
    print(f"{'model':<22} {'dataset':<16} {'seed':<6} {'F1':<8}")
    print("-" * 56)
    for m in all_metrics:
        print(f"{m['model']:<22} {m['dataset']:<16} {m['seed']:<6} {m['clean_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
