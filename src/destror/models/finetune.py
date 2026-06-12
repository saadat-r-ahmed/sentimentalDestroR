"""Finetune a pretrained model for Bangla sentiment classification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from destror.utils.logging import get_logger

log = get_logger(__name__)

# Maps model key → HF model id (same as loaders.py)
_BASE_MODELS = {
    "banglabert":   "csebuetnlp/banglabert",
    "banglishbert": "csebuetnlp/banglishbert",
    "xlm-r":        "xlm-roberta-base",
    "muril":        "google/muril-base-cased",
    "indicbertv2":  "ai4bharat/IndicBERTv2-MLM-only",
    "titulm-1b":    "hishab/titulm-llama-3.2-1b-v2.0",
}


def finetune(
    model_key: str,
    dataset_name: str,
    train_records: list[dict],
    eval_records: list[dict],
    test_records: list[dict],
    output_dir: Path,
    num_epochs: int = 5,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_seq_length: int = 256,
    fp16: bool = True,
    bf16: bool = False,
    seed: int = 42,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    use_lora: bool = False,
    lora_config: Optional[dict] = None,
) -> dict:
    """Finetune model_key on the given splits. Returns a metrics dict."""

    model_name = _BASE_MODELS.get(model_key, model_key)
    log.info(f"Finetuning {model_name} on {dataset_name} (seed={seed})")

    # Build label vocabulary from training data
    all_labels = sorted(set(r["label"] for r in train_records + eval_records + test_records))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(all_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Llama-based models have no pad token — use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(records: list[dict]) -> Dataset:
        texts = [r["text"] for r in records]
        labels = [label2id[r["label"]] for r in records]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        enc["labels"] = labels
        return Dataset.from_dict(enc)

    train_ds = tokenize(train_records)
    eval_ds  = tokenize(eval_records)
    test_ds  = tokenize(test_records)

    # QLoRA: load 4-bit quantized base for LLMs (use_lora=True implies QLoRA on GPU)
    if use_lora and torch.cuda.is_available():
        model = _load_qlora_model(model_name, num_labels, id2label, label2id, lora_config or {})
    else:
        load_dtype = torch.bfloat16 if bf16 else None
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            dtype=load_dtype,
        )

    # Llama has no pad token — align model config with tokenizer
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    run_name = f"{model_key}_{dataset_name}_seed{seed}"
    ckpt_dir = output_dir / run_name

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=fp16 and torch.cuda.is_available() and not bf16 and not use_lora,
        bf16=bf16 and torch.cuda.is_available() and not use_lora,
        gradient_checkpointing=use_lora,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=seed,
        report_to="wandb",
        run_name=run_name,
        logging_steps=50,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Delete optimizer/scheduler states from the saved checkpoint — not needed
    # for inference and each one is ~400MB (same size as the model weights).
    import glob as _glob
    for _f in _glob.glob(str(ckpt_dir / "**" / "optimizer.pt"), recursive=True) + \
               _glob.glob(str(ckpt_dir / "**" / "scheduler.pt"), recursive=True) + \
               _glob.glob(str(ckpt_dir / "**" / "rng_state*.pth"), recursive=True):
        Path(_f).unlink(missing_ok=True)

    # Evaluate on test set
    test_preds_out = trainer.predict(test_ds)
    test_preds = np.argmax(test_preds_out.predictions, axis=-1)
    test_labels = [label2id[r["label"]] for r in test_records]
    test_report = classification_report(
        test_labels, test_preds,
        target_names=all_labels,
        output_dict=True,
    )

    metrics = {
        "model": model_key,
        "dataset": dataset_name,
        "seed": seed,
        "clean_f1_macro": test_report["macro avg"]["f1-score"],
        "clean_f1_weighted": test_report["weighted avg"]["f1-score"],
        "per_class": {l: test_report[l] for l in all_labels},
        "label2id": label2id,
    }

    # Save metrics alongside checkpoint
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"  clean F1 (macro): {metrics['clean_f1_macro']:.4f}")

    if push_to_hub and hub_model_id:
        trainer.push_to_hub()

    return metrics


def _load_qlora_model(model_name, num_labels, id2label, label2id, lora_config: dict):
    """Load model in 4-bit (NF4) + LoRA adapters for 8GB VRAM."""
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def _apply_lora(model, lora_config: dict):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    except ImportError:
        log.warning("peft not installed — running full finetune for LLM victim (not LoRA)")
    return model
