"""Victim model loaders — returns a VictimFn callable for each model."""
from __future__ import annotations

from typing import Optional
import torch

# Canonical victim model roster (revision.md §1.1)
VICTIM_MODELS = {
    "banglabert":       "csebuetnlp/banglabert",
    "banglishbert":     "csebuetnlp/banglishbert",
    "xlm-r":            "xlm-roberta-base",
    "muril":            "google/muril-base-cased",
    "indicbertv2":      "ai4bharat/IndicBERTv2-MLM-only",
    "titulm-1b":        "hishab/titulm-llama-3.2-1b",
    # HF fine-tuned checkpoints (uploaded after Week 1 finetuning)
    "banglabert-finetuned":   "saadat-r-ahmed/banglabert-destror",
    "banglishbert-finetuned": "saadat-r-ahmed/banglishbert-destror",
    "xlm-r-finetuned":        "saadat-r-ahmed/xlm-r-destror",
    "muril-finetuned":        "saadat-r-ahmed/muril-destror",
    "indicbertv2-finetuned":  "saadat-r-ahmed/indicbertv2-destror",
    # Existing community fine-tune (used in original paper)
    "ka05ar":           "ka05ar/banglabert-sentiment",
}


def load_victim(model_key: str, label_map: Optional[dict] = None):
    """Return a VictimFn: text → (label, confidence).

    If model_key is a HF path not in VICTIM_MODELS, it is used directly.
    """
    model_name = VICTIM_MODELS.get(model_key, model_key)
    return _PipelineVictim(model_name, label_map)


class _PipelineVictim:
    """Wraps a HuggingFace text-classification pipeline as a VictimFn."""

    def __init__(self, model_name: str, label_map: Optional[dict]):
        self.model_name = model_name
        self.label_map = label_map or {}
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=device,
            truncation=True,
            max_length=512,
        )

    def __call__(self, text: str) -> tuple[str, float]:
        self._load()
        result = self._pipeline(text)[0]
        label = result["label"]
        for k, v in self.label_map.items():
            label = label.replace(k, v)
        return label, result["score"]
