"""Dataset loaders with stratified splits and Parquet caching."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[4] / "data"

# HuggingFace dataset identifiers for downloadable datasets
_HF_DATASETS = {
    "blp23":          {"path": "BanglaLLP/blp23-sentiment", "text_col": "text", "label_col": "label"},
    "youtube":        {"path": "BanglaLLP/youtube-sentiment", "text_col": "text", "label_col": "label"},
    "cognisenti":     {"path": "BanglaLLP/CogniSenti", "text_col": "text", "label_col": "label"},
    "basa_cricket":   {"path": "BanglaLLP/BASA-cricket", "text_col": "text", "label_col": "label"},
}

# Local TSV fallbacks (from BanglaClassificationAugment)
_LOCAL_TSV = {
    "blp23":   DATA_DIR / "raw" / "blp23_sentiment_dev.tsv",
    "youtube": DATA_DIR / "raw" / "youtube_sentiment_test.tsv",
}


def load_dataset(name: str, split: str = "test", seed: int = 42) -> list[dict]:
    """Load a named dataset, returning a list of {id, text, label} dicts.

    Falls back to local TSV for blp23/youtube if HF download fails.
    """
    records = _load_records(name, split, seed)
    return records


def _load_records(name: str, split: str, seed: int) -> list[dict]:
    parquet_path = DATA_DIR / "processed" / f"{name}_{split}_seed{seed}.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return df.to_dict(orient="records")

    df = _fetch_raw(name)
    df = _normalize_columns(df, name)
    df = _add_splits(df, seed)
    df = df[df["split"] == split].reset_index(drop=True)
    df["id"] = df.index.astype(str)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return df.to_dict(orient="records")


def _fetch_raw(name: str) -> pd.DataFrame:
    # Try HuggingFace first
    try:
        from datasets import load_dataset as hf_load
        cfg = _HF_DATASETS[name]
        ds = hf_load(cfg["path"])
        df = ds["train"].to_pandas()
        return df
    except Exception:
        pass

    # Try local TSV
    if name in _LOCAL_TSV and _LOCAL_TSV[name].exists():
        return pd.read_csv(_LOCAL_TSV[name], sep="\t")

    raise FileNotFoundError(
        f"Dataset '{name}' not found locally or on HuggingFace. "
        f"Run `make download-data` first."
    )


def _normalize_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cfg = _HF_DATASETS.get(name, {})
    text_col = cfg.get("text_col", "text")
    label_col = cfg.get("label_col", "label")

    # blp23 TSV has 'sentence' instead of 'text'
    for alias in ["sentence", "Sentence", "review", "Review"]:
        if alias in df.columns and text_col not in df.columns:
            df = df.rename(columns={alias: "text"})
            break

    df = df.rename(columns={text_col: "text", label_col: "label"}, errors="ignore")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()
    return df


def _add_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Stratified 80/10/10 train/dev/test split."""
    labels = df["label"]
    idx = df.index.tolist()

    train_idx, tmp_idx = train_test_split(idx, test_size=0.20, stratify=labels[idx], random_state=seed)
    dev_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, stratify=labels[tmp_idx], random_state=seed
    )

    df["split"] = "train"
    df.loc[dev_idx, "split"] = "dev"
    df.loc[test_idx, "split"] = "test"
    return df
