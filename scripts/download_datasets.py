"""Download and cache all four datasets.

All datasets sourced from:
  https://github.com/banglanlp/bangla-sentiment-classification

  CogniSenti:   data/CogniSenti/twitter_fbPost_merged_{train,dev,test}.tsv
  BASA_cricket: data/ABSA_Datasets/BASA_cricket_{train,dev,test}.tsv
  YouTube:      data/youtube_sentiment/sentiment_{train,dev,test}.tsv
  BLP23:        provided separately in BanglaClassificationAugment/Dataset/

Notes:
  - Labels are merged across splits and re-split stratified 80/10/10 (seed 42).
  - BASA_cricket labels normalized from lowercase to TitleCase.
  - YouTube: full 2796-sample dataset (original paper only used the 420-sample test split).
"""
import shutil
import pandas as pd
from pathlib import Path

RAW_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Sibling repos
_BANGLA_SC = Path(__file__).resolve().parents[2] / "bangla-sentiment-classification" / "data"
_BCA       = Path(__file__).resolve().parents[2] / "BanglaClassificationAugment" / "Dataset"


def _merge_splits(paths: list[Path], label_col: str = "class_label") -> pd.DataFrame:
    dfs = [pd.read_csv(p, sep="\t") for p in paths]
    df  = pd.concat(dfs, ignore_index=True)
    if label_col in df.columns and label_col != "label":
        df = df.rename(columns={label_col: "label"})
    df["label"] = df["label"].str.strip().str.title()
    df["id"]    = range(len(df))
    return df[["id", "text", "label"]]


def _save(df: pd.DataFrame, name: str):
    out = RAW_DIR / f"{name}.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"  {name}: {len(df)} rows  labels={dict(df['label'].value_counts())}  → {out}")
    # Clear any stale Parquet caches
    for f in PROC_DIR.glob(f"{name}_*.parquet"):
        f.unlink()
        print(f"    cleared cache: {f.name}")


def _copy_blp23():
    src = _BCA / "blp23_sentiment_dev.tsv"
    dst = RAW_DIR / "blp23_sentiment_dev.tsv"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"  blp23: copied from sibling repo → {dst}")
    elif dst.exists():
        print(f"  blp23: already present")
    else:
        print(f"  blp23: NOT FOUND at {src}")


if __name__ == "__main__":
    print("=== Dataset preparation ===\n")

    print("[blp23]")
    _copy_blp23()

    if _BANGLA_SC.exists():
        print("\n[youtube]")
        _save(_merge_splits([
            _BANGLA_SC / "youtube_sentiment/sentiment_train.tsv",
            _BANGLA_SC / "youtube_sentiment/sentiment_dev.tsv",
            _BANGLA_SC / "youtube_sentiment/sentiment_test.tsv",
        ]), "youtube")

        print("\n[cognisenti]")
        _save(_merge_splits([
            _BANGLA_SC / "CogniSenti/twitter_fbPost_merged_train.tsv",
            _BANGLA_SC / "CogniSenti/twitter_fbPost_merged_dev.tsv",
            _BANGLA_SC / "CogniSenti/twitter_fbPost_merged_test.tsv",
        ]), "cognisenti")

        print("\n[basa_cricket]")
        _save(_merge_splits([
            _BANGLA_SC / "ABSA_Datasets/BASA_cricket_train.tsv",
            _BANGLA_SC / "ABSA_Datasets/BASA_cricket_dev.tsv",
            _BANGLA_SC / "ABSA_Datasets/BASA_cricket_test.tsv",
        ]), "basa_cricket")
    else:
        print(f"\nWARNING: bangla-sentiment-classification repo not found at {_BANGLA_SC.parent}")
        print("Clone it with:")
        print("  git clone https://github.com/banglanlp/bangla-sentiment-classification.git")

    print("\nDone. Run `make dataset-stats` to verify.")
