"""Download and cache all four datasets.

blp23 + youtube:  TSVs already in data/raw/ (copied from BanglaClassificationAugment)
CogniSenti:       Request from authors or download from original repo — see note below
basa_cricket:     Available at https://github.com/LanguageTechnologyResearch/BASA

Manual steps for missing datasets:
  CogniSenti:
    Paper: "CogniSenti: A Multi-lingual Multi-task Benchmark for Bangla Cognitive Sentiment Analysis"
    Request data from: https://github.com/NLP-BRTEC/cogni-senti  OR  contact authors
    Place as: data/raw/cognisenti.tsv  (columns: id, text, label)

  BASA_cricket:
    Repo:  https://github.com/LanguageTechnologyResearch/BASA
    Download cricket split and place as: data/raw/basa_cricket.tsv  (columns: id, text, label)

Once files are present, re-run this script to build Parquet caches.
"""
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

_SIBLING = Path(__file__).resolve().parents[2] / "BanglaClassificationAugment" / "Dataset"

# (source_file, dest_file) for sibling-repo copies
_LOCAL_COPIES = [
    (_SIBLING / "blp23_sentiment_dev.tsv",    RAW_DIR / "blp23_sentiment_dev.tsv"),
    (_SIBLING / "youtube_sentiment_test.tsv", RAW_DIR / "youtube_sentiment_test.tsv"),
]


def _copy_local(src: Path, dst: Path):
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"  Copied {src.name} → {dst}")
    elif dst.exists():
        print(f"  Already present: {dst.name}")
    else:
        print(f"  NOT FOUND (sibling repo): {src}")


def _check_manual(dst: Path, instructions: str):
    if dst.exists():
        print(f"  Present: {dst.name}")
    else:
        print(f"  MISSING: {dst.name}")
        print(f"    {instructions}")


if __name__ == "__main__":
    print("=== Dataset download / verify ===\n")

    print("[blp23 + youtube] Copying from BanglaClassificationAugment sibling repo:")
    for src, dst in _LOCAL_COPIES:
        _copy_local(src, dst)

    print("\n[CogniSenti] Manual download required:")
    _check_manual(
        RAW_DIR / "cognisenti.tsv",
        "Get from https://github.com/NLP-BRTEC/cogni-senti and save as data/raw/cognisenti.tsv\n"
        "    Expected columns: id, text, label  (labels: Positive/Negative/Neutral)",
    )

    print("\n[BASA_cricket] Manual download required:")
    _check_manual(
        RAW_DIR / "basa_cricket.tsv",
        "Get from https://github.com/LanguageTechnologyResearch/BASA\n"
        "    Expected columns: id, text, label  (labels: Positive/Negative/Neutral)",
    )

    print("\nDone. Run `make dataset-stats` after all files are present.")
