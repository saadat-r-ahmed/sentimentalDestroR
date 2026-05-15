"""Download and cache all four datasets. Run once before experiments."""
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Copy TSVs from BanglaClassificationAugment if present (dev shortcut)
_SIBLING = Path(__file__).resolve().parents[2] / "BanglaClassificationAugment" / "Dataset"


def _copy_local(src: Path, dst: Path):
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"Copied {src.name} → {dst}")


def _hf_download(hf_path: str, name: str):
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_path)
        for split_name, split_ds in ds.items():
            out = RAW_DIR / f"{name}_{split_name}.parquet"
            if not out.exists():
                split_ds.to_pandas().to_parquet(out, index=False)
                print(f"Downloaded {hf_path} ({split_name}) → {out}")
    except Exception as e:
        print(f"WARNING: Could not download {hf_path}: {e}")


if __name__ == "__main__":
    # 1. Copy local TSVs if sibling repo exists
    _copy_local(_SIBLING / "blp23_sentiment_dev.tsv",     RAW_DIR / "blp23_sentiment_dev.tsv")
    _copy_local(_SIBLING / "youtube_sentiment_test.tsv",  RAW_DIR / "youtube_sentiment_test.tsv")

    # 2. Download from HuggingFace (will no-op if already cached by datasets library)
    _hf_download("BanglaLLP/blp23-sentiment",   "blp23")
    _hf_download("BanglaLLP/youtube-sentiment", "youtube")
    _hf_download("BanglaLLP/CogniSenti",        "cognisenti")
    _hf_download("BanglaLLP/BASA-cricket",      "basa_cricket")

    print("Done. Check data/raw/ for downloaded files.")
