"""Compute and print dataset statistics required by revision.md §1.2."""
import re
import json
from pathlib import Path

from destror.datasets.loaders import load_dataset

DATASETS = ["blp23", "youtube", "cognisenti", "basa_cricket"]
SEED = 42

# Rough Bangla Unicode range: U+0980–U+09FF
_BANGLA_RE = re.compile(r"[ঀ-৿]")
_LATIN_RE  = re.compile(r"[A-Za-z]")


def code_mix_rate(text: str) -> float:
    """Fraction of characters that are Latin (proxy for code-mixing)."""
    if not text:
        return 0.0
    latin = len(_LATIN_RE.findall(text))
    return latin / len(text)


def avg_tokens(texts: list[str]) -> float:
    return sum(len(t.split()) for t in texts) / len(texts)


def avg_chars(texts: list[str]) -> float:
    return sum(len(t) for t in texts) / len(texts)


def main():
    stats = {}
    for name in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        try:
            all_records = (
                load_dataset(name, split="train", seed=SEED)
                + load_dataset(name, split="dev",   seed=SEED)
                + load_dataset(name, split="test",  seed=SEED)
            )
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")
            continue

        texts  = [r["text"]  for r in all_records]
        labels = [r["label"] for r in all_records]

        from collections import Counter
        label_dist = Counter(labels)
        cm_rates   = [code_mix_rate(t) for t in texts]

        split_counts = {}
        for split in ["train", "dev", "test"]:
            split_counts[split] = len(load_dataset(name, split=split, seed=SEED))

        ds_stats = {
            "n_total": len(all_records),
            "splits": split_counts,
            "label_distribution": dict(label_dist),
            "avg_tokens": round(avg_tokens(texts), 2),
            "avg_chars": round(avg_chars(texts), 2),
            "code_mix_rate_mean": round(sum(cm_rates) / len(cm_rates), 4),
            "code_mix_rate_p90": round(
                sorted(cm_rates)[int(0.90 * len(cm_rates))], 4
            ),
        }
        stats[name] = ds_stats

        print(f"  Total:         {ds_stats['n_total']}")
        print(f"  Splits:        {split_counts}")
        print(f"  Labels:        {label_dist}")
        print(f"  Avg tokens:    {ds_stats['avg_tokens']}")
        print(f"  Avg chars:     {ds_stats['avg_chars']}")
        print(f"  Code-mix rate: {ds_stats['code_mix_rate_mean']:.3%} (mean), "
              f"{ds_stats['code_mix_rate_p90']:.3%} (p90)")

    out = Path("results") / "dataset_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStats written → {out}")


if __name__ == "__main__":
    main()
