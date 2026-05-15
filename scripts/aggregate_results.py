"""Aggregate all run JSONLs into LaTeX tables and matplotlib figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RAW_DIR = RESULTS_DIR / "raw_runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_runs() -> pd.DataFrame:
    rows = []
    for f in RAW_DIR.glob("*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def asr_table(df: pd.DataFrame) -> pd.DataFrame:
    attackable = df[~df["meta"].apply(lambda m: m.get("skipped", False))]
    pivot = (
        attackable.groupby(["attack", "model", "dataset", "seed"])["success"]
        .mean()
        .reset_index()
        .groupby(["attack", "model", "dataset"])["success"]
        .agg(["mean", "std"])
        .reset_index()
    )
    pivot.columns = ["attack", "model", "dataset", "asr_mean", "asr_std"]
    return pivot


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        df.to_latex(index=False, float_format="%.3f", escape=False),
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    df = load_runs()
    if df.empty:
        print("No run files found in results/raw_runs/. Run experiments first.")
        return

    asr = asr_table(df)
    print(asr.to_string())

    latex = to_latex(asr, "Attack Success Rate (mean ± std over 3 seeds)", "tab:asr")
    (TABLES_DIR / "asr_main.tex").write_text(latex, encoding="utf-8")
    print(f"Wrote {TABLES_DIR / 'asr_main.tex'}")


if __name__ == "__main__":
    main()
