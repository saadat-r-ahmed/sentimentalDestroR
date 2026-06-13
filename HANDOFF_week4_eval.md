# Handoff — Week 4 robustness matrix eval (paused)

**Paused:** 2026-06-14 ~00:20
**Branch:** `journal-revision`  **HEAD:** `e319059`
**Why paused:** GPU freed for other work. Eval was ~73% through (44/60 cells) + textfooler pending.

---

## TL;DR to resume

```bash
cd /home/maxwell/RESEARCH/destroR/sentimentalDestroR
bash /tmp/run_eval_rob.sh > /tmp/eval_rob.log 2>&1 &
```

If `/tmp/run_eval_rob.sh` is gone (tmp cleared), recreate it:

```bash
cat > /tmp/run_eval_rob.sh << 'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd /home/maxwell/RESEARCH/destroR/sentimentalDestroR
set -a; source .env; set +a
uv run python scripts/eval_robustness.py run.seeds=[42] run.wandb=false hydra.run.dir=/tmp/hydra_eval_rob
echo "=== COMPLETE ==="
SCRIPT
chmod +x /tmp/run_eval_rob.sh
```

**Resume is safe and idempotent.** `eval_robustness.py` folds every existing
`results/adv_eval_runs/*.jsonl` back into the matrix at startup (the
"folded (resume)" log lines) and skips any cell already present. It only runs
missing cells. Nothing already computed is recomputed or lost.

---

## State of the data

- **Completed eval cells live in:** `results/adv_eval_runs/*.jsonl`
  (177 files at pause; one file = one attack × model × dataset × regime cell)
- **Final matrix file:** `results/robustness_matrix.json` — **only written when
  the run finishes cleanly.** It was NOT written before the pause (still `{}`),
  but that's fine: it is fully reconstructed from the JSONL files on resume.

### What's done (4 working attacks: paraphrase, back_translation, one_hot_swap, bae)

| Regime                | Status                  |
|-----------------------|-------------------------|
| `adv_back_translation`| ✅ complete (20/20 cells each attack) |
| `adv_all`             | ✅ complete (20/20 cells each attack) |
| `adv_paraphrase`      | ⏳ ~4–5/20 cells — **in progress when paused** |

So remaining for the 4 attacks: ~15 cells of `adv_paraphrase` × 4 attacks.

### What's NOT done at all: **textfooler** (0/60 cells)

textfooler had two bugs (NoneType mask_token; reading hidden_state as vocab
logits). **Both are fixed and committed in `e319059`.** The original eval run
held the old code in memory, so every textfooler cell failed. On resume, a
fresh process imports the fixed `textfooler.py`, so all 60 textfooler cells
will run correctly (fill-mask based, ~fast).

---

## Time estimate to finish

At full test-set size (~30–40 min/cell):
- ~15 `adv_paraphrase` cells × 4 attacks: a few hours
- 60 textfooler cells: textfooler is fill-mask (faster than back_translation/bae), but still several hours
- **Total: roughly 8–12 hours.**

> Decision on record (2026-06-13): user chose **full test set, no sample cap**
> for maximum rigor. Do not add `dataset.max_samples` unless the user changes this.

---

## After the matrix completes

1. Confirm `results/robustness_matrix.json` has all cells:
   - 4 regimes (clean + 3 adv) × 5 models × 4 datasets × 5 attacks.
   - `clean` rows are folded in from `results/raw_runs/` if present.
2. The script prints an ASCII summary table (mean ASR per regime × attack) at the end.
3. **Commit** the matrix + new `adv_eval_runs` files:
   ```bash
   git add results/robustness_matrix.json results/adv_eval_runs/ scripts/eval_robustness.py
   git commit -m "feat(week-4): complete robustness matrix (5 attacks × 4 regimes)"
   git push
   ```
   (No Claude co-author line — house rule.)
4. **Ping the user** — they asked to be notified when the matrix is done.

---

## Known early findings (from the 44 completed cells)

- **BAE** is the strongest attack (~0.58–0.66 ASR even post-defense).
- **one_hot_swap** is weakest (~0.00 ASR) — consistent with its near-zero ASR
  on the train split during gen_adv_data. Worth a paper note / possible attack fix
  (gradient importance + larger candidate pool) before writing.
- Adversarial training measurably lowers paraphrase / back_translation success.

---

## Pipeline recap (Week 4 scripts)

| Stage | Script | Output | Status |
|-------|--------|--------|--------|
| 1 | `scripts/gen_adv_data.py` | `results/adv_train_data/*.jsonl` (60 files) | ✅ done |
| 2 | `scripts/adv_train.py` | `results/adv_trained_models/{regime}/...` (60 runs) | ✅ done |
| 3 | `scripts/eval_robustness.py` | `results/robustness_matrix.json` | ⏳ ~73% + textfooler pending |

Regimes trained: `adv_paraphrase`, `adv_back_translation`, `adv_all`
(adv_one_hot_swap standalone was skipped — its train-split ASR was ~0.3%, so it
contributes ~no augmentation; it's still included inside `adv_all`).

## Disk note

Earlier hit 100% disk from `optimizer.pt` checkpoint files (~162 GB). Fixed in
commit `7e93e1e`: `save_total_limit=1` + auto-delete optimizer/scheduler/rng
states after training. Currently ~156 GB free. If resuming heavy training,
keep an eye on `df -h`.
