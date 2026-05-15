.PHONY: install test lint reproduce smoke clean

PYTHON = python
UV = uv

install:
	$(UV) sync --all-extras

test:
	$(UV) run pytest tests/ -v

lint:
	$(UV) run ruff check src/ tests/
	$(UV) run ruff format --check src/ tests/

# ---------- Week 1 ----------
download-data:
	$(UV) run python scripts/download_datasets.py

dataset-stats:
	$(UV) run python scripts/dataset_stats.py

finetune:
	$(UV) run python scripts/finetune_victim.py

# ---------- Week 2-3 ----------
# Run full experiment pipeline (used for reproducibility audit)
reproduce:
	@echo "==> Reproducing all main-text results..."
	$(UV) run python scripts/run_attack.py   --config-name attack/paraphrase
	$(UV) run python scripts/run_attack.py   --config-name attack/back_translation
	$(UV) run python scripts/run_attack.py   --config-name attack/one_hot_swap
	$(UV) run python scripts/aggregate_results.py

# Smoke test: 10 samples, fast CI check
smoke:
	$(UV) run python scripts/run_attack.py --config-name attack/paraphrase dataset.max_samples=10 hydra.run.dir=results/smoke
	$(UV) run python scripts/run_attack.py --config-name attack/back_translation dataset.max_samples=10 hydra.run.dir=results/smoke

dvc-pull:
	dvc pull

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
