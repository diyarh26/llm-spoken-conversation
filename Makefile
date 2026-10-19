# Common tasks. Run `make help` for the list. Windows users without make can copy the commands.
PY ?= python

.PHONY: help install test lint format figures poster-figures explorer serve analysis generate clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the analysis + dev dependencies (editable)
	$(PY) -m pip install -e ".[analysis,dev]"

test:  ## Run the test suite (no GPU or licensed data needed)
	$(PY) -m pytest

lint:  ## Lint and check formatting
	ruff check . && ruff format --check .

format:  ## Auto-format and apply safe lint fixes
	ruff format . && ruff check --fix .

figures:  ## Rebuild the README figures from the committed results
	$(PY) analysis/figures.py

poster-figures:  ## Rebuild the poster figures from the committed results
	$(PY) analysis/poster_figures.py

explorer:  ## Build the conversation explorer's data bundle
	$(PY) explorer/build.py

serve: explorer  ## Serve the explorer locally at http://localhost:8000
	$(PY) -m http.server 8000 --directory explorer

analysis:  ## Recompute metrics and statistics (needs the licensed Switchboard corpus)
	$(PY) analysis/evaluate_generated.py --data_dir data/generated --out_dir results --switchboard
	$(PY) analysis/stats.py --data-dir data/generated > results/stats_report.txt
	$(PY) analysis/alignment_checks.py --out results
	$(PY) analysis/transition_weighted.py results/dialogue_acts

generate:  ## Generate all 12 conditions (CUDA GPU; resumable, run inside tmux)
	bash generation/run_all.sh

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build *.egg-info explorer/data
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
