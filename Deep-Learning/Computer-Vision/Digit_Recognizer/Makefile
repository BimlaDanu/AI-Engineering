# ═══════════════════════════════════════════════════════════
#  MNIST MLP — Makefile
#
#  One uv-managed environment in .venv. Runtime dependencies and the
#  dev tools (pytest, ruff, mypy) are declared in pyproject.toml and
#  pinned in uv.lock. Commit the lock file.
#
#  VIRTUAL_ENV is stripped so uv always uses its own .venv.
#
#  Usage:
#    make help       -> list every target (the default)
#    make sync       -> create/update .venv from pyproject.toml + uv.lock
#    make data       -> download the Kaggle competition CSVs
#    make check      -> lint + typecheck + test (the definition of done)
#    make experiments-> run the hyperparameter study
#    make evaluate   -> score the best model on the held-out test split
#    make ensemble   -> train a seed ensemble and compare it with TTA
# ═══════════════════════════════════════════════════════════

UV := env -u VIRTUAL_ENV uv
# Which size `make notebook-run` asks the notebook for: 0 is the full study.
QUICK ?= 0
SRC := src tests
PROJECT_NAME := bidanu-NLP.AI.1.5

.DEFAULT_GOAL := help

## help: show this message
help:
	@echo "MNIST MLP — available targets:"
	@echo
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  make /' | column -t -s ':'
	@echo

# ─────────────────── environment ───────────────────

## sync: create/update .venv from pyproject.toml + uv.lock
sync:
	$(UV) sync

## sync-ci: install exactly what uv.lock pins, no re-resolution
sync-ci:
	$(UV) sync --frozen

## lock: re-resolve dependencies and rewrite uv.lock
lock:
	$(UV) lock

## versions: print the interpreter, the accelerator and the key packages
versions:
	@$(UV) run python -m src.utils

# ─────────────────── data ───────────────────

## data: put the digit-recognizer CSVs in data/raw, downloading only if needed
#
# The download is skipped when train.csv or the committed train.csv.gz is
# already there. A fresh clone always has the gzip, so this target is a no-op
# on one, and no machine needs Kaggle credentials just to start training.
data:
	@if [ -f data/raw/train.csv ] || [ -f data/raw/train.csv.gz ]; then \
		echo "The training data is already here — skipping the download."; \
		echo "Verify it with 'make data-check', or re-fetch with 'make data-force'."; \
	else \
		$(MAKE) --no-print-directory data-force; \
	fi

## data-force: download the CSVs from Kaggle even if data/raw already has them
#
# Needs the competition rules accepted on the website and the Kaggle CLI
# authenticated: `kaggle auth login` (OAuth, nothing to store), or a token from
# https://www.kaggle.com/settings/api in KAGGLE_API_TOKEN or ~/.kaggle/access_token.
# Without one of those the CLI stops with an authentication notice and writes
# nothing; downloading train.csv by hand into data/raw/ changes nothing else.
data-force:
	$(UV) run --group data kaggle competitions download -c digit-recognizer -p data/raw
	cd data/raw && unzip -o digit-recognizer.zip && rm -f digit-recognizer.zip

## data-gz: rebuild the committed data/raw/*.csv.gz from the plain CSVs
#
# Only needed after a fresh download. The competition data never changes, so
# this is not part of any other target.
data-gz:
	gzip -9 -c data/raw/train.csv > data/raw/train.csv.gz
	gzip -9 -c data/raw/test.csv > data/raw/test.csv.gz
	@ls -lh data/raw/train.csv.gz data/raw/test.csv.gz

## bundle: zip exactly the files git tracks -- the same set a push sends
#
# Uploading the project folder as it stands means uploading .venv, which is
# 1.2 GB and useless on a cloud image that builds its own. Asking git which
# files count keeps this honest: there is no hand-written list to fall behind
# the project, and nothing gitignored -- .env, reports, checkpoints, caches --
# can reach an upload by accident.
#
# It zips the working copy rather than HEAD, so uncommitted edits travel too.
BUNDLE := $(PROJECT_NAME).zip

bundle:
	@rm -f $(BUNDLE)
	git ls-files -z | xargs -0 zip -q $(BUNDLE)
	@ls -lh $(BUNDLE)
	@echo "$$(git ls-files | wc -l | tr -d ' ') tracked files. Upload this one file, not the folder."

## data-check: report what is in data/raw and whether it is usable
data-check:
	$(UV) run python -m src.data

# ─────────────────── quality gates ───────────────────

## format: auto-format and auto-fix sources with Ruff
format:
	$(UV) run ruff format $(SRC)
	$(UV) run ruff check --fix $(SRC)

## lint: check sources with Ruff, writing nothing
lint:
	$(UV) run ruff check $(SRC)
	$(UV) run ruff format --check $(SRC)

## typecheck: static type check with mypy
typecheck:
	$(UV) run mypy

## test: run the unit tests
test:
	$(UV) run pytest

## test-fast: skip the tests that actually train, for a tight edit loop
test-fast:
	$(UV) run pytest -m "not slow"

## check: lint + typecheck + test
check: lint typecheck test

# ─────────────────── modelling ───────────────────

## train: train one model with the baseline configuration
#
# Override any field from the command line, for example:
#   make train ARGS="--hidden-sizes 512 256 --dropout 0.3 --optimizer adamw"
train:
	$(UV) run python -m src.train $(ARGS)

## experiments: run the hyperparameter study and write reports/experiments.csv
#
# One factor at a time from a fixed baseline, repeated over seeds, so that a
# difference between two settings can be read against its own run-to-run spread.
experiments:
	$(UV) run python -m src.experiments $(ARGS)

## evaluate: score the selected model on the held-out test split
evaluate:
	$(UV) run python -m src.evaluate $(ARGS)

## ensemble: train a seed ensemble and compare it with test-time augmentation
ensemble:
	$(UV) run python -m src.ensemble $(ARGS)

## compare-cnn: train the reference CNN and put it beside the MLP
compare-cnn:
	$(UV) run python -m src.reference_cnn $(ARGS)

## notebook: open the results notebook
notebook:
	$(UV) run jupyter lab notebooks/

## notebook-run: execute every cell in order and save the outputs into the file
#
# `make notebook` is the browser route. This one is for a GPU session: it runs
# the notebook start to finish and writes the outputs back, which is what a
# reader of the committed file sees. It takes hours at QUICK=0.
#
#   make notebook-run           # the full study
#   make notebook-run QUICK=1   # the short pipeline check, minutes
notebook-run:
	MNIST_MLP_QUICK=$(QUICK) $(UV) run jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=-1 --inplace notebooks/mnist_mlp_report.ipynb

# ─────────────────── housekeeping ───────────────────

## clean: remove the .venv and all tool caches (data and reports are kept)
clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Clean done. Data, reports and system packages untouched."

## clean-runs: remove training artifacts (checkpoints and logs)
clean-runs:
	rm -rf checkpoints lightning_logs runs
	@echo "Training artifacts removed."

.PHONY: help sync sync-ci lock versions data data-force data-gz data-check bundle format lint typecheck test test-fast \
	check train experiments evaluate ensemble compare-cnn notebook notebook-run clean clean-runs
