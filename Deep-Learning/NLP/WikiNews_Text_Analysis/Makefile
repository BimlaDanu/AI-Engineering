# WikiNews NLP — build and analysis targets.
#
# The pipeline targets follow the requirements in AE.NLP.2.5.md, in dependency
# order. Each stage reads the previous stage's file from data/processed and
# writes its own, so any stage can be re-run alone.

# Drop an activated VIRTUAL_ENV so uv uses .venv rather than warning about a
# mismatch with whatever the shell has sourced.
UV := env -u VIRTUAL_ENV uv
RUN := $(UV) run

# Directories Ruff and mypy check
SRC := src tests

# One pipeline per language, all the `sm` size: requirement 3 compares NER
# accuracy across languages, which only holds if model capacity is constant.
# Installed by `make sync` from the `models` dependency group — a pipeline
# fetched with `spacy download` is declared nowhere and the next sync prunes it.
SPACY_MODELS := en_core_web_sm es_core_news_sm fr_core_news_sm de_core_news_sm

# The corpus. 15,200 articles in 33 languages, ~57 MB, not tracked by git.
CORPUS_REPO := https://github.com/PrimerAI/WikiNews-multilingual.git
CORPUS_DIR  := data/raw/wikinews
CORPUS_FILE := $(CORPUS_DIR)/multilingual_wikinews.jsonl

.DEFAULT_GOAL := help

## help: show this message
help:
	@echo "WikiNews NLP — available targets:"
	@echo
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  make /' | column -t -s ':'
	@echo

# ─────────────────── environment ───────────────────

## setup: everything a fresh clone needs — deps, spaCy models, corpus
setup: sync models data
	@echo "Setup complete. Run 'make all' to build the analysis."

## sync: create/update the .venv from pyproject.toml + uv.lock
sync:
	$(UV) sync

## lock: re-resolve dependencies and rewrite uv.lock (commit the result)
lock:
	$(UV) lock

## sync-ci: install exactly what uv.lock pins — a stale lock is an error
sync-ci:
	$(UV) sync --frozen

## models: check the four spaCy pipelines are installed and loadable
#
# Fails here rather than as an OSError six stages into a run.
models:
	@$(RUN) python -c "import spacy; \
[spacy.load(m) for m in '$(SPACY_MODELS)'.split()]; \
print('spaCy pipelines OK: $(SPACY_MODELS)')"

## data: fetch the WikiNews corpus into data/raw (skipped if already present)
data:
	@if [ -f "$(CORPUS_FILE)" ]; then \
		echo "Corpus already present at $(CORPUS_FILE)"; \
	else \
		echo "Cloning corpus into $(CORPUS_DIR)"; \
		git clone --depth 1 $(CORPUS_REPO) $(CORPUS_DIR); \
	fi

## versions: print the interpreter and the packages the analysis depends on
versions:
	@$(RUN) python -c "import sys, numpy, spacy; \
print(f'python  {sys.version.split()[0]}'); \
print(f'numpy   {numpy.__version__}'); \
print(f'spacy   {spacy.__version__}')"
	@$(RUN) python -c "import spacy.util as u; \
print('models  ' + ', '.join(sorted(u.get_installed_models())))"
	@$(RUN) mypy --version
	@$(RUN) ruff --version
	@$(RUN) pytest --version

# ─────────────────── quality gates ───────────────────

## format: auto-format sources with Ruff
format:
	$(RUN) ruff format $(SRC)
	$(RUN) ruff check --fix $(SRC)

## lint: check sources with Ruff (no changes written)
lint:
	$(RUN) ruff check $(SRC)
	$(RUN) ruff format --check $(SRC)

## typecheck: static type check with mypy
typecheck:
	$(RUN) mypy

## test: run the unit tests
test:
	$(RUN) pytest

## test-fast: the same tests without the corpus runs or the API calls
test-fast:
	$(RUN) pytest -m "not slow and not live"

## check: everything CI will run — lint, typecheck, test
check: lint typecheck test

# ─────────────────── analysis pipeline ───────────────────

## corpus: select the analysed categories and languages, write the working set
#
# Also backfills ISO dates onto non-English articles from their English
# counterpart; the raw file dates only English pages that way.
corpus:
	$(RUN) python -m src.pipeline.corpus

## preprocess: tokenize, sentence-split, POS-tag and lemmatize (requirement 1)
preprocess:
	$(RUN) python -m src.pipeline.preprocess

## ner: extract named entities with their article metadata (requirement 2)
ner:
	$(RUN) python -m src.pipeline.ner

## ner-analysis: aggregate entities, trace them over time, cluster them
ner-analysis:
	$(RUN) python -m src.pipeline.ner_analysis

## ner-errors: investigate wrongly predicted entities across languages (requirement 3)
#
# Articles sharing a pageid describe the same event, so a name found in English
# and missed in the linked article is a candidate miss.
ner-errors:
	$(RUN) python -m src.pipeline.ner_errors

## summarize: extractive and abstractive summaries per category (requirement 4)
#
# Extractive runs offline; abstractive needs OPENROUTER_API_KEY. Run one with
# SUMMARIZER=extractive.
SUMMARIZER ?= both
summarize:
	$(RUN) python -m src.pipeline.summarize --summarizer $(SUMMARIZER)

## grammar: readability and style statistics for the summaries
grammar:
	$(RUN) python -m src.pipeline.grammar

## similarity: score summaries against their source articles (requirement 5)
similarity:
	$(RUN) python -m src.pipeline.similarity

## topics: predict the news category for held-out articles
topics:
	$(RUN) python -m src.pipeline.topics

## figures: render every figure into reports/figures
figures:
	$(RUN) python -m src.pipeline.figures

## report: write reports/findings.md from the stage outputs
report:
	$(RUN) python -m src.pipeline.report

## all: run the full pipeline in dependency order
all: corpus preprocess ner ner-analysis ner-errors summarize grammar similarity topics figures report
	@echo "Pipeline complete. See reports/findings.md"

## notebook: open the analysis notebook
notebook:
	$(RUN) jupyter lab notebooks/

# ─────────────────── housekeeping ───────────────────

## clean-outputs: delete generated data and reports, keep the raw corpus
clean-outputs:
	rm -rf data/processed data/interim reports/figures reports/findings.md
	@echo "Generated outputs removed. Raw corpus untouched."

## clean: remove the uv-managed .venv and all tool caches
clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -not -path "./.git/*" -exec rm -rf {} +
	@echo "Clean done. System packages untouched."

.PHONY: help setup sync lock sync-ci models data versions \
	format lint typecheck test test-fast check \
	corpus preprocess ner ner-analysis ner-errors summarize grammar \
	similarity topics figures report all notebook \
	clean-outputs clean
