# ═══════════════════════════════════════════════════════════
#   Interview Practice app — Makefile
#
#  Single environment: the uv-managed .venv, built from pyproject.toml
#  and uv.lock (`make sync`). It holds both the app runtime deps
#  (streamlit, openai, pypdf, python-dotenv) and the dev tools
#  (pytest, ruff), so every target below goes through uv.
#
#  We strip any activated VIRTUAL_ENV first, so uv always resolves to its
#  own .venv and never warns about a mismatched or stale environment.
#
#  Usage:
#    make         -> show this target list (same as `make help`)
#    make help    -> show this target list
#    make run     -> run the Streamlit app
#    make eval    -> compare prompt techniques    (REAL API calls)
#    make test    -> run the unit tests           (no network)
#    make lint    -> run Ruff over the sources
#    make format  -> auto-format + sort imports (REWRITES files)
#    make check   -> lint + test, same as CI
#    make sync    -> create/update the .venv from the lockfile
#    make clean   -> remove the .venv
# ═══════════════════════════════════════════════════════════

# Call uv with any stale VIRTUAL_ENV removed, so it always uses its own .venv.
UV := env -u VIRTUAL_ENV uv

# The files Ruff lints and formats (same list for both, so they can't drift).
SRC := Iapp.py auth.py core.py categories.py evals.py jailbreak_experiment.py tests

# Running `make` with no target prints the help list instead of
# launching the app. Delete the next line to make `run` the default again.
.DEFAULT_GOAL := help

help:  ## show this list of targets
	@echo "Interview Practice app — available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'
	@echo ""

run:  ## run the Streamlit app
	$(UV) run streamlit run Iapp.py

eval:  ## compare prompt techniques (REAL API calls)
	$(UV) run python evals.py --mode techniques --limit 3

test:  ## run the unit tests (no network, no API key)
	$(UV) run pytest

lint:  ## run Ruff over the sources
	$(UV) run ruff check $(SRC)

# Writes to your files. `ruff format` does not sort imports, and the Ruff
# config selects the "I" rules, so sort them first — --select I limits the
# autofix to import order only (no unused-import removal or other rewrites).
format:  ## auto-format + sort imports with Ruff (REWRITES files)
	$(UV) run ruff check --select I --fix $(SRC)
	$(UV) run ruff format $(SRC)

# Mirrors .github/workflows/check.yml: lint first, then the unit tests.
# Run this before pushing to catch what CI would catch.
check: lint test  ## lint + unit tests (what CI runs)
	@echo "check: lint and tests passed."

sync:  ## create/update the .venv from uv.lock
	$(UV) sync

clean:  ## remove the .venv (rebuild it with `make sync`)
	rm -rf .venv
	@echo "Clean done. System packages untouched."

.PHONY: help run eval test lint format check sync clean
