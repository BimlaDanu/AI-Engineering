# ═══════════════════════════════════════════════════════════
#   Synapse — AI/ML Assistant — Makefile
#
#  Single environment managed by uv (.venv). It holds both the app
#  runtime deps and the dev-only tools (pytest, ruff), so every target
#  goes through `uv run`. No separate app_env — uv handles it all.
#
#  We strip any already-activated VIRTUAL_ENV first so uv always uses
#  its own .venv and never warns about a mismatch.
#
#  `make` on its own prints the target list (see the `help` target); each
#  target documents itself with a trailing `## ...` comment.
# ═══════════════════════════════════════════════════════════

# Call uv with the stale VIRTUAL_ENV removed so it uses its own .venv
# without warning about a mismatch.
UV := env -u VIRTUAL_ENV uv

# What Ruff checks and formats. conftest.py lives at the root, outside both directories,
# so name it or it is silently never linted.
SRC := src tests conftest.py

# Bare `make` should explain itself rather than silently building something.
.DEFAULT_GOAL := help

help:  ## Show this help (targets and what they do)
	@printf "\nSynapse — make targets\n\nUsage: make <target>\n\n"
	@awk 'BEGIN {FS = ":[^#]*##"} /^[a-zA-Z_-]+:[^#]*##/ {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nOne test:  make test T=tests/test_x.py::test_y\n"
	@printf "Fix lint: make fix\n\n"

sync:  ## Create/update the uv .venv from pyproject.toml
	$(UV) sync

ingest:  ## Rebuild the Chroma vector store from data/ (asks first: costs API tokens)
	@printf "This wipes and rebuilds chroma_db/ from data/.\n"
	@printf "With EMBEDDING_BACKEND=api (the default) it SPENDS embedding tokens.\n"
	@printf "Continue? [y/N] " && read ans && [ "$$ans" = "y" ] || { echo "Cancelled."; exit 1; }
	$(UV) run python -m src.rag.ingest

ingest-yes:  ## Rebuild the vector store without the confirmation prompt (for scripts)
	$(UV) run python -m src.rag.ingest

mcp-serve:  ## Serve Synapse's own tools over MCP (stdio)
	$(UV) run python -m src.mcp_server

run:  ## Run the Streamlit app (src/app.py)
	$(UV) run streamlit run src/app.py

test:  ## Run the offline unit tests (pytest). One test: make test T=tests/test_x.py::test_y
	$(UV) run pytest $(if $(T),$(T) -v,)

lint:  ## Check sources with Ruff
	$(UV) run ruff check $(SRC)

format:  ## Auto-format sources with Ruff
	$(UV) run ruff format $(SRC)

format-check:  ## Verify formatting without rewriting files (what CI runs)
	$(UV) run ruff format --check $(SRC)

fix:  ## Auto-fix what `make check` would fail on (Ruff --fix, then format)
	$(UV) run ruff check --fix $(SRC)
	$(UV) run ruff format $(SRC)

check: lint format-check test  ## Run the full CI gate locally (lint + format check + tests)

doctor:  ## Read-only readiness report: index drift, credentials, accounts (spends nothing)
	$(UV) run python -m src.doctor

clean:  ## Remove caches only (leaves .venv intact — no re-sync needed)
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Caches cleared. .venv left alone — use 'make clean-venv' to remove it too."

clean-venv: clean  ## Also remove the uv-managed .venv (requires 'make sync' afterwards)
	rm -rf .venv
	@echo "Removed .venv. Run 'make sync' to rebuild it. System packages untouched."

.PHONY: help sync ingest ingest-yes mcp-serve run test lint format format-check fix \
        check doctor clean clean-venv
