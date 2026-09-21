# ═══════════════════════════════════════════════════════════
#   QuantumLab Copilot — Makefile
#
#  One environment, managed by uv (.venv). It holds the app runtime deps
#  and the dev-only tools (pytest, ruff, mypy), so every target goes
#  through `uv run`. Dependencies are declared in pyproject.toml and
#  pinned in uv.lock — commit the lock file.
#
#  We strip any already-activated VIRTUAL_ENV first so uv always uses
#  its own .venv and never warns about a mismatch.
#
#  Usage:
#    make help       -> list every target (the default)
#
#    make sync       -> create/update the uv .venv from pyproject.toml + uv.lock
#    make lock       -> re-resolve dependencies and rewrite uv.lock
#
#    make format     -> auto-format sources with Ruff
#    make lint       -> check sources with Ruff (no changes written)
#    make typecheck  -> static type check with mypy
#    make test       -> run the unit tests (pytest)
#    make test-fast  -> the same, minus the slow physics runs (78s -> 30s)
#    make check      -> lint + typecheck + test — the definition of done
#    make check-fast -> the same gate minus the slow tests, for the edit loop
#    make coverage   -> the suite under coverage, with a floor (report in htmlcov/)
#
#    make run        -> launch the Streamlit app
#    make ingest     -> build the Chroma vector store from the paper corpus
#    make evals      -> run the held-out evals and write the scorecard
#
#    make clean      -> remove the uv-managed .venv and all tool caches
# ═══════════════════════════════════════════════════════════

# Call uv with the stale VIRTUAL_ENV removed so it uses its own .venv
# without warning about a mismatch.
UV := env -u VIRTUAL_ENV uv

# Directories Ruff and mypy should check/format
SRC := src tests

.DEFAULT_GOAL := help

## help: show this message
help:
	@echo "QuantumLab Copilot — available targets:"
	@echo
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  make /' | column -t -s ':'
	@echo

# ─────────────────── environment ───────────────────

## sync: create/update the .venv from pyproject.toml + uv.lock
sync:
	$(UV) sync

## lock: re-resolve dependencies and rewrite uv.lock (commit the result)
lock:
	$(UV) lock

## sync-ci: install exactly what uv.lock pins -- no re-resolution
#
# What CI installs before running the gate. `--frozen` is the whole point: plain
# `uv sync` may re-resolve when pyproject.toml has moved ahead of the lock, and a
# gate that quietly resolves its own dependencies is not testing the versions
# anybody else gets. Here a stale lock is an error rather than a silent upgrade.
#
# The interpreter comes from `.python-version`, which is pinned for the same
# reason: uv.lock resolves numpy 2.4.6 below Python 3.12 and 2.5.2 at or above
# it, and 2.5.2's type stubs are written in 3.12 syntax that mypy -- targeting
# 3.11, per [tool.mypy] -- cannot parse. Unpinned, `make typecheck` passed or
# failed depending on which interpreter uv happened to find.
sync-ci:
	$(UV) sync --frozen

## versions: print the interpreter and the packages the gate depends on
#
# One line each, because two of them are coupled: mypy targets 3.11 and numpy's
# stubs are only parseable at that target below numpy 2.5. A green run should say
# what it was green on rather than leave it to be inferred.
versions:
	@$(UV) run python -c "import sys, numpy; print(f'python  {sys.version.split()[0]}'); print(f'numpy   {numpy.__version__}')"
	@$(UV) run mypy --version
	@$(UV) run ruff --version
	@$(UV) run pytest --version

# ─────────────────── quality gates ───────────────────

## format: auto-format sources with Ruff
format:
	$(UV) run ruff format $(SRC)
	$(UV) run ruff check --fix $(SRC)

## lint: check sources with Ruff (no changes written)
lint:
	$(UV) run ruff check $(SRC)
	$(UV) run ruff format --check $(SRC)

## typecheck: static type check with mypy
typecheck:
	$(UV) run mypy

## test: run the unit tests
test:
	$(UV) run pytest

## test-fast: the same tests without the slow ones, for a tight edit loop
#
# Skips what is marked `slow`, which is two kinds of test and both are physics
# waiting on a CPU rather than logic waiting to break:
#
#   * the eval cases run through the real graph, and
#   * every UI test that renders the Quantum Ising Lab, because drawing that page
#     diagonalises a chain first and Streamlit's cache is per session, so each
#     such test pays the full solve on its own.
#
# Measured: 78s for the whole suite, 30s without them -- and they are the tests
# least likely to break while editing one module, which is what makes them the
# right ones to defer. `make check` and CI still run every one, so this is a
# shortcut during work and never the gate.
test-fast:
	$(UV) run pytest -m "not slow"

## check: everything CI will run — lint, typecheck, test
check: lint typecheck test

## check-fast: the same gate without the slow tests, for the edit loop
#
# Lint and types are quick and catch different things, so they stay; only the
# slow half of the suite is deferred. What `make check` is to a commit, this is
# to a save -- and it is emphatically not a substitute: `check` is the definition
# of done, here and in CI.
check-fast: lint typecheck test-fast

## coverage: run the suite under coverage and fail below the floor
#
# The flags live here rather than in `addopts`, deliberately: measuring coverage
# roughly triples the suite's wall clock, and paying that on every `make test`
# would make the edit loop worse to buy a number nobody reads between commits.
# This is the target that wants the number.
#
# `--cov=src` and not the working directory, so the tests do not measure
# themselves -- a suite that covers its own lines reports a percentage that goes
# up when you add a test and says nothing about the code under it.
#
# The floor is 90 against 96 measured. Set at the measurement it would ratchet on
# any honest refactor that deletes a covered line; set far below it, it stops
# being a gate. Six points is roughly "a module may lose its tests before anyone
# has to argue about it", which is the thing worth catching.
#
# `term-missing:skip-covered` prints only the files with gaps, so the report is
# the list of things to look at rather than a wall of 100%s.
coverage:
	$(UV) run pytest --cov=src --cov-report=term-missing:skip-covered \
		--cov-report=html --cov-fail-under=90
	@echo "HTML report: htmlcov/index.html"

# ─────────────────── application ───────────────────

## run: launch the Streamlit app
run:
	$(UV) run streamlit run src/ui/app.py

## ingest: build the Chroma vector store from the paper corpus
ingest:
	$(UV) run python -m src.rag.ingest

## evals: run the held-out suite and write reports/scorecard.md
#
# Deliberately NOT `--group evals`: nothing in that optional group is used, and asking
# for it would make a scoring run download a package. The suite needs nothing beyond
# the app's own dependencies -- it grades by comparison, not with a judge model.
#
# Exits non-zero if any case fails, so this can be a gate rather than a report.
#
# Each case is roughly seven sequential model calls, so the run is mostly waiting.
# It answers four cases at once by default (EVAL_WORKERS). That does not send
# requests any faster -- the throttle is shared across the whole process, so the
# outbound rate stays at REQUESTS_PER_SECOND -- it only overlaps the waiting.
# To go faster than that, the throttle is the thing to raise, and only as far as
# the gateway tier allows:
#
#   REQUESTS_PER_SECOND=8 EVAL_WORKERS=8 make evals
#   EVAL_WORKERS=1 make evals        # one at a time, to read a trajectory
#
# The scorecard is identical either way -- results are collected back into case
# order before grading -- so this is a wall-clock setting and nothing else.
evals:
	$(UV) run python -m src.evals.run

## observable-check: ask whether the model, not the code, picks what to plot
#
# The sweep tool takes an observable -- ground-state properties or the excitation
# spectrum -- and the unit tests can only prove a chosen one is honoured. Four
# live calls, two of them deliberate near-misses. See the script.
observable-check:
	$(UV) run python -m scripts.observable_check

## live-check: ask the real agent a few questions and log each trajectory
#
# What `make check` and `make evals` cannot show: which actions the loop chose and
# who chose them. Live -- about seven model calls per question. See the script.
live-check:
	$(UV) run python -m scripts.live_check

# ─────────────────── housekeeping ───────────────────

## clean: remove the uv-managed .venv and all tool caches
clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Clean done. System packages untouched."

.PHONY: help sync lock sync-ci versions format lint typecheck test test-fast check check-fast \
	coverage run ingest evals \
	observable-check \
	live-check clean
