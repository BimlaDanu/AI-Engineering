# Quay — Makefile
#
# One uv-managed environment (.venv) holds the runtime and the dev tools, so every
# target goes through `uv run`. Dependencies are declared in pyproject.toml and
# pinned in uv.lock, which is committed.
#
#   make help    list every target

# VIRTUAL_ENV is stripped so uv uses its own .venv and does not warn about a mismatch.
UV := env -u VIRTUAL_ENV uv

# Ruff checks what mypy checks. `[tool.mypy] files` reads src, tests, scripts, and a
# gate that type-checks a directory it does not lint has a hole in it.
SRC := src tests scripts

.DEFAULT_GOAL := help

## help: show this message
help:
	@echo "Quay — available targets:"
	@echo
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  make /' | column -t -s ':'
	@echo

# ─────────────────── environment ───────────────────

## sync: create or update the .venv from pyproject.toml + uv.lock
sync:
	$(UV) sync

## lock: re-resolve dependencies and rewrite uv.lock (commit the result)
lock:
	$(UV) lock

## sync-ci: install exactly what uv.lock pins, with no re-resolution
#
# `--frozen` makes a stale lock an error rather than a silent upgrade. The
# interpreter is pinned in `.python-version` for a related reason: uv.lock resolves
# numpy 2.4.6 below Python 3.12 and 2.5.2 above it, and 2.5.2's stubs are written in
# 3.12 syntax that mypy, targeting 3.11, cannot parse.
sync-ci:
	$(UV) sync --frozen

## versions: print the interpreter and the packages the gate depends on
versions:
	@$(UV) run python -c "import sys, numpy; print(f'python  {sys.version.split()[0]}'); print(f'numpy   {numpy.__version__}')"
	@$(UV) run mypy --version
	@$(UV) run ruff --version
	@$(UV) run pytest --version

# ─────────────────── quality gates ───────────────────

## format: format and auto-fix with Ruff
format:
	$(UV) run ruff format $(SRC)
	$(UV) run ruff check --fix $(SRC)

## lint: check with Ruff, writing nothing
lint:
	$(UV) run ruff check $(SRC)
	$(UV) run ruff format --check $(SRC)

## typecheck: static type check with mypy
typecheck:
	$(UV) run mypy

## test: run the tests across every core
#
# The suite is CPU-bound arithmetic with no credential and no network, so it
# parallelises. Measured here: 493 s serial against 283 s on eight workers. It does
# not scale further because the run cannot be shorter than its longest single test —
# `make slowest` prints those.
test:
	$(UV) run pytest -n auto

## test-fast: the same tests without the whole-suite runs, for an edit loop
#
# Skips what is marked `slow`: the tests that put all 27 evaluation cases through the
# real graph. `make check` still runs them.
test-fast:
	$(UV) run pytest -n auto -m "not slow"

## test-serial: one test at a time, to debug a parallel-only failure
#
# A failure that appears only under `-n auto` is a test depending on another having
# run first.
test-serial:
	$(UV) run pytest -p no:xdist

## slowest: rank the twenty slowest tests
slowest:
	$(UV) run pytest -n auto --durations=20 -q

## doctest: run the examples in every docstring under src/
#
# A separate invocation, because plain `pytest` collects `tests/` and never looks at
# `src/`. Until this target existed, four of the 133 doctests had gone stale.
doctest:
	$(UV) run pytest -n auto --doctest-modules src -q

## check: what CI runs — lint, typecheck, test, doctest
check: lint typecheck test doctest

# ─────────────────── application ───────────────────

## run: launch the Streamlit app
#
# One question emits forty to a hundred JSON records. They go to reports/quay.log;
# the terminal keeps warnings and errors. `make logs` follows the rest.
run:
	$(UV) run streamlit run src/ui/app.py

## logs: follow the structured log the app writes while it runs
#
# One JSON object per line. Rotated at five megabytes, ignored by version control.
logs:
	@touch reports/quay.log && tail -f reports/quay.log

## figures: redraw every figure into reports/figures/ as PDF
#
# `src/figure_export.py` holds the output policy and the plotting style; the scripts
# only draw.
figures:
	$(UV) run python scripts/draw_ansatz_circuit.py
	$(UV) run python scripts/draw_device_limits.py
	$(UV) run python scripts/draw_lab_figures.py
	$(UV) run python scripts/draw_agent_graph.py
	$(UV) run python scripts/draw_project_workflow.py
	$(UV) run python scripts/draw_app_icon.py

## icon: regenerate the page icon (PNG for the browser tab, PDF to inspect)
icon:
	$(UV) run python scripts/draw_app_icon.py

## corpus: fetch the retrieval corpus from arXiv into data/corpus/
#
# Reaches the network. Not part of `make check`.
corpus:
	$(UV) run python scripts/fetch_arxiv_corpus.py

## ingest: build the Chroma vector store from data/corpus/
#
# Needs the embedding credential. Re-run after changing the corpus or the embedding
# model.
ingest:
	$(UV) run python -m src.rag.ingest

## evals: run the held-out suite and write reports/scorecard.md
#
# Live: needs OPENROUTER_API_KEY and costs tokens. Exits non-zero if a case fails
# or a verdict drifts. There is no keyless version of this number -- two of the
# three suites grade what a language model wrote, so a run with no model measures
# the report template. `make evals-offline` runs that reduced thing on purpose and
# says so; only a complete live run is allowed to replace the scorecard.
#
# Each case is roughly seven sequential model calls, so the run is mostly waiting.
# EVAL_WORKERS overlaps that waiting; it does not raise the outbound rate, which is
# capped process-wide by REQUESTS_PER_SECOND. The scorecard is identical either way.
#
#   REQUESTS_PER_SECOND=8 EVAL_WORKERS=8 make evals
#   EVAL_WORKERS=1 make evals        # one at a time, to read a trajectory
evals:
	$(UV) run python -m src.evals.run

## evals-offline: run what can be scored with no credential
#
# The physics, the planner, the budget arithmetic, the verdict rules and the
# keyword half of retrieval, with no model anywhere: free, reproducible, and
# useful for checking the harness itself still runs. It cannot score the writing,
# so it leaves reports/scorecard.md alone rather than overwriting measured
# figures with figures nobody measured.
evals-offline:
	$(UV) run python -m src.evals.run --offline

## campaign: run one feasibility campaign end to end
#
# The agent's entry point without the interface. Live: costs tokens.
campaign:
	$(UV) run python -m src.agent.campaign

## mcp: serve the agent's tools over the Model Context Protocol, on stdin/stdout
#
# For a client rather than a person: it reads JSON from stdin and writes JSON to
# stdout, so running it in a terminal looks like a hang. Configure a client with this
# command and this directory. Needs no credential — no tool here calls a model.
mcp:
	$(UV) run python -m src.mcp_server

# ─────────────────── diagnostics ───────────────────

## routes: print the route each shape of question takes through the graph
#
# One question per branch, printing the nodes that ran in order. Offline and free.
routes:
	$(UV) run python -m scripts.show_graph_routes

## live-check: ask a handful of questions and print each trajectory
#
# Which configurations the loop proposed, which it refused and why, and how far up the
# depth ladder it got — none of which a pass rate shows. Live by default.
#
#   uv run python -m scripts.live_check --offline
live-check:
	$(UV) run python -m scripts.live_check

## readme-check: ask every question the README advertises and check it is answerable
#
# Reads the questions out of the document rather than keeping a second copy. Exits
# non-zero when an advertised question comes back unanswerable. Offline by default.
#
#   uv run python -m scripts.check_readme_questions --live
readme-check:
	$(UV) run python -m scripts.check_readme_questions --quiet

## docstring-check: fail on a docstring that grew into an essay
#
# Ruff's `D` rules check a docstring exists and is shaped like Google style, not
# that it is short. This flags prose over budget, reST section headings, changelog
# phrasing and markdown bold. Free and offline.
#
#   uv run python -m scripts.check_docstrings --path src/agent
docstring-check:
	$(UV) run python -m scripts.check_docstrings --quiet

## starter-check: press every starter button and check what comes back
#
# The eight questions on the Chat page's own buttons are the first thing anybody
# does with this application, so they are the ones that must not come back empty,
# declined, or shaped like something else. Reports the route each takes and how much
# of the seventeen-node graph the set covers between them. Offline by default.
#
#   uv run python -m scripts.check_starter_questions --offered     # just the buttons
#   uv run python -m scripts.check_starter_questions --live        # costs tokens
starter-check:
	$(UV) run python -m scripts.check_starter_questions --quiet

## curve-check: ask the six curve questions and report what came back
#
# Prints the intent, the route, whether the exact field sweep was fetched, and whether
# every point was computed twice. Exits non-zero if a curve question did not get a
# cross-checked curve. Offline and free.
curve-check:
	$(UV) run python -m scripts.check_field_sweep_answers

## timing: time one question through the graph and say where the wait went
#
# A line per model call from the pool's ledger, then wall clock minus that total as
# local work. Run `--offline` first: that figure is the floor.
#
#   uv run python -m scripts.time_a_question --offline
timing:
	$(UV) run python -m scripts.time_a_question

## verify-models: resolve every catalogued slug against the live model index
#
# `src/model_catalogue.py` reconstructs the identifier the API expects from the
# display name; this checks the reconstruction with a read-only GET. Also lists the
# gateway's free endpoints. Reaches the network, needs the credential, exits non-zero
# when a chat slug does not resolve.
verify-models:
	$(UV) run python -m scripts.verify_model_slugs --free

## bakeoff: ask several models the same question and compare what changed
#
# If deterministic code computes every number, swapping the model must leave the
# numbers where they are. One full campaign per model. Live: costs tokens. Exits
# non-zero when something moved.
#
#   uv run python -m scripts.model_bakeoff --models openai/gpt-4o,openai/gpt-4.1-mini
bakeoff:
	$(UV) run python -m scripts.model_bakeoff

## sweep: price every configuration against every machine, into a committed table
#
# Chain length x boundary x depth x device. Arithmetic over each machine's wiring
# diagram: no model, no credential, no simulation, so the output is byte-identical on
# any machine. Resumes by default; rows already present are kept.
#
#   uv run python -m src.evals.sweep --rebuild
sweep:
	$(UV) run python -m src.evals.sweep

# ─────────────────── housekeeping ───────────────────

## clean: remove the .venv and every tool cache
clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Clean done. System packages untouched."

.PHONY: help sync lock sync-ci versions \
	format lint typecheck test test-fast test-serial slowest doctest check \
	run logs figures icon corpus ingest evals evals-offline campaign mcp \
	routes live-check readme-check starter-check curve-check timing verify-models bakeoff sweep \
	clean
