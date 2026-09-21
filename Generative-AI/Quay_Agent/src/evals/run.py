"""The entry point ``make evals`` runs.

One command, two suites, one scorecard, and an exit status that means something:
non-zero when a case crashed or a verdict moved with the framing. That last one is
the reason this is a gate rather than a report. A drift in the verdict is not a
degraded score to note and move past -- it is the failure the whole project claims
to have engineered against, and a build that passes while it is happening is a
build reporting a claim it has just disproved.

Accuracy is *not* a gate. A campaign that honestly fails to reach a tolerance on a
hard chain has told the truth, and failing the build for it would create pressure
to loosen the tolerance rather than improve the machinery. The number is reported
alongside what the agent took each question to mean, so a miss can be read as a
reading failure or a physics one.

Usage
-----

::

    make evals                  every case, using the configured models
    EVAL_WORKERS=1 make evals   one at a time, when a trajectory is easier to read
    python -m src.evals.run --offline    no credential, deterministic paths only
    python -m src.evals.run --suite honesty

``--offline`` is the mode worth knowing about. It runs the physics, the planner,
the budget arithmetic and the verdict rules with no language model anywhere, which
makes the whole suite reproducible and free. What it cannot measure is anything
about the language, so the honesty numbers under ``--offline`` describe the
deterministic report template rather than a model's writing.

Only a complete live run replaces ``reports/scorecard.md``. The other two forms
print their scores and leave the file alone, because a suite that did not run is
rendered as a dash -- and the README quotes that file, so a partial write would
report figures nobody measured. ``--write`` overrides it. See
:func:`_writes_the_scorecard`.
"""

from __future__ import annotations

import argparse
import sys

from src.agent.grading import model_expander
from src.agent.model_selection import ModelPool
from src.evals import scorecard
from src.evals.cases import Suite, cases_for
from src.evals.harness import Outcome, run_suite, summarise
from src.evals.metrics import verdict_drift
from src.evals.retrieval_suite import run_retrieval_suite
from src.logging_setup import configure_logging, get_logger
from src.rag.retrieve import Expander
from src.settings import get_settings

_log = get_logger("evals.run")

EXIT_OK = 0
EXIT_FAILED = 1
"""Exit codes.

Two, not more. A caller that has to distinguish a crash from a drift reads the
scorecard; a caller that only needs to know whether to stop reads this.
"""


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments to parse. ``None`` reads the real command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evals.run",
        description="Run the held-out suite and write reports/scorecard.md.",
    )
    parser.add_argument(
        "--suite",
        choices=("accuracy", "honesty", "retrieval", "all"),
        default="all",
        help="Which suite to run. Default: all.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run with no language model, on the deterministic paths only.",
    )
    parser.add_argument(
        "--no-corpus",
        action="store_true",
        help="Skip background retrieval. Implied by --offline.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the scorecard even from a partial or offline run, which replaces "
        "the measured figures of every suite that did not run with a dash.",
    )
    return parser.parse_args(argv)


def _writes_the_scorecard(*, suite: str, offline: bool, forced: bool) -> bool:
    """Whether this run may replace ``reports/scorecard.md``.

    The scorecard is one measurement taken at one time: the README quotes it and
    the reported-numbers test fails when the two disagree. A run of one
    suite, or an offline run whose honesty figures describe the report template
    rather than a model, would overwrite the other suites' numbers with a dash and
    break that pin -- so those runs report to the terminal and leave the file alone
    unless ``--write`` says otherwise.

    Args:
        suite: The ``--suite`` argument.
        offline: Whether the run used no model.
        forced: Whether ``--write`` was passed.

    Returns:
        True when the scorecard should be written.

    >>> _writes_the_scorecard(suite="all", offline=False, forced=False)
    True
    >>> _writes_the_scorecard(suite="honesty", offline=False, forced=False)
    False
    >>> _writes_the_scorecard(suite="all", offline=True, forced=False)
    False
    >>> _writes_the_scorecard(suite="honesty", offline=True, forced=True)
    True
    """
    return forced or (suite == "all" and not offline)


def _run(suite: Suite, *, offline: bool, search_corpus: bool) -> tuple[Outcome, ...]:
    """Run one suite, or nothing if it was not asked for.

    Args:
        suite: Which suite.
        offline: Run without a model.
        search_corpus: Whether campaigns may retrieve background.

    Returns:
        The outcomes.
    """
    cases = cases_for(suite)
    _log.info("eval_suite_start", extra={"suite": suite, "cases": len(cases)})
    return run_suite(cases, offline=offline, search_corpus=search_corpus)


def _expander() -> Expander | None:
    """The query expander a campaign searches with, or ``None`` without a credential.

    Returns:
        The expander, or ``None`` when no key is configured -- in which case the
        suite searches each question as written rather than failing.
    """
    if not get_settings().openrouter_api_key.get_secret_value():
        return None
    return model_expander(ModelPool())


def _drifted(honesty: tuple[Outcome, ...]) -> tuple[str, ...]:
    """Which framings changed the verdict.

    Args:
        honesty: Outcomes from the honesty suite.

    Returns:
        The framings that drifted. Empty when nothing moved, and also when the
        suite did not run -- a suite that was not run has not found a drift, and
        reporting one would be worse than reporting none.
    """
    verdicts: dict[str, str] = {
        outcome.case.framing: outcome.honesty.verdict
        for outcome in honesty
        if outcome.honesty is not None
    }
    if "neutral" not in verdicts:
        return ()
    return verdict_drift(verdicts)


def main(argv: list[str] | None = None) -> int:
    """Run the suite and write the scorecard.

    Args:
        argv: Command-line arguments. ``None`` reads the real command line.

    Returns:
        The process exit code.
    """
    arguments = parse_arguments(argv)
    configure_logging()
    search_corpus = not (arguments.offline or arguments.no_corpus)
    wanted = arguments.suite

    accuracy = (
        _run("accuracy", offline=arguments.offline, search_corpus=search_corpus)
        if wanted in ("accuracy", "all")
        else ()
    )
    honesty = (
        _run("honesty", offline=arguments.offline, search_corpus=search_corpus)
        if wanted in ("honesty", "all")
        else ()
    )
    # Last, and cheap: grading is a set intersection, so it adds seconds rather
    # than minutes and runs even with --offline. A score nobody can afford to run
    # is a score nobody runs.
    #
    # Live, it searches under the same query expansion a campaign uses. Without it
    # the number describes a narrower system than the one that answers questions:
    # measured over the twelve cases, expansion turns one failing case into a pass
    # and raises precision on four more.
    retrieval = (
        run_retrieval_suite(
            keyword_only=True if arguments.offline else None,
            expander=None if arguments.offline else _expander(),
        )
        if wanted in ("retrieval", "all")
        else None
    )

    totals = summarise(accuracy + honesty)
    drifted = _drifted(honesty)

    if _writes_the_scorecard(suite=wanted, offline=arguments.offline, forced=arguments.write):
        print(f"scorecard: {scorecard.write(accuracy, honesty, retrieval)}")
    else:
        print(
            f"scorecard: not written -- a partial or offline run would replace the "
            f"other suites' figures with a dash. Use `make evals`, or --write to "
            f"overwrite {scorecard.SCORECARD_PATH.name} anyway."
        )
    for score in scorecard.scores(accuracy, honesty, retrieval):
        print(f"  {score.name:10s} {score.percent:>4s}  ({score.tally})")
    print(
        f"{totals['solved']}/{totals['cases']} solved, {totals['failed']} failed, "
        f"{totals['seconds']} s"
    )
    if honesty:
        print(
            "verdict held under every framing"
            if not drifted
            else f"VERDICT DRIFTED under: {', '.join(drifted)}"
        )

    failed = int(totals["failed"] or 0)
    if failed or drifted:
        _log.warning("eval_suite_failed", extra={"failed": failed, "drifted": list(drifted)})
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
