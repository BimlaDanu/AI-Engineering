"""Ask the real agent a handful of questions and print what each run actually did.

Why this exists beside a green ``make check`` and a scored ``make evals``: both of
those are about **outcomes**. A campaign that reaches the right verdict by refusing
everything and falling back on the baseline scores exactly as well as one that
climbed the depth ladder properly, and neither the test suite nor the scorecard can
tell them apart. What separates them is the **trajectory** -- which configurations
were proposed, which were refused and for what reason, how far up the ladder it got
before the budget or the coherence time stopped it.

So this prints one line per question: the verdict, how many configurations were run
and refused, what stopped it, how much of the budget went, and how long it took.
Read down the column and a loop that always refuses everything, or always runs
exactly one configuration, is obvious in a way it is not from a pass rate.

**Not a test, deliberately.** It calls the live gateway -- several model calls per
question -- so it costs tokens and its answers depend on a model's judgement on the
day. Everything under ``tests`` stays offline and deterministic. This is a script a
person runs on purpose.

    make live-check                                  # every question
    uv run python -m scripts.live_check ring         # only ones matching "ring"
    uv run python -m scripts.live_check --offline    # free, deterministic, no key

The last of those is worth knowing about: with ``--offline`` the whole thing runs on
the deterministic paths, costs nothing, and still exercises every trajectory
decision -- because none of those decisions was ever a language model's job. It is
the version to run when the question is "did I break the loop?" rather than "how
well does the model read a sentence?".
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from src.agent.graph import run_campaign
from src.agent.state import CampaignState
from src.hardware.devices import LINEAR
from src.logging_setup import configure_logging, get_logger
from src.physics.registry import field_sweep_bench
from src.ui.starters import QUESTIONS as STARTER_QUESTIONS

REGRESSIONS: tuple[str, ...] = (
    # Each one is a shape of question that has gone wrong before, kept as the thing
    # it is there to show rather than as a question anybody would ask twice.
    "Is a quantum computer worth using for a chain of 200 magnets?",
    "Should we use a quantum computer? (no other details given)",
    "What is the capital of France?",
)
"""Questions that once produced a bad trajectory.

The first is past every register modelled here, and the loop should refuse it on
arithmetic rather than climb a ladder it cannot finish. The second names no problem
at all and should be asked for detail rather than answered. The third is outside the
subject entirely and should be declined without a campaign being run at all.
"""

QUESTIONS: tuple[str, ...] = STARTER_QUESTIONS + REGRESSIONS
"""Everything this script can ask.

The starters come first because they are the questions a visitor is most likely to
ask -- they are on buttons -- and because the claim made about them is a claim about
the loop: that between them they take different routes rather than six variations of
one. Nothing offline can fully check that, since which route a question takes
depends on how a model reads it, which is exactly why this script is not a test.
"""

SHOT_BUDGET = 1_000_000_000
"""The budget each question gets.

Large on purpose. One energy reading on a ten-spin chain at one per cent accuracy
costs tens of millions of measurements and an optimisation needs dozens of them, so
a budget that sounds generous refuses the first configuration and every question
comes back "we could not afford to try" -- which tells you nothing about the loop
and looks exactly like a broken one.
"""

_logger = get_logger("live_check")


def summarise(question: str, state: CampaignState, seconds: float) -> str:
    """Render one campaign as a single readable line.

    Args:
        question: What was asked.
        state: The finished campaign.
        seconds: Wall-clock time it took.

    Returns:
        One line, padded so a column of them reads as a table.
    """
    verdict = state["verdict"]
    ledger = state["shots"]
    call = verdict.call if verdict else "none"
    deepest = max((run.depth for run in state["runs"]), default=0)
    return (
        f"{call:<12} "
        f"ran {len(state['runs']):>2}  "
        f"refused {len(state['ruled_out']):>2}  "
        f"deepest p{deepest:<3} "
        f"{ledger.spent / max(ledger.budget, 1):>5.0%} of budget  "
        f"{seconds:>5.1f}s  "
        f"{question[:44]}"
    )


def why_it_stopped(state: CampaignState) -> str:
    """Name what ended the climb, from the last refusal recorded.

    The single most useful thing about a trajectory. A loop that always stops for
    the same reason is a loop with one code path, whatever its verdicts look like.

    Args:
        state: The finished campaign.

    Returns:
        A short phrase, or a note that nothing was refused.
    """
    if not state["ruled_out"]:
        return "nothing was refused"
    reason = state["ruled_out"][-1].reason
    for marker, label in (
        ("coherence", "ran out of coherence"),
        ("shots", "ran out of budget"),
        ("no layout", "did not fit the register"),
    ):
        if marker in reason:
            return label
    return reason[:60]


def ask_one(question: str, offline: bool) -> tuple[CampaignState, float]:
    """Run one campaign and time it.

    Args:
        question: The problem, in ordinary language.
        offline: Run with no language model.

    Returns:
        The finished campaign and how long it took, in seconds.
    """
    started = time.monotonic()
    state = run_campaign(
        question,
        shot_budget=SHOT_BUDGET,
        device=LINEAR,
        chat_model=None if offline else "auto",
        search_corpus=True,
        fetch_external=False,
        # Lent exactly as the interface lends it, because the point of this script is
        # to watch what a visitor would get. Two of the starters ask for a curve in
        # the field, and without the bench they would be answered from the notes here
        # and from the exact solution in the browser -- which is the one difference
        # this check exists to rule out.
        reference_bench=field_sweep_bench(),
    )
    return state, time.monotonic() - started


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments to parse. ``None`` reads the real command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.live_check",
        description="Ask the agent a handful of questions and print each trajectory.",
    )
    parser.add_argument(
        "filter",
        nargs="?",
        default="",
        help="only ask questions containing this text; one bad trajectory is usually "
        "the one worth reading the whole log for",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="run on the deterministic paths: free, needs no key, and still exercises "
        "every trajectory decision",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ask every matching question and print the results.

    Args:
        argv: Command-line arguments. ``None`` reads the real ones.

    Returns:
        Zero. This reports rather than judges -- a trajectory is something to read,
        not something to pass or fail, and a script that returned non-zero for an
        unusual one would eventually be run with its output ignored.
    """
    configure_logging()
    arguments = parse_arguments(argv)
    wanted = [q for q in QUESTIONS if arguments.filter.lower() in q.lower()]
    if not wanted:
        print(f"nothing matches {arguments.filter!r}")
        return 0

    mode = "offline, no model" if arguments.offline else "live, calling a model"
    print(f"Asking {len(wanted)} question{'' if len(wanted) == 1 else 's'} — {mode}\n")
    for question in wanted:
        state, seconds = ask_one(question, arguments.offline)
        print(summarise(question, state, seconds))
        print(f"{'':>12} stopped because: {why_it_stopped(state)}")
        _logger.info(
            "live_check_question",
            extra={
                "question": question,
                "verdict": state["verdict"].call if state["verdict"] else None,
                "runs": len(state["runs"]),
                "refused": len(state["ruled_out"]),
                "seconds": round(seconds, 2),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
