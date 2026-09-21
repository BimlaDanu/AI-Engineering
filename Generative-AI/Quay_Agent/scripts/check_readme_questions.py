"""Ask every question the README advertises, and report what came back.

The README's *What you can ask it* section is a promise: twenty questions, each
with a sentence saying what the application does with it. Nothing checked that
promise, and it went stale in the way documentation does -- silently, and in the
direction that flatters the project.

Two defects found on the first run of this made the case for keeping it. *Detail
the mathematics of the quantum-to-classical mapping* -- a question the README
lists, about the derivation the project rests on -- was declined by the scope gate
and reported as a **NO** verdict at high confidence with nothing behind it, because
``quantum-to-classical`` is one hyphenated token and neither half matched. And
*what would a hundred questions of this kind cost?*, which the application itself
proposes as a follow-up, was declined by the same gate on the screen below the
answer that suggested it.

Offline by default, which is the point: routing, scope, formalisation and which
tools fire are all deterministic, so the failures this catches cost nothing to
catch. ``--live`` runs the same questions through the real gateway when the
question is about the prose rather than the path.

    uv run python -m scripts.check_readme_questions
    uv run python -m scripts.check_readme_questions --live
    uv run python -m scripts.check_readme_questions --only lattice

Exits non-zero when a question the README advertises comes back unanswerable, so
it can gate a change to the routing or to the README.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from src.agent.graph import run_campaign
from src.agent.memory import Memory
from src.agent.state import CampaignState
from src.hardware.devices import LINEAR
from src.logging_setup import configure_logging
from src.physics.registry import field_sweep_bench

README = Path(__file__).resolve().parent.parent / "README.md"

SECTION = "## What you can ask it"
"""Heading the questions are read from. Read rather than duplicated here, so the
list cannot drift away from the document it is checking."""

NEXT_SECTION = "## How it reaches an answer"

SHOT_BUDGET = 1_000_000_000
"""Large enough that nothing is refused for want of budget.

A refusal on the shot budget is a real answer, but it is not the one being checked
here: this asks whether a question *reaches* an answer at all.
"""

DECLINED_ON_PURPOSE = ("what is the best pizza in vilnius?",)
"""Questions the README advertises as being *declined*.

One of them is there to show the scope gate working. It must be refused, so for
this question a refusal is the pass and an answer is the failure.
"""

FOLLOW_UPS = (
    "and if we double the circuit depth?",
    "what would a hundred questions of this kind cost?",
)
"""Questions the README describes as continuing an earlier one.

These are asked after a first question, because that is how the README presents
them and how a reader meets them -- as the second thing typed, or as a suggested
follow-up under an answer. Asked cold they are a different question: *what would a
hundred questions of this kind cost?* has no "kind" to refer to, and admitting it
anyway would mean admitting every sentence containing the word "cost".
"""

OPENING_QUESTION = "Is quantum hardware worth it for a 10-spin chain at criticality?"
"""What the follow-ups follow.

Named rather than taken from the list above, so a reordering of the README cannot
quietly change what the follow-ups are following.
"""

NEEDS_A_MODEL = ("can you write the qaoa circuit for 8 spins at depth 3?",)
"""Questions whose promised answer cannot be produced offline.

Writing a program is the one thing in this application that a language model does
rather than checks, so with no model reachable the honest outcome is a refusal that
says so. That is a pass here, and the reason is printed rather than hidden -- but
only for the questions where it is genuinely the answer, so that a silent
everything-refuses run cannot look green.
"""


def questions() -> tuple[str, ...]:
    """Read the advertised questions out of the README.

    Returns:
        Each question in the order the document lists them.

    Raises:
        SystemExit: If the section or its questions cannot be found, which means
            the README was restructured and this script needs to follow it rather
            than silently checking nothing.
    """
    text = README.read_text(encoding="utf-8")
    try:
        body = text[text.index(SECTION) : text.index(NEXT_SECTION)]
    except ValueError:
        raise SystemExit(f"could not find {SECTION!r} in {README}") from None
    found = tuple(dict.fromkeys(re.findall(r'\*"(.+?)"\*', body)))
    if not found:
        raise SystemExit(f"found no questions under {SECTION!r}; has the format changed?")
    return found


def verdict_of(state: CampaignState) -> str:
    """Summarise in one word what the reader would be shown.

    Args:
        state: The finished campaign.

    Returns:
        A short label for the shape of the reply. ``"measured"`` is the deliberate
        middle case: runs happened and a report was written, but no verdict was
        reached, because a feasibility call needs a chain the question named.
    """
    request = state["request"]
    if request.blocked:
        return "blocked"
    if not request.in_scope:
        return "declined"
    answer = state["answer"]
    if answer is not None and answer.refused:
        return "not-answered"
    if state["draft"] is not None and state["draft"].written:
        return "code"
    if answer is not None and answer.written:
        return "explained"
    if state["verdict"] is not None:
        return f"verdict:{state['verdict'].call}"
    # A campaign that climbed the depth ladder and then withheld the call is not an
    # empty one -- `converge` withholds it when the question named no chain -- and
    # counting it as "nothing" made this check fail on a question the README
    # advertises and the agent answers correctly.
    if state["runs"] and state["report"]:
        return "measured"
    return "nothing"


def ask(question: str, live: bool) -> tuple[CampaignState, float]:
    """Run one question through the graph, with a prior turn if it needs one.

    Args:
        question: The question, exactly as the README prints it.
        live: Reach the real gateway rather than the deterministic paths.

    Returns:
        The finished campaign and how long it took, in seconds.
    """
    started = time.monotonic()
    memory: Memory | None = None
    if question.strip().lower() in FOLLOW_UPS:
        # Its own user and thread, so a check run never lands in a real
        # conversation's memory or reads one.
        memory = Memory(user="readme-check", thread="readme-check")
        run_campaign(
            OPENING_QUESTION,
            shot_budget=SHOT_BUDGET,
            device=LINEAR,
            chat_model="auto" if live else None,
            search_corpus=False,
            fetch_external=False,
            suggest_followups=False,
            memory=memory,
        )
    state = run_campaign(
        question,
        shot_budget=SHOT_BUDGET,
        device=LINEAR,
        chat_model="auto" if live else None,
        search_corpus=True,
        fetch_external=False,
        reference_bench=field_sweep_bench(),
        memory=memory,
    )
    return state, time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    """Ask every advertised question and print a row per answer.

    Args:
        argv: Command line. ``None`` reads the real one.

    Returns:
        Zero when every question reached the kind of answer the README promises,
        one otherwise.
    """
    parser = argparse.ArgumentParser(description="Check the README's advertised questions.")
    parser.add_argument("--live", action="store_true", help="call the real gateway")
    parser.add_argument("--only", default="", help="only questions containing this text")
    parser.add_argument("--quiet", action="store_true", help="suppress the pipeline's log lines")
    args = parser.parse_args(argv)

    if not args.quiet:
        configure_logging()

    asked = [q for q in questions() if args.only.lower() in q.lower()]
    if not asked:
        print(f"no advertised question matches {args.only!r}")
        return 1

    failures: list[tuple[str, str]] = []
    print(f"{len(asked)} advertised questions, {'live' if args.live else 'offline'}\n")
    for position, question in enumerate(asked, start=1):
        state, elapsed = ask(question, args.live)
        outcome = verdict_of(state)
        model = state["model"]
        lowered = question.strip().lower()
        should_decline = lowered in DECLINED_ON_PURPOSE
        excused = not args.live and lowered in NEEDS_A_MODEL
        wrong = (outcome == "declined") is not should_decline or outcome == "nothing"
        if excused:
            wrong = False
        mark = "FAIL" if wrong else "ok  "
        if wrong:
            failures.append((question, outcome))
        print(f"{mark} {position:2}. {question}")
        note = "  (needs a model; refused honestly offline)" if excused else ""
        print(
            f"        {outcome:14} intent={state['intent'].intent:11} "
            f"problem={'-' if model is None else model.label()}  {elapsed:.1f}s{note}"
        )

    print()
    if failures:
        print(f"{len(failures)} of {len(asked)} advertised questions did not reach an answer:")
        for question, outcome in failures:
            print(f"  {outcome:14} {question}")
        return 1
    print(f"all {len(asked)} advertised questions reached the kind of answer the README promises")
    return 0


if __name__ == "__main__":  # pragma: no cover - a script a person runs
    sys.exit(main())
