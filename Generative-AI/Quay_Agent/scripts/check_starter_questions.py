"""Press every starter button the interface offers and report what came back.

The starters are the fifteen questions on the Chat page's own buttons, and they are
the first thing anybody does with this application -- a reviewer opens it, clicks
one, and forms a view before reading a word of the README. So they are the questions
that must not come back empty, declined, or shaped like something else.

``scripts/check_readme_questions.py`` does the same job for what the *document*
advertises. This does it for what the *interface* offers, and the two lists are not
the same: a README question is read, a starter is clicked.

What is checked, offline
------------------------
Routing, scope, formalisation, which nodes ran and what the state came back holding
are all deterministic, so the failures worth catching cost nothing to catch:

* the question was not blocked or declined by the scope gate;
* something was actually produced -- a verdict, an explanation or a written draft,
  never ``nothing``;
* the reply is *structured*: a feasibility answer carries a model and a report, an
  explanation carries prose.

``--live`` runs the same fifteen through the real gateway, which is the only way to
judge the prose. Offline the prose is the deterministic template, so the four
starters that exist to be *written* rather than computed are marked as needing a
model rather than failed.

    uv run python -m scripts.check_starter_questions
    uv run python -m scripts.check_starter_questions --live
"""

from __future__ import annotations

import argparse
import sys
import time

from src.agent.graph import build_graph, run_campaign
from src.agent.state import CampaignState
from src.hardware.devices import LINEAR
from src.logging_setup import configure_logging
from src.ui.starters import OFFERED_KEYS, STARTERS

SHOT_BUDGET = 10**9
"""The budget the interface ships, so a starter is priced as a visitor would price it."""


def verdict_of(state: CampaignState) -> str:
    """Summarise in one word what the reader would be shown.

    Args:
        state: The finished campaign.

    Returns:
        A short label for the shape of the reply.
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
    if state["report"] is not None:
        return "reported"
    return "nothing"


def ask(question: str, live: bool) -> tuple[CampaignState, list[str], float]:
    """Run one starter through the graph as the Chat page would.

    Args:
        question: The starter's text, exactly as its button prints it.
        live: Reach the real gateway rather than the deterministic paths.

    Returns:
        The finished campaign, the nodes that ran in order, and the seconds taken.
    """
    started = time.monotonic()
    visited: list[str] = []
    state = run_campaign(
        question,
        shot_budget=SHOT_BUDGET,
        device=LINEAR,
        chat_model="auto" if live else None,
        search_corpus=live,
        fetch_external=False,
        suggest_followups=False,
        visited=visited,
    )
    return state, visited, time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    """Press every starter and print a row per answer.

    Args:
        argv: Command line. ``None`` reads the real one.

    Returns:
        Zero when every starter produced a structured answer, one otherwise.
    """
    parser = argparse.ArgumentParser(description="Check the interface's starter questions.")
    parser.add_argument("--live", action="store_true", help="call the real gateway")
    parser.add_argument("--only", default="", help="only starters containing this text")
    parser.add_argument(
        "--offered",
        action="store_true",
        help="only the eight that appear as buttons, which is what a visitor presses",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the pipeline's log lines")
    args = parser.parse_args(argv)

    if not args.quiet:
        configure_logging()

    keys = OFFERED_KEYS if args.offered else tuple(STARTERS)
    asked = [(key, STARTERS[key]) for key in keys if args.only.lower() in STARTERS[key].lower()]
    if not asked:
        print(f"no starter matches {args.only!r}")
        return 1

    failures: list[tuple[str, str]] = []
    covered: set[str] = set()
    print(f"{len(asked)} starter questions, {'live' if args.live else 'offline'}\n")
    for position, (key, question) in enumerate(asked, start=1):
        state, route, elapsed = ask(question, args.live)
        outcome = verdict_of(state)
        model = state["model"]
        covered.update(route)
        # Offline the prose branch has no model to write with, so an honest refusal
        # there is the correct behaviour rather than a defect. Every other shape --
        # blocked, declined, nothing at all -- is a defect in both modes.
        excused = not args.live and outcome == "not-answered"
        wrong = outcome in {"blocked", "declined", "nothing"} or (
            outcome == "not-answered" and args.live
        )
        if wrong:
            failures.append((key, outcome))
        mark = "FAIL" if wrong else "ok  "
        note = "  (prose needs a model; refused honestly offline)" if excused else ""
        print(f"{mark} {position:2}. [{key}] {question}")
        print(
            f"        {outcome:16} intent={state['intent'].intent:11} "
            f"problem={'-' if model is None else model.label()}  "
            f"nodes={len(set(route)):2}  {elapsed:.1f}s{note}"
        )
        print(f"        route: {' -> '.join(dict.fromkeys(route)) or '(none recorded)'}")

    print()
    every = {name for name in build_graph().get_graph().nodes if not name.startswith("__")}
    missed = sorted(every - covered)
    print(
        f"node coverage: {len(covered & every)} of {len(every)}"
        + (f"  --  never run: {', '.join(missed)}" if missed else "  --  every node exercised")
    )
    print()
    if failures:
        print(f"{len(failures)} of {len(asked)} starters did not produce a structured answer:")
        for key, outcome in failures:
            print(f"  {outcome:16} {key}")
        return 1
    print(f"all {len(asked)} starters produced a structured answer")
    return 0


if __name__ == "__main__":  # pragma: no cover - a script a person runs
    sys.exit(main())
