"""Print one campaign's answer as text, for reading it rather than measuring it.

``scripts/live_check.py`` reports a *trajectory* -- which branch ran, what was refused,
how much of the budget went -- and deliberately not the prose, because forty questions
of prose is not something anybody reads. That leaves no way to look at the one thing a
reader of the application actually sees, which is how an answer that was structurally
fine and substantively empty went unnoticed: every trajectory was green.

So this prints the answer, the follow-ups and the sources for a single question, and
nothing about the route. It calls the live gateway and costs tokens.

Usage::

    uv run python -m scripts.show_answer "Could you teach me about VQE, QAOA and VarQITE?"
    uv run python -m scripts.show_answer --starter methods_taught
"""

from __future__ import annotations

import argparse
import sys

from src.agent.graph import run_campaign
from src.logging_setup import configure_logging
from src.ui.starters import STARTERS


def main() -> int:
    """Ask one question and print what the reader would be shown.

    Returns:
        Zero when an answer came back, one when the campaign produced none --
        so a shell loop over several questions can tell the difference.
    """
    parser = argparse.ArgumentParser(description="Print one campaign's answer.")
    parser.add_argument("question", nargs="?", default="", help="the question to ask")
    parser.add_argument(
        "--starter",
        default="",
        help=f"ask a starter by key instead: {', '.join(sorted(STARTERS))}",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the pipeline's own log lines, leaving only the answer",
    )
    args = parser.parse_args()

    question = STARTERS[args.starter] if args.starter else args.question
    if not question:
        parser.error("give a question or a --starter key")
    if not args.quiet:
        configure_logging()

    state = run_campaign(question)
    answer = state["answer"]

    print("=" * 78)
    print(f"QUESTION: {question}")
    print("=" * 78)
    # The code branch writes no prose at all, so a script that only knows how to print
    # `answer` reports "no answer" for a question that was answered perfectly well --
    # which is the same blindness this script was written to fix, one branch over.
    drafted = state["draft"]
    if drafted is not None and drafted.written:
        print(f"\nwritten by: model   {drafted.explain()}\n")
        if drafted.preamble:
            print(drafted.preamble, end="\n\n")
        if drafted.mathematics:
            print(drafted.mathematics, end="\n\n")
        print(f"```{drafted.language}\n{drafted.code}\n```")
        return 0
    if answer is None:
        print("\nNo answer. The campaign reached none.\n")
        for note in state["notes"]:
            print(f"  note: {note}")
        return 1
    print(f"\nwritten by: {answer.written_by}   passages cited: {answer.cited}\n")
    print(answer.text)
    if answer.rests_on:
        print(f"\nRESTS ON: {answer.rests_on}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
