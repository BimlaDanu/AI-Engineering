"""Ask the real agent a handful of questions and log what each run actually did.

Why this exists, next to a green ``make check`` and a 23-case ``make evals``: both
of those are about outcomes, and the complaints this project keeps getting are about
*trajectories*. "It always goes route, decide, search" and "it declines things it
knows" are statements about which nodes ran and which actions the loop chose, and
neither is visible in a pass/fail column.

So this prints one line per question: the route, the actions with who chose each,
how long the answer took, and how many passages were cited. It calls the live
gateway -- about seven model calls per question -- which is why it is a script a
person runs deliberately rather than a test.

Not a test, deliberately: :mod:`tests` must stay offline and deterministic, and
these assertions would be about a model's judgement on the day.

    make live-check                          # all of them
    uv run python -m scripts.live_check arxiv  # only questions matching "arxiv"

The filter exists because one failing trajectory is usually the one worth reading the
whole log for, and six questions is about three minutes of model calls.
"""

from __future__ import annotations

import sys
import time

from src.agent.graph import Answer, ask
from src.logging_setup import configure_logging, get_logger
from src.settings import get_settings
from src.ui.starters import QUESTIONS as STARTER_QUESTIONS

LOG = get_logger("live_check")

REGRESSIONS: tuple[str, ...] = (
    # Each one is a failure that was reported, kept as the thing it is meant to show.
    "Why does the gap close at h = J?",
    "What is the ground-state energy at h = 1, and why is it that value?",
    "Can you teach me about IBM quantum technologies?",
    "Find recent arXiv papers on Trotter error for the Ising chain.",
    "Can you plan a quantum simulation of the Ising chain on NISQ devices?",
    "What is the capital of France?",
    # Answered with a magnetisation curve, and with a sentence saying the
    # derivatives had not been determined -- the wrong quantity, plus an admission
    # that the right one was missing. Both had the same cause: the sweep had no
    # derivatives observable for the model to ask for.
    "Ground state energy and its first and second derivatives of the transverse "
    "field Ising model as a function of the magnetic field h/J for L = 6 using "
    "the free fermion approach.",
)
"""The questions, chosen because each one was once answered wrongly."""

QUESTIONS: tuple[str, ...] = REGRESSIONS + STARTER_QUESTIONS
"""Everything this script can ask.

The starters are here because they are the five questions a visitor is most likely
to ask -- they are on buttons -- and because the claim made about them is a claim
about the loop: that between them they take different paths rather than five
variations of one search. Nothing offline can check that, since which actions the
loop chooses depends on what the grader thinks of the passages it got. So they are
asked here, where the trajectory is printed and the claim can be read off.

Pass a substring to ask a subset:

    uv run python -m scripts.live_check quantum      # the starters about the model
"""


def trajectory(answer: Answer) -> str:
    """Render the actions taken, with who chose each.

    Args:
        answer: A finished run.

    Returns:
        ``retrieve[policy] -> consult[model] -> finish[policy]``, or ``"(no loop)"``
        for a question answered without one -- a refusal, or a question about the
        agent itself.
    """
    steps = " -> ".join(f"{step.action}[{step.decided_by}]" for step in answer.steps)
    return steps or "(no loop)"


def chosen(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Pick the questions to ask.

    Args:
        patterns: Case-insensitive substrings from the command line.

    Returns:
        Every question when nothing was asked for, otherwise those matching any
        pattern. An unmatched pattern yields nothing rather than falling back to all
        six, so a typo costs no model calls.
    """
    if not patterns:
        return QUESTIONS
    wanted = tuple(pattern.lower() for pattern in patterns)
    return tuple(q for q in QUESTIONS if any(pattern in q.lower() for pattern in wanted))


def main() -> None:
    """Ask every question and log one summary line each."""
    configure_logging(secrets=[get_settings().openrouter_api_key.get_secret_value()])
    for question in chosen(tuple(sys.argv[1:])):
        started = time.perf_counter()
        answer = ask(question)
        LOG.info(
            "live_check_answer",
            extra={
                "question": question,
                "seconds": round(time.perf_counter() - started, 1),
                "status": answer.status,
                "route": answer.routing.route if answer.routing is not None else "",
                "trajectory": trajectory(answer),
                "citations": len(answer.citations),
                "tools": len(answer.tools),
            },
        )


if __name__ == "__main__":
    main()
