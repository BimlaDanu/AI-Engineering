"""The agent's own entry point: a question in, a feasibility report out.

Everything the interface does, from a terminal and without one, so the graph can
be scripted, piped and debugged with no browser. The interface and this module
are two callers of the same function.

    make campaign                                  the default question
    python -m src.agent.campaign "10 magnets in a row, equal field. Feasible?"
    python -m src.agent.campaign --framing vendor --offline
    python -m src.agent.campaign --json > run.json
    python -m src.agent.campaign --fast-model google/gemini-2.5-flash-lite

Costs a credential and tokens unless ``--offline`` is given, so it is not part of
``make check``.

The default output is the report as it would be read, with the campaign trail
above it. ``--json`` prints the snapshot instead -- primitives only, for piping
elsewhere. Both come from the same finished campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Literal, cast

from src.agent.graph import DEFAULT_SHOT_BUDGET, run_campaign
from src.agent.memory import Memory
from src.agent.model_selection import ModelPool, Tier
from src.agent.state import CampaignState, snapshot
from src.logging_setup import configure_logging, get_logger
from src.settings import get_settings

_log = get_logger("agent.campaign")

DEFAULT_QUESTION = (
    "We have a line of 8 magnets, each pulling on its neighbours about as strongly "
    "as a sideways field pushes on all of them. The ends are free. Would a quantum "
    "computer find its lowest-energy arrangement faster than a classical one?"
)
"""What the campaign runs when nothing was asked.

A worked example rather than a placeholder. Somebody running ``make campaign`` to
see whether the thing works should get a complete, honest answer to a real
question, not a prompt to supply one.
"""

__all__ = ["DEFAULT_SHOT_BUDGET", "main", "parse_arguments"]
"""The budget is re-exported rather than restated.

It used to be defined here as fifty million, which is the reason this note exists.
The number was raised to a billion in :mod:`src.agent.graph` and in the interface,
and the test that was written to stop the three drifting apart compared only two of
them -- so the command line kept the old figure, and ``make campaign`` refused the
first rung of its own advertised worked example: nine hundred thousand shots short,
zero configurations run, and a confident ``no`` that was an accounting result rather
than a physical one. A worked example that runs nothing is worse than no default,
which is exactly what the docstring on the surviving constant already said.

Re-exporting keeps the command line, the library and the interface on one number by
construction instead of by vigilance, and the name stays importable from here so the
existing callers and the arithmetic in :data:`src.agent.graph.DEFAULT_SHOT_BUDGET`
cannot disagree again.
"""

EXIT_OK = 0
EXIT_NO_VERDICT = 1


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments to parse. ``None`` reads the real command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.agent.campaign",
        description="Run one feasibility campaign from a question to a written verdict.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="The problem, in ordinary language. Omit for a worked example.",
    )
    parser.add_argument(
        "--framing",
        choices=("neutral", "vendor", "skeptical"),
        default="neutral",
        help="The voice the question is asked in. Recorded, never acted on.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOT_BUDGET,
        help="Total measurements the campaign may spend.",
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
        "--fetch",
        action="store_true",
        help="Allow a search that finds nothing locally to reach an external index.",
    )
    parser.add_argument(
        "--remember",
        action="store_true",
        help="Recall this conversation and record the outcome. Off by default, so a "
        "one-off question leaves nothing behind.",
    )
    parser.add_argument("--user", default="", help="Who is asking, for memory.")
    parser.add_argument("--thread", default="", help="Which conversation this belongs to.")
    for tier in ("fast", "standard", "strong"):
        parser.add_argument(
            f"--{tier}-model",
            default="",
            help=f"Model slug to serve the {tier} tier, overriding configuration.",
        )
    parser.add_argument("--json", action="store_true", help="Print the snapshot instead of prose.")
    return parser.parse_args(argv)


def _pool(arguments: argparse.Namespace) -> ModelPool:
    """Build the set of models this run may call.

    Args:
        arguments: The parsed command line.

    Returns:
        A pool honouring any per-tier override given on the command line. An
        offline pool serves nothing, which is what makes ``--offline`` a single
        switch rather than a mode every call site has to check.
    """
    overrides: dict[Tier, str] = {}
    for tier in ("fast", "standard", "strong"):
        chosen = getattr(arguments, f"{tier}_model", "")
        if chosen:
            overrides[tier] = chosen
    return ModelPool(offline=arguments.offline, overrides=overrides)


def _memory(arguments: argparse.Namespace) -> Memory:
    """Decide what this run remembers.

    Args:
        arguments: The parsed command line.

    Returns:
        A memory when ``--remember`` was given, otherwise one that stores nothing.
        Off by default because a question asked from a shell is usually a one-off,
        and writing somebody's question to a file they did not ask for is the kind
        of default that should have to be typed.
    """
    if not arguments.remember:
        return Memory.disabled()
    settings = get_settings()
    return Memory(
        user=arguments.user or "anonymous",
        thread=arguments.thread or "cli",
        settings=settings,
    )


def render(state: CampaignState) -> str:
    """Write the finished campaign out as prose.

    Args:
        state: The finished campaign.

    Returns:
        The trail of what happened, then the report. The trail comes first
        deliberately: a reader who disagrees with the verdict needs to see what it
        was based on before they read it, not after.
    """
    lines = ["What the campaign did", "=====================", ""]
    lines.extend(f" * {note}" for note in state["notes"])
    report = state.get("report")
    if report:
        lines.extend(["", "The report", "==========", "", report])
    verdict = state.get("verdict")
    if verdict is None:
        lines.extend(["", "No verdict was reached."])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run one campaign and print it.

    Args:
        argv: Command-line arguments. ``None`` reads the real command line.

    Returns:
        The process exit code. Non-zero when no verdict was reached, so this can
        be used in a script that needs to know whether it got an answer.
    """
    arguments = parse_arguments(argv)
    configure_logging()
    framing = cast(Literal["neutral", "vendor", "skeptical"], arguments.framing)
    memory = _memory(arguments)

    _log.info(
        "campaign_cli_start",
        extra={"framing": framing, "shots": arguments.shots, "offline": arguments.offline},
    )
    state = run_campaign(
        arguments.question,
        framing=framing,
        shot_budget=arguments.shots,
        models=_pool(arguments),
        search_corpus=not (arguments.offline or arguments.no_corpus),
        fetch_external=arguments.fetch,
        memory=memory,
    )

    if arguments.json:
        print(json.dumps(snapshot(state), indent=2))
    else:
        print(render(state))

    return EXIT_OK if state.get("verdict") is not None else EXIT_NO_VERDICT


if __name__ == "__main__":
    sys.exit(main())
