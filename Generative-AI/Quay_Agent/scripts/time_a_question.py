"""Time one question through the graph and say where the wait actually went.

Why this exists: "the chat is slow" is a complaint about wall clock, and wall clock
here is the sum of two unrelated things -- sequential model round trips, and local
work (retrieval, an exact sweep, a figure). They are fixed by opposite means, so
guessing which dominates wastes the fix.

Two numbers come out of it. The **model total** is read from the pool's own ledger,
which times every attempt anyway, so this script adds no measurement of its own to
be wrong. Wall clock minus that total is **local work**: everything this repository
does itself. Run it once with ``--offline`` and the model total is zero by
construction, leaving the local floor -- no provider can make an answer faster than
that, so a large offline figure is a defect here rather than a slow gateway.

    uv run python -m scripts.time_a_question --offline
    uv run python -m scripts.time_a_question "Is 12 spins worth it?"
    uv run python -m scripts.time_a_question --serial   # prefetch off, for comparison

``--serial`` turns off :mod:`src.agent.prefetch`, so the saving from overlapping the
front-of-graph calls can be measured rather than assumed.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from typing import cast

from src.agent.graph import run_campaign
from src.agent.model_selection import ModelPool, Tier
from src.hardware.devices import LINEAR
from src.physics.registry import field_sweep_bench

DEFAULT_QUESTION = "Can you plot the ground-state energy and its first two derivatives vs h/J?"
"""What to time when the caller names nothing.

A curve question, because that is the shape the complaint arrived about and it is the
longest of the prose routes: retrieval, a tool consultation, an exact sweep and a
figure all happen on it.
"""

SHOT_BUDGET = 1_000_000_000


def main(argv: Sequence[str] | None = None) -> int:
    """Run one campaign and print the breakdown.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--offline", action="store_true", help="skip every model call")
    parser.add_argument("--serial", action="store_true", help="turn the prefetch off")
    parser.add_argument("--fast", default="", help="slug to serve the fast tier")
    parser.add_argument("--standard", default="", help="slug to serve the standard tier")
    parser.add_argument("--strong", default="", help="slug to serve the strong tier")
    arguments = parser.parse_args(argv)

    overrides: dict[str, str] = {
        tier: slug
        for tier, slug in (
            ("fast", arguments.fast),
            ("standard", arguments.standard),
            ("strong", arguments.strong),
        )
        if slug
    }
    pool = ModelPool(offline=arguments.offline, overrides=cast("dict[Tier, str]", overrides))
    order: list[str] = []
    started = time.perf_counter()
    state = run_campaign(
        arguments.question,
        shot_budget=SHOT_BUDGET,
        device=LINEAR,
        models=pool,
        reference_bench=field_sweep_bench(),
        parallel_model_calls=not arguments.serial,
        visited=order,
    )
    wall = time.perf_counter() - started

    print(f"question: {arguments.question}")
    print(
        f"mode:     {'offline' if arguments.offline else 'live'}"
        f"{', prefetch off' if arguments.serial else ''}\n"
    )
    print(f"{'seconds':>8}  {'task':<20} {'model':<28} ok")
    for entry in pool.ledger.entries:
        print(
            f"{entry['elapsed_s']:8.3f}  {entry['task']:<20} "
            f"{entry['model']:<28} {'yes' if entry['ok'] else 'NO'}"
        )
    model_time = pool.ledger.total_seconds
    print(f"\n{model_time:8.3f}s  model calls ({len(pool.ledger.entries)} attempts)")
    print(f"{wall - model_time:8.3f}s  local work")
    print(f"{wall:8.3f}s  WALL CLOCK over {len(order)} node runs")
    print(f"\nroute: {' -> '.join(order)}")
    print(f"intent: {state['intent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
