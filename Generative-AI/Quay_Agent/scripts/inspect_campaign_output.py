"""Print a finished campaign's report exactly as the Chat page renders it.

Kept in the repository rather than run from a temp directory because the quality of
the written answer is the thing most often judged by eye, and judging it by eye needs
a command that reproduces the same text tomorrow.
"""

from __future__ import annotations

import sys

from src.agent.graph import run_campaign
from src.hardware.devices import device_for


def main() -> None:
    """Run one campaign and dump every field a reader would see.

    Offline by default, because that is the mode the test suite runs in and the one
    whose output has to stand on its own without a key. Pass ``--live`` to spend real
    model calls.
    """
    arguments = [word for word in sys.argv[1:] if word != "--live"]
    live = "--live" in sys.argv
    question = (
        arguments[0]
        if arguments
        else "Is quantum hardware worth it for a 10-spin critical Ising chain?"
    )
    visited: list[str] = []
    state = run_campaign(
        question,
        shot_budget=10**9,
        device=device_for("heavy-hex-27"),
        chat_model="auto" if live else None,
        search_corpus=True,
        fetch_external=False,
        visited=visited,
    )
    print("=" * 78)
    print("ROUTE:", " -> ".join(visited))
    print("=" * 78)
    verdict = state["verdict"]
    print("INTENT:", state["intent"])
    print("VERDICT:", verdict)
    print("=" * 78)
    print("REPORT:")
    print(state["report"])
    print("=" * 78)
    print("NOTES:")
    for note in state["notes"]:
        print(" -", note)
    print("=" * 78)
    for key in ("runs", "ruled_out", "citations", "followups"):
        if key in state:
            print(key.upper(), "=", state[key])
            print("-" * 78)


if __name__ == "__main__":
    main()
