"""What the person watching a campaign is told while it runs.

The claim under test is not that the wording is nice. It is that the commentary
cannot lie: a line appears only for a node that actually ran, a repeated node is
numbered so four rungs of the depth ladder do not read as one stuck loop, and the
node that finishes the answer is identified correctly -- because the chat page draws
the answer on the strength of that name, and naming the wrong one would either show
an empty card or hold a finished answer back.
"""

from __future__ import annotations

import pytest

from src.agent.graph import run_campaign
from src.agent.model_selection import ModelPool
from src.ui import progress


def test_every_node_the_graph_can_run_has_a_phrase() -> None:
    # The one that rots. A node added to the graph without a line here would be
    # shown to a visitor by its function name, which is the kind of leak nobody
    # notices until a screenshot.
    ran: list[str] = []
    for question in ("Is a ring of 8 spins worth it?", "What is the capital of France?"):
        run_campaign(question, shot_budget=10**9, models=ModelPool(offline=True), visited=ran)
    assert ran, "the campaign ran no nodes at all"
    missing = sorted({node for node in ran if node not in progress.PHRASES})
    assert not missing, f"no plain-English line for: {missing}"


def test_the_answer_node_is_one_the_graph_actually_runs() -> None:
    # The chat page draws the answer when this node reports itself. A name the graph
    # never emits would mean the answer is never drawn early, silently -- the page
    # would still work, just as slowly as before, with no test to say so.
    ran: list[str] = []
    run_campaign(
        "Is a ring of 8 spins worth it?",
        shot_budget=10**9,
        models=ModelPool(offline=True),
        visited=ran,
    )
    assert progress.ANSWER_READY in ran


def test_a_repeated_node_is_numbered_and_a_single_one_is_not() -> None:
    trail = progress.Trail()
    assert trail.record("retrieve") == "searched the notes for background"
    assert trail.record("plan") == "designed a configuration to try"
    assert trail.record("plan") == "designed a configuration to try (2nd configuration)"
    assert trail.record("plan") == "designed a configuration to try (3rd configuration)"


def test_a_node_that_is_not_a_rung_is_never_numbered() -> None:
    # Retrieval running twice is a defect worth seeing plainly. Dressing it up as
    # "2nd configuration" would describe it as the ladder doing its job.
    trail = progress.Trail()
    trail.record("retrieve")
    assert trail.record("retrieve") == "searched the notes for background"


def test_the_trail_reports_what_ran() -> None:
    trail = progress.Trail()
    assert not trail.ran("scribe")
    assert trail.latest() == "reading the question"
    trail.record("scribe")
    assert trail.ran("scribe")
    assert trail.latest() == "wrote the report"


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
    ],
)
def test_ordinals_read_correctly(count: int, expected: str) -> None:
    assert progress._ordinal(count) == expected


def test_an_unknown_node_is_shown_rather_than_swallowed() -> None:
    # Falling back to the raw name rather than to nothing. A dropped line would make
    # the *next* line a lie about what the campaign is doing.
    assert progress.phrase("a-node-nobody-wrote-a-line-for") == "a-node-nobody-wrote-a-line-for"
