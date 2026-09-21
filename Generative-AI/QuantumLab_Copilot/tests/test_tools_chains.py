"""Tests for the comparison-chain tool.

Two things are being pinned here. The first is that a number arriving through a
tool is verified exactly like one arriving through the main run -- a tool must not
be a back door to an unchecked result. The second is that the tool stops short of
the cost gate rather than walking through it, since there is no human inside a
tool call to ask.

Chains stay at ``L <= 8``, as everywhere in this suite.
"""

from __future__ import annotations

from src.agent.selection import APPROVAL_SITES
from src.physics.model import TFIMSpec
from src.tools.chains import MAX_COMPARISON_SITES, MIN_COMPARISON_SITES, compare_chain

BASE = TFIMSpec(n_sites=8, coupling=1.0, field=1.0)


def test_a_comparison_point_is_cross_checked() -> None:
    run = compare_chain(BASE, 6)
    assert run.ok
    assert run.check is not None
    assert run.check.is_corroborated
    assert len(run.check.results) == 2


def test_the_hamiltonian_is_inherited_and_only_the_length_changes() -> None:
    # The knob owns the physics. A tool that could also change J or h would
    # reintroduce exactly the failure the knob exists to prevent: a model
    # mis-reading a number out of a sentence and computing a different problem.
    base = TFIMSpec(n_sites=8, coupling=2.0, field=0.5, boundary="open")
    run = compare_chain(base, 4)
    assert run.spec is not None
    assert (run.spec.coupling, run.spec.field, run.spec.boundary) == (2.0, 0.5, "open")
    assert run.spec.n_sites == 4


def test_the_gate_the_tool_stops_at_is_below_the_one_that_asks_a_human() -> None:
    # The relationship is the invariant, not the numbers: raising one without the
    # other would let a tool call start a run the interface would have paused.
    assert MAX_COMPARISON_SITES < APPROVAL_SITES


def test_a_chain_that_would_need_approval_is_declined() -> None:
    run = compare_chain(BASE, APPROVAL_SITES)
    assert not run.ok
    assert "approval" in run.detail
    assert run.check is None


def test_a_chain_beyond_the_hard_cap_is_declined_too() -> None:
    assert not compare_chain(BASE, 64).ok


def test_a_nonsensical_length_is_declined_rather_than_raised() -> None:
    for length in (0, -3, 1):
        run = compare_chain(BASE, length)
        assert not run.ok, length


def test_the_shortest_allowed_chain_still_runs() -> None:
    assert compare_chain(BASE, MIN_COMPARISON_SITES).ok


def test_asking_for_the_chain_already_solved_is_declined() -> None:
    # Not an error, just pointless: it would pay for a duplicate of the number
    # the answer already has.
    run = compare_chain(BASE, BASE.n_sites)
    assert not run.ok
    assert "already solved" in run.detail


def test_an_unverifiable_chain_says_so_in_one_line() -> None:
    odd = TFIMSpec(n_sites=7)  # the closed form needs an even ring
    run = compare_chain(odd, 5)
    assert run.ok
    assert "unverified" in run.explain()


def test_the_explanation_carries_the_number_and_the_verdict() -> None:
    explanation = compare_chain(BASE, 4).explain()
    assert "E0 = " in explanation
    assert "corroborated" in explanation
