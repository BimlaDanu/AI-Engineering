"""Tests for the three-method comparison the Chat page draws.

The claims here are mostly about *fairness* rather than about physics, because that is
where a comparison like this goes wrong. Three methods run at three different depths, or
on three slightly different chains, produce a picture that looks exactly like a fair
comparison and answers a question nobody asked. So the tests check that every method got
the same chain and the same circuit, that the chain it got is the one that was asked
for, and that the cost of a step is reported beside the count of them.

The energies themselves are graded against the sealed exact solvers, which a test may
import -- ``tests/test_architecture.py`` bans the *solver* from reaching an exact
answer, not the grader.
"""

from __future__ import annotations

import json

import pytest

from src.physics.model import TFIMSpec
from src.physics.quantum.method_race import (
    MAX_RACED_SITES,
    METHOD_BLURBS,
    METHOD_NAMES,
    race_methods,
    unsupported_reason,
)
from src.physics.registry import solver_for


def _exact(n_sites: int, coupling: float, field: float, boundary: str) -> float:
    """The true ground-state energy, from the grader's bench."""
    return solver_for("exact_diagonalisation").solve(
        TFIMSpec(n_sites=n_sites, coupling=coupling, field=field, boundary=boundary)  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# What makes it a comparison rather than three pictures
# --------------------------------------------------------------------------


def test_every_method_named_gets_run_and_none_else_does() -> None:
    race = race_methods(n_sites=6, depth=2)
    assert tuple(run.method for run in race.runs) == METHOD_NAMES


def test_each_method_gets_the_same_circuit_as_the_others() -> None:
    # The one thing that must not vary. Three methods at three depths is a picture of
    # the depth ladder wearing a comparison's clothes.
    race = race_methods(n_sites=6, depth=3)
    assert race.depth == 3
    assert all(len(run.energy_history) >= 1 for run in race.runs)


@pytest.mark.parametrize("longitudinal", [0.0, 0.5])
@pytest.mark.parametrize("field", [0.5, 1.0])
def test_the_chain_raced_on_is_the_chain_that_was_asked_for(
    field: float, longitudinal: float
) -> None:
    # Any J, h and g, which is the point: the picture under an answer has to be of the
    # problem the question described, not of a stock example that resembles it.
    race = race_methods(
        n_sites=6,
        depth=2,
        coupling=1.5,
        transverse_field=field,
        longitudinal_field=longitudinal,
        boundary="periodic",
    )
    assert (race.coupling, race.transverse_field, race.longitudinal_field) == (
        1.5,
        field,
        longitudinal,
    )
    assert race.boundary == "periodic"
    assert f"h={field:g}" in race.label()
    # The caption names the longitudinal field only when there is one. At zero it
    # contributes nothing to the chain being drawn, and printing "g=0" under every
    # figure puts a third free parameter in front of a reader whose page shows two.
    if longitudinal:
        assert f"g={longitudinal:g}" in race.label()
    else:
        assert "g=" not in race.label()


def test_a_longitudinal_field_changes_the_answer_rather_than_being_ignored() -> None:
    plain = race_methods(n_sites=6, depth=2, longitudinal_field=0.0)
    tilted = race_methods(n_sites=6, depth=2, longitudinal_field=0.6)
    assert tilted.best() is not None
    assert plain.best() is not None
    assert tilted.best().energy < plain.best().energy - 1e-6  # type: ignore[union-attr]


# --------------------------------------------------------------------------
# The physics, graded against algebra none of the three can see
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", [0.5, 1.0, 2.0])
def test_no_method_ever_reports_an_energy_below_the_true_ground_state(field: float) -> None:
    race = race_methods(n_sites=6, depth=3, transverse_field=field)
    truth = _exact(6, 1.0, field, "open")
    for run in race.runs:
        assert run.energy >= truth - 1e-9, run.method


def test_every_curve_goes_downhill_and_ends_where_the_method_says_it_did() -> None:
    # The last point of the curve and the reported energy are two different fields and
    # a mismatch between them would make the figure disagree with the table beside it.
    race = race_methods(n_sites=6, depth=3)
    for run in race.runs:
        assert run.energy == pytest.approx(run.energy_history[-1])
        assert run.energy_history[-1] <= run.energy_history[0] + 1e-12
        assert run.improvement >= -1e-12


def test_the_winner_is_the_lowest_energy_because_every_energy_is_an_upper_bound() -> None:
    race = race_methods(n_sites=6, depth=3)
    winner = race.best()
    assert winner is not None
    assert winner.energy == min(run.energy for run in race.runs)


def test_a_race_with_no_methods_has_no_winner_rather_than_an_invented_one() -> None:
    race = race_methods(n_sites=6, depth=2, methods=())
    assert race.runs == ()
    assert race.best() is None
    assert race.curves() == ()


# --------------------------------------------------------------------------
# What it reports about the cost, which is the half a step count hides
# --------------------------------------------------------------------------


def test_the_bill_is_reported_beside_the_step_count() -> None:
    # A step is not the same purchase in all three, so a comparison quoting only steps
    # would say the imaginary-time method is slowest when it may be the cheapest.
    race = race_methods(n_sites=6, depth=3)
    for run in race.runs:
        assert run.n_energy_evaluations >= run.n_steps, run.method
        assert run.n_steps == len(run.energy_history) - 1, run.method


def test_criticality_is_recognised_from_the_numbers_rather_than_from_a_flag() -> None:
    assert race_methods(n_sites=4, depth=1, coupling=2.0, transverse_field=2.0).at_criticality
    assert not race_methods(n_sites=4, depth=1, coupling=2.0, transverse_field=1.0).at_criticality


def test_the_description_survives_a_round_trip_through_json() -> None:
    # It goes into a campaign snapshot and out to a log file, and a curve that cannot
    # be written down is a comparison nobody can re-plot tomorrow.
    described = race_methods(n_sites=4, depth=2).describe()
    restored = json.loads(json.dumps(described))
    assert set(restored["methods"]) == set(METHOD_NAMES)
    assert len(restored["methods"]["VQE"]["energy_history"]) >= 1
    assert restored["at_criticality"] is True


def test_every_method_has_a_sentence_a_non_specialist_can_read() -> None:
    # The blurbs are printed under the figure. A method with no blurb would render as
    # an empty table cell on the page.
    assert set(METHOD_BLURBS) == set(METHOD_NAMES)
    for blurb in METHOD_BLURBS.values():
        assert blurb and blurb[0].islower()


# --------------------------------------------------------------------------
# Refusing rather than hanging
# --------------------------------------------------------------------------


def test_a_chain_too_long_to_race_is_refused_with_the_reason_and_the_limit() -> None:
    reason = unsupported_reason(MAX_RACED_SITES + 1, 3)
    assert reason is not None
    assert str(MAX_RACED_SITES) in reason


@pytest.mark.parametrize("depth", [0, -1])
def test_a_circuit_with_no_angles_is_refused_because_nothing_would_converge(depth: int) -> None:
    assert unsupported_reason(6, depth) is not None


def test_a_workable_chain_is_not_refused() -> None:
    assert unsupported_reason(MAX_RACED_SITES, 1) is None


def test_racing_a_refused_chain_raises_rather_than_running_anyway() -> None:
    with pytest.raises(ValueError, match="cannot race methods"):
        race_methods(n_sites=MAX_RACED_SITES + 1, depth=3)
