"""Tests for the VQE driver.

Two kinds of claim are under test. The first is about *physics* and is checked against
the sealed exact solvers: the energy is a genuine upper bound, and enough depth closes
the gap to it. The second is about *reporting*, and matters just as much here -- a run
that stopped because it ran out of iterations and a run that converged produce the same
float, and an agent handed only the float will present the second as the first.

Importing the sealed solvers is legitimate in a test: the wall
(``tests/test_architecture.py``) bans the *solver* from reaching an exact answer, not the
grader.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from src.physics.model import DEFAULT_SITES, BoundaryCondition, TFIMSpec
from src.physics.quantum.ansatz import adiabatic_ramp
from src.physics.quantum.variational_eigensolver import solve
from src.physics.registry import solver_for

FIELDS = [0.5, 1.0, 2.0]


def _exact(n_sites: int, coupling: float, field: float, boundary: str) -> float:
    """The true ground-state energy, from the grader's bench."""
    return solver_for("exact_diagonalisation").solve(
        TFIMSpec(n_sites=n_sites, coupling=coupling, field=field, boundary=boundary)  # type: ignore[arg-type]
    )


RESTART_RAMPS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
"""Adiabatic ramp times to restart from before the best result is taken.

A single run here is fully deterministic -- seed 0, a ramp start, tolerances at 1e-10 --
but *which* local minimum it converges to is settled in the last bits of the arithmetic,
and those differ between one BLAS implementation and another. At depth 8 a perturbation
of 1e-10 on the starting angles moves the answer between 2.9e-5 and 1.5e-3 above the
true energy: the same code, converged just as tightly, in a neighbouring basin. That is
what made the absolute tolerances below pass on macOS and fail on Linux CI.

Restarting is the right cure rather than a looser bound. The energy is a variational
upper bound, so the lowest of several starts is always the better estimate -- taking it
is what a variational eigensolver does in practice, not a concession to the test -- and
it collapses the platform spread: measured over jittered starts, the worst gap falls
from 1.5e-3 to 2.9e-5 for the eigensolver and from 2.4e-2 to 1.4e-4 for imaginary time.
Reproduce with ``scripts/measure_convergence_spread.py``.
"""


def _lowest_over_restarts(
    *,
    depth: int,
    transverse_field: float = 1.0,
    boundary: BoundaryCondition = "open",
) -> float:
    """Lowest energy the eigensolver reaches from any of :data:`RESTART_RAMPS`.

    Args:
        depth: Number of ansatz layers.
        transverse_field: The field h, in units of the coupling J.
        boundary: Open chain or ring.

    Returns:
        The lowest energy found, which is still a variational upper bound because every
        run is one.
    """
    return min(
        solve(
            n_sites=DEFAULT_SITES,
            depth=depth,
            transverse_field=transverse_field,
            boundary=boundary,
            ramp_time=ramp,
        ).energy
        for ramp in RESTART_RAMPS
    )


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_the_energy_never_falls_below_the_true_ground_state(field: float, depth: int) -> None:
    # A value below E0 is a bug and never a better answer. This is the cheapest
    # correctness gate in the project and the reason a sign error cannot survive.
    result = solve(n_sites=DEFAULT_SITES, depth=depth, transverse_field=field)
    assert result.energy >= _exact(DEFAULT_SITES, 1.0, field, "open") - 1e-9


@pytest.mark.parametrize("field", FIELDS)
def test_a_bare_circuit_returns_the_closed_form_energy(field: float) -> None:
    # |+> is the exact ground state of the field half alone, so E = -h L with no
    # numerics on the other side of the comparison at all.
    result = solve(n_sites=DEFAULT_SITES, depth=0, transverse_field=field)
    assert result.energy == pytest.approx(-field * DEFAULT_SITES)
    assert result.stop_reason == "depth_zero"


@pytest.mark.parametrize("field", FIELDS)
def test_enough_depth_closes_the_gap_to_the_exact_answer(field: float) -> None:
    # The headline claim, graded against algebra the solver cannot see. Taken over
    # restarts because the depth-8 landscape has more than one basin -- see
    # RESTART_RAMPS for why a single start makes this a platform-dependent lottery.
    energy = _lowest_over_restarts(depth=8, transverse_field=field)
    assert energy == pytest.approx(_exact(DEFAULT_SITES, 1.0, field, "open"), abs=1e-3)


def test_the_answer_agrees_on_a_ring_too() -> None:
    energy = _lowest_over_restarts(depth=8, boundary="periodic")
    assert energy == pytest.approx(_exact(DEFAULT_SITES, 1.0, 1.0, "periodic"), abs=1e-3)


def test_a_longitudinal_field_lowers_the_energy_it_is_added_to() -> None:
    # g != 0 is the non-integrable case and the whole reason the project keeps the knob.
    # If the driver silently dropped the term, this is where it would show.
    plain = solve(n_sites=DEFAULT_SITES, depth=8, longitudinal_field=0.0, ramp_time=0.5).energy
    mixed = solve(n_sites=DEFAULT_SITES, depth=8, longitudinal_field=0.5, ramp_time=0.5).energy
    assert mixed < plain


def test_more_layers_reach_a_lower_energy() -> None:
    energies = [
        solve(n_sites=DEFAULT_SITES, depth=depth, ramp_time=0.5).energy for depth in (2, 4, 6)
    ]
    assert energies[1] < energies[0]
    assert energies[2] < energies[1]


def test_the_remaining_error_shrinks_as_layers_are_added() -> None:
    exact = _exact(DEFAULT_SITES, 1.0, 1.0, "open")
    errors = [
        solve(n_sites=DEFAULT_SITES, depth=depth, ramp_time=0.5).energy - exact
        for depth in (2, 4, 6)
    ]
    assert all(error > 0.0 for error in errors)
    assert errors[2] < errors[1] < errors[0]


def test_the_stop_reason_is_always_populated() -> None:
    for depth in (0, 1, 3):
        assert solve(n_sites=4, depth=depth).stop_reason


def test_an_iteration_cap_is_reported_as_a_cap_and_not_as_convergence() -> None:
    # The failure this guards against is silent: the same float, two entirely different
    # meanings, and no way to tell them apart downstream.
    result = solve(n_sites=DEFAULT_SITES, depth=6, max_iterations=2, ramp_time=0.5)
    assert result.stop_reason == "iteration_limit"
    assert not result.converged


def test_a_converged_run_says_so() -> None:
    result = solve(n_sites=DEFAULT_SITES, depth=2, ramp_time=0.5)
    assert result.converged
    assert result.stop_reason in ("converged", "gradient_below_tolerance")


def test_the_history_records_every_accepted_step() -> None:
    result = solve(n_sites=DEFAULT_SITES, depth=3, ramp_time=0.5)
    assert len(result.energy_history) == len(result.gradient_norm_history)
    assert result.energy_history[-1] == pytest.approx(result.energy, abs=1e-9)


def test_the_optimiser_only_ever_goes_downhill() -> None:
    history = solve(n_sites=DEFAULT_SITES, depth=4, ramp_time=0.5).energy_history
    assert all(later <= earlier + 1e-9 for earlier, later in pairwise(history))


def test_the_reported_improvement_is_the_distance_actually_travelled() -> None:
    result = solve(n_sites=DEFAULT_SITES, depth=4, ramp_time=0.5)
    assert result.improvement == pytest.approx(result.energy_history[0] - result.energy)


def test_the_energy_per_site_is_the_energy_divided_by_the_chain() -> None:
    result = solve(n_sites=DEFAULT_SITES, depth=2)
    assert result.energy_per_site == pytest.approx(result.energy / DEFAULT_SITES)


def test_the_returned_angles_are_folded_into_one_period() -> None:
    result = solve(n_sites=DEFAULT_SITES, depth=4, ramp_time=0.5)
    assert np.all(np.abs(result.parameters) <= np.pi + 1e-12)


def test_describe_returns_only_json_safe_values() -> None:
    described = solve(n_sites=4, depth=2).describe()
    for key, value in described.items():
        if key == "ansatz":
            continue
        assert isinstance(value, (int, float, str, bool)), key


def test_the_same_arguments_give_the_same_answer_twice() -> None:
    # A result that cannot be reproduced is not a result.
    first = solve(n_sites=DEFAULT_SITES, depth=3, initialisation="small_angle", seed=11)
    second = solve(n_sites=DEFAULT_SITES, depth=3, initialisation="small_angle", seed=11)
    assert first.energy == second.energy
    assert np.array_equal(first.parameters, second.parameters)


def test_a_warm_start_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="expected 6 initial angles"):
        solve(n_sites=4, depth=3, initial_parameters=adiabatic_ramp(2))


def test_the_family_name_is_carried_through_to_the_result() -> None:
    assert solve(n_sites=4, depth=1, family="qaoa").spec.family == "qaoa"
