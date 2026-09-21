r"""Tests for VarQITE: the method that reaches the ground state without searching.

Three kinds of claim are under test, and the first two are held against algebra that
shares nothing with the code producing them -- which is this project's own rule applied
to its own new module.

**The derivatives** are checked against a central difference of the circuit itself. The
module builds :math:`\partial_k\lvert\psi\rangle` by inserting a generator into one
forward sweep; the test rebuilds it by running the whole circuit twice at shifted
angles. The two have no line of code in common.

**The metric** is checked against infidelity. :func:`fubini_study_metric` assembles
:math:`g` from analytic derivative states; the test measures
:math:`1 - \lvert\langle\psi\vert\phi\rangle\rvert^2` between two nearby circuits, which
is what :math:`g` is *defined* to be the quadratic form of and which is computed here
from two state overlaps and nothing else.

**The physics** is graded against the sealed exact solvers -- legitimate in a test,
since ``tests/test_architecture.py`` bans the *solver* from reaching an exact answer,
not the grader.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from src.physics.model import DEFAULT_SITES, TFIMSpec
from src.physics.quantum.ansatz import AnsatzSpec
from src.physics.quantum.imaginary_time_evolution import (
    ImaginaryTimeResult,
    evolve_in_imaginary_time,
    fubini_study_metric,
    imaginary_time_velocity,
    state_derivatives,
)
from src.physics.quantum.statevector import diagonal_energies, evolve
from src.physics.quantum.variational_eigensolver import solve
from src.physics.registry import solver_for

FIELDS = [0.5, 1.0, 2.0]

RESTART_RAMPS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
"""Adiabatic ramp times to restart from before the best result is taken.

See the identical constant in ``tests/test_variational_eigensolver.py`` for the full
reasoning. In short: the run is deterministic, but which local minimum it settles in is
decided in the last bits of the arithmetic, and those differ between one BLAS
implementation and another. At depth 8 a 1e-11 perturbation of the start moves this
solver's answer between 2.7e-6 and 2.4e-2 above the true energy. Taking the lowest of
several starts is what a variational method does in practice -- every run is an upper
bound, so the lowest is the best estimate -- and it pulls the worst case down to 1.4e-4.
Reproduce with ``scripts/measure_convergence_spread.py``.
"""


def _lowest_over_restarts(*, depth: int, transverse_field: float = 1.0) -> float:
    """Lowest energy imaginary-time evolution reaches from any of :data:`RESTART_RAMPS`.

    Args:
        depth: Number of ansatz layers.
        transverse_field: The field h, in units of the coupling J.

    Returns:
        The lowest energy found, still a variational upper bound because every run is one.
    """
    return min(
        evolve_in_imaginary_time(
            n_sites=6, depth=depth, transverse_field=transverse_field, ramp_time=ramp
        ).energy
        for ramp in RESTART_RAMPS
    )


def _exact(n_sites: int, coupling: float, field: float, boundary: str) -> float:
    """The true ground-state energy, from the grader's bench."""
    return solver_for("exact_diagonalisation").solve(
        TFIMSpec(n_sites=n_sites, coupling=coupling, field=field, boundary=boundary)  # type: ignore[arg-type]
    )


def _angles(spec: AnsatzSpec, seed: int = 0) -> np.ndarray:
    """A reproducible set of angles well away from any special point."""
    return np.asarray(np.random.default_rng(seed).normal(size=spec.n_parameters), dtype=np.float64)


# --------------------------------------------------------------------------
# The derivatives, against a circuit run twice
# --------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("longitudinal", [0.0, 0.4])
def test_the_analytic_derivatives_match_running_the_circuit_at_shifted_angles(
    depth: int, longitudinal: float
) -> None:
    # One forward sweep with a generator inserted, against two whole circuits per
    # angle. If the field half's sign convention were wrong -- the one mistake this
    # kind of code actually makes -- the two would disagree in exactly half the rows.
    spec = AnsatzSpec(n_qubits=4, depth=depth, boundary="open", longitudinal=longitudinal != 0.0)
    diagonal = diagonal_energies(4, 1.0, longitudinal, "open")
    field = 0.8
    theta = _angles(spec)

    analytic = state_derivatives(theta, spec, diagonal, field)

    shift = 1e-6
    for index in range(spec.n_parameters):
        forward, backward = theta.copy(), theta.copy()
        forward[index] += shift
        backward[index] -= shift
        difference = (
            evolve(forward, spec, diagonal, field) - evolve(backward, spec, diagonal, field)
        ) / (2 * shift)
        assert np.allclose(analytic[index], difference, atol=1e-7)


def test_a_derivative_is_asked_for_every_angle_and_no_more() -> None:
    spec = AnsatzSpec(n_qubits=3, depth=2)
    derivatives = state_derivatives(np.zeros(spec.n_parameters), spec, diagonal_energies(3), 1.0)
    assert derivatives.shape == (spec.n_parameters, spec.state_dimension)


def test_the_wrong_number_of_angles_is_refused_rather_than_broadcast() -> None:
    spec = AnsatzSpec(n_qubits=3, depth=2)
    with pytest.raises(ValueError, match="expected 4 angles"):
        state_derivatives(np.zeros(3), spec, diagonal_energies(3), 1.0)


# --------------------------------------------------------------------------
# The metric, against infidelity between two nearby circuits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pair", [(0, 0), (1, 4), (2, 5), (3, 3)])
def test_the_metric_measures_how_far_the_state_actually_moves(pair: tuple[int, int]) -> None:
    # g is defined as the quadratic form of the infidelity between two nearby states.
    # The module computes it from analytic derivatives; this measures it from two
    # overlaps. Nothing but the definition connects the two routes.
    spec = AnsatzSpec(n_qubits=4, depth=3, boundary="open")
    diagonal = diagonal_energies(4, 1.0, 0.3, "open")
    field = 0.8
    theta = _angles(spec, seed=1)
    state = evolve(theta, spec, diagonal, field)
    metric = fubini_study_metric(state_derivatives(theta, spec, diagonal, field), state)

    nudge = np.zeros(spec.n_parameters)
    for index in pair:
        nudge[index] += 1e-4
    moved = evolve(theta + nudge, spec, diagonal, field)
    infidelity = 1.0 - abs(complex(np.vdot(state, moved))) ** 2

    assert float(nudge @ metric @ nudge) == pytest.approx(infidelity, rel=1e-3)


def test_the_metric_is_symmetric_and_never_says_a_direction_costs_less_than_nothing() -> None:
    # A negative eigenvalue would be a distance shorter than zero, and the solve that
    # follows would return a velocity pointing uphill with nothing to flag it.
    spec = AnsatzSpec(n_qubits=4, depth=3)
    diagonal = diagonal_energies(4)
    theta = _angles(spec, seed=2)
    metric = fubini_study_metric(
        state_derivatives(theta, spec, diagonal, 1.0), evolve(theta, spec, diagonal, 1.0)
    )

    assert np.allclose(metric, metric.T)
    assert np.linalg.eigvalsh(metric).min() > -1e-9


def test_a_flat_metric_makes_the_step_ordinary_gradient_descent() -> None:
    # With g = I the correction does nothing and McLachlan's equation is plain descent
    # at half the gradient. Stating it here fixes the factor of two, which is the part
    # of this method that can be wrong while everything still looks like it works.
    gradient = np.array([2.0, -4.0, 6.0])
    velocity = imaginary_time_velocity(np.eye(3), gradient, regularisation=0.0)
    assert velocity == pytest.approx(-0.5 * gradient)


# --------------------------------------------------------------------------
# The physics, graded against the exact answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_the_energy_never_falls_below_the_true_ground_state(field: float, depth: int) -> None:
    result = evolve_in_imaginary_time(n_sites=DEFAULT_SITES, depth=depth, transverse_field=field)
    assert result.energy >= _exact(DEFAULT_SITES, 1.0, field, "open") - 1e-9


@pytest.mark.parametrize("field", FIELDS)
def test_a_bare_circuit_returns_the_closed_form_energy(field: float) -> None:
    result = evolve_in_imaginary_time(n_sites=DEFAULT_SITES, depth=0, transverse_field=field)
    assert result.energy == pytest.approx(-field * DEFAULT_SITES)
    assert result.stop_reason == "depth_zero"
    assert result.n_metric_evaluations == 0


@pytest.mark.parametrize("field", FIELDS)
def test_the_energy_falls_at_every_accepted_step(field: float) -> None:
    # Exact imaginary time cannot raise the energy, so a rise here would mean the
    # Euler step outran the linearisation and was accepted anyway.
    result = evolve_in_imaginary_time(n_sites=6, depth=3, transverse_field=field)
    assert all(later <= earlier + 1e-12 for earlier, later in pairwise(result.energy_history))


@pytest.mark.parametrize("field", [0.5, 1.0])
def test_it_reaches_the_same_place_as_the_optimiser_on_the_same_circuit(field: float) -> None:
    # The claim that makes the race fair: two entirely different rules for moving the
    # angles -- L-BFGS on the energy, and a differential equation with no search in it
    # -- find the same floor from the same start, so a gap between their curves is
    # about how they got there and not about where they can get to.
    evolved = evolve_in_imaginary_time(n_sites=6, depth=3, transverse_field=field)
    searched = solve(n_sites=6, depth=3, transverse_field=field)
    assert evolved.energy == pytest.approx(searched.energy, abs=1e-4)


def test_where_the_two_rules_disagree_both_are_still_honest_upper_bounds() -> None:
    # They do not always agree, and pinning the case down is worth more than loosening
    # the tolerance above until it disappears. Deep in the disordered phase the two
    # settle in different basins from the same starting angles: imaginary time follows
    # the steepest path through *state* space and L-BFGS follows a quasi-Newton path
    # through *parameter* space, and at h = 2J those are different paths. Neither is
    # wrong -- both are variational upper bounds and neither can dip below the truth --
    # and a comparison that could not show this would not be worth drawing.
    field = 2.0
    evolved = evolve_in_imaginary_time(n_sites=6, depth=3, transverse_field=field)
    searched = solve(n_sites=6, depth=3, transverse_field=field)
    truth = _exact(6, 1.0, field, "open")

    assert evolved.energy != pytest.approx(searched.energy, abs=1e-4)
    assert evolved.energy >= truth - 1e-9
    assert searched.energy >= truth - 1e-9
    # And the disagreement is small enough to be a basin and not a bug: a sign error
    # or a wrong generator would put them orders of magnitude apart, not one part in
    # ten thousand.
    assert abs(evolved.energy - searched.energy) < 1e-2


def test_enough_depth_closes_the_gap_to_the_exact_answer() -> None:
    # Taken over restarts: at depth 8 this landscape has several basins and a single
    # start picks between them on last-bit arithmetic. See RESTART_RAMPS. The bound is
    # 2e-3 rather than the eigensolver's 1e-3 because imaginary time follows the
    # steepest path through state space and stops in a shallower basin more often --
    # the measured worst case over jittered starts is 1.4e-4, so this holds with margin
    # while still being fifty times tighter than the 7.5e-2 a depth-2 circuit reaches.
    energy = _lowest_over_restarts(depth=8, transverse_field=1.0)
    assert energy == pytest.approx(_exact(6, 1.0, 1.0, "open"), abs=2e-3)


def test_a_longitudinal_field_is_carried_into_the_chain_it_solves() -> None:
    # g breaks the free-fermion mapping, so the two chains have different ground
    # states and a run that ignored g would return the wrong one without saying so.
    plain = evolve_in_imaginary_time(n_sites=6, depth=3, longitudinal_field=0.0)
    tilted = evolve_in_imaginary_time(n_sites=6, depth=3, longitudinal_field=0.6)
    assert tilted.energy < plain.energy - 1e-6
    assert tilted.spec.longitudinal is True


# --------------------------------------------------------------------------
# What the result reports about itself
# --------------------------------------------------------------------------


def test_how_it_stopped_is_recorded_rather_than_guessed_from_the_energy() -> None:
    # A run cut off at its limit and a run that converged produce the same float, and
    # a caller handed only the float will report the first as the second.
    capped = evolve_in_imaginary_time(n_sites=6, depth=3, max_steps=2)
    assert capped.stop_reason == "step_limit"
    assert capped.converged is False
    assert capped.n_steps == 2


def test_the_cost_reported_is_larger_than_the_number_of_steps() -> None:
    # Every rejected half-step was still paid for. A cost equal to the step count
    # would be the retries quietly disappearing from the bill.
    result = evolve_in_imaginary_time(n_sites=6, depth=3)
    assert result.n_energy_evaluations > result.n_steps
    assert result.n_metric_evaluations >= 1


def test_the_description_is_flat_data_with_the_history_left_out() -> None:
    described = evolve_in_imaginary_time(n_sites=4, depth=2).describe()
    assert described["stop_reason"] in {"converged", "step_limit", "step_collapsed"}
    assert "energy_history" not in described
    assert described["ansatz"]["n_parameters"] == 4


def test_an_empty_run_reports_no_improvement_rather_than_dividing_by_nothing() -> None:
    blank = ImaginaryTimeResult(
        energy=-1.0,
        parameters=np.zeros(0),
        spec=AnsatzSpec(n_qubits=2, depth=0),
        n_steps=0,
        imaginary_time=0.0,
        n_energy_evaluations=0,
        n_metric_evaluations=0,
        converged=False,
        stop_reason="depth_zero",
    )
    assert blank.improvement == 0.0


@pytest.mark.parametrize("step", [0.0, -0.1])
def test_a_step_that_does_not_advance_time_is_refused(step: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        evolve_in_imaginary_time(n_sites=4, depth=2, step=step)


def test_the_wrong_number_of_starting_angles_is_refused() -> None:
    with pytest.raises(ValueError, match="expected 4 initial angles"):
        evolve_in_imaginary_time(n_sites=4, depth=2, initial_parameters=np.zeros(3))
