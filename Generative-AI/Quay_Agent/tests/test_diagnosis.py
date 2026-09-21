"""The deterministic run diagnosis: does it call each failure mode correctly?

Two kinds of test live here and they are doing different jobs.

The bound tests are physics. ``coefficient_bound`` is checked against arithmetic
written out by hand in the test, and then against the sealed exact solvers, which
share no algebra with it at all -- one sums Pauli coefficients, one diagonalises a
sparse matrix, one solves free fermions in momentum space. A bound that survives
all three is a bound.

The signal tests are software. Each failure mode is fed a run built to show
exactly that symptom and nothing else, so a diagnosis that fires on the wrong
evidence fails here rather than in a report.

Using the sealed solvers is legitimate in a test: the architecture seal stops the
*agent* from reaching an exact answer, and this file is the grader, not the agent.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agent.diagnosis import (
    BOUND_SLACK,
    DESCENT_PER_STEP,
    PLATEAU_GRADIENT_NORM,
    coefficient_bound,
    diagnose_run,
    diagnose_sweep,
)
from src.physics.model import TFIMSpec
from src.physics.quantum.ansatz import AnsatzSpec
from src.physics.quantum.quantum_approximate_optimisation import DepthSweep
from src.physics.quantum.variational_eigensolver import StopReason, VqeResult, solve
from src.physics.registry import solver_for


def make_result(
    *,
    energy: float = -5.0,
    stop_reason: StopReason = "converged",
    n_iterations: int = 10,
    energy_history: tuple[float, ...] = (-1.0, -5.0),
    gradient_norm_history: tuple[float, ...] = (1.0, 1e-9),
    n_qubits: int = 6,
    depth: int = 2,
) -> VqeResult:
    """Build a run with one symptom deliberately present and the rest neutral."""
    return VqeResult(
        energy=energy,
        parameters=np.zeros(2 * depth),
        spec=AnsatzSpec(n_qubits=n_qubits, depth=depth),
        n_iterations=n_iterations,
        n_energy_evaluations=n_iterations * 2,
        converged=stop_reason in {"converged", "gradient_below_tolerance"},
        stop_reason=stop_reason,
        energy_history=energy_history,
        gradient_norm_history=gradient_norm_history,
    )


# ── the bound, checked three ways ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("n_sites", "coupling", "transverse_field", "longitudinal_field", "boundary", "expected"),
    [
        (4, 1.0, 1.0, 0.0, "open", -7.0),  # 3 bonds + 4 field terms
        (4, 1.0, 1.0, 0.0, "periodic", -8.0),  # a ring has one more bond
        (6, 2.0, 0.5, 0.0, "open", -13.0),  # 5*2 + 6*0.5
        (6, 1.0, 1.0, 0.25, "open", -12.5),  # the longitudinal field adds g*L
    ],
)
def test_bound_matches_hand_arithmetic(
    n_sites: int,
    coupling: float,
    transverse_field: float,
    longitudinal_field: float,
    boundary: str,
    expected: float,
) -> None:
    bound = coefficient_bound(
        n_sites=n_sites,
        coupling=coupling,
        transverse_field=transverse_field,
        longitudinal_field=longitudinal_field,
        boundary=boundary,  # type: ignore[arg-type]
    )
    assert bound == pytest.approx(expected)


@pytest.mark.parametrize("boundary", ["open", "periodic"])
@pytest.mark.parametrize("n_sites", [4, 6, 8])
@pytest.mark.parametrize("field", [0.25, 1.0, 3.0])
def test_bound_lies_below_the_exact_ground_state(n_sites: int, field: float, boundary: str) -> None:
    """The whole point of the bound: nothing real may sit under it.

    Graded against every sealed solver that accepts the spec. The two reach the same
    energy by unrelated routes -- one diagonalises a sparse matrix in the spin basis,
    one solves free fermions in momentum space -- and the Pfeuty solution only covers
    the ring, so each is asked rather than assumed. If the bound were ever above a
    true ground state it would condemn a correct run as a bug, which is the one way
    this module could do real harm.
    """
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=field, boundary=boundary)  # type: ignore[arg-type]
    bound = coefficient_bound(
        n_sites=n_sites,
        coupling=1.0,
        transverse_field=field,
        boundary=boundary,  # type: ignore[arg-type]
    )

    graded_by = 0
    for method in ("pfeuty_exact", "exact_diagonalisation"):
        facts = solver_for(method)
        if facts.unsupported_reason(spec) is not None:
            continue
        assert facts.ground_state_energy is not None
        exact = facts.ground_state_energy(spec)
        assert bound < exact, f"{method}: bound {bound} is not below exact {exact}"
        graded_by += 1

    assert graded_by, f"no sealed solver could grade {spec.label()}"


def test_bound_is_loose_and_that_is_expected() -> None:
    """A reader should not mistake the bound for an estimate of the answer.

    At the critical point the true energy sits well above it, because the Ising
    terms and the transverse field do not commute and cannot both be minimised.
    Pinned so nobody later "improves" the bound into something it cannot be.
    """
    spec = TFIMSpec(n_sites=8, coupling=1.0, field=1.0, boundary="periodic")
    facts = solver_for("pfeuty_exact")
    assert facts.ground_state_energy is not None

    bound = coefficient_bound(n_sites=8, coupling=1.0, transverse_field=1.0, boundary="periodic")
    exact = facts.ground_state_energy(spec)
    assert exact - bound > 1.0


# ── the signals ────────────────────────────────────────────────────────────────


def test_a_real_converged_run_is_healthy() -> None:
    result = solve(n_sites=6, depth=3, boundary="open")
    bound = coefficient_bound(n_sites=6, boundary="open")

    diagnosis = diagnose_run(result, bound)

    assert diagnosis.signal == "healthy"
    assert result.energy > bound


def test_energy_below_the_bound_is_reported_as_a_bug() -> None:
    bound = coefficient_bound(n_sites=6, boundary="open")
    result = make_result(energy=bound - 1.0)

    diagnosis = diagnose_run(result, bound)

    assert diagnosis.signal == "below_variational_bound"
    assert not diagnosis.is_actionable


def test_rounding_below_the_bound_is_not_called_a_bug() -> None:
    """Floating-point slack must not trip the falsifier, or it cries wolf."""
    bound = coefficient_bound(n_sites=6, boundary="open")
    result = make_result(energy=bound - BOUND_SLACK / 10)

    assert diagnose_run(result, bound).signal != "below_variational_bound"


def test_the_bound_check_outranks_the_stop_reason() -> None:
    """An impossible energy is a bug whatever else the run reports."""
    bound = coefficient_bound(n_sites=6, boundary="open")
    result = make_result(
        energy=bound - 1.0,
        stop_reason="iteration_limit",
        energy_history=(-1.0, -50.0),
    )

    assert diagnose_run(result, bound).signal == "below_variational_bound"


def test_hitting_the_cap_while_still_falling_is_still_descending() -> None:
    result = make_result(
        stop_reason="iteration_limit",
        energy_history=(-1.0, -3.0, -5.0),
        gradient_norm_history=(1.0, 0.5, 0.2),
    )

    diagnosis = diagnose_run(result, -100.0)

    assert diagnosis.signal == "still_descending"
    assert diagnosis.is_actionable


def test_hitting_the_cap_after_flattening_is_not_still_descending() -> None:
    """The last step is what decides it, not the fact that the cap was reached."""
    result = make_result(
        stop_reason="iteration_limit",
        energy_history=(-1.0, -5.0, -5.0 - DESCENT_PER_STEP / 100),
        gradient_norm_history=(1.0, 0.5, 0.2),
    )

    assert diagnose_run(result, -100.0).signal != "still_descending"


def test_a_flat_start_that_never_moves_is_a_plateau() -> None:
    result = make_result(
        energy=-5.0,
        energy_history=(-5.0, -5.0),
        gradient_norm_history=(PLATEAU_GRADIENT_NORM / 100, PLATEAU_GRADIENT_NORM / 100),
    )

    diagnosis = diagnose_run(result, -100.0)

    assert diagnosis.signal == "barren_plateau"
    assert "flat" in diagnosis.repair


def test_a_warm_start_that_was_already_good_is_not_a_plateau() -> None:
    """A small opening gradient plus real improvement is a good start, not a trap.

    This is the mistake worth guarding: the depth sweep warm-starts every depth, so
    its best runs all begin with small gradients. Diagnosing those as plateaus would
    condemn precisely the runs the project relies on.
    """
    result = make_result(
        energy=-9.0,
        energy_history=(-8.0, -9.0),
        gradient_norm_history=(PLATEAU_GRADIENT_NORM / 100, 1e-12),
    )

    assert diagnose_run(result, -100.0).signal == "healthy"


def test_depth_zero_reports_that_nothing_ran() -> None:
    result = make_result(stop_reason="depth_zero", depth=1, energy_history=())

    assert diagnose_run(result, -100.0).signal == "never_started"


def test_every_diagnosis_carries_evidence_and_a_repair() -> None:
    """No branch may return an empty string: the report quotes both fields."""
    bound = coefficient_bound(n_sites=6, boundary="open")
    runs = [
        make_result(energy=bound - 1.0),
        make_result(stop_reason="depth_zero", energy_history=()),
        make_result(stop_reason="iteration_limit", energy_history=(-1.0, -3.0, -5.0)),
        make_result(energy_history=(-5.0, -5.0), gradient_norm_history=(1e-12, 1e-12)),
        make_result(stop_reason="optimiser_failed"),
        make_result(),
    ]

    for run in runs:
        diagnosis = diagnose_run(run, bound)
        assert diagnosis.evidence.strip()
        assert diagnosis.repair.strip()
        assert diagnosis.describe()["signal"] == diagnosis.signal


# ── the sweep, which sees what no single run can ───────────────────────────────


def test_a_regression_in_the_sweep_proves_a_local_minimum() -> None:
    """A deeper circuit contains the shallower one, so worse is proof, not noise."""
    results = (make_result(energy=-5.0, depth=1), make_result(energy=-4.0, depth=2))
    sweep = DepthSweep(
        depths=(1, 2),
        energies=(-5.0, -5.0),
        results=results,
        ramp_time=1.0,
        regressions=(2,),
    )

    diagnosis = diagnose_sweep(sweep, -100.0)

    assert diagnosis.signal == "local_minimum"
    assert "2" in diagnosis.evidence


def test_a_clean_sweep_is_judged_on_its_best_run() -> None:
    results = (make_result(energy=-5.0, depth=1), make_result(energy=-9.0, depth=2))
    sweep = DepthSweep(
        depths=(1, 2),
        energies=(-5.0, -9.0),
        results=results,
        ramp_time=1.0,
    )

    assert diagnose_sweep(sweep, -100.0).signal == "healthy"
