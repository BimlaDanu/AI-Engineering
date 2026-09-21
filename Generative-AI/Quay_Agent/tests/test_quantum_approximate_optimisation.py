"""Tests for the QAOA driver -- the depth sweep and the initialisation that makes it work.

The circuit is the one ``tests/test_variational_eigensolver.py`` already exercises, so
nothing here re-checks the gates. What is under test is the part QAOA adds: that scanning
the ramp time finds a better basin than any fixed choice, that the warm start climbs the
depth ladder without falling out of it, and that the reported curve is the monotone one
the family guarantees.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from src.physics.model import DEFAULT_SITES, TFIMSpec
from src.physics.quantum.quantum_approximate_optimisation import (
    RAMP_TIME_GRID,
    depth_sweep,
    scan_ramp_time,
)
from src.physics.quantum.variational_eigensolver import solve
from src.physics.registry import solver_for


def _exact(n_sites: int) -> float:
    """The true ground-state energy of the default open chain."""
    return solver_for("exact_diagonalisation").solve(
        TFIMSpec(n_sites=n_sites, coupling=1.0, field=1.0, boundary="open")
    )


def test_the_scan_never_returns_worse_than_any_ramp_time_it_tried() -> None:
    best_time, best = scan_ramp_time(n_sites=DEFAULT_SITES, depth=1)
    for candidate in RAMP_TIME_GRID:
        assert (
            best.energy <= solve(n_sites=DEFAULT_SITES, depth=1, ramp_time=candidate).energy + 1e-12
        )
    assert best_time in RAMP_TIME_GRID


def test_the_scan_is_worth_running_because_a_fixed_ramp_time_can_be_badly_wrong() -> None:
    # The scan is not a formality. On this chain a ramp time of 2 converges to about
    # -2.6 while the scan finds about -7.1 -- same optimiser, same circuit, different
    # basin. If this ever stops being true the scan is free to become a constant, but
    # until then it is load-bearing.
    _, best = scan_ramp_time(n_sites=DEFAULT_SITES, depth=1)
    unlucky = solve(n_sites=DEFAULT_SITES, depth=1, ramp_time=2.0)
    assert best.energy < unlucky.energy


def test_an_empty_ramp_time_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        scan_ramp_time(n_sites=4, grid=())


def test_the_sweep_covers_every_depth_up_to_the_maximum() -> None:
    sweep = depth_sweep(n_sites=DEFAULT_SITES, max_depth=4)
    assert list(sweep.depths) == [1, 2, 3, 4]
    assert len(sweep.energies) == len(sweep.results) == 4


def test_the_reported_curve_is_monotone() -> None:
    # A deeper ansatz strictly contains every shallower one -- pad the extra layer with
    # zero angles and the circuit is unchanged -- so a rising curve would be reporting a
    # local minimum as a limit of the family.
    energies = depth_sweep(n_sites=DEFAULT_SITES, max_depth=5).energies
    assert all(later <= earlier + 1e-12 for earlier, later in pairwise(energies))


def test_every_point_on_the_curve_stays_above_the_exact_answer() -> None:
    exact = _exact(DEFAULT_SITES)
    for value in depth_sweep(n_sites=DEFAULT_SITES, max_depth=5).energies:
        assert value >= exact - 1e-9


def test_the_deepest_point_is_close_to_the_exact_answer() -> None:
    sweep = depth_sweep(n_sites=DEFAULT_SITES, max_depth=6)
    assert sweep.energies[-1] == pytest.approx(_exact(DEFAULT_SITES), abs=1e-3)


def test_the_warm_start_beats_a_cold_start_at_the_same_depth() -> None:
    # The reason INTERP exists: the same optimiser at the same depth, differing only in
    # where it began. A warm start that lost to a cold one would mean the interpolation
    # is discarding the basin it was supposed to inherit.
    warm = depth_sweep(n_sites=DEFAULT_SITES, max_depth=4).energies[-1]
    cold = solve(n_sites=DEFAULT_SITES, depth=4, ramp_time=2.0).energy
    assert warm <= cold + 1e-9


def test_the_marginal_gain_is_the_difference_between_neighbouring_depths() -> None:
    sweep = depth_sweep(n_sites=DEFAULT_SITES, max_depth=4)
    gains = sweep.marginal_gain()
    assert len(gains) == 3
    assert all(gain >= -1e-12 for gain in gains)
    assert gains[0] == pytest.approx(sweep.energies[0] - sweep.energies[1])


def test_the_gain_from_each_extra_layer_gets_smaller() -> None:
    # The shape a depth decision turns on: once the gain has fallen below what a shot
    # budget could resolve, the next layer is not worth its two-qubit depth.
    gains = depth_sweep(n_sites=DEFAULT_SITES, max_depth=6).marginal_gain()
    assert gains[-1] < gains[0]


def test_the_best_run_is_the_one_that_reached_the_lowest_energy() -> None:
    sweep = depth_sweep(n_sites=DEFAULT_SITES, max_depth=4)
    assert sweep.best.energy == min(result.energy for result in sweep.results)


def test_the_selected_ramp_time_is_reported() -> None:
    sweep = depth_sweep(n_sites=DEFAULT_SITES, max_depth=3)
    assert sweep.ramp_time in RAMP_TIME_GRID


def test_a_local_minimum_is_recorded_rather_than_smoothed_away() -> None:
    # Monotonicity is enforced on the reported curve, but the fact that it had to be
    # enforced is itself a finding, so the depths where it bit are carried out.
    sweep = depth_sweep(n_sites=DEFAULT_SITES, max_depth=5)
    assert all(depth in sweep.depths for depth in sweep.regressions)


def test_a_sweep_of_no_layers_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        depth_sweep(n_sites=4, max_depth=0)


def test_describe_carries_the_curve_and_the_cost_of_the_deepest_circuit() -> None:
    described = depth_sweep(n_sites=DEFAULT_SITES, max_depth=3).describe()
    assert described["depths"] == [1, 2, 3]
    assert len(described["energies"]) == 3
    assert described["deepest_circuit"]["depth"] == 3
    assert described["deepest_circuit"]["two_qubit_depth"] == 12
