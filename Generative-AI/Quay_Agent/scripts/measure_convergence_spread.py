r"""Measure how far a variational run lands from the exact answer, and how far that moves.

Why this exists. ``test_enough_depth_closes_the_gap_to_the_exact_answer`` passed on macOS
and failed on Linux CI, in both solvers, by about 1.5e-3. Neither run is random: the seed
is fixed, the start is an adiabatic ramp, and the tolerances are 1e-10. The difference is
not an under-converged run but a *different local minimum of the same landscape*, chosen
by arithmetic that differs in the last bits between one BLAS implementation and another.

This script measures that. It perturbs the starting angles by an amount far below any
physical scale -- comparable to the disagreement between two linear-algebra libraries --
and reports how far the converged energy moves. A landscape with a single basin ignores
such a perturbation entirely; this one does not, and the numbers here are what set the
tolerances and the restart counts in ``tests/test_variational_eigensolver.py`` and
``tests/test_imaginary_time_evolution.py``.

Run:
    uv run python scripts/measure_convergence_spread.py
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.physics.model import BoundaryCondition, TFIMSpec
from src.physics.quantum.imaginary_time_evolution import evolve_in_imaginary_time
from src.physics.quantum.variational_eigensolver import solve
from src.physics.registry import solver_for

SITES = 6
"""Chain length. Small enough to diagonalise exactly, long enough to have a landscape."""

JITTER = 1e-10
"""Perturbation applied to the ramp time, in units of the ramp time itself.

Chosen to be physically meaningless and numerically comparable to the last-bit
disagreement between two BLAS implementations: if the answer moves, the mover is the
optimiser's path and not the physics.
"""

DRAWS = 8
"""Perturbed starts per configuration. Enough to find the neighbouring basin, not a
statistically converged distribution -- the worst of eight is the number a tolerance has
to survive, and more draws can only make it worse."""

RESTART_RAMPS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
"""The restart schedule the tests use, repeated here so the two can be compared."""

Runner = Callable[[int, float, BoundaryCondition, float], float]
"""A solver call flattened to (depth, field, boundary, ramp time) -> energy."""


def exact(field: float, boundary: BoundaryCondition = "open") -> float:
    """Return the true ground-state energy, from the grader's bench.

    Importing a sealed solver is legitimate here for the same reason it is in the tests:
    the architecture rule bans a *solver* from reaching an exact answer, not a grader.

    Args:
        field: The transverse field h, in units of the coupling J.
        boundary: Open chain or ring.

    Returns:
        The exact ground-state energy of the chain.
    """
    return solver_for("exact_diagonalisation").solve(
        TFIMSpec(n_sites=SITES, coupling=1.0, field=field, boundary=boundary)
    )


def run_variational(depth: int, field: float, boundary: BoundaryCondition, ramp: float) -> float:
    """Return the energy the eigensolver converges to. See :data:`Runner`."""
    return solve(
        n_sites=SITES, depth=depth, transverse_field=field, boundary=boundary, ramp_time=ramp
    ).energy


def run_imaginary_time(depth: int, field: float, boundary: BoundaryCondition, ramp: float) -> float:
    """Return the energy imaginary-time evolution settles at. See :data:`Runner`."""
    return evolve_in_imaginary_time(
        n_sites=SITES, depth=depth, transverse_field=field, boundary=boundary, ramp_time=ramp
    ).energy


SOLVERS: tuple[tuple[str, Runner], ...] = (
    ("variational (L-BFGS)", run_variational),
    ("imaginary time", run_imaginary_time),
)

CASES: tuple[tuple[float, BoundaryCondition], ...] = (
    (0.5, "open"),
    (1.0, "open"),
    (2.0, "open"),
    (1.0, "periodic"),
)


def worst_gap(
    runner: Runner,
    depth: int,
    field: float,
    boundary: BoundaryCondition,
    rng: np.random.Generator,
    *,
    restarts: tuple[float, ...] | None = None,
) -> float:
    """Return the largest gap to the exact answer over :data:`DRAWS` jittered starts.

    Args:
        runner: The solver under test.
        depth: Number of ansatz layers.
        field: The transverse field h.
        boundary: Open chain or ring.
        rng: Source of the jitter, passed in so a whole table reproduces from one seed.
        restarts: Ramp times to restart from, taking the lowest energy of the set. None
            runs a single start, which is what the tests did before this measurement.

    Returns:
        The worst gap seen, in units of the coupling. This is the number a tolerance has
        to clear, because CI's arithmetic is not this machine's and the draw is not ours.
    """
    ramps = restarts if restarts is not None else (2.0,)
    truth = exact(field, boundary)
    gaps = []
    for _ in range(DRAWS):
        jitter = float(rng.normal(0.0, JITTER))
        gaps.append(min(runner(depth, field, boundary, ramp + jitter) for ramp in ramps) - truth)
    return max(gaps)


def main() -> None:
    """Print the single-start spread, then the same table with restarts turned on."""
    for title, restarts in (
        (f"one start, worst of {DRAWS} jittered draws", None),
        (
            f"lowest of {len(RESTART_RAMPS)} restarts, worst of {DRAWS} jittered draws",
            RESTART_RAMPS,
        ),
    ):
        print(f"\n{title}")
        print(f"{'solver':<22}{'h':>5}{'boundary':>10}{'depth 2':>12}{'depth 8':>12}")
        print("-" * 61)
        rng = np.random.default_rng(0)
        for label, runner in SOLVERS:
            for field, boundary in CASES:
                shallow = worst_gap(runner, 2, field, boundary, rng, restarts=restarts)
                deep = worst_gap(runner, 8, field, boundary, rng, restarts=restarts)
                print(f"{label:<22}{field:>5}{boundary:>10}{shallow:>12.3e}{deep:>12.3e}")


if __name__ == "__main__":
    main()
