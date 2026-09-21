r"""Run the alternating ansatz at growing depth, and report how fast it approaches.

The QAOA driver. The circuit it runs is the one
:mod:`src.physics.quantum.variational_eigensolver` runs -- identical gates, identical
parameter count -- so this module is not a second algorithm. What it adds is the two
things QAOA has that a generic variational eigensolver does not, and both are about
*where the optimiser starts* rather than about what it optimises:

1. A principled initialisation. The angles are read off a Trotterised adiabatic path,
   which fixes the whole :math:`2p`-parameter schedule from the single time scale
   :math:`T`. Scanning that one number is cheap and it matters: at :math:`L = 6`,
   :math:`p = 1`, a ramp time of 2 converges to :math:`-2.60` while a ramp time of 1
   converges to :math:`-7.06` on the same Hamiltonian. The optimiser is not at fault --
   the basin is.
2. A warm start up the depth ladder. Having converged at depth :math:`p`, the schedule
   is stretched onto :math:`p+1` rather than started again, so a sweep costs roughly one
   optimisation instead of :math:`p` of them and each depth inherits a good basin.

The output is a curve rather than a number. :attr:`DepthSweep.energies` is the deliverable:
monotone decreasing in :math:`p`, approaching the true ground state from above, and its
*rate* of approach is the actual result. A single energy at a single depth says nothing
about whether more depth would have helped, which is the only question a feasibility
verdict turns on.

Monotonicity is enforced rather than hoped for. A deeper ansatz contains every shallower one
-- pad the extra layer with zero angles and the circuit is unchanged -- so
:math:`E(p+1) \le E(p)` must hold. When the optimiser lands somewhere worse, that is a
local minimum rather than a limit of the family, and the sweep keeps the better answer and
records that it had to. See :attr:`DepthSweep.regressions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.physics.lattice import Lattice
from src.physics.model import BoundaryCondition
from src.physics.quantum.ansatz import interpolate_to_depth
from src.physics.quantum.variational_eigensolver import VqeResult, solve

FloatArray = NDArray[np.float64]

RAMP_TIME_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
"""Ramp times the coarse scan tries.

Logarithmic because the failure it is guarding against is order-of-magnitude -- a ramp far
too fast leaves the state near where it started, one far too slow overshoots into a
different basin -- and six points spanning a factor of thirty-two find the right decade at
a cost of six optimisations of a two-parameter problem.
"""


@dataclass(frozen=True, slots=True)
class DepthSweep:
    """One ansatz run at every depth from one up to a maximum.

    Attributes:
        depths: The depths tried, ascending.
        energies: The best energy at each depth, in the same order. Monotone
            non-increasing by construction -- see :attr:`regressions`.
        results: The full run at each depth, carrying its own stop reason and history.
        ramp_time: The ramp time the scan selected for the first depth, and therefore the
            basin the whole sweep inherited.
        regressions: Depths at which the optimiser returned something worse than the depth
            below, before the monotone floor was applied. Empty is the expected case;
            a non-empty tuple is a local-minimum report, and worth surfacing rather than
            smoothing away.
    """

    depths: tuple[int, ...]
    energies: tuple[float, ...]
    results: tuple[VqeResult, ...]
    ramp_time: float
    regressions: tuple[int, ...] = ()

    @property
    def best(self) -> VqeResult:
        """The run that reached the lowest energy."""
        return self.results[int(np.argmin(self.energies))]

    def marginal_gain(self) -> tuple[float, ...]:
        """How much each added layer bought, in energy.

        The quantity a depth decision actually turns on. A gain that has fallen to the
        size of the shot noise means the next layer is not worth its two-qubit depth, and
        that is a claim about the budget rather than about the physics.

        Returns:
            One entry per depth after the first: the drop from the previous depth.
        """
        return tuple(
            self.energies[index - 1] - self.energies[index]
            for index in range(1, len(self.energies))
        )

    def describe(self) -> dict[str, Any]:
        """Render the sweep as plain data for a tool result or a log line.

        Returns:
            A flat mapping carrying the curve, the per-layer gains, and the cost of the
            deepest circuit -- everything a depth decision needs and nothing that would
            need a custom encoder to serialise.
        """
        best = self.best
        return {
            "depths": list(self.depths),
            "energies": [round(value, 12) for value in self.energies],
            "energies_per_site": [round(value / best.spec.n_qubits, 12) for value in self.energies],
            "marginal_gain": [round(value, 12) for value in self.marginal_gain()],
            "ramp_time": self.ramp_time,
            "best_depth": best.spec.depth,
            "best_energy": round(best.energy, 12),
            "regressions": list(self.regressions),
            "deepest_circuit": self.results[-1].spec.describe(),
        }


def scan_ramp_time(
    n_sites: int,
    depth: int = 1,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: BoundaryCondition = "open",
    grid: tuple[float, ...] = RAMP_TIME_GRID,
    lattice: Lattice | None = None,
) -> tuple[float, VqeResult]:
    r"""Find the adiabatic time scale that starts the optimiser in the right basin.

    The one-parameter search that replaces a :math:`2p`-parameter one. Because the ramp
    fixes every angle from :math:`T` alone, scanning :math:`T` explores the family of
    *schedules* rather than the space of angle vectors, and it is the difference between
    converging and converging somewhere useless.

    Args:
        n_sites: Number of sites.
        depth: Depth to scan at. One by default: the basin is chosen at the shallowest
            depth, where each optimisation is cheapest, and inherited upward by the warm
            start.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        grid: Ramp times to try.
        lattice: The shape the spins sit on, or ``None`` for a chain. The ramp is
            chosen for the problem it will be used on, so a basin found on a chain
            and reused on a lattice would be a warm start into the wrong basin.

    Returns:
        The winning ramp time and the run it produced.

    Raises:
        ValueError: If the grid is empty, which would leave nothing to choose between.

    Examples:
        >>> best_time, result = scan_ramp_time(n_sites=4, depth=1)
        >>> bool(result.energy < 0.0)
        True
    """
    if not grid:
        raise ValueError("the ramp-time grid cannot be empty")
    runs = [
        (
            candidate,
            solve(
                n_sites=n_sites,
                depth=depth,
                coupling=coupling,
                transverse_field=transverse_field,
                longitudinal_field=longitudinal_field,
                boundary=boundary,
                family="qaoa",
                initialisation="adiabatic_ramp",
                ramp_time=candidate,
                lattice=lattice,
            ),
        )
        for candidate in grid
    ]
    return min(runs, key=lambda pair: pair[1].energy)


def depth_sweep(
    n_sites: int,
    max_depth: int,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: BoundaryCondition = "open",
    grid: tuple[float, ...] = RAMP_TIME_GRID,
    lattice: Lattice | None = None,
) -> DepthSweep:
    r"""Run every depth from one to ``max_depth``, warm-starting each from the last.

    The INTERP schedule. Depth one is found by scanning the ramp time; every depth after
    it starts from the previous depth's converged schedule stretched onto the finer grid.
    Each optimisation therefore begins in a basin that is already good, which is why the
    whole sweep costs about as much as one cold-started optimisation at the deepest level.

    Monotonicity is enforced at the end. A deeper ansatz strictly contains every shallower
    one, so a depth whose optimiser returned something worse has found a local minimum,
    not a limit of the family; the better energy is kept and the depth is recorded in
    :attr:`DepthSweep.regressions` rather than being quietly hidden.

    Args:
        n_sites: Number of sites.
        max_depth: Deepest ansatz to run. Must be at least one.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        grid: Ramp times for the depth-one scan.
        lattice: The shape the spins sit on, or ``None`` for a chain. Carried into
            every depth, so the warm start each depth inherits was found on the same
            problem it is being reused on.

    Returns:
        The completed sweep.

    Raises:
        ValueError: If ``max_depth`` is below one.

    Examples:
        >>> sweep = depth_sweep(n_sites=6, max_depth=3)
        >>> list(sweep.depths)
        [1, 2, 3]
        >>> bool(sweep.energies[2] <= sweep.energies[0])
        True
    """
    if max_depth < 1:
        raise ValueError(f"a sweep needs at least one layer, got {max_depth}")

    ramp_time, first = scan_ramp_time(
        n_sites=n_sites,
        depth=1,
        coupling=coupling,
        transverse_field=transverse_field,
        longitudinal_field=longitudinal_field,
        boundary=boundary,
        grid=grid,
        lattice=lattice,
    )
    results = [first]
    for depth in range(2, max_depth + 1):
        warm_start = interpolate_to_depth(results[-1].parameters, depth)
        results.append(
            solve(
                n_sites=n_sites,
                depth=depth,
                coupling=coupling,
                transverse_field=transverse_field,
                longitudinal_field=longitudinal_field,
                boundary=boundary,
                family="qaoa",
                initial_parameters=warm_start,
                lattice=lattice,
            )
        )

    raw = [result.energy for result in results]
    regressions = tuple(index + 1 for index in range(1, len(raw)) if raw[index] > raw[index - 1])
    monotone = tuple(np.minimum.accumulate(raw).tolist())

    return DepthSweep(
        depths=tuple(range(1, max_depth + 1)),
        energies=monotone,
        results=tuple(results),
        ramp_time=ramp_time,
        regressions=regressions,
    )
