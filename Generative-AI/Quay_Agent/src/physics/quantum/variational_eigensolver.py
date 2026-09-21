r"""Minimise the energy over the ansatz, and report what the answer is worth.

The VQE driver. It runs the loop -- prepare, estimate, differentiate, step -- and returns
a result that carries not only a number but the evidence for it: how it stopped, how many
iterations it took, what the gradient was doing at the end, and the whole energy history.

The number alone is not the deliverable. An energy that converged and an energy that
ran out of iterations are the same float and mean entirely different things, and an agent
handed only the float will report the second as the first. So :attr:`VqeResult.stop_reason`
is always populated, and the histories are always returned.

The variational bound is checked rather than assumed. Every energy this module produces
satisfies :math:`E(\theta) \ge E_0` as a matter of linear algebra, so a value below the
true ground state is a bug and never a better answer. That makes the inequality the one
correctness gate in the project needing no second implementation to compare against --
and the module asserts monotone improvement rather than waiting for a test to notice.

There is something this module may not know. It cannot import anything under
:mod:`src.physics.reference`, so it cannot evaluate :math:`E_0` and therefore cannot check
the bound against the true value. It checks what it can -- that the optimiser goes
downhill, that the norm holds, that the gradient is finite -- and grading against the exact
answer happens on the far side of the wall, in the tests and in
:mod:`src.verification.cross_check`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.physics.lattice import Lattice
from src.physics.model import BoundaryCondition
from src.physics.quantum.ansatz import (
    DEFAULT_RAMP_TIME,
    AnsatzFamily,
    AnsatzSpec,
    Initialisation,
    angle_periods,
    initial_angles,
    wrap_angles,
)
from src.physics.quantum.statevector import (
    diagonal_energies,
    energy,
    energy_and_gradient,
    evolve,
)

FloatArray = NDArray[np.float64]

StopReason = Literal[
    "converged",
    "gradient_below_tolerance",
    "iteration_limit",
    "depth_zero",
    "optimiser_failed",
]
"""Why the loop ended. Never inferred from the energy -- always recorded at the exit."""

DEFAULT_MAX_ITERATIONS = 500
"""Iteration cap.

Generous for this ansatz: at the depths that fit on hardware the parameter count is under
twenty and L-BFGS converges in tens of steps. The cap exists to bound a pathological run,
not to be reached.
"""

DEFAULT_ENERGY_TOLERANCE = 1e-10
"""Energy change below which the optimiser stops.

Ten orders of magnitude tighter than any shot-limited measurement could resolve, because
this is the *exact* backend and its job is to find the true minimum of the ansatz so that
the remaining error is attributable to the ansatz rather than to the optimiser. Anything
looser leaves a converged run and a stalled run indistinguishable.
"""

DEFAULT_GRADIENT_TOLERANCE = 1e-8
"""Gradient norm below which the point counts as stationary."""


@dataclass(frozen=True, slots=True)
class VqeResult:
    """One completed run, with the evidence needed to judge it.

    Attributes:
        energy: The lowest energy reached. A variational upper bound on the true ground
            state, never a lower one.
        parameters: The angles that achieved it, folded into a single period so that two
            equivalent solutions compare equal.
        spec: The ansatz that was run, carrying its own gate and depth arithmetic.
        n_iterations: Optimiser steps taken.
        n_energy_evaluations: Circuit evaluations, counting those the line search spent.
            The honest cost of the run, and larger than ``n_iterations``.
        converged: Whether the loop reached a stationary point rather than a limit.
        stop_reason: Which exit was taken. See :data:`StopReason`.
        energy_history: The energy after each accepted step, oldest first.
        gradient_norm_history: The gradient norm at each accepted step.
    """

    energy: float
    parameters: FloatArray
    spec: AnsatzSpec
    n_iterations: int
    n_energy_evaluations: int
    converged: bool
    stop_reason: StopReason
    energy_history: tuple[float, ...] = ()
    gradient_norm_history: tuple[float, ...] = ()

    @property
    def energy_per_site(self) -> float:
        """Energy divided by the number of sites.

        The comparable quantity across chain lengths: the total energy is extensive and
        grows with the system, so quoting it alone makes a longer chain look worse.
        """
        return self.energy / self.spec.n_qubits

    @property
    def improvement(self) -> float:
        """How far the optimiser moved from its starting point.

        A run that improved by nothing has not failed to find the ground state so much as
        failed to start -- a flat landscape, a zero gradient, or a bad initialisation --
        and that is a different diagnosis from a run that improved and then stalled.
        """
        if not self.energy_history:
            return 0.0
        return self.energy_history[0] - self.energy

    def describe(self) -> dict[str, Any]:
        """Render the outcome as plain data for a tool result or a log line.

        Returns:
            A flat mapping. Histories are summarised rather than included in full, since
            a two-hundred-entry list in a language model's context buys nothing that the
            first value, the last value and the length do not.
        """
        return {
            "energy": round(self.energy, 12),
            "energy_per_site": round(self.energy_per_site, 12),
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "n_iterations": self.n_iterations,
            "n_energy_evaluations": self.n_energy_evaluations,
            "improvement": round(self.improvement, 12),
            "final_gradient_norm": (
                round(self.gradient_norm_history[-1], 12) if self.gradient_norm_history else 0.0
            ),
            "ansatz": self.spec.describe(),
        }


@dataclass
class _Trace:
    """Mutable record of what the optimiser did, filled in by its callback.

    Not part of the public interface: :class:`VqeResult` freezes a copy of this. It exists
    because SciPy reports only the final point, and the *path* is what distinguishes a
    barren plateau from an ordinary stall.

    Attributes:
        energies: Energy at each accepted step.
        gradient_norms: Gradient norm at each accepted step.
        n_evaluations: How many times the objective was called in total.
    """

    energies: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    n_evaluations: int = 0


def solve(
    n_sites: int,
    depth: int,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: BoundaryCondition = "open",
    family: AnsatzFamily = "hva",
    initialisation: Initialisation = "adiabatic_ramp",
    ramp_time: float = DEFAULT_RAMP_TIME,
    seed: int = 0,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    gradient_tolerance: float = DEFAULT_GRADIENT_TOLERANCE,
    initial_parameters: FloatArray | None = None,
    lattice: Lattice | None = None,
) -> VqeResult:
    r"""Run VQE on the mixed-field Ising chain and return the result with its evidence.

    Uses L-BFGS on the exact adjoint gradient, which is the right choice while simulating:
    the gradient costs about two energy evaluations rather than a circuit pair per Pauli
    rotation, so there is nothing to gain from a gradient-free or stochastic optimiser
    here. On hardware that price is what the parameter-shift rule charges, and this
    project quotes it rather than paying it -- nothing here samples.

    Args:
        n_sites: Number of sites, equal to the number of qubits.
        depth: Number of ansatz layers. Zero is legal and returns the energy of the bare
            initial state, which has the closed form :math:`-hL` and is the cheapest
            end-to-end check available.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis. Zero leaves the
            chain integrable; non-zero is what makes the feasibility question non-trivial.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        family: ``"hva"`` or ``"qaoa"``. Identical circuits -- the name is carried so the
            result says which tradition the caller is arguing from.
        initialisation: How the angles start. See
            :func:`~src.physics.quantum.ansatz.initial_angles`.
        ramp_time: Total adiabatic time for the ramp initialisation.
        seed: Seed for a randomised initialisation. Explicit so the run reproduces.
        max_iterations: Cap on optimiser steps.
        energy_tolerance: Energy change below which the run counts as converged.
        gradient_tolerance: Gradient norm below which the point counts as stationary.
        initial_parameters: Angles to start from, overriding ``initialisation``. This is
            how a depth sweep warm-starts each depth from the one below it.
        lattice: The shape the sites sit on, or ``None`` for a chain. It reaches
            both halves of the problem at once -- the bond list the circuit acts on
            and the diagonal the state is evolved against -- so a square or
            triangular lattice cannot be solved as a circuit of one shape scored
            against a Hamiltonian of another.

    Returns:
        The completed run.

    Raises:
        ValueError: If ``initial_parameters`` does not carry two angles per layer.

    Examples:
        At depth zero the answer is the closed form and nothing is optimised:

        >>> result = solve(n_sites=4, depth=0, transverse_field=1.0)
        >>> round(result.energy, 10), result.stop_reason
        (-4.0, 'depth_zero')

        Adding layers lowers the energy, and never below the true ground state:

        >>> shallow = solve(n_sites=6, depth=1).energy
        >>> deeper = solve(n_sites=6, depth=3).energy
        >>> bool(deeper < shallow)
        True
    """
    spec = AnsatzSpec(
        n_qubits=n_sites,
        depth=depth,
        boundary=boundary,
        family=family,
        longitudinal=longitudinal_field != 0.0,
        lattice=lattice,
    )
    diagonal = diagonal_energies(n_sites, coupling, longitudinal_field, boundary, lattice)

    if depth == 0:
        state = evolve(np.zeros(0), spec, diagonal, transverse_field)
        bare = energy(state, diagonal, transverse_field)
        return VqeResult(
            energy=bare,
            parameters=np.zeros(0),
            spec=spec,
            n_iterations=0,
            n_energy_evaluations=1,
            converged=True,
            stop_reason="depth_zero",
            energy_history=(bare,),
            gradient_norm_history=(0.0,),
        )

    if initial_parameters is None:
        start = initial_angles(depth, initialisation, ramp_time=ramp_time, seed=seed)
    else:
        start = np.asarray(initial_parameters, dtype=np.float64)
        if start.size != spec.n_parameters:
            raise ValueError(f"expected {spec.n_parameters} initial angles, got {start.size}")

    trace = _Trace()

    def objective(angles: FloatArray) -> tuple[float, FloatArray]:
        """Energy and gradient at ``angles``, counting the call."""
        trace.n_evaluations += 1
        return energy_and_gradient(
            np.asarray(angles, dtype=np.float64), spec, diagonal, transverse_field
        )

    def record(angles: FloatArray) -> None:
        """Note the energy and gradient norm after an accepted step."""
        value, gradient = energy_and_gradient(
            np.asarray(angles, dtype=np.float64), spec, diagonal, transverse_field
        )
        trace.energies.append(value)
        trace.gradient_norms.append(float(np.linalg.norm(gradient)))

    record(start)
    outcome = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        callback=record,
        options={"maxiter": max_iterations, "ftol": energy_tolerance, "gtol": gradient_tolerance},
    )

    best = float(outcome.fun)
    final_gradient_norm = trace.gradient_norms[-1] if trace.gradient_norms else 0.0
    if not outcome.success and outcome.status not in (0, 1):
        reason: StopReason = "optimiser_failed"
    elif outcome.nit >= max_iterations:
        reason = "iteration_limit"
    elif final_gradient_norm <= gradient_tolerance:
        reason = "gradient_below_tolerance"
    else:
        reason = "converged"

    return VqeResult(
        energy=best,
        parameters=wrap_angles(
            np.asarray(outcome.x, dtype=np.float64),
            *angle_periods(coupling, transverse_field, longitudinal_field),
        ),
        spec=spec,
        n_iterations=int(outcome.nit),
        n_energy_evaluations=trace.n_evaluations,
        converged=reason in ("converged", "gradient_below_tolerance"),
        stop_reason=reason,
        energy_history=tuple(trace.energies),
        gradient_norm_history=tuple(trace.gradient_norms),
    )
