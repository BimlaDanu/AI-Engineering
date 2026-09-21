r"""Imaginary-time evolution of the circuit\'s own angles -- VarQITE.

The one method here that does not search. Replace :math:`t` by :math:`-i\tau` in
the Schroedinger equation and every component of the state decays at a rate set
by its own energy, leaving the lowest standing. No landscape, no optimiser.

:math:`e^{-\tau\hat H}` is not unitary, so McLachlan\'s formulation keeps the
state inside the ansatz and asks which change of angles best imitates the decay.
That is a linear system at every step:

.. math::

    \sum_j g_{kj}\,\dot\theta_j \;=\; -\,C_k ,
    \qquad
    C_k = \tfrac{1}{2}\,\frac{\partial E}{\partial \theta_k} ,

with :math:`g` the Fubini-Study metric, giving
:math:`\dot\theta = -\tfrac{1}{2} g^{-1} \nabla E` -- gradient descent corrected
for equal steps in the angles not being equal steps in the state.

The comparison against :mod:`src.physics.quantum.variational_eigensolver` is fair
because only the rule that moves the angles differs. It descends with no line
search; the metric costs :math:`P^2` overlaps per step, cheap on a simulator and
many extra circuits on a device; and it is not immune to the ansatz\'s ceiling,
converging just as confidently to the same wrong answer.

:mod:`src.physics.classical.variational_imaginary_time` is the classical
baseline, not this. Both are named for what they do, since ``vita.py`` and
``varqite.py`` side by side would be indistinguishable under pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.physics.lattice import Lattice
from src.physics.model import BoundaryCondition
from src.physics.quantum.ansatz import (
    DEFAULT_RAMP_TIME,
    AnsatzFamily,
    AnsatzSpec,
    Initialisation,
    angle_periods,
    initial_angles,
    split_angles,
    wrap_angles,
)
from src.physics.quantum.statevector import (
    apply_diagonal_layer,
    apply_field_layer,
    diagonal_energies,
    energy,
    energy_and_gradient,
    evolve,
    sigma_x_sum,
    uniform_superposition,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

ImaginaryTimeStop = Literal[
    "converged",
    "step_limit",
    "depth_zero",
    "step_collapsed",
]
"""Why the evolution ended. Recorded at the exit, never inferred from the energy."""

DEFAULT_STEP = 0.05
r"""The imaginary-time increment :math:`\Delta\tau` each step advances by.

Small enough that the linearisation McLachlan's condition rests on holds for this
ansatz at the depths that fit on hardware, and large enough that a run reaches the
bottom in tens of steps rather than thousands. A step that turns out to be too large is
not a silent error here: it raises the energy, which is impossible in exact imaginary
time, so :func:`evolve_in_imaginary_time` halves it and tries again.
"""

DEFAULT_MAX_STEPS = 400
"""Cap on steps taken. A bound on a pathological run, not a target to be reached."""

DEFAULT_ENERGY_TOLERANCE = 1e-10
"""Energy change per step below which the state counts as having stopped falling.

The same tolerance the eigensolver uses, deliberately: two methods compared on one axis
must be allowed to stop for the same reason, or the comparison measures the stopping
rules rather than the methods.
"""

DEFAULT_REGULARISATION = 1e-6
r"""Ridge added to the metric's diagonal before the linear system is solved.

The Fubini-Study metric is singular wherever the ansatz has a redundant direction -- a
change of angles that moves no state -- and this circuit has them at shallow depth. Left
alone the solve would return an enormous velocity along a direction that does nothing,
which shows up as an energy that jumps rather than falls. The ridge is the standard cure
and is small enough that it changes no direction the metric actually resolves.
"""

MIN_STEP_FRACTION = 1e-3
"""How far the step may be cut before a run is called stalled rather than converging.

A step that has been halved ten times and still raises the energy is not a step that
needs halving again: it means the state is at the bottom of what this ansatz can reach,
and the honest exit is to say so.
"""


@dataclass(frozen=True, slots=True)
class ImaginaryTimeResult:
    r"""One completed evolution, carrying the evidence needed to judge it.

    Deliberately the same shape as
    :class:`~src.physics.quantum.variational_eigensolver.VqeResult`: an energy, the
    angles that reached it, the ansatz, how it stopped and the whole history. Anything
    that draws or grades one must be able to draw or grade the other without knowing
    which method produced it, or the comparison the module exists for cannot be made.

    Attributes:
        energy: The lowest energy reached. A variational upper bound on the true ground
            state, never a lower one -- it is the expectation value of a normalised
            state in the ansatz, exactly as in VQE.
        parameters: The angles that achieved it, folded into a single period.
        spec: The ansatz that was evolved, carrying its own gate and depth arithmetic.
        n_steps: Imaginary-time steps accepted.
        imaginary_time: Total :math:`\tau` elapsed. Not the same as ``n_steps`` when
            steps were cut, and it is the physically meaningful axis of the two.
        n_energy_evaluations: Energies computed, including those spent on steps that
            were rejected and retried. The honest cost of the run.
        n_metric_evaluations: Fubini-Study metrics assembled. On a simulator these are
            cheap; on hardware each one is :math:`O(P^2)` extra circuits, and it is the
            number that decides whether this method is affordable on a device.
        converged: Whether the energy stopped falling rather than a limit being hit.
        stop_reason: Which exit was taken. See :data:`ImaginaryTimeStop`.
        energy_history: The energy after each accepted step, oldest first, beginning
            with the energy of the starting angles.
    """

    energy: float
    parameters: FloatArray
    spec: AnsatzSpec
    n_steps: int
    imaginary_time: float
    n_energy_evaluations: int
    n_metric_evaluations: int
    converged: bool
    stop_reason: ImaginaryTimeStop
    energy_history: tuple[float, ...] = ()

    @property
    def energy_per_site(self) -> float:
        """Energy divided by the number of sites.

        The comparable quantity across chain lengths: the total energy is extensive, so
        quoting it alone makes a longer chain look worse than a shorter one.
        """
        return self.energy / self.spec.n_qubits

    @property
    def improvement(self) -> float:
        """How far the energy fell from the starting angles.

        A run that improved by nothing did not fail to reach the ground state so much as
        fail to start, and that is a different diagnosis from one that fell and stalled.
        """
        if not self.energy_history:
            return 0.0
        return self.energy_history[0] - self.energy

    def describe(self) -> dict[str, Any]:
        """Render the outcome as plain data for a tool result or a log line.

        Returns:
            A flat mapping. The history is summarised rather than included, since a
            four-hundred-entry list buys nothing the first value, the last value and the
            length do not.
        """
        return {
            "energy": round(self.energy, 12),
            "energy_per_site": round(self.energy_per_site, 12),
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "n_steps": self.n_steps,
            "imaginary_time": round(self.imaginary_time, 12),
            "n_energy_evaluations": self.n_energy_evaluations,
            "n_metric_evaluations": self.n_metric_evaluations,
            "improvement": round(self.improvement, 12),
            "ansatz": self.spec.describe(),
        }


def state_derivatives(
    theta: FloatArray,
    spec: AnsatzSpec,
    diagonal: FloatArray,
    transverse_field: float,
) -> ComplexArray:
    r"""Differentiate the prepared state with respect to every angle at once.

    Writing the circuit as :math:`\lvert\psi\rangle = \hat U_M \cdots \hat U_1
    \lvert\psi_0\rangle` with :math:`\hat U_k = e^{-i\theta_k \hat G_k}`, the derivative
    with respect to one angle is the same circuit with its generator inserted:

    .. math::

        \partial_k \lvert \psi \rangle
        = \hat U_M \cdots \hat U_{k+1} \,(-i \hat G_k)\, \hat U_k \cdots \hat U_1
          \lvert \psi_0 \rangle .

    Because :math:`\hat G_k` commutes with its own gate, the insertion can be made
    *after* :math:`\hat U_k` has been applied, which is what lets one forward sweep carry
    every derivative at once: at each gate the sweep pushes the derivatives it already
    holds through that gate and starts one more. That is :math:`O(P)` states carried
    through :math:`O(P)` gates, against :math:`O(P)` separate circuit runs for the
    obvious implementation.

    A device cannot do this. These are amplitudes, read between the layers, and
    measurement destroys them. On hardware the same quantities come from Hadamard tests,
    which is where VarQITE's real cost lives -- see :attr:`ImaginaryTimeResult
    .n_metric_evaluations`.

    Args:
        theta: The flat parameter vector, packed as
            :func:`~src.physics.quantum.ansatz.join_angles` packs it.
        spec: The ansatz being differentiated.
        diagonal: The diagonal of :math:`\hat H_\text{diag}`, from
            :func:`~src.physics.quantum.statevector.diagonal_energies`.
        transverse_field: The field strength :math:`h`.

    Returns:
        An array of shape ``(n_parameters, state_dimension)`` whose row :math:`k` is
        :math:`\partial_k \lvert\psi\rangle`, in the same order as ``theta``.

    Raises:
        ValueError: If ``theta`` does not carry exactly two angles per layer.

    Examples:
        A one-layer circuit has two angles, so two derivative states:

        >>> import numpy as np
        >>> spec = AnsatzSpec(n_qubits=3, depth=1)
        >>> derivatives = state_derivatives(np.zeros(2), spec, diagonal_energies(3), 1.0)
        >>> derivatives.shape
        (2, 8)
    """
    if theta.size != spec.n_parameters:
        raise ValueError(f"expected {spec.n_parameters} angles for this ansatz, got {theta.size}")
    gamma, beta = split_angles(theta)
    derivatives = np.zeros((spec.n_parameters, spec.state_dimension), dtype=np.complex128)
    carried: list[int] = []
    state = uniform_superposition(spec.n_qubits)

    for layer in range(spec.depth):
        for index in carried:
            derivatives[index] = apply_diagonal_layer(
                derivatives[index], float(gamma[layer]), diagonal
            )
        state = apply_diagonal_layer(state, float(gamma[layer]), diagonal)
        # d/dgamma of exp(-i gamma H_diag) is -i H_diag times the state after the gate.
        derivatives[layer] = -1j * diagonal * state
        carried.append(layer)

        for index in carried:
            derivatives[index] = apply_field_layer(
                derivatives[index], float(beta[layer]), transverse_field
            )
        state = apply_field_layer(state, float(beta[layer]), transverse_field)
        # The field generator is -h times the sum of sigma^x, so the -i and the minus
        # sign cancel into +i. Taken from `sigma_x_sum` rather than rewritten, so that
        # this module and the adjoint gradient cannot disagree about the sign.
        derivatives[spec.depth + layer] = 1j * transverse_field * sigma_x_sum(state)
        carried.append(spec.depth + layer)

    return derivatives


def fubini_study_metric(derivatives: ComplexArray, state: ComplexArray) -> FloatArray:
    r"""Assemble the geometry the ansatz's parameters induce on the space of states.

    .. math::

        g_{kj} = \mathrm{Re}\left[
            \langle \partial_k \psi \vert \partial_j \psi \rangle
            - \langle \partial_k \psi \vert \psi \rangle
              \langle \psi \vert \partial_j \psi \rangle
        \right] .

    The subtracted term is the projection onto the state itself, and removing it is what
    makes this a distance between *physical states* rather than between parameter
    vectors: a change of angles that only multiplies the state by a phase moves nothing
    observable and must cost nothing here.

    Why the whole method needs it: without :math:`g`, a step of the same size in every
    angle is assumed to move the state by the same amount in every direction, which for
    this circuit is simply false. That assumption is what plain gradient descent makes,
    and correcting it is the difference between this method and the eigensolver.

    Args:
        derivatives: The rows of :func:`state_derivatives`.
        state: The normalised state they were taken at.

    Returns:
        The metric, of shape ``(n_parameters, n_parameters)``. Real, symmetric and
        positive semi-definite by construction -- semi-definite rather than definite
        because a redundant parameter direction is a genuine zero eigenvalue and not a
        numerical accident.
    """
    overlaps = derivatives @ np.conj(state)
    gram = derivatives.conj() @ derivatives.T
    projected = gram - np.outer(np.conj(overlaps), overlaps)
    metric = np.asarray(projected.real, dtype=np.float64)
    # Symmetrised because the two halves differ only in floating-point noise, and an
    # asymmetric matrix would send `solve` down a general path for no reason.
    return np.asarray(0.5 * (metric + metric.T), dtype=np.float64)


def imaginary_time_velocity(
    metric: FloatArray,
    gradient: FloatArray,
    regularisation: float = DEFAULT_REGULARISATION,
) -> FloatArray:
    r"""Solve McLachlan's condition for how fast each angle should move.

    The system is :math:`(g + \varepsilon I)\,\dot\theta = -\tfrac{1}{2}\nabla E`, the
    right-hand side being :math:`-C` written in terms of the ordinary energy gradient --
    the two differ by exactly a factor of two, which is worth stating because getting it
    wrong changes the effective step size and nothing else, and so hides.

    Args:
        metric: The Fubini-Study metric at the current angles.
        gradient: The energy gradient at the same angles.
        regularisation: Ridge added to the diagonal. See :data:`DEFAULT_REGULARISATION`.

    Returns:
        The velocity :math:`\dot\theta`, in the same packing as the angles.
    """
    ridged = metric + regularisation * np.eye(metric.shape[0])
    velocity, *_ = np.linalg.lstsq(ridged, -0.5 * gradient, rcond=None)
    return np.asarray(velocity, dtype=np.float64)


def evolve_in_imaginary_time(
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
    step: float = DEFAULT_STEP,
    max_steps: int = DEFAULT_MAX_STEPS,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    regularisation: float = DEFAULT_REGULARISATION,
    initial_parameters: FloatArray | None = None,
    lattice: Lattice | None = None,
) -> ImaginaryTimeResult:
    r"""Run VarQITE on the mixed-field Ising chain and return the result with its evidence.

    One Euler step of McLachlan's equation per iteration, with the step halved and the
    iteration retried whenever the energy fails to fall. That retry is not defensive
    padding: exact imaginary time can never raise the energy, so an increase is proof
    that :math:`\Delta\tau` was too large for the linearisation, and a run that accepted
    it would report a number the method does not actually produce.

    Args:
        n_sites: Number of sites, equal to the number of qubits.
        depth: Number of ansatz layers. Zero is legal and returns the energy of the bare
            initial state, which has the closed form :math:`-hL`.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        family: ``"hva"`` or ``"qaoa"``. Identical circuits -- carried so the result says
            which tradition the caller is arguing from.
        initialisation: How the angles start. See
            :func:`~src.physics.quantum.ansatz.initial_angles`.
        ramp_time: Total adiabatic time for the ramp initialisation.
        seed: Seed for a randomised initialisation. Explicit so the run reproduces.
        step: The imaginary-time increment. See :data:`DEFAULT_STEP`.
        max_steps: Cap on accepted steps.
        energy_tolerance: Energy change below which the state has stopped falling.
        regularisation: Ridge on the metric. See :data:`DEFAULT_REGULARISATION`.
        initial_parameters: Angles to start from, overriding ``initialisation``.
        lattice: The shape the sites sit on, or ``None`` for a chain. It reaches
            both halves of the problem at once -- the bond list the circuit acts on
            and the diagonal the state is evolved against -- so a square or
            triangular lattice cannot be solved as a circuit of one shape scored
            against a Hamiltonian of another.

    Returns:
        The completed evolution.

    Raises:
        ValueError: If ``initial_parameters`` does not carry two angles per layer, or if
            ``step`` is not positive.

    Examples:
        At depth zero nothing evolves and the answer is the closed form:

        >>> result = evolve_in_imaginary_time(n_sites=4, depth=0, transverse_field=1.0)
        >>> round(result.energy, 10), result.stop_reason
        (-4.0, 'depth_zero')

        The energy falls, and never below the true ground state:

        >>> run = evolve_in_imaginary_time(n_sites=6, depth=3)
        >>> bool(run.energy < run.energy_history[0])
        True
    """
    if step <= 0.0:
        raise ValueError(f"the imaginary-time step must be positive, got {step}")

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
        bare = energy(
            evolve(np.zeros(0), spec, diagonal, transverse_field), diagonal, transverse_field
        )
        return ImaginaryTimeResult(
            energy=bare,
            parameters=np.zeros(0),
            spec=spec,
            n_steps=0,
            imaginary_time=0.0,
            n_energy_evaluations=1,
            n_metric_evaluations=0,
            converged=True,
            stop_reason="depth_zero",
            energy_history=(bare,),
        )

    if initial_parameters is None:
        theta = initial_angles(depth, initialisation, ramp_time=ramp_time, seed=seed)
    else:
        theta = np.asarray(initial_parameters, dtype=np.float64)
        if theta.size != spec.n_parameters:
            raise ValueError(f"expected {spec.n_parameters} initial angles, got {theta.size}")

    smallest_step = step * MIN_STEP_FRACTION
    current, gradient = energy_and_gradient(theta, spec, diagonal, transverse_field)
    history = [current]
    n_energies = 1
    n_metrics = 0
    elapsed = 0.0
    reason: ImaginaryTimeStop = "step_limit"

    for _ in range(max_steps):
        state = evolve(theta, spec, diagonal, transverse_field)
        metric = fubini_study_metric(
            state_derivatives(theta, spec, diagonal, transverse_field), state
        )
        n_metrics += 1
        velocity = imaginary_time_velocity(metric, gradient, regularisation)

        # Shrink until the energy actually falls. In exact imaginary time it always
        # does; here it can fail only because the Euler step outran the linearisation.
        trial = step
        while True:
            moved = theta + trial * velocity
            value, moved_gradient = energy_and_gradient(moved, spec, diagonal, transverse_field)
            n_energies += 1
            if value <= current:
                break
            trial *= 0.5
            if trial < smallest_step:
                break

        if trial < smallest_step:
            reason = "step_collapsed"
            break

        fall = current - value
        theta, current, gradient = moved, value, moved_gradient
        history.append(current)
        elapsed += trial
        if fall <= energy_tolerance:
            reason = "converged"
            break

    return ImaginaryTimeResult(
        energy=current,
        parameters=wrap_angles(
            theta, *angle_periods(coupling, transverse_field, longitudinal_field)
        ),
        spec=spec,
        n_steps=len(history) - 1,
        imaginary_time=elapsed,
        n_energy_evaluations=n_energies,
        n_metric_evaluations=n_metrics,
        converged=reason in ("converged", "step_collapsed"),
        stop_reason=reason,
        energy_history=tuple(history),
    )
