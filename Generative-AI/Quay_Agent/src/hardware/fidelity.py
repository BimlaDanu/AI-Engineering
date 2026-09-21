r"""What survives the noise, and the depth ceiling that leaves.

A circuit that fits in the register and finishes inside :math:`T_2` can still
return nothing worth reading. This module multiplies the error sources together
and returns the three numbers a verdict needs: how much signal is left, how deep
the circuit may go before there is none, and how many more shots the noise costs.

Three independent factors, each the fraction of a measured expectation value that
one error source leaves standing:

.. math::

    F \;=\;
    \underbrace{(1-\epsilon_2)^{n_2}\,(1-\epsilon_1)^{n_1}}_{\text{gates}}
    \;\cdot\;
    \underbrace{e^{-\tau_\text{idle}/T_2}}_{\text{dephasing}}
    \;\cdot\;
    \underbrace{(1-2\epsilon_\text{ro})^{k}}_{\text{readout}}

Neither the readout factor nor the dephasing exponent is the obvious one.

Readout carries :math:`2\epsilon` and not :math:`\epsilon` because a misread bit does
not lose the term, it *negates* it. A :math:`\pm 1` outcome flipped with probability
:math:`\epsilon` has mean :math:`(1-2\epsilon)` times the true value, so a
:math:`k`-qubit term is damped by :math:`(1-2\epsilon_\text{ro})^k`. A gate error
behaves differently -- it scatters the state towards one whose expectation is zero,
so there the probability of surviving *is* the damping, which is why the first two
factors read as they do.

Dephasing is charged on idle time only. A published two-qubit error rate is
already mostly decoherence suffered during the gate, measured on the machine with
the qubits alive, so multiplying by :math:`e^{-Lt/T_2}` double-counts it. What is
genuinely uncounted is time spent waiting while neighbours are gated, and during
readout, so :math:`\tau_\text{idle}` is total qubit-time :math:`L \cdot t` minus
the qubit-time occupied by gates -- a quantity the schedule already knows.

It is also charged per observable, not per register. Every one- and two-body term
in this Hamiltonian is estimated from at most :data:`OBSERVABLE_LOCALITY` bits of
the outcome, so damping the whole :math:`L`-qubit string prices a measurement the
algorithm never performs. A twelve-qubit register at 1.5% readout error is a factor
of 0.69 by that rule and 0.94 by this one.

Independent errors is the remaining assumption. It is optimistic in ignoring
crosstalk and correlated errors, and pessimistic in ignoring that an optimiser
re-tunes its angles around a systematic gate error. Neither is quantified, so the
result is a magnitude.

The cost is arithmetic, not a badge. A depolarised component contributes nothing
to the expectation of a traceless Hamiltonian, so a measurement returns roughly
:math:`F\,\langle\hat H\rangle` and dividing by :math:`F` to recover the true
value divides the noise by :math:`F` too. Fixed precision therefore costs
:math:`1/F^2` times as many shots: a factor of four at :math:`F = 0.5`, a
hundred at :math:`F = 0.1`. :func:`shot_inflation` is that multiplication.

:func:`coherence_budget` fills :class:`~src.agent.state.CoherenceBudget` from a
real machine\'s clock rather than from nominal constants, so the planner has one
depth-limit type instead of two to keep in step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from src.agent.state import USABLE_COHERENCE_FRACTION, CoherenceBudget
from src.hardware.devices import Device
from src.hardware.transpile import Transpiled, fits, transpile
from src.physics.lattice import Lattice
from src.physics.quantum.ansatz import AnsatzSpec

USABLE_FIDELITY_FLOOR = 0.5
r"""How much signal must survive before a result is worth reporting.

Below one half, most of what comes back is a state the algorithm did not prepare,
and the shot cost of seeing through it has already quadrupled -- so the circuit is
simultaneously less informative and more expensive, which is the point at which
running it stops being defensible. It is a convention rather than a threshold
anybody measured, so it is a named constant and every function that applies it
takes it as an argument with this as the default.
"""

OBSERVABLE_LOCALITY = 2
r"""How many bits of one measurement outcome an energy term is read from.

Every term in this Hamiltonian is one-body or two-body, so the worst case is two.
Named rather than written as a literal because it is the one number here that is a
property of the *problem* rather than of the machine, and a Hamiltonian with
longer-range terms would change it and nothing else in this module.
"""

MAX_DEPTH_SEARCHED = 128
r"""How far the depth ladder is climbed before the search gives up.

Fidelity falls monotonically with depth, so the search below is a walk up until it
crosses the floor. This bounds it for the case where it never does -- an ideal
machine, where every depth is free -- so that "no ceiling" returns a number rather
than looping.

An order of magnitude above the deepest circuit anything here proposes, and the
whole cost of the ideal case, since that walk always runs to the end. Raising it
would not change a single reported ceiling on a real machine and would make the
control slower to price, which is the wrong trade in both directions.
"""


@dataclass(frozen=True, slots=True)
class Fidelity:
    r"""What is left of a circuit's signal after the machine has run it.

    Attributes:
        gates: Probability that no gate misfired.
        coherence: Probability that no qubit dephased while it was idle. Time spent
            inside a gate is not charged here -- the gate's own error rate already
            includes it.
        readout: What is left of one energy term after its bits may have been
            misread. A flip negates a term rather than losing it, so the factor is
            :math:`(1-2\epsilon_\text{ro})` per bit, over the bits one term is read
            from rather than the whole register; see the module docstring.
        total: The three multiplied together.
        duration_ns: How long the circuit occupied the machine.
        idle_ns: Qubit-time spent waiting rather than being operated on, summed over
            the register. This is what the coherence factor is charged on, and it is
            reported because it is the number that says whether a schedule is
            wasteful or whether the machine is simply slow.
        n_qubits: Width of the register.
        two_qubit_gates: Entangling gates, the term that dominates :attr:`gates`.
    """

    gates: float
    coherence: float
    readout: float
    total: float
    duration_ns: float
    idle_ns: float
    n_qubits: int
    two_qubit_gates: int

    @property
    def dominant_loss(self) -> Literal["gates", "coherence", "readout"]:
        """Name whichever factor cost the most, so a report can say what to fix.

        The three losses compete and the winner changes with the circuit: a shallow
        wide circuit loses to readout, a deep one loses to gates, and a slow machine
        loses to dephasing. Naming it turns a fidelity number into an instruction.
        """
        losses: dict[Literal["gates", "coherence", "readout"], float] = {
            "gates": 1.0 - self.gates,
            "coherence": 1.0 - self.coherence,
            "readout": 1.0 - self.readout,
        }
        return max(losses, key=lambda name: losses[name])

    @property
    def usable(self) -> bool:
        """Whether enough signal survived to be worth a shot budget."""
        return self.total >= USABLE_FIDELITY_FLOOR

    def shot_inflation(self) -> float:
        r"""How many times more shots the noise costs, for the same precision.

        Noise damps the measured expectation value towards zero; undoing the damping
        divides the statistical error by the same factor, so the shot count to reach
        a fixed precision scales as :math:`1/F^2`.

        Returns:
            :math:`1/F^2`. Infinite when nothing survived, which is the honest
            answer -- no number of shots recovers a signal of zero.
        """
        return math.inf if self.total <= 0.0 else 1.0 / (self.total * self.total)

    def energy_bias(self, energy_scale: float) -> float:
        r"""How far the noise pulls a reported energy away from the true one.

        The depolarised part of the state has zero expectation value for a traceless
        Hamiltonian, so an uncorrected measurement reads :math:`F\,\langle\hat H
        \rangle` and sits :math:`(1-F)\,\lvert\langle\hat H\rangle\rvert` away from
        the truth. This is a *bias*, not a fluctuation: it does not shrink with more
        shots, and it pulls the reading towards zero. Every energy this model reports
        is negative, so towards zero is upward: a variational result is an upper
        bound on the ground-state energy, and noise makes that bound weaker rather
        than falsely tight.

        Args:
            energy_scale: Magnitude of the energy being measured.

        Returns:
            The size of the shift, in the same units.
        """
        return (1.0 - self.total) * abs(energy_scale)

    def describe(self) -> dict[str, Any]:
        """Render the fidelity estimate as plain data.

        Returns:
            A flat mapping of the three factors, the total, and what the total costs
            in shots.
        """
        inflation = self.shot_inflation()
        return {
            "total": round(self.total, 5),
            "gates": round(self.gates, 5),
            "coherence": round(self.coherence, 5),
            "readout": round(self.readout, 5),
            "dominant_loss": self.dominant_loss,
            "usable": self.usable,
            "shot_inflation": None if math.isinf(inflation) else round(inflation, 2),
            "duration_us": round(self.duration_ns / 1000.0, 3),
            "idle_qubit_us": round(self.idle_ns / 1000.0, 3),
            "n_qubits": self.n_qubits,
            "two_qubit_gates": self.two_qubit_gates,
        }


def estimate(compiled: Transpiled) -> Fidelity:
    """Work out what survives one compiled circuit on the machine it was built for.

    Args:
        compiled: The circuit, already placed, routed and scheduled.

    Returns:
        The three loss factors and their product.
    """
    device = compiled.device
    width = compiled.ansatz.n_qubits
    gates = (1.0 - device.two_qubit_error) ** compiled.two_qubit_gates * (
        1.0 - device.single_qubit_error
    ) ** compiled.single_qubit_gates
    idle = idle_qubit_ns(compiled)
    coherence = math.exp(-idle / device.t2_ns)
    readout = max(0.0, 1.0 - 2.0 * device.readout_error) ** OBSERVABLE_LOCALITY
    return Fidelity(
        gates=gates,
        coherence=coherence,
        readout=readout,
        total=gates * coherence * readout,
        duration_ns=compiled.duration_ns,
        idle_ns=idle,
        n_qubits=width,
        two_qubit_gates=compiled.two_qubit_gates,
    )


def idle_qubit_ns(compiled: Transpiled) -> float:
    """Total qubit-time spent waiting rather than being operated on.

    Every qubit is alive for the whole duration of the circuit, so the register
    offers ``n_qubits * duration`` qubit-nanoseconds. Gates consume some of that --
    an entangling gate occupies two qubits for its duration, a rotation occupies
    one -- and whatever is left is idle, which is where the uncounted dephasing
    happens. Readout is idle by this reckoning, which is correct: a qubit measured
    late in the register is dephasing while the earlier ones are read.

    Args:
        compiled: The circuit, already placed, routed and scheduled.

    Returns:
        The idle qubit-time in nanoseconds. Never negative -- a schedule dense
        enough to make the arithmetic go the other way is reported as fully
        occupied rather than as impossible.
    """
    device = compiled.device
    offered = compiled.ansatz.n_qubits * compiled.duration_ns
    occupied = (
        2 * compiled.two_qubit_gates * device.two_qubit_gate_ns
        + compiled.single_qubit_gates * device.single_qubit_gate_ns
    )
    return max(0.0, offered - occupied)


def coherence_budget(
    device: Device, usable_fraction: float = USABLE_COHERENCE_FRACTION
) -> CoherenceBudget:
    """Express one machine's clock as the depth limit a planner refuses with.

    Args:
        device: The machine.
        usable_fraction: How much of :math:`T_2` a circuit may occupy. The default
            is the project-wide convention.

    Returns:
        A budget carrying this machine's gate duration and coherence time in place
        of the nominal figures.
    """
    return CoherenceBudget(
        two_qubit_gate_ns=device.two_qubit_gate_ns,
        coherence_ns=device.t2_ns,
        usable_fraction=usable_fraction,
    )


@dataclass(frozen=True, slots=True)
class DepthCeiling:
    """The deepest circuit worth running on one machine, and why it stops there.

    Two limits bind and they are not the same. Coherence is a wall: past it the
    circuit has not finished when the qubits have. Fidelity is a slope: the answer
    degrades continuously and at some point stops being worth the shots. The lower
    of the two is what a planner should believe, and naming which one it was is what
    turns a refusal into advice -- a coherence limit asks for a faster gate, a
    fidelity limit asks for a better one.

    Attributes:
        device: Name of the machine.
        n_sites: Length of the chain the ceiling was computed for.
        boundary: Which boundary condition, since a ring routes and a segment does
            not.
        by_coherence: Deepest ansatz layer count whose whole schedule -- entangling
            layers, single-qubit layers and the readout at the end -- finishes
            inside the usable window. Readout alone is a sixth of that window on a
            superconducting machine, so charging entangling time only would report
            a rung the circuit does not actually reach.
        by_fidelity: Deepest ansatz layer count whose surviving signal clears the
            floor.
        floor: The fidelity floor that was applied.
        limit: The lower of the two, which is the answer.
        binding: Which of them set it.
    """

    device: str
    n_sites: int
    boundary: Literal["open", "periodic"]
    by_coherence: int
    by_fidelity: int
    floor: float
    limit: int
    binding: Literal["coherence", "fidelity", "neither"]

    def describe(self) -> dict[str, Any]:
        """Render the ceiling as plain data.

        Returns:
            A flat mapping of both limits and which one binds.
        """
        return {
            "device": self.device,
            "n_sites": self.n_sites,
            "boundary": self.boundary,
            "max_depth": self.limit,
            "max_depth_by_coherence": self.by_coherence,
            "max_depth_by_fidelity": self.by_fidelity,
            "fidelity_floor": self.floor,
            "binding_constraint": self.binding,
        }


@lru_cache(maxsize=2048)
def depth_ceiling(
    n_sites: int,
    device: Device,
    boundary: Literal["open", "periodic"] = "open",
    floor: float = USABLE_FIDELITY_FLOOR,
    usable_fraction: float = USABLE_COHERENCE_FRACTION,
    longitudinal: bool = False,
    lattice: Lattice | None = None,
) -> DepthCeiling:
    r"""Find the deepest ansatz this machine will carry for this chain.

    Both limits are computed by walking the depth ladder from one and compiling at
    each rung, rather than by inverting a formula. Routing cost is not a closed-form
    function of depth on an arbitrary lattice -- it depends on which qubits the
    layout happened to use -- so the walk is the only version that stays correct
    when a device is added.

    The walk is memoised, because a planner climbing its own depth ladder asks for
    the same ceiling at every rung and an ideal machine never crosses either limit,
    so the search runs to :data:`MAX_DEPTH_SEARCHED` every time it is asked. Every
    argument is immutable and the answer depends on nothing else, so the cache
    cannot go stale.

    Args:
        n_sites: Length of the chain.
        device: The machine.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        floor: Surviving fidelity below which a result is not worth reporting.
        usable_fraction: How much of :math:`T_2` a circuit may occupy.
        longitudinal: Whether the Hamiltonian carries a :math:`g\sum_i\hat\sigma^z_i`
            term, which adds single-qubit rotations and therefore a little duration.
        lattice: The shape the register sits on, or ``None`` for a chain. It
            changes the answer more than any other argument here, because it
            changes how many two-qubit gates a layer holds: a line needs two
            rounds per layer, a square lattice four and a triangular one six, and
            each round is a full two-qubit gate duration against a fixed
            coherence time. Left at ``None`` for a lattice question, this reports
            a line's ceiling under the lattice's name -- and a ceiling is exactly
            the number a feasibility verdict turns on. Frozen and hashable, so the
            memoisation below still holds.

    Returns:
        Both limits and the lower of them.

    Raises:
        ValueError: If the chain does not fit on the machine.
    """
    refusal = fits(n_sites, device)
    if refusal is not None:
        raise ValueError(refusal)
    by_coherence = 0
    by_fidelity = 0
    for depth in range(1, MAX_DEPTH_SEARCHED + 1):
        compiled = transpile(
            AnsatzSpec(
                n_qubits=n_sites,
                depth=depth,
                boundary=boundary,
                longitudinal=longitudinal,
                lattice=lattice,
            ),
            device,
        )
        within_coherence = compiled.duration_ns <= usable_fraction * device.t2_ns
        above_floor = estimate(compiled).total >= floor
        if within_coherence:
            by_coherence = depth
        if above_floor:
            by_fidelity = depth
        if not within_coherence and not above_floor:
            break
    limit = min(by_coherence, by_fidelity)
    if limit == MAX_DEPTH_SEARCHED:
        binding: Literal["coherence", "fidelity", "neither"] = "neither"
    elif by_coherence < by_fidelity:
        binding = "coherence"
    elif by_fidelity < by_coherence:
        binding = "fidelity"
    else:
        binding = "coherence"
    return DepthCeiling(
        device=device.name,
        n_sites=n_sites,
        boundary=boundary,
        by_coherence=by_coherence,
        by_fidelity=by_fidelity,
        floor=floor,
        limit=limit,
        binding=binding,
    )


def shot_inflation(compiled: Transpiled) -> float:
    r"""How many times more shots this circuit costs because the machine is noisy.

    A convenience over :meth:`Fidelity.shot_inflation` for the common case where the
    caller has a compiled circuit and wants the one number.

    Args:
        compiled: The circuit, already placed, routed and scheduled.

    Returns:
        :math:`1/F^2`, or infinity when nothing survives.
    """
    return estimate(compiled).shot_inflation()
