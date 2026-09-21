r"""What a circuit would cost, described before any circuit is built.

An :class:`AnsatzSpec` is a *specification*, not a circuit: the layer structure, the
parameter count and the gate arithmetic, computed from integers alone. That separation
is the point. The agent has to choose a depth under a budget, and it must be able to
price the choice without a simulator, without Qiskit and without a device -- pricing
twenty candidate depths should cost twenty multiplications, not twenty circuit builds.

One family, two names. ``"hva"`` and ``"qaoa"`` produce the identical circuit here:

.. math::

    \lvert \psi(\gamma, \beta) \rangle = \prod_{k=1}^{p}
        e^{-i \beta_k \hat H_\text{field}} \, e^{-i \gamma_k \hat H_\text{diag}}
        \lvert + \rangle^{\otimes L}

What differs is the *objective* they are usually pointed at, not the gates. QAOA in the
textbook minimises a classical cost function and treats the transverse field as a mixing
tool; here the transverse field is part of the physics being minimised. The name is
carried so the agent can say which tradition it is arguing from; it changes nothing
about the gates that are applied.

The gate counts belong here rather than in a transpiler. The one non-obvious number
is the two-qubit *depth*, and it is a statement about physics rather than about a
compiler: every :math:`\hat\sigma^z \hat\sigma^z` term commutes with every other, so the
bonds may be reordered freely, so they can be grouped into rounds of mutually disjoint
pairs and applied simultaneously. A chain needs two such rounds however long it is. An
agent that misses this prices a feasible problem as infeasible by a factor of
:math:`L/2`, which is why the count is derived by colouring the bond list rather than
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.physics.lattice import Lattice
from src.physics.model import BoundaryCondition
from src.physics.quantum.hamiltonians import chain_bonds

FloatArray = NDArray[np.float64]

DEFAULT_DEPTH = 2
"""Layers to assume when nobody has said how many.

Two is the smallest depth that is a *circuit* rather than a single Trotter step, and
it is what the interface's depth knob starts on. It lives here, beside the layer
arithmetic, rather than in the interface: the agent needs the same number when a
question names a circuit but no depth, and a default that differs between the page and
the answer would put a drawing of one circuit beside the algebra of another.
"""

AnsatzFamily = Literal["hva", "qaoa"]
"""Which tradition the caller is naming. Identical circuits -- see the module docstring."""

Initialisation = Literal["adiabatic_ramp", "small_angle", "zeros"]
"""How the angles are first set. :func:`initial_angles` implements each."""

MAX_SIMULABLE_QUBITS = 24
"""Above this, a dense state vector stops fitting in this machine's memory.

At 24 qubits a ``complex128`` state vector is 256 MiB, and the gradient needs three of
them live at once. Carried as a named ceiling rather than left implicit so that
:meth:`AnsatzSpec.simulation_bytes` can be compared against something and the agent is
told *why* a size was refused instead of watching the process die.
"""

DEFAULT_RAMP_TIME = 2.0
"""Total adiabatic time the linear ramp is scaled to when the caller does not choose.

The middle of the range the ramp-time scan explores. It is a starting point for an
optimiser rather than a converged value, and nothing depends on its being right --
anything that cares about the schedule scans for it instead.
"""

SMALL_ANGLE_SCALE = 1e-2
"""Magnitude of the near-identity initialisation.

Small enough that the circuit starts where the cost landscape still has curvature, which
is the standard first defence against a barren plateau, and large enough that the
gradient is not lost to floating-point noise.
"""


def bond_rounds(
    bonds: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Group the bonds into rounds of gates that can fire at the same moment.

    Two bonds may fire together exactly when they share no qubit, so this is an edge
    colouring of the interaction graph -- and for the graphs this project uses the
    answer is small and known: a path needs two colours, an even ring two, an odd
    ring three. Rather than encode those three cases and hope the classification is
    right, the colouring is performed greedily on the actual bond list, which is
    correct for any bond set a future model might introduce.

    The grouping itself is returned rather than only its size, because the interface
    draws the schedule as a circuit diagram and a drawing built from a second,
    separate colouring could show a layer the depth arithmetic never priced.

    Args:
        bonds: The coupled pairs, as returned by
            :func:`~src.physics.quantum.hamiltonians.chain_bonds`.

    Returns:
        One tuple of mutually disjoint bonds per round, in firing order. Empty when
        there are no bonds at all.

    Examples:
        A four-site chain splits into the even bonds and then the odd one:

        >>> bond_rounds(((0, 1), (1, 2), (2, 3)))
        (((0, 1), (2, 3)), ((1, 2),))
    """
    rounds: list[list[tuple[int, int]]] = []
    occupied: list[set[int]] = []
    for left, right in bonds:
        for group, qubits in zip(rounds, occupied, strict=True):
            if left not in qubits and right not in qubits:
                group.append((left, right))
                qubits.update((left, right))
                break
        else:
            rounds.append([(left, right)])
            occupied.append({left, right})
    return tuple(tuple(group) for group in rounds)


def _bond_rounds(bonds: tuple[tuple[int, int], ...]) -> int:
    """Count the rounds of simultaneous two-qubit gates one layer needs.

    Args:
        bonds: The coupled pairs.

    Returns:
        The number of rounds. Zero when there are no bonds at all.
    """
    return len(bond_rounds(bonds))


@dataclass(frozen=True, slots=True)
class AnsatzSpec:
    r"""The shape and price of one variational circuit family.

    Attributes:
        n_qubits: Width of the register, equal to the number of sites in the chain.
        depth: Number of layers, written ``p`` in the QAOA literature and ``L`` in the
            HVA literature. Zero is legal and means the bare initial state, which is a
            useful edge case because its energy has a closed form.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring. It changes the
            bond count, and for an odd ring it also changes the depth -- see
            :attr:`two_qubit_rounds_per_layer`.
        family: Which tradition the caller is naming. Does not change a single gate.
        longitudinal: Whether the Hamiltonian carries a :math:`g \sum_i \hat\sigma^z_i`
            term. It costs one single-qubit rotation per site per layer and **no**
            two-qubit depth at all, which is exactly why it is the knob this project
            holds in reserve: it breaks integrability for free.
        lattice: The shape the register sits on, or ``None`` for a chain. Every
            count on this class is derived from :attr:`bonds`, so supplying a
            lattice is what makes the gate count, the two-qubit depth and the
            circuit drawing describe a square or triangular problem rather than a
            chain of the same width. Optional and defaulting to ``None`` because a
            chain is the overwhelmingly common case and a required argument that is
            almost always the same value is an argument that stops being read.
    """

    n_qubits: int
    depth: int
    boundary: BoundaryCondition = "open"
    family: AnsatzFamily = "hva"
    longitudinal: bool = False
    lattice: Lattice | None = None

    def __post_init__(self) -> None:
        """Reject a specification that could not describe a real circuit.

        The lattice is checked against the two fields that duplicate it rather than
        being allowed to quietly win. A specification whose lattice says sixteen
        sites and whose ``n_qubits`` says twelve describes two different circuits,
        and whichever one a given property happened to read would be right half the
        time -- so neither is preferred and the disagreement is refused.

        Raises:
            ValueError: If the register is narrower than two qubits, the depth is
                negative, or a lattice was given that contradicts ``n_qubits`` or
                ``boundary``.
        """
        if self.n_qubits < 2:
            raise ValueError(f"a chain needs at least 2 qubits, got {self.n_qubits}")
        if self.depth < 0:
            raise ValueError(f"depth cannot be negative, got {self.depth}")
        if self.lattice is None:
            return
        if self.lattice.n_sites != self.n_qubits:
            raise ValueError(
                f"the lattice holds {self.lattice.n_sites} sites but n_qubits is "
                f"{self.n_qubits}; they describe different circuits"
            )
        if self.lattice.boundary != self.boundary:
            raise ValueError(
                f"the lattice is {self.lattice.boundary} but boundary is "
                f"{self.boundary}; they describe different circuits"
            )

    @property
    def geometry(self) -> str:
        """The shape the bonds were taken from, named for a log line or a caption."""
        return "chain" if self.lattice is None else self.lattice.geometry

    @property
    def bonds(self) -> tuple[tuple[int, int], ...]:
        """The coupled pairs one layer acts on, in the order they are listed.

        The single place the circuit layer learns its geometry. Everything priced
        on this class -- gate counts, two-qubit depth, the rounds a drawing shows --
        reads this property, so a lattice supplied here reaches all of them at once
        and none of them can be left behind on a chain.
        """
        if self.lattice is not None:
            return self.lattice.bonds()
        return chain_bonds(self.n_qubits, self.boundary)

    @property
    def n_parameters(self) -> int:
        r"""How many numbers the optimiser searches over.

        Two per layer -- one :math:`\gamma`, one :math:`\beta` -- and therefore
        **independent of the number of qubits**. That independence is the whole reason
        this family is worth running rather than a hardware-efficient ansatz: the
        classical optimisation does not grow with the physical system.
        """
        return 2 * self.depth

    @property
    def two_qubit_rounds_per_layer(self) -> int:
        r"""Rounds of simultaneous two-qubit gates in one layer.

        Two for a chain or an even ring, three for an odd ring, one where there is a
        single bond. The odd ring is the interesting case and it is not a rounding
        detail: a cycle of odd length cannot be two-coloured, so one bond is always left
        over and the layer costs half again as much.
        """
        return _bond_rounds(self.bonds)

    @property
    def rounds(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """The bonds themselves, grouped into the rounds :attr:`two_qubit_depth` prices.

        Exposed so that anything drawing the circuit draws the schedule that was
        costed, rather than a plausible one of its own.
        """
        return bond_rounds(self.bonds)

    @property
    def two_qubit_gates(self) -> int:
        r"""Total two-qubit gate count.

        Each bond rotation compiles to two ``CX`` gates around one
        :math:`\hat R^z`, so this is twice the bond count per layer.
        """
        return 2 * len(self.bonds) * self.depth

    @property
    def two_qubit_depth(self) -> int:
        r"""Two-qubit depth, which is what a coherence budget is actually spent on.

        Each round costs two ``CX`` layers, so a chain costs four per layer *whatever
        its length is*. Compare the naive left-to-right compilation, which costs
        ``2 * n_bonds`` per layer and grows with the chain: at 14 sites that is 26
        against 4, on a bit-identical unitary.
        """
        return 2 * self.two_qubit_rounds_per_layer * self.depth

    @property
    def single_qubit_gates(self) -> int:
        r"""Total single-qubit rotation count.

        One :math:`\hat R^x` per site per layer for the transverse field, plus one
        :math:`\hat R^z` per site per layer when a longitudinal field is present, plus
        the layer of Hadamards that prepares the initial state.
        """
        per_layer = self.n_qubits * (2 if self.longitudinal else 1)
        return per_layer * self.depth + self.n_qubits

    @property
    def state_dimension(self) -> int:
        """Length of the state vector a simulator would have to hold."""
        return 2**self.n_qubits

    def simulation_bytes(self) -> int:
        """Memory one ``complex128`` state vector of this width occupies.

        Returns:
            The size in bytes. The gradient needs three such vectors live at once, so
            the practical ceiling is a third of whatever the machine has.
        """
        return 16 * self.state_dimension

    def naive_two_qubit_depth(self) -> int:
        """What the depth would be without the commutation argument.

        Kept as a property rather than left as a remark because the *ratio* between this
        and :attr:`two_qubit_depth` is the single largest optimisation in the project,
        and a number the agent quotes is more convincing than an adjective.

        Returns:
            Two ``CX`` layers per bond per layer, applied strictly in sequence.
        """
        return 2 * len(self.bonds) * self.depth

    def describe(self) -> dict[str, Any]:
        """Render the specification as plain data for a tool result or a log line.

        Every value is an ``int``, a ``str`` or a ``bool``, so the result survives
        serialisation into a language model's context without a custom encoder.

        Returns:
            A flat mapping of the counts above, plus the depth saving that the
            even/odd argument buys.

        Examples:
            >>> AnsatzSpec(n_qubits=6, depth=3).describe()["two_qubit_depth"]
            12
            >>> AnsatzSpec(n_qubits=6, depth=3).describe()["depth_saving_factor"]
            2.5
        """
        naive = self.naive_two_qubit_depth()
        saving = naive / self.two_qubit_depth if self.two_qubit_depth else 1.0
        return {
            "family": self.family,
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "boundary": self.boundary,
            "geometry": self.geometry,
            "n_parameters": self.n_parameters,
            "n_bonds": len(self.bonds),
            "two_qubit_gates": self.two_qubit_gates,
            "two_qubit_depth": self.two_qubit_depth,
            "naive_two_qubit_depth": naive,
            "depth_saving_factor": round(saving, 3),
            "single_qubit_gates": self.single_qubit_gates,
            "state_dimension": self.state_dimension,
            "simulation_bytes": self.simulation_bytes(),
            "simulable": self.n_qubits <= MAX_SIMULABLE_QUBITS,
        }


def split_angles(theta: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Separate a flat parameter vector into its two schedules.

    The optimiser wants one contiguous vector; the circuit wants two named schedules.
    Packing is ``[gamma_1..gamma_p, beta_1..beta_p]`` -- blocked rather than interleaved,
    so that a slice of the vector is a whole schedule and can be plotted or interpolated
    without a stride.

    Args:
        theta: The flat vector, of even length.

    Returns:
        The ``gamma`` schedule and the ``beta`` schedule, as views.

    Raises:
        ValueError: If the length is odd, which means the two schedules cannot pair up.
    """
    if theta.size % 2 != 0:
        raise ValueError(f"expected an even number of angles, got {theta.size}")
    half = theta.size // 2
    return theta[:half], theta[half:]


def join_angles(gamma: FloatArray, beta: FloatArray) -> FloatArray:
    """Pack two schedules back into one flat parameter vector.

    Args:
        gamma: The diagonal-layer schedule.
        beta: The field-layer schedule.

    Returns:
        Their concatenation, the inverse of :func:`split_angles`.

    Raises:
        ValueError: If the two schedules have different lengths, which would silently
            produce a vector no depth can interpret.
    """
    if gamma.size != beta.size:
        raise ValueError(f"schedules must be the same length, got {gamma.size} and {beta.size}")
    return np.concatenate([gamma, beta])


def adiabatic_ramp(depth: int, ramp_time: float = DEFAULT_RAMP_TIME) -> FloatArray:
    r"""Initial angles read off a Trotterised adiabatic path.

    This is the one thing QAOA has that a generic variational eigensolver does not: a
    principled starting point rather than a random one. Interpolating
    :math:`\hat H(s) = (1-s)\hat H_\text{field} + s \hat H_\text{diag}` over a total time
    :math:`T` and Trotterising into :math:`p` steps of :math:`\Delta = T/p` gives, at
    step :math:`k`,

    .. math::

        \gamma_k = \Delta s_k , \qquad \beta_k = \Delta (1 - s_k) ,

    so :math:`\gamma` rises while :math:`\beta` falls and the whole schedule is fixed by
    the single scale :math:`T`. That reduces the initial search from :math:`2p`
    parameters to one, which is why a coarse scan over ``ramp_time`` is a cheap and
    effective first move.

    The path is sampled at the **midpoints** :math:`s_k = (k - \tfrac12)/p` rather than
    at the endpoints :math:`k/p`. Sampling the endpoint puts :math:`s_p = 1` exactly,
    which sets :math:`\beta_p = 0` and spends the last layer on the identity -- a layer
    of two-qubit depth bought and thrown away. The midpoint rule is also the more
    accurate quadrature of the same integral, so it costs nothing to prefer it.

    Args:
        depth: Number of layers. Zero returns an empty vector.
        ramp_time: The total adiabatic time :math:`T`. Larger means a slower, more
            faithful ramp and correspondingly larger angles.

    Returns:
        The flat parameter vector, packed as :func:`join_angles` packs it.

    Raises:
        ValueError: If ``depth`` is negative or ``ramp_time`` is not positive.

    Examples:
        >>> angles = adiabatic_ramp(depth=2, ramp_time=2.0)
        >>> gamma, beta = split_angles(angles)
        >>> bool(gamma[0] < gamma[1]), bool(beta[0] > beta[1])
        (True, True)
    """
    if depth < 0:
        raise ValueError(f"depth cannot be negative, got {depth}")
    if ramp_time <= 0.0:
        raise ValueError(f"ramp time must be positive, got {ramp_time}")
    if depth == 0:
        return np.zeros(0, dtype=np.float64)
    step = ramp_time / depth
    fraction = (np.arange(1, depth + 1, dtype=np.float64) - 0.5) / depth
    return join_angles(step * fraction, step * (1.0 - fraction))


def small_angle(depth: int, scale: float = SMALL_ANGLE_SCALE, seed: int = 0) -> FloatArray:
    """Initial angles just off the identity, where the landscape still has curvature.

    The standard first defence against a barren plateau. A circuit initialised at random
    over the whole parameter range behaves like a Haar-random state, whose gradient
    vanishes exponentially in the qubit count; a circuit initialised near the identity
    does not, because it has not had the depth to scramble.

    Args:
        depth: Number of layers.
        scale: Magnitude of the perturbation. The angles are drawn uniformly from
            ``[-scale, scale]``.
        seed: Seed for the generator. Explicit and required-by-default because an
            initialisation that cannot be reproduced makes the run that followed it
            unreproducible too.

    Returns:
        The flat parameter vector.

    Raises:
        ValueError: If ``depth`` is negative or ``scale`` is not positive.
    """
    if depth < 0:
        raise ValueError(f"depth cannot be negative, got {depth}")
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    generator = np.random.default_rng(seed)
    return generator.uniform(-scale, scale, size=2 * depth)


def initial_angles(
    depth: int,
    strategy: Initialisation = "adiabatic_ramp",
    ramp_time: float = DEFAULT_RAMP_TIME,
    scale: float = SMALL_ANGLE_SCALE,
    seed: int = 0,
) -> FloatArray:
    """Produce a starting parameter vector by the named strategy.

    One entry point so that the choice of initialisation is a *value* the agent selects
    and the result carries, rather than a branch hidden in a driver. Which strategy was
    used explains most of the difference between two runs that otherwise look identical.

    Args:
        depth: Number of layers.
        strategy: ``"adiabatic_ramp"`` for the Trotterised ramp of
            :func:`adiabatic_ramp`, ``"small_angle"`` for a near-identity start, or
            ``"zeros"`` for the exact identity -- which is a legitimate choice only when
            something downstream will move it, since the gradient there is not always
            informative.
        ramp_time: Passed to :func:`adiabatic_ramp`.
        scale: Passed to :func:`small_angle`.
        seed: Passed to :func:`small_angle`.

    Returns:
        The flat parameter vector.

    Raises:
        ValueError: If the strategy is not one of the three named.
    """
    if strategy == "adiabatic_ramp":
        return adiabatic_ramp(depth, ramp_time)
    if strategy == "small_angle":
        return small_angle(depth, scale, seed)
    if strategy == "zeros":
        return np.zeros(2 * max(depth, 0), dtype=np.float64)
    raise ValueError(f"unknown initialisation strategy {strategy!r}")


def interpolate_to_depth(theta: FloatArray, new_depth: int) -> FloatArray:
    """Resample a converged schedule onto a different number of layers.

    The INTERP warm start: having optimised at depth :math:`p`, initialise depth
    :math:`p+1` by stretching the converged schedule over the finer grid rather than
    starting again. It inherits a good basin instead of searching for one, and it is
    what makes a depth sweep cost roughly one optimisation rather than :math:`p` of them.

    Linear interpolation is used because the schedules being interpolated are smooth
    ramps by construction; a spline would add wiggle the physics does not have.

    Args:
        theta: The converged flat parameter vector, of even length.
        new_depth: The depth to resample onto. May be larger or smaller.

    Returns:
        A flat parameter vector of length ``2 * new_depth``.

    Raises:
        ValueError: If ``new_depth`` is negative, or if ``theta`` is empty while a
            non-zero depth is requested -- there is nothing to interpolate from.

    Examples:
        >>> import numpy as np
        >>> grown = interpolate_to_depth(adiabatic_ramp(2), 4)
        >>> grown.size
        8
    """
    if new_depth < 0:
        raise ValueError(f"depth cannot be negative, got {new_depth}")
    if new_depth == 0:
        return np.zeros(0, dtype=np.float64)
    gamma, beta = split_angles(theta)
    if gamma.size == 0:
        raise ValueError("cannot interpolate an empty schedule onto a non-zero depth")
    if gamma.size == 1:
        return join_angles(np.full(new_depth, gamma[0]), np.full(new_depth, beta[0]))
    source = np.linspace(0.0, 1.0, gamma.size)
    target = np.linspace(0.0, 1.0, new_depth)
    return join_angles(np.interp(target, source, gamma), np.interp(target, source, beta))


def angle_periods(
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
) -> tuple[float | None, float | None]:
    r"""How far each schedule can be shifted without changing the state it prepares.

    The period is set by the generator's spectrum rather than by the turn of a
    circle. The diagonal layer applies :math:`e^{-i\gamma \hat H_\text{diag}}`, whose
    eigenvalues are :math:`J(2u - B)` over integers :math:`u` for a lattice of
    :math:`B` bonds, so consecutive eigenvalues differ by :math:`2J` and shifting
    :math:`\gamma` by :math:`\tau` multiplies each amplitude by
    :math:`e^{-2i\tau J u}` times a global phase. That is the identity for every
    :math:`u` only when :math:`2\tau J` is a whole number of turns, giving a period
    of :math:`\pi / J` rather than :math:`2\pi`. The field layer is the same argument
    with :math:`2h` in place of :math:`2J`.

    So :math:`2\pi` is a safe fold only when :math:`2J` and :math:`2h` happen to be
    whole numbers, which is true at the calibration point :math:`J = h = 1` and
    false at most of the values the interface's sliders offer. Folding by it
    elsewhere reports angles that do not reproduce the energy printed beside them.

    Args:
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis.

    Returns:
        ``(gamma_period, beta_period)``. Either is ``None`` when no finite period
        exists, and ``None`` means *do not fold* rather than *fold by the default*.
        With both :math:`J` and :math:`g` non-zero the diagonal generator has two
        incommensurate scales in it and a common period exists only when
        :math:`g / J` is rational; rather than test that on floating-point input,
        this returns ``None`` and the angles are reported as the optimiser left
        them. An unfolded angle is merely untidy, whereas a wrongly folded one is
        incorrect.

    Examples:
        >>> angle_periods(1.0, 1.0)
        (3.141592653589793, 3.141592653589793)
        >>> gamma, beta = angle_periods(0.7, 0.85)
        >>> round(gamma, 6), round(beta, 6)
        (4.48799, 3.695991)
        >>> angle_periods(1.0, 1.0, longitudinal_field=0.4)
        (None, 3.141592653589793)
    """
    if longitudinal_field == 0.0:
        gamma = np.pi / abs(coupling) if coupling != 0.0 else None
    elif coupling == 0.0:
        gamma = np.pi / abs(longitudinal_field)
    else:
        gamma = None
    beta = np.pi / abs(transverse_field) if transverse_field != 0.0 else None
    return gamma, beta


def wrap_angles(
    theta: FloatArray,
    gamma_period: float | None = None,
    beta_period: float | None = None,
) -> FloatArray:
    r"""Fold each schedule into one period of its own generator.

    Two parameter vectors that describe the identical state should compare equal.
    Without folding they do not: the schedules are periodic, so an optimiser that
    drifts by a whole period produces a vector that looks far from where it started,
    a convergence test that never fires, and a reported solution that is needlessly
    hard to read.

    The periods must be supplied because they depend on :math:`J`, :math:`g` and
    :math:`h`, which a parameter vector does not carry -- see :func:`angle_periods`
    for the derivation and for why assuming :math:`2\pi` is wrong. Omitting one
    leaves that half untouched, so the failure mode of a caller that does not know
    its couplings is an untidy answer rather than an incorrect one.

    Args:
        theta: Any parameter vector, :math:`\gamma` then :math:`\beta`.
        gamma_period: Period of the diagonal schedule, or ``None`` to leave it be.
        beta_period: Period of the field schedule, or ``None`` to leave it be.

    Returns:
        The angles, each half folded into ``[-tau/2, tau/2)`` for its own period.

    Examples:
        >>> import numpy as np
        >>> folded = wrap_angles(np.array([4.0, 1.0]), *angle_periods(1.0, 1.0))
        >>> np.round(folded, 6)
        array([0.858407, 1.      ])

        With no periods given nothing moves, which is the honest default:

        >>> wrap_angles(np.array([4.0, 1.0]))
        array([4., 1.])
    """
    gamma, beta = split_angles(theta)
    return join_angles(_fold(gamma, gamma_period), _fold(beta, beta_period))


def _fold(angles: FloatArray, period: float | None) -> FloatArray:
    """Fold one schedule into ``[-period/2, period/2)``, or leave it alone.

    Args:
        angles: One half of a parameter vector.
        period: Its period, or ``None`` when none is known.

    Returns:
        The folded angles, as a new array.
    """
    if period is None or period <= 0.0:
        return np.asarray(angles, dtype=np.float64)
    half = 0.5 * period
    return np.asarray((angles + half) % period - half, dtype=np.float64)
