r"""Evolve the ansatz exactly, with no matrix exponential and no Trotter error.

A general simulator would build a :math:`2^L \times 2^L` operator per layer and
multiply. This one never forms a matrix, because splitting the Hamiltonian into

.. math::

    \hat H_\text{diag} = -J \sum_i \hat\sigma^z_i \hat\sigma^z_{i+1} - g \sum_i \hat\sigma^z_i ,
    \qquad
    \hat H_\text{field} = -h \sum_i \hat\sigma^x_i ,

leaves every term inside each half commuting with every other term in that half.
Each layer\'s exponential then factorises exactly: the diagonal half is a phase
applied elementwise, the field half a product of independent single-qubit
rotations. Both are equalities, so the only error here is floating-point.

That the two halves do not commute with each other is the whole difficulty of the
problem, and why the alternation of layers is optimised rather than solved.

Three reasons this exists rather than a call into Qiskit. The gradient in
:func:`energy_and_gradient` uses the adjoint method, which must reach inside the
state between layers, and a circuit-level API does not offer that. The cost is
:math:`O(pL2^L)` either way but with no gate-object allocation per bond. And
independence from Qiskit is what lets the simulator test check this
against Qiskit -- two implementations sharing no code.

The ceiling is :data:`~src.physics.quantum.ansatz.MAX_SIMULABLE_QUBITS`, where
the state vector stops fitting in memory. That is the simulator\'s limit, not the
algorithm\'s: a device would run these circuits in time linear in the qubit count.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.physics.lattice import Lattice
from src.physics.model import BoundaryCondition
from src.physics.quantum.ansatz import AnsatzSpec, split_angles
from src.physics.quantum.hamiltonians import chain_bonds

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

NORM_TOLERANCE = 1e-10
"""How far the state norm may drift from one before something is wrong.

Every gate here is unitary by construction, so drift is not accumulation of rounding --
it is a reshape applied to the wrong axis, or an angle applied twice. The check is cheap
and it localises such a bug to the layer that caused it rather than to the energy that
finally looked odd.
"""


def diagonal_energies(
    n_sites: int,
    coupling: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: BoundaryCondition = "open",
    lattice: Lattice | None = None,
) -> FloatArray:
    r"""The diagonal of :math:`\hat H_\text{diag}`, as a vector over basis states.

    Computed by bit arithmetic over all :math:`2^L` basis indices at once rather than by
    assembling a sparse matrix and reading its diagonal. The two routes share no algebra,
    which is what makes the comparison in the simulator test worth running.

    Convention: qubit :math:`i` is bit :math:`i` of the basis index, and bit ``0`` is the
    :math:`\hat\sigma^z = +1` eigenstate. So the spin value is :math:`s_i = 1 - 2 b_i`.
    This matches :mod:`src.physics.quantum.hamiltonians` and is the *opposite* of the
    labelling in the sealed exact-diagonalisation module -- the two differ by a global
    bit flip, and the tests compare through that permutation rather than assuming it away.

    Args:
        n_sites: Number of sites, equal to the number of qubits.
        coupling: The Ising coupling :math:`J`.
        longitudinal_field: The field :math:`g` along the coupling axis. Zero by default,
            which is the integrable case.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring. Ignored when
            ``lattice`` is given, which carries its own.
        lattice: The shape the sites sit on, or ``None`` for a chain. Only the bond
            list changes: the bit arithmetic above is a statement about spins and
            knows nothing about how they are arranged, which is why a lattice costs
            one argument here and not a second implementation. This is the argument
            that makes the *variational* solvers two-dimensional -- everything they
            evolve is a phase read out of this vector.

    Returns:
        A real vector of length :math:`2^L`, one energy per computational basis state.

    Raises:
        ValueError: If ``n_sites`` is below two, or a lattice was given whose site
            count is not ``n_sites``.

    Examples:
        Two sites, ferromagnetic, no field: the aligned states sit at :math:`-J`.

        >>> diagonal_energies(2, coupling=1.0)
        array([-1.,  1.,  1., -1.])

        A two-by-two square is a four-cycle, so each of its four states with two
        spins up sits two bonds higher than the aligned ones:

        >>> from src.physics.lattice import Lattice
        >>> diagonal_energies(4, lattice=Lattice("square", 2, 2))[[0, 3]]
        array([-4.,  0.])
    """
    if n_sites < 2:
        raise ValueError(f"a chain needs at least 2 sites, got {n_sites}")
    if lattice is not None and lattice.n_sites != n_sites:
        raise ValueError(f"the lattice holds {lattice.n_sites} sites but n_sites is {n_sites}")
    index = np.arange(2**n_sites, dtype=np.int64)
    spins = np.empty((n_sites, index.size), dtype=np.float64)
    for site in range(n_sites):
        spins[site] = 1.0 - 2.0 * ((index >> site) & 1)
    energies = np.zeros(index.size, dtype=np.float64)
    bonds = lattice.bonds() if lattice is not None else chain_bonds(n_sites, boundary)
    for left, right in bonds:
        energies -= coupling * spins[left] * spins[right]
    if longitudinal_field != 0.0:
        energies -= longitudinal_field * spins.sum(axis=0)
    return energies


def uniform_superposition(n_qubits: int) -> ComplexArray:
    r"""The state every run starts from: :math:`\lvert + \rangle^{\otimes L}`.

    It is the exact ground state of :math:`\hat H_\text{field}` on its own for
    :math:`h > 0`, so the ansatz begins already correct in the limit where the transverse
    field dominates, and one layer of Hadamards prepares it on any device.

    Args:
        n_qubits: Width of the register.

    Returns:
        A normalised complex state vector with every amplitude equal.
    """
    dimension = 2**n_qubits
    return np.full(dimension, 1.0 / np.sqrt(dimension), dtype=np.complex128)


def apply_diagonal_layer(state: ComplexArray, angle: float, diagonal: FloatArray) -> ComplexArray:
    r"""Apply :math:`e^{-i \gamma \hat H_\text{diag}}` exactly.

    The generator is diagonal, so its exponential is too, and the gate is one elementwise
    multiply -- :math:`O(2^L)` with no temporaries beyond the phase vector.

    Args:
        state: The state to evolve. Not modified.
        angle: The angle :math:`\gamma`.
        diagonal: The generator's diagonal, from :func:`diagonal_energies`.

    Returns:
        The evolved state.
    """
    return np.asarray(np.exp(-1j * angle * diagonal) * state, dtype=np.complex128)


def apply_field_layer(state: ComplexArray, angle: float, transverse_field: float) -> ComplexArray:
    r"""Apply :math:`e^{-i \beta \hat H_\text{field}}` exactly, one qubit at a time.

    The generator is a sum of single-qubit terms that commute, so

    .. math::

        e^{-i\beta \hat H_\text{field}}
        = \prod_j e^{+ i \beta h \hat\sigma^x_j}
        = \prod_j \left[ \cos(\beta h)\,\hat I + i \sin(\beta h)\, \hat\sigma^x_j \right] ,

    and each factor is applied by viewing the state as ``(low, 2, high)`` and mixing the
    middle axis. No :math:`2 \times 2` matrix is ever multiplied against a
    :math:`2^L`-dimensional operator.

    Args:
        state: The state to evolve. Not modified.
        angle: The angle :math:`\beta`.
        transverse_field: The field strength :math:`h`.

    Returns:
        The evolved state.
    """
    n_qubits = int(np.log2(state.size))
    cosine = np.cos(angle * transverse_field)
    sine = 1j * np.sin(angle * transverse_field)
    evolved = state
    for qubit in range(n_qubits):
        block = 2**qubit
        view = evolved.reshape(-1, 2, block)
        lower, upper = view[:, 0, :], view[:, 1, :]
        evolved = np.stack([cosine * lower + sine * upper, cosine * upper + sine * lower], axis=1)
        evolved = evolved.reshape(-1)
    return np.asarray(evolved, dtype=np.complex128)


def evolve(
    theta: FloatArray,
    spec: AnsatzSpec,
    diagonal: FloatArray,
    transverse_field: float,
) -> ComplexArray:
    r"""Run the whole circuit and return the prepared state.

    Args:
        theta: The flat parameter vector, packed as
            :func:`~src.physics.quantum.ansatz.join_angles` packs it.
        spec: The ansatz being run. Only its depth and width are used here; the gate
            counts it also carries are for pricing, not for evolution.
        diagonal: The diagonal of :math:`\hat H_\text{diag}`, from
            :func:`diagonal_energies`.
        transverse_field: The field strength :math:`h`.

    Returns:
        The normalised state :math:`\lvert \psi(\theta) \rangle`.

    Raises:
        ValueError: If ``theta`` does not carry exactly two angles per layer, or if the
            diagonal is not as wide as the register.

    Examples:
        At depth zero the circuit is the bare initial state:

        >>> import numpy as np
        >>> spec = AnsatzSpec(n_qubits=2, depth=0)
        >>> state = evolve(np.zeros(0), spec, diagonal_energies(2), 1.0)
        >>> bool(np.allclose(state, uniform_superposition(2)))
        True
    """
    if theta.size != spec.n_parameters:
        raise ValueError(f"expected {spec.n_parameters} angles for this ansatz, got {theta.size}")
    if diagonal.size != spec.state_dimension:
        raise ValueError(
            f"diagonal has {diagonal.size} entries but the register needs {spec.state_dimension}"
        )
    gamma, beta = split_angles(theta)
    state = uniform_superposition(spec.n_qubits)
    for layer in range(spec.depth):
        state = apply_diagonal_layer(state, float(gamma[layer]), diagonal)
        state = apply_field_layer(state, float(beta[layer]), transverse_field)
    return state


def apply_hamiltonian(
    state: ComplexArray, diagonal: FloatArray, transverse_field: float
) -> ComplexArray:
    r"""Apply :math:`\hat H` itself to a state, without forming it as a matrix.

    Needed twice: to compute an energy as :math:`\langle \psi \vert \hat H \psi \rangle`,
    and to seed the adjoint gradient's backward pass. The diagonal half is a multiply;
    the field half reuses the same reshape trick as :func:`apply_field_layer`, but adds
    the flipped amplitude rather than mixing it with a rotation.

    Args:
        state: The state to act on. Not modified.
        diagonal: The diagonal of :math:`\hat H_\text{diag}`.
        transverse_field: The field strength :math:`h`.

    Returns:
        :math:`\hat H \lvert \psi \rangle`, generally not normalised.
    """
    result = diagonal * state
    n_qubits = int(np.log2(state.size))
    for qubit in range(n_qubits):
        block = 2**qubit
        view = state.reshape(-1, 2, block)
        flipped = np.stack([view[:, 1, :], view[:, 0, :]], axis=1).reshape(-1)
        result = result - transverse_field * flipped
    return np.asarray(result, dtype=np.complex128)


def energy(state: ComplexArray, diagonal: FloatArray, transverse_field: float) -> float:
    r"""The exact expectation :math:`\langle \psi \vert \hat H \vert \psi \rangle`.

    This is what a *simulator* can report and a real device cannot: no sampling, no error
    bar, no shot budget. Nothing in this project samples an expectation; what a device
    would have to spend to resolve one is priced instead, by
    :func:`src.hardware.fidelity.shot_inflation` and the shot ledger.

    Args:
        state: A normalised state.
        diagonal: The diagonal of :math:`\hat H_\text{diag}`.
        transverse_field: The field strength :math:`h`.

    Returns:
        The energy. Real by construction -- :math:`\hat H` is Hermitian -- and the
        imaginary part is discarded rather than checked here, because
        :func:`energy_and_gradient` is where a violation would actually mean something.
    """
    return float(np.vdot(state, apply_hamiltonian(state, diagonal, transverse_field)).real)


def energy_and_gradient(
    theta: FloatArray,
    spec: AnsatzSpec,
    diagonal: FloatArray,
    transverse_field: float,
) -> tuple[float, FloatArray]:
    r"""The energy and its exact gradient, in two sweeps rather than :math:`2P` circuits.

    The **adjoint method**. Writing the circuit as
    :math:`\lvert \psi \rangle = \hat U_M \cdots \hat U_1 \lvert \psi_0 \rangle` with
    :math:`\hat G_k` the generator of gate :math:`k`,

    .. math::

        \frac{\partial E}{\partial \theta_k}
        = 2\,\mathrm{Re}\,\langle \lambda_k \vert (-i \hat G_k) \vert \varphi_k \rangle ,

    where :math:`\lvert \varphi_k \rangle` is the state after gate :math:`k` and
    :math:`\langle \lambda_k \vert = \langle \psi \vert \hat H \hat U_M \cdots \hat
    U_{k+1}`. Both obey the same backward recursion, so one forward pass and one backward
    pass produce the whole gradient -- about the cost of two energy evaluations, exact to
    machine precision.

    A device cannot do this, because the backward pass reads the state between layers and
    measurement destroys it. Hardware differentiates by the parameter-shift rule, which is
    two circuits per *Pauli rotation* -- and one angle here drives every gate in its layer
    at once, so a layer costs two circuits per gate sharing the angle rather than two in
    total. Shifting a layer angle as though it were a single rotation is not the rule and
    does not give the gradient. That price is what the agent quotes; this function is how
    the simulation gets its work done.

    Args:
        theta: The flat parameter vector.
        spec: The ansatz being run.
        diagonal: The diagonal of :math:`\hat H_\text{diag}`.
        transverse_field: The field strength :math:`h`.

    Returns:
        The energy, and the gradient packed in the same order as ``theta``.

    Raises:
        ValueError: If the state norm drifts by more than :data:`NORM_TOLERANCE`, which
            can only mean a gate was applied incorrectly.
    """
    gamma, beta = split_angles(theta)
    forward: list[ComplexArray] = [uniform_superposition(spec.n_qubits)]
    for layer in range(spec.depth):
        after_diagonal = apply_diagonal_layer(forward[-1], float(gamma[layer]), diagonal)
        forward.append(after_diagonal)
        forward.append(apply_field_layer(after_diagonal, float(beta[layer]), transverse_field))

    state = forward[-1]
    drift = abs(float(np.vdot(state, state).real) - 1.0)
    if drift > NORM_TOLERANCE:
        raise ValueError(f"state norm drifted by {drift:.2e}; a layer was applied incorrectly")

    total = energy(state, diagonal, transverse_field)
    gradient_gamma = np.zeros(spec.depth, dtype=np.float64)
    gradient_beta = np.zeros(spec.depth, dtype=np.float64)
    adjoint = apply_hamiltonian(state, diagonal, transverse_field)

    for layer in reversed(range(spec.depth)):
        before_field = forward[2 * layer + 1]
        after_field = forward[2 * layer + 2]
        field_generator = -transverse_field * sigma_x_sum(after_field)
        gradient_beta[layer] = 2.0 * float(np.vdot(adjoint, -1j * field_generator).real)
        adjoint = apply_field_layer(adjoint, -float(beta[layer]), transverse_field)

        gradient_gamma[layer] = 2.0 * float(np.vdot(adjoint, -1j * diagonal * before_field).real)
        adjoint = apply_diagonal_layer(adjoint, -float(gamma[layer]), diagonal)

    return total, np.concatenate([gradient_gamma, gradient_beta])


def sigma_x_sum(state: ComplexArray) -> ComplexArray:
    r"""Apply :math:`\sum_j \hat\sigma^x_j` to a state.

    Factored out because it is the field half's *generator*, which the adjoint sweep
    needs on its own -- :func:`apply_hamiltonian` bundles it with the diagonal half and
    with the coupling constant, and the gradient needs it bare.

    Public rather than private because the same generator is what
    :mod:`src.physics.quantum.imaginary_time_evolution` differentiates the circuit with:
    a second copy of this loop in another module would be a second place for the field
    half's sign convention to be got wrong, and the two would then disagree silently.

    Args:
        state: The state to act on. Not modified.

    Returns:
        The summed bit-flip of the state.
    """
    n_qubits = int(np.log2(state.size))
    result = np.zeros_like(state)
    for qubit in range(n_qubits):
        block = 2**qubit
        view = state.reshape(-1, 2, block)
        result = result + np.stack([view[:, 1, :], view[:, 0, :]], axis=1).reshape(-1)
    return np.asarray(result, dtype=np.complex128)


def measurement_deviation(
    state: ComplexArray, diagonal: FloatArray, transverse_field: float
) -> float:
    r"""Return :math:`\sum_g \sigma_g` over the two measurement settings.

    The variance-aware counterpart of
    :meth:`~src.physics.quantum.hamiltonians.PauliSum.coefficient_l1`. Both are the
    numerator of the same shot count :math:`(\cdot / \epsilon)^2`: the norm charges
    every Pauli the full variance its :math:`\pm 1` spectrum allows, while this
    charges the variance the prepared state actually carries. Quoting the two
    together says how loose the price a plan was approved on turned out to be.

    Evaluable only by a simulator, and only after a run -- :math:`\sigma_g` is a
    property of a state the planner has not prepared yet, which is why a budget is
    priced from the bound instead.

    Args:
        state: A normalised state.
        diagonal: The diagonal of :math:`\hat H_\text{diag}`. Every term in it is
            diagonal in the same basis, so the whole half is one setting.
        transverse_field: The field strength :math:`h`, whose terms are the second
            setting.

    Returns:
        The two settings' standard deviations, added.

    Examples:
        On a :math:`\hat\sigma^z` product state the diagonal half is sharp and the
        field half contributes :math:`h\sqrt{N}`:

        >>> import numpy as np
        >>> state = np.zeros(4, dtype=np.complex128)
        >>> state[0] = 1.0
        >>> round(measurement_deviation(state, diagonal_energies(2), 1.0), 6)
        1.414214
    """
    weights = np.abs(state) ** 2
    diagonal_mean = float(weights @ diagonal)
    diagonal_variance = float(weights @ diagonal**2) - diagonal_mean**2
    flipped = sigma_x_sum(state)
    field_mean = -transverse_field * float(np.vdot(state, flipped).real)
    field_variance = transverse_field**2 * float(np.vdot(flipped, flipped).real) - field_mean**2
    return float(np.sqrt(max(0.0, diagonal_variance)) + np.sqrt(max(0.0, field_variance)))
