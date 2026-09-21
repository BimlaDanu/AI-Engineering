"""Tests for the exact simulator, each against algebra it does not share.

The module claims its layers are *exact*, not Trotterised. That is a strong claim and it
is the one under test here: every layer is held against a dense matrix exponential built
by a different route, the energy against a sparse ``matvec``, and the adjoint gradient
against two independent differentiation methods.

The house rule is that agreement between two routes is evidence and a passing
self-consistency check is not, so nothing here compares the simulator with itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from src.physics.lattice import Lattice
from src.physics.quantum.ansatz import AnsatzSpec, adiabatic_ramp, small_angle, split_angles
from src.physics.quantum.hamiltonians import ising_chain
from src.physics.quantum.statevector import (
    apply_diagonal_layer,
    apply_field_layer,
    apply_hamiltonian,
    diagonal_energies,
    energy,
    energy_and_gradient,
    evolve,
    measurement_deviation,
    uniform_superposition,
)
from src.physics.quantum.variational_eigensolver import solve

SIZES = [2, 3, 4, 6]
FIELDS = [0.3, 1.0, 2.5]


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _random_state(n_sites: int, seed: int) -> np.ndarray:
    """A normalised random complex state.

    Layers are checked on a *generic* state rather than on the uniform superposition,
    because the superposition is an eigenstate of the field half and would hide an error
    in exactly the axis being tested.
    """
    generator = np.random.default_rng(seed)
    state = generator.normal(size=2**n_sites) + 1j * generator.normal(size=2**n_sites)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)


def _dense(
    n_sites: int, coupling: float, transverse: float, longitudinal: float, boundary: str
) -> np.ndarray:
    """The Hamiltonian as a dense array, by the sparse Kronecker route."""
    return ising_chain(n_sites, coupling, transverse, longitudinal, boundary).to_matrix().toarray()  # type: ignore[arg-type]


def _dense_field(n_sites: int, transverse: float) -> np.ndarray:
    """The field half alone, assembled by hand.

    Built here rather than by asking :func:`ising_chain` for a chain with zero coupling,
    which it rightly refuses -- a chain with no coupling is not an Ising chain. Writing
    the Kronecker sum out also makes this an independent route rather than a second call
    into the code under test.
    """
    total = np.zeros((2**n_sites, 2**n_sites), dtype=np.complex128)
    for site in range(n_sites):
        operator: np.ndarray = np.array([[1.0]], dtype=np.complex128)
        for other in range(n_sites):
            # Qubit `site` is bit `site` of the index, so it is the *last* Kronecker
            # factor -- little-endian, matching the simulator's convention.
            operator = np.kron(SIGMA_X if other == site else np.eye(2), operator)
        total -= transverse * operator
    return total


# --------------------------------------------------------------------------
# The diagonal — bit arithmetic against a Kronecker assembly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("boundary", ["open", "periodic"])
@pytest.mark.parametrize("longitudinal", [0.0, 0.4])
def test_the_diagonal_matches_the_sparse_construction(
    n_sites: int, boundary: str, longitudinal: float
) -> None:
    # Two routes with nothing in common: a loop over bit patterns against a Kronecker
    # product of 2x2 matrices summed into a sparse array.
    ours = diagonal_energies(n_sites, 1.0, longitudinal, boundary)  # type: ignore[arg-type]
    theirs = _dense(n_sites, 1.0, 0.0, longitudinal, boundary).diagonal().real
    assert np.allclose(ours, theirs)


def test_the_aligned_states_of_a_two_site_chain_sit_at_minus_j() -> None:
    # Closed form, no code on the other side: the two ferromagnetic states are degenerate
    # at -J and the two anti-aligned states at +J.
    assert np.allclose(diagonal_energies(2, coupling=1.0), [-1.0, 1.0, 1.0, -1.0])


def test_a_chain_shorter_than_two_sites_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2 sites"):
        diagonal_energies(1)


# --------------------------------------------------------------------------
# The layers — factorised gates against a matrix exponential
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("angle", [0.0, 0.31, 1.7, -0.8])
def test_the_diagonal_layer_equals_a_matrix_exponential(n_sites: int, angle: float) -> None:
    diagonal = diagonal_energies(n_sites, 1.0, 0.3, "open")
    state = uniform_superposition(n_sites)
    theirs = expm(-1j * angle * np.diag(diagonal)) @ state
    assert np.allclose(apply_diagonal_layer(state, angle, diagonal), theirs)


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("angle", [0.0, 0.31, 1.7, -0.8])
@pytest.mark.parametrize("transverse", FIELDS)
def test_the_field_layer_equals_a_matrix_exponential(
    n_sites: int, angle: float, transverse: float
) -> None:
    # The claim being checked is that the product of independent single-qubit rotations
    # really is the exponential of the summed generator -- true only because the terms
    # commute, which is exactly the structural fact the module is built on.
    generator = _dense_field(n_sites, transverse)
    state = _random_state(n_sites, seed=3)
    theirs = expm(-1j * angle * generator) @ state
    assert np.allclose(apply_field_layer(state, angle, transverse), theirs)


@pytest.mark.parametrize("n_sites", SIZES)
def test_every_layer_preserves_the_norm(n_sites: int) -> None:
    diagonal = diagonal_energies(n_sites, 1.0, 0.0, "open")
    state = uniform_superposition(n_sites)
    state = apply_diagonal_layer(state, 0.7, diagonal)
    state = apply_field_layer(state, -1.3, 0.9)
    assert float(np.vdot(state, state).real) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# The full circuit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
def test_a_depth_zero_circuit_is_the_bare_initial_state(n_sites: int) -> None:
    spec = AnsatzSpec(n_qubits=n_sites, depth=0)
    state = evolve(np.zeros(0), spec, diagonal_energies(n_sites), 1.0)
    assert np.allclose(state, uniform_superposition(n_sites))


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("transverse", FIELDS)
def test_the_bare_initial_state_has_the_closed_form_energy(n_sites: int, transverse: float) -> None:
    # The cheapest end-to-end check in the project, and it needs no second implementation:
    # |+> is the exact ground state of the field half alone, so E = -h L exactly.
    spec = AnsatzSpec(n_qubits=n_sites, depth=0)
    diagonal = diagonal_energies(n_sites, 1.0, 0.0, "open")
    state = evolve(np.zeros(0), spec, diagonal, transverse)
    assert energy(state, diagonal, transverse) == pytest.approx(-transverse * n_sites)


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_whole_circuit_equals_a_product_of_matrix_exponentials(
    n_sites: int, depth: int
) -> None:
    # The claim that the layers compose exactly, checked layer by layer against dense
    # exponentials of the same two generators.
    transverse, longitudinal = 0.9, 0.25
    spec = AnsatzSpec(n_qubits=n_sites, depth=depth)
    diagonal = diagonal_energies(n_sites, 1.0, longitudinal, "open")
    theta = adiabatic_ramp(depth, ramp_time=1.5)
    gamma, beta = split_angles(theta)

    field = _dense_field(n_sites, transverse)
    theirs = uniform_superposition(n_sites)
    for layer in range(depth):
        theirs = expm(-1j * gamma[layer] * np.diag(diagonal)) @ theirs
        theirs = expm(-1j * beta[layer] * field) @ theirs

    assert np.allclose(evolve(theta, spec, diagonal, transverse), theirs)


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("longitudinal", [0.0, 0.4])
def test_the_energy_matches_a_dense_matrix_product(n_sites: int, longitudinal: float) -> None:
    # Term-by-term expectation against one matvec on the assembled operator.
    transverse = 1.1
    spec = AnsatzSpec(n_qubits=n_sites, depth=2)
    diagonal = diagonal_energies(n_sites, 1.0, longitudinal, "open")
    state = evolve(adiabatic_ramp(2), spec, diagonal, transverse)
    dense = _dense(n_sites, 1.0, transverse, longitudinal, "open")
    assert energy(state, diagonal, transverse) == pytest.approx(
        float((state.conj() @ (dense @ state)).real), abs=1e-12
    )


@pytest.mark.parametrize("n_sites", SIZES)
def test_applying_the_hamiltonian_matches_the_dense_operator(n_sites: int) -> None:
    transverse, longitudinal = 0.7, 0.2
    diagonal = diagonal_energies(n_sites, 1.0, longitudinal, "open")
    state = evolve(adiabatic_ramp(2), AnsatzSpec(n_qubits=n_sites, depth=2), diagonal, transverse)
    dense = _dense(n_sites, 1.0, transverse, longitudinal, "open")
    assert np.allclose(apply_hamiltonian(state, diagonal, transverse), dense @ state)


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("longitudinal", [0.0, 0.4])
def test_the_measurement_deviation_matches_the_grouped_operators(
    n_sites: int, longitudinal: float
) -> None:
    # The state-vector route against assembled matrices, one per measurement setting.
    transverse = 1.1
    diagonal = diagonal_energies(n_sites, 1.0, longitudinal, "open")
    state = evolve(adiabatic_ramp(2), AnsatzSpec(n_qubits=n_sites, depth=2), diagonal, transverse)
    chain = ising_chain(
        n_sites=n_sites,
        coupling=1.0,
        transverse_field=transverse,
        longitudinal_field=longitudinal,
    )
    theirs = 0.0
    for group in chain.measurement_groups():
        matrix = group.operator.to_matrix().toarray()
        mean = float((state.conj() @ (matrix @ state)).real)
        second = float((state.conj() @ (matrix @ (matrix @ state))).real)
        theirs += float(np.sqrt(second - mean**2))

    assert measurement_deviation(state, diagonal, transverse) == pytest.approx(theirs, abs=1e-10)


def test_a_prepared_ground_state_asks_far_fewer_shots_than_the_coefficient_bound() -> None:
    # The claim the post-run figure rests on, and the route the campaign takes to it:
    # the state is rebuilt from the angles the solver returned, not carried out of it.
    result = solve(n_sites=8, depth=3, coupling=1.0, transverse_field=1.0, family="hva")
    diagonal = diagonal_energies(8, 1.0)
    state = evolve(result.parameters, result.spec, diagonal, 1.0)
    bound = ising_chain(n_sites=8, coupling=1.0, transverse_field=1.0).coefficient_l1()

    assert energy(state, diagonal, 1.0) == pytest.approx(result.energy, abs=1e-10)
    deviation = measurement_deviation(state, diagonal, 1.0)
    assert deviation < bound
    # Shots go as the square, so the worst case over-prices by more than fivefold.
    assert (bound / deviation) ** 2 > 5.0


def test_a_parameter_vector_of_the_wrong_length_is_refused() -> None:
    spec = AnsatzSpec(n_qubits=4, depth=2)
    with pytest.raises(ValueError, match="expected 4 angles"):
        evolve(np.zeros(6), spec, diagonal_energies(4), 1.0)


def test_a_diagonal_of_the_wrong_width_is_refused() -> None:
    spec = AnsatzSpec(n_qubits=4, depth=1)
    with pytest.raises(ValueError, match="diagonal has"):
        evolve(np.zeros(2), spec, diagonal_energies(3), 1.0)


# --------------------------------------------------------------------------
# The gradient — the adjoint sweep against two independent routes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_adjoint_gradient_matches_central_finite_differences(n_sites: int, depth: int) -> None:
    # Numerical differentiation shares no algebra with the adjoint recursion: it never
    # touches an intermediate state and never applies an inverse gate.
    transverse, longitudinal = 0.85, 0.3
    spec = AnsatzSpec(n_qubits=n_sites, depth=depth)
    diagonal = diagonal_energies(n_sites, 1.0, longitudinal, "open")
    theta = adiabatic_ramp(depth, ramp_time=1.5)

    _, ours = energy_and_gradient(theta, spec, diagonal, transverse)
    step = 1e-6
    theirs = np.zeros_like(theta)
    for index in range(theta.size):
        forward, backward = theta.copy(), theta.copy()
        forward[index] += step
        backward[index] -= step
        theirs[index] = (
            energy(evolve(forward, spec, diagonal, transverse), diagonal, transverse)
            - energy(evolve(backward, spec, diagonal, transverse), diagonal, transverse)
        ) / (2.0 * step)
    assert np.allclose(ours, theirs, atol=1e-7)


@pytest.mark.parametrize("n_sites", [3, 4])
@pytest.mark.parametrize("depth", [1, 2])
def test_the_adjoint_gradient_matches_the_dense_derivative_of_the_state(
    n_sites: int, depth: int
) -> None:
    # A third route, analytic rather than numeric: differentiate the product of dense
    # exponentials directly, inserting the generator by hand at each layer.
    transverse = 1.0
    spec = AnsatzSpec(n_qubits=n_sites, depth=depth)
    diagonal = diagonal_energies(n_sites, 1.0, 0.0, "open")
    theta = small_angle(depth, scale=0.6, seed=1)
    gamma, beta = split_angles(theta)

    diag_generator = np.diag(diagonal).astype(np.complex128)
    field_generator = _dense_field(n_sites, transverse)
    hamiltonian = _dense(n_sites, 1.0, transverse, 0.0, "open")

    def layered(insert_at: int | None, generator: np.ndarray | None) -> np.ndarray:
        """The state, optionally with a generator inserted after gate ``insert_at``."""
        state = uniform_superposition(n_sites)
        gate = 0
        for layer in range(depth):
            state = expm(-1j * gamma[layer] * diag_generator) @ state
            if gate == insert_at and generator is not None:
                state = -1j * generator @ state
            gate += 1
            state = expm(-1j * beta[layer] * field_generator) @ state
            if gate == insert_at and generator is not None:
                state = -1j * generator @ state
            gate += 1
        return state

    psi = layered(None, None)
    theirs = np.zeros(2 * depth)
    for layer in range(depth):
        d_gamma = layered(2 * layer, diag_generator)
        d_beta = layered(2 * layer + 1, field_generator)
        theirs[layer] = 2.0 * float((psi.conj() @ (hamiltonian @ d_gamma)).real)
        theirs[depth + layer] = 2.0 * float((psi.conj() @ (hamiltonian @ d_beta)).real)

    _, ours = energy_and_gradient(theta, spec, diagonal, transverse)
    assert np.allclose(ours, theirs, atol=1e-10)


@pytest.mark.parametrize("n_sites", SIZES)
def test_the_energy_returned_beside_the_gradient_is_the_same_energy(n_sites: int) -> None:
    spec = AnsatzSpec(n_qubits=n_sites, depth=2)
    diagonal = diagonal_energies(n_sites, 1.0, 0.2, "open")
    theta = adiabatic_ramp(2)
    value, _ = energy_and_gradient(theta, spec, diagonal, 0.8)
    assert value == pytest.approx(energy(evolve(theta, spec, diagonal, 0.8), diagonal, 0.8))


@pytest.mark.parametrize("n_sites", SIZES)
def test_the_gradient_vanishes_where_the_field_dominates_completely(n_sites: int) -> None:
    # With no coupling at all, |+> is already the exact ground state and every layer is a
    # rotation about an axis it already lies on. A non-zero gradient here would mean the
    # generators are wired to the wrong halves.
    spec = AnsatzSpec(n_qubits=n_sites, depth=2)
    diagonal = diagonal_energies(n_sites, coupling=0.0)
    _, gradient = energy_and_gradient(np.zeros(4), spec, diagonal, 1.0)
    assert np.allclose(gradient, 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# A lattice reaches the solvers, checked against algebra that shares no code
# --------------------------------------------------------------------------

SQUARE_AND_TRIANGLE = [
    Lattice("square", 2, 2),
    Lattice("square", 2, 3),
    Lattice("triangular", 2, 2),
    Lattice("chain", 1, 4),
]


def _kronecker_hamiltonian(
    lattice: Lattice,
    coupling: float,
    transverse_field: float,
    longitudinal_field: float,
) -> np.ndarray:
    r"""Assemble the Hamiltonian as an explicit dense matrix.

    Deliberately the slow, obvious route: one Kronecker product per operator per
    site, built from the two-by-two Pauli matrices and nothing else. It shares no
    line of code with :func:`diagonal_energies`, which does bit arithmetic over
    basis indices and never forms a matrix -- so agreement between them is evidence
    rather than a self-consistency check that would pass either way.

    Args:
        lattice: The shape, which supplies the bond list.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis.

    Returns:
        The full :math:`2^L \times 2^L` matrix.
    """
    identity = np.eye(2)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])

    def at(matrix: np.ndarray, site: int) -> np.ndarray:
        """Place a single-site operator on ``site``, little-endian like the module."""
        out = np.array([[1.0]])
        for position in range(lattice.n_sites):
            out = np.kron(matrix if position == site else identity, out)
        return out

    dimension = 2**lattice.n_sites
    matrix = np.zeros((dimension, dimension))
    for left, right in lattice.bonds():
        matrix -= coupling * at(sigma_z, left) @ at(sigma_z, right)
    for site in range(lattice.n_sites):
        matrix -= transverse_field * at(sigma_x, site)
        matrix -= longitudinal_field * at(sigma_z, site)
    return matrix


@pytest.mark.parametrize("lattice", SQUARE_AND_TRIANGLE)
def test_the_diagonal_on_a_lattice_matches_a_kronecker_assembly(lattice: Lattice) -> None:
    built = _kronecker_hamiltonian(lattice, 0.7, 0.0, 0.4)
    computed = diagonal_energies(lattice.n_sites, 0.7, 0.4, lattice.boundary, lattice)
    assert np.allclose(computed, np.diag(built))


def test_a_lattice_whose_size_disagrees_with_the_site_count_is_refused() -> None:
    with pytest.raises(ValueError, match="lattice holds"):
        diagonal_energies(6, lattice=Lattice("square", 2, 2))


@pytest.mark.parametrize("lattice", SQUARE_AND_TRIANGLE)
def test_both_variational_solvers_reach_the_true_energy_on_a_lattice(lattice: Lattice) -> None:
    # The gap this closes, end to end. `lattice=` used to reach the Hamiltonian display
    # and the sealed grader but not `diagonal_energies` or `AnsatzSpec`, so the two
    # solvers that actually run the circuits could only ever solve a chain. A number
    # from a square lattice is only meaningful if both halves agree on the shape: the
    # bonds the circuit entangles and the diagonal it is scored against.
    exact = float(np.linalg.eigvalsh(_kronecker_hamiltonian(lattice, 0.7, 0.85, 0.4))[0])
    found = solve(
        n_sites=lattice.n_sites,
        depth=8,
        coupling=0.7,
        transverse_field=0.85,
        longitudinal_field=0.4,
        boundary=lattice.boundary,
        lattice=lattice,
    )
    # Variational, so it may not be exact -- but it may never be below.
    assert found.energy >= exact - 1e-8
    assert found.energy == pytest.approx(exact, abs=1e-2)
    assert found.spec.bonds == lattice.bonds()


def test_a_square_lattice_is_not_solved_as_a_chain_of_the_same_width() -> None:
    # The failure that would otherwise be silent: a plausible number, from the wrong
    # problem. Four sites in a ring of bonds is not four sites in a line.
    square = Lattice("square", 2, 2)
    shaped = solve(n_sites=4, depth=6, lattice=square).energy
    flat = solve(n_sites=4, depth=6).energy
    assert shaped != pytest.approx(flat, abs=1e-3)
    exact = float(np.linalg.eigvalsh(_kronecker_hamiltonian(square, 1.0, 1.0, 0.0))[0])
    assert shaped == pytest.approx(exact, abs=1e-6)
