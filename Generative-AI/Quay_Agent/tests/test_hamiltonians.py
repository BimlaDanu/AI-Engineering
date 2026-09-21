"""The Pauli representation is held against the reference answer, not against itself.

`src/physics/quantum/hamiltonians.py` is where a factor of two or a flipped sign would do the
most damage, because everything downstream -- the circuits, the gradients, the shot
budget, the verdict -- inherits it silently and still looks like it is working. So the
constructions are checked four ways, none of which shares algebra with the code under
test:

* against `physics/reference/exact_diagonalisation.py`, which builds the same TFIM
  matrix from a different bit convention with a different method;
* against `physics/reference/free_fermions.py`, which never builds a matrix at all;
* against a Kronecker-product assembly written here, which shares nothing with the bit
  arithmetic in `to_matrix`;
* against the two free limits of the chain, where the ground state is a product state
  and the energy can be counted rather than computed.

These tests import the exact solvers deliberately. `tests/` is the grader's side of the
wall; `tests/test_architecture.py` is what enforces that `src/physics/quantum/` stays on the
other side of it.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.physics.lattice import Geometry, Lattice
from src.physics.model import TFIMSpec
from src.physics.quantum.hamiltonians import (
    PauliSum,
    PauliTerm,
    chain_bonds,
    ising_chain,
    maxcut_cost,
)
from src.physics.reference import exact_diagonalisation, free_fermions

PAULI_MATRICES = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def kron_matrix(operator: PauliSum) -> NDArray[np.complex128]:
    """Assemble the same matrix by repeated Kronecker product.

    Deliberately the slow, obvious construction: one 2x2 factor per qubit, in the order
    Qiskit's little-endian labels imply. It shares no line of reasoning with the bit-mask
    arithmetic in `PauliSum.to_matrix`, which is the only reason comparing them is
    evidence of anything.
    """
    dimension = 2**operator.n_qubits
    total = np.zeros((dimension, dimension), dtype=np.complex128)
    for label, coefficient in operator.to_labels():
        product = np.array([[1.0 + 0j]])
        for letter in label:  # leftmost character is the highest qubit
            product = np.kron(product, PAULI_MATRICES[letter])
        total += coefficient * product
    return total


def ed_permutation(n_qubits: int) -> NDArray[np.int64]:
    """Map this module's basis ordering onto the one `exact_diagonalisation.py` uses.

    `exact_diagonalisation.py` sets bit = 1 for sigma_z = +1; here |0> is the +1 eigenstate. The two
    labellings differ by flipping every bit, and nothing else.
    """
    return np.arange(2**n_qubits) ^ (2**n_qubits - 1)


@pytest.mark.parametrize("n_sites", [4, 6])
@pytest.mark.parametrize("field", [0.0, 0.7, 1.0, 2.5])
def test_tfim_matrix_is_the_one_exact_diagonalisation_builds(n_sites: int, field: float) -> None:
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=field, boundary="periodic")
    mine = ising_chain(
        n_sites, coupling=1.0, transverse_field=field, boundary="periodic"
    ).to_matrix()
    theirs = exact_diagonalisation.hamiltonian(spec).toarray()

    permutation = ed_permutation(n_sites)
    assert np.allclose(mine.toarray().real, theirs[np.ix_(permutation, permutation)])
    assert np.allclose(mine.toarray().imag, 0.0)


@pytest.mark.parametrize("n_sites", [4, 6, 8])
@pytest.mark.parametrize("field", [0.3, 1.0, 1.8])
def test_tfim_ground_energy_matches_the_free_fermion_solution(n_sites: int, field: float) -> None:
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=field, boundary="periodic")
    matrix = ising_chain(
        n_sites, coupling=1.0, transverse_field=field, boundary="periodic"
    ).to_matrix()
    lowest = float(np.linalg.eigvalsh(matrix.toarray()).real[0])
    assert lowest == pytest.approx(free_fermions.ground_state_energy(spec), abs=1e-10)


def test_the_two_matrix_constructions_agree_including_on_y_terms() -> None:
    # The Ising family carries no sigma^y, but `to_matrix` must still be right about it:
    # the phase i**n_y is exactly where a bit-arithmetic assembly goes wrong, and a
    # gradient or a metric element evaluated on a rotated operator will hit it.
    for operator in (
        ising_chain(4, transverse_field=0.6, longitudinal_field=0.3),
        ising_chain(5, transverse_field=1.2, boundary="periodic"),
        maxcut_cost(4, [(0, 1, 1.0), (1, 2, 0.5), (0, 3, 2.0)]),
        PauliSum(3, (PauliTerm.from_mapping(0.7, {0: "Y", 2: "X"}),)),
        PauliSum(
            3,
            (
                PauliTerm.from_mapping(0.4, {0: "Y", 1: "Y"}),
                PauliTerm.from_mapping(-0.9, {0: "X", 1: "Y", 2: "Z"}),
                PauliTerm.from_mapping(1.3, {1: "Y"}),
            ),
        ),
    ):
        assert np.allclose(operator.to_matrix().toarray(), kron_matrix(operator))


def test_every_hamiltonian_is_hermitian() -> None:
    for operator in (
        ising_chain(4, transverse_field=0.6, longitudinal_field=0.3),
        maxcut_cost(4, [(0, 1, 1.0), (2, 3, 1.0)]),
        PauliSum(3, (PauliTerm.from_mapping(0.7, {0: "Y", 2: "X"}),)),
    ):
        matrix = operator.to_matrix().toarray()
        assert np.allclose(matrix, matrix.conj().T)


def test_the_two_free_limits_of_the_ising_chain() -> None:
    # No field: every bond satisfied, so E0 = -J * (number of bonds).
    ordered = ising_chain(6, coupling=1.0, transverse_field=0.0, boundary="open")
    assert np.linalg.eigvalsh(ordered.to_matrix().toarray()).real[0] == pytest.approx(-5.0)

    # No coupling: the product state along x, so E0 = -h * N. Built directly, because
    # `ising_chain` refuses J = 0 -- the spec dataclass does too, and they must agree.
    polarised = PauliSum(6, tuple(PauliTerm.from_mapping(-0.5, {i: "X"}) for i in range(6)))
    assert np.linalg.eigvalsh(polarised.to_matrix().toarray()).real[0] == pytest.approx(-3.0)


def test_the_mixed_field_chain_costs_two_measurement_settings() -> None:
    chain = ising_chain(6, transverse_field=1.0, longitudinal_field=0.3)
    groups = chain.measurement_groups()

    assert len(groups) == 2, "the z-diagonal and x-diagonal families should each collapse to one"
    assert {group.basis for group in groups} == {("Z",) * 6, ("X",) * 6}
    # The partition must be a partition: the groups sum back to the whole Hamiltonian.
    total = sum(
        (group.operator.to_matrix() for group in groups[1:]), groups[0].operator.to_matrix()
    )
    assert np.allclose(total.toarray(), chain.to_matrix().toarray())


def test_measurement_grouping_keeps_the_identity_and_stays_a_partition() -> None:
    cut = maxcut_cost(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
    groups = cut.measurement_groups()
    assert len(groups) == 1
    assert sum(len(g.operator.terms) for g in groups) == len(cut.terms)


def test_shot_budget_scale_is_linear_in_the_chain_length() -> None:
    # sum |c| = J(N-1) + hN + gN for an open chain: the quantity that sets the shot cost.
    for n_sites in (4, 8, 12):
        chain = ising_chain(n_sites, coupling=1.0, transverse_field=1.0, longitudinal_field=0.5)
        assert chain.coefficient_l1() == pytest.approx(1.0 * (n_sites - 1) + 1.5 * n_sites)


def test_maxcut_eigenvalues_are_the_cut_values() -> None:
    # A 4-cycle is bipartite, so every edge can be cut; a triangle cannot.
    square = maxcut_cost(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)])
    values = np.linalg.eigvalsh(square.to_matrix().toarray()).real
    assert values[-1] == pytest.approx(4.0)
    assert np.allclose(values, np.round(values))  # diagonal, integer-weighted


def test_maxcut_is_returned_as_the_quantity_to_maximise() -> None:
    # The sign convention is load-bearing: minimising the unnegated operator finds the
    # *worst* cut, which looks exactly like a working algorithm doing badly.
    cut = maxcut_cost(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
    assert np.linalg.eigvalsh(cut.to_matrix().toarray()).real[-1] == pytest.approx(2.0)
    negated = cut.scaled(-1.0)
    assert np.linalg.eigvalsh(negated.to_matrix().toarray()).real[0] == pytest.approx(-2.0)


def test_bonds_of_a_ring_and_a_segment() -> None:
    assert chain_bonds(4, "open") == ((0, 1), (1, 2), (2, 3))
    assert chain_bonds(4, "periodic") == ((0, 1), (1, 2), (2, 3), (3, 0))
    assert chain_bonds(4, "open", distance=2) == ((0, 2), (1, 3))
    assert chain_bonds(3, "open", distance=5) == ()


def test_simplify_combines_and_cancels() -> None:
    zz = PauliTerm.from_mapping(1.0, {0: "Z", 1: "Z"})
    assert PauliSum(2, (zz, zz)).simplified().terms[0].coefficient == pytest.approx(2.0)
    assert PauliSum(2, (zz, zz.scaled(-1.0))).simplified().terms == ()


def test_constructions_that_should_be_refused() -> None:
    with pytest.raises(ValueError, match="sorted by qubit"):
        PauliTerm(1.0, ((1, "Z"), (0, "Z")))
    with pytest.raises(ValueError, match="not one of"):
        # The type checker rejects this too, which is the belt; the runtime check is the
        # braces, and it is the one that fires when a letter arrives from a config file.
        PauliTerm(1.0, ((0, "Q"),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="qubit 3 of a 2-qubit"):
        PauliSum(2, (PauliTerm.from_mapping(1.0, {3: "Z"}),))
    with pytest.raises(ValueError, match="strictly positive"):
        ising_chain(4, coupling=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        ising_chain(4, transverse_field=-1.0)
    with pytest.raises(ValueError, match="self-loop"):
        maxcut_cost(3, [(1, 1, 1.0)])
    with pytest.raises(ValueError, match="outside a 3-node graph"):
        maxcut_cost(3, [(0, 5, 1.0)])
    with pytest.raises(ValueError, match="cannot add"):
        _ = ising_chain(3) + ising_chain(4)
    with pytest.raises(ValueError, match="verification tool"):
        PauliSum(21, ()).to_matrix()


def test_labels_are_little_endian_and_round_trip_through_the_register_width() -> None:
    assert PauliTerm.from_mapping(1.0, {0: "Z", 1: "Z"}).label(4) == "IIZZ"
    assert PauliTerm.from_mapping(1.0, {3: "X"}).label(4) == "XIII"
    assert ising_chain(2, transverse_field=0.0).to_labels() == [("ZZ", -1.0)]


def test_is_real_flags_the_families_that_carry_odd_y_counts() -> None:
    assert ising_chain(4, transverse_field=0.5).is_real
    assert PauliSum(2, (PauliTerm.from_mapping(1.0, {0: "Y", 1: "Y"}),)).is_real  # even
    assert not PauliSum(2, (PauliTerm.from_mapping(1.0, {0: "Y"}),)).is_real


# --------------------------------------------------------------------------
# Geometry — the two assemblers on a lattice
#
# Both assemblers accept a `Lattice`,
# and it is worth being exact about what "two independent routes" still means once
# they do. There is one geometry module and both of them read it, so the *edge list*
# is shared. What stays independent is the algebra: `PauliSum.to_matrix` uses bit-mask
# arithmetic on Pauli strings, `exact_diagonalisation` acts on basis integers with an
# XOR, and the two do not even agree on which bit means spin-up. The edge list itself
# is pinned separately, by the chain -- where two independent generators do exist --
# and by the 2x2 square's identity with a four-site ring, which has a closed form.
# --------------------------------------------------------------------------

GEOMETRIES: list[tuple[Geometry, int, int]] = [
    ("chain", 1, 6),
    ("square", 2, 2),
    ("square", 2, 3),
    ("triangular", 2, 2),
    ("triangular", 3, 3),
]


@pytest.mark.parametrize(("geometry", "rows", "cols"), GEOMETRIES)
@pytest.mark.parametrize("field", [0.0, 0.7, 1.6])
def test_the_two_assemblers_agree_on_every_geometry(
    geometry: Geometry, rows: int, cols: int, field: float
) -> None:
    """Cross-check 1: Kronecker-product algebra against XOR-on-integers algebra."""
    lattice = Lattice(geometry, rows, cols)
    spec = TFIMSpec(n_sites=lattice.n_sites, coupling=1.0, field=field, boundary="open")

    mine = ising_chain(
        lattice.n_sites, coupling=1.0, transverse_field=field, lattice=lattice
    ).to_matrix()
    theirs = exact_diagonalisation.hamiltonian(spec, lattice=lattice).toarray()

    permutation = ed_permutation(lattice.n_sites)
    assert np.allclose(mine.toarray().real, theirs[np.ix_(permutation, permutation)])


@pytest.mark.parametrize(("geometry", "rows", "cols"), GEOMETRIES)
def test_the_slow_kronecker_assembly_agrees_on_every_geometry(
    geometry: Geometry, rows: int, cols: int
) -> None:
    """The third route: one 2x2 factor per qubit, sharing no line with either solver."""
    lattice = Lattice(geometry, rows, cols)
    operator = ising_chain(lattice.n_sites, transverse_field=0.9, lattice=lattice)

    assert np.allclose(operator.to_matrix().toarray(), kron_matrix(operator))


def test_a_two_by_two_square_reproduces_the_four_site_rings_closed_form() -> None:
    """Cross-check 3, the calibration anchor, now through the production code paths.

    `tests/test_lattice.py` checks this with a Hamiltonian assembled inside the test.
    This checks the same identity through `ising_chain(lattice=...)` and the grader's
    sparse assembler, which is what a 2D question will actually run. The two graphs
    are isomorphic rather than identically labelled -- a square's cycle is 0-1-3-2-0
    under row-major numbering, a ring's is 0-1-2-3-0 -- and an energy is invariant
    under relabelling, which is exactly what makes the anchor useful.
    """
    ring = TFIMSpec(n_sites=4, coupling=1.0, field=1.0, boundary="periodic")
    closed_form = free_fermions.ground_state_energy(ring)

    square = Lattice("square", 2, 2)
    through_pauli = float(
        np.linalg.eigvalsh(
            ising_chain(4, coupling=1.0, transverse_field=1.0, lattice=square).to_matrix().toarray()
        ).real[0]
    )
    through_sparse = float(
        np.linalg.eigvalsh(
            exact_diagonalisation.hamiltonian(
                TFIMSpec(n_sites=4, coupling=1.0, field=1.0, boundary="open"),
                lattice=square,
            ).toarray()
        )[0]
    )

    assert through_pauli == pytest.approx(closed_form, abs=1e-9)
    assert through_sparse == pytest.approx(closed_form, abs=1e-9)


@pytest.mark.parametrize(("geometry", "rows", "cols"), GEOMETRIES)
def test_a_geometry_replaces_the_bonds_and_leaves_the_field_terms_alone(
    geometry: Geometry, rows: int, cols: int
) -> None:
    """Why generalising cost one argument: the field is one operator per site.

    A vacuity guard as much as a property. Several of the geometries above have the
    same site count as a chain of the same length, so a `lattice=` argument that was
    silently ignored would still pass the agreement tests -- both assemblers would
    then be building the same wrong matrix. This asserts the bond count actually
    changed.
    """
    lattice = Lattice(geometry, rows, cols)
    operator = ising_chain(lattice.n_sites, transverse_field=0.5, lattice=lattice)

    coupling_terms = [term for term in operator.terms if term.weight == 2]
    field_terms = [term for term in operator.terms if term.weight == 1]

    assert len(coupling_terms) == lattice.n_bonds
    assert len(field_terms) == lattice.n_sites
    if geometry != "chain":
        assert lattice.n_bonds > lattice.n_sites - 1


def test_a_geometry_that_disagrees_with_the_site_count_is_refused() -> None:
    """Two sources for one number is one source too many if they can differ.

    Trusting either silently would build a Hamiltonian for a problem neither of them
    describes -- the right number of qubits wired up the wrong way, or the wrong
    number wired up correctly.
    """
    with pytest.raises(ValueError, match="they must agree"):
        ising_chain(6, lattice=Lattice("square", 2, 2))
    with pytest.raises(ValueError, match="they must agree"):
        exact_diagonalisation.hamiltonian(
            TFIMSpec(n_sites=6, boundary="open"), lattice=Lattice("square", 2, 2)
        )


def test_a_two_site_ring_is_refused_only_by_the_route_that_gets_it_wrong() -> None:
    """The one size where the bond generators disagree, and which of them is right.

    See `src.physics.lattice.TWO_SITE_RING`. `chain_bonds` returns the same pair
    twice, which is what the periodic sum says at L = 2, and `Lattice.bonds` returns
    it once, because it is an edge set. Those are different energies rather than
    different notation.

    **The doubled reading is the one the rest of the project is validated against.**
    `tests/test_registry.py` runs L = 2 periodic through both exact solvers and the
    closed form and they agree, so refusing the size outright -- which an earlier
    version of this change did, at the spec level -- destroyed working, tested
    coverage rather than protecting anything. `Lattice` cannot express a doubled
    edge, so it is the one route that declines the size.
    """
    with pytest.raises(ValueError, match="two-site ring"):
        Lattice("chain", 1, 2, "periodic")

    assert chain_bonds(2, "periodic") == ((0, 1), (1, 0))
    assert TFIMSpec(n_sites=2, boundary="periodic").n_bonds == 2

    # Open boundaries at two sites are unambiguous in every route, and the evals
    # use that as the smallest problem the graders agree on.
    assert chain_bonds(2, "open") == ((0, 1),)
    assert Lattice("chain", 1, 2).bonds() == ((0, 1),)
