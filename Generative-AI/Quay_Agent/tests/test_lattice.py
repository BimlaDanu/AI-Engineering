"""Tests for the lattice geometry — the edge lists every other layer builds from.

The first phase of the two-dimensional work. Nothing here touches a solver: the module
under test returns pairs of site indices and knows no physics. But an edge list is
only meaningful alongside the indexing convention that produced it, and a wrong bond
list produces a perfectly self-consistent answer to a problem nobody posed -- which is
the failure this file exists to make impossible.

**The energies below are assembled in the test itself**, from ``PauliTerm`` and a
dense matrix, rather than by calling any of the project's solvers. That is the house
rule applied to a new module: a geometry checked against the same code that consumes
it would only prove the two agree. Here the Hamiltonian is built by hand from the bond
list, diagonalised by ``numpy.linalg.eigh``, and held against the free-fermion closed
form -- three routes that share no algebra.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.lattice import DEFAULT_SIDES, Geometry, Lattice
from src.physics.model import TFIMSpec
from src.physics.quantum.hamiltonians import PauliSum, PauliTerm, chain_bonds
from src.physics.reference.free_fermions import ground_state_energy as closed_form

RING_OF_FOUR = -5.226251860
"""Ground-state energy of a four-site ring at ``J = h = 1``.

Recorded to the precision the closed form and a dense diagonalisation agree to. It is
written down here because it is the one number in this file that anchors the geometry
to something outside the project -- Pfeuty's 1970 solution -- and a reader should be
able to see it rather than infer it from a passing assertion.
"""


def energy_from_bonds(
    bonds: tuple[tuple[int, int], ...],
    n_sites: int,
    coupling: float = 1.0,
    field: float = 1.0,
) -> float:
    """Diagonalise ``H`` assembled by hand from an arbitrary bond list.

    Deliberately not routed through :func:`src.physics.quantum.hamiltonians.ising_chain`
    or through the grader's solver. Both of those will grow geometry support in Phase
    2, and a test that used one of them would then be checking that a module agrees
    with itself.

    Args:
        bonds: Coupled pairs.
        n_sites: How many sites.
        coupling: The Ising coupling ``J``. Negative is antiferromagnetic.
        field: The transverse field ``h``.

    Returns:
        The lowest eigenvalue of the dense Hamiltonian.
    """
    terms = [PauliTerm.from_mapping(-coupling, {i: "Z", j: "Z"}) for i, j in bonds]
    terms += [PauliTerm.from_mapping(-field, {i: "X"}) for i in range(n_sites)]
    dense = PauliSum(n_qubits=n_sites, terms=tuple(terms)).to_matrix().toarray()
    return float(np.linalg.eigvalsh(dense)[0])


# --- the calibration anchor ------------------------------------------------


def normalised(bonds: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    """The same edges as an unordered set of ``(low, high)`` pairs.

    Needed because the two generators in this project write an edge differently:
    :func:`~src.physics.quantum.hamiltonians.chain_bonds` closes a ring with
    ``(L-1, 0)`` while :meth:`~src.physics.lattice.Lattice.bonds` normalises to
    ``(0, L-1)``. Those are the same coupling, and a test comparing the tuples
    rather than the edges would be asserting a representation.

    Args:
        bonds: Pairs in either order.

    Returns:
        The edge set.
    """
    return {(min(left, right), max(left, right)) for left, right in bonds}


def test_a_two_by_two_square_is_a_four_cycle_like_a_ring_of_four() -> None:
    # The only structural identity available between a 2D lattice and a 1D chain, and
    # it holds **up to relabelling** rather than on the nose. The square's cycle runs
    # 0-1-3-2-0 because sites are numbered row-major, while a ring runs 0-1-2-3-0. The
    # graphs are isomorphic; the labelled edge sets are not, and an energy does not
    # care which -- so this test checks the shape and the one below checks the number.
    #
    # A first version compared the labelled sets and failed, correctly. The claim
    # "a 2x2 square *is* a ring of four" is true of the graph and false of the
    # indices, and that distinction is exactly what a bond-list bug hides behind.
    square = Lattice("square", 2, 2).bonds()
    degrees = [sum(site in bond for bond in square) for site in range(4)]
    assert len(square) == 4, f"a four-cycle has four edges, got {square}"
    assert degrees == [2, 2, 2, 2], f"every site of a cycle has two neighbours: {degrees}"
    ring = chain_bonds(4, "periodic")
    ring_degrees = [sum(site in bond for bond in ring) for site in range(4)]
    assert degrees == ring_degrees, "the ring should have the same degree sequence"


def test_the_two_by_two_square_reproduces_pfeutys_closed_form() -> None:
    # The same identity carried through to an energy, by three routes that share no
    # algebra: a dense matrix assembled here from the square's bond list, the same for
    # the ring, and the free-fermion closed form, which forms no matrix at all.
    square = energy_from_bonds(Lattice("square", 2, 2).bonds(), 4)
    ring = energy_from_bonds(chain_bonds(4, "periodic"), 4)
    exact = closed_form(TFIMSpec(n_sites=4, coupling=1.0, field=1.0, boundary="periodic"))
    assert square == pytest.approx(ring, abs=1e-12)
    assert square == pytest.approx(exact, abs=1e-9)
    assert square == pytest.approx(RING_OF_FOUR, abs=1e-9)


# --- the chain must not move -----------------------------------------------


@pytest.mark.parametrize("length", [3, 4, 8, 12])
@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_a_chain_built_here_is_the_chain_the_project_already_had(
    length: int, boundary: str
) -> None:
    # The regression that matters most: this module must not change a single existing
    # answer. `chain_bonds` is what every 1D result in the repository was computed
    # from, so the new generator has to describe the same couplings.
    #
    # Length two is excluded and tested separately below, because there the two
    # generators genuinely disagree rather than merely differing in notation.
    from typing import cast

    from src.physics.lattice import Boundary

    built = Lattice("chain", 1, length, cast("Boundary", boundary)).bonds()
    assert normalised(built) == normalised(chain_bonds(length, cast("Boundary", boundary)))


def test_the_two_site_ring_is_refused_here_rather_than_answered_differently() -> None:
    # Phase 0 found that `chain_bonds` and `Lattice.bonds` disagree at L = 2 periodic:
    # the periodic sum lists site 0's right neighbour as 1 and site 1's as 0, so the
    # *same* coupling appears twice and the interaction is 2J, while an edge set counts
    # one pair joined once as one bond. Different energies, not different notation.
    #
    # Phase 2 settled it, and not by picking the prettier convention. `free_fermions`
    # and `exact_diagonalisation` already agree at that size using the doubled reading,
    # and `tests/test_registry.py` runs both there -- so three routes concur and are
    # anchored to a closed form. What is left is that `Lattice.bonds` builds a *set* and
    # cannot express a doubled edge, making it the one route that would be wrong. So it
    # declines the size instead. See `src.physics.lattice.TWO_SITE_RING`.
    with pytest.raises(ValueError, match="two-site ring"):
        Lattice("chain", 1, 2, "periodic")

    doubled = chain_bonds(2, "periodic")
    assert doubled == ((0, 1), (1, 0)), "the periodic sum no longer double-counts L=2"
    assert len(normalised(doubled)) == 1, "they describe the same single coupling"

    # The consequence, kept on the record so the size of the avoided disagreement is
    # visible: a doubled bond is twice the coupling, which is a different problem.
    two_bonds = energy_from_bonds(doubled, 2)
    one_bond = energy_from_bonds(((0, 1),), 2)
    assert two_bonds < one_bond - 0.1, (
        f"the doubled coupling should sit lower: {two_bonds} vs {one_bond}"
    )

    # Open boundaries at two sites are unambiguous everywhere, so they stay available.
    assert Lattice("chain", 1, 2).bonds() == ((0, 1),)


# --- bond hygiene ----------------------------------------------------------


@pytest.mark.parametrize("geometry", ["chain", "square", "triangular"])
@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_every_bond_is_listed_once_and_never_joins_a_site_to_itself(
    geometry: str, boundary: str
) -> None:
    # A bond counted twice doubles that term in the Hamiltonian, and a self-loop is
    # not a bond at all. Both produce a wrong energy that looks entirely plausible.
    from typing import cast

    from src.physics.lattice import Boundary

    rows = 1 if geometry == "chain" else 3
    lattice = Lattice(cast("Geometry", geometry), rows, 4, cast("Boundary", boundary))
    bonds = lattice.bonds()
    assert len(bonds) == len(set(bonds)), "a bond appears more than once"
    assert all(left < right for left, right in bonds), "bonds are not in (low, high) order"
    assert all(left != right for left, right in bonds), "a site is coupled to itself"
    assert bonds == tuple(sorted(bonds)), "bonds are not sorted"


def test_wrapping_a_side_of_two_does_not_double_its_bonds() -> None:
    # The trap a periodic 2x2 lattice sets: wrapping in both directions connects each
    # neighbour pair twice, so the naive generator produces eight bonds for four
    # sites and every energy comes back doubled.
    periodic = Lattice("square", 2, 2, "periodic").bonds()
    assert len(periodic) == 4, f"expected four distinct bonds, got {periodic}"
    assert sorted(periodic) == sorted(Lattice("square", 2, 2).bonds())


# --- bipartiteness, which is computed rather than declared -----------------


@pytest.mark.parametrize(
    ("lattice", "expected"),
    [
        (Lattice("chain", 1, 6), True),
        (Lattice("chain", 1, 6, "periodic"), True),
        (Lattice("chain", 1, 5, "periodic"), False),
        (Lattice("square", 4, 4), True),
        (Lattice("square", 2, 2), True),
        (Lattice("triangular", 3, 3), False),
        (Lattice("triangular", 3, 4), False),
    ],
    ids=lambda value: getattr(value, "geometry", str(value)),
)
def test_bipartiteness_is_read_off_the_edge_list(lattice: Lattice, expected: bool) -> None:
    # An odd ring is the interesting case: it is a chain, so a check that trusted the
    # geometry's *name* would call it bipartite. It is not -- an odd cycle cannot be
    # two-coloured -- and that is why the property colours the graph instead.
    assert lattice.is_bipartite is expected


def test_frustration_needs_both_a_shape_and_a_sign() -> None:
    # Frustration is not a property of the lattice alone. A ferromagnet is satisfied
    # by every spin agreeing whatever the shape, and a bipartite antiferromagnet is
    # satisfied by alternating. Only a non-bipartite lattice with J < 0 frustrates.
    triangular = Lattice("triangular", 3, 3)
    square = Lattice("square", 4, 4)
    assert triangular.frustrated_by(-1.0)
    assert not triangular.frustrated_by(1.0)
    assert not square.frustrated_by(-1.0)
    assert not square.frustrated_by(1.0)


# --- the physical consequence of bipartiteness -----------------------------


def test_a_ferromagnet_and_an_antiferromagnet_agree_on_a_bipartite_lattice() -> None:
    # The free cross-check bipartiteness buys, and the reason `is_bipartite` is
    # computed rather than declared. Flipping every spin on one sublattice maps J to
    # -J and leaves the spectrum alone, so these must agree to machine precision. If
    # they ever disagree the bug is in this project, not in the physics.
    for lattice in (Lattice("square", 2, 2), Lattice("chain", 1, 6)):
        bonds, sites = lattice.bonds(), lattice.n_sites
        assert lattice.is_bipartite, "this test is only meaningful on a bipartite lattice"
        ferro = energy_from_bonds(bonds, sites, coupling=1.0)
        antiferro = energy_from_bonds(bonds, sites, coupling=-1.0)
        assert ferro == pytest.approx(antiferro, abs=1e-10), lattice.describe()


def test_a_frustrated_antiferromagnet_costs_energy_a_ferromagnet_does_not() -> None:
    # The other half, and the whole reason a triangular lattice is worth adding. On a
    # lattice whose triangles cannot be two-coloured the two signs are *not* related
    # by a sublattice flip, so the antiferromagnet cannot satisfy every bond and sits
    # strictly higher. That gap is frustration, measured rather than asserted.
    lattice = Lattice("triangular", 3, 3)
    assert not lattice.is_bipartite
    bonds, sites = lattice.bonds(), lattice.n_sites
    ferro = energy_from_bonds(bonds, sites, coupling=1.0)
    antiferro = energy_from_bonds(bonds, sites, coupling=-1.0)
    assert antiferro > ferro + 1e-6, (
        f"a frustrated antiferromagnet should sit above the ferromagnet: {antiferro} vs {ferro}"
    )


# --- sizes and defaults ----------------------------------------------------


def test_the_default_size_of_each_shape_is_the_small_one() -> None:
    # Deliberately the smallest useful cluster, not the largest runnable one: a
    # question that names a shape and no size should be answered in milliseconds
    # rather than starting a minute of arithmetic nobody asked for.
    assert DEFAULT_SIDES["chain"] == (1, 12)
    assert DEFAULT_SIDES["square"] == (2, 2)
    assert DEFAULT_SIDES["triangular"] == (3, 3)


@pytest.mark.parametrize(
    ("lattice", "sites", "bonds"),
    [
        (Lattice("chain", 1, 12), 12, 11),
        (Lattice("square", 2, 2), 4, 4),
        (Lattice("triangular", 3, 3), 9, 16),
        (Lattice("square", 4, 4), 16, 24),
        (Lattice("triangular", 3, 4), 12, 23),
    ],
    ids=lambda value: getattr(value, "geometry", str(value)),
)
def test_the_recorded_sizes_are_the_sizes_it_builds(
    lattice: Lattice, sites: int, bonds: int
) -> None:
    # These counts were quoted in the plan and used to argue that 4x4 is
    # affordable. A table in a plan that disagrees with the code is how a
    # feasibility estimate becomes fiction.
    assert (lattice.n_sites, lattice.n_bonds) == (sites, bonds), lattice.describe()


def test_a_two_dimensional_lattice_with_one_row_is_refused_by_name() -> None:
    # Not pedantry: a 1xL "square lattice" is a chain, and a report calling it a
    # square lattice would be describing a problem nobody asked about.
    with pytest.raises(ValueError, match="at least 2 rows"):
        Lattice("square", 1, 6)
    with pytest.raises(ValueError, match="one row"):
        Lattice("chain", 3, 6)


def test_a_shape_that_degenerates_to_a_line_is_called_a_chain() -> None:
    assert "chain" in Lattice("chain", 1, 8).describe()
    assert "square" in Lattice("square", 4, 4).describe()
    assert "triangular" in Lattice("triangular", 3, 3).describe()
