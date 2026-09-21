"""Tests for sparse exact diagonalisation.

Chain lengths are capped at ``L = 8``, so the largest Hilbert space touched
here is 256-dimensional and the whole file runs in well under a second.

The important test in this module is
:func:`test_matches_the_free_fermion_solution`. It is the first point at which
the project's central claim -- that every answer is checked against an
independent exact result -- is literally true rather than aspirational.
"""

import numpy as np
import pytest

from src.physics import ed, exact
from src.physics.model import MAX_SITES_STATEVECTOR, BoundaryCondition, TFIMSpec

SIZES = [2, 3, 4, 6, 8]
"""Chain lengths used here. Odd lengths are included because exact
diagonalisation accepts them and the free-fermion solver does not -- that
asymmetry is the whole reason the agent has a method choice to make."""

EVEN_SIZES = [2, 4, 6, 8]
BOUNDARIES: list[BoundaryCondition] = ["periodic", "open"]
FIELD_RATIOS = [0.0, 0.3, 0.7, 1.0, 1.5, 3.0]

# --------------------------------------------------------------------------
# An independent Hamiltonian, built from Pauli matrices by Kronecker product.
#
# The production code never forms a dense matrix and never multiplies Pauli
# matrices: it exploits the fact that Z is diagonal and X is an XOR on a bit
# pattern. The reference below does the obvious, slow, textbook thing instead.
# Agreement between them is evidence that the bit-twiddling is right.
# --------------------------------------------------------------------------

_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_PAULI_Z = np.array([[-1.0, 0.0], [0.0, 1.0]])  # bit 0 -> -1, bit 1 -> +1


def _operator_at(site: int, operator: np.ndarray, n_sites: int) -> np.ndarray:
    factors = [np.eye(2) for _ in range(n_sites)]
    factors[site] = operator
    out = np.array([[1.0]])
    for index in range(n_sites - 1, -1, -1):
        out = np.kron(out, factors[index])
    return out


def _dense_reference_hamiltonian(spec: TFIMSpec) -> np.ndarray:
    dimension = 2**spec.n_sites
    matrix = np.zeros((dimension, dimension))
    for i, j in ed.bonds(spec):
        z_i = _operator_at(i, _PAULI_Z, spec.n_sites)
        z_j = _operator_at(j, _PAULI_Z, spec.n_sites)
        matrix -= spec.coupling * (z_i @ z_j)
    for site in range(spec.n_sites):
        matrix -= spec.field * _operator_at(site, _PAULI_X, spec.n_sites)
    return matrix


# --------------------------------------------------------------------------
# Applicability and cost.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
def test_applies_within_the_size_cap(n_sites: int) -> None:
    assert ed.unsupported_reason(TFIMSpec(n_sites=n_sites)) is None


def test_refuses_a_chain_above_the_cap() -> None:
    spec = TFIMSpec(n_sites=MAX_SITES_STATEVECTOR + 2)
    reason = ed.unsupported_reason(spec)
    assert reason is not None
    assert "exponentially" in reason


def test_solving_an_oversized_chain_raises_rather_than_exhausting_memory() -> None:
    spec = TFIMSpec(n_sites=MAX_SITES_STATEVECTOR + 2)
    with pytest.raises(ValueError, match="exact_diagonalisation cannot solve"):
        ed.solve(spec)
    with pytest.raises(ValueError, match="exact_diagonalisation cannot solve"):
        ed.hamiltonian(spec)


def test_memory_estimate_quadruples_with_each_extra_spin() -> None:
    # One more site doubles the dimension and doubles the nonzeros per row, so
    # the estimate must grow faster than linearly. This is the number the
    # approval gate shows the user before a run.
    small = ed.estimate_memory_bytes(TFIMSpec(n_sites=8))
    large = ed.estimate_memory_bytes(TFIMSpec(n_sites=10))
    assert large > 4 * small


def test_memory_estimate_at_the_cap_is_modest() -> None:
    # The cap is not near any real memory limit; it is a deliberate refusal
    # point chosen so a run can never take the machine down.
    assert ed.estimate_memory_bytes(TFIMSpec(n_sites=MAX_SITES_STATEVECTOR)) < 5_000_000


# --------------------------------------------------------------------------
# Hamiltonian construction.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_hamiltonian_matches_the_kronecker_product_reference(
    n_sites: int, boundary: BoundaryCondition
) -> None:
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.7, boundary=boundary)
    np.testing.assert_allclose(
        ed.hamiltonian(spec).toarray(), _dense_reference_hamiltonian(spec), atol=1e-12
    )


@pytest.mark.parametrize("n_sites", SIZES)
def test_hamiltonian_is_symmetric(n_sites: int) -> None:
    matrix = ed.hamiltonian(TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.4))
    assert abs(matrix - matrix.T).max() == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("n_sites", SIZES)
def test_without_a_field_the_hamiltonian_is_purely_diagonal(n_sites: int) -> None:
    matrix = ed.hamiltonian(TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.0))
    off_diagonal = matrix - matrix.multiply(np.eye(2**n_sites))
    assert abs(off_diagonal).max() == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("n_sites", SIZES)
def test_hamiltonian_stays_sparse(n_sites: int) -> None:
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.5)
    matrix = ed.hamiltonian(spec)
    assert matrix.nnz <= 2**n_sites * (n_sites + 1)


def test_ring_has_one_more_bond_than_the_segment() -> None:
    assert len(ed.bonds(TFIMSpec(n_sites=6, boundary="periodic"))) == 6
    assert len(ed.bonds(TFIMSpec(n_sites=6, boundary="open"))) == 5


def test_spin_table_reads_bits_as_plus_and_minus_one() -> None:
    table = ed.spin_table(2)
    # state 0 = both down, state 3 = both up, state 1 = site 0 up only.
    np.testing.assert_array_equal(table[0], [-1, -1])
    np.testing.assert_array_equal(table[1], [1, -1])
    np.testing.assert_array_equal(table[2], [-1, 1])
    np.testing.assert_array_equal(table[3], [1, 1])


# --------------------------------------------------------------------------
# The cross-check: exact diagonalisation against the free-fermion solution.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", EVEN_SIZES)
@pytest.mark.parametrize("g", FIELD_RATIOS)
def test_matches_the_free_fermion_solution(n_sites: int, g: float) -> None:
    # Two methods with nothing in common -- one diagonalises a 2**L matrix, the
    # other sums L/2 closed-form quasiparticle energies -- must return the same
    # number. This is the project's headline claim, in one assertion.
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=g, boundary="periodic")
    assert ed.ground_state_energy(spec) == pytest.approx(exact.ground_state_energy(spec), abs=1e-10)


@pytest.mark.parametrize("n_sites", EVEN_SIZES)
@pytest.mark.parametrize("g", FIELD_RATIOS)
def test_magnetisation_matches_the_free_fermion_solution(n_sites: int, g: float) -> None:
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=g, boundary="periodic")
    assert ed.solve(spec).transverse_magnetisation == pytest.approx(
        exact.transverse_magnetisation(spec), abs=1e-10
    )


def test_solves_an_odd_chain_the_free_fermion_solver_refuses() -> None:
    # The payoff of having two methods: this spec has no closed-form solution
    # in our implementation, and exact diagonalisation does not care.
    spec = TFIMSpec(n_sites=7, coupling=1.0, field=0.5)
    assert exact.unsupported_reason(spec) is not None
    assert ed.unsupported_reason(spec) is None
    assert ed.solve(spec).energy < 0.0


def test_solves_open_boundaries_the_free_fermion_solver_refuses() -> None:
    spec = TFIMSpec(n_sites=8, coupling=1.0, field=0.5, boundary="open")
    assert exact.unsupported_reason(spec) is not None
    result = ed.solve(spec)
    # One fewer satisfied bond, so a segment sits above the ring at equal L.
    assert result.energy > exact.ground_state_energy(TFIMSpec(n_sites=8, field=0.5))


# --------------------------------------------------------------------------
# Ground-state observables.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_zero_field_ground_state_is_fully_ordered(
    n_sites: int, boundary: BoundaryCondition
) -> None:
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.0, boundary=boundary)
    result = ed.solve(spec)
    assert result.energy == pytest.approx(-1.0 * spec.n_bonds, abs=1e-10)
    assert result.zz_correlation == pytest.approx(1.0, abs=1e-10)
    assert result.transverse_magnetisation == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("n_sites", SIZES)
def test_dominant_field_polarises_the_ground_state(n_sites: int) -> None:
    spec = TFIMSpec(n_sites=n_sites, coupling=1e-6, field=1.0)
    result = ed.solve(spec)
    assert result.transverse_magnetisation == pytest.approx(1.0, abs=1e-6)
    assert result.zz_correlation == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("g", FIELD_RATIOS)
def test_observables_reconstruct_the_energy(
    n_sites: int, boundary: BoundaryCondition, g: float
) -> None:
    # <H> = -J * n_bonds * <sigma_z sigma_z> - h * L * <sigma_x>. If the observables and the
    # eigenvalue disagree, one of the three is measured against the wrong
    # basis convention.
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=g, boundary=boundary)
    result = ed.solve(spec)
    reconstructed = (
        -spec.coupling * spec.n_bonds * result.zz_correlation
        - spec.field * spec.n_sites * result.transverse_magnetisation
    )
    assert result.energy == pytest.approx(reconstructed, abs=1e-9)


@pytest.mark.parametrize("n_sites", SIZES)
def test_correlation_and_magnetisation_stay_in_physical_range(n_sites: int) -> None:
    result = ed.solve(TFIMSpec(n_sites=n_sites, coupling=1.0, field=1.0))
    assert -1.0 <= result.zz_correlation <= 1.0
    assert 0.0 <= result.transverse_magnetisation <= 1.0


# --------------------------------------------------------------------------
# Solver selection.
# --------------------------------------------------------------------------


def test_tiny_systems_use_the_dense_solver() -> None:
    # L = 6 gives dimension 64: a Krylov subspace would span the whole space.
    assert ed.solve(TFIMSpec(n_sites=6)).solver == "dense"


def test_larger_systems_use_the_iterative_solver() -> None:
    assert ed.solve(TFIMSpec(n_sites=8)).solver == "lanczos"


def test_both_solvers_agree_where_they_overlap() -> None:
    spec = TFIMSpec(n_sites=8, coupling=1.0, field=0.6)
    matrix = ed.hamiltonian(spec)
    dense_energy = float(np.linalg.eigvalsh(matrix.toarray())[0])
    assert ed.solve(spec).energy == pytest.approx(dense_energy, abs=1e-10)


def test_result_reports_the_dimension_it_diagonalised() -> None:
    assert ed.solve(TFIMSpec(n_sites=8)).dimension == 256


def test_repeated_runs_are_bit_identical() -> None:
    # The iterative solver is seeded with a fixed start vector so cached
    # results stay comparable across sessions.
    spec = TFIMSpec(n_sites=8, coupling=1.0, field=0.9)
    assert ed.solve(spec).energy == ed.solve(spec).energy


# --------------------------------------------------------------------------
# The levels above the ground state
# --------------------------------------------------------------------------


def test_the_lowest_level_is_the_ground_state_energy() -> None:
    # The same number two ways, so `low_levels` cannot drift from `solve`.
    spec = TFIMSpec(n_sites=6, coupling=1.0, field=0.7)
    assert ed.low_levels(spec)[0] == pytest.approx(ed.solve(spec).energy, abs=1e-10)


def test_the_levels_come_back_ascending() -> None:
    levels = ed.low_levels(TFIMSpec(n_sites=6, coupling=1.0, field=1.3))
    assert list(levels) == sorted(levels)


def test_the_ordered_chain_has_a_degenerate_ground_pair() -> None:
    # At h = 0 the two ordered configurations have the same energy exactly, so a
    # solver that returned one of them and called it the spectrum would be hiding
    # the degeneracy that the whole ordered phase rests on.
    levels = ed.low_levels(TFIMSpec(n_sites=6, coupling=1.0, field=0.0))
    assert levels[1] - levels[0] == pytest.approx(0.0, abs=1e-12)


def test_the_field_opens_the_gap() -> None:
    ordered = ed.low_levels(TFIMSpec(n_sites=6, coupling=1.0, field=0.4))
    disordered = ed.low_levels(TFIMSpec(n_sites=6, coupling=1.0, field=1.6))
    assert disordered[1] - disordered[0] > ordered[1] - ordered[0]


def test_the_gap_narrows_as_the_chain_grows_at_criticality() -> None:
    # The finite-size statement the spectrum plot is there to make: the gap at the
    # critical field is not zero, and shrinks towards zero with L.
    short = ed.low_levels(TFIMSpec(n_sites=4, coupling=1.0, field=1.0))
    longer = ed.low_levels(TFIMSpec(n_sites=8, coupling=1.0, field=1.0))
    assert 0.0 < longer[1] - longer[0] < short[1] - short[0]


def test_asking_for_more_levels_than_exist_returns_what_there_is() -> None:
    # A two-spin chain has four states, so six cannot be returned and asking must
    # not raise inside a tool call.
    assert len(ed.low_levels(TFIMSpec(n_sites=2), count=6)) == 4


def test_the_level_count_is_clamped_rather_than_honoured() -> None:
    assert len(ed.low_levels(TFIMSpec(n_sites=8), count=10_000)) == ed.MAX_LEVELS
    assert len(ed.low_levels(TFIMSpec(n_sites=8), count=0)) == 1


def test_a_chain_past_the_cap_is_refused_rather_than_attempted() -> None:
    with pytest.raises(ValueError, match="above the project cap"):
        ed.low_levels(TFIMSpec(n_sites=MAX_SITES_STATEVECTOR + 1))
