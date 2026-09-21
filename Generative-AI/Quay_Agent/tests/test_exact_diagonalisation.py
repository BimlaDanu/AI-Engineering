"""The grader's sparse solver: its size cap, and whether its answer is unique.

Two things are under test, and both are
about what may honestly be *said* about a number rather than about the number.

**Degeneracy.** A ground-state energy is well defined whether or not the ground
state is. An observable measured on it is not: the solver returns one arbitrary
vector from the ground space, and a different vector would give a different
magnetisation. The plan flagged this as a frustrated-lattice problem; measured, it
is a small-``h/J`` problem, it already existed in one dimension, and the worst case
at the field this project calibrates on is a plain unfrustrated ``4 x 4`` square.

**The size cap.** Sparse diagonalisation and circuit simulation used to share one,
which made the cheaper method refuse work it could do. They are separated now, and
the two limits must stay distinct or the split silently collapses back.

The cross-checks between this solver and the Pauli-string assembler live in
``tests/test_hamiltonians.py``, beside the assembler they check.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.lattice import Geometry, Lattice
from src.physics.method_catalogue import (
    exact_diagonalisation_unsupported_reason,
    variational_quantum_eigensolver_unsupported_reason,
)
from src.physics.model import MAX_SITES_SPARSE, MAX_SITES_STATEVECTOR, TFIMSpec
from src.physics.reference import exact_diagonalisation as ed

# --------------------------------------------------------------------------
# Degeneracy: "a" ground state, not "the"
# --------------------------------------------------------------------------


def test_a_zero_field_chain_is_exactly_degenerate() -> None:
    """The clearest case, and one the closed form agrees about.

    With no transverse field the Hamiltonian is classical, and all-up and all-down
    cost exactly the same. So the energy is certain and the magnetisation is a coin
    toss between two states -- which is what the flag exists to say.
    """
    result = ed.solve(TFIMSpec(n_sites=4, coupling=1.0, field=0.0))

    assert result.energy == pytest.approx(-4.0)
    assert result.gap == pytest.approx(0.0)
    assert not result.ground_state_is_unique


def test_turning_the_field_on_splits_the_pair() -> None:
    """The guard must not fire on the case the project actually calibrates at."""
    result = ed.solve(TFIMSpec(n_sites=6, coupling=1.0, field=1.0))

    assert result.gap > 0.1
    assert result.ground_state_is_unique


def test_the_four_by_four_square_is_degenerate_at_the_field_we_calibrate_on() -> None:
    """The finding that made this guard necessary rather than theoretical.

    A ``4 x 4`` square at ``J = h = 1`` is the flagship two-dimensional size -- the
    smallest 2D cluster with an interior, since every site of a ``3 x 3`` is on its
    boundary. Its gap is about ``8e-5``: the two symmetry-broken states are
    indistinguishable, and its magnetisation was being reported as though it were a
    property of the problem.

    Four neighbours per site is why. At the same field a *chain* has two, so the
    square sits far deeper in the ordered phase and its symmetry-broken pair is far
    closer together.
    """
    square = Lattice("square", 4, 4)
    spec = TFIMSpec(n_sites=16, coupling=1.0, field=1.0, boundary="open")

    result = ed.solve(spec, lattice=square)

    assert result.gap < 1e-3
    assert not result.ground_state_is_unique
    # And a chain of the same length at the same field is fine, which is what makes
    # this a fact about the geometry rather than about the size.
    assert ed.solve(TFIMSpec(n_sites=16, field=1.0, boundary="open")).ground_state_is_unique


def test_frustration_is_not_what_drives_the_degeneracy() -> None:
    """The plan had this backwards, and the correction is checkable.

    The plan warned that *frustrated* ground states are often degenerate.
    Frustration suppresses ordering, and it is ordering that brings the
    symmetry-broken pair together -- so at ``h = J`` the frustrated triangular case
    has the **healthier** gap of the two. Asserted here because the wrong
    attribution would put the guard on the geometry instead of on the gap, and would
    then miss the square entirely.

    Built from a bond list in the test rather than through a spec, because
    :class:`~src.physics.model.TFIMSpec` still refuses ``J < 0`` -- so an
    antiferromagnet is expressible as a geometry and not yet as a problem a solver
    will run. That gap is Phase 3's.
    """
    from src.physics.quantum.hamiltonians import PauliSum, PauliTerm

    def gap_of(coupling: float) -> float:
        lattice = Lattice("triangular", 3, 3)
        terms = [PauliTerm.from_mapping(-coupling, {i: "Z", j: "Z"}) for i, j in lattice.bonds()]
        terms += [PauliTerm.from_mapping(-1.0, {i: "X"}) for i in range(lattice.n_sites)]
        dense = PauliSum(n_qubits=lattice.n_sites, terms=tuple(terms)).to_matrix().toarray()
        levels = np.linalg.eigvalsh(dense.real)
        return float(levels[1] - levels[0])

    frustrated = gap_of(-1.0)
    unfrustrated = gap_of(1.0)

    assert Lattice("triangular", 3, 3).frustrated_by(-1.0)
    assert frustrated > unfrustrated


def test_the_gap_is_reported_from_both_solver_paths() -> None:
    """Dense and Lanczos must agree about the gap, not only about the energy.

    They are separate code paths chosen on the Hilbert-space dimension, and the
    Lanczos one had to grow from one eigenpair to two to get this. A path that
    returned the levels out of order would report a negative gap and call every
    ground state unique.
    """
    small = ed.solve(TFIMSpec(n_sites=4, field=1.0, boundary="open"))
    large = ed.solve(TFIMSpec(n_sites=10, field=1.0, boundary="open"))

    assert small.solver == "dense"
    assert large.solver == "lanczos"
    for result in (small, large):
        assert result.gap > 0.0
        assert np.isfinite(result.gap)


@pytest.mark.parametrize(
    ("geometry", "rows", "cols"),
    [("chain", 1, 6), ("square", 2, 3), ("triangular", 3, 3)],
)
def test_the_correlation_averages_over_the_bonds_the_state_was_found_on(
    geometry: Geometry, rows: int, cols: int
) -> None:
    """A lattice's state averaged over a chain's bonds is a number for neither.

    The observable divides by the bond count, and the two geometries have different
    ones -- so passing the lattice to `solve` and not to the observable would divide
    a lattice's sum by a chain's denominator and quietly rescale the answer.
    """
    lattice = Lattice(geometry, rows, cols)
    spec = TFIMSpec(n_sites=lattice.n_sites, field=1.0, boundary="open")

    result = ed.solve(spec, lattice=lattice)

    # A correlation is an average of +-1 quantities, so it cannot leave that range.
    # A mismatched denominator is exactly what pushes it out.
    assert -1.0 <= result.zz_correlation <= 1.0


# --------------------------------------------------------------------------
# The size cap, split in two
# --------------------------------------------------------------------------


def test_the_grading_cap_is_never_below_the_cap_on_what_it_grades() -> None:
    """The one ordering that must hold, whatever the two numbers are.

    They are equal today, both sixteen, because sixteen is where a laptop stops:
    eighteen sites cost this method 2.5 s and half a gigabyte. Equality is fine.
    Sparse diagonalisation falling *below* the circuit simulator is not, because it
    is the grader's method -- the ceiling on what can be checked would then be set
    by the cost of the thing being checked, and a circuit could be run at a size
    nothing was able to mark.
    """
    assert MAX_SITES_SPARSE >= MAX_SITES_STATEVECTOR


def test_each_method_is_refused_against_its_own_cap() -> None:
    """Each check reads its own constant, which is why there are two of them.

    Asserted at each cap's own boundary rather than in the gap between them: the
    two numbers coincide at present, so there is no gap to stand in, and a test
    that needed one would fail for a reason that is not a defect.
    """
    assert exact_diagonalisation_unsupported_reason(TFIMSpec(n_sites=MAX_SITES_SPARSE)) is None
    assert (
        variational_quantum_eigensolver_unsupported_reason(TFIMSpec(n_sites=MAX_SITES_STATEVECTOR))
        is None
    )

    circuit = variational_quantum_eigensolver_unsupported_reason(
        TFIMSpec(n_sites=MAX_SITES_STATEVECTOR + 1, boundary="open")
    )
    assert circuit is not None
    assert "amplitudes" in circuit


def test_past_the_sparse_cap_diagonalisation_refuses_rather_than_hangs() -> None:
    """An agent that declines a run it cannot afford beats one that hangs."""
    spec = TFIMSpec(n_sites=MAX_SITES_SPARSE + 1, boundary="open")

    refusal = exact_diagonalisation_unsupported_reason(spec)

    assert refusal is not None
    assert str(MAX_SITES_SPARSE) in refusal
    with pytest.raises(ValueError, match="cannot solve"):
        ed.solve(spec)


def test_the_four_by_four_square_is_inside_the_sparse_cap() -> None:
    """The size the cap exists for, asserted so a later trim cannot lose it.

    Sixteen sites is the smallest 2D cluster with an interior site, and the whole
    reason the sparse cap is not twelve. It is the only size above
    :data:`~src.physics.model.WORKING_SITES` this project runs anything at.
    """
    assert Lattice("square", 4, 4).n_sites <= MAX_SITES_SPARSE
    assert exact_diagonalisation_unsupported_reason(TFIMSpec(n_sites=16)) is None
