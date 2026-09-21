r"""Sparse exact diagonalisation of the transverse-field Ising model.

Exact and general, where :mod:`src.physics.reference.free_fermions` is exact and
narrow: this builds the Hamiltonian matrix, so it handles open boundaries, odd
lengths and models with no closed form. It pays with a Hilbert space of
dimension :math:`2^L`, hence a hard size cap -- and the two methods together give
the agent a real choice to make.

Basis states are plain integers ``0 .. 2**L - 1``, bit ``i`` being the spin on
site ``i`` with ``bit = 1`` meaning :math:`\sigma^z = +1`. The Ising term is then
diagonal and the transverse field connects ``s`` to ``s ^ (1 << i)``, a single
XOR.

``H`` has at most ``L + 1`` nonzeros per row. At ``L = 12`` that is roughly
53,000 against 16.8 million elements, so it is assembled in COO, converted to
CSR, and solved by a Lanczos iteration needing only matrix-vector products;
densely it would cost 134 MB of mostly zeros.

A chain by default, and any shape :mod:`src.physics.lattice` describes when one
is passed -- the field term is one operator per site whatever the sites are
joined to, so geometry changes the diagonal and nothing else. The bond list is
shared with :mod:`src.physics.quantum.hamiltonians`; what keeps the two routes
independent is the algebra rather than the edge list, see :func:`bonds`.

Any ``L`` up to :data:`~src.physics.model.MAX_SITES_SPARSE`, sixteen -- a
``4 x 4`` square, solved in 0.26 s -- either boundary, any ``J > 0`` and
``h >= 0``. That cap is this method\'s own constant rather than the circuit
simulator\'s, even though they agree today: reading the simulator\'s cap would set
the ceiling on what can be graded by the cost of what is being graded.

:func:`solve` also reports the gap to the next level, because a degenerate ground
space means one arbitrary vector was returned and any observable measured on it
is an artefact of that choice. See :data:`DEGENERACY_TOLERANCE`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

from src.physics.lattice import Lattice
from src.physics.method_catalogue import (
    exact_diagonalisation_memory_bytes,
    exact_diagonalisation_unsupported_reason,
)
from src.physics.model import TFIMSpec

FloatArray = NDArray[np.float64]

METHOD_NAME = "exact_diagonalisation"
"""Registry key. Kept next to the implementation so the two cannot drift."""

DENSE_SOLVER_MAX_DIMENSION = 64
"""Below this Hilbert-space dimension, diagonalise densely instead of iterating.

A Lanczos iteration needs a Krylov subspace of ``ncv`` vectors, and the default
``ncv`` already exceeds the whole space for a handful of spins. At that size
ARPACK is both pointless and prone to convergence warnings, whereas a dense
symmetric eigensolve on a 64 x 64 matrix is instant and unconditionally
reliable. The matrix is still *stored* sparsely either way.
"""

DEGENERACY_TOLERANCE = 1e-3
r"""Gap below which the ground state is reported as one of several, in units of ``J``.

A **reporting threshold, not a theorem.** Two levels separated by far less than the
coupling that produced them are mixed by any perturbation, so an observable measured
on one of them is not a property of the problem.

Chosen from the measured spectra rather than picked, and it separates every case
tried on this machine. Gaps at ``J = 1``:

===================================  ======  ==================
shape and field                       gap     verdict
===================================  ======  ==================
chain of 16, ``h = 1``                0.19    unique
``3 x 3`` triangular AFM, ``h = 1``   0.62    unique
``4 x 4`` square, ``h = 1``           7.8e-5  one of several
``2 x 2`` square, ``h = 0.2``         9.9e-4  one of several
``3 x 3`` triangular AFM, ``h = 0.2`` 4e-6    one of several
``3 x 3`` triangular FM, ``h = 0.2``  0.0     one of several
===================================  ======  ==================

**Note what drives it, because the plan had this wrong.** Near-degeneracy comes from
being in the *ordered* phase -- small ``h/J``, where the two symmetry-broken states
are nearly equal in energy -- and not from frustration.
The plan for this work attributed it to frustrated antiferromagnets; measured,
the frustrated ``3 x 3`` triangular case at ``h = 1`` has the **healthiest gap in the
table**, because frustration suppresses the ordering that causes the degeneracy. The
worst case at ``h = 1`` is a plain unfrustrated ``4 x 4`` square, whose four
neighbours per site put it deep in the ordered phase at the same field. So this guard
belongs on the gap, and it was already needed in one dimension.
"""

DEFAULT_LEVELS = 6
"""How many low-lying levels :func:`low_levels` returns unless asked otherwise.

Six is enough to show the structure that matters in this model: the nearly
degenerate ground pair, the one-particle band above it, and the start of the
two-particle continuum. More lines on a spectrum plot stop being readable before
they start being informative.
"""

MAX_LEVELS = 16
"""Most levels any caller can ask for.

Not a cost limit -- a dozen extra eigenvalues is milliseconds -- but a limit on
what can be honestly *drawn*. Past this the low-lying spectrum is no longer
low-lying, and the sparse solver's accuracy for interior states is not something
this project verifies.
"""


@dataclass(frozen=True, slots=True)
class EDResult:
    """Everything one exact-diagonalisation run produced.

    Bundling the observables with the energy keeps the expensive part -- the
    eigenvector -- from being recomputed once per question, and gives the agent
    a single object to cache, log and show.

    Attributes:
        energy: Ground-state energy ``E0``. Well defined whether or not the ground
            state is unique -- unlike the two observables below.
        energy_density: ``E0 / L``, the quantity comparable across sizes.
        transverse_magnetisation: ``(1/L) sum_i <sigma_x_i>``, measured on **a**
            ground state. See :attr:`ground_state_is_unique` before quoting it.
        zz_correlation: ``(1/n_bonds) sum_bonds <sigma_z_i sigma_z_j>``, with the
            same caveat.
        gap: Distance from ``E0`` to the next level, in the same units. ``inf`` for
            a one-dimensional space.
        coupling: The ``J`` the run used, kept so that :attr:`ground_state_is_unique`
            can judge the gap against the energy scale that produced it rather than
            against an absolute number.
        dimension: Hilbert-space dimension actually diagonalised, ``2**L``.
        solver: ``"dense"`` or ``"lanczos"`` -- which path ran.
    """

    energy: float
    energy_density: float
    transverse_magnetisation: float
    zz_correlation: float
    gap: float
    coupling: float
    dimension: int
    solver: str

    @property
    def ground_state_is_unique(self) -> bool:
        """Whether the observables above describe *the* ground state or one of several.

        A degenerate ground space means the solver returned one arbitrary vector
        from it, and any observable measured on that vector is an artefact of the
        arbitrary choice. The energy is unaffected; the magnetisation and the
        correlation are not.

        Judged against :data:`DEGENERACY_TOLERANCE` times the coupling, because
        ``J`` sets the scale of every term in the Hamiltonian and an absolute
        threshold would mean different things at different couplings.
        """
        return self.gap > DEGENERACY_TOLERANCE * abs(self.coupling)


def unsupported_reason(spec: TFIMSpec) -> str | None:
    """Explain why this method cannot handle ``spec``, or return ``None``.

    Delegates to :mod:`src.physics.method_catalogue`; see the note on the same
    function in :mod:`src.physics.reference.free_fermions` for why the
    applicability checks live outside the sealed package.

    Args:
        spec: The problem to check.

    Returns:
        A human-readable reason the method is inapplicable, or ``None`` if it
        applies.
    """
    return exact_diagonalisation_unsupported_reason(spec)


def _require_supported(spec: TFIMSpec) -> None:
    """Raise if ``spec`` is outside this method's domain of validity.

    Args:
        spec: The problem to check.

    Raises:
        ValueError: If :func:`unsupported_reason` gives a reason.
    """
    reason = unsupported_reason(spec)
    if reason is not None:
        raise ValueError(f"{METHOD_NAME} cannot solve {spec.label()}: {reason}")


def estimate_memory_bytes(spec: TFIMSpec) -> int:
    """Predict the peak memory an exact-diagonalisation run would need.

    Delegates to :mod:`src.physics.method_catalogue`, for the same reason as
    :func:`unsupported_reason`: the estimate is arithmetic on ``L`` and reveals
    nothing about the eigenvalue, so it belongs on the agent's side of the wall.

    Args:
        spec: The problem to cost. Need not be supported; the estimate is what
            tells the agent it is not.

    Returns:
        An approximate byte count, accurate to a factor of about two.
    """
    return exact_diagonalisation_memory_bytes(spec)


def spin_table(n_sites: int) -> NDArray[np.int8]:
    """Return the ``+/-1`` spin value of every site in every basis state.

    Args:
        n_sites: Chain length ``L``.

    Returns:
        Array of shape ``(2**L, L)`` whose ``[s, i]`` entry is ``+1`` if bit
        ``i`` of ``s`` is set and ``-1`` otherwise.
    """
    states = np.arange(2**n_sites, dtype=np.int64)
    bits = (states[:, None] >> np.arange(n_sites, dtype=np.int64)) & 1
    return (2 * bits - 1).astype(np.int8)


def bonds(spec: TFIMSpec, lattice: Lattice | None = None) -> list[tuple[int, int]]:
    """List the Ising bonds, from this module's own generator or from a geometry.

    There are two routes and keeping them apart is deliberate. With no ``lattice``, this
    builds the chain's edges from its own arithmetic -- deliberately not shared with
    :func:`src.physics.quantum.hamiltonians.chain_bonds`, which does the same job on
    the agent's side of the import wall, because two solvers that share a bond
    generator share its bugs and would then agree for the wrong reason.

    In two dimensions that duplication is not available, and it is worth being exact
    about what the project's "two independent routes" claim then means. There is one
    geometry module, :mod:`src.physics.lattice`, and both solvers read it. What stays
    independent is the **algebra**: this module assembles the Hamiltonian as a sparse
    matrix by acting on basis integers with an XOR, while
    :class:`~src.physics.quantum.hamiltonians.PauliSum` assembles it as a Kronecker
    product of two-by-two matrices. Those share no code and no bit convention, and
    they are what the cross-check compares. The *edge list* is pinned separately, by
    the one place two independent generators do exist -- the chain -- and by the
    ``2 x 2`` square's identity with a four-site ring, which has a closed form.

    Args:
        spec: The problem. Supplies the coupling and the size, and -- unless
            ``lattice`` overrides it -- the shape: a spec that names a square or
            triangular lattice brings its own edge list through
            :attr:`~src.physics.model.TFIMSpec.shape_to_solve`, so a caller who
            simply passes the spec gets the problem the spec describes rather than
            a line of the same size.
        lattice: A geometry to take the edges from instead of the spec's own. Its
            site count must match ``spec.n_sites``, since the Hilbert space is
            built from the spec. Kept as an argument because the cross-check tests
            need to solve one size on several shapes without building a spec for
            each.

    Returns:
        Pairs ``(i, j)`` of coupled sites: ``L`` of them wrapping around for a
        ring, ``L - 1`` for a segment, or whatever the geometry lists.

    Raises:
        ValueError: If ``lattice`` describes a different number of sites from
            ``spec``. Silently trusting one over the other would build a
            Hamiltonian for a problem neither of them describes.

    Examples:
        >>> from src.physics.model import TFIMSpec
        >>> bonds(TFIMSpec(n_sites=4, boundary="open"))
        [(0, 1), (1, 2), (2, 3)]

        A spec that names a shape carries it, with no second argument:

        >>> bonds(TFIMSpec(n_sites=4, boundary="open", geometry="square", rows=2))
        [(0, 1), (0, 2), (1, 3), (2, 3)]

        And an explicit geometry overrides whatever the spec said:

        >>> from src.physics.lattice import Lattice
        >>> bonds(TFIMSpec(n_sites=4), lattice=Lattice("square", 2, 2))
        [(0, 1), (0, 2), (1, 3), (2, 3)]
    """
    lattice = lattice if lattice is not None else spec.shape_to_solve
    if lattice is None:
        return [(i, (i + 1) % spec.n_sites) for i in range(spec.n_bonds)]
    if lattice.n_sites != spec.n_sites:
        raise ValueError(
            f"the geometry has {lattice.n_sites} sites and the spec has "
            f"{spec.n_sites}; they must agree"
        )
    return list(lattice.bonds())


def hamiltonian(spec: TFIMSpec, lattice: Lattice | None = None) -> csr_matrix:
    """Assemble ``H = -J sum_<ij> sigma_z_i sigma_z_j - h sum_i sigma_x_i`` in CSR form.

    The transverse-field term is untouched by geometry -- it is one operator per
    site, whatever the sites are connected to -- so a lattice changes the diagonal
    and nothing else. That is why generalising this to two dimensions costs one
    argument rather than a rewrite.

    Args:
        spec: The problem.
        lattice: A geometry to take the bonds from. See :func:`bonds`.

    Returns:
        A real symmetric ``2**L x 2**L`` CSR matrix.

    Raises:
        ValueError: If the spec exceeds the size cap, or the geometry and the spec
            disagree about the number of sites.
    """
    _require_supported(spec)
    n_sites = spec.n_sites
    dimension = 2**n_sites
    states = np.arange(dimension, dtype=np.int64)

    # Ising term: diagonal, because sigma_z is diagonal in this basis.
    spins = spin_table(n_sites).astype(np.float64)
    diagonal = np.zeros(dimension, dtype=np.float64)
    for i, j in bonds(spec, lattice):
        diagonal -= spec.coupling * spins[:, i] * spins[:, j]

    # Transverse field: sigma_x_i maps |s> to |s XOR 2**i>, one off-diagonal entry
    # per site per basis state. Building every (row, col) pair at once keeps
    # this a handful of vectorised operations rather than a 2**L Python loop.
    flipped = states[:, None] ^ (np.int64(1) << np.arange(n_sites, dtype=np.int64))
    rows = np.concatenate([states, flipped.ravel(order="F")])
    cols = np.concatenate([states, np.tile(states, n_sites)])
    values = np.concatenate([diagonal, np.full(dimension * n_sites, -spec.field, dtype=np.float64)])
    matrix = coo_matrix((values, (rows, cols)), shape=(dimension, dimension))
    return matrix.tocsr()


def _ground_state(matrix: csr_matrix) -> tuple[float, FloatArray, float, str]:
    """Find the lowest eigenpair, and how far the next level sits above it.

    Two levels are returned rather than one, and the second is not a luxury. The energy
    of a degenerate ground state is perfectly well defined; an *observable* measured
    on it is not, because the solver returns one arbitrary vector from the ground
    space and a different one would give a different answer. So the gap is what
    tells a caller whether :attr:`EDResult.transverse_magnetisation` and
    :attr:`EDResult.zz_correlation` are numbers it may quote.

    That case is not hypothetical, and it is not confined to frustrated lattices.
    A ``4 x 4`` square at ``J = h = 1`` -- the flagship two-dimensional size -- has
    a gap of ``7.8e-5``: the two symmetry-broken states are indistinguishable, and
    its observables were being reported as though they were unique.

    The second level costs about 2.4x the first on the Lanczos path (measured: 0.11 s
    to 0.29 s at sixteen sites, 6.3 s to 15.0 s at twenty) and nothing at all on the
    dense path, which computes the whole spectrum anyway. Paid, because the
    alternative is quoting a number that depends on which vector ARPACK happened to
    return.

    Args:
        matrix: The Hamiltonian in CSR form.

    Returns:
        Tuple of ``(eigenvalue, normalised eigenvector, gap, solver name)``. The gap
        is ``inf`` for a one-dimensional space, which has no second level rather
        than a distant one.
    """
    dimension = matrix.shape[0]
    if dimension <= DENSE_SOLVER_MAX_DIMENSION:
        values, vectors = np.linalg.eigh(matrix.toarray())
        return (
            float(values[0]),
            np.asarray(vectors[:, 0], dtype=np.float64),
            _gap(values),
            "dense",
        )
    # A fixed start vector makes the run reproducible, which matters because
    # results are cached and compared across sessions. ARPACK would otherwise
    # seed itself randomly.
    start = np.ones(dimension, dtype=np.float64) / np.sqrt(dimension)
    values, vectors = eigsh(matrix, k=2, which="SA", v0=start, tol=0.0)
    # ARPACK does not promise ascending order, and the whole point here is which
    # of the two is lower.
    order = np.argsort(values)
    return (
        float(values[order[0]]),
        np.asarray(vectors[:, order[0]], dtype=np.float64),
        _gap(values[order]),
        "lanczos",
    )


def _gap(ascending: FloatArray) -> float:
    """Distance from the lowest level to the next, given levels in ascending order."""
    if ascending.size < 2:
        return float("inf")
    return float(ascending[1] - ascending[0])


def low_levels(spec: TFIMSpec, count: int = DEFAULT_LEVELS) -> FloatArray:
    """Return the lowest few eigenvalues of the Hamiltonian, in ascending order.

    The ground state alone answers "what is the energy?"; the levels above it are
    what answer "how far is the first excitation?" and "what does the low-lying
    spectrum look like as the field is turned up?" -- the gap is the quantity that
    closes at the transition, and it cannot be read off a single number.

    Args:
        spec: The problem.
        count: How many levels to return. Held between one and both
            :data:`MAX_LEVELS` and the dimension of the space -- for a short
            chain the dimension is the tighter of the two, and a request below
            one is raised rather than refused, since the ground state is always
            available.

    Returns:
        The ``min(count, 2**L)`` lowest eigenvalues, ascending. Degenerate levels
        appear as many times as they are degenerate: at small ``h`` the two lowest
        are a nearly degenerate pair, and collapsing them would hide the physics.

    Raises:
        ValueError: If the spec exceeds the size cap.

    Examples:
        At zero field the ground state is the doubly degenerate ordered pair, so
        the gap between the two lowest levels is exactly zero:

        >>> from src.physics.model import TFIMSpec
        >>> levels = low_levels(TFIMSpec(n_sites=4, coupling=1.0, field=0.0))
        >>> round(float(levels[1] - levels[0]), 12)
        0.0

        Turning the field on splits them:

        >>> levels = low_levels(TFIMSpec(n_sites=4, coupling=1.0, field=0.5))
        >>> float(levels[1] - levels[0]) > 0.0
        True
    """
    _require_supported(spec)
    matrix = hamiltonian(spec)
    dimension = matrix.shape[0]
    wanted = max(1, min(count, MAX_LEVELS, dimension))
    if dimension <= DENSE_SOLVER_MAX_DIMENSION:
        values = np.linalg.eigvalsh(matrix.toarray())
        return np.asarray(np.sort(values)[:wanted], dtype=np.float64)
    # ARPACK cannot return every eigenvalue of a matrix, so `wanted` must stay
    # strictly below the dimension; the dense branch above covers the short chains
    # where that bites. Fixed start vector for the same reason as `_ground_state`:
    # a cached result must not depend on a random seed.
    start = np.ones(dimension, dtype=np.float64) / np.sqrt(dimension)
    values = eigsh(
        matrix,
        k=min(wanted, dimension - 1),
        which="SA",
        v0=start,
        tol=0.0,
        return_eigenvectors=False,
    )
    return np.asarray(np.sort(values), dtype=np.float64)


def solve(spec: TFIMSpec, lattice: Lattice | None = None) -> EDResult:
    """Diagonalise the Hamiltonian and measure a ground state's observables.

    *A* ground state, not *the*: when the ground space is degenerate the solver
    returns one arbitrary vector from it. :attr:`EDResult.ground_state_is_unique`
    says which case this is, and it is not a rare one -- a ``4 x 4`` square at
    ``J = h = 1`` is degenerate to five decimal places.

    Args:
        spec: The problem. Supplies the size, the couplings and the boundary.
        lattice: A geometry to take the bonds from. See :func:`bonds`.

    Returns:
        The energy together with the two observables a ground state supports, the
        gap that says whether those observables are unique, and a record of which
        solver ran.

    Raises:
        ValueError: If the spec exceeds the size cap, or the geometry and the spec
            disagree about the number of sites.

    Examples:
        At zero field the ground state is fully ordered, so every bond is
        satisfied and the transverse magnetisation vanishes:

        >>> from src.physics.model import TFIMSpec
        >>> result = solve(TFIMSpec(n_sites=4, coupling=1.0, field=0.0))
        >>> round(result.energy, 12), round(result.zz_correlation, 12)
        (-4.0, 1.0)

        And it is exactly degenerate there -- the all-up and all-down states cost
        the same -- so those observables are one choice out of two:

        >>> result.gap, result.ground_state_is_unique
        (0.0, False)

        Turning the field on splits them:

        >>> solve(TFIMSpec(n_sites=4, coupling=1.0, field=1.0)).ground_state_is_unique
        True
    """
    _require_supported(spec)
    matrix = hamiltonian(spec, lattice)
    energy, state, gap, solver = _ground_state(matrix)
    return EDResult(
        energy=energy,
        energy_density=energy / spec.n_sites,
        transverse_magnetisation=transverse_magnetisation(spec, state),
        zz_correlation=zz_correlation(spec, state, lattice),
        gap=gap,
        coupling=spec.coupling,
        dimension=matrix.shape[0],
        solver=solver,
    )


def ground_state_energy(spec: TFIMSpec) -> float:
    """Ground-state energy ``E0``, for parity with the free-fermion solver.

    Args:
        spec: The problem.

    Returns:
        The total ground-state energy, not the density.
    """
    return solve(spec).energy


def transverse_magnetisation(spec: TFIMSpec, state: FloatArray) -> float:
    """Measure ``(1/L) sum_i <sigma_x_i>`` on a given state.

    Args:
        spec: The problem the state belongs to.
        state: A normalised real state vector of length ``2**L``.

    Returns:
        The per-site transverse magnetisation. Because ``sigma_x_i`` permutes basis
        states, the expectation is a dot product of the vector with itself
        reindexed by an XOR -- no matrix is formed.
    """
    states = np.arange(2**spec.n_sites, dtype=np.int64)
    total = 0.0
    for site in range(spec.n_sites):
        total += float(np.dot(state, state[states ^ (np.int64(1) << site)]))
    return total / spec.n_sites


def zz_correlation(spec: TFIMSpec, state: FloatArray, lattice: Lattice | None = None) -> float:
    """Measure ``(1/n_bonds) sum_bonds <sigma_z_i sigma_z_j>`` on a given state.

    Args:
        spec: The problem the state belongs to.
        state: A normalised real state vector of length ``2**L``.
        lattice: The geometry whose bonds to average over. Must be the one the
            state was found on -- averaging a square lattice's state over a chain's
            bonds would return a number for neither problem.

    Returns:
        The mean nearest-neighbour Ising correlation, ``1`` for a fully ordered
        state and ``0`` for an uncorrelated one.
    """
    weights = state**2
    spins = spin_table(spec.n_sites).astype(np.float64)
    edges = bonds(spec, lattice)
    total = sum(float(np.dot(weights, spins[:, i] * spins[:, j])) for i, j in edges)
    return total / len(edges)
