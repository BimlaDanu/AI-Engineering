r"""Sparse exact diagonalisation of the 1D transverse-field Ising model.

Where :mod:`src.physics.exact` is exact but narrow -- it solves the uniform
TFIM ring and nothing else -- this module is exact and *general*: it builds the
Hamiltonian matrix itself, so it handles open boundaries, odd chain lengths,
and (later) models with no closed-form solution at all. It pays for that with
a Hilbert space of dimension :math:`2^L`, which is why it carries a hard size
cap and why the two methods together give the agent a real choice to make.

**Basis.** Computational basis states are plain integers ``0 .. 2**L - 1``. Bit
``i`` of the integer is the spin on site ``i``, with ``bit = 1`` meaning
:math:`\sigma^z = +1` and ``bit = 0`` meaning :math:`\sigma^z = -1`. The Ising
term is then diagonal, and the transverse field connects ``s`` to ``s ^ (1 <<
i)`` -- a single XOR. This is the construction from the project's own
``PHYSICS-NOTES/ED-Python-Scripts/TFIM_ED_1.py``, kept in the same convention
but vectorised over the basis and assembled sparsely.

**Sparsity.** ``H`` has at most ``L + 1`` nonzeros per row: one diagonal entry
and one off-diagonal entry per site. At ``L = 12`` that is roughly 53,000
nonzeros against 16.8 million matrix elements -- 0.3% dense. Storing it densely
would cost 134 MB to hold almost entirely zeros, so the matrix is assembled in
COO and converted to CSR, and the ground state is found with a Lanczos
iteration that only ever needs matrix-vector products.

**Applicability.** Any ``L`` up to
:data:`~src.physics.model.MAX_SITES_STATEVECTOR`, either boundary condition,
any ``J > 0`` and ``h >= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

from src.physics.model import MAX_SITES_STATEVECTOR, TFIMSpec

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
        energy: Ground-state energy ``E0``.
        energy_density: ``E0 / L``, the quantity comparable across sizes.
        transverse_magnetisation: ``(1/L) sum_i <sigma_x_i>``.
        zz_correlation: ``(1/n_bonds) sum_bonds <sigma_z_i sigma_z_{i+1}>``.
        dimension: Hilbert-space dimension actually diagonalised, ``2**L``.
        solver: ``"dense"`` or ``"lanczos"`` -- which path ran.
    """

    energy: float
    energy_density: float
    transverse_magnetisation: float
    zz_correlation: float
    dimension: int
    solver: str


def unsupported_reason(spec: TFIMSpec) -> str | None:
    """Explain why this method cannot handle ``spec``, or return ``None``.

    Exact diagonalisation is refused on size alone. The cap is a deliberate
    refusal rather than an attempt that might exhaust memory: an agent that
    declines a run it cannot afford is more useful than one that hangs.

    Args:
        spec: The problem to check.

    Returns:
        A human-readable reason the method is inapplicable, or ``None`` if it
        applies.
    """
    if spec.n_sites > MAX_SITES_STATEVECTOR:
        return (
            f"the Hilbert space has dimension 2**{spec.n_sites} = {2**spec.n_sites}, "
            f"above the project cap of 2**{MAX_SITES_STATEVECTOR}; use a method "
            f"whose cost does not grow exponentially with L"
        )
    return None


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

    This is what the agent shows the user *before* committing to a run, so it
    deliberately errs high: it counts the CSR matrix, the spin-table used to
    build the diagonal, and the Krylov vectors the iterative solver allocates.

    Args:
        spec: The problem to cost. Need not be supported; the estimate is what
            tells the agent it is not.

    Returns:
        An approximate byte count. Accurate to a factor of about two, which is
        all a go/no-go decision requires.
    """
    dimension = 2**spec.n_sites
    nonzeros = dimension * (spec.n_sites + 1)
    csr_bytes = nonzeros * (8 + 4) + (dimension + 1) * 4  # data + indices + indptr
    spin_table_bytes = dimension * spec.n_sites  # int8, one entry per site
    krylov_bytes = 20 * dimension * 8  # ARPACK's default subspace, float64
    return int(csr_bytes + spin_table_bytes + krylov_bytes)


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


def bonds(spec: TFIMSpec) -> list[tuple[int, int]]:
    """List the Ising bonds of the lattice.

    Args:
        spec: The problem.

    Returns:
        Pairs ``(i, j)`` of coupled sites: ``L`` of them wrapping around for a
        ring, ``L - 1`` for a segment.
    """
    return [(i, (i + 1) % spec.n_sites) for i in range(spec.n_bonds)]


def hamiltonian(spec: TFIMSpec) -> csr_matrix:
    """Assemble ``H = -J sum_i sigma_z_i sigma_z_{i+1} - h sum_i sigma_x_i`` in CSR form.

    Args:
        spec: The problem.

    Returns:
        A real symmetric ``2**L x 2**L`` CSR matrix.

    Raises:
        ValueError: If the spec exceeds the size cap.
    """
    _require_supported(spec)
    n_sites = spec.n_sites
    dimension = 2**n_sites
    states = np.arange(dimension, dtype=np.int64)

    # Ising term: diagonal, because sigma_z is diagonal in this basis.
    spins = spin_table(n_sites).astype(np.float64)
    diagonal = np.zeros(dimension, dtype=np.float64)
    for i, j in bonds(spec):
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


def _ground_state(matrix: csr_matrix) -> tuple[float, FloatArray, str]:
    """Find the lowest eigenpair of a real symmetric sparse matrix.

    Args:
        matrix: The Hamiltonian in CSR form.

    Returns:
        Tuple of ``(eigenvalue, normalised eigenvector, solver name)``.
    """
    dimension = matrix.shape[0]
    if dimension <= DENSE_SOLVER_MAX_DIMENSION:
        values, vectors = np.linalg.eigh(matrix.toarray())
        return float(values[0]), np.asarray(vectors[:, 0], dtype=np.float64), "dense"
    # A fixed start vector makes the run reproducible, which matters because
    # results are cached and compared across sessions. ARPACK would otherwise
    # seed itself randomly.
    start = np.ones(dimension, dtype=np.float64) / np.sqrt(dimension)
    values, vectors = eigsh(matrix, k=1, which="SA", v0=start, tol=0.0)
    return float(values[0]), np.asarray(vectors[:, 0], dtype=np.float64), "lanczos"


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


def solve(spec: TFIMSpec) -> EDResult:
    """Diagonalise the Hamiltonian and measure the ground-state observables.

    Args:
        spec: The problem.

    Returns:
        The energy together with the two observables the ground state supports,
        and a record of which solver ran.

    Raises:
        ValueError: If the spec exceeds the size cap.

    Examples:
        At zero field the ground state is fully ordered, so every bond is
        satisfied and the transverse magnetisation vanishes:

        >>> from src.physics.model import TFIMSpec
        >>> result = solve(TFIMSpec(n_sites=4, coupling=1.0, field=0.0))
        >>> round(result.energy, 12), round(result.zz_correlation, 12)
        (-4.0, 1.0)
    """
    _require_supported(spec)
    matrix = hamiltonian(spec)
    energy, state, solver = _ground_state(matrix)
    return EDResult(
        energy=energy,
        energy_density=energy / spec.n_sites,
        transverse_magnetisation=transverse_magnetisation(spec, state),
        zz_correlation=zz_correlation(spec, state),
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


def zz_correlation(spec: TFIMSpec, state: FloatArray) -> float:
    """Measure ``(1/n_bonds) sum_bonds <sigma_z_i sigma_z_{i+1}>`` on a given state.

    Args:
        spec: The problem the state belongs to.
        state: A normalised real state vector of length ``2**L``.

    Returns:
        The mean nearest-neighbour Ising correlation, ``1`` for a fully ordered
        state and ``0`` for an uncorrelated one.
    """
    weights = state**2
    spins = spin_table(spec.n_sites).astype(np.float64)
    total = sum(float(np.dot(weights, spins[:, i] * spins[:, j])) for i, j in bonds(spec))
    return total / spec.n_bonds
