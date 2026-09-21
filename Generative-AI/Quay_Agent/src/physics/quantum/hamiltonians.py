r"""Pauli-term representation of every model the agent is asked about.

States problems; does not solve them. A :class:`PauliSum` travels from the
formaliser through the ansatz builder to the estimator, so one Hamiltonian is
checked against sparse linear algebra, compiled to a Qiskit circuit and priced in
shots without three transcriptions of the same physics.

Two conventions, fixed once here because both are common sources of a mismatch
against exact diagonalisation.

Paulis throughout, eigenvalues :math:`\pm 1`, matching :mod:`src.physics.model`:

.. math::

    \hat H = -J \sum_i \hat\sigma^z_i \hat\sigma^z_{i+1}
             - g \sum_i \hat\sigma^z_i
             - h \sum_i \hat\sigma^x_i

Spin operators :math:`\hat S^a = \hat\sigma^a / 2` appear nowhere.

Qubit ``i`` is bit ``i`` of the basis index and :math:`|0\rangle` is the
:math:`+1` eigenstate of :math:`\hat\sigma^z` -- Qiskit\'s little-endian order, so
:meth:`PauliSum.to_labels` needs no reversal. It differs from
:mod:`src.physics.reference.exact_diagonalisation`, and
the Hamiltonian test compares the two through the bit-flip permutation
that relates them rather than assuming it away.

The longitudinal field ``g`` defaults to zero. Qiskit is imported lazily by
:meth:`PauliSum.to_sparse_pauli_op`, so the problem statement stays importable
with no quantum SDK present.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from src.physics.lattice import Lattice
from src.physics.model import MAX_SITES_SPARSE, BoundaryCondition

if TYPE_CHECKING:  # pragma: no cover - import exists for annotations only
    from qiskit.quantum_info import SparsePauliOp

IntArray = NDArray[np.int64]
ComplexArray = NDArray[np.complex128]

PauliLetter = Literal["X", "Y", "Z"]
"""A single non-identity Pauli factor.

Identity is represented by *absence* rather than by an ``"I"`` letter: a term
stores only the qubits it acts on. That makes weight, support and qubit-wise
commutation direct properties of the stored data rather than string searches,
and it means the same term object is valid on a 6-qubit and a 60-qubit register.
"""

MEASUREMENT_DEFAULT: PauliLetter = "Z"
"""Basis assigned to a qubit that a measurement group leaves unconstrained.

An idle qubit can be read out in any basis; naming the computational one keeps
the returned basis string total, so a caller never has to handle ``None``.
"""

COEFFICIENT_TOLERANCE = 1e-12
"""Below this magnitude a coefficient is dropped rather than carried.

A term with a zero coefficient is not wrong, it is *expensive*: it survives into
the measurement grouping, is allocated shots, and buys a number known in advance
to be zero. Dropping it at construction is the cheapest place to do it.
"""


def _parity(values: IntArray) -> IntArray:
    """Parity of the set bits of each element -- ``popcount(v) % 2``, vectorised.

    Written out rather than taken from ``np.bitwise_count``, which arrived in
    NumPy 2.0 while ``pyproject.toml`` admits 1.26. Six XOR-shifts fold the 64
    bits down into bit zero.

    Args:
        values: Non-negative 64-bit integers.

    Returns:
        ``0`` or ``1`` per element, with the same shape as the input.
    """
    folded = values.copy()
    for shift in (32, 16, 8, 4, 2, 1):
        folded ^= folded >> shift
    return folded & 1


@dataclass(frozen=True, slots=True)
class PauliTerm:
    r"""One weighted Pauli string, such as :math:`-J\,\hat\sigma^z_0\hat\sigma^z_1`.

    Attributes:
        coefficient: The real weight. Real by construction, not by convention:
            every term is Hermitian, so a sum of them with real coefficients is
            Hermitian too, and a Hamiltonian that could quietly acquire an
            imaginary part is one whose "energy" would need checking every time
            it was printed.
        operators: The qubits acted on, as ``(qubit, letter)`` pairs sorted by
            qubit index with no repeats. Sorting is enforced rather than assumed
            so that two spellings of the same term compare equal and collapse in
            :meth:`PauliSum.simplified`.
    """

    coefficient: float
    operators: tuple[tuple[int, PauliLetter], ...] = ()

    def __post_init__(self) -> None:
        """Reject a term that could not be measured or could not be compared.

        Raises:
            ValueError: If a qubit index is negative, repeated, out of order, or
                a letter is not one of ``X``, ``Y``, ``Z``.
        """
        previous = -1
        for qubit, letter in self.operators:
            if qubit < 0:
                raise ValueError(f"qubit index must be non-negative, got {qubit}")
            if qubit <= previous:
                raise ValueError(
                    f"operators must be sorted by qubit with no repeats, got {self.operators}"
                )
            if letter not in get_args(PauliLetter):
                raise ValueError(f"{letter!r} is not one of {get_args(PauliLetter)}")
            previous = qubit

    @classmethod
    def from_mapping(cls, coefficient: float, operators: Mapping[int, PauliLetter]) -> PauliTerm:
        """Build a term from a ``{qubit: letter}`` mapping, sorting it on the way in.

        Args:
            coefficient: The real weight.
            operators: Which Pauli acts on which qubit. Order is irrelevant.

        Returns:
            The term, with its operators in canonical order.

        Examples:
            >>> PauliTerm.from_mapping(-1.0, {1: "Z", 0: "Z"}).operators
            ((0, 'Z'), (1, 'Z'))
        """
        return cls(coefficient, tuple(sorted(operators.items())))

    @property
    def is_identity(self) -> bool:
        """Whether this term acts on no qubit at all -- a constant energy offset."""
        return not self.operators

    @property
    def weight(self) -> int:
        """How many qubits the string acts on non-trivially.

        The quantity that decides how hard a term is to measure on hardware: a
        weight-``k`` term needs ``k`` qubits read out in a common basis, and on a
        device with limited connectivity it may need them adjacent.
        """
        return len(self.operators)

    @property
    def support(self) -> frozenset[int]:
        """The set of qubits the string touches."""
        return frozenset(qubit for qubit, _ in self.operators)

    def label(self, n_qubits: int) -> str:
        """Render as a dense Pauli string in Qiskit's little-endian order.

        Args:
            n_qubits: Width of the register to pad to.

        Returns:
            A string of length ``n_qubits`` whose **rightmost** character is
            qubit ``0``. That reversal is Qiskit's convention, and doing it here
            once is the alternative to doing it wrongly at three call sites.

        Raises:
            ValueError: If the term acts on a qubit outside the register.

        Examples:
            >>> PauliTerm.from_mapping(1.0, {0: "Z", 1: "Z"}).label(4)
            'IIZZ'
            >>> PauliTerm.from_mapping(1.0, {2: "X"}).label(4)
            'IXII'
        """
        letters = ["I"] * n_qubits
        for qubit, letter in self.operators:
            if qubit >= n_qubits:
                raise ValueError(f"term acts on qubit {qubit} of a {n_qubits}-qubit register")
            letters[qubit] = letter
        return "".join(reversed(letters))

    def scaled(self, factor: float) -> PauliTerm:
        """Return the same string with its coefficient multiplied by ``factor``.

        Args:
            factor: The multiplier.

        Returns:
            A new term; the original is frozen and unchanged.
        """
        return PauliTerm(self.coefficient * factor, self.operators)

    def qubitwise_commutes_with(self, other: PauliTerm) -> bool:
        """Whether both strings can be measured in a single readout basis.

        Qubit-wise commutation is stricter than commutation -- ``XX`` and ``YY``
        commute but need different bases -- and it is the stricter notion that
        decides how many circuits a Hamiltonian costs, because one basis per
        qubit is what a device can actually apply before measuring.

        Args:
            other: The term to compare against.

        Returns:
            ``True`` if the two agree on every qubit where both are non-identity.

        Examples:
            The Ising terms all live in the ``z`` basis, and the field terms all
            in the ``x`` basis, which is why the whole chain costs two settings:

            >>> zz = PauliTerm.from_mapping(1.0, {0: "Z", 1: "Z"})
            >>> z = PauliTerm.from_mapping(1.0, {1: "Z"})
            >>> x = PauliTerm.from_mapping(1.0, {1: "X"})
            >>> zz.qubitwise_commutes_with(z), zz.qubitwise_commutes_with(x)
            (True, False)
        """
        mine = dict(self.operators)
        return all(mine.get(qubit, letter) == letter for qubit, letter in other.operators)


@dataclass(frozen=True, slots=True)
class MeasurementGroup:
    """A set of terms that one circuit and one readout basis can measure together.

    Attributes:
        basis: One Pauli letter per qubit, the basis that qubit is rotated into
            before measurement. Qubits no term in the group constrains carry
            :data:`MEASUREMENT_DEFAULT`.
        operator: The terms themselves, as a sum in their own right, so that a
            group can be priced, matrixed and printed exactly like the whole.
    """

    basis: tuple[PauliLetter, ...]
    operator: PauliSum


@dataclass(frozen=True, slots=True)
class PauliSum:
    r"""A Hamiltonian as a weighted sum of Pauli strings.

    Attributes:
        n_qubits: Width of the register the terms are defined on. Carried
            explicitly rather than inferred from the highest index used, because
            a Hamiltonian with no term on the last qubit is a perfectly ordinary
            Hamiltonian and inferring the width would silently shrink it.
        terms: The terms. Not automatically simplified -- see
            :meth:`simplified` -- because the *unsimplified* form is what shows
            the physics: ``-J`` on every bond and ``-h`` on every site is how the
            model was written down, and collapsing it is a decision the caller
            makes when it is about to matter.

    Examples:
        >>> h = ising_chain(n_sites=3, coupling=1.0, transverse_field=0.5)
        >>> h.n_qubits, len(h.terms)
        (3, 5)
        >>> h.coefficient_l1()
        3.5
    """

    n_qubits: int
    terms: tuple[PauliTerm, ...]

    def __post_init__(self) -> None:
        """Reject a register that is too small for the terms it carries.

        Raises:
            ValueError: If ``n_qubits`` is below one, or any term acts outside it.
        """
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be at least 1, got {self.n_qubits}")
        for term in self.terms:
            for qubit, _ in term.operators:
                if qubit >= self.n_qubits:
                    raise ValueError(
                        f"term acts on qubit {qubit} of a {self.n_qubits}-qubit register"
                    )

    @classmethod
    def from_terms(cls, n_qubits: int, terms: Iterable[PauliTerm]) -> PauliSum:
        """Build a sum, discarding terms whose coefficient is numerically zero.

        Args:
            n_qubits: Width of the register.
            terms: The terms, in any order.

        Returns:
            The sum, with terms below :data:`COEFFICIENT_TOLERANCE` removed.
        """
        kept = tuple(t for t in terms if abs(t.coefficient) > COEFFICIENT_TOLERANCE)
        return cls(n_qubits, kept)

    def __add__(self, other: PauliSum) -> PauliSum:
        """Concatenate two sums over the same register.

        Args:
            other: The sum to add.

        Returns:
            A sum carrying every term of both, unsimplified.

        Raises:
            ValueError: If the two are defined on different numbers of qubits.
                Adding Hamiltonians of different widths is always a mistake, and
                a silent broadcast would produce a plausible wrong answer.
        """
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"cannot add a {self.n_qubits}-qubit and a {other.n_qubits}-qubit Hamiltonian"
            )
        return PauliSum(self.n_qubits, self.terms + other.terms)

    def scaled(self, factor: float) -> PauliSum:
        """Multiply every coefficient by ``factor``.

        Args:
            factor: The multiplier. ``-1`` turns a quantity to maximise into one
                a minimiser can consume, which is the whole of the MaxCut sign
                convention (see :func:`maxcut_cost`).

        Returns:
            A new sum.
        """
        return PauliSum(self.n_qubits, tuple(t.scaled(factor) for t in self.terms))

    def simplified(self) -> PauliSum:
        """Combine repeated Pauli strings and drop the ones that cancel.

        Returns:
            An equivalent sum with one term per distinct string, in first-seen
            order so that the result stays readable rather than hash-ordered.

        Examples:
            >>> zz = PauliTerm.from_mapping(1.0, {0: "Z", 1: "Z"})
            >>> len(PauliSum(2, (zz, zz.scaled(-1.0))).simplified().terms)
            0
        """
        totals: dict[tuple[tuple[int, PauliLetter], ...], float] = {}
        for term in self.terms:
            totals[term.operators] = totals.get(term.operators, 0.0) + term.coefficient
        return PauliSum.from_terms(
            self.n_qubits, (PauliTerm(coefficient, ops) for ops, coefficient in totals.items())
        )

    @property
    def is_real(self) -> bool:
        r"""Whether the matrix has no imaginary part in the computational basis.

        True when every term carries an even number of :math:`\\hat\\sigma^y`
        factors, which holds for the whole Ising family and for ``XX + YY``. A
        real Hamiltonian has a real ground state, which halves the memory of a
        state-vector run and makes a real-arithmetic eigensolver legitimate.
        """
        return all(sum(1 for _, p in t.operators if p == "Y") % 2 == 0 for t in self.terms)

    def coefficient_l1(self) -> float:
        r"""Return :math:`\sum_\alpha |c_\alpha|` over the non-identity terms.

        This single number sets the shot budget. Give each term a share of the
        shots proportional to its weight and the variances sum to
        :math:`(\sum_\alpha |c_\alpha|)^2 / S`, so estimating the energy to
        precision :math:`\epsilon` takes :math:`S = (\sum_\alpha |c_\alpha|)^2 /
        \epsilon^2` -- an agent that wants to know what an answer will *cost*
        before running anything asks for this and squares it.

        It is a worst case, and a loose one: it charges every Pauli the full
        variance its :math:`\pm 1` spectrum allows, where the real variance on a
        prepared ground state is far smaller. Measured on a ten-spin chain, the
        true :math:`(\sum_g \sigma_g)^2` asks 12 to 31 times fewer shots across
        :math:`h = 0.2` to :math:`2`. The bound is still the right thing to price a
        plan with, because :math:`\sigma_g` is a property of the state the run has
        not prepared yet -- but it is loose by more than a noisy device's premium,
        so a total quoted from it is an over-estimate rather than a floor.

        The identity is excluded because it is known exactly and needs no shots
        -- including it would inflate every estimate by a constant that no
        measurement ever pays for.

        Returns:
            The sum of absolute coefficients over terms that act on at least one
            qubit.

        Examples:
            For the Ising chain this is :math:`JB + hN + gN`, where :math:`B` is
            the bond count -- :math:`N-1` for an open chain and :math:`N` for a
            ring. Linear in ``N`` either way, which is why the shot cost of a
            fixed *per-site* precision barely grows with the chain. The boundary
            is worth writing out rather than folding into ``N-1``: a ring costs
            one more bond's worth of measurement than a row of the same length,
            and quoting the open-chain form beside a periodic spec understates
            the budget by exactly that bond.

            >>> ising_chain(n_sites=4, coupling=1.0, transverse_field=1.0).coefficient_l1()
            7.0
        """
        return sum(abs(t.coefficient) for t in self.terms if not t.is_identity)

    def to_labels(self) -> list[tuple[str, float]]:
        """Render as ``(pauli string, coefficient)`` pairs.

        Returns:
            One pair per term, strings padded to :attr:`n_qubits` in Qiskit's
            little-endian order. This is the exact input format of
            ``SparsePauliOp.from_list``, and it is also the readable form to put
            in a log or a run card.
        """
        return [(t.label(self.n_qubits), t.coefficient) for t in self.terms]

    def measurement_groups(self) -> tuple[MeasurementGroup, ...]:
        r"""Partition the terms into sets that share one readout basis.

        Greedy first-fit against each group's accumulated basis: a term joins the
        first group that has not already claimed a different letter on any qubit
        it touches. Greedy is not optimal in general -- minimum clique cover is
        NP-hard -- but it is exactly optimal on the structure that matters here,
        and it is *why* it is worth doing: every :math:`\hat\sigma^z\hat\sigma^z`
        and :math:`\hat\sigma^z` term is diagonal in the same basis, and every
        :math:`\hat\sigma^x` term in another, so the entire mixed-field Ising
        chain costs **two** circuits per energy evaluation rather than
        :math:`2N-1`. That factor of ``N`` is bought by noticing a commutation
        structure, not by buying hardware.

        Returns:
            The groups, in the order the greedy pass created them. Identity
            terms need no measurement at all and are attached to the first group
            so that summing the groups still reproduces the whole Hamiltonian.

        Examples:
            >>> chain = ising_chain(n_sites=6, transverse_field=1.0, longitudinal_field=0.3)
            >>> len(chain.measurement_groups())
            2
        """
        bases: list[dict[int, PauliLetter]] = []
        members: list[list[PauliTerm]] = []
        for term in self.terms:
            for index, basis in enumerate(bases):
                if all(basis.get(q, letter) == letter for q, letter in term.operators):
                    basis.update(term.operators)
                    members[index].append(term)
                    break
            else:
                bases.append(dict(term.operators))
                members.append([term])
        return tuple(
            MeasurementGroup(
                basis=tuple(basis.get(q, MEASUREMENT_DEFAULT) for q in range(self.n_qubits)),
                operator=PauliSum(self.n_qubits, tuple(group)),
            )
            for basis, group in zip(bases, members, strict=True)
        )

    def to_matrix(self) -> csr_matrix:
        r"""Assemble the :math:`2^N \times 2^N` matrix, sparsely.

        Each Pauli string has exactly one non-zero entry per column, so the whole
        matrix is built from bit arithmetic rather than from ``n_qubits`` nested
        Kronecker products: a string flips the basis index by the mask of its
        :math:`\hat\sigma^x` and :math:`\hat\sigma^y` positions, and contributes a
        phase :math:`i^{n_y}(-1)^{|s \wedge m_{yz}|}` where :math:`m_{yz}` masks
        the qubits carrying :math:`\hat\sigma^y` or :math:`\hat\sigma^z`.

        This exists to be *checked*, not to be run: it is how a hand-built
        Hamiltonian is held against
        :mod:`src.physics.reference.exact_diagonalisation` at ``N = 4``, and how
        a circuit's state vector is scored during development. Nothing on the
        agent's path calls it.

        Returns:
            A Hermitian complex CSR matrix of dimension ``2**n_qubits``.
            Duplicate entries from repeated strings are summed by the COO
            constructor, so an unsimplified sum gives the same matrix as a
            simplified one.

        Raises:
            ValueError: If the register is wider than
                :data:`~src.physics.model.MAX_SITES_SPARSE` qubits. Nothing in the
                project runs above that, so a wider register here is a caller who
                has reached for a verification tool by mistake.
        """
        if self.n_qubits > MAX_SITES_SPARSE:
            raise ValueError(
                f"refusing to build a 2**{self.n_qubits} matrix; to_matrix is a verification "
                f"tool for registers up to {MAX_SITES_SPARSE} qubits, not an execution path"
            )
        dimension = 2**self.n_qubits
        states = np.arange(dimension, dtype=np.int64)
        rows: list[IntArray] = []
        cols: list[IntArray] = []
        values: list[ComplexArray] = []
        for term in self.terms:
            flip = np.int64(0)
            phase_mask = np.int64(0)
            n_y = 0
            for qubit, letter in term.operators:
                bit = np.int64(1) << np.int64(qubit)
                if letter in ("X", "Y"):
                    flip |= bit
                if letter in ("Y", "Z"):
                    phase_mask |= bit
                if letter == "Y":
                    n_y += 1
            signs = 1.0 - 2.0 * _parity(states & phase_mask).astype(np.float64)
            amplitude = term.coefficient * (1j**n_y) * signs
            rows.append(states ^ flip)
            cols.append(states)
            values.append(amplitude.astype(np.complex128))
        if not rows:
            return csr_matrix((dimension, dimension), dtype=np.complex128)
        matrix = coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(dimension, dimension),
            dtype=np.complex128,
        )
        return matrix.tocsr()

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """Convert to Qiskit's ``SparsePauliOp``, importing Qiskit only if called.

        The import is deferred so that the problem statement -- and every test of
        it -- stays runnable with no quantum SDK installed. Only the code that
        actually builds or runs a circuit pays for Qiskit.

        Returns:
            The same operator in Qiskit's representation, on the same number of
            qubits and in the same little-endian ordering.

        Raises:
            ImportError: If Qiskit is not installed, with the command to fix it.
        """
        try:
            from qiskit.quantum_info import SparsePauliOp
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                "Qiskit is needed to convert a Hamiltonian into a circuit operator. "
                "Install it with: uv add qiskit qiskit-aer"
            ) from error
        if not self.terms:
            return SparsePauliOp.from_list([("I" * self.n_qubits, 0.0)])
        return SparsePauliOp.from_list(self.to_labels(), num_qubits=self.n_qubits)


def chain_bonds(
    n_sites: int, boundary: BoundaryCondition = "open", distance: int = 1
) -> tuple[tuple[int, int], ...]:
    r"""List the coupled pairs of a one-dimensional lattice.

    Args:
        n_sites: Number of sites.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring. Open is
            the project default because it is what hardware looks like -- a real
            device has ends -- and periodic is used where a result is being held
            against the Pfeuty solution, which assumes a ring.
        distance: Separation of the coupled sites. ``1`` is nearest-neighbour;
            ``2`` gives the next-nearest-neighbour bonds of the frustrated
            :math:`J_1`--:math:`J_2` chain.

    Returns:
        Pairs ``(i, j)`` with ``i`` ascending. A ring of ``N`` sites has ``N``
        bonds at any distance; a segment has ``N - distance``, and none at all
        once the distance exceeds the chain.

    Raises:
        ValueError: If ``n_sites`` is below two or ``distance`` below one.

    Note:
        A two-site ring gives ``((0, 1), (1, 0))`` here -- the same pair twice,
        because that is what :math:`\sum_i \sigma^z_i \sigma^z_{i+1 \bmod L}` says
        at ``L = 2``, and it is the convention the closed form and the grader's
        sparse solver both agree with. :meth:`src.physics.lattice.Lattice.bonds`
        cannot express a doubled edge and refuses that size instead; see
        :data:`src.physics.lattice.TWO_SITE_RING`.

    Examples:
        >>> chain_bonds(4, "open")
        ((0, 1), (1, 2), (2, 3))
        >>> chain_bonds(4, "periodic")
        ((0, 1), (1, 2), (2, 3), (3, 0))
    """
    if n_sites < 2:
        raise ValueError(f"a chain needs at least 2 sites, got {n_sites}")
    if distance < 1:
        raise ValueError(f"bond distance must be at least 1, got {distance}")
    if boundary == "periodic":
        return tuple((i, (i + distance) % n_sites) for i in range(n_sites))
    return tuple((i, i + distance) for i in range(max(0, n_sites - distance)))


def ising_chain(
    n_sites: int,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: BoundaryCondition = "open",
    lattice: Lattice | None = None,
) -> PauliSum:
    r"""Build the mixed-field Ising chain.

    .. math::

        \hat H = -J \sum_{\langle ij \rangle} \hat\sigma^z_i \hat\sigma^z_j
                 - g \sum_i \hat\sigma^z_i
                 - h \sum_i \hat\sigma^x_i

    The whole difficulty axis of the project is the third term. At ``g = 0`` this
    is the transverse-field Ising model: Jordan-Wigner takes it to free fermions,
    it is solved in :math:`O(N)` by :mod:`src.physics.reference.free_fermions`, and any honest
    feasibility verdict on it is "no quantum advantage, trivially". Switching on
    ``g`` puts a field along the *coupling* axis, which under Jordan-Wigner is a
    non-local string operator; integrability dies, and the question of whether a
    quantum computer helps becomes one worth asking. The circuit pays almost
    nothing for it -- one layer of :math:`\hat R^z` per ansatz layer, no
    two-qubit depth at all -- which is precisely what makes it the right knob.

    Args:
        n_sites: Number of spins, equal to the number of qubits.
        coupling: The Ising coupling ``J``, strictly positive (ferromagnetic).
        transverse_field: The transverse field ``h``, non-negative.
        longitudinal_field: The longitudinal field ``g``, any sign. A negative
            ``g`` is the same physics with the two ordered states exchanged.
        boundary: Lattice boundary condition. Ignored when ``lattice`` is given,
            which carries its own.
        lattice: A geometry to take the bonds from, in place of this module's own
            chain generator. It is the only argument that makes this function
            two-dimensional, because the transverse and longitudinal terms are one
            operator per site and know nothing about who is connected to whom -- so
            a square or triangular lattice changes the first sum and leaves the other
            two exactly as they were.

            The bond list then comes from :meth:`src.physics.lattice.Lattice.bonds`
            rather than from :func:`chain_bonds`, and the same list reaches the
            grader's solver. What keeps the two routes independent in two dimensions
            is the *algebra* and not the edge list -- this one builds Kronecker
            products of two-by-two matrices, the grader's acts on basis integers with
            an XOR. See :func:`src.physics.reference.exact_diagonalisation.bonds`,
            which states the same thing from the other side.

    Returns:
        The Hamiltonian, with zero-coefficient families omitted entirely rather
        than carried as terms that would be measured and found to be zero.

    Raises:
        ValueError: If ``coupling`` is not positive, ``transverse_field`` is
            negative, or ``n_sites`` is below two. The first two mirror the
            validation in :class:`src.physics.model.TFIMSpec`, so a spec the
            the reference solver would refuse cannot be turned into a circuit here. Also
            if ``lattice`` describes a different number of sites from ``n_sites``.

    Examples:
        A three-site open chain has two bonds and three field terms:

        >>> chain = ising_chain(3, coupling=1.0, transverse_field=0.5)
        >>> chain.to_labels()
        [('IZZ', -1.0), ('ZZI', -1.0), ('IIX', -0.5), ('IXI', -0.5), ('XII', -0.5)]

        At zero field the ground state is the fully ordered product state, so the
        energy is minus the number of satisfied bonds:

        >>> import numpy as np
        >>> chain = ising_chain(6, coupling=1.0, transverse_field=0.0)
        >>> float(np.linalg.eigvalsh(chain.to_matrix().toarray())[0].real)
        -5.0

        A geometry replaces the bonds and nothing else. A ``2 x 2`` square has
        four bonds where a four-site open chain has three, and the field terms are
        untouched:

        >>> from src.physics.lattice import Lattice
        >>> square = ising_chain(4, transverse_field=0.5, lattice=Lattice("square", 2, 2))
        >>> square.to_labels()[:4]
        [('IIZZ', -1.0), ('IZIZ', -1.0), ('ZIZI', -1.0), ('ZZII', -1.0)]
    """
    if coupling <= 0.0:
        raise ValueError(f"coupling J must be strictly positive, got {coupling}")
    if transverse_field < 0.0:
        raise ValueError(f"transverse field h must be non-negative, got {transverse_field}")
    if lattice is not None and lattice.n_sites != n_sites:
        raise ValueError(
            f"the geometry has {lattice.n_sites} sites and {n_sites} were asked for; "
            "they must agree"
        )
    edges = chain_bonds(n_sites, boundary) if lattice is None else lattice.bonds()
    terms = [PauliTerm.from_mapping(-coupling, {i: "Z", j: "Z"}) for i, j in edges]
    terms += [PauliTerm.from_mapping(-transverse_field, {i: "X"}) for i in range(n_sites)]
    terms += [PauliTerm.from_mapping(-longitudinal_field, {i: "Z"}) for i in range(n_sites)]
    return PauliSum.from_terms(n_sites, terms)


def maxcut_cost(n_nodes: int, edges: Sequence[tuple[int, int, float]]) -> PauliSum:
    r"""Build the MaxCut cut-value operator of a weighted graph.

    .. math::

        \hat H_C = \sum_{(i,j) \in E} w_{ij}
                   \frac{1 - \hat\sigma^z_i \hat\sigma^z_j}{2}

    Diagonal, so its eigenvalues *are* the cut values of the assignments and
    finding the maximum cut is finding the highest-energy state of a classical
    antiferromagnetic Ising model on the graph. MaxCut is therefore an Ising
    problem rather than a neighbour of one, which is why it belongs in this
    project at all.

    Sign: this returns the quantity to *maximise*. Every minimiser in the
    project -- VQE, the imaginary-time flow, the classical optimisers -- wants
    the negation, ``maxcut_cost(...).scaled(-1.0)``. The convention is stated
    once here and the sign is never flipped implicitly, because a MaxCut result
    that is wrong by a sign looks like a working algorithm producing a poor cut.

    The graph is passed in as edges rather than generated here: instance
    generation and the brute-force optimum live in ``src.physics.maxcut``, which
    holds the reference answers and which nothing in this package may import.

    Args:
        n_nodes: Number of graph vertices, equal to the number of qubits.
        edges: Triples ``(i, j, weight)``. Repeated edges are summed rather than
            rejected, which is what a multigraph means.

    Returns:
        The cut operator, including the constant :math:`\sum_{ij} w_{ij}/2` as a
        single identity term. That constant is half the total edge weight and is
        the reason a MaxCut expectation value is never near zero.

    Raises:
        ValueError: If an endpoint lies outside the graph or an edge is a
            self-loop. A self-loop contributes ``w(1 - 1)/2 = 0`` and so is
            silently free, which makes it exactly the kind of input that hides a
            construction bug upstream.

    Examples:
        A single edge: constant one half, minus one half on the pair.

        >>> maxcut_cost(2, [(0, 1, 1.0)]).to_labels()
        [('II', 0.5), ('ZZ', -0.5)]

        The triangle cannot be cut fully -- its best cut is 2 of 3 edges:

        >>> import numpy as np
        >>> triangle = maxcut_cost(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
        >>> float(np.linalg.eigvalsh(triangle.to_matrix().toarray())[-1].real)
        2.0
    """
    terms: list[PauliTerm] = []
    total_weight = 0.0
    for i, j, weight in edges:
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError(f"edge ({i}, {j}) lies outside a {n_nodes}-node graph")
        if i == j:
            raise ValueError(f"self-loop at node {i} contributes nothing and is never intended")
        total_weight += weight
        terms.append(PauliTerm.from_mapping(-weight / 2.0, {min(i, j): "Z", max(i, j): "Z"}))
    return PauliSum.from_terms(n_nodes, [PauliTerm(total_weight / 2.0), *terms]).simplified()
