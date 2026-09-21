r"""What makes the *quantum* Ising chain quantum, checked rather than asserted.

Every statement this module supports is one a textbook makes in a line of algebra
and this project refuses to make in a line of prose: the two terms of the
Hamiltonian do not commute, the on-site Pauli operators anticommute, the parity
operator is a symmetry, and the ground state obeys -- and in one limit *saturates*
-- the uncertainty relation. Here each one is evaluated on the matrices themselves
and reported as a number, because a page of identities nobody computed is exactly
the kind of claim the rest of this application exists to avoid making.

Dense, and deliberately. Every other solver here is sparse and careful; this
one builds full ``2**L`` matrices with :func:`numpy.kron` and is capped at
:data:`MAX_SITES` spins. Commutator norms are matrix-wide quantities rather than
one vector's worth of information, and at ``L = 8`` a dense operator is a
256x256 array -- microseconds, and far simpler to read than the sparse equivalent.
The cap is what keeps that true.

It builds its own Hamiltonian, and then checks it. Nothing here reuses
the basis of :mod:`src.physics.reference.exact_diagonalisation`, because the two
would then share a convention and a mistake in it would cancel.
:attr:`Quantumness.energy_disagreement` is this module's ground-state energy
against the free-fermion closed form -- the same cross-check every other number
in the project goes through, applied to the matrices these identities are
evaluated on.

Units. Spin operators are :math:`\hat S^a = (\hbar/2)\hat\sigma^a`, and
:math:`\hbar = 1` throughout -- a choice of units, not a claim that Planck's
constant is one. :data:`HBAR_SI` is what it is in joule-seconds, for anyone
converting the bound below into an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.physics.model import TFIMSpec
from src.physics.reference import exact_diagonalisation, free_fermions

ComplexArray = NDArray[np.complex128]

MAX_SITES = 8
"""Longest chain this module will build operators for.

Dense ``2**L x 2**L`` complex matrices: eight spins is a 256x256 array and a
handful of them, which is nothing. Twelve would be 4096x4096 complex, which is
several hundred megabytes per operator and would turn a page tab into a hang.
"""

HBAR = 1.0
r"""Planck's constant in the units this project works in.

One, so that :math:`\hat S^a = (\hbar/2)\hat\sigma^a` and energies are read
directly in units of ``J``. Stated as a constant rather than left implicit because
the uncertainty bound below is proportional to it, and a reader is entitled to see
where it went.
"""

HBAR_SI = 1.054571817e-34
r"""The same constant in joule-seconds, exactly, by the 2019 SI definition.

Carried so the page can say what :math:`\Delta S^y \Delta S^z \ge \hbar^2/4` is in
units somebody could measure in. Nothing computes with it.
"""

PAULI: dict[str, ComplexArray] = {
    "i": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}
r"""The Pauli matrices and the identity, keyed by axis.

:math:`\hat\sigma^y` is the only complex one, and it is the reason this module
works in ``complex128`` throughout: the commutator
:math:`[\hat\sigma^x, \hat\sigma^y] = 2i\hat\sigma^z` cannot be represented
otherwise, and a real-valued shortcut would quietly drop the ``i`` that makes the
algebra quantum.
"""


DEFAULT_CONFIGURATIONS = 8
"""How many spin configurations a cartoon of the ground state draws.

Eight rows fit on a screen and, at the fields this page is used at, carry most of
the weight. The number that says whether they do is
:attr:`Superposition.shown_weight`, which is reported rather than assumed -- a
cartoon that quietly hid nine tenths of the state would be a nicer picture and a
worse one.
"""


def _require_supported(spec: TFIMSpec) -> None:
    """Refuse a chain too long to build dense operators for.

    Args:
        spec: The problem.

    Raises:
        ValueError: If the chain is longer than :data:`MAX_SITES`.
    """
    if spec.n_sites > MAX_SITES:
        raise ValueError(
            f"L={spec.n_sites} is above the dense-operator cap of {MAX_SITES}; "
            "these identities are matrix-wide and the matrices are full."
        )


def site_operator(n_sites: int, axis: str, site: int) -> ComplexArray:
    r"""Place one Pauli matrix on one site of the chain.

    Args:
        n_sites: Chain length ``L``.
        axis: ``"x"``, ``"y"``, ``"z"`` or ``"i"``.
        site: Which spin it acts on, from the left.

    Returns:
        :math:`I \otimes \cdots \otimes \hat\sigma^a \otimes \cdots \otimes I`,
        a ``2**L x 2**L`` array. Site ``0`` is the leftmost factor; the convention
        only has to be self-consistent, and :attr:`Quantumness.energy_disagreement`
        is what confirms it is.
    """
    matrices = [PAULI["i"]] * n_sites
    matrices[site] = PAULI[axis]
    out: ComplexArray = matrices[0]
    for factor in matrices[1:]:
        out = np.kron(out, factor).astype(np.complex128)
    return out


def parity(n_sites: int) -> ComplexArray:
    r"""The :math:`\mathbb{Z}_2` symmetry operator :math:`\hat P = \prod_i \hat\sigma^x_i`.

    Args:
        n_sites: Chain length.

    Returns:
        The product over every site. Flipping every spin at once leaves the
        Hamiltonian alone -- the Ising term has two ``z`` factors per bond, so both
        signs cancel -- which is the symmetry the ordered phase breaks and the
        reason the spectrum splits into sectors at all.
    """
    out: ComplexArray = PAULI["x"]
    for _ in range(n_sites - 1):
        out = np.kron(out, PAULI["x"]).astype(np.complex128)
    return out


def terms(spec: TFIMSpec) -> tuple[ComplexArray, ComplexArray]:
    r"""The two halves of the Hamiltonian, separately.

    Args:
        spec: The chain.

    Returns:
        :math:`(\hat H_{ZZ}, \hat H_X)` with
        :math:`\hat H_{ZZ} = -J\sum_{\langle ij\rangle} \hat\sigma^z_i \hat\sigma^z_j`
        and :math:`\hat H_X = -h\sum_i \hat\sigma^x_i`. Kept apart because the
        interesting quantity is the commutator *between* them: their sum is a
        matrix, while their failure to commute is the physics.
    """
    _require_supported(spec)
    dimension = 2**spec.n_sites
    ising = np.zeros((dimension, dimension), dtype=np.complex128)
    field = np.zeros((dimension, dimension), dtype=np.complex128)
    for left, right in exact_diagonalisation.bonds(spec):
        ising -= spec.coupling * (
            site_operator(spec.n_sites, "z", left) @ site_operator(spec.n_sites, "z", right)
        )
    for site in range(spec.n_sites):
        field -= spec.field * site_operator(spec.n_sites, "x", site)
    return ising, field


def commutator(left: ComplexArray, right: ComplexArray) -> ComplexArray:
    """``[A, B] = AB - BA``."""
    return left @ right - right @ left


def anticommutator(left: ComplexArray, right: ComplexArray) -> ComplexArray:
    """``{A, B} = AB + BA``."""
    return left @ right + right @ left


def norm(matrix: ComplexArray) -> float:
    """The Frobenius norm, as a plain float.

    Args:
        matrix: Any operator.

    Returns:
        Its norm. Used as "is this operator zero?", which is a question about the
        whole matrix and cannot be answered by one expectation value: an operator
        can annihilate a particular state and be nowhere near zero.
    """
    return float(np.linalg.norm(matrix))


def _spectrum(spec: TFIMSpec) -> tuple[ComplexArray, ComplexArray, float, ComplexArray]:
    r"""Diagonalise the chain once, and hand back the pieces its callers need.

    Args:
        spec: The chain.

    Returns:
        :math:`(\hat H_{ZZ}, \hat H_X, E_0, |\psi_0\rangle)`. One function so
        that the identities and the pictures are looking at the *same* ground
        state; two separate diagonalisations would agree, but nothing would say so.

    Raises:
        ValueError: If the chain is too long for dense operators.
    """
    _require_supported(spec)
    ising, field = terms(spec)
    values, vectors = np.linalg.eigh(ising + field)
    return ising, field, float(values[0]), vectors[:, 0]


def configuration_spins(n_sites: int, index: int) -> tuple[int, ...]:
    r"""Read a basis state's index back as a row of up and down spins.

    The basis is the one :func:`site_operator` builds: successive
    :func:`numpy.kron` factors, so site ``0`` is the most significant bit and
    :math:`\hat\sigma^z = \mathrm{diag}(1, -1)` makes a zero bit an up spin.

    Args:
        n_sites: Chain length ``L``.
        index: A row of the state vector, ``0 <= index < 2**L``.

    Returns:
        ``+1`` for up and ``-1`` for down, site ``0`` first.

    Examples:
        Basis state ``2`` of two spins is ``down, up`` -- which is exactly what
        the diagonals in :func:`site_operator` say it should be:

        >>> configuration_spins(2, 2)
        (-1, 1)
        >>> configuration_spins(3, 0)
        (1, 1, 1)
    """
    return tuple(1 if (index >> (n_sites - 1 - site)) & 1 == 0 else -1 for site in range(n_sites))


@dataclass(frozen=True, slots=True)
class Configuration:
    r"""One arrangement of the spins, and how much of the ground state it is.

    Attributes:
        spins: ``+1`` for up and ``-1`` for down, site ``0`` first.
        probability: :math:`|\langle \text{this} | \psi_0 \rangle|^2`. A
            probability and not an amplitude, because the sign of an amplitude
            depends on a phase convention and this is what a measurement would
            give.
    """

    spins: tuple[int, ...]
    probability: float

    def domain_walls(self, *, periodic: bool = True) -> int:
        """Count the bonds whose two spins disagree.

        The Ising term pays ``2J`` for each of these, so this is the *energy* of
        the configuration in the classical limit -- and the excitations of the
        ordered phase are these walls moving. Sorting a cartoon by this number is
        therefore sorting it by physics rather than by appearance.

        Args:
            periodic: Whether the last spin is also a neighbour of the first.

        Returns:
            How many neighbouring pairs point in opposite directions.

        Examples:
            >>> Configuration((1, 1, 1, 1), 0.5).domain_walls()
            0
            >>> Configuration((1, 1, -1, -1), 0.5).domain_walls()
            2
            >>> Configuration((1, 1, -1, -1), 0.5).domain_walls(periodic=False)
            1
        """
        # Deliberately ragged: pairing a sequence with its own tail is how the
        # bonds are enumerated, and there is one fewer bond than there are spins.
        bonds = list(zip(self.spins, self.spins[1:], strict=False))
        if periodic and len(self.spins) > 2:
            bonds.append((self.spins[-1], self.spins[0]))
        return sum(1 for left, right in bonds if left != right)


@dataclass(frozen=True, slots=True)
class Superposition:
    r"""The ground state as a picture rather than as a number.

    A commutator norm says *that* the ground state cannot be a configuration of
    spins. This says what it is instead: which arrangements carry the weight, how
    many of them there effectively are, and where one spin ends up pointing.

    Attributes:
        spec: The chain.
        configurations: The heaviest arrangements, most probable first.
        shown_weight: How much of the state those add up to, in ``[0, 1]``. The
            honesty of the cartoon, reported so a reader can distrust it.
        effective_count: :math:`1 / \sum_i p_i^2`, the participation ratio over
            *all* :math:`2^L` configurations. Read it as "the ground state is
            roughly this many spin arrangements at once": ``1`` in the classical
            limit, and :math:`2^L` when the field has won outright.
        arrow: :math:`(\langle\hat\sigma^x\rangle, \langle\hat\sigma^y\rangle,
            \langle\hat\sigma^z\rangle)` on site ``0`` -- where a single spin
            points, if it points anywhere.
    """

    spec: TFIMSpec
    configurations: tuple[Configuration, ...]
    shown_weight: float
    effective_count: float
    arrow: tuple[float, float, float]

    @property
    def arrow_length(self) -> float:
        r"""How long that single-spin arrow is, in :math:`[0, 1]`.

        Returns:
            :math:`|\vec r|` for the one-site density matrix
            :math:`\hat\rho = (\hat I + \vec r \cdot \vec{\hat\sigma})/2`. One
            means the spin is in a state of its own; **zero means it has no
            direction at all**, because everything it knows is stored in its
            correlations with its neighbours. That shrinking arrow is
            entanglement, drawn.
        """
        x, y, z = self.arrow
        return float(np.sqrt(x * x + y * y + z * z))

    @property
    def full_count(self) -> int:
        """How many configurations exist to be superposed, :math:`2^L`."""
        return 2**self.spec.n_sites


def superposition(spec: TFIMSpec, limit: int = DEFAULT_CONFIGURATIONS) -> Superposition:
    r"""Look at the ground state itself, in the basis of spin configurations.

    Args:
        spec: The chain. Must be at most :data:`MAX_SITES` spins.
        limit: How many configurations to return.

    Returns:
        The heaviest arrangements and the summary numbers around them.

    Raises:
        ValueError: If the chain is too long for dense operators.

    Examples:
        Switch the field off and the Hamiltonian is diagonal: the ground states
        *are* configurations, the solver returns one of them, and there is no
        superposition to draw. That is the classical limit, and it is why the
        cartoon is worth drawing at all.

        >>> from src.physics.model import TFIMSpec
        >>> flat = superposition(TFIMSpec(n_sites=4, coupling=1.0, field=0.0))
        >>> round(flat.effective_count, 6), flat.configurations[0].domain_walls()
        (1.0, 0)

        Let the field win and every one of the :math:`2^L` arrangements carries
        equal weight, because each spin is then separately along ``x``:

        >>> strong = superposition(TFIMSpec(n_sites=4, coupling=1.0, field=50.0))
        >>> round(strong.effective_count, 1), strong.full_count
        (16.0, 16)
        >>> round(strong.arrow_length, 3)
        1.0

        In between it is neither, which is the whole point:

        >>> middle = superposition(TFIMSpec(n_sites=4, coupling=1.0, field=1.0))
        >>> 1.0 < middle.effective_count < 16.0, middle.arrow_length < 1.0
        (True, True)
    """
    _, _, _, state = _spectrum(spec)
    weights = np.abs(state) ** 2
    heaviest = np.argsort(weights)[::-1][:limit]
    chosen = tuple(
        Configuration(
            spins=configuration_spins(spec.n_sites, int(index)),
            probability=float(weights[index]),
        )
        for index in heaviest
    )

    def expectation(operator: ComplexArray) -> float:
        """One expectation value in this ground state, real by construction."""
        return float(np.real(np.vdot(state, operator @ state)))

    return Superposition(
        spec=spec,
        configurations=chosen,
        shown_weight=float(np.sum(weights[heaviest])),
        # Normalisation is exact to machine precision here, so the sum of squares
        # needs no guard: it is never zero.
        effective_count=float(1.0 / np.sum(weights**2)),
        arrow=(
            expectation(site_operator(spec.n_sites, "x", 0)),
            expectation(site_operator(spec.n_sites, "y", 0)),
            expectation(site_operator(spec.n_sites, "z", 0)),
        ),
    )


@dataclass(frozen=True, slots=True)
class Quantumness:
    r"""Five quantum statements about one chain, each with its number.

    Attributes:
        spec: The chain these were measured on.
        energy: This module's own ground-state energy, from its own matrices.
        energy_disagreement: That energy against the free-fermion closed form, or
            ``None`` where the closed form does not apply. The cross-check on
            everything else here.
        terms_commutator: :math:`\|[\hat H_{ZZ}, \hat H_X]\|`. Non-zero is the
            whole reason the model is quantum.
        onsite_anticommutator: :math:`\|\{\hat\sigma^x_0, \hat\sigma^z_0\}\|`,
            which is zero: two operators on the same spin anticommute.
        offsite_commutator: :math:`\|[\hat\sigma^x_0, \hat\sigma^z_1]\|`, also
            zero: different spins are independent degrees of freedom.
        parity_commutator: :math:`\|[\hat H, \hat P]\|`, zero because the symmetry
            is exact at every field.
        parity_expectation: :math:`\langle \hat P \rangle` in the ground state,
            which is ``+1``: the ground state lives in the even sector.
        sigma_x: :math:`\langle \hat\sigma^x_0 \rangle`, the number that sets the
            uncertainty bound below -- and the same one the magnetisation panel
            plots.
        spread_y: :math:`\Delta \hat S^y` in the ground state, in units of
            :math:`\hbar`.
        spread_z: :math:`\Delta \hat S^z`, likewise.
        bound: :math:`(\hbar/2)|\langle \hat S^x \rangle|`, the right-hand side of
            the uncertainty relation for that pair.
    """

    spec: TFIMSpec
    energy: float
    energy_disagreement: float | None
    terms_commutator: float
    onsite_anticommutator: float
    offsite_commutator: float
    parity_commutator: float
    parity_expectation: float
    sigma_x: float
    spread_y: float
    spread_z: float
    bound: float

    @property
    def product(self) -> float:
        r""":math:`\Delta \hat S^y \, \Delta \hat S^z`, the left-hand side."""
        return self.spread_y * self.spread_z

    @property
    def slack(self) -> float:
        r"""How much room the uncertainty relation has left.

        Returns:
            Left-hand side minus right-hand side, which is never negative. It
            shrinks towards zero as the field grows: a chain deep in the
            disordered phase sits in an eigenstate of :math:`\hat\sigma^x`, and
            there the inequality is *saturated* rather than merely satisfied.
        """
        return self.product - self.bound

    @property
    def classical(self) -> bool:
        """Whether the two terms commute, which is the classical limit.

        Returns:
            ``True`` only when one of ``J`` and ``h`` is zero. Then the
            Hamiltonian is diagonal in some product basis, every eigenstate is a
            configuration of spins, and nothing here is quantum at all.
        """
        return self.terms_commutator < 1e-12


def measure(spec: TFIMSpec) -> Quantumness:
    r"""Evaluate every identity on the chain's own matrices.

    Args:
        spec: The chain. Must be at most :data:`MAX_SITES` spins.

    Returns:
        The measured statements.

    Raises:
        ValueError: If the chain is too long for dense operators.

    Examples:
        The two terms fail to commute whenever both are present, and commute
        exactly when either is switched off:

        >>> measure(TFIMSpec(n_sites=4, coupling=1.0, field=1.0)).classical
        False
        >>> measure(TFIMSpec(n_sites=4, coupling=1.0, field=0.0)).classical
        True

        On-site operators anticommute and different sites do not interfere:

        >>> found = measure(TFIMSpec(n_sites=4, coupling=1.0, field=1.0))
        >>> found.onsite_anticommutator < 1e-12, found.offsite_commutator < 1e-12
        (True, True)

        And the uncertainty relation holds, with room to spare at ``h = J``:

        >>> found.slack > 0.0
        True
    """
    ising, field, energy, state = _spectrum(spec)
    hamiltonian = ising + field

    def expectation(operator: ComplexArray) -> float:
        """One expectation value in the ground state, real by construction."""
        return float(np.real(np.vdot(state, operator @ state)))

    reference = (
        free_fermions.ground_state_energy(spec)
        if free_fermions.unsupported_reason(spec) is None
        else None
    )
    sigma_x = expectation(site_operator(spec.n_sites, "x", 0))
    sigma_y = expectation(site_operator(spec.n_sites, "y", 0))
    sigma_z = expectation(site_operator(spec.n_sites, "z", 0))
    # (hbar/2)^2 <sigma^2> = (hbar/2)^2, because every Pauli matrix squares to the
    # identity. So the spread is set entirely by the expectation value, and no
    # second matrix product is needed to get it.
    half = 0.5 * HBAR
    return Quantumness(
        spec=spec,
        energy=energy,
        energy_disagreement=None if reference is None else abs(energy - reference),
        terms_commutator=norm(commutator(ising, field)),
        onsite_anticommutator=norm(
            anticommutator(site_operator(spec.n_sites, "x", 0), site_operator(spec.n_sites, "z", 0))
        ),
        offsite_commutator=norm(
            commutator(site_operator(spec.n_sites, "x", 0), site_operator(spec.n_sites, "z", 1))
        ),
        parity_commutator=norm(commutator(hamiltonian, parity(spec.n_sites))),
        parity_expectation=expectation(parity(spec.n_sites)),
        sigma_x=sigma_x,
        spread_y=half * float(np.sqrt(max(0.0, 1.0 - sigma_y**2))),
        spread_z=half * float(np.sqrt(max(0.0, 1.0 - sigma_z**2))),
        bound=half * half * abs(sigma_x),
    )
