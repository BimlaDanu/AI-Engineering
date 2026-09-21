r"""Exact free-fermion (Pfeuty) solution of the 1D transverse-field Ising model.

This module is the reference the rest of the physics is checked against. Exact
diagonalisation, mean-field, and every method added later are measured against
it, so it is deliberately small, dependency-light and heavily tested.

Method. A Jordan-Wigner transformation maps the spin chain to free
fermions and a Bogoliubov rotation diagonalises the result, giving
quasiparticle energies

.. math::

    \varepsilon(k) = 2\sqrt{J^2 + h^2 - 2Jh\cos k}.

For a ferromagnetic ring the ground state lies in the even-fermion-parity
sector at every finite ``L``, and that sector carries *antiperiodic* boundary
conditions on the fermions, i.e. momenta :math:`k = (2n+1)\pi/L`. Those momenta
never land on :math:`k = 0` or :math:`k = \pi`, so the unpaired-zero-mode
bookkeeping that complicates the odd sector does not arise here at all. The
ground-state energy is then a single sum over the positive half of the
Brillouin zone,

.. math::

    E_0 = -\sum_{k>0} \varepsilon(k).

Cost. ``O(L)`` arithmetic and ``O(L)`` memory. ``L = 10**6`` is a
millisecond. That is what makes this the reference the other methods are
judged against: it is free, and it never approximates.

Applicability. Uniform ferromagnetic 1D TFIM, periodic boundary, even
``L``. Anything else must use a different method — see
:func:`unsupported_reason`, which is the hook the agent's method-selection
step reads.
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import ellipe

from src.physics.method_catalogue import free_fermion_unsupported_reason
from src.physics.model import TFIMSpec

FloatArray = NDArray[np.float64]

METHOD_NAME = "pfeuty_exact"
"""Registry key. Kept next to the implementation so the two cannot drift."""


def unsupported_reason(spec: TFIMSpec) -> str | None:
    """Explain why this method cannot handle ``spec``, or return ``None``.

    Delegates to :mod:`src.physics.method_catalogue`, which is where every
    applicability check lives. The reason for that split is the seal: *whether*
    a method applies is a statement about the problem and gives away nothing
    about the answer, so the agent is entitled to ask it -- and it cannot import
    this module to do so. Defining the predicate once and re-exporting it here
    keeps the module self-contained for the :class:`Method` protocol without
    letting the two copies drift.

    Args:
        spec: The problem to check.

    Returns:
        A human-readable reason the method is inapplicable, or ``None`` if it
        applies.
    """
    return free_fermion_unsupported_reason(spec)


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


def _require_valid_parameters(coupling: float, field: float) -> None:
    """Reject a coupling or field the model is not defined for.

    The thermodynamic-limit functions take bare numbers rather than a
    :class:`~src.physics.model.TFIMSpec` -- an infinite chain has no length --
    so they do not get that class's validation for free and check here instead.

    Args:
        coupling: The Ising coupling ``J``.
        field: The transverse field ``h``.

    Raises:
        ValueError: If ``coupling`` is not strictly positive, or ``field`` is
            negative.
    """
    if coupling <= 0.0:
        raise ValueError(f"coupling J must be strictly positive, got {coupling}")
    if field < 0.0:
        raise ValueError(f"field h must be non-negative, got {field}")


def positive_momenta(n_sites: int) -> FloatArray:
    """Return the positive antiperiodic momenta of the even-parity sector.

    These are ``k = (2n+1) * pi / L`` for ``n = 0 .. L/2 - 1``, i.e. the upper
    half of the Brillouin zone. The negative half is its mirror image and
    contributes identically, which is why the energy sum below carries no
    factor of one half.

    Args:
        n_sites: Chain length ``L``. Must be even.

    Returns:
        Array of ``L / 2`` momenta in ascending order, all in ``(0, pi)``.
    """
    index = np.arange(n_sites // 2, dtype=np.float64)
    return (2.0 * index + 1.0) * np.pi / n_sites


def dispersion(momenta: FloatArray, coupling: float, field: float) -> FloatArray:
    """Bogoliubov quasiparticle energies ``eps(k)``.

    Args:
        momenta: Momenta at which to evaluate the dispersion.
        coupling: The Ising coupling ``J``.
        field: The transverse field ``h``.

    Returns:
        ``2 * sqrt(J**2 + h**2 - 2*J*h*cos k)``, elementwise. The radicand is a
        perfect square at ``k = 0`` and never negative, so no clipping is
        needed.
    """
    radicand = coupling**2 + field**2 - 2.0 * coupling * field * np.cos(momenta)
    return 2.0 * np.sqrt(radicand)


def ground_state_energy(spec: TFIMSpec) -> float:
    """Exact ground-state energy ``E0`` of the finite chain.

    Args:
        spec: The problem. Must be a periodic ring of even length.

    Returns:
        The total ground-state energy, not the density.

    Raises:
        ValueError: If the spec is outside this method's domain of validity.

    Examples:
        At zero field every bond is satisfied, so ``E0 = -J * L``:

        >>> from src.physics.model import TFIMSpec
        >>> round(ground_state_energy(TFIMSpec(n_sites=8, coupling=1.0, field=0.0)), 12)
        -8.0
    """
    _require_supported(spec)
    energies = dispersion(positive_momenta(spec.n_sites), spec.coupling, spec.field)
    return float(-np.sum(energies))


def energy_density(spec: TFIMSpec) -> float:
    """Exact ground-state energy per site, ``E0 / L``.

    Args:
        spec: The problem.

    Returns:
        The energy density, which is the quantity that converges to a finite
        thermodynamic limit and so is what finite-size studies should compare.
    """
    return ground_state_energy(spec) / spec.n_sites


def transverse_magnetisation(spec: TFIMSpec) -> float:
    """Exact transverse magnetisation ``<sigma_x> = (1/L) sum_i <sigma_x_i>``.

    Obtained by the Hellmann-Feynman theorem as ``-(1/L) dE0/dh``, which for
    the free-fermion spectrum is a closed-form sum over the same momenta used
    by :func:`ground_state_energy`. No numerical differentiation is involved,
    so this is exact to machine precision rather than to a step size.

    Args:
        spec: The problem.

    Returns:
        A number in ``[0, 1]``: zero at ``h = 0`` (spins along ``z``) and
        approaching one as ``h`` dominates (spins polarised along ``x``).

    Raises:
        ValueError: If the spec is outside this method's domain of validity.
    """
    _require_supported(spec)
    momenta = positive_momenta(spec.n_sites)
    j, h = spec.coupling, spec.field
    numerator = 2.0 * (h - j * np.cos(momenta))
    denominator = np.sqrt(j**2 + h**2 - 2.0 * j * h * np.cos(momenta))
    return float(np.sum(numerator / denominator) / spec.n_sites)


def excitation_energies(spec: TFIMSpec) -> FloatArray:
    r"""Bogoliubov quasiparticle energies at the chain's allowed momenta.

    The free-fermion mapping turns the chain into ``L / 2`` independent
    oscillators, one per positive momentum, and :math:`\epsilon(k)` is the cost of
    exciting one of them. These are therefore the low-lying excitation energies of
    the model in the language the exact solution is written in -- and the source of
    the *even-parity* many-body spectrum, since each such state's energy is the
    ground state plus a sum of an **even** number of these. The odd-parity states
    are not reachable this way at all: they carry the periodic momenta
    :math:`k = 2n\pi/L` rather than these antiperiodic ones, which is the same
    sector bookkeeping :func:`gap` and :func:`parity_even_gap` turn on.

    Distinct from :func:`src.physics.reference.exact_diagonalisation.low_levels`,
    which returns *many-body* levels by diagonalising the whole Hamiltonian. Both
    are exact and they answer different questions: this one is ``O(L)`` and says
    which modes exist, the other is ``O(2**L)`` and says which states do.

    Args:
        spec: The problem. Must be a periodic ring of even length.

    Returns:
        The ``L / 2`` energies, ascending in ``k``, all strictly positive for a
        finite chain: the smallest allowed momentum is :math:`\pi / L` rather than
        zero, which is exactly why the gap of a finite chain never closes.

    Raises:
        ValueError: If the spec is outside this method's domain of validity.

    Examples:
        >>> from src.physics.model import TFIMSpec
        >>> energies = excitation_energies(TFIMSpec(n_sites=6, coupling=1.0, field=1.0))
        >>> len(energies)
        3
        >>> bool((energies > 0.0).all())
        True
    """
    _require_supported(spec)
    return dispersion(positive_momenta(spec.n_sites), spec.coupling, spec.field)


def gap(spec: TFIMSpec) -> float:
    r"""Energy of the cheapest **single quasiparticle**, :math:`\min_k \epsilon(k)`.

    Read the name carefully: this is one quasiparticle's energy, not the spacing
    between the chain's two lowest states. The two are different numbers and the
    difference is not small.

    Quasiparticles on a ring are created in pairs -- the Hamiltonian commutes with
    the parity :math:`\hat P = \prod_i \hat\sigma^x_i`, so a state with an odd
    number of them lives in the other parity sector and carries a different set of
    allowed momenta entirely. So the cheapest excitation the ground state can
    actually reach without changing parity costs *two* of these, which is what
    :func:`parity_even_gap` returns and what a test holds against a parity-resolved
    diagonalisation. The true first excited state of the ring is cheaper still and
    is in the odd sector: below the critical field it is the other half of the
    symmetry-broken doublet, and its splitting from the ground state falls
    exponentially with ``L`` -- at :math:`L = 10`, :math:`h = 0.2J` it is under
    :math:`10^{-5}` while this function returns :math:`1.6`.

    This is nonetheless the right quantity to plot against
    :func:`gap_thermodynamic`, which is the same single-quasiparticle energy in the
    infinite chain, and it is the quantity the phase diagram is drawn from.

    Args:
        spec: The problem. Must be a periodic ring of even length.

    Returns:
        :math:`\epsilon(\pi/L)`, **strictly positive at every field**. A finite
        chain has no gapless point: the momentum that would send this to zero,
        :math:`k = 0`, is not one this ring allows.

    Raises:
        ValueError: If the spec is outside this method's domain of validity.

    Examples:
        >>> from src.physics.model import TFIMSpec
        >>> gap(TFIMSpec(n_sites=8, coupling=1.0, field=1.0)) > 0.0
        True
    """
    return float(np.min(excitation_energies(spec)))


def parity_even_gap(spec: TFIMSpec) -> float:
    r"""The cheapest excitation the ground state can actually reach.

    Everything in this project that evolves the chain -- imaginary time from
    :math:`\lvert + \rangle^{\otimes L}`, the hardware-efficient ansatz, the
    variational imaginary-time baseline -- conserves the parity
    :math:`\hat P = \prod_i \hat\sigma^x_i`. So the states such a method can mix
    into the ground state are the *even* ones, quasiparticles come in pairs, and the
    energy that sets how fast an evolution converges is

    .. math::

        \Delta_{\text{even}} = 2 \min_k \epsilon(k) = 2\,\epsilon(\pi/L),

    twice what :func:`gap` returns. Getting this wrong is a factor of two in every
    convergence-time estimate in the same direction -- always optimistic -- which is
    why it has a function of its own and a cross-check rather than a comment.

    Held in the tests against a parity-resolved diagonalisation of the sparse
    matrix, which shares no algebra with any of this -- no momenta, no sectors, no
    Bogoliubov rotation. Not against a level *index*: below the critical field the
    ring's first excited state is the odd-parity partner of its ground state and the
    second is this pair, while above it the odd sector fills in with single
    quasiparticles and the pair is pushed several levels up. Sorting the spectrum by
    parity is what makes one statement cover both phases.

    Args:
        spec: The problem. Must be a periodic ring of even length.

    Returns:
        The lowest even-parity excitation energy.

    Raises:
        ValueError: If the spec is outside this method's domain of validity.

    Examples:
        >>> from src.physics.model import TFIMSpec
        >>> ring = TFIMSpec(n_sites=8, coupling=1.0, field=1.0)
        >>> round(parity_even_gap(ring) / gap(ring), 12)
        2.0
    """
    return 2.0 * gap(spec)


DEFAULT_LEVELS = 6
"""How many many-body levels :func:`low_lying_levels` returns unless asked otherwise.

Six is what a spectrum figure can label without the lines merging, and it is enough
to show the two features that matter as the field is swept: the near-degenerate
partner of the ground state below :math:`h = J`, and the pair excitation that
becomes the cheapest one above it.
"""

MAX_LEVELS = 24
"""Ceiling on one spectrum request.

The search below is cheap per level, but a plot of two dozen lines communicates
nothing that six do not, and a prompt carrying them is a prompt of digits.
"""


def _sector_mode_energies(spec: TFIMSpec, parity: int) -> FloatArray:
    r"""Signed single-mode energies of one fermion-parity sector.

    The two sectors of the ring differ in one thing only -- which momenta the
    fermions are allowed -- and everything else about them follows from it. Parity
    :math:`\hat P = \prod_i \hat\sigma^x_i` commutes with the Hamiltonian, and the
    Jordan-Wigner string closing the ring contributes :math:`\hat P`, so the even
    sector quantises momenta *antiperiodically* and the odd sector periodically:

    .. math::

        k \in \Big\{\tfrac{(2n+1)\pi}{L}\Big\}_{n=0}^{L-1} \ (\text{even}),
        \qquad
        k \in \Big\{\tfrac{2n\pi}{L}\Big\}_{n=0}^{L-1} \ (\text{odd}).

    The full Brillouin zone rather than its positive half, because each momentum
    carries its own mode: occupying :math:`+k` and :math:`-k` costs
    :math:`2\epsilon(k)`, and a construction that kept one mode per :math:`|k|`
    would miss exactly those states. (It does, and it looks right until it is held
    against a diagonalisation: the first even excitation of the ring is
    :math:`2\epsilon(\pi/L)`, and that state has no counterpart in the positive
    half alone.)

    The **signed** return value is the whole subtlety of the odd sector. For a mode
    with pairing amplitude :math:`\Delta_k = 2J\sin k \neq 0` the Bogoliubov
    rotation gives the usual positive :math:`\epsilon(k)`. At :math:`k = 0` and
    :math:`k = \pi` -- which only the periodic set contains -- the pairing vanishes,
    no rotation happens, and the mode keeps its bare energy
    :math:`\xi_k = 2(h - J\cos k)`. That is **negative below the critical field**
    at :math:`k = 0`, which means the cheapest configuration has the mode *filled*,
    and filling it flips the fermion parity at essentially no cost. That single
    sign is what makes the ground state of the ferromagnetic ring
    near-degenerate below :math:`h = J` and non-degenerate above it -- the
    symmetry breaking, arrived at by bookkeeping rather than by assumption.

    Args:
        spec: The problem. Must be a periodic ring of even length.
        parity: ``+1`` for the even sector, ``-1`` for the odd one.

    Returns:
        ``L`` signed energies, one per allowed momentum, in ascending momentum.
        Positive throughout the even sector; possibly negative at ``k = 0`` in the
        odd one.
    """
    index = np.arange(spec.n_sites, dtype=np.float64)
    if parity > 0:
        momenta = (2.0 * index + 1.0) * np.pi / spec.n_sites
    else:
        momenta = 2.0 * index * np.pi / spec.n_sites
    j, h = spec.coupling, spec.field
    bare = 2.0 * (h - j * np.cos(momenta))
    pairing = 2.0 * j * np.sin(momenta)
    rotated = np.sqrt(bare**2 + pairing**2)
    # Where the pairing vanishes there is nothing to rotate, and the bare energy --
    # sign and all -- is the mode's own. `np.isclose` rather than an exact zero
    # test: sin(pi) is 1.2e-16 rather than 0.0 in floating point.
    return np.where(np.isclose(pairing, 0.0, atol=1e-12), bare, rotated)


def _cheapest_sums(costs: FloatArray, want_odd_count: bool, count: int) -> list[float]:
    """Smallest sums over subsets of ``costs`` with a fixed parity of subset size.

    A best-first search rather than an enumeration, and that is the difference
    between a function that works at ``L = 20`` and one that allocates a million
    floats to return six numbers. The classic trick for subset sums in ascending
    order, carrying the subset *size* in the state so the parity constraint can be
    applied without ever materialising a subset: from a state ending at index
    ``i``, either swap that last element for the next one along -- same size -- or
    extend by it, one larger. Every subset is reached exactly once, and the heap
    hands them out cheapest first.

    Args:
        costs: Non-negative excitation costs, **sorted ascending**.
        want_odd_count: Whether the wanted subsets have an odd number of members.
        count: How many sums to return.

    Returns:
        The ``count`` smallest qualifying sums, ascending. Shorter than ``count``
        only when the set has fewer qualifying subsets than that, which happens
        only for a chain of two or three modes.
    """
    found: list[float] = []
    if not want_odd_count:
        found.append(0.0)  # the empty subset: the sector's own reference state
    if costs.size == 0:
        return found[:count]
    # (running sum, index of the last member, number of members)
    frontier: list[tuple[float, int, int]] = [(float(costs[0]), 0, 1)]
    heapq.heapify(frontier)
    while frontier and len(found) < count:
        total, last, size = heapq.heappop(frontier)
        if (size % 2 == 1) == want_odd_count:
            found.append(total)
        if last + 1 < costs.size:
            step = float(costs[last + 1])
            heapq.heappush(frontier, (total - float(costs[last]) + step, last + 1, size))
            heapq.heappush(frontier, (total + step, last + 1, size + 1))
    return found[:count]


def low_lying_levels(spec: TFIMSpec, count: int = DEFAULT_LEVELS) -> FloatArray:
    r"""The lowest many-body energies of the ring, from the free-fermion solution.

    The answer to *plot the low-lying spectrum against the field*, computed in
    :math:`O(L \log L)` per level rather than by diagonalising a :math:`2^L`
    matrix. Both fermion-parity sectors are built -- see
    :func:`_sector_mode_energies` for why they differ and why the difference is the
    physics rather than bookkeeping -- and the two lists are merged:

    .. math::

        E = -\tfrac{1}{2}\sum_m \lvert \epsilon_m \rvert
            + \sum_{m \in \text{flipped}} \lvert \epsilon_m \rvert ,

    with the sector's allowed momenta setting :math:`\{\epsilon_m\}` and the number
    of flips constrained to keep the sector's parity. The reference energy is the
    same expression at zero flips, which is why the ground state needs no special
    case: it is the empty flip set of the even sector.

    This is checked against something that shares no algebra with it. A test holds
    the whole construction against ``numpy.linalg.eigvalsh`` of the dense
    :math:`2^L` Hamiltonian -- no momenta, no sectors, no Bogoliubov rotation -- for
    :math:`L = 4, 6, 8` at nine fields spanning both phases, and the two agree on the
    entire spectrum to :math:`10^{-13}`. Agreement on every level rather than on the
    lowest few is what rules out the failure this function is exposed to: a sector
    construction can be wrong by one state and still get the ground state right in
    one phase.

    That check also found something worth knowing about the *other* reference
    solver. :func:`src.physics.reference.exact_diagonalisation.low_levels` uses a
    sparse iterative eigensolver, and below the critical field it silently returns
    fewer copies of a degenerate level than the degeneracy holds -- so a level-by-
    level comparison against it fails while both solvers are right. Compare against
    the dense spectrum, or compare sets.

    Args:
        spec: The problem. Must be a periodic ring of even length.
        count: How many levels to return, at most :data:`MAX_LEVELS`.

    Returns:
        The ``count`` lowest energies, ascending, with degeneracies repeated -- two
        equal entries mean two states, which is exactly what a spectrum plot below
        :math:`h = J` has to show.

    Raises:
        ValueError: If the spec is outside this method's domain of validity, or
            ``count`` is not between one and :data:`MAX_LEVELS`.

    Examples:
        Below the critical field the two lowest levels are all but degenerate --
        the symmetry-broken pair -- and above it they are cleanly split:

        >>> from src.physics.model import TFIMSpec
        >>> ordered = low_lying_levels(TFIMSpec(n_sites=10, coupling=1.0, field=0.2))
        >>> bool(ordered[1] - ordered[0] < 1e-6)
        True
        >>> polarised = low_lying_levels(TFIMSpec(n_sites=10, coupling=1.0, field=2.0))
        >>> bool(polarised[1] - polarised[0] > 1.0)
        True

        And the lowest level is the ground-state energy the rest of this module
        computes by a different route:

        >>> ring = TFIMSpec(n_sites=8, coupling=1.0, field=0.7)
        >>> round(float(low_lying_levels(ring)[0]) - ground_state_energy(ring), 12)
        0.0
    """
    _require_supported(spec)
    if not 1 <= count <= MAX_LEVELS:
        raise ValueError(f"count must be between 1 and {MAX_LEVELS}, got {count}")
    levels: list[float] = []
    for parity in (+1, -1):
        signed = _sector_mode_energies(spec, parity)
        costs = np.sort(np.abs(signed))
        reference = -0.5 * float(np.sum(np.abs(signed)))
        # A mode whose signed energy is negative is *filled* in the sector's
        # reference state, so the reference already carries an occupancy of that
        # parity and the flips are counted on top of it.
        filled = int(np.sum(signed < 0.0)) % 2
        want_odd = (filled % 2 == 0) if parity < 0 else (filled % 2 == 1)
        levels.extend(reference + total for total in _cheapest_sums(costs, want_odd, count))
    return np.sort(np.array(levels, dtype=np.float64))[:count]


def gap_thermodynamic(coupling: float, field: float) -> float:
    r"""The infinite chain's gap, :math:`2|J - h|`.

    The whole phase diagram in one expression. Momentum is continuous in an
    infinite chain, so :math:`k = 0` is available and
    :math:`\epsilon(0) = 2|J - h|` is the cheapest excitation there is -- and it
    costs nothing on the line :math:`h = J`. That vanishing is the phase
    transition, and it is the reason the critical line can be *drawn* rather than
    marked by hand: it is where this function is zero.

    Args:
        coupling: The Ising coupling ``J``.
        field: The transverse field ``h``.

    Returns:
        The gap. Unlike :func:`gap` this takes no spec, because the thermodynamic
        limit has no length -- and takes no boundary condition, because in that
        limit two ends make no difference.

    Examples:
        >>> gap_thermodynamic(1.0, 1.0)
        0.0
        >>> gap_thermodynamic(1.0, 0.25)
        1.5
    """
    return 2.0 * abs(coupling - field)


def energy_density_curvature(spec: TFIMSpec) -> float:
    r"""Second derivative of the energy density with respect to the field.

    In closed form, differentiating :math:`\epsilon(k)` twice:

    .. math::

        \frac{\partial^2 \epsilon}{\partial h^2}
            = \frac{2 J^2 \sin^2 k}{(J^2 + h^2 - 2 J h \cos k)^{3/2}}

    so :math:`\partial^2 (E_0/L) / \partial h^2` is minus the sum of that over the
    allowed momenta, divided by ``L``. Analytic rather than a finite difference, so
    it is exact to machine precision rather than to a step size -- which matters
    here more than usual: a second derivative computed numerically is the classic
    place where a plot acquires structure that is not in the physics.

    The first derivative needs no function of its own. By Hellmann-Feynman it *is*
    the transverse magnetisation up to sign, so :func:`transverse_magnetisation`
    already returns it, and having one quantity behind both curves is what lets the
    derivative plot be checked rather than believed.

    Args:
        spec: The problem. Must be a periodic ring of even length.

    Returns:
        The curvature, always negative or zero: the ground-state energy is concave
        in the field. It dips sharpest near the critical field, and that dip
        sharpening as ``L`` grows is the finite chain's signature of the transition
        -- in the infinite chain the corresponding derivative diverges.

    Raises:
        ValueError: If the spec is outside this method's domain of validity.

    Examples:
        The curvature is a response function, so it never has the wrong sign:

        >>> from src.physics.model import TFIMSpec
        >>> energy_density_curvature(TFIMSpec(n_sites=8, coupling=1.0, field=1.0)) < 0.0
        True
    """
    _require_supported(spec)
    momenta = positive_momenta(spec.n_sites)
    j, h = spec.coupling, spec.field
    radicand = j**2 + h**2 - 2.0 * j * h * np.cos(momenta)
    terms = 2.0 * j**2 * np.sin(momenta) ** 2 / radicand**1.5
    return float(-np.sum(terms) / spec.n_sites)


def magnetisation_slope(spec: TFIMSpec) -> float:
    r"""First derivative of the transverse magnetisation with respect to the field.

    One line of algebra rather than a new sum. The magnetisation is minus the first
    derivative of the energy density (Hellmann-Feynman), so its own first derivative
    is minus the energy density's second:

    .. math::

        \frac{\partial \langle \sigma^x \rangle}{\partial h}
            = - \frac{\partial^2 (E_0/L)}{\partial h^2}

    which :func:`energy_density_curvature` already has in closed form. Deriving it
    that way rather than differentiating the magnetisation sum again is the point:
    the two curves are one quantity, and a bug in either would show up as the pair
    disagreeing.

    Args:
        spec: The problem. Must be a periodic ring of even length.

    Returns:
        The susceptibility of the chain to the transverse field, never negative:
        turning the field up cannot turn the alignment with it down. It peaks near
        the critical field, and that peak sharpening with ``L`` is the finite
        chain's signature of the transition.

    Raises:
        ValueError: If the spec is outside this method's domain of validity.

    Examples:
        >>> from src.physics.model import TFIMSpec
        >>> magnetisation_slope(TFIMSpec(n_sites=8, coupling=1.0, field=1.0)) > 0.0
        True
    """
    return -energy_density_curvature(spec)


def magnetisation_curvature(spec: TFIMSpec) -> float:
    r"""Second derivative of the transverse magnetisation with respect to the field.

    The third derivative of the energy density, with the sign flipped. Differentiating
    :math:`\partial^2 \epsilon / \partial h^2` once more:

    .. math::

        \frac{\partial^3 \epsilon}{\partial h^3}
            = - \frac{6 J^2 \sin^2 k \, (h - J \cos k)}{(J^2 + h^2 - 2 J h \cos k)^{5/2}}

    Analytic for the same reason the curvature below it is: this is the third
    derivative of a computed quantity, and finite-differencing one is where a plot
    reliably acquires structure that is not in the physics. Held against a central
    difference of :func:`transverse_magnetisation` by a test, which is the check
    that matters -- the algebra here is easy to get subtly wrong and impossible to
    eyeball.

    Args:
        spec: The problem. Must be a periodic ring of even length.

    Returns:
        The rate at which the susceptibility itself is changing. Positive below the
        critical field, where the magnetisation curve is still steepening, and
        negative above it, where it is flattening towards saturation; the crossing
        is this chain's own estimate of where the rise is fastest.

    Raises:
        ValueError: If the spec is outside this method's domain of validity.

    Examples:
        It changes sign across the transition, which is what makes the steepest
        rise a zero to look for rather than a maximum to scan for:

        >>> from src.physics.model import TFIMSpec
        >>> magnetisation_curvature(TFIMSpec(n_sites=8, coupling=1.0, field=0.4)) > 0.0
        True
        >>> magnetisation_curvature(TFIMSpec(n_sites=8, coupling=1.0, field=1.7)) < 0.0
        True
    """
    _require_supported(spec)
    momenta = positive_momenta(spec.n_sites)
    j, h = spec.coupling, spec.field
    radicand = j**2 + h**2 - 2.0 * j * h * np.cos(momenta)
    terms = 6.0 * j**2 * np.sin(momenta) ** 2 * (h - j * np.cos(momenta)) / radicand**2.5
    return float(-np.sum(terms) / spec.n_sites)


def energy_density_thermodynamic(coupling: float, field: float) -> float:
    r"""Ground-state energy per site of the *infinite* chain, in closed form.

    The momentum sum becomes an integral that evaluates to a complete elliptic
    integral of the second kind:

    .. math::

        e_0 = -\frac{2J}{\pi}(1 + r)\,E\!\left(\frac{4r}{(1+r)^2}\right),
        \qquad r = h/J .

    Args:
        coupling: The Ising coupling ``J``, strictly positive.
        field: The transverse field ``h``, non-negative.

    Returns:
        The infinite-chain energy density. At the critical point ``h = J`` this
        is exactly ``-4J/pi``. The ratio is written ``r`` rather than the ``g``
        much of the literature uses, because ``g`` is this project's
        longitudinal field -- see :data:`src.physics.model.HAMILTONIAN_LATEX`.

    Raises:
        ValueError: If ``coupling`` is not positive or ``field`` is negative.
    """
    _require_valid_parameters(coupling, field)
    g = field / coupling
    parameter = 4.0 * g / (1.0 + g) ** 2
    return float(-2.0 * coupling / np.pi * (1.0 + g) * ellipe(parameter))


def energy_density_thermodynamic_quadrature(coupling: float, field: float) -> float:
    """The same infinite-chain energy density, by numerical quadrature.

    This duplicates :func:`energy_density_thermodynamic` on purpose. The two
    routes share no algebra — one is a closed form in elliptic integrals, the
    other integrates the dispersion directly — so agreement between them is
    real evidence that neither was transcribed wrongly. That is the project's
    whole thesis applied to its own foundation, and it costs ten lines.

    Args:
        coupling: The Ising coupling ``J``, strictly positive.
        field: The transverse field ``h``, non-negative.

    Returns:
        The infinite-chain energy density, accurate to quadrature tolerance.

    Raises:
        ValueError: If ``coupling`` is not positive or ``field`` is negative.
            Checked here as well as in the closed form, and that is the point:
            these two exist to be compared, so a parameter one of them refuses
            and the other quietly accepts would turn a cross-check into a
            comparison of two different problems.
    """
    _require_valid_parameters(coupling, field)

    def integrand(k: float) -> float:
        return float(np.sqrt(coupling**2 + field**2 - 2.0 * coupling * field * np.cos(k)))

    value, _abserr = quad(integrand, 0.0, np.pi)
    return float(-value / np.pi)
