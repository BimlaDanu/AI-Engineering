r"""Exact free-fermion (Pfeuty) solution of the 1D transverse-field Ising model.

This module is the reference the rest of the physics is checked against. Exact
diagonalisation, mean-field, and every method added later are measured against
it, so it is deliberately small, dependency-light and heavily tested.

**Method.** A Jordan-Wigner transformation maps the spin chain to free
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

**Cost.** ``O(L)`` arithmetic and ``O(L)`` memory. ``L = 10**6`` is a
millisecond. That is what makes this the reference the other methods are
judged against: it is free, and it never approximates.

**Applicability.** Uniform ferromagnetic 1D TFIM, periodic boundary, even
``L``. Anything else must use a different method — see
:func:`unsupported_reason`, which is the hook the agent's method-selection
step reads.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import ellipe

from src.physics.model import TFIMSpec

FloatArray = NDArray[np.float64]

METHOD_NAME = "pfeuty_exact"
"""Registry key. Kept next to the implementation so the two cannot drift."""


def unsupported_reason(spec: TFIMSpec) -> str | None:
    """Explain why this method cannot handle ``spec``, or return ``None``.

    The agent's method-selection step calls this on every registered method and
    is only allowed to choose from those that return ``None``. Returning a
    sentence rather than a boolean means a rejection can be shown to the user
    and cited in the run's justification.

    Args:
        spec: The problem to check.

    Returns:
        A human-readable reason the method is inapplicable, or ``None`` if it
        applies.
    """
    if spec.boundary != "periodic":
        return (
            "The free-fermion solution implemented here assumes a periodic ring; "
            f"this spec uses {spec.boundary} boundaries."
        )
    if spec.n_sites % 2 != 0:
        return (
            "The even-parity momentum set k = (2n+1)pi/L is only a complete, "
            f"symmetric half-Brillouin-zone for even L; this spec has L={spec.n_sites}."
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
    the many-body spectrum, since every state's energy is the ground state plus a
    sum of some of these.

    Distinct from :func:`src.physics.ed.low_levels`, which returns *many-body*
    levels by diagonalising the whole Hamiltonian. Both are exact and they answer
    different questions: this one is ``O(L)`` and says which modes exist, the other
    is ``O(2**L)`` and says which states do.

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
    r"""Energy of the cheapest quasiparticle, :math:`\min_k \epsilon(k)`.

    Args:
        spec: The problem. Must be a periodic ring of even length.

    Returns:
        The smallest allowed excitation energy, which for a finite ring is
        :math:`\epsilon(\pi/L)` and is **strictly positive at every field**. A
        finite chain has no gapless point: the momentum that would close the gap,
        :math:`k = 0`, is not one this ring allows.

    Raises:
        ValueError: If the spec is outside this method's domain of validity.

    Examples:
        >>> from src.physics.model import TFIMSpec
        >>> gap(TFIMSpec(n_sites=8, coupling=1.0, field=1.0)) > 0.0
        True
    """
    return float(np.min(excitation_energies(spec)))


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

        e_0 = -\frac{2J}{\pi}(1 + g)\,E\!\left(\frac{4g}{(1+g)^2}\right),
        \qquad g = h/J .

    Args:
        coupling: The Ising coupling ``J``, strictly positive.
        field: The transverse field ``h``, non-negative.

    Returns:
        The infinite-chain energy density. At the critical point ``g = 1`` this
        is exactly ``-4J/pi``.

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
