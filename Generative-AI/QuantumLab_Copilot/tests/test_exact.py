"""Tests for the free-fermion reference solution.

Chain lengths are capped at ``L = 8`` throughout, matching the project rule
that no test may run a system a laptop would notice. Nothing here needs a long
chain: every claim is either an exactly solvable limit, an algebraic identity,
or a comparison between two independent code paths at the same size.
"""

import math
from itertools import pairwise

import numpy as np
import pytest

from src.physics.exact import (
    dispersion,
    energy_density,
    energy_density_curvature,
    energy_density_thermodynamic,
    energy_density_thermodynamic_quadrature,
    excitation_energies,
    gap,
    gap_thermodynamic,
    ground_state_energy,
    magnetisation_curvature,
    magnetisation_slope,
    positive_momenta,
    transverse_magnetisation,
    unsupported_reason,
)
from src.physics.model import TFIMSpec

SIZES = [2, 4, 6, 8]
"""Every chain length used in this suite. See the module docstring."""

# --------------------------------------------------------------------------
# Applicability — the hook the agent's method-selection step reads.
# --------------------------------------------------------------------------


def test_applies_to_an_even_ring() -> None:
    assert unsupported_reason(TFIMSpec(n_sites=8)) is None


def test_rejects_open_boundaries_with_a_reason() -> None:
    reason = unsupported_reason(TFIMSpec(n_sites=8, boundary="open"))
    assert reason is not None
    assert "periodic" in reason


def test_rejects_odd_chains_with_a_reason() -> None:
    reason = unsupported_reason(TFIMSpec(n_sites=7))
    assert reason is not None
    assert "even L" in reason


@pytest.mark.parametrize(
    "spec",
    [TFIMSpec(n_sites=7), TFIMSpec(n_sites=8, boundary="open")],
)
def test_solving_an_unsupported_spec_raises_rather_than_guessing(spec: TFIMSpec) -> None:
    with pytest.raises(ValueError, match="pfeuty_exact cannot solve"):
        ground_state_energy(spec)
    with pytest.raises(ValueError, match="pfeuty_exact cannot solve"):
        transverse_magnetisation(spec)


# --------------------------------------------------------------------------
# Momenta and dispersion.
# --------------------------------------------------------------------------


def test_momenta_are_the_positive_antiperiodic_half_zone() -> None:
    k = positive_momenta(8)
    assert k.shape == (4,)
    np.testing.assert_allclose(k, np.array([1, 3, 5, 7]) * np.pi / 8)
    # Never on a zero mode: that is what keeps the even sector free of
    # unpaired-fermion bookkeeping.
    assert np.all(k > 0.0)
    assert np.all(k < np.pi)


def test_dispersion_is_symmetric_under_exchanging_j_and_h() -> None:
    k = positive_momenta(8)
    np.testing.assert_allclose(dispersion(k, 1.0, 0.4), dispersion(k, 0.4, 1.0))


def test_dispersion_gap_closes_only_at_criticality() -> None:
    # Evaluated at a single small momentum rather than on a long chain: the
    # gap is a property of eps(k), not of any particular lattice.
    near_zero = np.array([1e-6])
    assert dispersion(near_zero, 1.0, 1.0).item() == pytest.approx(0.0, abs=1e-5)
    # Away from criticality the gap is 2|J - h| and stays open.
    assert dispersion(near_zero, 1.0, 0.5).item() == pytest.approx(1.0, abs=1e-9)
    assert dispersion(near_zero, 1.0, 2.0).item() == pytest.approx(2.0, abs=1e-9)


# --------------------------------------------------------------------------
# Ground-state energy — exactly solvable limits.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
def test_zero_field_saturates_every_bond(n_sites: int) -> None:
    # All spins aligned along z: E0 = -J * L, one satisfied bond per site.
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.0)
    assert ground_state_energy(spec) == pytest.approx(-1.0 * n_sites, abs=1e-12)


@pytest.mark.parametrize("n_sites", SIZES)
def test_dominant_field_polarises_every_spin(n_sites: int) -> None:
    # J -> 0: every spin sits in the +x eigenstate, E0 = -h * L.
    spec = TFIMSpec(n_sites=n_sites, coupling=1e-9, field=1.0)
    assert ground_state_energy(spec) == pytest.approx(-1.0 * n_sites, rel=1e-8)


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("g", [0.25, 0.5, 0.9, 1.0, 1.5, 3.0])
def test_kramers_wannier_self_duality(n_sites: int, g: float) -> None:
    # The spectrum depends on (J, h) only through J^2 + h^2 - 2Jh cos k, which
    # is symmetric under J <-> h. Duality therefore pins the critical point at
    # g = 1 exactly, and this identity holds at every finite L.
    ordered = ground_state_energy(TFIMSpec(n_sites=n_sites, coupling=1.0, field=g))
    disordered = ground_state_energy(TFIMSpec(n_sites=n_sites, coupling=g, field=1.0))
    assert ordered == pytest.approx(disordered, rel=1e-14)


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("g", [0.5, 1.0, 2.0])
def test_energy_matches_an_independently_written_scalar_sum(n_sites: int, g: float) -> None:
    # The production path is a vectorised NumPy reduction. This one is a plain
    # Python loop with math.fsum and no array machinery at all, so a mistake in
    # the momentum set, the prefactor or the summation range shows up as a
    # disagreement rather than as two matching wrong answers.
    reference = -math.fsum(
        2.0 * math.sqrt(1.0 + g**2 - 2.0 * g * math.cos((2 * n + 1) * math.pi / n_sites))
        for n in range(n_sites // 2)
    )
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=g)
    assert ground_state_energy(spec) == pytest.approx(reference, rel=1e-14)


def test_energy_is_monotonically_lowered_by_the_field() -> None:
    energies = [
        energy_density(TFIMSpec(n_sites=8, coupling=1.0, field=float(h)))
        for h in np.linspace(0.0, 3.0, 25)
    ]
    assert all(later < earlier for earlier, later in pairwise(energies))


def test_finite_size_error_shrinks_with_length() -> None:
    # Leading finite-size correction at criticality falls off with L, so the
    # trend is visible well within the size cap.
    infinite = energy_density_thermodynamic(1.0, 1.0)
    errors = [
        abs(energy_density(TFIMSpec(n_sites=n, coupling=1.0, field=1.0)) - infinite) for n in SIZES
    ]
    assert all(later < earlier for earlier, later in pairwise(errors))


def test_short_chain_is_already_close_to_the_thermodynamic_limit() -> None:
    # L = 8 sits within a couple of percent of the infinite chain even at the
    # critical point, which is why small systems are informative here at all.
    infinite = energy_density_thermodynamic(1.0, 1.0)
    assert energy_density(TFIMSpec(n_sites=8, coupling=1.0, field=1.0)) == pytest.approx(
        infinite, rel=2e-2
    )


# --------------------------------------------------------------------------
# Thermodynamic limit — two independent routes to the same number.
# --------------------------------------------------------------------------


def test_critical_energy_density_is_minus_four_over_pi() -> None:
    assert energy_density_thermodynamic(1.0, 1.0) == pytest.approx(-4.0 / math.pi, rel=1e-12)


@pytest.mark.parametrize("g", [0.0, 0.1, 0.5, 0.99, 1.0, 1.01, 2.0, 10.0])
def test_closed_form_and_quadrature_agree(g: float) -> None:
    # Elliptic-integral closed form vs direct numerical integration of the
    # dispersion. The two share no algebra, so agreement is real evidence.
    assert energy_density_thermodynamic(1.0, g) == pytest.approx(
        energy_density_thermodynamic_quadrature(1.0, g), rel=1e-10
    )


def test_thermodynamic_helpers_reject_unphysical_arguments() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        energy_density_thermodynamic(0.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        energy_density_thermodynamic(1.0, -1.0)


# --------------------------------------------------------------------------
# Transverse magnetisation.
# --------------------------------------------------------------------------


def test_magnetisation_vanishes_without_a_field() -> None:
    spec = TFIMSpec(n_sites=8, coupling=1.0, field=0.0)
    assert transverse_magnetisation(spec) == pytest.approx(0.0, abs=1e-14)


def test_magnetisation_saturates_in_a_strong_field() -> None:
    spec = TFIMSpec(n_sites=8, coupling=1.0, field=1e6)
    assert transverse_magnetisation(spec) == pytest.approx(1.0, rel=1e-10)


def test_magnetisation_increases_with_the_field() -> None:
    values = [
        transverse_magnetisation(TFIMSpec(n_sites=8, coupling=1.0, field=float(h)))
        for h in np.linspace(0.0, 3.0, 25)
    ]
    assert all(later > earlier for earlier, later in pairwise(values))


@pytest.mark.parametrize("n_sites", SIZES)
@pytest.mark.parametrize("g", [0.3, 0.8, 1.0, 1.4, 2.5])
def test_magnetisation_matches_the_numerical_derivative_of_the_energy(
    n_sites: int, g: float
) -> None:
    # Hellmann-Feynman: <sigma_x> = -(1/L) dE0/dh. The closed form and a central
    # difference of the energy are independent code paths.
    step = 1e-5
    plus = ground_state_energy(TFIMSpec(n_sites=n_sites, coupling=1.0, field=g + step))
    minus = ground_state_energy(TFIMSpec(n_sites=n_sites, coupling=1.0, field=g - step))
    numerical = -(plus - minus) / (2.0 * step) / n_sites
    analytic = transverse_magnetisation(TFIMSpec(n_sites=n_sites, coupling=1.0, field=g))
    assert analytic == pytest.approx(numerical, rel=1e-7)


# --------------------------------------------------------------------------
# The excitations, and the derivatives that show the transition
# --------------------------------------------------------------------------


def test_there_is_one_mode_per_positive_momentum() -> None:
    for n_sites in (2, 4, 6, 8):
        spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.8)
        assert len(excitation_energies(spec)) == n_sites // 2


def test_every_excitation_costs_something_on_a_finite_chain() -> None:
    # The reason a finite chain has no closing gap: the smallest allowed momentum is
    # pi/L, not zero, so no mode is ever free -- not even at the critical field.
    for field in (0.0, 0.5, 1.0, 1.5):
        energies = excitation_energies(TFIMSpec(n_sites=8, coupling=1.0, field=field))
        assert bool((energies > 0.0).all()), field


def test_the_modes_are_degenerate_in_zero_field() -> None:
    # At h = 0 the dispersion loses its k dependence entirely: every excitation
    # costs 2J, which is one broken bond.
    energies = excitation_energies(TFIMSpec(n_sites=8, coupling=1.5, field=0.0))
    assert energies == pytest.approx(np.full(4, 2 * 1.5))


def test_the_excitations_sum_to_the_ground_state_energy() -> None:
    # The identity that makes the spectrum plot trustworthy rather than decorative:
    # the same numbers, summed, are the ground-state energy the page cross-checks
    # against exact diagonalisation.
    spec = TFIMSpec(n_sites=8, coupling=1.0, field=0.7)
    assert -float(np.sum(excitation_energies(spec))) == pytest.approx(
        ground_state_energy(spec), abs=1e-12
    )


def test_the_lowest_branch_bottoms_out_at_twice_the_sine() -> None:
    # min over h of eps(k) sits at h = J cos k, where eps = 2J |sin k|. Checked
    # because it is what the caption on the spectrum panel claims.
    n_sites, coupling = 6, 1.0
    momentum = math.pi / n_sites
    at_minimum = TFIMSpec(n_sites=n_sites, coupling=coupling, field=coupling * math.cos(momentum))
    assert float(excitation_energies(at_minimum)[0]) == pytest.approx(
        2.0 * coupling * math.sin(momentum)
    )


def test_the_energy_is_concave_in_the_field_everywhere() -> None:
    # A response function with the wrong sign is a bug, not a discovery.
    for field in (0.0, 0.3, 1.0, 1.9):
        spec = TFIMSpec(n_sites=8, coupling=1.0, field=field)
        assert energy_density_curvature(spec) < 0.0, field


def test_the_curvature_matches_a_finite_difference() -> None:
    # The closed form is analytic; this is the independent check that the algebra
    # behind it is right, done the slow way at one point.
    n_sites, field, step = 8, 0.9, 1e-5

    def density(value: float) -> float:
        return energy_density(TFIMSpec(n_sites=n_sites, coupling=1.0, field=value))

    numerical = (density(field + step) - 2.0 * density(field) + density(field - step)) / step**2
    analytic = energy_density_curvature(TFIMSpec(n_sites=n_sites, coupling=1.0, field=field))
    assert analytic == pytest.approx(numerical, rel=1e-4)


def test_the_curvature_dip_moves_towards_the_critical_point_as_the_chain_grows() -> None:
    """The finite-size claim the derivative panel is there to make.

    In the infinite chain the dip is a divergence at exactly ``g = 1``. In a finite
    one it is a smooth minimum somewhere below, and it deepens and moves right with
    ``L`` -- which is what "seeing a transition in a system too small to have one"
    actually means.
    """

    def sharpest(n_sites: int) -> tuple[float, float]:
        fields = [index / 100.0 for index in range(1, 201)]
        values = [
            energy_density_curvature(TFIMSpec(n_sites=n_sites, coupling=1.0, field=field))
            for field in fields
        ]
        deepest = min(values)
        return fields[values.index(deepest)], deepest

    short_where, short_depth = sharpest(4)
    long_where, long_depth = sharpest(8)
    assert short_where < long_where <= 1.0
    assert long_depth < short_depth


def test_a_finite_rings_gap_never_closes_at_any_field() -> None:
    # The claim the phase-diagram panel makes in one sentence and this makes in a
    # loop: k = 0 is not an allowed momentum of a finite ring, so there is no field
    # at which the cheapest excitation is free.
    for field in (0.0, 0.5, 0.9, 1.0, 1.1, 2.0):
        assert gap(TFIMSpec(n_sites=8, coupling=1.0, field=field)) > 0.1


def test_the_gap_this_ring_reports_is_its_cheapest_excitation() -> None:
    spec = TFIMSpec(n_sites=8, coupling=1.0, field=0.7)
    assert gap(spec) == pytest.approx(float(min(excitation_energies(spec))), abs=1e-12)


def test_the_infinite_chains_gap_closes_at_the_critical_field_and_nowhere_else() -> None:
    # The only place the two phases meet, and the reason the line in the figure is
    # a line rather than a decoration.
    assert gap_thermodynamic(1.0, 1.0) == 0.0
    for field in (0.0, 0.5, 0.99, 1.01, 2.0):
        assert gap_thermodynamic(1.0, field) > 0.0
    # Symmetric about the critical field, which is Kramers-Wannier duality showing
    # up in the one quantity the panel plots.
    assert gap_thermodynamic(1.0, 0.4) == pytest.approx(gap_thermodynamic(1.0, 1.6))


def test_the_rings_gap_approaches_the_infinite_chains_as_it_grows() -> None:
    # The two curves the panel draws together: the finite one sits above, and the
    # distance between them is what "finite" costs. It has to shrink with L or the
    # figure would be claiming something false about both.
    #
    # Within the L <= 8 cap this suite works under, so the claim is the trend and
    # not a converged number: the smallest allowed momentum is pi/L, and the gap it
    # carries falls towards 2|J - h| as that momentum shrinks.
    coupling, field = 1.0, 0.6
    errors = [
        gap(TFIMSpec(n_sites=sites, coupling=coupling, field=field))
        - gap_thermodynamic(coupling, field)
        for sites in SIZES
    ]
    assert all(error > 0.0 for error in errors)
    assert errors == sorted(errors, reverse=True)


# --- how the magnetisation responds to the field --------------------------


@pytest.mark.parametrize("field", [0.4, 1.0, 1.7])
def test_the_magnetisation_derivatives_match_a_central_difference(field: float) -> None:
    """The check that matters for algebra nobody can eyeball.

    Both are differentiated by hand -- the second is the *third* derivative of the
    energy density -- so the only honest test is against the quantity they claim to
    be the derivatives of. Finite differences are the wrong way to compute these and
    the right way to check them.
    """
    step = 1e-5

    def magnetisation(h: float) -> float:
        return transverse_magnetisation(TFIMSpec(n_sites=8, coupling=1.0, field=h))

    spec = TFIMSpec(n_sites=8, coupling=1.0, field=field)
    first = (magnetisation(field + step) - magnetisation(field - step)) / (2.0 * step)
    second = (
        magnetisation(field + step) - 2.0 * magnetisation(field) + magnetisation(field - step)
    ) / step**2
    assert magnetisation_slope(spec) == pytest.approx(first, abs=1e-7)
    assert magnetisation_curvature(spec) == pytest.approx(second, abs=1e-4)


@pytest.mark.parametrize("field", [0.2, 0.7, 1.0, 1.5, 2.0])
def test_turning_the_field_up_never_turns_the_alignment_down(field: float) -> None:
    # A susceptibility with the wrong sign is a bug however plausible the curve
    # looks, and this one is bounded below by zero for every field.
    spec = TFIMSpec(n_sites=8, coupling=1.0, field=field)
    assert magnetisation_slope(spec) >= 0.0


def test_the_magnetisation_slope_is_the_energy_curvature_negated() -> None:
    # Hellmann-Feynman, stated as an identity rather than trusted: the two are one
    # quantity, and a change to either that broke the relation would break this.
    for field in (0.3, 1.0, 1.8):
        spec = TFIMSpec(n_sites=10, coupling=1.0, field=field)
        assert magnetisation_slope(spec) == pytest.approx(-energy_density_curvature(spec))


def test_the_magnetisation_stops_steepening_before_it_saturates() -> None:
    # The second derivative changes sign, which is what makes "where does it rise
    # fastest" a crossing to solve for rather than a maximum to scan for. On a
    # finite chain that crossing sits a little below g = 1.
    assert magnetisation_curvature(TFIMSpec(n_sites=8, coupling=1.0, field=0.5)) > 0.0
    assert magnetisation_curvature(TFIMSpec(n_sites=8, coupling=1.0, field=1.5)) < 0.0
