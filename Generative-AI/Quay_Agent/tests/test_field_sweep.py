"""Tests for the exact field sweep, and for the many-body spectrum behind it.

Two things are under test here and only one of them is the sweep.

The **spectrum** is a new route to a quantity this project already had another
route to, so it is held against the one that shares no algebra with it: a dense
``numpy.linalg.eigvalsh`` of the whole :math:`2^L` matrix, which knows nothing about
momenta, parity sectors or Bogoliubov rotations. The comparison is on the *entire*
spectrum rather than on the lowest few levels, and that is deliberate -- a sector
construction can be wrong by one state and still get the ground state right in one
of the two phases, which is exactly the failure that would survive a check on the
gap alone.

The **sweep** is then checked for the properties a curve has to have before it is
worth plotting: that both solvers ran at every point and agreed, that the analytic
derivatives match finite differences of the quantities they are derivatives of, and
that the identities the figures rely on (Hellmann-Feynman above all) hold point by
point rather than on average.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.model import TFIMSpec
from src.physics.reference import exact_diagonalisation, free_fermions
from src.physics.reference.field_sweep import (
    DEFAULT_RATIO_MAX,
    MAX_POINTS,
    Curve,
    FieldSweep,
    _canonical_curves,
    sweep_field,
)

# Both phases and the critical point, plus the two edges. The ordered phase is where
# a parity mistake hides, so it is sampled more finely than the polarised one.
FIELDS = [0.0, 0.2, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0]

RING = TFIMSpec(n_sites=8, coupling=1.0, field=1.0)


def dense_spectrum(spec: TFIMSpec) -> np.ndarray:
    """Every eigenvalue of the chain, from a dense diagonalisation.

    Args:
        spec: The chain.

    Returns:
        All :math:`2^L` eigenvalues, ascending, with their multiplicities. A dense
        solver rather than the sparse one the reference module uses, because
        ``eigsh`` does not reliably return every copy of a degenerate level -- which
        this model has a whole phase of.
    """
    return np.sort(np.linalg.eigvalsh(exact_diagonalisation.hamiltonian(spec).toarray()))


# --------------------------------------------------------------------------
# The many-body spectrum, against algebra it shares nothing with
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", [4, 6, 8])
@pytest.mark.parametrize("field", FIELDS)
def test_the_free_fermion_levels_match_a_dense_diagonalisation(n_sites: int, field: float) -> None:
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=field)
    wanted = free_fermions.MAX_LEVELS
    mine = free_fermions.low_lying_levels(spec, count=wanted)
    theirs = dense_spectrum(spec)[:wanted]
    # Every level, with its multiplicity, in both phases. A construction that got one
    # sector's parity wrong would pass at h > J and fail here.
    assert np.allclose(mine, theirs, atol=1e-10), f"levels disagree at h={field}"


@pytest.mark.parametrize("field", FIELDS)
def test_the_lowest_level_is_the_ground_state_energy(field: float) -> None:
    spec = TFIMSpec(n_sites=10, coupling=1.0, field=field)
    lowest = float(free_fermions.low_lying_levels(spec, count=4)[0])
    assert lowest == pytest.approx(free_fermions.ground_state_energy(spec), abs=1e-12)


def test_the_ordered_phase_has_a_near_degenerate_pair_and_the_polarised_one_does_not() -> None:
    # The symmetry-broken doublet, and the one physical fact a naive positive-momentum
    # construction gets wrong: below the critical field the first excited state costs
    # essentially nothing, and above it the pair is cleanly split.
    ordered = free_fermions.low_lying_levels(TFIMSpec(n_sites=12, coupling=1.0, field=0.2))
    polarised = free_fermions.low_lying_levels(TFIMSpec(n_sites=12, coupling=1.0, field=2.0))
    assert ordered[1] - ordered[0] < 1e-6
    assert polarised[1] - polarised[0] > 1.0


def test_the_level_count_is_bounded_and_refused_outside_it() -> None:
    assert len(free_fermions.low_lying_levels(RING, count=3)) == 3
    with pytest.raises(ValueError, match="count must be between"):
        free_fermions.low_lying_levels(RING, count=free_fermions.MAX_LEVELS + 1)
    with pytest.raises(ValueError, match="count must be between"):
        free_fermions.low_lying_levels(RING, count=0)


def test_an_odd_ring_is_refused_rather_than_approximated() -> None:
    with pytest.raises(ValueError, match="cannot solve"):
        free_fermions.low_lying_levels(TFIMSpec(n_sites=7, coupling=1.0, field=1.0))


# --------------------------------------------------------------------------
# The sweep: two solvers, every point
# --------------------------------------------------------------------------


def test_every_point_is_computed_twice_and_they_agree() -> None:
    swept = sweep_field(RING, curves=["energy", "magnetisation"], points=21)
    assert swept.ok
    assert swept.methods == (free_fermions.METHOD_NAME, exact_diagonalisation.METHOD_NAME)
    assert swept.is_corroborated
    worst = swept.max_disagreement
    assert worst is not None and worst < 1e-9


def test_a_chain_too_long_to_diagonalise_is_swept_by_the_closed_form_alone() -> None:
    swept = sweep_field(TFIMSpec(n_sites=40, coupling=1.0, field=1.0), points=5)
    assert swept.ok
    assert swept.methods == (free_fermions.METHOD_NAME,)
    # Not corroborated, and it says so rather than going quiet about it.
    assert not swept.is_corroborated
    assert swept.max_disagreement is None
    assert "closed form alone" in swept.table()


def test_the_magnetisation_runs_from_none_to_nearly_all() -> None:
    swept = sweep_field(RING, curves=["magnetisation"], points=41)
    assert swept.points[0].magnetisation == pytest.approx(0.0, abs=1e-12)
    assert swept.points[-1].magnetisation > 0.9
    # The lower bound carries floating noise rather than a hard zero: at h = 0 the
    # magnetisation is a cancelling sum and lands a few times 1e-17 either side of it.
    assert all(-1e-12 <= point.magnetisation <= 1.0 for point in swept.points)


@pytest.mark.parametrize("field", [0.3, 0.9, 1.4])
def test_the_first_derivative_is_minus_the_magnetisation_exactly(field: float) -> None:
    # Hellmann-Feynman, and the identity every derivative figure here rests on: the
    # magnetisation panel and the first-derivative panel are one quantity with a sign
    # between them, not two measurements.
    spec = TFIMSpec(n_sites=10, coupling=1.0, field=field)
    swept = sweep_field(spec, curves=["energy_derivatives"], points=3)
    point = swept.points[0]
    assert point.energy_slope == pytest.approx(-point.magnetisation, abs=1e-12)


@pytest.mark.parametrize("field", [0.3, 0.9, 1.4])
def test_the_analytic_derivatives_match_finite_differences(field: float) -> None:
    # The analytic route is the one that ships, because a finite difference would make
    # the plotted physics depend on the point spacing. This is the check that the
    # closed-form differentiation is the derivative of the thing it claims.
    step = 1e-4

    def density(at: float) -> float:
        return free_fermions.energy_density(TFIMSpec(n_sites=10, coupling=1.0, field=at))

    def alignment(at: float) -> float:
        return free_fermions.transverse_magnetisation(TFIMSpec(n_sites=10, coupling=1.0, field=at))

    spec = TFIMSpec(n_sites=10, coupling=1.0, field=field)
    first = (density(field + step) - density(field - step)) / (2.0 * step)
    second = (density(field + step) - 2.0 * density(field) + density(field - step)) / step**2
    third = (alignment(field + step) - 2.0 * alignment(field) + alignment(field - step)) / step**2
    assert -free_fermions.transverse_magnetisation(spec) == pytest.approx(first, abs=1e-7)
    assert free_fermions.energy_density_curvature(spec) == pytest.approx(second, abs=1e-6)
    assert free_fermions.magnetisation_curvature(spec) == pytest.approx(third, abs=1e-4)


def test_the_susceptibility_peak_approaches_the_critical_field_from_below() -> None:
    # The finite chain's own estimate of where the transition is. It sits below
    # h/J = 1 and climbs towards it, and its height grows without bound -- both
    # halves are quoted in the notes and in the sweep's own summary lines, so both
    # are held here.
    peaks = []
    for n_sites in (8, 12, 60):
        swept = sweep_field(
            TFIMSpec(n_sites=n_sites, coupling=1.0, field=1.0),
            curves=["magnetisation_derivatives"],
            points=MAX_POINTS,
        )
        found = swept.peak_susceptibility
        assert found is not None
        peaks.append(found)
    # On a grid of this spacing the L = 60 peak rounds onto h/J = 1 exactly, so the
    # bound is inclusive here and the strict statement is made below, off the grid.
    assert all(0.8 <= where <= 1.0 for where, _ in peaks)
    assert [where for where, _ in peaks] == sorted(where for where, _ in peaks)
    assert [height for _, height in peaks] == sorted(height for _, height in peaks)


@pytest.mark.parametrize(("n_sites", "expected"), [(6, 0.835), (12, 0.949), (40, 0.994)])
def test_the_true_peak_of_the_susceptibility_sits_below_the_critical_field(
    n_sites: int, expected: float
) -> None:
    # Off the sweep's grid and straight from the solver, because the claim is about
    # the model rather than about a sampling: the peak is strictly below h/J = 1 and
    # climbs towards it with length. These are the numbers tabulated in
    # ``data/corpus/physics-notes/derivatives-of-the-ground-state-energy.md``, so a
    # change in either has to be a change in both.
    grid = np.linspace(0.5, 1.5, 2001)
    slopes = [
        free_fermions.magnetisation_slope(TFIMSpec(n_sites=n_sites, coupling=1.0, field=field))
        for field in grid
    ]
    where = float(grid[int(np.argmax(slopes))])
    assert where < 1.0
    assert where == pytest.approx(expected, abs=1e-3)


def test_the_curvature_dip_and_the_susceptibility_peak_are_one_feature() -> None:
    swept = sweep_field(RING, curves=["energy_derivatives"], points=41)
    bend = swept.sharpest_curvature
    peak = swept.peak_susceptibility
    assert bend is not None and peak is not None
    assert bend[0] == peak[0]
    assert bend[1] == pytest.approx(-peak[1], abs=1e-12)


def test_a_spectrum_sweep_carries_levels_and_a_non_zero_critical_gap() -> None:
    swept = sweep_field(RING, curves=["spectrum"], points=21)
    assert swept.shows_spectrum
    assert swept.levels_drawn >= 2
    critical = swept.critical_gap
    assert critical is not None
    where, size = critical
    assert where == pytest.approx(1.0, abs=1e-9)
    # Small but strictly positive: a finite ring has no gapless point, because the
    # momentum that would close the gap is not one it allows.
    assert 0.0 < size < 0.5
    assert all(point.excitations[0] == 0.0 for point in swept.points)


def test_a_sweep_without_a_spectrum_request_draws_no_levels() -> None:
    swept = sweep_field(RING, curves=["energy"], points=5)
    assert not swept.shows_spectrum
    assert swept.levels_drawn == 0
    assert swept.critical_gap is None


# --------------------------------------------------------------------------
# What the caller asked for, and what comes back
# --------------------------------------------------------------------------


def test_the_requested_curves_are_normalised_and_ordered() -> None:
    # Two callers asking for the same set in different orders get the same figure.
    assert _canonical_curves(["magnetisation", "spectrum"]) == ("spectrum", "magnetisation")
    assert _canonical_curves("spectrum") == ("spectrum",)
    assert _canonical_curves(["spectrum", "spectrum"]) == ("spectrum",)
    # Nothing recognisable falls back to the ground-state pair rather than to nothing.
    assert _canonical_curves(None) == ("energy", "magnetisation")
    assert _canonical_curves(["nonsense"]) == ("energy", "magnetisation")


def test_the_table_names_the_features_before_it_lists_the_rows() -> None:
    swept = sweep_field(RING, curves=["spectrum", "energy_derivatives"], points=41)
    table = swept.table()
    # The order is load-bearing: a tool result is clipped from the end, so the named
    # features have to appear above the rows they were computed from.
    features = table.index("most negative at h/J")
    rows = table.index("The curve itself")
    assert features < rows
    assert "Hellmann-Feynman" in table
    assert "agree to" in table


def test_the_table_carries_only_the_columns_that_were_asked_for() -> None:
    energy = sweep_field(RING, curves=["energy"], points=5).table()
    assert "E0/L=" in energy
    assert "<sigma^x>=" not in energy
    magnetisation = sweep_field(RING, curves=["magnetisation"], points=5).table()
    assert "<sigma^x>=" in magnetisation
    assert "E0/L=" not in magnetisation


def test_the_payload_is_json_shaped_and_parallel() -> None:
    payload = sweep_field(RING, curves=["energy", "spectrum"], points=7).payload()
    assert payload["cross_checked"] is True
    assert payload["curves_requested"] == ["spectrum", "energy"]
    assert len(payload["ratio"]) == 7
    for key in ("energy_density", "magnetisation", "energy_slope", "excitations"):
        assert len(payload[key]) == len(payload["ratio"])
    # Primitives only: this is about to be serialised into a model's context.
    assert all(isinstance(value, float) for value in payload["energy_density"])


def test_a_printed_zero_carries_no_minus_sign() -> None:
    # The magnetisation at h = 0 is a cancelling sum that lands on -1e-17, and a
    # minus sign on a quantity a reader has just been told is zero reads as a bug.
    table = sweep_field(RING, curves=["magnetisation"], points=5).table()
    assert "-0.000000" not in table


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (TFIMSpec(n_sites=7, coupling=1.0, field=1.0), "does not apply"),
        (TFIMSpec(n_sites=8, coupling=1.0, field=1.0, boundary="open"), "does not apply"),
    ],
)
def test_a_chain_the_closed_form_cannot_solve_comes_back_as_a_reason(
    spec: TFIMSpec, expected: str
) -> None:
    # Never raises: this runs behind a tool call, where an exception would discard an
    # answer that was otherwise complete.
    swept = sweep_field(spec, points=5)
    assert not swept.ok
    assert expected in swept.detail
    assert "error" in swept.payload()
    assert swept.explain().startswith("no field sweep")


def test_the_point_count_and_range_are_clamped_rather_than_refused() -> None:
    assert len(sweep_field(RING, points=1).points) == 3
    assert len(sweep_field(RING, points=10_000).points) == MAX_POINTS
    far = sweep_field(RING, points=3, ratio_max=99.0)
    assert far.points[-1].ratio <= 8.0
    near = sweep_field(RING, points=3, ratio_max=0.0)
    assert near.points[-1].ratio == pytest.approx(0.5)


def test_the_default_range_spans_both_phases() -> None:
    swept = sweep_field(RING, points=5)
    assert swept.points[0].ratio == 0.0
    assert swept.points[-1].ratio == pytest.approx(DEFAULT_RATIO_MAX)


def test_an_empty_sweep_answers_every_question_without_raising() -> None:
    # The shape a caller has to handle when nothing ran, exercised directly so that
    # no property of it is only ever reached through a failure.
    empty = FieldSweep(spec=None, points=(), curves=("energy",), methods=(), detail="none")
    assert not empty.ok
    assert empty.levels_drawn == 0
    assert empty.max_disagreement is None
    assert not empty.is_corroborated
    assert empty.critical_gap is None
    assert empty.peak_susceptibility is None
    assert empty.sharpest_curvature is None
    assert empty.unavailable() is None
    assert empty.table() == ""


def test_every_named_curve_is_one_the_type_declares() -> None:
    from typing import get_args

    declared = get_args(Curve)
    for name in declared:
        swept = sweep_field(RING, curves=[name], points=5)
        assert swept.curves == (name,)
        assert swept.table()


# --------------------------------------------------------------------------
# A curve on a lattice, and the two curves a lattice does not get
# --------------------------------------------------------------------------
#
# The defect these pin, which was the worst of the three. A question about a
# lattice's curve was answered by sweeping a *line* of the same number of sites:
# the per-point spec was rebuilt from four fields and the shape was not one of
# them, and the closed form -- which exists only for a line -- was then quoted as
# the exact answer for it. At sixteen sites those two energies are -1.28 and -2.13
# per spin. Not a small error; a different problem, plotted under the right title.


def test_a_lattice_gets_a_curve_computed_on_the_lattice() -> None:
    square = TFIMSpec(n_sites=12, boundary="open", geometry="square", rows=3)
    swept = sweep_field(square, curves=["energy", "magnetisation"], points=9)

    assert swept.ok
    assert swept.spec is not None
    assert swept.spec.geometry == "square"
    # Every point is on the lattice, not on a line of the same length. The check is
    # the number itself: a 3x4 square has 17 bonds against a line's 11, so at zero
    # field the energy density differs by a wide margin -- -17/12 against -11/12.
    at_zero = swept.points[0]
    assert at_zero.energy_density == pytest.approx(-17 / 12, abs=1e-9)
    assert at_zero.energy_density != pytest.approx(-11 / 12, abs=1e-3)


def test_every_lattice_point_is_computed_twice_and_the_two_agree() -> None:
    # The project's standing claim, held on the shape that has no closed form. The
    # second route assembles the same operator as a Kronecker product where the
    # first acts on basis integers with an XOR: no shared code, no shared bit
    # convention, so agreement is evidence rather than a tautology.
    for geometry, rows in [("square", 3), ("triangular", 3)]:
        spec = TFIMSpec(n_sites=12, boundary="open", geometry=geometry, rows=rows)  # type: ignore[arg-type]
        swept = sweep_field(spec, curves="energy", points=7)
        assert swept.ok
        assert all(point.disagreement is not None for point in swept.points)
        assert swept.is_corroborated, swept.max_disagreement
        assert swept.max_disagreement is not None
        assert swept.max_disagreement < 1e-12


def test_hellmann_feynman_holds_on_a_lattice_too() -> None:
    # The one derivative a lattice gets exactly. It is a statement about any
    # Hamiltonian depending linearly on a parameter, so it does not care what the
    # sites are connected to -- and it is what makes the energy's first derivative
    # reportable here when the second is not.
    spec = TFIMSpec(n_sites=9, boundary="open", geometry="triangular", rows=3)
    swept = sweep_field(spec, curves=["energy", "magnetisation"], points=9)
    for point in swept.points:
        assert point.energy_slope == pytest.approx(-point.magnetisation, abs=1e-12)


def test_the_derivative_curves_are_declined_by_name_rather_than_approximated() -> None:
    # A finite difference would make the plotted physics depend on how many points
    # were asked for, which is the one thing this module refuses to do. So the
    # curve is dropped and the reason is in the detail, where an answer can quote it.
    spec = TFIMSpec(n_sites=9, boundary="open", geometry="square", rows=3)
    swept = sweep_field(
        spec, curves=["energy", "energy_derivatives", "magnetisation_derivatives"], points=5
    )

    assert swept.ok
    assert "energy" in swept.curves
    assert "energy_derivatives" not in swept.curves
    assert "magnetisation_derivatives" not in swept.curves
    assert "finite difference" in swept.detail
    # And the fields those curves would have plotted are nan rather than a number
    # that looks computed. A zero here would plot as a flat susceptibility.
    assert np.isnan(swept.points[0].energy_curvature)
    assert np.isnan(swept.points[0].magnetisation_slope)
    assert np.isnan(swept.points[0].thermodynamic_energy_density)


def test_a_lattice_too_large_to_diagonalise_says_so_instead_of_raising() -> None:
    # This runs behind a tool call, where an exception discards an answer that was
    # otherwise complete. So the refusal is a value, and it names the size. Twenty
    # sites is the smallest square past the cap and nothing is ever built at it --
    # the whole test is that the matrix is declined rather than assembled.
    spec = TFIMSpec(n_sites=20, boundary="open", geometry="square", rows=4)
    swept = sweep_field(spec, curves="energy", points=5)
    assert not swept.ok
    assert "20 sites" in swept.detail
    assert "4x5" in swept.detail


def test_a_lattice_spectrum_is_drawn_from_the_lattice_levels() -> None:
    spec = TFIMSpec(n_sites=9, boundary="open", geometry="square", rows=3)
    swept = sweep_field(spec, curves="spectrum", points=5, levels=4)
    assert swept.shows_spectrum
    assert swept.levels_drawn >= 4
    for point in swept.points:
        # Ascending, and the ground state is the lowest of them.
        assert list(point.levels) == sorted(point.levels)
        assert point.excitations[0] == pytest.approx(0.0, abs=1e-12)


def test_the_line_still_sweeps_by_the_closed_form_and_nothing_moved() -> None:
    # The 2D branch must not have changed the 1D one. A line keeps the closed form
    # as its first method, which is the route the whole existing suite pins.
    line = sweep_field(TFIMSpec(n_sites=8), points=11)
    assert line.ok
    assert line.methods[0] == free_fermions.METHOD_NAME
    assert line.is_corroborated
    assert not np.isnan(line.points[0].energy_curvature)


def test_the_infinite_chain_energy_agrees_by_two_routes_that_share_no_algebra() -> None:
    r"""The closed form against quadrature, which is why the second one exists.

    ``energy_density_thermodynamic`` evaluates an elliptic integral;
    ``energy_density_thermodynamic_quadrature`` integrates the dispersion
    :math:`\sqrt{J^2 + h^2 - 2Jh\cos k}` numerically. They share no algebra, so
    agreement is evidence that neither was transcribed wrongly.

    The quadrature route's own docstring says that is what it is for, and until this
    test existed nothing in the project ever called it -- so the cross-check the
    house rule asks for was written down and never performed. The two are compared
    across the transition rather than at one field: an elliptic-integral parameter
    mistake is easy to hide at :math:`h/J = 1`, where the argument is exactly one.
    """
    for ratio in (0.0, 0.25, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0):
        closed = free_fermions.energy_density_thermodynamic(1.0, ratio)
        numeric = free_fermions.energy_density_thermodynamic_quadrature(1.0, ratio)
        assert closed == pytest.approx(numeric, abs=1e-9), f"h/J = {ratio}"

    # And the one value the closed form is quoted for: at criticality the energy
    # density of the infinite chain is exactly -4J/pi.
    assert free_fermions.energy_density_thermodynamic(1.0, 1.0) == pytest.approx(
        -4.0 / np.pi, abs=1e-12
    )

    # Both refuse the same parameters, which is the other half of being comparable:
    # one route quietly accepting what the other rejects would turn a cross-check
    # into a comparison of two different problems.
    for coupling, field in ((0.0, 1.0), (-1.0, 1.0), (1.0, -1.0)):
        with pytest.raises(ValueError):
            free_fermions.energy_density_thermodynamic(coupling, field)
        with pytest.raises(ValueError):
            free_fermions.energy_density_thermodynamic_quadrature(coupling, field)
