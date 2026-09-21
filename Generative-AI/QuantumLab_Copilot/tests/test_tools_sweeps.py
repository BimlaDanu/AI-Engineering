"""Tests for the field sweep.

The claim being tested is that a *curve* is verified the way a single number is:
every point computed twice by methods sharing no algebra. A plot is the easiest
place in an application to smuggle in an unchecked number, because a reader checks
a headline figure and trusts a line.

Physics identities are used as the checks where possible -- the magnetisation
vanishes in zero field and saturates when the field dominates -- because they hold
at every parameter value rather than at the one a golden number was recorded at.
"""

from __future__ import annotations

import pytest

from src.physics.model import TFIMSpec
from src.tools.sweeps import (
    MAX_COSTLY_SITES,
    MAX_POINTS,
    MAX_RATIO,
    methods_for,
    sweep_field,
)

RING = TFIMSpec(n_sites=6, coupling=1.0, field=1.0)


def test_every_point_is_computed_by_two_independent_methods() -> None:
    sweep = sweep_field(RING, points=11)
    assert sweep.methods == ("pfeuty_exact", "exact_diagonalisation")
    assert all(point.disagreement is not None for point in sweep.points)
    assert sweep.is_corroborated


def test_the_worst_point_on_the_curve_is_reported_not_the_average() -> None:
    # A reader deciding whether to trust a plot is asking about its worst point.
    sweep = sweep_field(RING, points=11)
    worst = sweep.max_disagreement
    assert worst is not None
    assert worst < 1e-12


def test_the_magnetisation_vanishes_in_zero_field() -> None:
    # An exact identity, not a recorded number: with no transverse field the spins
    # lie along z and <sigma_x> is identically zero at every chain length.
    sweep = sweep_field(RING, points=11)
    assert abs(sweep.points[0].magnetisation) < 1e-12


def test_the_magnetisation_saturates_when_the_field_dominates() -> None:
    sweep = sweep_field(RING, points=11)
    assert sweep.points[-1].magnetisation > 0.9
    assert sweep.points[-1].ratio == MAX_RATIO


def test_the_magnetisation_never_decreases_along_the_curve() -> None:
    # Monotonic in h for a ferromagnetic chain. A sign error or a mismatched
    # method would show up here as a kink long before it showed up in a number.
    sweep = sweep_field(RING, points=21)
    values = [point.magnetisation for point in sweep.points]
    assert values == sorted(values)


def test_the_curve_finds_its_own_turning_point_near_the_critical_field() -> None:
    # The finite chain's answer to "where is the transition?" -- computed rather
    # than assumed. It is near g = 1 without being at it, which is the honest
    # result for six spins.
    turning = sweep_field(RING, points=41).turning_point
    assert turning is not None
    assert 0.7 < turning < 1.3


def test_an_odd_chain_is_swept_by_one_method_and_says_so() -> None:
    sweep = sweep_field(TFIMSpec(n_sites=5), points=9)
    assert sweep.ok
    assert sweep.methods == ("exact_diagonalisation",)
    assert not sweep.is_corroborated
    assert sweep.max_disagreement is None
    assert "one method only" in sweep.explain()


def test_the_point_count_is_clamped_rather_than_honoured() -> None:
    assert len(sweep_field(RING, points=10_000).points) == MAX_POINTS
    assert len(sweep_field(RING, points=0).points) == 3


def test_a_chain_too_long_to_sweep_is_declined_with_the_reason() -> None:
    # A sweep multiplies the cost of one solve by the number of points, so it
    # stops earlier than a single run does.
    long_open = TFIMSpec(n_sites=MAX_COSTLY_SITES + 1, boundary="open")
    sweep = sweep_field(long_open, points=11)
    assert not sweep.ok
    assert "no method can sweep" in sweep.detail
    assert methods_for(long_open) == ()


def test_an_even_ring_is_always_sweepable_because_the_closed_form_is_free() -> None:
    # O(L) per point, so length is no obstacle on this path -- and the even ring
    # is the case where an independent check exists at all.
    assert methods_for(TFIMSpec(n_sites=12)) == ("pfeuty_exact",)


def test_a_lowered_ceiling_drops_the_expensive_solver_from_the_sweep() -> None:
    # The interface offers this ceiling as a slider, and for a long time nothing
    # read it: both the Lab plot and the agent's sweep tool used the module default
    # instead, so the slider moved and no curve changed. A six-spin ring is inside
    # the default and outside a ceiling of four, which is the whole difference.
    ring = TFIMSpec(n_sites=6)
    assert methods_for(ring) == ("pfeuty_exact", "exact_diagonalisation")
    assert methods_for(ring, max_costly_sites=4) == ("pfeuty_exact",)


def test_a_sweep_run_under_a_lowered_ceiling_says_so_with_that_number() -> None:
    # Not the module default: a reader who set the limit to four and is told the
    # limit is eight has been handed a number that contradicts their own screen.
    sweep = sweep_field(TFIMSpec(n_sites=6), points=5, observable="spectrum", max_costly_sites=4)
    assert sweep.ok and not sweep.has_spectrum  # the closed form alone has no levels
    missing = sweep.unavailable_observable()
    assert missing is not None
    assert "L = 4" in missing
    assert str(MAX_COSTLY_SITES) not in missing


def test_the_prompt_table_is_thinned_rather_than_dumped() -> None:
    table = sweep_field(RING, points=121).table()
    assert table.count("g=") <= 12
    assert "rises fastest at g =" in table


def test_a_declined_sweep_renders_no_table() -> None:
    assert sweep_field(TFIMSpec(n_sites=11, boundary="open"), points=9).table() == ""


# --------------------------------------------------------------------------
# The spectrum: excitations, not ground-state properties
# --------------------------------------------------------------------------


def test_the_levels_are_measured_from_the_ground_state() -> None:
    sweep = sweep_field(RING, points=9, observable="spectrum")
    for point in sweep.points:
        assert point.excitations[0] == 0.0
        # Ascending, because they came out of an eigensolver sorted and the plot
        # draws them as separate lines.
        assert list(point.excitations) == sorted(point.excitations)


def test_the_gap_is_the_first_excitation() -> None:
    sweep = sweep_field(RING, points=9, observable="spectrum")
    for point in sweep.points:
        assert point.gap == point.excitations[1]


def test_the_ordered_chain_has_a_degenerate_pair_and_the_field_splits_it() -> None:
    # The physics the spectrum panel exists to show: at h = 0 the two lowest levels
    # are exactly degenerate, and the gap grows once the field dominates.
    sweep = sweep_field(RING, points=21, observable="spectrum")
    first, last = sweep.points[0], sweep.points[-1]
    assert first.gap == pytest.approx(0.0, abs=1e-12)
    assert last.gap is not None and last.gap > 1.0


def test_the_quoted_gap_is_the_one_at_the_critical_field() -> None:
    # Not the smallest gap on the curve, which is always the exact degeneracy at
    # h = 0 and says nothing about criticality.
    sweep = sweep_field(RING, points=21, observable="spectrum")
    critical = sweep.critical_gap
    assert critical is not None
    ratio, gap = critical
    assert ratio == pytest.approx(1.0)
    assert 0.0 < gap < 1.0


def test_a_free_fermion_only_sweep_carries_no_spectrum() -> None:
    # L = 10 is past MAX_COSTLY_SITES, so only the closed form runs -- and it gives
    # the ground state, not the levels above it. Honest absence, not a zero.
    sweep = sweep_field(TFIMSpec(n_sites=10), points=5, observable="spectrum")
    assert sweep.ok
    assert sweep.methods == ("pfeuty_exact",)
    assert not sweep.has_spectrum
    assert sweep.levels_drawn == 0
    assert sweep.critical_gap is None
    assert not sweep.wants_spectrum


# --------------------------------------------------------------------------
# The observable is the agent's choice, and it is honoured
# --------------------------------------------------------------------------


def test_the_default_sweep_is_about_the_ground_state() -> None:
    sweep = sweep_field(RING, points=5)
    assert sweep.observable == "ground_state"
    assert sweep.wants_ground_state
    assert not sweep.wants_spectrum


def test_asking_for_the_spectrum_suppresses_the_ground_state_curves() -> None:
    sweep = sweep_field(RING, points=5, observable="spectrum")
    assert sweep.wants_spectrum
    assert not sweep.wants_ground_state


def test_asking_for_both_shows_both() -> None:
    sweep = sweep_field(RING, points=5, observable="both")
    assert sweep.wants_spectrum
    assert sweep.wants_ground_state


def test_the_table_names_the_quantity_that_was_asked_for() -> None:
    # The narrator writes about what it is shown, so the request has to be in the
    # material rather than only in the tool call.
    assert "excitation spectrum" in sweep_field(RING, points=9, observable="spectrum").table()
    assert "ground-state properties" in sweep_field(RING, points=9).table()


def test_the_gap_is_withheld_from_a_ground_state_table() -> None:
    # The levels were computed either way; showing them to a narrator answering a
    # question about the magnetisation invites a paragraph nobody asked for.
    ground = sweep_field(RING, points=9).table()
    assert "gap=" not in ground
    assert "gap=" in sweep_field(RING, points=9, observable="spectrum").table()


# --------------------------------------------------------------------------
# Derivatives: a third quantity, not the magnetisation under another name
# --------------------------------------------------------------------------


def test_asking_for_the_derivatives_does_not_get_the_magnetisation_panel() -> None:
    """The figure rule, which is where the original bug was.

    A question about the first and second derivatives of the energy was answered
    with a magnetisation curve -- the first derivative wearing a different name, and
    silent about the second. That is a claim about what is *drawn*, and it still
    holds: ``wants_ground_state`` stays false, so the ground-state panels do not
    appear. The narrator's table is a separate matter and now carries both sets of
    columns; see the test below for why copying this rule into the text was wrong.
    """
    sweep = sweep_field(RING, points=9, observable="derivatives")
    assert sweep.wants_derivatives
    assert not sweep.wants_ground_state
    assert not sweep.wants_spectrum


def test_the_derivatives_reach_the_narrator_as_numbers() -> None:
    # Without these columns the narrator wrote that the derivatives "were not
    # determined" -- correctly, since it may only use numbers it was given.
    table = sweep_field(RING, points=9, observable="derivatives").table()
    assert "d(E0/L)/dh=" in table
    assert "d2(E0/L)/dh2=" in table
    assert "most negative at g =" in table


def test_the_first_derivative_is_minus_the_magnetisation() -> None:
    # Hellmann-Feynman, and the check that the two columns are the same physics.
    sweep = sweep_field(RING, points=9, observable="derivatives")
    for point in sweep.points:
        assert point.slope == pytest.approx(-point.magnetisation, abs=1e-12)


def test_the_curvature_dips_near_the_critical_field() -> None:
    sweep = sweep_field(RING, points=21, observable="derivatives")
    sharpest = sweep.sharpest_curvature
    assert sharpest is not None
    where, bend = sharpest
    assert bend < 0.0
    assert 0.5 < where < 1.5
    assert all(point.curvature is not None and point.curvature <= 0.0 for point in sweep.points)


def test_derivatives_are_absent_where_the_closed_form_does_not_apply() -> None:
    # Open boundaries: no free-fermion solution here, so no analytic derivatives.
    # Absent rather than finite-differenced, which would depend on the spacing.
    sweep = sweep_field(TFIMSpec(n_sites=6, boundary="open"), points=5, observable="derivatives")
    assert sweep.ok
    assert not sweep.has_derivatives
    assert not sweep.wants_derivatives
    assert "d(E0/L)/dh=" not in sweep.table()


# --------------------------------------------------------------------------
# A sweep that could not answer the question says so
# --------------------------------------------------------------------------


def test_an_open_chain_asked_for_derivatives_says_they_were_not_computed() -> None:
    # The failure this closes: the derivatives come only from the closed form, which
    # an open chain has none of, so the sweep quietly produced the magnetisation
    # curve under a heading announcing the derivatives -- the same substitution the
    # `observable` choice exists to prevent, reached from the other side. The
    # narrator may only use what it is handed, so the absence has to be handed to it.
    sweep = sweep_field(TFIMSpec(n_sites=6, boundary="open"), points=5, observable="derivatives")
    assert sweep.ok
    assert not sweep.has_derivatives
    missing = sweep.unavailable_observable()
    assert missing is not None
    assert "derivatives" in missing
    assert "periodic ring" in missing  # the reason, not merely the fact
    assert "NOT COMPUTED" in sweep.table()


def test_a_long_chain_asked_for_the_spectrum_says_it_was_not_computed() -> None:
    # Same failure, other observable: past MAX_COSTLY_SITES a sweep runs the closed
    # form alone, which yields no many-body levels at all.
    sweep = sweep_field(TFIMSpec(n_sites=MAX_COSTLY_SITES + 2), points=5, observable="spectrum")
    assert sweep.ok
    assert not sweep.has_spectrum
    missing = sweep.unavailable_observable()
    assert missing is not None
    assert "spectrum" in missing
    assert str(MAX_COSTLY_SITES) in missing


def test_a_sweep_that_answered_the_question_adds_no_apology() -> None:
    for observable in ("ground_state", "spectrum", "derivatives", "both"):
        sweep = sweep_field(RING, points=5, observable=observable)
        assert sweep.unavailable_observable() is None, observable
        assert "NOT COMPUTED" not in sweep.table(), observable


def test_a_ground_state_sweep_hands_the_narrator_the_magnetisation() -> None:
    # The complement of `test_asking_for_the_derivatives_suppresses_the_magnetisation`,
    # which only checks that the column is absent: without this, renaming the column
    # would leave that test passing while the narrator lost the number.
    table = sweep_field(RING, points=9, observable="ground_state").table()
    assert "<sigma_x>" in table
    assert "magnetisation rises fastest" in table


def test_a_sweep_is_never_shorter_than_a_curve() -> None:
    # Two points are a line segment and one is a dot; the turning point needs three.
    # The floor is silent by design, so it is asserted rather than assumed.
    for asked in (0, 1, 2, 3):
        assert len(sweep_field(RING, points=asked).points) == 3


def test_a_derivatives_sweep_carries_the_magnetisations_own_derivatives() -> None:
    """Reported in use, and the reason both were added to the point.

    Asked to *plot the ground state magnetisation and its first and second
    derivatives*, the sweep returned the energy density's derivatives and the answer
    said the magnetisation and its derivatives "were not computed in the supplied
    data" -- true of the table it was handed, and false of the physics, which had
    the first one sitting there with a sign on it.
    """
    curve = sweep_field(TFIMSpec(n_sites=6), points=9, observable="derivatives")
    assert curve.wants_derivatives
    for point in curve.points:
        assert point.magnetisation_slope is not None
        assert point.magnetisation_curvature is not None
        # Hellmann-Feynman, per point: one physics, two readings.
        assert point.curvature is not None
        assert point.magnetisation_slope == pytest.approx(-point.curvature)


def test_the_narrator_is_given_the_magnetisation_on_a_derivatives_sweep() -> None:
    # The table is what the narrator writes from, so a column missing there is a
    # quantity that does not exist as far as the answer is concerned.
    table = sweep_field(TFIMSpec(n_sites=6), points=9, observable="derivatives").table()
    assert "<sigma_x>=" in table
    assert "d<sigma_x>/dh=" in table
    assert "d2<sigma_x>/dh2=" in table
    assert "d2(E0/L)/dh2=" in table
    # And it must not be told the quantity was skipped, which is what sent the
    # answer off saying it could not report it.
    assert "not the magnetisation" not in table
