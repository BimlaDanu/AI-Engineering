"""The cross-check, and the four things it can report.

The module states the project's central claim -- a number is trusted only when two
methods that share no algebra produce it -- and its doctests covered the verdict but
not the words it says about one. :meth:`~src.verification.cross_check.CrossCheck.summary`
is what a log line and a prompt read, so what it says has to be true of what ran.
"""

from __future__ import annotations

import pytest

from src.physics.model import TFIMSpec
from src.verification.cross_check import (
    AGREEMENT_TOLERANCE,
    CrossCheck,
    MethodResult,
    cross_check,
)


def test_two_routes_that_share_no_algebra_agree_and_the_report_says_so() -> None:
    check = cross_check(TFIMSpec(n_sites=4, field=0.7))

    assert len(check.results) == 2, "the even ring should reach both reference methods"
    assert {result.method for result in check.results} == {
        "pfeuty_exact",
        "exact_diagonalisation",
    }
    assert check.is_corroborated
    assert check.energy is not None
    summary = check.summary()
    assert "VERIFIED" in summary
    assert all(result.method in summary for result in check.results)


def test_one_method_alone_is_never_corroborated_however_exact_it_is() -> None:
    # An odd ring is outside the closed form's momentum set, so only the matrix
    # method applies. A single unopposed answer is the case the whole module exists
    # to refuse to bless.
    check = cross_check(TFIMSpec(n_sites=5))

    assert len(check.results) == 1
    assert check.max_disagreement is None
    assert not check.is_corroborated
    assert check.energy is not None, "it still has an answer -- it just has no second opinion"
    assert "only one method applies" in check.summary()
    assert check.rejected, "the report must say why the other method was unavailable"


def test_a_specification_no_method_solves_says_that_and_not_something_else() -> None:
    # Both this and the one-method case leave the spread undefined, and reporting
    # them identically had the summary print "no method could solve this
    # specification" and "only one method applies" on consecutive lines.
    empty = CrossCheck(spec=TFIMSpec(n_sites=4), results=(), rejected=())
    summary = empty.summary()

    assert empty.energy is None
    assert not empty.is_corroborated
    assert "no method could solve this specification" in summary
    assert "no method applies" in summary
    assert "only one method applies" not in summary, "it contradicts the line above it"


def test_two_methods_that_disagree_are_reported_as_a_contradiction() -> None:
    clash = CrossCheck(
        spec=TFIMSpec(n_sites=4),
        results=(MethodResult("a", -5.0), MethodResult("b", -4.0)),
        rejected=(),
    )

    assert clash.max_disagreement == pytest.approx(1.0)
    assert not clash.is_corroborated
    assert "CONTRADICTION" in clash.summary()
    assert "VERIFIED" not in clash.summary()


@pytest.mark.parametrize("inside", (True, False))
def test_the_agreement_tolerance_is_applied_at_its_own_boundary(inside: bool) -> None:
    # Relative to the larger energy, not absolute: the tolerance has to mean the
    # same thing for a four-spin chain and a sixteen-spin one.
    energy = -8.0
    scale = max(1.0, abs(energy))
    gap = AGREEMENT_TOLERANCE * scale * (0.5 if inside else 2.0)
    check = CrossCheck(
        spec=TFIMSpec(n_sites=4),
        results=(MethodResult("a", energy), MethodResult("b", energy + gap)),
        rejected=(),
    )

    assert check.is_corroborated is inside
