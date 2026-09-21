"""Tests for the cross-method verification step.

Nothing here runs a chain longer than eight sites.
"""

from __future__ import annotations

import pytest

from src.physics import ed, exact
from src.physics.model import TFIMSpec
from src.physics.registry import get_method
from src.verification.cross_check import (
    AGREEMENT_TOLERANCE,
    CrossCheck,
    MethodResult,
    cross_check,
)

SIZES = [2, 4, 6, 8]


def fake(*energies: float) -> CrossCheck:
    """Build a CrossCheck from given energies, bypassing the solvers."""
    return CrossCheck(
        spec=TFIMSpec(n_sites=4),
        results=tuple(MethodResult(method=f"m{i}", energy=e) for i, e in enumerate(energies)),
        rejected=(),
    )


# --------------------------------------------------------------------------
# The headline claim
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
def test_two_independent_methods_corroborate_an_even_ring(n_sites: int) -> None:
    check = cross_check(TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.7))
    assert {result.method for result in check.results} == {exact.METHOD_NAME, ed.METHOD_NAME}
    assert check.is_corroborated
    assert check.max_disagreement is not None
    assert check.max_disagreement < 1e-12


@pytest.mark.parametrize("field", [0.0, 0.5, 1.0, 2.0])
def test_corroboration_holds_across_the_critical_point(field: float) -> None:
    # h = J is where the free-fermion gap closes and a wrong momentum
    # convention would show up first.
    assert cross_check(TFIMSpec(n_sites=6, field=field)).is_corroborated


def test_the_reported_energy_is_the_agreed_one() -> None:
    spec = TFIMSpec(n_sites=6, field=0.7)
    check = cross_check(spec)
    assert check.energy == pytest.approx(exact.ground_state_energy(spec), abs=1e-12)


# --------------------------------------------------------------------------
# A single answer is not a verified answer
# --------------------------------------------------------------------------


def test_one_method_alone_is_not_corroborated() -> None:
    # An open chain is exact diagonalisation only. The number is right; it is
    # simply unchecked, and the difference must survive to the caller.
    check = cross_check(TFIMSpec(n_sites=6, boundary="open"))
    assert len(check.results) == 1
    assert check.energy is not None
    assert not check.is_corroborated
    assert check.max_disagreement is None


def test_an_unavailable_method_reports_its_reason() -> None:
    check = cross_check(TFIMSpec(n_sites=7))
    assert [item.method.name for item in check.rejected] == [exact.METHOD_NAME]
    assert "even L" in check.rejected[0].reason


def test_nothing_applies_when_the_problem_is_too_large() -> None:
    spec = TFIMSpec(n_sites=13, boundary="open")
    check = cross_check(spec)
    assert check.results == ()
    assert check.energy is None
    assert not check.is_corroborated
    assert len(check.rejected) == 2


# --------------------------------------------------------------------------
# The tolerance actually discriminates
# --------------------------------------------------------------------------


def test_a_disagreement_beyond_tolerance_is_not_corroborated() -> None:
    assert not fake(-5.0, -5.0 + 1e-3).is_corroborated


def test_the_tolerance_scales_with_the_size_of_the_energy() -> None:
    # Energies grow with L; a fixed absolute tolerance would tighten silently
    # as the chain gets longer.
    assert fake(-1000.0, -1000.0 - AGREEMENT_TOLERANCE * 500).is_corroborated
    assert not fake(-1000.0, -1000.0 - AGREEMENT_TOLERANCE * 2000).is_corroborated


def test_the_spread_is_taken_over_every_pair() -> None:
    assert fake(-5.0, -5.0, -4.0).max_disagreement == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The text that reaches a log or a prompt
# --------------------------------------------------------------------------


def test_the_summary_says_verified_and_names_both_methods() -> None:
    text = cross_check(TFIMSpec(n_sites=4, field=0.7)).summary()
    assert "VERIFIED" in text
    assert exact.METHOD_NAME in text and ed.METHOD_NAME in text


def test_the_summary_flags_an_unchecked_answer() -> None:
    assert "UNCORROBORATED" in cross_check(TFIMSpec(n_sites=6, boundary="open")).summary()


def test_the_summary_flags_a_contradiction() -> None:
    assert "CONTRADICTION" in fake(-5.0, -4.0).summary()


def test_the_summary_is_deterministic() -> None:
    spec = TFIMSpec(n_sites=6, field=0.3)
    assert cross_check(spec).summary() == cross_check(spec).summary()


def test_the_result_cannot_be_edited_after_the_fact() -> None:
    with pytest.raises(AttributeError):
        cross_check(TFIMSpec(n_sites=4)).results = ()  # type: ignore[misc]


# --------------------------------------------------------------------------
# Exact limits, checked through the verification path
# --------------------------------------------------------------------------


def test_a_zero_field_ring_hits_the_classical_ground_state() -> None:
    spec = TFIMSpec(n_sites=6, coupling=1.5, field=0.0)
    check = cross_check(spec)
    assert check.is_corroborated
    assert check.energy == pytest.approx(-spec.coupling * spec.n_bonds, abs=1e-12)


def test_kramers_wannier_duality_survives_the_verification_path() -> None:
    direct = cross_check(TFIMSpec(n_sites=6, coupling=1.0, field=0.4))
    dual = cross_check(TFIMSpec(n_sites=6, coupling=0.4, field=1.0))
    assert direct.energy is not None and dual.energy is not None
    assert direct.energy == pytest.approx(dual.energy, abs=1e-12)


def test_the_registry_and_the_cross_check_run_the_same_function() -> None:
    # Guards against the verification layer quietly calling something other
    # than the method it names in its output.
    spec = TFIMSpec(n_sites=4, field=0.7)
    for result in cross_check(spec).results:
        assert result.energy == get_method(result.method).ground_state_energy(spec)
