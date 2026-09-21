"""Tests for the method registry.

Nothing here runs a solver on a chain longer than eight sites. The registry's
job is bookkeeping -- names, applicability, cost facts -- and the few tests that
do compute a number use the smallest spec that makes the point.
"""

from __future__ import annotations

import pytest

from src.physics import ed, exact
from src.physics.model import MAX_SITES_STATEVECTOR, TFIMSpec
from src.physics.registry import (
    COST_ORDER,
    MethodInfo,
    MethodSurvey,
    Rejection,
    all_methods,
    get_method,
    method_names,
    survey,
)

SIZES = [2, 4, 6, 8]


# --------------------------------------------------------------------------
# Registry integrity
# --------------------------------------------------------------------------


def test_the_registry_is_not_empty() -> None:
    assert len(all_methods()) >= 2


def test_registry_names_are_unique() -> None:
    names = method_names()
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("module", [exact, ed])
def test_every_solver_module_is_registered_under_its_own_method_name(module: object) -> None:
    # Guards against the registry and the implementation drifting apart, which
    # would surface as an agent naming a method that cannot be looked up.
    name = module.METHOD_NAME  # type: ignore[attr-defined]
    assert name in method_names()


@pytest.mark.parametrize("info", all_methods(), ids=lambda info: info.name)
def test_registered_functions_come_from_the_module_they_claim(info: MethodInfo) -> None:
    module = {exact.METHOD_NAME: exact, ed.METHOD_NAME: ed}[info.name]
    assert info.unsupported_reason is module.unsupported_reason
    assert info.ground_state_energy is module.ground_state_energy


@pytest.mark.parametrize("info", all_methods(), ids=lambda info: info.name)
def test_every_method_declares_its_cost_and_accuracy_class(info: MethodInfo) -> None:
    assert info.cost in COST_ORDER
    assert info.accuracy in {"exact", "variational_bound", "uncontrolled"}


@pytest.mark.parametrize("info", all_methods(), ids=lambda info: info.name)
def test_every_method_explains_itself_in_prose(info: MethodInfo) -> None:
    # These strings end up in a prompt and in the justification shown to the
    # user, so an empty or placeholder one is a real defect.
    assert len(info.summary) > 40
    assert len(info.when_to_use) > 40


def test_the_registry_cannot_be_extended_by_a_caller() -> None:
    assert isinstance(all_methods(), tuple)


def test_cost_order_ranks_cheapest_first() -> None:
    assert COST_ORDER["constant"] < COST_ORDER["linear"] < COST_ORDER["exponential"]


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", method_names())
def test_get_method_round_trips_every_registered_name(name: str) -> None:
    assert get_method(name).name == name


def test_get_method_rejects_an_unknown_name_and_lists_the_valid_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_method("dmrg")
    message = str(excinfo.value)
    assert "dmrg" in message
    for name in method_names():
        assert name in message


# --------------------------------------------------------------------------
# Applicability
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
def test_an_even_periodic_ring_can_be_solved_both_ways(n_sites: int) -> None:
    result = survey(TFIMSpec(n_sites=n_sites))
    assert set(result.names()) == {exact.METHOD_NAME, ed.METHOD_NAME}
    assert result.rejected == ()


def test_an_open_chain_leaves_only_exact_diagonalisation() -> None:
    result = survey(TFIMSpec(n_sites=6, boundary="open"))
    assert result.names() == (ed.METHOD_NAME,)
    assert result.rejected[0].method.name == exact.METHOD_NAME
    assert "periodic" in result.rejected[0].reason


def test_an_odd_chain_leaves_only_exact_diagonalisation() -> None:
    result = survey(TFIMSpec(n_sites=7))
    assert result.names() == (ed.METHOD_NAME,)
    assert "even L" in result.rejected[0].reason


@pytest.mark.parametrize("n_sites", SIZES)
def test_the_survey_partitions_the_registry(n_sites: int) -> None:
    result = survey(TFIMSpec(n_sites=n_sites, boundary="open"))
    surveyed = {info.name for info in result.applicable}
    surveyed |= {item.method.name for item in result.rejected}
    assert surveyed == set(method_names())
    assert len(result.applicable) + len(result.rejected) == len(all_methods())


@pytest.mark.parametrize("info", all_methods(), ids=lambda info: info.name)
def test_applies_to_agrees_with_the_underlying_check(info: MethodInfo) -> None:
    for boundary in ("periodic", "open"):
        for n_sites in (4, 5):
            spec = TFIMSpec(n_sites=n_sites, boundary=boundary)
            assert info.applies_to(spec) == (info.unsupported_reason(spec) is None)


def test_a_rejection_reason_is_a_sentence_not_a_flag() -> None:
    result = survey(TFIMSpec(n_sites=5))
    reason = result.rejected[0].reason
    assert isinstance(reason, str)
    assert len(reason.split()) > 5


def test_the_survey_reports_whether_anything_applies() -> None:
    assert survey(TFIMSpec(n_sites=4)).has_applicable_method
    empty = MethodSurvey(spec=TFIMSpec(n_sites=4), applicable=(), rejected=())
    assert not empty.has_applicable_method


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def test_the_closed_form_declares_no_memory_estimate() -> None:
    # O(L) arithmetic on an array of L/2 momenta needs no approval gate.
    assert get_method(exact.METHOD_NAME).memory_bytes(TFIMSpec(n_sites=8)) is None


@pytest.mark.parametrize("n_sites", SIZES)
def test_exact_diagonalisation_predicts_a_positive_footprint(n_sites: int) -> None:
    estimate = get_method(ed.METHOD_NAME).memory_bytes(TFIMSpec(n_sites=n_sites))
    assert estimate is not None
    assert estimate > 0


def test_the_predicted_footprint_grows_at_least_exponentially() -> None:
    info = get_method(ed.METHOD_NAME)
    small = info.memory_bytes(TFIMSpec(n_sites=6))
    large = info.memory_bytes(TFIMSpec(n_sites=8))
    assert small is not None and large is not None
    # Two extra sites quadruple the Hilbert space; the estimate must see that.
    assert large >= 4 * small


def test_the_footprint_estimate_survives_a_spec_the_method_would_refuse() -> None:
    # The agent needs the number precisely to explain the refusal, so costing
    # an over-cap spec must not raise.
    spec = TFIMSpec(n_sites=MAX_SITES_STATEVECTOR + 4)
    estimate = get_method(ed.METHOD_NAME).memory_bytes(spec)
    assert estimate is not None
    assert estimate > 0


# --------------------------------------------------------------------------
# Description — the text that reaches a prompt or a log
# --------------------------------------------------------------------------


def test_the_description_names_the_problem_and_every_available_method() -> None:
    spec = TFIMSpec(n_sites=4)
    text = survey(spec).describe()
    assert spec.label() in text
    for name in method_names():
        assert name in text


def test_the_description_quotes_the_reason_a_method_declined() -> None:
    result = survey(TFIMSpec(n_sites=6, boundary="open"))
    text = result.describe()
    assert "unavailable:" in text
    assert result.rejected[0].reason in text


def test_the_description_reports_a_memory_footprint_where_one_is_known() -> None:
    text = survey(TFIMSpec(n_sites=8)).describe()
    assert "MB" in text or "KB" in text


def test_the_description_says_so_when_nothing_applies() -> None:
    spec = TFIMSpec(n_sites=4)
    reason = "no method covers this"
    empty = MethodSurvey(
        spec=spec,
        applicable=(),
        rejected=tuple(Rejection(method=info, reason=reason) for info in all_methods()),
    )
    text = empty.describe()
    assert "(none)" in text
    assert reason in text


def test_the_description_is_deterministic() -> None:
    spec = TFIMSpec(n_sites=6, boundary="open")
    assert survey(spec).describe() == survey(spec).describe()


# --------------------------------------------------------------------------
# The registered callables really solve the problem
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
def test_every_applicable_method_agrees_with_every_other(n_sites: int) -> None:
    # The headline claim, routed through the registry rather than by importing
    # the two solvers by hand: whatever the agent picks, it gets the same
    # number to machine precision.
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.7)
    energies = [info.ground_state_energy(spec) for info in survey(spec).applicable]
    assert len(energies) == 2
    assert energies[0] == pytest.approx(energies[1], abs=1e-10)


def test_calling_a_method_the_survey_rejected_raises_with_its_name() -> None:
    spec = TFIMSpec(n_sites=5)
    rejected = survey(spec).rejected[0]
    with pytest.raises(ValueError, match=f"{rejected.method.name} cannot solve"):
        rejected.method.ground_state_energy(spec)
