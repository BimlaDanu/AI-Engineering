"""Tests for method-selection policy.

Two kinds of case. The first goes through the real registry and asserts what the
agent will actually do with a real chain. The second builds a survey by hand,
which is the only way to test policy for accuracy classes no solver in this
project implements yet -- and it is why :func:`select_from` takes a survey
rather than fetching one itself.

Nothing here diagonalises anything: planning is a decision about what *would*
run, so even the L=13 case costs nothing.
"""

from __future__ import annotations

from src.agent.selection import (
    APPROVAL_SITES,
    Approval,
    Plan,
    caveat_for,
    rank,
    select,
    select_from,
)
from src.physics.model import TFIMSpec
from src.physics.registry import (
    AccuracyClass,
    CostClass,
    MethodInfo,
    MethodSurvey,
    Rejection,
    get_method,
)


def fake(
    name: str,
    cost: CostClass = "linear",
    accuracy: AccuracyClass = "exact",
) -> MethodInfo:
    """Build a method that exists only for this test."""
    return MethodInfo(
        name=name,
        summary=f"{name} summary",
        when_to_use=f"use {name}",
        cost=cost,
        accuracy=accuracy,
        unsupported_reason=lambda spec: None,
        ground_state_energy=lambda spec: 0.0,
    )


def survey_of(
    *methods: MethodInfo, n_sites: int = 4, rejected: tuple[Rejection, ...] = ()
) -> MethodSurvey:
    """Build a survey without consulting the registry."""
    return MethodSurvey(spec=TFIMSpec(n_sites=n_sites), applicable=methods, rejected=rejected)


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def test_accuracy_outranks_cost() -> None:
    # The project's claim is verification: a cheap approximation must never
    # displace an exact method, however slow the exact one is.
    cheap_guess = fake("guess", cost="constant", accuracy="uncontrolled")
    costly_truth = fake("truth", cost="exponential", accuracy="exact")
    assert [info.name for info in rank((cheap_guess, costly_truth))] == ["truth", "guess"]


def test_cost_breaks_a_tie_between_equally_accurate_methods() -> None:
    expensive = fake("big", cost="exponential")
    cheap = fake("small", cost="constant")
    assert [info.name for info in rank((expensive, cheap))] == ["small", "big"]


def test_a_bound_outranks_an_uncontrolled_estimate() -> None:
    # A bound can be falsified by an exact result; a guess cannot.
    bound = fake("bound", accuracy="variational_bound")
    guess = fake("guess", accuracy="uncontrolled")
    assert [info.name for info in rank((guess, bound))] == ["bound", "guess"]


def test_ranking_is_stable_for_indistinguishable_methods() -> None:
    # A total order matters: the same question must plan the same way twice.
    first, second = fake("first"), fake("second")
    assert [info.name for info in rank((first, second))] == ["first", "second"]
    assert rank((first, second)) == rank(rank((first, second)))


# --------------------------------------------------------------------------
# Planning against the real registry
# --------------------------------------------------------------------------


def test_an_even_ring_leads_with_the_closed_form_and_checks_it() -> None:
    plan = select(TFIMSpec(n_sites=8))
    assert plan.chosen is not None
    assert plan.chosen.name == "pfeuty_exact"
    assert [info.name for info in plan.corroborators] == ["exact_diagonalisation"]
    assert plan.will_be_corroborated


def test_an_open_chain_falls_back_to_diagonalisation_alone() -> None:
    plan = select(TFIMSpec(n_sites=6, boundary="open"))
    assert plan.chosen is not None
    assert plan.chosen.name == "exact_diagonalisation"
    assert not plan.will_be_corroborated
    assert "periodic" in plan.justify()


def test_an_oversized_chain_produces_no_plan_but_still_explains_itself() -> None:
    # No solver runs here: 2**13 is never allocated, because planning only
    # decides what would run.
    plan = select(TFIMSpec(n_sites=13))
    assert not plan.is_runnable
    assert plan.methods == ()
    justification = plan.justify()
    assert "no method applies" in justification
    assert "exact_diagonalisation" in justification


def test_the_chosen_method_always_comes_first_in_the_run_list() -> None:
    plan = select(TFIMSpec(n_sites=8))
    assert plan.chosen is not None
    assert plan.methods[0] is plan.chosen


# --------------------------------------------------------------------------
# Cost approval
# --------------------------------------------------------------------------


def test_a_small_chain_runs_without_asking() -> None:
    plan = select(TFIMSpec(n_sites=6, boundary="open"))
    assert not plan.needs_approval


def test_a_large_chain_asks_before_diagonalising() -> None:
    plan = select(TFIMSpec(n_sites=APPROVAL_SITES, boundary="open"))
    assert plan.needs_approval
    assert plan.approval is not None
    assert plan.approval.method.name == "exact_diagonalisation"


def test_the_question_quantifies_the_cost_it_is_asking_about() -> None:
    # "This is expensive, continue?" is not a question anyone can answer.
    plan = select(TFIMSpec(n_sites=12, boundary="open"))
    assert plan.approval is not None
    assert "4096-dimensional" in plan.approval.reason
    assert "MB" in plan.approval.reason
    assert plan.approval.estimated_memory_bytes is not None


def test_approval_is_still_required_when_only_the_check_is_expensive() -> None:
    # The cheap closed form leads at L=12, but the corroborating run is the
    # expensive one, and it is a run the user is about to pay for.
    plan = select(TFIMSpec(n_sites=12))
    assert plan.chosen is not None
    assert plan.chosen.cost == "linear"
    assert plan.needs_approval


def test_a_cheap_method_is_never_gated_however_long_the_chain() -> None:
    plan = select_from(survey_of(fake("cheap", cost="linear"), n_sites=12))
    assert not plan.needs_approval


# --------------------------------------------------------------------------
# Caveats
# --------------------------------------------------------------------------


def test_an_exact_method_needs_no_caveat() -> None:
    assert caveat_for(get_method("pfeuty_exact")) is None
    assert select(TFIMSpec(n_sites=8)).caveat is None


def test_a_bound_is_reported_as_a_bound() -> None:
    plan = select_from(survey_of(fake("dmrg", accuracy="variational_bound")))
    assert plan.caveat is not None
    assert "at or below" in plan.caveat
    assert "caveat:" in plan.justify()


def test_an_uncontrolled_method_must_not_be_quoted_bare() -> None:
    plan = select_from(survey_of(fake("mean_field", accuracy="uncontrolled")))
    assert plan.caveat is not None
    assert "no rigorous error bound" in plan.caveat


# --------------------------------------------------------------------------
# The justification
# --------------------------------------------------------------------------


def test_the_justification_names_the_method_and_the_check() -> None:
    text = select(TFIMSpec(n_sites=8)).justify()
    assert "run pfeuty_exact" in text
    assert "check against exact_diagonalisation" in text
    assert "shares no algebra" in text


def test_an_unchecked_answer_says_so_in_the_justification() -> None:
    text = select(TFIMSpec(n_sites=7)).justify()
    assert "no independent check available" in text
    assert "unverified" in text


def test_the_justification_repeats_every_rejection_with_its_reason() -> None:
    text = select(TFIMSpec(n_sites=6, boundary="open")).justify()
    assert "pfeuty_exact unavailable" in text


def test_a_plan_cannot_be_retargeted_after_it_has_been_justified() -> None:
    plan = select(TFIMSpec(n_sites=8))
    try:
        plan.chosen = None  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("Plan must be frozen")


def test_planning_the_same_problem_twice_gives_the_same_plan() -> None:
    first, second = select(TFIMSpec(n_sites=8)), select(TFIMSpec(n_sites=8))
    assert first.justify() == second.justify()
    assert isinstance(first, Plan)
    assert isinstance(first.approval, Approval | None)
