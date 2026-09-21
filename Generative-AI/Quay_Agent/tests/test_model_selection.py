"""Which model serves which call, and what a pool records about it.

Every pool here is offline or driven by a stub. The selection logic is what is
under test, and a suite that built a real client would be testing a credential.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, SecretStr

from src.agent.middleware import CallBudget
from src.agent.model_selection import (
    DEFAULT_TIER_SLUGS,
    FALLBACK_TIERS,
    PINNED_SLUG,
    TASK_TIERS,
    ModelPool,
    Task,
    Tier,
    selectable_slugs,
    slug_for,
    tier_for,
)
from src.model_catalogue import card_for, models_for
from src.settings import Settings

TASKS: list[Task] = list(TASK_TIERS)
TIERS: list[Tier] = ["fast", "standard", "strong"]


class Answer(BaseModel):
    """A one-field schema, so a call has something to validate against."""

    value: str = "ok"


class StubModel:
    """Stands in for a chat model without being one.

    The pool only ever hands a model to the call at the bottom of the stack, so
    a stub that is never invoked is enough to exercise selection, substitution
    and the ledger.
    """

    def __init__(self, name: str = "stub") -> None:
        """Record the name a trace would show.

        Args:
            name: What this model calls itself.
        """
        self.model_name = name


def settings_with(**overrides: Any) -> Settings:
    """Build settings without needing a credential in the environment.

    Args:
        **overrides: Fields to set.

    Returns:
        The settings.
    """
    return Settings(openrouter_api_key=SecretStr("not-a-real-key"), **overrides)


# --------------------------------------------------------------------------
# The assignment
# --------------------------------------------------------------------------


def test_every_task_is_assigned_a_tier() -> None:
    for task in TASKS:
        assert tier_for(task) in TIERS


def test_the_calls_a_person_waits_on_are_all_on_the_fast_tier() -> None:
    # The whole latency argument, as an assertion. Retrieval added three model
    # calls between a question and its answer, and reading the intent added a
    # fourth in front of all of them; every one is on the cheap tier, and a change
    # that moves one of them fails here.
    for task in ("intent_reading", "shelf_choice", "passage_grading", "query_rewrite"):
        assert tier_for(task) == "fast"


def test_the_strong_tier_is_exactly_the_calls_whose_output_a_person_reads() -> None:
    # The rule, not the list: a call earns the strong tier when its own words are
    # the deliverable rather than an input to something checked later. The report,
    # the explanation and the drafted code are read as written; every other call
    # feeds a decision that arithmetic or a test can catch out afterwards.
    #
    # Written as a set comparison rather than "the report is the only one" -- which
    # is what this asserted while the report was the only thing the agent could
    # produce. Adding a branch that writes prose is exactly the change that should
    # have to come back here and say so.
    strong = {task for task, tier in TASK_TIERS.items() if tier == "strong"}
    assert strong == {"report_writing", "explanation", "code_drafting"}


def test_every_tier_has_somewhere_to_fall_back_to() -> None:
    for tier in TIERS:
        assert FALLBACK_TIERS[tier]
        assert tier not in FALLBACK_TIERS[tier]


def test_every_default_slug_is_one_the_subscription_confirms() -> None:
    # A default that fails at the first call is worse than a default that is a
    # year old. Newer models are chosen deliberately, in configuration.
    for tier in TIERS:
        card = card_for(DEFAULT_TIER_SLUGS[tier])
        assert card is not None, f"the {tier} default is not in the catalogue"
        assert card.confirmed, f"the {tier} default has an unverified slug"
        assert card.role == "chat"


def test_the_tier_defaults_climb_rather_than_wander() -> None:
    # A tier is a claim about how much model a call is worth, so a ladder whose
    # rungs are out of order makes every one of those claims wrong -- and it is
    # the kind of wrong nothing else notices, because each slug on its own is
    # perfectly valid. DEFAULT_TIER_SLUGS' docstring claims strength and price
    # order agree; price is the half a test can check, so it is checked here.
    from src.agent.usage import price_for

    rungs = [price_for(DEFAULT_TIER_SLUGS[tier]) for tier in TIERS]
    assert all(rung is not None for rung in rungs), "an unpriced default cannot be ordered"
    rates = [rung.input_usd for rung in rungs if rung is not None]
    assert rates == sorted(rates), f"the tier defaults are not in ascending order: {rates}"


def test_the_standard_tier_default_is_the_settings_default() -> None:
    # Two names for one tier's default is two things to forget to change
    # together, and they had already drifted: this table said `gpt-4.1-mini`
    # while every real run used the settings value. The dead one was also the
    # only one the confirmed-slug test above was looking at, so it certified a
    # slug nothing called.
    from src.settings import DEFAULT_CHAT_MODEL

    assert DEFAULT_TIER_SLUGS["standard"] == DEFAULT_CHAT_MODEL


# --------------------------------------------------------------------------
# Resolving a slug
# --------------------------------------------------------------------------


def test_configuration_wins_over_the_tier_default() -> None:
    chosen = settings_with(fast_model="anthropic/claude-3.5-haiku")
    assert slug_for("fast", chosen) == "anthropic/claude-3.5-haiku"


def test_an_unset_tier_falls_back_to_its_default() -> None:
    assert slug_for("fast", settings_with(fast_model="")) == DEFAULT_TIER_SLUGS["fast"]


def test_the_standard_tier_reads_the_project_wide_model_setting() -> None:
    # One model setting already existed and everything referred to it. Giving
    # the standard tier a second name for the same thing would be two places to
    # change and one of them would be missed.
    chosen = settings_with(chat_model="openai/gpt-4.1-mini")
    assert slug_for("standard", chosen) == "openai/gpt-4.1-mini"


def test_an_uncatalogued_slug_is_accepted_rather_than_refused() -> None:
    # The subscription can gain a model before this project hears about it. A
    # catalogue that could halt the application is worse than the problem it
    # guards against.
    assert slug_for("fast", settings_with(fast_model="someone/brand-new")) == "someone/brand-new"


def test_a_pool_override_beats_configuration() -> None:
    pool = ModelPool(
        settings=settings_with(fast_model="a/configured"),
        overrides={"fast": "b/chosen-by-a-person"},
    )
    assert pool.slug("fast") == "b/chosen-by-a-person"


# --------------------------------------------------------------------------
# What a person may choose between
# --------------------------------------------------------------------------


def test_the_selectable_list_offers_only_chat_models() -> None:
    chat = {card.slug for card in models_for("chat")}
    assert set(selectable_slugs()) == chat


def test_confirmed_models_are_offered_before_unconfirmed_ones() -> None:
    # An unconfirmed slug is a reconstruction that may not resolve. Offering it
    # first invites someone to pick the broken option.
    offered = selectable_slugs()
    confirmed = [slug for slug in offered if (card_for(slug) or models_for("chat")[0]).confirmed]
    assert list(offered[: len(confirmed)]) == confirmed


def test_the_selectable_list_has_no_duplicates() -> None:
    assert len(set(selectable_slugs())) == len(selectable_slugs())


# --------------------------------------------------------------------------
# The pool
# --------------------------------------------------------------------------


def test_an_offline_pool_serves_nothing_and_calls_nothing() -> None:
    pool = ModelPool(offline=True)
    for task in TASKS:
        assert pool.for_task(task) is None
    assert pool.invoke("shelf_choice", Answer, "system", "text") is None
    assert pool.selections() == ()


def test_a_pinned_model_serves_every_task() -> None:
    stub = StubModel()
    pool = ModelPool(pinned=stub)  # type: ignore[arg-type]
    for task in TASKS:
        assert pool.for_task(task) is stub
    assert {choice.slug for choice in pool.selections()} == {PINNED_SLUG}


def test_a_pinned_model_still_records_the_tier_the_task_belongs_to() -> None:
    # Which tier a task *is* does not change because a caller supplied one
    # model. A trace that lost that would make a single-model run unreadable
    # against a tiered one.
    pool = ModelPool(pinned=StubModel())  # type: ignore[arg-type]
    pool.for_task("report_writing")
    assert pool.selections()[-1].tier == "strong"


def test_a_pool_takes_its_ceiling_from_configuration() -> None:
    pool = ModelPool(settings=settings_with(max_model_calls_per_run=7))
    assert pool.budget is not None
    assert pool.budget.limit == 7


def test_an_explicit_ceiling_is_left_alone() -> None:
    pool = ModelPool(settings=settings_with(), budget=CallBudget(limit=3))
    assert pool.budget is not None
    assert pool.budget.limit == 3


def test_two_pools_do_not_share_a_ceiling() -> None:
    # One pool per campaign, so a long-running process cannot accumulate a
    # ceiling across unrelated runs.
    first = ModelPool(settings=settings_with(max_model_calls_per_run=2))
    second = ModelPool(settings=settings_with(max_model_calls_per_run=2))
    assert first.budget is not None
    first.budget.spent = 2
    assert second.budget is not None
    assert second.budget.remaining == 2


def test_a_pool_describes_what_it_bound_and_what_it_spent() -> None:
    pool = ModelPool(offline=True, settings=settings_with())
    described = pool.describe()
    tiers = described["tiers"]
    assert isinstance(tiers, dict)
    assert set(tiers) == set(TIERS)
    assert described["offline"] is True
    assert described["calls"] == 0


def test_the_description_carries_the_latency_record() -> None:
    # The ledger's summary is folded in, so one call answers "which models, how
    # many calls, how long" -- which is what a status panel needs and the first
    # thing anyone asks.
    described = ModelPool(offline=True, settings=settings_with()).describe()
    for key in ("seconds", "slowest_task", "models"):
        assert key in described


@pytest.mark.parametrize("task", TASKS)
def test_no_task_can_reach_a_model_from_an_offline_pool(task: Task) -> None:
    assert ModelPool(offline=True).for_task(task) is None


def test_the_call_ceiling_clears_the_honest_worst_case() -> None:
    """A run that does everything right must never hit the ceiling.

    The ceiling counts attempts including retries, which is correct -- a retry
    costs money exactly like a first attempt. The consequence is that it has to
    clear the *honest* call count with room for retries on top, or a flaky gateway
    exhausts it partway and the answer degrades with nothing on screen saying why.
    That is the failure this pins: the condition that causes retries is the one
    condition under which the budget must not run out.

    Counted from the declared tasks rather than from a number written here, so a
    new call site fails this test instead of quietly eating the headroom.
    """
    from src.agent.model_selection import TASK_TIERS
    from src.settings import Settings

    # Every declared task, less the three that cannot fire on one branch: code
    # drafting and explanation are alternative branches, and report writing has no
    # call site (the report is composed from templates).
    honest = len(TASK_TIERS) - 3
    # Tool consultation is the one task that runs more than once per campaign.
    honest += 1

    ceiling = Settings(openrouter_api_key=SecretStr("test-key")).max_model_calls_per_run
    assert ceiling >= honest, (
        f"a prose answer can need {honest} attempts before any retry, and the "
        f"ceiling is {ceiling}: an honest run can be truncated by the budget"
    )
    assert ceiling >= 2 * honest, (
        f"the ceiling ({ceiling}) leaves no room to retry each of the {honest} "
        f"honest calls once, and a rate-limited gateway is exactly when retries "
        f"happen and when an answer must still be reachable"
    )
