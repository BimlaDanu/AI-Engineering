"""Tests for token accounting.

The thing worth pinning here is not arithmetic. It is that an unpriced model
produces *no* cost rather than a zero one, and that a bookkeeping failure cannot
take an answer down with it.
"""

from __future__ import annotations

import pytest

from src.agent.usage import (
    HEURISTIC_PRICES,
    PRICES,
    Call,
    Meter,
    Price,
    Usage,
    estimated_price_for,
    metered,
    price_basis,
    price_for,
)
from src.logging_setup import CallLog, log_llm_call


def record(
    model: str = "openai/gpt-4o-mini",
    purpose: str = "answer",
    prompt: int | None = 1000,
    completion: int | None = 500,
) -> CallLog:
    """One finished call record, as `log_llm_call` would have left it."""
    return CallLog(
        model=model,
        purpose=purpose,
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_ms=12.5,
        outcome="ok",
    )


# --- prices ----------------------------------------------------------------


def test_a_price_charges_more_for_output_than_for_input() -> None:
    # True of every provider, and the reason a verbose answer costs more than a
    # verbose prompt.
    for price in (price_for(name) for name in ("openai/gpt-4o-mini", "anthropic/claude-haiku-4.5")):
        assert price is not None
        assert price.output_usd > price.input_usd


def test_cost_is_per_million_tokens() -> None:
    assert Price(2.0, 10.0).cost(500_000, 100_000) == pytest.approx(2.0)


def test_every_model_the_project_ships_as_a_default_is_priced() -> None:
    # An unpriced model is the honest answer for a slug somebody typed in. It is not
    # the honest answer for one this project chose: the session panel then reads
    # "cost unknown (unpriced model)" on a perfectly ordinary run, which a reader
    # cannot tell apart from "no model was called" -- the one distinction that panel
    # exists to make. The strong tier's default was missing, so every explanation and
    # every code draft landed there.
    from src.agent.model_selection import DEFAULT_TIER_SLUGS

    unpriced = [slug for slug in DEFAULT_TIER_SLUGS.values() if price_for(slug) is None]
    assert not unpriced


def test_an_unknown_model_has_no_price_rather_than_a_zero_one() -> None:
    assert price_for("some/model-nobody-priced") is None


# --- what a run spent ------------------------------------------------------


def test_an_empty_usage_says_no_model_was_called() -> None:
    # The honest description of a refusal composed from code, which is a real
    # outcome and must not render as a blank.
    assert Usage().summary() == "No model was called."
    assert Usage().calls_made == 0
    assert Usage().cost_usd == 0.0


def test_usage_totals_tokens_across_calls() -> None:
    usage = Usage((Call.of(record()), Call.of(record(prompt=200, completion=50))))
    assert usage.prompt_tokens == 1200
    assert usage.completion_tokens == 550
    assert usage.total_tokens == 1750


def test_an_unpriced_model_leaves_the_run_not_fully_priced() -> None:
    usage = Usage((Call.of(record()), Call.of(record(model="mine/custom-slug"))))
    assert not usage.fully_priced
    # The priced half still counts, so the total is a floor rather than nothing.
    assert usage.cost_usd > 0.0
    assert "some models unpriced" in usage.summary()


def test_a_run_with_only_unpriced_models_says_the_cost_is_unknown() -> None:
    usage = Usage((Call.of(record(model="mine/custom-slug")),))
    assert usage.cost_usd == 0.0
    assert "cost unknown" in usage.summary()


def test_a_provider_that_reports_no_usage_is_distinguishable_from_a_free_call() -> None:
    # Zero tokens is not a thing a model call does. A gateway that forwards no
    # usage metadata is, and the two must not look alike.
    silent = Call.of(record(prompt=None, completion=None))
    assert silent.total_tokens == 0
    assert not silent.reported
    assert not Usage((silent,)).fully_reported


def test_a_negative_token_count_from_a_provider_is_clamped_rather_than_subtracted() -> None:
    # A negative count is not something a call can do, so it is a provider defect.
    # Carried through, it would take tokens *off* the run's total and show a bill
    # falling as the work rises, which reads as a bug in the meter rather than in
    # the gateway.
    call = Call.of(record(prompt=-100, completion=-50))
    assert call.prompt_tokens == 0
    assert call.completion_tokens == 0
    assert call.cost_usd == 0.0
    # Still reported: the gateway said something, it was just nonsense. That is a
    # different fact from having said nothing at all.
    assert call.reported


def test_by_purpose_puts_the_heaviest_step_first() -> None:
    usage = Usage(
        (
            Call.of(record(purpose="route", prompt=100, completion=10)),
            Call.of(record(purpose="compose", prompt=4000, completion=800)),
        )
    )
    assert [step.purpose for step in usage.by_purpose()] == ["compose", "route"]
    assert usage.by_purpose()[0].calls == 1


def test_merging_accumulates_a_conversation() -> None:
    one = Usage((Call.of(record()),))
    assert one.merge(one).calls_made == 2
    assert one.merge(Usage()).total_tokens == one.total_tokens


def test_a_failed_call_is_still_counted() -> None:
    # The provider charges for the tokens it read before it failed, and a retried
    # call is two calls. Accounting that hid them would understate every bad run.
    failed = CallLog(model="openai/gpt-4o-mini", purpose="compose", prompt_tokens=300)
    failed.outcome = "error"
    usage = Usage((Call.of(failed),))
    assert usage.calls_made == 1
    assert usage.calls[0].outcome == "error"


# --- the meter -------------------------------------------------------------


def test_the_meter_sees_calls_made_inside_the_block() -> None:
    with metered() as meter:
        with log_llm_call("openai/gpt-4o-mini", purpose="route") as call:
            call.prompt_tokens, call.completion_tokens = 90, 10
    assert meter.usage.calls_made == 1
    assert meter.usage.total_tokens == 100


def test_the_meter_sees_a_call_that_raised() -> None:
    with metered() as meter, pytest.raises(RuntimeError):
        with log_llm_call("openai/gpt-4o-mini", purpose="route") as call:
            call.prompt_tokens = 50
            raise RuntimeError("provider fell over")
    assert meter.usage.calls_made == 1
    assert meter.usage.calls[0].outcome == "error"


def test_the_meter_ignores_calls_made_outside_the_block() -> None:
    with metered() as meter:
        pass
    with log_llm_call("openai/gpt-4o-mini", purpose="route"):
        pass
    assert meter.usage.calls_made == 0


def test_a_broken_observer_does_not_break_the_call() -> None:
    # The whole point of counting through an observer: an accounting bug shows up
    # as a missing number on screen, never as a traceback where an answer was.
    from src.logging_setup import observing_calls

    def explode(call: CallLog) -> None:
        raise ValueError("bookkeeping is hard")

    with observing_calls(explode):
        with log_llm_call("openai/gpt-4o-mini", purpose="route") as call:
            call.prompt_tokens = 10
    assert call.prompt_tokens == 10


def test_two_meters_nest_without_mixing() -> None:
    with metered() as outer:
        with log_llm_call("openai/gpt-4o-mini", purpose="route"):
            pass
        with metered() as inner:
            with log_llm_call("openai/gpt-4o-mini", purpose="compose"):
                pass
    assert [call.purpose for call in inner.usage.calls] == ["compose"]
    assert [call.purpose for call in outer.usage.calls] == ["route", "compose"]


def test_the_meter_snapshot_is_immutable() -> None:
    meter = Meter()
    meter.record(record())
    snapshot = meter.usage
    meter.record(record())
    assert snapshot.calls_made == 1


# --- reasoned estimates, kept apart from the rate card ---------------------


def test_no_estimate_shadows_a_published_rate() -> None:
    # The separation is the whole design. If a slug appeared in both tables, whichever
    # happened to be consulted first would decide whether a figure was an invoice or a
    # guess -- and nothing on screen would change.
    assert not (set(PRICES) & set(HEURISTIC_PRICES))


def test_the_published_rate_wins_where_there_is_one() -> None:
    assert estimated_price_for("openai/gpt-4o") == price_for("openai/gpt-4o")


def test_an_estimated_model_is_still_not_fully_priced() -> None:
    # The point of the whole exercise: the run gets a number to show, and it does
    # *not* get to call itself priced. Relaxing this would make a guess and a rate
    # card indistinguishable in the one panel that exists to tell them apart.
    #
    # The summary used to print "cost unknown (unpriced model)" here, which is the
    # half of that sentence this module went to the trouble of avoiding: the
    # estimate existed on the same object and no display showed it. It now prints
    # the figure and says what it rests on, and `fully_priced` still goes false.
    meter = Meter()
    meter.record(record(model="z-ai/glm-5.2"))
    usage = meter.usage
    assert not usage.fully_priced
    assert usage.rests_on_estimates
    assert usage.estimated_cost_usd is not None
    assert "unpriced" not in usage.summary()
    assert "estimated rates" in usage.summary()
    assert "est." in usage.cost_label()


def test_a_model_in_neither_table_says_so_rather_than_showing_zero() -> None:
    # The failure the label exists to prevent: a session that cost money reading
    # as free. With nothing to price the call by, the figure is a floor and is
    # marked as one.
    meter = Meter()
    meter.record(record(model="nobody/has-priced-this"))
    usage = meter.usage
    assert not usage.fully_priced
    assert not usage.rests_on_estimates
    assert usage.estimated_cost_usd is None
    assert usage.cost_label().startswith("≥ $")
    assert "floor" in usage.cost_basis()


def test_a_published_run_needs_no_qualification() -> None:
    meter = Meter()
    meter.record(record(model="openai/gpt-4o-mini"))
    usage = meter.usage
    assert usage.fully_priced
    assert usage.cost_basis() == ""
    assert usage.cost_label().startswith("$")


def test_the_estimate_is_the_rate_applied_to_the_tokens_actually_spent() -> None:
    # Checked against arithmetic done here rather than against the property's own
    # route: 1000 prompt tokens at $0.97/M and 500 completion at $3.04/M.
    #
    # The two rates are written out because the point is to compute the answer a
    # second way, and reading them out of the table under test would compute it
    # the same way twice. That does mean a corrected price lands here as a
    # failure -- it did, when `scripts/verify_model_slugs.py` found this model's
    # rate was 0.60/2.20 only in this project's imagination -- so the table is
    # then asserted to agree, which turns "somebody edited a price" into a
    # message that says so instead of an unexplained arithmetic mismatch.
    rate_in, rate_out = 0.97, 3.04
    listed = HEURISTIC_PRICES["z-ai/glm-5.2"]
    assert (listed.input_usd, listed.output_usd) == (rate_in, rate_out), (
        "this model's rate moved; check it with scripts/verify_model_slugs.py "
        "and update the two constants above as well as the table"
    )

    meter = Meter()
    meter.record(record(model="z-ai/glm-5.2", prompt=1000, completion=500))
    expected = (1000 * rate_in + 500 * rate_out) / 1_000_000
    assert meter.usage.estimated_cost_usd == pytest.approx(expected)


def test_a_model_neither_table_lists_has_no_estimate_rather_than_a_partial_one() -> None:
    # A total that silently drops one model is worse than no total: it is a number
    # somebody would act on.
    meter = Meter()
    meter.record(record(model="openai/gpt-4o"))
    meter.record(record(model="nobody/has-heard-of-this"))
    assert meter.usage.estimated_cost_usd is None
    assert not meter.usage.rests_on_estimates


def test_every_model_the_interface_offers_can_be_costed() -> None:
    # Before HEURISTIC_PRICES, fifteen of the twenty-three models selectable in the
    # chat panel reported no cost at all -- so picking one made the session's spend
    # panel go blank, which reads as "nothing was called". A model offered in the
    # interface is a model this project chose, and it owes the reader a figure or a
    # stated reason there is none.
    from src.agent.model_selection import selectable_slugs

    uncosted = [slug for slug in selectable_slugs() if price_basis(slug) == "unknown"]
    assert not uncosted


def test_price_basis_names_the_three_cases_apart() -> None:
    assert price_basis("openai/gpt-4o") == "published"
    assert price_basis("openai/gpt-5.4-nano") == "estimated"
    assert price_basis("nobody/has-heard-of-this") == "unknown"


def test_every_estimate_charges_more_for_output_than_input() -> None:
    # The one structural fact that survives any correction to the individual figures.
    # An estimate that got this backwards would be a transcription error, not a
    # difference of opinion about the price.
    for slug, price in HEURISTIC_PRICES.items():
        assert price.output_usd > price.input_usd, slug
