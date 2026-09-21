"""Tests for token accounting.

The thing worth pinning here is not arithmetic. It is that an unpriced model
produces *no* cost rather than a zero one, and that a bookkeeping failure cannot
take an answer down with it.
"""

from __future__ import annotations

import pytest

from src.agent.usage import Call, Meter, Price, Usage, metered, price_for
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
