"""Tests for token estimation, cost accounting, and transient-failure retry."""

import pytest

from src.utils import StageTimer, call_with_retry, estimate_cost, estimate_tokens


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world " * 100)
    assert 0 < short < long


def test_estimate_cost_known_model():
    # 1M input tokens at $0.15/M
    assert estimate_cost("openai/gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)


def test_estimate_cost_unknown_model_is_zero():
    assert estimate_cost("no/such-model", 1000, 1000) == 0.0


def test_stage_timer_records_stages():
    timer = StageTimer()
    with timer.measure("stage_a"):
        pass
    assert "stage_a" in timer.stages
    assert timer.stages["stage_a"] >= 0.0


# --- call_with_retry ---------------------------------------------------------------------
# ``sleep`` is injected as a no-op recorder so these run instantly and never touch the clock.


def _no_sleep_recorder():
    delays: list[float] = []
    return delays, delays.append


def test_call_with_retry_returns_on_first_success():
    delays, sleep = _no_sleep_recorder()
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert call_with_retry(fn, sleep=sleep) == "ok"
    assert len(calls) == 1  # no retry
    assert delays == []  # never slept


def test_call_with_retry_retries_transient_then_succeeds():
    delays, sleep = _no_sleep_recorder()
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("HTTP 429 rate limit exceeded")
        return "recovered"

    assert call_with_retry(flaky, sleep=sleep) == "recovered"
    assert attempts["n"] == 3
    assert len(delays) == 2  # slept before each of the two retries


def test_call_with_retry_reraises_hard_error_immediately():
    delays, sleep = _no_sleep_recorder()

    def bad_key():
        raise ValueError("401 invalid api key")

    with pytest.raises(ValueError, match="invalid api key"):
        call_with_retry(bad_key, sleep=sleep)
    assert delays == []  # hard error is not retried


def test_call_with_retry_gives_up_after_max_attempts():
    delays, sleep = _no_sleep_recorder()
    attempts = {"n": 0}

    def always_429():
        attempts["n"] += 1
        raise RuntimeError("503 service overloaded")

    with pytest.raises(RuntimeError, match="overloaded"):
        call_with_retry(always_429, max_attempts=3, sleep=sleep)
    assert attempts["n"] == 3  # first try + 2 retries
    assert len(delays) == 2  # slept before each retry, not after the last failure


def test_call_with_retry_honours_custom_retry_on_predicate():
    # A non-transient error (a truncated-JSON parse failure) is retried only because the
    # caller opts in via retry_on — mirrors how the LLM judge re-rolls a malformed response.
    delays, sleep = _no_sleep_recorder()
    attempts = {"n": 0}

    def flaky_parse():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("Invalid JSON: EOF while parsing a string")
        return "parsed"

    result = call_with_retry(
        flaky_parse, sleep=sleep, retry_on=lambda e: "invalid json" in str(e).lower()
    )
    assert result == "parsed"
    assert attempts["n"] == 2
    assert len(delays) == 1


def test_call_with_retry_without_predicate_does_not_retry_parse_error():
    # Without retry_on, a non-transient parse error is a hard failure — reraised at once.
    delays, sleep = _no_sleep_recorder()

    def parse_error():
        raise ValueError("Invalid JSON: EOF while parsing a string")

    with pytest.raises(ValueError, match="Invalid JSON"):
        call_with_retry(parse_error, sleep=sleep)
    assert delays == []


def test_call_with_retry_backoff_is_bounded_by_max_delay():
    delays, sleep = _no_sleep_recorder()

    def always_timeout():
        raise TimeoutError("request timed out")

    with pytest.raises(TimeoutError):
        call_with_retry(always_timeout, max_attempts=6, base_delay=1.0, max_delay=4.0, sleep=sleep)
    # Full jitter: each wait is in [0, ceiling] where ceiling is capped at max_delay.
    assert all(0.0 <= d <= 4.0 for d in delays)
