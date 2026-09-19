"""Offline tests for the per-session token-bucket rate limiter.

The bucket takes an explicit ``now`` on every call, so time is injected and the behaviour is
fully deterministic — no sleeping, no wall clock. These pin the burst allowance, refilling,
the capacity cap, and the read-only ``retry_after`` used to tell the user how long to wait.
"""

from __future__ import annotations

import pytest

from src.ratelimit import TokenBucket


def test_starts_full_and_serves_the_burst() -> None:
    bucket = TokenBucket(capacity=3, refill_per_sec=1.0)
    assert all(bucket.try_consume(now=0.0) for _ in range(3))
    assert not bucket.try_consume(now=0.0)  # 4th at the same instant is throttled


def test_refills_at_the_configured_rate() -> None:
    bucket = TokenBucket(capacity=2, refill_per_sec=1.0)
    assert bucket.try_consume(now=0.0)
    assert bucket.try_consume(now=0.0)
    assert not bucket.try_consume(now=0.0)
    assert bucket.try_consume(now=1.0)  # one token regenerated after 1s
    assert not bucket.try_consume(now=1.0)


def test_refill_is_capped_at_capacity() -> None:
    bucket = TokenBucket(capacity=2, refill_per_sec=100.0)
    assert bucket.try_consume(now=0.0)
    # A long idle period cannot bank more than `capacity` tokens.
    assert bucket.try_consume(now=1000.0)
    assert bucket.try_consume(now=1000.0)
    assert not bucket.try_consume(now=1000.0)


def test_retry_after_is_read_only_and_accurate() -> None:
    bucket = TokenBucket(capacity=1, refill_per_sec=0.5)  # one token every 2s
    assert bucket.try_consume(now=0.0)
    assert not bucket.try_consume(now=0.0)
    assert bucket.retry_after(now=0.0) == pytest.approx(2.0)
    # Calling retry_after must not consume or advance state.
    assert bucket.retry_after(now=0.0) == pytest.approx(2.0)
    assert bucket.try_consume(now=2.0)


def test_retry_after_zero_when_a_token_is_available() -> None:
    bucket = TokenBucket(capacity=1, refill_per_sec=1.0)
    assert bucket.retry_after(now=0.0) == 0.0


def test_non_refilling_bucket_waits_forever_once_empty() -> None:
    bucket = TokenBucket(capacity=1, refill_per_sec=0.0)
    assert bucket.try_consume(now=0.0)
    assert bucket.retry_after(now=5.0) == float("inf")


def test_per_minute_factory_sets_the_rate() -> None:
    bucket = TokenBucket.per_minute(burst=6, per_min=60)
    assert bucket.capacity == 6
    assert bucket.refill_per_sec == pytest.approx(1.0)
