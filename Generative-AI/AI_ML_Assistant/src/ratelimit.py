"""A simple rate limiter based on the token bucket algorithm.

It limits how quickly a single Streamlit session can send requests. This helps
prevent excessive API usage, unexpected request loops, and high API costs.

The rate limiter is independent of Streamlit and other frameworks, making it
easy to test. It uses a provided time value instead of managing its own clock,
so its behaviour is predictable during testing.

The UI layer (`src.ui.pages.chat`) creates one token bucket per user session
and stores it in `st.session_state`. Before sending a request to the AI
pipeline, it checks the bucket to confirm that the request is allowed.
This module does not import Streamlit and only provides the rate-limiting logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def _monotonic() -> float:
    """Default clock — monotonic so it is immune to wall-clock adjustments."""
    return time.monotonic()


@dataclass
class TokenBucket:
    """A classic token bucket: ``capacity`` tokens that refill at ``refill_per_sec``.

    Each request consumes tokens; a request is allowed only if enough tokens are available.
    The bucket starts full so the first ``capacity`` requests are served without delay (the
    burst allowance), after which requests are paced by the refill rate.

    Args:
        capacity: Maximum tokens the bucket can hold — the instantaneous burst size.
        refill_per_sec: Tokens regenerated per second once the burst is spent.
        tokens: Current token count. Defaults to ``capacity`` (a full bucket) via a negative
            sentinel, so callers never need to pass it.
        last: Timestamp (same clock as ``now``) of the last refill; ``0.0`` until first use.
    """

    capacity: int
    refill_per_sec: float
    tokens: float = -1.0  # sentinel: < 0 means "start full" (resolved in __post_init__)
    last: float = field(default=0.0)

    def __post_init__(self) -> None:
        """Resolve the full-bucket sentinel into an explicit token count."""
        if self.tokens < 0:
            self.tokens = float(self.capacity)

    @classmethod
    def per_minute(cls, burst: int, per_min: float) -> TokenBucket:
        """Build a bucket with a ``burst`` allowance refilling at ``per_min`` per minute."""
        return cls(capacity=max(1, burst), refill_per_sec=max(0.0, per_min) / 60.0)

    def _available(self, now: float) -> float:
        """Tokens that would be present at ``now`` (pure — does not mutate the bucket)."""
        elapsed = max(0.0, now - self.last)
        return min(float(self.capacity), self.tokens + elapsed * self.refill_per_sec)

    def try_consume(self, amount: float = 1.0, now: float | None = None) -> bool:
        """Consume ``amount`` tokens if available; return whether the request is allowed."""
        now = _monotonic() if now is None else now
        self.tokens = self._available(now)
        self.last = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def retry_after(self, amount: float = 1.0, now: float | None = None) -> float:
        """Seconds until ``amount`` tokens are available (``0.0`` if allowed right now).

        Read-only: safe to call after a failed :meth:`try_consume` to tell the user how long
        to wait. Returns ``inf`` if the bucket never refills (``refill_per_sec <= 0``).
        """
        now = _monotonic() if now is None else now
        available = self._available(now)
        if available >= amount:
            return 0.0
        if self.refill_per_sec <= 0:
            return float("inf")
        return (amount - available) / self.refill_per_sec
