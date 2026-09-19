"""Utilities for token counting, cost tracking, timing, and retrying failed operations."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from src.config import MODEL_PRICES

_T = TypeVar("_T")

# Substrings that mark a *transient* API failure worth retrying — rate limits, upstream
# overload, gateway hiccups, timeouts. Anything else (bad key, malformed request, quota
# exhausted) is a hard failure we surface immediately rather than hammering. Matched
# case-insensitively against ``str(exc)``.
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "429",
    "rate limit",
    "rate-limit",
    "overloaded",
    "capacity",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "503",
    "502",
    "504",
    "connection reset",
    "connection aborted",
)


def _is_transient(exc: Exception) -> bool:
    """True if ``exc`` looks like a retry-worthy transient API failure."""
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def call_with_retry(
    fn: Callable[[], _T],
    *,
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
    retry_on: Callable[[Exception], bool] | None = None,
) -> _T:
    """Call ``fn`` with exponential backoff + full jitter on transient failures.

    Retries errors that *look* transient (rate limit, overload, timeout — see
    :data:`_TRANSIENT_MARKERS`) plus anything ``retry_on`` opts in to; a hard error (bad key,
    malformed request) is reraised at once, and the final attempt's exception always
    propagates so the caller still surfaces a real failure to the user. The per-attempt wait
    is ``uniform(0, min(max_delay, base_delay * 2**n))`` — full jitter, which spreads a burst
    of concurrent callers out instead of retrying them in lockstep.

    Args:
        fn: A zero-argument thunk performing the (network) call.
        max_attempts: Total attempts including the first (``>= 1``).
        base_delay: Base backoff in seconds for the first retry.
        max_delay: Upper bound on the (pre-jitter) backoff.
        sleep: Injectable sleep, so tests run without real delays.
        retry_on: Optional extra predicate; when it returns True for the raised exception,
            that error is retried too. Lets a caller mark domain-specific failures as
            retry-worthy — e.g. the LLM judge treats a truncated / malformed structured-output
            response as retryable, since a fresh completion usually parses cleanly.

    Returns:
        Whatever ``fn`` returns on the first successful attempt.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            retryable = _is_transient(exc) or (retry_on is not None and retry_on(exc))
            if attempt >= max_attempts or not retryable:
                raise
            ceiling = min(max_delay, base_delay * 2 ** (attempt - 1))
            # Jitter to decorrelate retries across sessions; nothing here is a secret.
            sleep(random.uniform(0.0, ceiling))  # noqa: S311
    # Unreachable: the loop either returns or reraises on the final attempt.
    raise AssertionError("call_with_retry exhausted without returning or raising")


def estimate_tokens(text: str) -> int:
    """Rough token estimate without a tokenizer dependency.

    Uses the larger of the ``words * 4/3`` and ``chars / 4`` heuristics, which tracks
    GPT-style BPE within roughly 10% on English prose.
    """
    if not text:
        return 0
    words = len(text.split())
    return max(round(words * 4 / 3), len(text) // 4, 1)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Approximate USD cost of a call, based on the static price table in config."""
    price_in, price_out = MODEL_PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class UsageMeter:
    """Accumulates token usage from the *auxiliary* LLM calls made while serving a request.

    The final-answer generation reports its own exact usage on the :class:`AnswerResult`, but
    the extra main-model calls a request may make on the way — query rewriting/expansion and
    listwise reranking — were previously uncounted, so the reported token total (and hence
    cost) undercounted. Helpers that make those calls report here via :meth:`add`, and the
    service folds this into the request total. Usage is *exact* when the provider returns
    ``usage_metadata`` and *estimated* (via :func:`estimate_tokens`) for structured-output
    calls, which do not surface it — the same honest-estimate approach the agent loop uses.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Add one call's input/output token counts (negatives are clamped to zero)."""
        self.input_tokens += max(0, int(input_tokens))
        self.output_tokens += max(0, int(output_tokens))

    def record_call(self, reply: object, prompt: str, output: str) -> None:
        """Record a call's usage: exact from ``reply.usage_metadata`` if present, else estimated.

        ``reply`` is the raw model reply (an ``AIMessage`` carries ``usage_metadata``; a
        structured-output Pydantic result does not). ``prompt``/``output`` are the request and
        response text used to estimate tokens when the provider does not report them.
        """
        usage = getattr(reply, "usage_metadata", None) or {}
        self.add(
            usage.get("input_tokens") or estimate_tokens(prompt),
            usage.get("output_tokens") or estimate_tokens(output),
        )


@dataclass
class StageTimer:
    """Collects per-stage wall-clock timings for the Developer-tab pipeline trace."""

    stages: dict[str, float] = field(default_factory=dict)

    def measure(self, name: str) -> _Stage:
        """Return a context manager that records the elapsed time under ``name``."""
        return _Stage(self, name)


class _Stage:
    """Context manager recording one stage's wall-clock duration into its StageTimer."""

    def __init__(self, timer: StageTimer, name: str) -> None:
        self._timer = timer
        self._name = name

    def __enter__(self) -> None:
        """Start the stage clock."""
        self._start = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        """Stop the clock and store the elapsed seconds under the stage name."""
        self._timer.stages[self._name] = time.perf_counter() - self._start
