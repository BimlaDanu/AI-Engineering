"""Cross-cutting concerns wrapped around every model call, in one composable stack.

A campaign makes model calls from six places, and each needs the same four
things: a ceiling it cannot spend past, a screen on what goes into the prompt, a
retry on a failure worth retrying, and a record of what it cost. Written at the
call sites those become twenty-four opportunities to forget one.

A :class:`Handler` takes an :class:`Invocation` and produces an :class:`Outcome`;
a :class:`Middleware` takes a handler and returns a handler. :func:`compose`
stacks them so the first listed is outermost and the innermost is the call
itself, which makes :func:`standard_stack` the whole behaviour of every model
call in five lines.

The order is the contract:

1. Ceiling, outermost, so retries count against it. An unbounded retry inside a
   bounded loop is the expensive failure.
2. Screen, before anything is sent -- the last point at which refusing is free.
3. Retry, inside the screen, because a call worth retrying is one already judged
   safe to make and re-screening identical text has a known answer.
4. Record, innermost, so each attempt is timed separately. A retried call is two
   calls, both paid for.

Nothing here reaches the network. The stack is pure composition and the call at
the bottom is supplied by the caller, which is what lets every layer be tested
with no credential.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.agent.llm import is_transient
from src.logging_setup import get_logger
from src.security import screen

_logger = get_logger("agent.middleware")

RETRY_BACKOFF_S = 1.0
"""Delay before the first retry, doubled on each attempt after it.

A second is long enough for a rate limiter to release and short enough that a
person waiting on an answer does not conclude the application has hung. It is not
configurable because the value that matters is the retry *count*, which is, and a
second knob would only make two things to get wrong.
"""


@dataclass(frozen=True, slots=True)
class Invocation:
    """One pending model call, described well enough to decide things about it.

    Attributes:
        task: What the call is for. This is what a budget, a log line and a cost
            report all group by, which is why it travels with the call rather
            than being reconstructed from the stack.
        tier: Which tier the task was served from.
        model: The slug about to be called.
        system: The system instruction.
        text: The user-role content, which is where untrusted material arrives.
        purpose: A short label for the record, defaulting to the task.
    """

    task: str
    tier: str
    model: str
    system: str
    text: str
    purpose: str = ""

    def label(self) -> str:
        """What this call is called in a log line."""
        return self.purpose or self.task


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one call produced, including the ways it can produce nothing.

    A value rather than an exception, because every failure mode here has the
    same consequence for the caller -- take the deterministic path -- and control
    flow that is the same for four causes should not be four ``except`` clauses.

    Attributes:
        value: Whatever the call returned, or ``None`` if it produced nothing.
        refusal: Why nothing came back, empty on success. A sentence rather than
            a code: it is shown to a person when a campaign explains why it fell
            back to its deterministic path.
        elapsed_s: Wall-clock seconds spent, retries included.
        attempts: How many times the call was actually made.
    """

    value: Any = None
    refusal: str = ""
    elapsed_s: float = 0.0
    attempts: int = 0

    @property
    def ok(self) -> bool:
        """Whether a usable value came back."""
        return self.value is not None and not self.refusal


Handler = Callable[[Invocation], Outcome]
"""Executes a call and reports what happened."""

Middleware = Callable[[Handler], Handler]
"""Wraps a handler in one layer of behaviour."""


def compose(middlewares: Sequence[Middleware], handler: Handler) -> Handler:
    """Stack middlewares around a handler, first listed outermost.

    Args:
        middlewares: The layers, outermost first.
        handler: The call at the bottom.

    Returns:
        A handler that runs every layer in order.

    Examples:
        >>> def shout(inner: Handler) -> Handler:
        ...     return lambda call: Outcome(value=str(inner(call).value).upper())
        >>> stacked = compose([shout], lambda call: Outcome(value="quiet"))
        >>> stacked(Invocation("t", "fast", "m", "s", "x")).value
        'QUIET'
    """
    wrapped = handler
    for middleware in reversed(middlewares):
        wrapped = middleware(wrapped)
    return wrapped


@dataclass
class CallBudget:
    """How many model calls one campaign may make, and how many it has made.

    Mutable on purpose: it is a counter, and a counter that returns a new counter
    is a counter every caller has to remember to reassign. One per campaign, so
    that a long-running process cannot accumulate a ceiling across unrelated
    runs.

    Attributes:
        limit: The ceiling. Counted across every task.
        spent: Calls attempted so far, retries included -- a retry costs money
            and latency exactly like a first attempt, and a budget that ignored
            them would be a budget a failing provider could walk straight
            through.
    """

    limit: int
    spent: int = 0
    _turnstile: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def remaining(self) -> int:
        """Calls still available."""
        return max(0, self.limit - self.spent)

    def refusal(self) -> str | None:
        """Why the next call cannot be made, or ``None`` if it can.

        Returns:
            A sentence naming the ceiling, or ``None``.
        """
        if self.remaining > 0:
            return None
        return f"the run's ceiling of {self.limit} model calls is spent"

    def claim(self) -> str | None:
        """Take one call off the ceiling, or say why it cannot be taken.

        Checking the ceiling and spending against it are one indivisible act, which
        matters now that a campaign issues several calls at once -- see
        :mod:`src.agent.prefetch`. Split apart, three threads can each read a
        remaining count of one and each go on to spend it, so a ceiling meant to be a
        hard limit becomes a suggestion under exactly the load it exists to bound.

        Returns:
            The refusal, or ``None`` when the caller now holds one call's worth of
            budget and must go on to use it.
        """
        with self._turnstile:
            refusal = self.refusal()
            if refusal is not None:
                return refusal
            self.spent += 1
            return None


def enforce_ceiling(budget: CallBudget) -> Middleware:
    """Refuse a call once the run's ceiling is reached.

    Claims the *first* attempt only. The retries are claimed by
    :func:`retry_transient`, which is handed the same budget -- see there for why
    the counting is split across two layers rather than done in one.

    Args:
        budget: The campaign's counter.

    Returns:
        The outermost layer.
    """

    def layer(inner: Handler) -> Handler:
        def run(call: Invocation) -> Outcome:
            refusal = budget.claim()
            if refusal is not None:
                _logger.warning(
                    "model_call_refused",
                    extra={"task": call.task, "detail": refusal},
                )
                return Outcome(refusal=refusal)
            return inner(call)

        return run

    return layer


def screen_prompt() -> Middleware:
    """Refuse a call whose prompt carries instruction-override patterns.

    Defence in depth rather than the primary control. A question is screened when
    it arrives and a retrieved passage is screened when it is stored, so text
    reaching here has already passed twice. This layer catches the case neither
    of those can: text *assembled* from clean parts, where the combination is what
    carries the pattern.

    Only the user-role content is screened. The system instruction is written by
    this project and screening it would mean the project's own wording could
    block its own calls.

    Returns:
        The layer.
    """

    def layer(inner: Handler) -> Handler:
        def run(call: Invocation) -> Outcome:
            signals = [
                signal
                for signal in screen(call.text).signals
                # Length is a rule about what a person may type, not about how
                # much retrieved material a prompt may legitimately carry.
                if signal.category != "oversized_input"
            ]
            if signals:
                categories = sorted({signal.category for signal in signals})
                _logger.warning(
                    "model_call_blocked",
                    extra={"task": call.task, "categories": categories},
                )
                return Outcome(
                    refusal=f"the assembled prompt carries {', '.join(categories)} patterns"
                )
            return inner(call)

        return run

    return layer


def retry_transient(
    max_retries: int,
    backoff_s: float = RETRY_BACKOFF_S,
    budget: CallBudget | None = None,
) -> Middleware:
    r"""Repeat a call that failed for a reason that might not recur.

    Only transient failures are retried. A schema the provider rejects and a
    model name that does not exist will fail identically every time, and retrying
    those spends the budget three times to reach the same answer.

    Every retry is claimed against the ceiling. :attr:`CallBudget.spent` is
    documented as attempts *including retries*, because a retry costs money and
    latency exactly like a first attempt -- but the layer that enforces the ceiling
    wraps this one and therefore sees a retried call once. So a ceiling of thirty
    admitted up to a hundred and twenty provider requests at the default of three
    retries: the ceiling failing in the precise case it exists for, a provider
    failing under load.

    The counting is split across two layers rather than moved wholesale into one,
    and that is not a compromise. :func:`enforce_ceiling` has to stay outermost so
    that a call the prompt screen blocks still costs a slot -- otherwise an
    attacker who can get prompts blocked has an unmetered channel into the
    application. So the ceiling claims the first attempt, this claims each further
    one, and the invariant a reader can check is that ``budget.spent`` equals the
    number of entries in the ledger.

    Args:
        max_retries: Additional attempts after the first. Zero disables retrying.
        backoff_s: Delay before the first retry, doubled thereafter.
        budget: The campaign's ceiling, so retries can be charged to it. ``None``
            leaves them uncharged, which suits a test exercising this layer alone
            and nothing else.

    Returns:
        The layer.
    """

    def layer(inner: Handler) -> Handler:
        def run(call: Invocation) -> Outcome:
            attempts = 0
            delay = backoff_s
            last = Outcome(refusal="the call was never attempted")
            while attempts <= max_retries:
                if attempts and budget is not None:
                    exhausted = budget.claim()
                    if exhausted is not None:
                        _logger.warning(
                            "model_retry_refused",
                            extra={"task": call.task, "detail": exhausted},
                        )
                        return Outcome(refusal=exhausted, attempts=attempts)
                attempts += 1
                try:
                    outcome = inner(call)
                except Exception as error:
                    if not is_transient(error) or attempts > max_retries:
                        return Outcome(
                            refusal=f"{type(error).__name__} while calling the model",
                            attempts=attempts,
                        )
                    last = Outcome(refusal=type(error).__name__, attempts=attempts)
                else:
                    if outcome.ok or attempts > max_retries:
                        return Outcome(
                            value=outcome.value,
                            refusal=outcome.refusal,
                            elapsed_s=outcome.elapsed_s,
                            attempts=attempts,
                        )
                    last = outcome
                _logger.info(
                    "model_call_retried",
                    extra={"task": call.task, "attempt": attempts, "delay_s": delay},
                )
                time.sleep(delay)
                delay *= 2
            return Outcome(refusal=last.refusal, attempts=attempts)

        return run

    return layer


@dataclass
class Ledger:
    """Every call a campaign made, in the order it made them.

    Attributes:
        entries: One record per attempt: the task, the model, whether it
            produced a value, and how long it took.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, call: Invocation, outcome: Outcome) -> None:
        """Add one attempt.

        Args:
            call: What was attempted.
            outcome: What came back.
        """
        self.entries.append(
            {
                "task": call.task,
                "tier": call.tier,
                "model": call.model,
                "ok": outcome.ok,
                "elapsed_s": round(outcome.elapsed_s, 3),
            }
        )

    @property
    def total_seconds(self) -> float:
        """Wall-clock time spent in model calls."""
        return sum(float(entry["elapsed_s"]) for entry in self.entries)

    def slowest(self) -> dict[str, Any] | None:
        """The single slowest call, or ``None`` if there were none.

        Returns:
            The record. This is the first thing to look at when a campaign felt
            slow, and having it as a value means nobody has to sort a log.
        """
        return max(self.entries, key=lambda entry: entry["elapsed_s"], default=None)

    def summary(self) -> dict[str, Any]:
        """Totals for a trace or a status panel.

        Returns:
            Call count, failures, total seconds, and the slowest task.
        """
        slowest = self.slowest()
        return {
            "calls": len(self.entries),
            "failed": sum(1 for entry in self.entries if not entry["ok"]),
            "seconds": round(self.total_seconds, 2),
            "slowest_task": slowest["task"] if slowest else "",
            "models": sorted({str(entry["model"]) for entry in self.entries}),
        }


def record_calls(ledger: Ledger) -> Middleware:
    """Time each attempt and write it down.

    Innermost layer, so that a retried call appears as the several calls it
    actually was.

    Args:
        ledger: Where to record.

    Returns:
        The layer.
    """

    def layer(inner: Handler) -> Handler:
        def run(call: Invocation) -> Outcome:
            started = time.monotonic()
            try:
                outcome = inner(call)
            except Exception:
                ledger.record(call, Outcome(elapsed_s=time.monotonic() - started, attempts=1))
                raise
            timed = Outcome(
                value=outcome.value,
                refusal=outcome.refusal,
                elapsed_s=time.monotonic() - started,
                attempts=max(1, outcome.attempts),
            )
            ledger.record(call, timed)
            _logger.info(
                "model_call",
                extra={
                    "task": call.label(),
                    "model": call.model,
                    "ok": timed.ok,
                    "elapsed_s": round(timed.elapsed_s, 3),
                },
            )
            return timed

        return run

    return layer


def standard_stack(budget: CallBudget, ledger: Ledger, max_retries: int) -> tuple[Middleware, ...]:
    """The layers every model call in this project runs inside, outermost first.

    The order is the design rather than a list. The ceiling is outermost so that a
    call the screen blocks still costs a slot: an attacker who could send blocked
    prompts for free would have an unmetered channel into the application. The
    screen comes next, then the retry loop, then the ledger innermost so that a
    retried call appears as the several calls it actually was.

    That leaves one gap, and it is closed by handing the budget to the retry layer
    as well: the ceiling sees a retried call once, so without it a ceiling of
    thirty admitted a hundred and twenty provider requests. With it, every attempt
    is claimed exactly once and ``budget.spent`` matches the ledger's length.

    Args:
        budget: The run's call ceiling.
        ledger: Where each attempt is recorded.
        max_retries: Additional attempts for a transient failure.

    Returns:
        The stack, ready for :func:`compose`. Returned as data rather than
        applied here so that a caller can inspect the order, and a test can
        exercise one layer without the rest.
    """
    return (
        enforce_ceiling(budget),
        screen_prompt(),
        retry_transient(max_retries, budget=budget),
        record_calls(ledger),
    )
