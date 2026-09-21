"""What one answer cost, counted while it was being produced.

:mod:`src.logging_setup` already records every model call as a line of JSON, which
answers "what did today cost?" for whoever reads the logs. It does not answer the
question a person sitting in front of the application asks, which is narrower and
more immediate: *what did **this** answer cost, and which step spent it?* That
needs the calls for one run collected in one place and handed back with the
answer, which is what a :class:`Meter` does.

Two decisions worth stating.

**Prices are estimates, and a missing price stays missing.** The numbers in
:data:`PRICES` are the gateway's published rates copied by hand; they change
without telling us, and the gateway's own invoice is the authority. So a model
this table has never heard of contributes tokens but no money, and
:attr:`Usage.fully_priced` goes false rather than the total quietly
under-reporting. An approximate cost clearly labelled is useful. A wrong cost
presented as exact is the same failure this project exists to avoid, one layer up.

**Counting must never cost an answer.** The meter observes; it cannot alter a
call, and :func:`src.logging_setup.observing_calls` swallows a failing observer.
An accounting bug should show up as a missing number on screen, never as a
traceback where a verified energy should have been.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import NamedTuple

from src.logging_setup import CallLog, observing_calls

TOKENS_PER_PRICE_UNIT = 1_000_000
"""Prices are quoted per million tokens, which is how providers publish them."""


class Price(NamedTuple):
    """What one model charges, in US dollars per million tokens.

    Attributes:
        input_usd: Cost of a million prompt tokens.
        output_usd: Cost of a million generated tokens. Always the larger of the
            two, often by a factor of five -- which is why a verbose system
            prompt is cheap and a verbose *answer* is not.
    """

    input_usd: float
    output_usd: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Price one call.

        Args:
            prompt_tokens: Tokens sent.
            completion_tokens: Tokens generated.

        Returns:
            The cost in US dollars.

        Examples:
            >>> Price(1.0, 5.0).cost(1_000_000, 200_000)
            2.0
        """
        prompt_cost = prompt_tokens * self.input_usd
        completion_cost = completion_tokens * self.output_usd
        return (prompt_cost + completion_cost) / TOKENS_PER_PRICE_UNIT


PRICES: dict[str, Price] = {
    "openai/gpt-5-mini": Price(0.25, 2.00),
    "anthropic/claude-haiku-4.5": Price(1.00, 5.00),
    "anthropic/claude-3.5-haiku": Price(0.80, 4.00),
    "openai/gpt-4o-mini": Price(0.15, 0.60),
    "openai/gpt-4.1-mini": Price(0.40, 1.60),
    "google/gemini-2.5-flash": Price(0.30, 2.50),
    "google/gemini-2.5-flash-lite": Price(0.10, 0.40),
}
"""Published rates for the models offered in the knob, as of August 2026.

Keyed by the OpenRouter slug, which is what the client reports as its model name.
Deliberately covers only :data:`src.agent.setting.MODEL_CHOICES`: a user who
types a custom slug gets token counts and no cost estimate, which is the truthful
answer rather than a guess dressed as a figure.
"""


def price_for(model: str) -> Price | None:
    """Look up a model's published rate.

    Args:
        model: The model identifier the client reported.

    Returns:
        The rate, or ``None`` if this table does not list the model.

    Examples:
        >>> price_for("openai/gpt-4o-mini").output_usd
        0.6
        >>> price_for("some/model-nobody-has-priced") is None
        True
    """
    return PRICES.get(model.strip())


@dataclass(frozen=True, slots=True)
class Call:
    """One model call, as the meter saw it.

    Attributes:
        model: Which model was called.
        purpose: Which step made the call -- ``"route"``, ``"answer"``, and so
            on. This is the field that turns a total into a diagnosis.
        prompt_tokens: Tokens sent, or zero if the provider reported none.
        completion_tokens: Tokens generated, or zero.
        reported: Whether the provider reported usage at all. Distinguishes a
            call that genuinely used no tokens (there is no such call) from one
            whose usage the gateway did not pass on.
        cost_usd: Estimated cost, or ``None`` when the model is unpriced.
        latency_ms: Wall-clock duration.
        outcome: ``"ok"`` or ``"error"``. Failed calls are counted: a retried
            call is two calls and the provider charges for both.
    """

    model: str
    purpose: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reported: bool = False
    cost_usd: float | None = None
    latency_ms: float | None = None
    outcome: str = "ok"

    @property
    def total_tokens(self) -> int:
        """Tokens in and out."""
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def of(cls, record: CallLog) -> Call:
        """Build a call from a finished log record.

        Args:
            record: The record :func:`src.logging_setup.log_llm_call` filled in.

        Returns:
            The immutable summary this module keeps.
        """
        # Clamped, not trusted: these numbers are whatever the gateway reported. A
        # negative count is not a thing a call can do, so it is a provider defect,
        # and passing one through would subtract from the run's total -- a bill that
        # goes down as the work goes up, which reads as a bug in the meter.
        prompt = max(record.prompt_tokens or 0, 0)
        completion = max(record.completion_tokens or 0, 0)
        price = price_for(record.model)
        cost = None if price is None else price.cost(prompt, completion)
        return cls(
            model=record.model,
            purpose=record.purpose,
            prompt_tokens=prompt,
            completion_tokens=completion,
            reported=record.prompt_tokens is not None or record.completion_tokens is not None,
            cost_usd=cost,
            latency_ms=record.latency_ms,
            outcome=record.outcome or "ok",
        )


class PurposeTotal(NamedTuple):
    """Everything one step of the pipeline spent.

    Attributes:
        purpose: The step.
        calls: How many calls it made.
        total_tokens: Tokens it moved.
        cost_usd: What that cost, where the models were priced.
    """

    purpose: str
    calls: int
    total_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class Usage:
    """What a run of the agent spent, in total and by step.

    An empty instance is the honest description of an answer produced without a
    model -- a blocked question, a refusal composed from code, a run with no
    credential. Those cost nothing, and the display says so rather than showing
    a blank.

    Attributes:
        calls: Every call made, in the order they were made.
    """

    calls: tuple[Call, ...] = ()

    @property
    def calls_made(self) -> int:
        """How many model calls the run made."""
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        """Tokens sent across the run."""
        return sum(call.prompt_tokens for call in self.calls)

    @property
    def completion_tokens(self) -> int:
        """Tokens generated across the run."""
        return sum(call.completion_tokens for call in self.calls)

    @property
    def total_tokens(self) -> int:
        """Tokens in and out across the run."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        """Estimated cost of the priced calls, in US dollars.

        Calls whose model is unpriced contribute nothing here, so read this
        alongside :attr:`fully_priced` -- otherwise a run on a custom slug looks
        free rather than unmeasured.
        """
        return sum(call.cost_usd or 0.0 for call in self.calls)

    @property
    def fully_priced(self) -> bool:
        """Whether every call made was against a model in :data:`PRICES`."""
        return all(call.cost_usd is not None for call in self.calls)

    @property
    def fully_reported(self) -> bool:
        """Whether every call came back with usage metadata attached."""
        return all(call.reported for call in self.calls)

    def by_purpose(self) -> tuple[PurposeTotal, ...]:
        """Break the run down by pipeline step.

        Returns:
            One total per step, heaviest first, so the step to look at is the
            one at the top.
        """
        totals: dict[str, PurposeTotal] = {}
        for call in self.calls:
            running = totals.get(call.purpose, PurposeTotal(call.purpose, 0, 0, 0.0))
            totals[call.purpose] = PurposeTotal(
                call.purpose,
                running.calls + 1,
                running.total_tokens + call.total_tokens,
                running.cost_usd + (call.cost_usd or 0.0),
            )
        return tuple(sorted(totals.values(), key=lambda total: total.total_tokens, reverse=True))

    def summary(self) -> str:
        """One line for under an answer.

        Returns:
            Calls, tokens and estimated cost, hedged exactly as far as the data
            deserves.

        Examples:
            >>> Usage().summary()
            'No model was called.'
            >>> priced = Call("openai/gpt-4o-mini", "answer", 1000, 500, True, 0.00045)
            >>> Usage((priced,)).summary()
            '1 model call · 1,500 tokens · ≈ $0.000450'
        """
        if not self.calls:
            return "No model was called."
        plural = "" if self.calls_made == 1 else "s"
        parts = [f"{self.calls_made} model call{plural}", f"{self.total_tokens:,} tokens"]
        if self.calls and self.fully_priced:
            parts.append(f"≈ ${self.cost_usd:.6f}")
        elif self.cost_usd > 0.0:
            parts.append(f"≥ ${self.cost_usd:.6f} (some models unpriced)")
        else:
            parts.append("cost unknown (unpriced model)")
        return " · ".join(parts)

    def merge(self, other: Usage) -> Usage:
        """Combine two runs' usage.

        Args:
            other: Usage to add, treated as having happened after this one.

        Returns:
            A new instance holding both runs' calls -- which is how a session
            total is accumulated over a conversation.
        """
        return Usage(calls=(*self.calls, *other.calls))


@dataclass(slots=True)
class Meter:
    """Collects model calls while a run is in progress.

    Mutable, and the only mutable thing in this module: it is filled in by a
    callback during the run and read once at the end, at which point
    :attr:`usage` freezes what it saw.

    Attributes:
        seen: The calls recorded so far.
    """

    seen: list[Call] = field(default_factory=list)

    def record(self, call: CallLog) -> None:
        """Record one finished call.

        Args:
            call: The completed log record.
        """
        self.seen.append(Call.of(call))

    @property
    def usage(self) -> Usage:
        """An immutable snapshot of what has been recorded."""
        return Usage(calls=tuple(self.seen))


@contextmanager
def metered() -> Iterator[Meter]:
    """Count every model call made inside this block.

    Yields:
        The meter. Read :attr:`Meter.usage` after the block; reading it inside
        gives a partial count, which is occasionally what you want.

    Examples:
        >>> from src.logging_setup import log_llm_call
        >>> with metered() as meter:
        ...     with log_llm_call("openai/gpt-4o-mini", purpose="route") as call:
        ...         call.prompt_tokens, call.completion_tokens = 100, 20
        >>> meter.usage.total_tokens
        120
    """
    meter = Meter()
    with observing_calls(meter.record):
        yield meter
