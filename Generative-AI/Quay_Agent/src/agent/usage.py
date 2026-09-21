"""What one answer cost, counted while it was being produced.

:mod:`src.logging_setup` already records every model call as a line of JSON, which
answers "what did today cost?" for whoever reads the logs. It does not answer the
question a person sitting in front of the application asks, which is narrower and
more immediate: *what did **this** answer cost, and which step spent it?* That
needs the calls for one run collected in one place and handed back with the
answer, which is what a :class:`Meter` does.

Two decisions worth stating.

Prices are estimates, and a missing price stays missing. The numbers in
:data:`PRICES` are the gateway's published rates copied by hand; they change
without telling us, and the gateway's own invoice is the authority. So a model
this table has never heard of contributes tokens but no money, and
:attr:`Usage.fully_priced` goes false rather than the total quietly
under-reporting. An approximate cost clearly labelled is useful. A wrong cost
presented as exact is the same failure this project exists to avoid, one layer up.

Counting must never cost an answer. The meter observes; it cannot alter a
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
    # The strong tier's default, and it was the one slug missing. Every explanation
    # and every code draft is served by it, so a session that used the model as
    # shipped reported "cost unknown (unpriced model)" -- which reads on screen as
    # "no model was called", the exact fact this panel exists to distinguish.
    "openai/gpt-4o": Price(2.50, 10.00),
    "google/gemini-2.5-flash": Price(0.30, 2.50),
    "google/gemini-2.5-flash-lite": Price(0.10, 0.40),
    # The guest tier. Zero in both directions is a published rate, not a missing
    # one, so it belongs here rather than in the table below: a guest campaign
    # reports its cost as $0.00 and `fully_priced` stays true, which is the
    # difference between "this cost nothing" and "we could not tell what it cost".
    "google/gemma-4-31b-it:free": Price(0.00, 0.00),
}
"""Published rates for the models offered in the knob, as of August 2026.

Keyed by the OpenRouter slug, which is what the client reports as its model name.
Every slug in :data:`src.agent.model_selection.DEFAULT_TIER_SLUGS` is listed, which
is what the usage test pins: a model the project ships as a default and
cannot price makes its own session panel say "cost unknown", and a reader has no way
to tell that from "no model was called". A user who types a custom slug still gets
token counts and no cost estimate, which is the truthful answer rather than a guess
dressed as a figure.
"""


HEURISTIC_PRICES: dict[str, Price] = {
    # Every figure below is now read from the gateway's own index by
    # the slug verifier rather than reasoned out, and reading it
    # settled the question these rows were guessing at: SEVEN OF THEM WERE WRONG,
    # three by more than a factor of three (`gpt-5.4-nano` was under by 4x,
    # `gpt-5.4-mini` by 3x, `deepseek-v4-flash` over by 3x). The values are
    # corrected here. They stay in this table rather than moving to PRICES because
    # promoting them changes what `price_basis` reports for a dozen models and what
    # `fully_priced` means for a run, which is a deliberate change and not a side
    # effect of a price correction -- and because an over-cautious label costs a
    # reader nothing while an over-confident one costs them a budget.
    #
    # Corroborated: two independent transcriptions of this same subscription agree.
    "openai/gpt-4.1-nano": Price(0.10, 0.40),
    "openai/gpt-5-nano": Price(0.05, 0.40),
    "anthropic/claude-3-haiku": Price(0.25, 1.25),
    "openai/gpt-5.2": Price(1.75, 14.00),
    "deepseek/deepseek-v4-flash": Price(0.09, 0.17),
    # Extrapolated: one source only, reasoning from a neighbouring model in the same
    # family. For the rows that postdate anything verifiable from here -- GPT-5.4,
    # DeepSeek v4 Pro, GLM 5.2, MiniMax M2.7, Gemma 4 -- these are the *shape* of a
    # price (this is a nano, this is a flagship) rather than a figure to budget on.
    "openai/gpt-3.5-turbo": Price(0.50, 1.50),
    "openai/gpt-4o-2024-11-20": Price(2.50, 10.00),
    "openai/gpt-5.2-codex": Price(1.75, 14.00),
    "openai/gpt-5.4": Price(2.50, 15.00),
    "openai/gpt-5.4-mini": Price(0.75, 4.50),
    "openai/gpt-5.4-nano": Price(0.20, 1.25),
    "deepseek/deepseek-v4-pro": Price(1.04, 2.08),
    "google/gemma-4-31b-it": Price(0.09, 0.34),
    "minimax/minimax-m2.7": Price(0.30, 1.20),
    "z-ai/glm-5.2": Price(0.97, 3.04),
}
"""Rates nobody published, reasoned out and kept in their own table.

The subscription publishes a model list and no rates, so without this table the
fourteen models here contribute tokens and no money at all. Each rate is inferred
from the vendor's public pricing for the nearest model of the same size and family.

This is deliberately not merged into :data:`PRICES`. That is a rate card copied
from vendors; this is arithmetic about a neighbouring model. Merged, the one panel
whose job is to say what a run cost could no longer tell a quoted figure from a
guess, and it would look equally confident either way.

So :func:`price_for` still means *published*, :attr:`Usage.fully_priced` still goes
false on every model here, and a total resting on any of these still reports itself
as a floor. What these buy is a labelled number in place of the word "unpriced",
which is worth having: an estimate a reader can argue with beats a blank they
cannot.
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


def estimated_price_for(model: str) -> Price | None:
    """Look up a model's rate, falling back to the reasoned estimate.

    Published rates win: this consults :data:`HEURISTIC_PRICES` only where
    :data:`PRICES` is silent, so it can never shadow a real rate card with a guess.

    Use it where a display would otherwise print "unpriced" and where a labelled
    estimate is more use than a blank -- never to compute a total that is presented
    as spend, and never to pick a winner. :func:`src.evals.bakeoff.cheapest` still
    reads published prices alone, because naming a model cheapest on an
    extrapolation recommends the one row nobody has costed.

    Args:
        model: The model identifier the client reported.

    Returns:
        The rate, or ``None`` if neither table lists the model.

    Examples:
        >>> estimated_price_for("openai/gpt-4o-mini").output_usd
        0.6
        >>> estimated_price_for("openai/gpt-5.4-nano").input_usd
        0.2
        >>> estimated_price_for("some/model-nobody-has-priced") is None
        True
    """
    slug = model.strip()
    return PRICES.get(slug) or HEURISTIC_PRICES.get(slug)


def price_basis(model: str) -> str:
    """Say where a model's rate came from, so a display can label it.

    A cost shown without its basis invites the reader to treat an extrapolation as
    an invoice. This exists so the label travels with the number rather than being
    reconstructed by whoever draws the panel.

    Args:
        model: The model identifier the client reported.

    Returns:
        ``"published"`` for a vendor rate card, ``"estimated"`` for a reasoned
        figure from :data:`HEURISTIC_PRICES`, ``"unknown"`` for neither.

    Examples:
        >>> price_basis("openai/gpt-4o")
        'published'
        >>> price_basis("z-ai/glm-5.2")
        'estimated'
        >>> price_basis("some/model-nobody-has-priced")
        'unknown'
    """
    slug = model.strip()
    if slug in PRICES:
        return "published"
    if slug in HEURISTIC_PRICES:
        return "estimated"
    return "unknown"


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
    def failed_calls(self) -> int:
        """How many calls came back as errors rather than answers.

        Recorded on every :class:`Call` since the meter was written and read by
        nothing until a live session spent its whole ceiling on calls that all
        failed, produced a correct answer from the deterministic paths, and said
        so nowhere a person could see. The meter knew; nobody asked it.
        """
        return sum(1 for call in self.calls if call.outcome != "ok")

    @property
    def every_call_failed(self) -> bool:
        """Whether the run reached the gateway and got nothing usable back.

        The distinction that matters on screen. One failed call among ten is a
        provider hiccup the retry layer exists to absorb; *all* of them failing is
        a credential, a spent balance or a severed network, and the answer on the
        page -- which will still be a correct answer, computed and cross-checked --
        was written without a language model having contributed a word to it.

        Returns:
            True when at least one call was made and none of them succeeded.
        """
        return bool(self.calls) and self.failed_calls == len(self.calls)

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
    def estimated_cost_usd(self) -> float | None:
        """What the run cost at published rates, falling back to reasoned ones.

        Every call is repriced from its own prompt and completion counts, so this
        is the estimate applied to the tokens that were actually spent rather than
        a total scaled by a ratio.

        This does **not** relax :attr:`fully_priced`. A run that needed
        :data:`HEURISTIC_PRICES` for any call still reports itself as an estimate
        wherever that flag is read; this only gives a display something to show in
        place of the word "unpriced".

        Returns:
            The cost in US dollars, or ``None`` if any call was against a model
            neither table lists -- because a partial total quietly dropping one
            model is the failure mode this whole module exists to avoid.
        """
        total = 0.0
        for call in self.calls:
            rate = estimated_price_for(call.model)
            if rate is None:
                return None
            total += rate.cost(call.prompt_tokens, call.completion_tokens)
        return total

    def cost_label(self) -> str:
        """The cost as a display should print it, with its basis attached.

        One place, because the same total is drawn on two panels and in
        :meth:`summary`, and three that computed it separately would eventually
        disagree. Mirrors :func:`src.evals.bakeoff.cost_label` for the same reason.

        Returns:
            A published total, an estimate marked ``est.``, or a floor -- and
            ``"no charge"`` for a run that made no call at all.

        Examples:
            >>> Usage().cost_label()
            'no charge'
            >>> published = Call("openai/gpt-4o-mini", "answer", 100_000, 20_000, True, 0.027)
            >>> Usage((published,)).cost_label()
            '$0.0270'
            >>> reasoned = Call("z-ai/glm-5.2", "answer", 20_000, 4_000, True, None)
            >>> Usage((reasoned,)).cost_label()
            '$0.0316 est.'
            >>> Usage((Call("nobody/knows", "answer", 10, 5, True, None),)).cost_label()
            '≥ $0.0000'
        """
        if not self.calls:
            return "no charge"
        if self.fully_priced:
            return f"${self.cost_usd:.4f}"
        estimated = self.estimated_cost_usd
        if estimated is not None:
            return f"${estimated:.4f} est."
        return f"≥ ${self.cost_usd:.4f}"

    def cost_basis(self) -> str:
        """One sentence saying how far the figure beside it can be trusted.

        Returns:
            The caption to print under :meth:`cost_label`, or the empty string
            when the total is a plain published one needing no qualification.
        """
        if not self.calls or self.fully_priced:
            return ""
        if self.estimated_cost_usd is not None:
            return (
                "Some calls used a model the vendor publishes no rate for. Those are "
                "priced from the reasoned figures in the catalogue, so the total is an "
                "estimate rather than an invoice."
            )
        return (
            "Some calls used a model with no price on record at all, so the cost is a "
            "floor rather than a total."
        )

    @property
    def rests_on_estimates(self) -> bool:
        """Whether any call priced only through :data:`HEURISTIC_PRICES`.

        The distinction :attr:`fully_priced` cannot draw on its own: it goes false
        both for a model with a reasoned estimate and for one with nothing at all,
        and those deserve different words on screen.
        """
        return any(price_basis(call.model) == "estimated" for call in self.calls)

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
        if self.fully_priced:
            parts.append(f"≈ ${self.cost_usd:.6f}")
        elif (estimated := self.estimated_cost_usd) is not None:
            parts.append(f"≈ ${estimated:.6f} (estimated rates)")
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
