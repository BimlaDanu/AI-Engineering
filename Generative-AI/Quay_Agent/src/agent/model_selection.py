"""Choosing which model answers which call, and building it once.

A campaign makes six or seven kinds of model call, and they want different
things. Breaking a tie between two knowledge-base names wants an answer in under
a second; reading a request into a Hamiltonian wants a model that follows a
schema exactly; writing the closing report wants prose a person will read.
Sending all three to one model pays the report's model for the tie-break, or
asks the tie-break's model to write the report.

So calls are grouped into tiers, tiers are bound to slugs, and every call site
names a :class:`Task` rather than a model. Three consequences:

Latency -- everything on the hot path is pinned to the fast tier, so the stages
added to make retrieval better cannot be the stages that make it slow. The tier
is a property of the call, so a deployment pointing the fast tier at something
slow has made a visible choice rather than an accidental one.

Cost -- the expensive model is reached for at most twice in a campaign, and on
the feasibility branch not at all: that report is assembled from templates over
numbers the arithmetic produced, which is what lets a campaign run with no model
reachable.

Attribution -- every call records the task and the slug that served it, so a bad
answer traces to a model rather than to "the LLM".

Selection is policy, which is why it is here and not beside the catalogue:
:mod:`src.model_catalogue` states what the subscription exposes and makes no
judgement. Building a model is lazy and memoised, so a campaign that never
escalates never constructs the strong model.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from src.agent.llm import MAX_TOOL_ROUNDS, Consultation
from src.agent.middleware import (
    CallBudget,
    Invocation,
    Ledger,
    Outcome,
    compose,
    standard_stack,
)
from src.logging_setup import get_logger
from src.model_catalogue import card_for, models_for
from src.settings import (
    DEFAULT_CHAT_MODEL,
    GUEST_CHAT_MODEL,
    Settings,
    get_settings,
    get_settings_or_none,
)

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound="BaseModel")
"""Structured-output type a call is asked for."""

_logger = get_logger("agent.model_selection")

Tier = Literal["fast", "standard", "strong"]
"""How much model a call is worth.

Three rather than two because the middle case is the common one and would
otherwise be forced to the wrong side: reading a request into a structured form
is more than a tie-break and less than a report. Three rather than five because a
tier nobody can describe in one sentence is a tier that gets assigned by feel.
"""

Task = Literal[
    "intent_reading",
    "shelf_choice",
    "passage_grading",
    "query_rewrite",
    "query_expansion",
    "passage_ordering",
    "problem_reading",
    "depth_suggestion",
    "followup_suggestion",
    "tool_consultation",
    "explanation",
    "code_drafting",
    "report_writing",
]
"""Every kind of model call the project makes.

An enumeration rather than free text so that a new call site has to declare
itself here, and so that the cost of a campaign can be accounted for by task
without parsing log messages.
"""

TASK_TIERS: dict[Task, Tier] = {
    # Asked before anything else can start, and only when the word signals tied --
    # so the cheap tier is being handed the genuinely ambiguous cases. A slower tier
    # here delays every branch, and the mistake costs one wrong-shaped document
    # rather than one wrong number.
    "intent_reading": "fast",
    # On the retrieval hot path. A person is waiting, and each of these is a
    # small judgement with a short answer.
    "shelf_choice": "fast",
    "passage_grading": "fast",
    "query_rewrite": "fast",
    # Multi-query expansion and reranking are both on the hot path too, and both are
    # short judgements about *wording* rather than about physics. Expansion asks for
    # three keyword strings; ordering asks for a permutation of identifiers it was
    # handed. Neither needs a model that can reason about a spin chain, and a slower
    # tier on either would put a second network round trip in front of every answer.
    "query_expansion": "fast",
    "passage_ordering": "fast",
    # Consequential: a misread request produces a campaign that answers the wrong
    # question perfectly. It is also the binding constraint on how long a reader
    # waits before anything happens, which is not what this comment used to say.
    # `src.agent.prefetch` issues it alongside the restatement and the intent
    # reading, and those two are fast-tier and land in about a second; this one was
    # measured at 3.4-5.0 s, so the opening wait is this call and nothing else.
    # Moving it to `fast` is the largest single latency lever in the project and it
    # is a real trade -- the deterministic reader in `src.agent.reading` already
    # takes every number a sentence states outright, and `_clamp_model` prefers the
    # sentence, so what the model contributes is sentences with no numbers in them.
    # Point the eval harness at it before making that trade.
    "problem_reading": "standard",
    # The only model call *inside* a loop, and the loop runs it three or four times
    # in a row -- so a slower tier here is paid once per rung with a person watching
    # the trail fill. The reply is one integer, proposed from a short table of what
    # each depth reached, and `_next_depth` discards it unless it is positive,
    # untried and below the refusal ceiling. A cheap model that proposes badly costs
    # one rung of the ladder; a slow one costs the wait on every rung.
    "depth_suggestion": "fast",
    # Written for a person to read on a button, and it has to be a question this
    # agent can actually answer -- a cheap model proposes plausible-sounding
    # questions that the agent then refuses, which is worse than suggesting nothing.
    "followup_suggestion": "standard",
    # The one call where the *model* decides how many calls happen. Standard rather
    # than fast, because choosing a tool and reading its result back is the kind of
    # judgement a cheap model gets wrong expensively: it searches with the words it
    # was given, gets nothing, and searches again with the same words until the
    # round limit stops it. Not strong either -- the deliverable is the tool result,
    # not this call's prose, and the strong tier serves the writing that follows.
    "tool_consultation": "standard",
    # Three calls where the prose or the code *is* the deliverable rather than a
    # side effect of one. An explanation is the whole answer on its branch, and a
    # weak model's explanation of a subtlety is confidently wrong in a way a reader
    # without the physics cannot catch. Code is worse still: it is the one output a
    # reader will run, this project never runs it first, and plausible-looking code
    # that does not work is the most expensive thing here to hand somebody.
    "explanation": "strong",
    "code_drafting": "strong",
    # Read by a person, once, at the end -- and not yet wired: the feasibility
    # report is assembled by `src.agent.report.compose` from templates over computed
    # numbers, with no model in the loop, which is what lets a campaign finish with
    # no credential. The task and its prompt (`prompts.REPORT_WRITING`) are declared
    # so that narrating the report stays a one-call change rather than a redesign,
    # and the model-selection test holds every declared task to a tier
    # whether or not a call site has appeared for it yet.
    "report_writing": "strong",
}
"""Which tier serves each task.

The assignment is the argument this module is making, so it is written as data
rather than spread across branches: a reader can check it in one glance, and a test
can assert that every task has a tier without reading any logic.
"""

FALLBACK_TIERS: dict[Tier, tuple[Tier, ...]] = {
    "fast": ("standard", "strong"),
    "standard": ("fast", "strong"),
    "strong": ("standard", "fast"),
}
"""Where a tier turns when its own slug will not build.

A campaign that cannot construct its strong model should still produce a report
written by something, and a report from the standard model is worth more than a
refusal. The orders differ by tier because the substitution that hurts least
differs: a fast call would rather be slow than absent, and a strong call would
rather be adequate than absent.
"""

DEFAULT_TIER_SLUGS: dict[Tier, str] = {
    "fast": "google/gemini-2.5-flash-lite",
    "standard": DEFAULT_CHAT_MODEL,
    "strong": "anthropic/claude-haiku-4.5",
}
"""The slug each tier uses when configuration names none.

**Ordered by strength, strongest at the top of the ladder.** Which is the whole
point of having three: a tier is a claim about how much model a call is worth, and
a ladder whose rungs are not in order makes every one of those claims wrong. The
three defaults now read, weakest to strongest, ``gemini-2.5-flash-lite`` ->
``gpt-5-mini`` -> ``claude-haiku-4.5``, and they happen to be in price order too:
0.10, 0.25 and 1.00 USD per million input tokens. When strength and cost agree on
an ordering there is nothing left to trade off, which is the only comfortable
position a tier policy can be in.

``anthropic/claude-haiku-4.5`` replaced ``openai/gpt-4o`` on the strong tier. It is
a generation newer, it is **cheaper** -- 1.00/5.00 against 2.50/10.00 USD per
million tokens -- and it was already the better-evidenced of the two here: both ran
a full campaign live under ``make bakeoff`` and both returned the baseline
identically to nine decimal places, so the older and dearer model was buying
nothing this project can measure. ``gpt-4o`` stays on the shortlist, so choosing it
back is one click and no code.

All three are ``confirmed`` in the catalogue, which remains the bar for a default:
a default that fails at the first call is worse than a default that is a year old.
The standard tier is not written out here at all -- it *is*
:data:`src.settings.DEFAULT_CHAT_MODEL`, because a tier default and a settings
default naming the same tier are two things to forget to change together, and for
a while they had already drifted apart (this said ``gpt-4.1-mini`` while every real
run used ``gpt-5-mini``, so the value here was dead and disagreed with the process).
"""


ALL_TIERS: tuple[Tier, ...] = ("fast", "standard", "strong")
"""Every tier, weakest first.

Written out rather than derived from ``typing.get_args(Tier)`` because the *order*
is meaningful -- it is the ladder :data:`DEFAULT_TIER_SLUGS` claims to climb -- and
an order that comes from a type annotation is an order nobody has decided.
"""


def guest_overrides(settings: Settings | None = None) -> dict[Tier, str]:
    """Pin every tier to the free endpoint.

    What an unauthenticated visitor to a public deployment runs on. Every tier
    rather than the strong one alone, because the point is that *no* call bills the
    host, and a ladder with one free rung still spends money on the other two.

    Collapsing three tiers into one model is a real loss and worth naming: the
    tiering exists because a passage-relevance judgement and a written explanation
    are not alike, and a guest gets one model doing both. What it is not is a loss
    of *correctness* -- no tier has ever computed a number -- so a guest's figures
    and cross-checks are the signed-in visitor's figures and cross-checks.

    Args:
        settings: Configuration to read the guest model from. Defaults to the
            process settings, so a deployment moves its guest tier with an
            environment variable and no code change -- which is what it takes to
            switch to a free endpoint once an account can reach one.

    Returns:
        Every tier mapped to the guest model.

    Examples:
        >>> sorted(guest_overrides()) == ["fast", "standard", "strong"]
        True
        >>> len(set(guest_overrides().values()))
        1
    """
    resolved = settings if settings is not None else get_settings_or_none()
    configured = resolved.guest_chat_model if resolved is not None else None
    return dict.fromkeys(ALL_TIERS, configured or GUEST_CHAT_MODEL)


PINNED_SLUG = "supplied-by-caller"
"""Recorded in place of a slug when a caller handed in the model itself.

A name rather than an empty string, so a trace of a run driven by a supplied
model reads as a deliberate choice instead of a missing field.
"""

DEFAULT_CALL_CEILING: int = cast(
    int, Settings.model_fields["max_model_calls_per_run"].get_default()
)
"""Model calls one campaign may make when configuration cannot be read.

A backstop rather than a policy: the number a deployment runs under comes from its
configuration, and this is what a pool falls back to when there is none at all.

Read off the field rather than written out again. Restated, the two drift the
moment one is tuned -- and they had: the configured default was raised to clear
the honest call count of a prose answer and this copy stayed at twelve, so a pool
that could not read settings kept the ceiling that truncates a legitimate run.
"""


@dataclass(frozen=True, slots=True)
class Selection:
    """The model chosen for one task, and how the choice was reached.

    Attributes:
        task: What the call is for.
        tier: The tier that task maps to.
        slug: The model actually used.
        substituted: Whether the tier's own slug could not be built and a
            fallback served the call. Recorded because a campaign that quietly
            ran on the wrong model looks exactly like one that ran on the right
            one, right up until someone tries to reproduce it.
    """

    task: Task
    tier: Tier
    slug: str
    substituted: bool = False

    def describe(self) -> dict[str, str | bool]:
        """Render the choice as scalar trace fields.

        Returns:
            Primitives only, ready for a log record.
        """
        return {
            "task": self.task,
            "tier": self.tier,
            "model": self.slug,
            "substituted": self.substituted,
        }


def tier_for(task: Task) -> Tier:
    """The tier a task is served from.

    Args:
        task: The kind of call.

    Returns:
        Its tier.

    Examples:
        >>> tier_for("passage_grading"), tier_for("report_writing")
        ('fast', 'strong')
    """
    return TASK_TIERS[task]


def slug_for(tier: Tier, settings: Settings | None = None) -> str:
    """The model slug a tier resolves to.

    Configuration wins over the default, and neither is validated against the
    catalogue here: an unlisted slug is a warning, not a refusal, because the
    subscription can gain a model before this project hears about it.

    Args:
        tier: The tier to resolve.
        settings: Configuration to read. Defaults to the process settings.

    Returns:
        A model slug.
    """
    # Resolved through the forgiving reader on purpose. This function answers "which
    # slug would a call use", which is a *preference*, and it runs inside the graph's
    # screen node -- so raising here on a missing credential stopped a campaign two
    # nodes in with a validation error instead of letting it take the deterministic
    # path. Naming a model costs nothing when no model will be called.
    resolved = settings if settings is not None else get_settings_or_none()
    configured = (
        {
            "fast": resolved.fast_model,
            "standard": resolved.chat_model,
            "strong": resolved.strong_model,
        }[tier]
        if resolved is not None
        else None
    )
    slug = configured or DEFAULT_TIER_SLUGS[tier]
    if card_for(slug) is None:
        _logger.warning(
            "model_not_catalogued",
            extra={"tier": tier, "model": slug, "detail": "the subscription may not expose it"},
        )
    return slug


FEATURED_SLUGS: tuple[str, ...] = (
    # Basic -- the cheap, confirmed, fast end. Anything a person is waiting on
    # belongs here, and one of these is the fast tier's default.
    # Free first, and it is not a courtesy: this is what an unauthenticated
    # visitor is served, so it is the row most likely to be the one actually
    # running. `anthropic/claude-3.5-haiku` used to sit at the end of this group
    # until the slug verifier found the gateway had retired it --
    # the selector was offering a model that fails at the first call.
    "google/gemma-4-31b-it:free",
    "google/gemini-2.5-flash-lite",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    # Middle -- more capable, still priced, still quick enough to sit on a tier a
    # person waits on when the question is harder than a tie-break.
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-mini",
    # Advanced -- what the written report is worth reaching for, and two entries
    # from other vendors so the comparison is not four flavours of one company.
    "openai/gpt-4o",
    "openai/gpt-5.4",
    "deepseek/deepseek-v4-pro",
)
"""The shortlist a chat selector offers, cheapest and most certain first.

Ten rather than twenty-three, and the cut is a usability decision rather than a
capability one. A selector holding every slug the subscription exposes asks a person
to choose between six variants of one model family, three of which were reconstructed
from display names and have never resolved against the live index -- so the commonest
outcome of a long list is a run that dies at its first call several minutes in.

Ordered basic to advanced, and spread across four vendors on purpose. The point of
letting somebody change the model at all is the claim the **Model bake-off** tab
tests: the numbers do not move when the model does. A list that offered only one
vendor's ladder would let a reader check that claim against a family of models that
share a tokeniser and a training recipe, which is the weakest possible version of it.

Eight of the ten carry a price in :data:`src.agent.usage.PRICES`; the three that do
not are marked in the interface rather than hidden, because "this model works and
nobody has costed it" is a different thing from "this model is free".

:func:`selectable_slugs` still returns everything. This is the shortlist, not the
allowlist -- the sidebar keeps the full catalogue, so a slug released after this
tuple was written is still reachable without an edit.
"""


def featured_slugs() -> tuple[str, ...]:
    """The shortlist, with anything the catalogue no longer lists dropped.

    Filtered rather than returned raw, so a slug retired from the subscription
    disappears from the selector instead of sitting there until somebody picks it
    and waits several minutes for the failure.

    Returns:
        The entries of :data:`FEATURED_SLUGS` the catalogue still knows about, in
        that order. Falls back to the full selectable list if the filter empties it,
        because a selector with no options is worse than one with too many.
    """
    known = set(selectable_slugs())
    kept = tuple(slug for slug in FEATURED_SLUGS if slug in known)
    return kept or selectable_slugs()


def selectable_slugs() -> tuple[str, ...]:
    """Every chat model a person may choose between, confirmed ones first.

    This is what a selector offers. Confirmed slugs lead because an unconfirmed
    one is a reconstruction that may not resolve, and a list that mixes the two
    without ordering invites someone to pick the broken option first.

    Returns:
        Chat slugs, confirmed before unconfirmed, catalogue order within each.
    """
    cards = models_for("chat")
    return tuple(card.slug for card in cards if card.confirmed) + tuple(
        card.slug for card in cards if not card.confirmed
    )


@dataclass
class ModelPool:
    """The models a single campaign may call, built at most once each.

    Not frozen and not shared: a pool is the campaign's own, and the mutation it
    carries is a cache rather than state anyone reasons about. One per campaign
    rather than one per process because a run that overrides a tier must not
    leave that override behind for the next run.

    Choosing a model is atomic; making the call is not. One campaign issues
    several calls at once -- see :mod:`src.agent.prefetch` -- and selection reads and
    writes shared state: the build cache, the ledger of what was chosen, and the run's
    call counter. Without the lock, two threads selecting at the same moment would each
    append a :class:`Selection` and then both read back the *last* one, so a call could
    be recorded under another call's model and tier. The lock covers only the in-process
    bookkeeping; the HTTP request happens outside it, which is the entire point of
    making the calls concurrently.

    Attributes:
        settings: Configuration the slugs are read from.
        overrides: Slugs that replace the configured ones, by tier. This is where
            a person's choice of model arrives.
        offline: Set when no model may be called at all. Every request then
            returns ``None`` and the deterministic path runs, which is a
            supported mode and the one the tests use.
        pinned: One model to serve every task, overriding the tiers. This is how
            a caller hands in a model it built itself -- a test double, or a
            single model a person picked -- without having to know which tier a
            task maps to.
        budget: The run's call ceiling. Built from configuration when omitted.
        ledger: Where every attempt is timed and recorded.
    """

    settings: Settings | None = None
    overrides: dict[Tier, str] = field(default_factory=dict)
    offline: bool = False
    pinned: BaseChatModel | None = None
    budget: CallBudget | None = None
    ledger: Ledger = field(default_factory=Ledger)
    _built: dict[str, BaseChatModel | None] = field(default_factory=dict, repr=False)
    _chosen: list[Selection] = field(default_factory=list, repr=False)
    _choosing: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        """Give the pool a call ceiling if the caller did not name one.

        Read from configuration rather than defaulted here, so that the ceiling a
        campaign runs under is the one a deployment set and not one hidden in a
        dataclass. A pool with no configuration to read gets no ceiling, which is
        the offline case and spends nothing anyway.
        """
        if self.budget is None:
            try:
                base = get_settings() if self.settings is None else self.settings
                self.budget = CallBudget(limit=base.max_model_calls_per_run)
            except Exception:
                self.budget = CallBudget(limit=DEFAULT_CALL_CEILING)

    def slug(self, tier: Tier) -> str:
        """The slug this pool would use for a tier, override included.

        Args:
            tier: The tier to resolve.

        Returns:
            A model slug.
        """
        return self.overrides.get(tier) or slug_for(tier, self.settings)

    def _build(self, slug: str) -> BaseChatModel | None:
        """Construct one model, or remember that it could not be built.

        The failure is cached alongside the success. Without that, a campaign
        with no credential would attempt to construct a client on every call and
        pay the same exception thirty times over.

        Args:
            slug: The model to build.

        Returns:
            The model, or ``None`` if construction failed.
        """
        if slug in self._built:
            return self._built[slug]
        from src.agent.llm import build_chat_model

        base = self.settings if self.settings is not None else get_settings_or_none()
        if base is None:
            # No configuration means no credential, so there is nothing to build. The
            # caller's contract already covers it: ``None`` is the offline answer.
            self._built[slug] = None
            return None
        try:
            model = build_chat_model(base.model_copy(update={"chat_model": slug}))
        except Exception as error:
            _logger.warning(
                "model_unavailable",
                extra={"model": slug, "detail": type(error).__name__},
            )
            model = None
        self._built[slug] = model
        return model

    def for_task(self, task: Task) -> BaseChatModel | None:
        """The model that should serve one call, or ``None`` to stay offline.

        Args:
            task: The kind of call about to be made.

        Returns:
            A model, or ``None`` when the pool is offline or nothing would build.
            ``None`` is a normal answer: every call site here has a deterministic
            path behind it, and running that path is a supported mode rather than
            a degraded one.
        """
        if self.offline:
            return None
        tier = tier_for(task)
        if self.pinned is not None:
            self._chosen.append(Selection(task, tier, PINNED_SLUG))
            return self.pinned
        for candidate, substituted in ((tier, False), *((t, True) for t in FALLBACK_TIERS[tier])):
            slug = self.slug(candidate)
            model = self._build(slug)
            if model is not None:
                self._chosen.append(Selection(task, tier, slug, substituted))
                if substituted:
                    _logger.info(
                        "model_substituted",
                        extra={"task": task, "tier": tier, "model": slug},
                    )
                return model
        return None

    def invoke(
        self,
        task: Task,
        schema: type[SchemaT],
        system: str,
        text: str,
    ) -> SchemaT | None:
        """Make one structured call through the full middleware stack.

        The single funnel. Every model call the campaign makes arrives here, which
        is what lets the ceiling, the prompt screen, the retry and the timing
        record be written once instead of at six call sites.

        Args:
            task: What the call is for. Decides the tier, and labels the record.
            schema: The structured shape the answer must satisfy.
            system: The instruction. Written by this project, so it is not
                screened -- see :func:`src.agent.middleware.screen_prompt`.
            text: The content, which is where untrusted material arrives.

        Returns:
            The validated answer, or ``None``. ``None`` covers every way a call
            can fail to produce one -- offline, over the ceiling, blocked by the
            screen, refused by the provider, or malformed -- because the caller's
            response to all of them is the same: run the deterministic path.
        """

        def execute(model: BaseChatModel, pending: Invocation) -> object | None:
            from src.agent.llm import ask_structured

            return ask_structured(model, schema, pending.system, pending.text, pending.task)

        return cast("SchemaT | None", self._through_the_stack(task, system, text, execute))

    def consult(
        self,
        task: Task,
        system: str,
        text: str,
        toolkit: Sequence[BaseTool],
        max_rounds: int = MAX_TOOL_ROUNDS,
    ) -> Consultation | None:
        """Let the model call tools, through the same funnel as every other call.

        The third of the three ways this project calls a model, and the only one
        where the model decides how much work happens. Everything the other two
        get -- the call ceiling, the prompt screen, the retry, the timing record --
        applies here too, which is the reason it goes through
        :meth:`_through_the_stack` rather than reaching for a model directly.

        One consultation is several provider calls, and each is logged and billed
        separately by :func:`src.agent.llm.ask_with_tools`. The ceiling this funnel
        enforces is therefore the *right* ceiling: it counts what was actually
        spent rather than counting a loop as one call.

        Args:
            task: What the consultation is for. Decides the tier and labels every
                call it makes.
            system: The instruction. Written by this project, so unscreened.
            text: The content, which is where untrusted material arrives.
            toolkit: The tools the model may reach. Nothing outside this is
                callable, whatever the model asks for.
            max_rounds: Ceiling on model calls within the loop.

        Returns:
            The consultation, or ``None`` for every way of not getting one --
            offline, over the ceiling, screened, refused, or a model that cannot
            call tools at all. The caller's answer to all of them is the same:
            write the answer from what it already has.
        """

        def execute(model: BaseChatModel, pending: Invocation) -> object | None:
            from src.agent.llm import ask_with_tools

            return ask_with_tools(
                model,
                pending.system,
                pending.text,
                toolkit,
                pending.task,
                max_rounds,
            )

        outcome = self._through_the_stack(task, system, text, execute)
        return outcome if isinstance(outcome, Consultation) else None

    def narrate(
        self,
        task: Task,
        system: str,
        text: str,
        sink: Callable[[str], None] | None = None,
    ) -> str | None:
        r"""Make one call that comes back as prose, through the same funnel.

        The counterpart to :meth:`invoke`, and the one to use whenever **the reply
        itself is what a person reads**. The difference is not presentational.

        A structured reply travels as a JSON string, and JSON has already assigned
        meanings to ``\b``, ``\f``, ``\r``, ``\t`` and ``\n``. LaTeX assigns
        different ones to the same sequences, so a model that writes ``\frac`` or
        ``\times`` into a JSON field and does not double the backslash sends a form
        feed and a tab -- and the reader is shown ``rac{1}{2}`` and ``imes``. Nothing
        downstream can repair it, because by the time the JSON is parsed a real tab
        and an eaten ``\times`` are the same byte. The fix is not to put the
        mathematics in JSON at all, which is what this method is for.

        The explanation branch is the one output in this project that is *mostly*
        mathematics, and it was going out through :meth:`invoke`. It survived
        whenever the model happened to escape correctly, which is most of the time
        and is not a guarantee -- exactly the failure that shows up as an occasional
        garbled answer nobody can reproduce.

        Args:
            task: What the call is for. Decides the tier and labels the record.
            system: The instruction. Written by this project, so unscreened.
            text: The content, which is where untrusted material arrives.
            sink: Given, the reply is streamed and each piece is passed to this as
                it arrives. Everything else about the call is unchanged -- the same
                ceiling, the same screen, the same retry and the same ledger entry
                with the same token counts -- so a streamed answer costs and is
                recorded exactly as an unstreamed one. What moves is when the reader
                sees it.

        Returns:
            The reply, stripped, or ``None`` for every way of not getting one --
            offline, over the ceiling, screened, refused, or empty. One value,
            because the caller's response to all of them is the deterministic path.
        """

        def execute(model: BaseChatModel, pending: Invocation) -> object | None:
            from src.agent.llm import ask_prose

            return ask_prose(model, pending.system, pending.text, pending.task, sink)

        answer = self._through_the_stack(task, system, text, execute)
        return answer if isinstance(answer, str) else None

    def _through_the_stack(
        self,
        task: Task,
        system: str,
        text: str,
        execute: Callable[[BaseChatModel, Invocation], object | None],
    ) -> object | None:
        """Run one call inside the ceiling, the screen, the retry and the ledger.

        Shared by :meth:`invoke` and :meth:`narrate` so that a prose call is metered,
        capped and screened exactly as a structured one is. Two funnels would be two
        places to forget the budget, and the one that got forgotten would be
        whichever was added second.

        Args:
            task: What the call is for.
            system: The instruction.
            text: The untrusted content.
            execute: What to do with the chosen model. Returns the value, or ``None``
                if the call produced nothing usable.

        Returns:
            Whatever ``execute`` produced, or ``None``.
        """
        # Held across both statements rather than inside `for_task`, because it is the
        # pair that has to be atomic: `for_task` appends the record and this reads it
        # back by position, so a second thread appending in between would hand this call
        # the other one's model name and tier.
        with self._choosing:
            model = self.for_task(task)
            if model is None:
                return None
            chosen = self._chosen[-1]
        call = Invocation(
            task=task,
            tier=chosen.tier,
            model=chosen.slug,
            system=system,
            text=text,
            purpose=task,
        )

        def run(pending: Invocation) -> Outcome:
            answer = execute(model, pending)
            return Outcome(
                value=answer,
                refusal="" if answer is not None else "the model returned nothing usable",
            )

        retries = self._max_retries()
        budget = self.budget if self.budget is not None else CallBudget(DEFAULT_CALL_CEILING)
        return compose(standard_stack(budget, self.ledger, retries), run)(call).value

    def _max_retries(self) -> int:
        """How many extra attempts a transient failure gets.

        Returns:
            The configured count, or none at all when configuration cannot be
            read -- which is the offline case, where nothing will be retried
            because nothing will be called.
        """
        try:
            base = get_settings() if self.settings is None else self.settings
        except Exception:
            return 0
        return base.max_retries

    def selections(self) -> tuple[Selection, ...]:
        """Every choice this pool made, in the order the calls happened.

        Returns:
            The record. This is what lets a report say which model wrote which
            part of it, and what makes a campaign reproducible by something other
            than hope.
        """
        return tuple(self._chosen)

    def describe(self) -> dict[str, object]:
        """Summarise the pool's activity for a trace.

        Returns:
            The slug bound to each tier, the number of calls made, and the
            distinct models that served them.
        """
        return {
            "tiers": {tier: self.slug(tier) for tier in ("fast", "standard", "strong")},
            "calls": len(self._chosen),
            "models_used": sorted({choice.slug for choice in self._chosen}),
            "offline": self.offline,
            "budget_remaining": self.budget.remaining if self.budget else 0,
            **self.ledger.summary(),
        }
