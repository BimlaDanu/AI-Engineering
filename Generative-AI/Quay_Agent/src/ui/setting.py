r"""What the two knobs hold, and the bounds they hold it between.

Declarative only: three frozen records and their limits, with no Streamlit anywhere
near them. That is what lets a test assert that the bound a visitor meets in the
interface is the same bound the rest of the project enforces, rather than a number
somebody typed into a widget call and forgot.

Two groups, and the split is the argument
------------------------------------------

:class:`Physics` changes **what is computed**. Move the chain length and the energy
moves. :class:`Model` changes **how it is described**. Move the reading level, or
switch the language model off entirely, and the energy does not move by one digit --
because no model computed it.

Keeping them visibly apart is not tidiness. It is the project's central claim made
checkable in one gesture: a visitor who suspects the numbers are being produced by a
language model can switch the language model off and watch every number stay exactly
where it was. A single undifferentiated settings panel would make that experiment
impossible to perform and the claim impossible to believe.

Which of the two is empty on a given page is itself informative. The Ising chain and
the Lab read :class:`Physics` and ignore :class:`Model` completely, and they say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from src.agent.model_selection import DEFAULT_TIER_SLUGS
from src.agent.reading import Reading
from src.hardware.devices import device_names
from src.rag.ingest import shelf_names
from src.rag.retrieve import MAX_ROUNDS
from src.settings import DEFAULT_AUDIENCE, Audience, Settings, get_settings

Boundary = Literal["open", "periodic"]
"""``"open"`` for a segment, ``"periodic"`` for a ring."""

MIN_SITES = 2
MAX_SITES = 16
"""Bounds on the chain length a visitor may ask for.

Two because a shorter chain has no bond. Sixteen because it is the highest number
any method here accepts (:data:`~src.physics.model.MAX_SITES_SPARSE`), and a knob
whose top position every panel refuses is a knob that only teaches disappointment.

The default is twelve and that is where the interesting range is
(:data:`~src.physics.model.WORKING_SITES`). Sixteen is the ``4 x 4`` square and
the exact solver's ceiling; the twenty this used to offer cost twelve seconds and
1.9 GB per answer and is gone.
"""

MAX_CIRCUIT_DEPTH = 12
"""Deepest circuit the interface will offer.

Well past the point where every real machine here has run out of signal, which is
the point: a control that stopped at the useful limit would hide the limit.
"""

SHOT_CHOICES: tuple[int, ...] = (10**6, 10**7, 5 * 10**7, 10**8, 10**9, 10**10)
"""Measurement budgets offered, as a ladder rather than a slider.

Shots scale as the inverse square of accuracy, so the interesting differences are
between orders of magnitude and a linear slider would spend most of its travel
between two budgets nobody can tell apart.

The default is high on purpose. One energy reading on a ten-spin chain at one per
cent accuracy costs tens of millions of measurements, and an optimisation needs
dozens of readings -- so a budget that sounds generous refuses the very first
configuration and the campaign concludes without having run anything. A default that
makes the honest answer "we could not afford to try" is a default that hides the
machinery behind a bookkeeping accident.
"""

DEFAULT_SITES = 10
DEFAULT_DEPTH = 2
DEFAULT_DEVICE = "linear"
DEFAULT_SHOTS = 10**9
DEFAULT_PRECISION = 1e-3
DEFAULT_MAX_CALLS = 40


def process_default(name: str) -> Any:
    """Read one field's default off :class:`~src.settings.Settings`.

    Every transport dial below opens **where the process actually runs**, rather
    than at a number retyped here. That is not tidiness: drawing the sidebar is what
    *sets* the value a campaign runs under, so a slider that opens at a literal the
    configuration no longer uses does not merely mislabel itself -- it silently
    reconfigures the run to the stale number the moment anybody looks at it. This
    knob was opening its temperature at ``0.2`` while the process default was
    ``0.6``, which is exactly that failure.

    Read from the class rather than an instance because :class:`Settings` refuses to
    construct without a credential, and this module is imported on a checkout that
    has none.

    Args:
        name: The field on :class:`Settings`.

    Returns:
        Its declared default, unnarrowed -- pydantic stores defaults as ``Any`` and
        each caller below states the type it expects.
    """
    return Settings.model_fields[name].default


def credential_available() -> bool:
    """Whether a language model could be called at all on this checkout.

    :class:`~src.settings.Settings` refuses to construct without a credential, so a
    successful read *is* the check -- there is no separate "is the key there?"
    question to ask, and asking one would be a second answer that can disagree with
    the first.

    Nothing is read out of the settings and nothing is logged. The credential's
    presence is the only fact that leaves this function.

    Returns:
        ``True`` when configuration resolves, ``False`` when it does not -- which is
        the checkout with no ``.env``, and the mode the test suite renders pages in.
    """
    try:
        get_settings()
    except Exception:
        return False
    return True


def default_offline() -> bool:
    """Where the language-model switch opens.

    This is the bug the switch used to have, and the same one
    :func:`process_default` was written for one field further down: the toggle was
    drawn at a hard-coded ``True``, so an application with a working credential
    opened with its model switched off and answered every question on the
    deterministic path. Nothing was broken and nothing said so -- the campaign ran,
    the numbers were right, the prose was templated, and the only visible symptom was
    that the agent sounded like a form letter.

    Offline stays a first-class mode: it is what a checkout with no key gets, it is
    what the suite runs in, and it is one click away for a sceptic who wants to watch
    every number stay put with the model off. What it stops being is the default on a
    machine that can do better.

    Returns:
        ``True`` when there is no credential to call with, ``False`` when there is.
    """
    return not credential_available()


DEFAULT_TEMPERATURE = float(process_default("temperature"))
DEFAULT_MAX_OUTPUT_TOKENS = int(process_default("max_output_tokens"))
DEFAULT_MAX_RETRIES = int(process_default("max_retries"))
DEFAULT_REQUESTS_PER_SECOND = float(process_default("requests_per_second"))
DEFAULT_REQUEST_TIMEOUT_S = float(process_default("request_timeout_s"))

MAX_OUTPUT_TOKEN_BOUNDS = (64, 8192)
"""What :class:`~src.settings.Settings` will accept as a reply-length cap.

Restated as a pair so a widget can be built from it, and pinned to the same numbers
by a test -- a slider whose top end exceeds what validation permits is a control that
produces an exception instead of a setting.
"""

MAX_RETRY_BOUND = 10
"""Most retries the configuration layer permits after a transient failure."""

MAX_REQUESTS_PER_SECOND = 10.0
"""Fastest outbound rate this interface offers.

Well below the hundred per second configuration would accept. The gateway's free
tier is strict, and a control whose top end trips a rate limit is a control that
produces a failed campaign rather than a faster one.
"""

REQUEST_TIMEOUT_BOUNDS = (5.0, 300.0)
"""Shortest and longest per-request timeout offered, in seconds."""

MAX_SEARCH_ROUNDS = MAX_ROUNDS
"""Most searches one question may be given, the first included.

Re-exported under the interface's own name so a widget can be built from the
retrieval layer's own ceiling rather than from a number retyped beside it. Offering
a third round would offer a loop the layer will not run.
"""

TIERS: tuple[str, ...] = ("fast", "standard", "strong")
"""The three model tiers, in cost order.

Named here as well as in the agent so that a selector can offer them without the
interface importing the agent's internals, and so a typo in one place is a
validation error rather than a silently ignored override.
"""

MAX_CALLS_CHOICES: tuple[int, ...] = (10, 20, 40, 80)
"""Call ceilings offered. Forty is roughly six campaigns' worth of a full depth ladder."""

DEFAULT_TIERS: tuple[tuple[str, str], ...] = tuple(
    (tier, slug)
    for tier, slug in sorted(DEFAULT_TIER_SLUGS.items(), key=lambda p: TIERS.index(p[0]))
)
"""Which model serves each tier unless somebody says otherwise.

The real slugs rather than a placeholder meaning "whatever is configured". A
selector showing "default" tells a reader nothing they can check, and the first
thing anybody wants to know about a model setting is which model it is. These are
the same values the agent falls back to, read from it rather than restated here, so
the two cannot drift.
"""

PRECISION_CHOICES: tuple[float, ...] = (1e-2, 1e-3, 1e-4)
"""Target accuracies offered, as a ladder.

Measurements scale as the inverse square of this, so each step down the list costs a
hundred times as much -- which is exactly why it is offered as three rungs rather
than as a slider that would spend most of its travel between two budgets nobody can
tell apart.
"""

MAX_LONGITUDINAL = 1.0
"""Largest field along the coupling axis the interface offers."""


@dataclass(frozen=True, slots=True)
class Physics:
    """What is computed. Every field here moves the number.

    Attributes:
        n_sites: Length of the chain, one qubit per magnet.
        boundary: Whether the chain is a segment or a ring.
        coupling: The Ising coupling :math:`J`.
        field: The transverse field :math:`h`. The ratio to the coupling is what
            matters, and at :math:`h/J = 1` the chain is critical and hardest.
        longitudinal: The field :math:`g` along the coupling axis. Zero leaves the
            chain exactly solvable, which is what makes a closed-form check possible;
            non-zero is the one knob held in reserve for the whole project, because it
            costs no two-qubit depth and it is what makes the feasibility question
            non-trivial.
        precision: Target accuracy per magnet the shot arithmetic is priced against.
        depth: Circuit layers.
        device: Which machine circuits are priced against, by name.
        shots: The measurement budget a campaign may spend.
    """

    n_sites: int = DEFAULT_SITES
    boundary: Boundary = "open"
    coupling: float = 1.0
    field: float = 1.0
    longitudinal: float = 0.0
    precision: float = DEFAULT_PRECISION
    depth: int = DEFAULT_DEPTH
    device: str = DEFAULT_DEVICE
    shots: int = DEFAULT_SHOTS

    def __post_init__(self) -> None:
        """Reject a setting the rest of the project would refuse anyway.

        Checked here rather than left to the widget bounds, because anything that
        calls this directly -- a test, a script -- never meets a widget, and a guard
        that lives only in a slider is absent exactly when somebody is debugging.

        Raises:
            ValueError: If the chain, the depth, the couplings or the budget are
                outside what this interface offers, or the device is unknown.
        """
        if not MIN_SITES <= self.n_sites <= MAX_SITES:
            raise ValueError(
                f"n_sites must be between {MIN_SITES} and {MAX_SITES}, got {self.n_sites}"
            )
        if not 0 <= self.depth <= MAX_CIRCUIT_DEPTH:
            raise ValueError(f"depth must be between 0 and {MAX_CIRCUIT_DEPTH}, got {self.depth}")
        if self.coupling <= 0.0:
            raise ValueError(f"coupling must be positive, got {self.coupling}")
        if self.field < 0.0:
            raise ValueError(f"field cannot be negative, got {self.field}")
        if self.longitudinal < 0.0:
            raise ValueError(f"longitudinal field cannot be negative, got {self.longitudinal}")
        if self.precision not in PRECISION_CHOICES:
            raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {self.precision}")
        if self.shots <= 0:
            raise ValueError(f"shots must be positive, got {self.shots}")
        if self.device not in device_names():
            raise ValueError(f"unknown device {self.device!r}; expected one of {device_names()}")

    @property
    def ratio(self) -> float:
        r"""The ratio :math:`h/J`, which is the only combination that matters.

        One is the critical point: the energy gap closes, correlations reach across
        the whole chain, and every approximate method has its hardest time. Away
        from it in either direction the problem is easy, which is why a suite that
        sampled evenly would flatter the machinery.
        """
        return self.field / self.coupling

    def as_reading(self) -> Reading:
        """These dials in the form the agent fills a silent request from.

        The five chain dials had no way through to a campaign: they drew the circuit
        at the top of the page, they ran the Lab, and the agent read the chain out of
        the question alone -- so a reader who set the length to ten and then asked a
        question that named no length got a verdict about six, under a picture of ten.
        A dial that changes nothing is worse than no dial.

        A :class:`~src.agent.reading.Reading` rather than the five numbers, because
        that is the type the agent already lays a follow-up over a question with, and
        the precedence wanted here is the same one: what the sentence said wins, then
        what the conversation said, then this.

        Returns:
            The chain as a reading. The longitudinal field is left out when it is
            zero, which is off rather than set -- naming a term to report that it is
            switched off is how a two-term problem comes to look like a three-term one.
        """
        return Reading(
            n_sites=self.n_sites,
            boundary=self.boundary,
            coupling=self.coupling,
            field=self.field,
            longitudinal=self.longitudinal or None,
        )


@dataclass(frozen=True, slots=True)
class Model:
    """How the answer is described. No field here moves a number.

    Attributes:
        offline: Run with no language model at all. Opens at
            :func:`default_offline`, which is ``True`` only when there is no
            credential to call with -- a dial has to open where the process actually
            runs, and one that opened *off* on a configured machine turned every
            answer into a form letter without saying so. A supported mode rather than
            a degraded one either way: every decision that changes a number is
            arithmetic, so what is lost is the prose and nothing else.
        audience: Who the prose is written for.
        temperature: How much the model is allowed to vary its wording. Reachable
            precisely so that a sceptic can turn it up and watch the numbers not
            move.
        tiers: Model slugs replacing the configured default for each tier, keyed by
            tier name. An empty mapping takes the defaults.

            Three tiers rather than one model, because the calls are not alike. The
            cheap tier serves everything a person is waiting on -- which shelf to
            search, whether a passage earned its place -- and those are short
            classifications where a larger model buys nothing but latency. The strong
            tier serves the two calls whose *output* is the deliverable rather than a
            step towards one: the prose explanation, and the drafted code. The
            feasibility report is not among them -- it is assembled from templates
            over numbers the arithmetic produced, which is why a campaign can write
            one with no model at all.
        max_calls: Ceiling on model calls in one campaign. A wall rather than a
            warning: a budget expressed as an instruction in a prompt is a
            suggestion.
        max_output_tokens: Ceiling on the length of one reply, or ``None`` to leave
            the provider's own default alone. A cap is the difference between a
            model that rambles at your expense and one that stops.
        max_retries: Extra attempts after a *transient* failure. Permanent failures
            are never retried, so raising this buys resilience to a flaky gateway
            and nothing else.
        requests_per_second: Client-side throttle on outbound calls. Zero switches
            throttling off. It bounds the *rate* calls leave at, not how many may be
            waiting at once -- it is the dial that stops an evaluation sweep from
            tripping a rate limit, not one that makes a single answer slower.
        request_timeout_s: How long one call may take before it is abandoned.
        parallel_calls: Whether the three model calls at the front of a campaign --
            restating the question, reading what kind of answer it wants, and reading
            the chain's numbers out of it -- go out together rather than one after
            another. None of them reads another's answer, so the wait is the slowest
            of the three instead of their sum, which is most of the pause before any
            visible work begins.

            On by default, and a dial rather than a decision because the sequential
            order is worth being able to get back to: it makes a trace read top to
            bottom. It moves no decision and changes no answer -- every node still
            runs and still decides on the same evidence -- so a sceptic can turn it
            off and watch the numbers not move, exactly as with ``temperature``.
    """

    offline: bool = field(default_factory=default_offline)
    audience: Audience = DEFAULT_AUDIENCE
    temperature: float = DEFAULT_TEMPERATURE
    tiers: tuple[tuple[str, str], ...] = DEFAULT_TIERS
    max_calls: int = DEFAULT_MAX_CALLS
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS
    max_retries: int = DEFAULT_MAX_RETRIES
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    parallel_calls: bool = True

    def __post_init__(self) -> None:
        """Reject settings no gateway would accept.

        Every bound here is the one :class:`~src.settings.Settings` enforces. Checked
        twice on purpose: a value refused at the knob names the dial that is wrong,
        while the same value refused four screens later inside a campaign names a
        pydantic field nobody moved.

        Raises:
            ValueError: If the temperature is outside :math:`[0, 2]`, the call
                ceiling is not positive, a tier is named that does not exist, or a
                transport dial is outside what the configuration layer accepts.
        """
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")
        if self.max_calls < 1:
            raise ValueError(f"max_calls must be at least 1, got {self.max_calls}")
        for tier, _ in self.tiers:
            if tier not in TIERS:
                raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
        low, high = MAX_OUTPUT_TOKEN_BOUNDS
        if self.max_output_tokens is not None and not low <= self.max_output_tokens <= high:
            raise ValueError(
                f"max_output_tokens must be between {low} and {high} or absent, "
                f"got {self.max_output_tokens}"
            )
        if not 0 <= self.max_retries <= MAX_RETRY_BOUND:
            raise ValueError(
                f"max_retries must be between 0 and {MAX_RETRY_BOUND}, got {self.max_retries}"
            )
        if not 0.0 <= self.requests_per_second <= MAX_REQUESTS_PER_SECOND:
            raise ValueError(
                f"requests_per_second must be between 0 and {MAX_REQUESTS_PER_SECOND}, "
                f"got {self.requests_per_second}"
            )
        shortest, longest = REQUEST_TIMEOUT_BOUNDS
        if not shortest <= self.request_timeout_s <= longest:
            raise ValueError(
                f"request_timeout_s must be between {shortest} and {longest}, "
                f"got {self.request_timeout_s}"
            )

    def overrides(self) -> dict[str, str]:
        """The tier slugs to hand a model pool.

        Stored as a tuple of pairs so the record stays hashable and frozen, and
        rendered back to a mapping here because that is what the pool takes. Empty
        entries are dropped rather than passed through: an override of ``""`` is how
        a selector says "leave it alone", and forwarding it would replace a working
        default with nothing.

        Returns:
            Tier name to slug, for every tier that names one.
        """
        return {tier: slug for tier, slug in self.tiers if slug}

    def applied_to(self, base: Settings) -> Settings:
        """Overlay this knob's transport dials onto a configuration.

        The dials on this record are worth nothing until something reads them, and
        for a long while nothing did: the temperature slider moved, the label under
        the knob announced that it had moved, and every campaign went on calling the
        gateway at the configured default. A control that changes nothing is worse
        than an absent one -- it is read as a control that did not work, and it makes
        every other control on the panel suspect.

        This is the single place they are connected. Everything downstream --
        :func:`src.agent.llm.build_chat_model`, the rate limiter, the retry
        middleware -- already reads them off :class:`~src.settings.Settings`, so
        handing the campaign's model pool a copy with these replaced wires all four
        at once and leaves no second path to keep in step.

        Args:
            base: The process configuration to copy.

        Returns:
            A copy carrying this knob's temperature, reply cap, retry count,
            throttle, timeout and call ceiling. Nothing else is touched -- the
            credential, the endpoint and the corpus paths are not a visitor's to set.
        """
        return base.model_copy(
            update={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "max_retries": self.max_retries,
                "requests_per_second": self.requests_per_second,
                "request_timeout_s": self.request_timeout_s,
                "max_model_calls_per_run": self.max_calls,
            }
        )


DEFAULT_PASSAGES = 4
"""How many passages a search keeps by default.

Four is what the campaign has always used. Named here so the dial that moves it has
somewhere to move from, and so a reader can see the default without opening the
retrieval layer.
"""

MAX_PASSAGES = 12
"""The most passages one answer may cite.

The dial's ceiling, and lower than the twenty the validator accepts, because the two
bounds answer different questions: the validator says what the retrieval layer will
not break on, and this says what is worth offering. Above about a dozen the search is
not finding better passages, it is admitting ones the grader nearly dropped -- so the
dial stops where the returns do, and a caller constructing a setting by hand can
still go further.
"""

DEFAULT_VECTOR_SHARE = 0.6
"""How much of a search is meaning-matching rather than keyword-matching.

Above a half because paraphrase is the common case -- a question rarely uses the
notes' own words. Not much above, because the keyword half is what finds a passage
that names an author or an arXiv number, and those are the searches whose failure a
reader notices.
"""

SHELF_CHOICES: tuple[str, ...] = ("decide for me", *shelf_names())
"""Where a search may be pointed.

``decide for me`` is first and is the default: the router picks a shelf from the
question's vocabulary and widens on the next round if it guessed wrong, which beats a
person guessing once. The named shelves are read off the corpus itself, so a renamed
shelf changes this list rather than leaving a dead option on a page.
"""


@dataclass(frozen=True, slots=True)
class Search:
    """What the agent is allowed to look up, and how hard it looks.

    A third group beside the physics and the model, because retrieval is the third
    thing that changes an answer and it had no dial at all: the campaign searched the
    notes and never went outside, both hard-coded, and a reader who wanted to know
    what the agent would say *without* its notes had no way to find out.

    Attributes:
        corpus: Whether to search the project's own notes. Left on almost always --
            the keyword half needs no model, so it returns citations even offline --
            and worth turning off exactly once, to see what the answer rests on
            without them.
        external: Whether to reach outside the corpus when the notes come back
            empty. Off by default: it is a network call on somebody's key, and what
            it returns is real text nobody has read, which is why anything fetched
            this way is cited as unreviewed rather than mixed in.
        passages: How many passages a search keeps.
        vector_share: How much of the search is meaning-matching rather than
            keyword-matching, between nothing and everything.
        shelf: Which knowledge base to search first, or ``"decide for me"``.
        rounds: How many searches one question gets, the first included. Two lets a
            search that found nothing be reformulated from the grader's own verdict
            and tried again; one switches that corrective loop off, which is how a
            reader sees what it was doing.
        followups: Whether the agent proposes what to ask next. On by default; the
            dial exists because a suggestion is a model call, and somebody metering
            a campaign should be able to see the number without it.
    """

    corpus: bool = True
    external: bool = False
    passages: int = DEFAULT_PASSAGES
    vector_share: float = DEFAULT_VECTOR_SHARE
    shelf: str = "decide for me"
    rounds: int = MAX_ROUNDS
    followups: bool = True

    def __post_init__(self) -> None:
        """Refuse a position no search could be run at.

        Raises:
            ValueError: If a field is outside the range the retrieval layer accepts.
                Checked here rather than at the search, because a knob that accepts
                an impossible value and fails four screens later is a knob that
                blames the wrong page.
        """
        if not 1 <= self.passages <= 20:
            raise ValueError(f"passages must be between 1 and 20, got {self.passages}")
        if not 0.0 <= self.vector_share <= 1.0:
            raise ValueError(f"vector share must be between 0 and 1, got {self.vector_share}")
        if self.shelf not in SHELF_CHOICES:
            raise ValueError(f"unknown shelf {self.shelf!r}; choose from {SHELF_CHOICES}")
        if not 1 <= self.rounds <= MAX_ROUNDS:
            raise ValueError(f"rounds must be between 1 and {MAX_ROUNDS}, got {self.rounds}")

    @property
    def shelves(self) -> tuple[str, ...]:
        """The shelves to search, as the retrieval layer wants them.

        Returns:
            One shelf, or nothing at all -- which is how the layer is told to decide
            for itself. Empty and "search everywhere" are the same instruction here,
            and keeping them the same is what lets the router widen a search that
            found nothing.
        """
        return () if self.shelf == "decide for me" else (self.shelf,)


@dataclass(frozen=True, slots=True)
class Setting:
    """Both knobs, as one value a page can be handed.

    Attributes:
        physics: What is computed.
        model: How it is described.
        search: What it is allowed to look up.
    """

    # Factories rather than shared instances, and that is load-bearing for
    # `Model.offline`: a bare `Model()` default is evaluated once, when this module
    # is imported, so it would freeze the credential check at import time and hand a
    # page rendered after the key was removed a knob that still says "model on".
    physics: Physics = field(default_factory=Physics)
    model: Model = field(default_factory=Model)
    search: Search = field(default_factory=Search)

    def changes_from_defaults(self) -> tuple[str, ...]:
        """List what has been moved away from its default position.

        A shut knob hides a moved dial, and somebody who turned the temperature up
        an hour ago and forgot needs telling before they read a strangely worded
        answer. Reported as a list rather than a flag so the label can say how many
        and the caption can say which.

        Returns:
            One phrase per changed field, in a stable order. Empty when nothing has
            been touched.
        """
        default = Setting()
        moved: list[str] = []
        for label, mine, theirs in (
            ("chain length", self.physics.n_sites, default.physics.n_sites),
            ("boundary", self.physics.boundary, default.physics.boundary),
            ("coupling", self.physics.coupling, default.physics.coupling),
            ("field", self.physics.field, default.physics.field),
            ("longitudinal field", self.physics.longitudinal, default.physics.longitudinal),
            ("precision", self.physics.precision, default.physics.precision),
            ("circuit depth", self.physics.depth, default.physics.depth),
            ("machine", self.physics.device, default.physics.device),
            ("shot budget", self.physics.shots, default.physics.shots),
            ("language model", self.model.offline, default.model.offline),
            ("reading level", self.model.audience, default.model.audience),
            ("temperature", self.model.temperature, default.model.temperature),
            ("models", self.model.overrides(), default.model.overrides()),
            ("call ceiling", self.model.max_calls, default.model.max_calls),
            ("reply cap", self.model.max_output_tokens, default.model.max_output_tokens),
            ("retries", self.model.max_retries, default.model.max_retries),
            (
                "requests per second",
                self.model.requests_per_second,
                default.model.requests_per_second,
            ),
            ("request timeout", self.model.request_timeout_s, default.model.request_timeout_s),
            ("calls in parallel", self.model.parallel_calls, default.model.parallel_calls),
            ("search the notes", self.search.corpus, default.search.corpus),
            ("search outside", self.search.external, default.search.external),
            ("passages kept", self.search.passages, default.search.passages),
            ("vector share", self.search.vector_share, default.search.vector_share),
            ("shelf", self.search.shelf, default.search.shelf),
            ("searches per question", self.search.rounds, default.search.rounds),
            ("follow-up questions", self.search.followups, default.search.followups),
        ):
            if mine != theirs:
                moved.append(f"{label} = {mine}")
        return tuple(moved)

    def with_physics(self, **changes: object) -> Setting:
        """Return a copy with some physics fields replaced.

        Args:
            **changes: Fields of :class:`Physics` to change.

        Returns:
            The new setting. Validation runs again, so an impossible override is
            refused here rather than several screens later.
        """
        return replace(self, physics=replace(self.physics, **changes))  # type: ignore[arg-type]

    def with_model(self, **changes: object) -> Setting:
        """Return a copy with some model fields replaced.

        Args:
            **changes: Fields of :class:`Model` to change.

        Returns:
            The new setting.
        """
        return replace(self, model=replace(self.model, **changes))  # type: ignore[arg-type]

    def with_search(self, **changes: object) -> Setting:
        """Return a copy with some search fields replaced.

        Args:
            **changes: Fields of :class:`Search` to change.

        Returns:
            The new setting. Validation runs again, so a shelf that no longer exists
            is refused here rather than returning an empty search.
        """
        return replace(self, search=replace(self.search, **changes))  # type: ignore[arg-type]
