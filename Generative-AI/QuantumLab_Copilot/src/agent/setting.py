"""The settings knob: every dial a user may turn, in one validated object.

The application has two kinds of parameter and they are usually muddled
together. This module keeps them apart while carrying them in one object:

* **Physics** -- the chain being solved, and how hard the solver is allowed to
  work on it. These change the *number*.
* **Model** -- which language model explains the number, how freely it writes,
  and how much it may spend doing so. These change the *prose*, and must never
  change the number.

That separation is the point. Someone who moves the temperature slider and
watches the ground-state energy stay fixed to the last digit has seen the
project's central claim demonstrated rather than asserted: the physics is
computed, and the model only narrates it.

**Not to be confused with** :class:`src.settings.Settings`. The distinction is
worth holding on to, because the two words are nearly the same:

===============================  ==========================================
:class:`src.settings.Settings`   :class:`Setting` (here)
===============================  ==========================================
Read from the environment         Chosen by the user in the interface
Process-wide, read once           Per question, changes as the knob turns
Holds credentials                 Holds no credential, ever
Never shown to a user             Rendered on screen, by design
===============================  ==========================================

:meth:`Setting.applied_to` is the one place they meet, turning a knob position
into the configuration a model client is built from.

**Why an object rather than a pile of Streamlit variables.** Widget state lives
inside a script that cannot be unit-tested, so a bound expressed as
``min_value=2`` in the UI is a bound no test can reach. Expressed here, every
limit is validated by pydantic, tested directly, and enforced identically
whether the caller is the UI, an evaluation run or a future API.

**Ceilings cannot be widened from the knob.** The bounds below mirror the ones
in :mod:`src.settings` and :data:`~src.physics.model.MAX_SITES_STATEVECTOR`
exactly. A knob that could raise the chain-length cap would let a user turn a
principled refusal into an out-of-memory crash, and a knob that could raise the
model-call ceiling would let them turn a runaway loop into a bill. The knob
chooses *within* the safe range; it does not move the edge of it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.physics.model import MAX_SITES_STATEVECTOR, BoundaryCondition, TFIMSpec
from src.rag.ingest import shelf_names
from src.rag.retrieve import (
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_SHARE,
    MAX_ROUNDS,
)
from src.settings import (
    DEFAULT_AUDIENCE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    Audience,
    Settings,
)
from src.tools.papers import MAX_PAPERS

MODEL_CHOICES: tuple[str, ...] = (
    "openai/gpt-5-mini",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
)
"""Chat models offered in the knob, as OpenRouter slugs.

Restricted to models available in **both** the Basic and Advanced course tiers,
so moving between tiers cannot strand the agent on a model it may not call.

Ordered by cost, cheapest first, which puts :data:`~src.settings.DEFAULT_CHAT_MODEL`
at the head and the strongest model measured -- Claude Haiku 4.5 -- one line below
it, a single click away for anyone comparing an answer against a better reasoner.

The first three entries have been exercised against the live gateway on a
structured routing call. The rest are unverified, and a wrong slug fails as an HTTP 400 that
:func:`~src.agent.llm.is_transient` correctly declines to retry. The knob therefore
also accepts a slug typed by hand -- see :attr:`ModelSetting.chat_model` -- so a
wrong entry here is a correctable inconvenience rather than a dead end.
"""

DEFAULT_SITES = 6
"""Chain length a visitor lands on.

Six rather than eight, and the reason is cost rather than physics. Every point of a
sweep at ``L`` builds and iterates a ``2**L``-dimensional Hamiltonian, so eight
spins is four times the work of six for a curve that looks the same at this scale --
and the page recomputes on every slider drag. Six is instant, is even (so the
closed-form solution applies and the independent cross-check exists at all), and is
long enough for the finite-size panel to have something to say.

The cap is still :data:`~src.physics.model.MAX_SITES_STATEVECTOR`. This is where a
visitor starts, not what they are limited to.
"""

FIELD_STEP = 0.25
"""How coarsely the transverse field is offered.

A quarter of a coupling, deliberately blunt. A finer step invites a drag across
dozens of values, each one a full diagonalisation of the chain and a fresh sweep,
and none of the intermediate positions is a question anybody meant to ask. The
physics depends only on ``h / J``, and the interesting values -- zero field, the
critical point at ``h = J``, deep in either phase -- all land on this grid.

Anything in between is still reachable: the Lab reads the same knob, and an
evaluation script constructs a :class:`PhysicsSetting` directly with whatever value
it likes.
"""

MIN_SWEEP_POINTS = 11
MAX_SWEEP_POINTS = 241
"""Bounds on sweep resolution.

Below about a dozen points the curve is a polyline and its minimum is an
artefact of the sampling. Above a couple of hundred, an ``O(L)`` sweep still
returns promptly but the plot gains nothing a reader can see.
"""

DEFAULT_SWEEP_POINTS = 21
"""Points a visitor lands on, across the whole field range.

Few and large rather than many and fine. The range runs to twice the critical
field, so twenty-one points is a step of ``0.1`` in ``h / J`` -- coarse enough
that the whole sweep is a handful of diagonalisations, fine enough that the
curve is a curve and the transition is visible where it belongs.

This was 121, which is where the page went from responsive to hanging: a slider
drag recomputed a hundred-odd eigenproblems per position. The resolution is
still there for anyone who wants it, at the top of the same slider.
"""

SWEEP_POINTS_COSTLY_CEILING = 21
"""Most points swept when each one costs a full diagonalisation.

The cheap closed-form solver is swept at whatever resolution the knob asks for.
Exact diagonalisation is not: at ``L = 8`` every point builds and iterates a
256-dimensional Hamiltonian, and a 241-point sweep behind a slider drag would
make the page feel broken. The knob's resolution is therefore honoured up to
this ceiling and clipped beyond it -- a deliberate, documented clip, not a
silent one.

Equal to :data:`DEFAULT_SWEEP_POINTS`, so the default sweep is never clipped and
the expensive path costs exactly what the cheap one does until somebody asks for
more.
"""

MAX_PASSAGES = 10
"""Most passages one answer may be built from.

Ten is well past useful and short of harmful. The corpus is nineteen notes, so a
question that needs ten sections of it is a question about the whole corpus, and
the passages after the first few are the ones the grader was least sure about.
Raising this does not find better evidence; it lowers the bar for what counts as
evidence, which is why there is a ceiling at all.

The default is :data:`~src.rag.retrieve.DEFAULT_TOP_K`, imported rather than
repeated: a knob whose starting position disagreed with the library's own default
would make an untouched interface answer differently from a direct call.
"""


class PhysicsSetting(BaseModel):
    """The chain to solve, and how hard to work on it.

    Every field here feeds a deterministic calculation. Two identical
    :class:`PhysicsSetting` values produce bit-identical numbers, which is what
    lets the UI cache results and the evaluation suite compare runs across
    sessions.

    Attributes:
        n_sites: Chain length ``L``. Capped at
            :data:`~src.physics.model.MAX_SITES_STATEVECTOR`, because exact
            diagonalisation stores a ``2**L`` state vector and every added site
            quadruples the memory.
        coupling: The Ising coupling ``J``, strictly positive.
        field: The transverse field ``h``, non-negative. The physics depends only
            on the ratio ``h / J``, so both are offered rather than just the
            ratio -- a user who has a specific Hamiltonian in mind can enter it
            as written.
        boundary: ``"periodic"`` (a ring) or ``"open"`` (a segment). Not
            cosmetic: the closed-form free-fermion solution applies only to
            even-length rings, so this choice decides whether an independent
            check exists at all.
        sweep_points: Resolution of the field sweep.
        max_sites_for_costly_sweep: Longest chain swept by repeated
            diagonalisation. Above this the sweep is declined rather than run
            slowly, so the page stays responsive.
    """

    model_config = ConfigDict(frozen=True)

    n_sites: int = Field(default=DEFAULT_SITES, ge=2, le=MAX_SITES_STATEVECTOR)
    coupling: float = Field(default=1.0, gt=0.0, le=10.0)
    field: float = Field(default=1.0, ge=0.0, le=20.0)
    boundary: BoundaryCondition = "periodic"

    sweep_points: int = Field(
        default=DEFAULT_SWEEP_POINTS, ge=MIN_SWEEP_POINTS, le=MAX_SWEEP_POINTS
    )
    max_sites_for_costly_sweep: int = Field(default=8, ge=2, le=MAX_SITES_STATEVECTOR)

    def spec(self) -> TFIMSpec:
        """Build the problem specification the physics layer consumes.

        Returns:
            The equivalent :class:`~src.physics.model.TFIMSpec`. The conversion
            lives here so the UI never constructs a spec itself, and so the
            bounds above are the only place a chain length is checked before the
            physics layer's own validation.
        """
        return TFIMSpec(
            n_sites=self.n_sites,
            coupling=self.coupling,
            field=self.field,
            boundary=self.boundary,
        )

    def points_for(self, *, costly: bool) -> int:
        """Resolution to sweep at, given how expensive one point is.

        Args:
            costly: Whether each point requires a full diagonalisation.

        Returns:
            The requested resolution, clipped to
            :data:`SWEEP_POINTS_COSTLY_CEILING` when every point is expensive.
        """
        if costly:
            return min(self.sweep_points, SWEEP_POINTS_COSTLY_CEILING)
        return self.sweep_points


class ModelSetting(BaseModel):
    """Which model narrates the answer, and the budget it narrates within.

    None of these fields can change a computed number. They are separated from
    :class:`PhysicsSetting` for exactly that reason: the boundary is the claim.

    Attributes:
        audience: How much physics the reader already has. The one dial in this
            group a non-technical user should ever touch, which is why the
            interface puts it in front of the knob rather than inside it.
        chat_model: OpenRouter slug of the chat model. Free text rather than an
            enumeration, so a slug that :data:`MODEL_CHOICES` gets wrong -- or a
            model added after this was written -- can still be reached.
        temperature: Sampling temperature, mid-scale by default and identical to
            :attr:`src.settings.Settings.temperature` -- where the reasoning for
            the number is. Two defaults for one dial is a bug waiting to happen,
            so anyone moving this slider should be moving it away from the
            same value the process starts at.

            What it changes is the wording. It cannot change a number: those come
            from :mod:`src.physics` and are cross-checked before any model sees
            them.
        max_output_tokens: Ceiling on one reply's length, or ``None`` for the
            provider's own default. Set mid-range rather than low: a limit
            reached mid-sentence truncates the answer, and a truncated
            explanation of a verified number is worse than a long one.
        max_model_calls_per_run: Ceiling on model calls while answering one
            question. The loop that does not converge is always found after the
            bill, so the ceiling is set before it.
        max_retries: Attempts after a *transient* failure. Permanent failures
            are never retried -- see :func:`src.agent.llm.is_transient`.
        request_timeout_s: Per-request timeout in seconds.
        requests_per_second: Client-side throttle. Zero disables it. The
            cheapest way to handle a rate limit is not to trip it -- but at one
            per second it was also most of the wait a user sat through, since an
            answer makes about seven sequential calls and the bucket added a
            second between each. See
            :attr:`src.settings.Settings.requests_per_second`.
    """

    model_config = ConfigDict(frozen=True)

    audience: Audience = DEFAULT_AUDIENCE
    chat_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=64, le=8192)
    max_model_calls_per_run: int = Field(default=12, ge=1, le=100)
    max_retries: int = Field(default=3, ge=0, le=10)
    request_timeout_s: float = Field(default=60.0, gt=0.0, le=300.0)
    requests_per_second: float = Field(default=4.0, ge=0.0, le=100.0)


class RetrievalSetting(BaseModel):
    """How much of the notes one answer is built from.

    A group of its own because it is neither of the first two. It cannot change a
    computed number -- the solver never reads a passage -- but unlike
    :class:`ModelSetting` it changes what the answer is *grounded in*, and so
    which citations appear under it. That is a third kind of dial and hiding it
    among the sampling parameters would misfile it.

    Attributes:
        passages: How many passages the answer may use. Read as the ``limit`` of
            :func:`src.rag.retrieve.retrieve`, which is a bound on what survives
            grading rather than on what is searched: the search always overfetches
            (:data:`~src.rag.retrieve.OVERFETCH`), because a grader that can only
            reject needs a pool bigger than the answer. Turning this up therefore
            widens the answer, never the search.
        vector_share: How much of the ranking the vector half decides, the keyword
            half taking the rest -- see
            :data:`~src.rag.retrieve.DEFAULT_VECTOR_SHARE`. At ``1.0`` the search
            is the similarity search this project started with, which is the
            comparison worth being able to run: drag it to one end and a question
            naming an author stops being answerable, because an author's name is in
            the frontmatter and was never embedded.

            A knob rather than a fixed constant because it is the one retrieval
            parameter whose right value depends on the question rather than on the
            corpus, and because it should be possible to *see* the hybrid
            search matter rather than read a claim that it does.
        rounds: How many searches one question may make. ``1`` turns the corrective
            loop off, so a failed search is a failed search; ``2`` lets the grader's
            verdict reformulate the query and try again
            (:data:`~src.rag.retrieve.MAX_ROUNDS`). Worth being adjustable because
            the retry is the most visible piece of machinery here and switching it
            off is how a reader sees what it was doing -- the same question with one
            round and with two is the demonstration.
        shelf: Which knowledge base to search, or ``""`` to leave the choice to the
            router. Named rather than boolean because the interesting failure is a
            *wrong* choice: forcing the hardware shelf for a question about the
            exact solution shows the router earning its place, and it is how someone
            checks that the shelf filter does anything at all.
    """

    model_config = ConfigDict(frozen=True)

    passages: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_PASSAGES)
    vector_share: float = Field(default=DEFAULT_VECTOR_SHARE, ge=0.0, le=1.0)
    rounds: int = Field(default=MAX_ROUNDS, ge=1, le=MAX_ROUNDS)
    shelf: str = ""

    @field_validator("shelf")
    @classmethod
    def _known_shelf(cls, value: str) -> str:
        """Refuse a shelf that does not exist.

        Args:
            value: The shelf name, or ``""`` for the router's choice.

        Returns:
            The value unchanged.

        Raises:
            ValueError: If it names no registered shelf. A typo here would produce
                a filter matching nothing, and a search that quietly returns
                nothing is indistinguishable from a corpus that does not cover the
                question.
        """
        if value and value not in shelf_names():
            raise ValueError(f"unknown shelf {value!r}; registered are {list(shelf_names())}")
        return value


class ToolSetting(BaseModel):
    """Which extra work the agent may ask for while answering.

    A third group, because a tool is neither of the other two: it can change the
    number (a sweep computes new ones) *and* it can reach outside the machine. What
    it cannot do is change how a number is checked -- a tool's result goes through
    the same cross-check as everything else, or it is labelled unverified.

    Attributes:
        use_arxiv: Whether the arXiv lookup is offered to the model at all. On by
            default: the corpus is small, and a question it does not cover is
            better met with a real paper than with a refusal. Off removes the tool
            from the model's list entirely, so no question can talk it into a
            network call.
        use_wikipedia: Whether the encyclopedia lookup is offered. On by default,
            and worth about what an encyclopedia is worth: background, never
            evidence for a number.
        use_web: Whether open web search is offered. **Off by default**, and the
            only source here with that default -- an arbitrary page has had no
            review of any kind. It stays unavailable regardless unless the
            deployment configured a search endpoint.
        use_mcp: Whether tools on a connected MCP server are offered. Off by
            default and unavailable unless a server is configured, since a tool
            list nobody is serving is not a tool list.
        max_papers: Most papers one lookup may return. Small on purpose -- nothing
            has checked them, and each one costs prompt budget.
        suggest_followups: Whether the agent proposes what to ask next. On by
            default and worth one short model call per answer; off makes the
            ``suggest`` node return nothing rather than removing it, so the run has
            the same shape either way.
    """

    model_config = ConfigDict(frozen=True)

    use_arxiv: bool = True
    use_wikipedia: bool = True
    use_web: bool = False
    use_mcp: bool = False
    max_papers: int = Field(default=3, ge=1, le=MAX_PAPERS)
    suggest_followups: bool = True


class Setting(BaseModel):
    """Everything behind the settings knob, both groups together.

    Built by the UI from its widgets, then handed to whatever runs the work.
    Frozen, so a value that has been shown to the user cannot be edited
    downstream: the settings displayed alongside an answer are the settings that
    produced it.

    Attributes:
        physics: Dials that change the number.
        model: Dials that change the prose.
        retrieval: How much of the notes an answer is built from.
        tools: Which extra work the agent may ask for.
    """

    model_config = ConfigDict(frozen=True)

    physics: PhysicsSetting = PhysicsSetting()
    model: ModelSetting = ModelSetting()
    retrieval: RetrievalSetting = RetrievalSetting()
    tools: ToolSetting = ToolSetting()

    def applied_to(self, base: Settings) -> Settings:
        """Return ``base`` with this knob's model choices substituted in.

        The process settings stay untouched. :class:`~src.settings.Settings` is
        frozen precisely so that a trace can be reproduced from the
        configuration it recorded, and a knob that mutated it in place would
        destroy that guarantee. A derived instance keeps both properties: the
        run is configured by the knob, and the configuration is still a frozen
        record.

        Credentials are carried across untouched -- the knob has no access to
        them and no field that could overwrite one.

        Args:
            base: The process settings, normally from
                :func:`~src.settings.get_settings`.

        Returns:
            A new frozen :class:`~src.settings.Settings` differing from ``base``
            only in the fields this knob owns. Passing it to
            :func:`~src.agent.llm.build_chat_model` or
            :func:`~src.agent.llm.build_middleware` is all that applying the
            knob requires; both already take a settings argument.

        Raises:
            pydantic.ValidationError: If a value is out of range. Cannot be
                triggered from the UI, whose widgets are bounded by the same
                numbers, but an evaluation script constructing a
                :class:`Setting` by hand is checked here too.
        """
        overrides = {
            "audience": self.model.audience,
            "chat_model": self.model.chat_model,
            "temperature": self.model.temperature,
            "max_output_tokens": self.model.max_output_tokens,
            "max_model_calls_per_run": self.model.max_model_calls_per_run,
            "max_retries": self.model.max_retries,
            "request_timeout_s": self.model.request_timeout_s,
            "requests_per_second": self.model.requests_per_second,
        }
        return Settings(**{**base.model_dump(), **overrides})

    def changes_from_defaults(self) -> tuple[str, ...]:
        """List the dials that have been moved, in plain language.

        The knob is collapsed by default, which creates one hazard: a user can
        leave the temperature at 1.4, forget, and later read an answer without
        knowing why it reads oddly. The UI shows this list next to the closed
        knob so a non-default configuration is never invisible.

        Returns:
            One short phrase per changed field, empty if everything is at its
            default. Physics fields come first, because those are the ones that
            change the number.
        """
        defaults = Setting()
        changed: list[str] = []
        groups = (
            (self.physics, defaults.physics),
            (self.model, defaults.model),
            (self.retrieval, defaults.retrieval),
            (self.tools, defaults.tools),
        )
        for group, reference in groups:
            for name in type(group).model_fields:
                value = getattr(group, name)
                if value != getattr(reference, name):
                    changed.append(f"{name.replace('_', ' ')} = {value}")
        return tuple(changed)
