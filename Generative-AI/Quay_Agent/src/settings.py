"""Runtime configuration, read once from the environment.

Every credential, model identifier and network timeout in the project comes
from here. Nothing else reads ``os.environ``, and no module hard-codes a model
name -- swapping the chat model is an edit to ``.env``, not to code.

Secrets are held as :class:`~pydantic.SecretStr`, whose ``repr`` is ``**********``.
That is deliberate defence in depth: settings objects end up in tracebacks,
LangSmith traces and debugger frames, and a plain ``str`` would leak the key
into all three. Call ``.get_secret_value()`` only at the boundary where the
client is constructed.

The ``.env`` file is read by pydantic-settings, never by this project's code.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
"""OpenRouter's OpenAI-compatible endpoint.

OpenRouter implements the OpenAI wire protocol, which is why the project can
use ``langchain-openai`` against it with no vendor-specific client.
"""

Audience = Literal[
    "beginner",
    "entrepreneur",
    "software, no physics",
    "practitioner",
    "quantum computing engineer",
    "researcher",
]
"""Who the answer is being written for.

Six readers rather than a smooth dial, because the differences that matter are
qualitative: whether a term needs unpacking, whether an equation helps or
interrupts, whether the interesting part is the cost or the derivation.

Ordered by how much of *this* domain is assumed, which is not the same as ordering
them by expertise. ``software, no physics`` describes somebody fluent in complexity
classes, sampling error and floating point who has never met a Hamiltonian;
``quantum computing engineer`` wants the circuit depth and the error budget and no
gloss at all; ``entrepreneur`` wants the decision, the cost and the condition under
which it changes, and none of the mechanism. Naming the assumed *background* rather
than a job title is what keeps this a scale: two readers with the same title can
want opposite answers, and a scale ordered by seniority would rank one of them
below the other for no reason a text can act on.

This changes **only the prose**. The number is identical at every level and is
cross-checked before any of this is consulted: an explanation written for a
beginner is not a less accurate one.
"""

ReasoningEffort = Literal["", "minimal", "low", "medium", "high"]
"""How much a reasoning model is asked to think before it answers.

Empty means "send nothing", which is the only safe value for a model that has no
such parameter -- see :data:`src.agent.llm.REASONING_MODELS` for how the two are
kept from meeting.

Minimal by default, and this is a latency setting rather than a quality one. Every
model call this project makes fills a small typed schema: which route, which
passages are relevant, one paragraph of prose. Measured on the routing call,
``openai/gpt-5-mini`` spent 14.6 seconds and 669 output tokens to fill four short
fields, and 3.3 seconds to fill them identically with this set to minimal. Seven
calls to an answer made that the difference between a 99-second reply and a
23-second one, for the same decisions.
"""

DEFAULT_AUDIENCE: Audience = "practitioner"
"""Level assumed when nobody has chosen one.

The middle setting: an answer that explains too much is skimmed, while one that
explains too little is misread, and the middle is the only default that fails
gracefully in both directions.
"""

DEFAULT_TEMPERATURE = 0.6
"""Sampling temperature when nobody has chosen one.

Mid-scale, for the reason :attr:`Settings.temperature` sets out at length: the
temperature buys wording and cannot reach a number, because every number in an
answer is computed in :mod:`src.physics` and cross-checked before a model is
called. Named here rather than written twice, since
:class:`src.ui.setting.Model` starts its slider at the same value and two
defaults for one dial is a bug waiting to happen.
"""

DEFAULT_MAX_OUTPUT_TOKENS = 4096
"""Ceiling on one reply, when nobody has chosen one.

Mid-range: high enough that neither an answer nor a code draft reaches it, low
enough to stop a model that has begun repeating itself. Both halves matter --
truncating a verified explanation mid-sentence is the worse failure of the two,
which is why this is not set tight.
"""

DEFAULT_CHAT_MODEL = "openai/gpt-5-mini"
"""Chat model used when the environment names none.

Chosen for cost, because the default is what development runs on. A single
``make evals`` is about 160 model calls, and the suite is run far more often
than it is shipped: at 0.25/2.00 USD per million tokens this is a quarter of
Claude Haiku 4.5's input rate and under half its output rate, for behaviour
measured to be equivalent on the decisions that matter.

Measured against the routing prompt on six questions, this model and Haiku 4.5
agreed on five. They differ in both directions, which is why the cheaper one is
defensible rather than merely cheaper: this model asks back for a bare
*compute the magnetisation* where Haiku routes it straight to the solver, and it
is the only one of the three tested that routed *teach me about IBM quantum
technologies* to a search instead of declining it.

A named constant rather than a literal in the field default because
:mod:`src.ui.setting` offers the same value as the settings knob's default,
and two independently written copies of a model slug are two things to forget to
change together. Anyone who wants the stronger model names it in the environment
or picks it from the knob -- see :func:`src.agent.model_selection.selectable_slugs`.
"""

GUEST_CHAT_MODEL = "google/gemini-2.5-flash-lite"
"""Model served to a visitor who has not signed in and has no key of their own.

**Why a public deployment needs this at all.** A demo is only a demo if a
stranger can press Ask. But every model call on a hosted app is billed to whoever
deployed it, so an open Ask button on the strong tier is an invitation to spend
somebody's month in an afternoon. The two usual answers are both bad: take the
button away and the demo demonstrates nothing, or leave it and hope.

**Why this model and not a free one.** :data:`GUEST_FREE_MODEL` was the first
choice and it does not work here -- see that constant for the reason, which is an
account policy rather than a bad slug. This is the cheapest confirmed model in the
catalogue at 0.10/0.40 USD per million tokens, and a guest campaign is about six
calls of a few thousand tokens: on the order of **$0.002 a question**, so a
thousand visitors cost a couple of dollars. That is a bounded, known number, which
is a better thing to deploy on than an endpoint that may or may not answer.

Set ``GUEST_CHAT_MODEL`` in the environment to override it -- which is how a
deployment moves to the free endpoint once its account allows one.

**And it matters less here than it would almost anywhere else.** Every number in
an answer is computed by :mod:`src.physics` and cross-checked before a model is
consulted; the model chooses wording, a route and a depth. So the guest tier costs
a visitor some quality of prose and none of the arithmetic -- which is the
project's central claim, arriving as a product decision rather than a paragraph.
"""

GUEST_FREE_MODEL = "google/gemma-4-31b-it:free"
"""A genuinely free endpoint, and what it takes to use one.

Zero per token in both directions -- verified against the gateway's own index by
the slug verifier rather than assumed from the ``:free`` suffix,
because a suffix is a naming convention and a price is a fact.

**It is not the default, because calling it returns 404 on this account:** *"No
endpoints available matching your guardrail restrictions and data policy."* Every
free endpoint on the gateway does, and the cause is an account-level privacy
setting rather than anything in this repository -- free providers are reached only
by accounts that have opted into letting prompts be used for training, and this one
has not. That is the *better* default for an account that also runs paid calls, so
the setting is left alone and named here instead of quietly worked around.

To use it: opt in at ``openrouter.ai/settings/privacy``, then set
``GUEST_CHAT_MODEL=google/gemma-4-31b-it:free``. The opt-in applies to the whole
account, so it is a decision about every prompt the deployment sends, not only a
guest's.

Free endpoints are also rate-limited upstream and can be withdrawn without notice.
Both are survivable: :data:`src.agent.model_selection.FALLBACK_TIERS` covers a tier
whose slug will not build, and a campaign that can reach no model at all still
answers from the deterministic path -- with every number intact and template prose.
"""

DEFAULT_PROMOTED_PATH = "promoted"
"""Where notes fetched from an external source are kept, relative to the root.

A *generated* directory, and top-level for that reason: everything under ``data/``
is written by hand and committed, and mixing machine-fetched text into it would
make "is this file authored or downloaded?" a question you answer by reading the
file rather than by looking at its path.

The name is the claim it makes. A note only arrives here after it has been
screened and found clean, so what sits in this directory has been *promoted* from
a raw search result into something the index is allowed to see. Anything that
failed the screen was never written.
"""

DEFAULT_CORPUS_PATH = "data/corpus"
"""Where the knowledge-base documents live, relative to the project root.

Read by ``make ingest`` and by nothing else at runtime: once the index is built,
the corpus is not consulted again. Keeping it in the repository is what makes an
ingest reproducible -- anyone can read exactly the text the agent retrieves
from.

``data/`` holds *only* committed documents. Generated artefacts live outside it,
so whether a file is version-controlled is answered by its path rather than by
reading :file:`.gitignore`.
"""

DEFAULT_VECTOR_STORE_PATH = "chroma_db"
"""Where the Chroma index is persisted, relative to the project root.

Generated, never authored, and excluded from version control. It sits *beside*
``data/`` rather than inside it, because ``data/`` holds only committed documents:
a top-level directory makes "generated" visible in a directory listing instead of
hiding it behind an ignore rule. Deleting it is always safe -- the next
``make ingest`` rebuilds it from the corpus, though re-embedding costs API calls.
"""

DEFAULT_COLLECTION = "tfim-literature"
"""Base name of the Chroma collection.

The name actually used is :attr:`Settings.collection_name`, which appends the
embedding model. See that property for why.
"""

CHROMA_MAX_COLLECTION_NAME = 63
"""Longest collection name Chroma accepts.

Chroma validates this itself and raises on a longer one. We truncate rather than
let that happen, because the failure would surface deep inside ingestion with a
message about a name the caller never typed.
"""


class Settings(BaseSettings):
    """Everything the application needs from its environment.

    Field names map to upper-case environment variables of the same name, so
    ``openrouter_api_key`` is populated from ``OPENROUTER_API_KEY``. Values in
    the real environment win over values in ``.env``, which is what lets CI and
    the tests override a setting without touching the file.

    The instance is frozen: configuration is read at startup and cannot drift
    mid-run, so a trace can be reproduced from the settings it recorded.

    Attributes:
        openrouter_api_key: Credential for the OpenRouter gateway. Required --
            construction fails loudly at startup rather than at the first call.
        openrouter_base_url: The OpenAI-compatible endpoint to call.
        chat_model: Model backing the agent's reasoning and tool calls, and the
            standard tier of :mod:`src.agent.model_selection`.
        fast_model: Model for the short calls a person is waiting on --
            choosing a knowledge base, grading passages, rewriting a query.
            Empty falls back to the tier default.
        strong_model: Model for the closing report, the one call whose output
            a person reads in full. Empty falls back to the tier default.
        embedding_model: Model backing the retrieval index. Changing this
            invalidates an existing Chroma collection: the stored vectors were
            produced by the old model and are not comparable with the new one.
        embedding_base_url: Endpoint serving embeddings, or ``None`` to use the
            chat endpoint. Separate because chat and embeddings need not come
            from the same gateway, and OpenRouter's coverage of the
            ``/embeddings`` route is not something to assume: this is the one
            setting to change when an ingest fails with a 404 on that path.
        embedding_api_key: Credential for that endpoint, or ``None`` to reuse
            the OpenRouter key. Set it when ``embedding_base_url`` points
            somewhere the OpenRouter key is not valid.
        audience: How much physics the reader already has. Changes the wording
            and nothing else -- see :data:`Audience`.
        temperature: Sampling temperature, mid-scale by default. No number in an
            answer is sampled: every energy, gap and curve comes from
            :mod:`src.physics` and the model writes the paragraph around it, so
            temperature buys wording and cannot touch a result. Zero costs
            something real, because greedy decoding on the *decisions* makes a
            wrong first token unrecoverable -- the same question takes the same
            wrong turn every time. Set it to zero for a reproducible transcript;
            the numbers are identical either way.
        max_output_tokens: Ceiling on one reply. High enough that no answer or
            code draft reaches it, and low enough to bound a model that has
            started repeating itself.
        reasoning_effort: How hard a reasoning model thinks before answering, or
            ``""`` to send no such parameter. Sent only to the models that accept
            it -- see :data:`src.agent.llm.REASONING_MODELS`.

            Left at the bottom of its range, and the cost was measured rather
            than guessed: mid-scale, the routing call took 14.6 s and 669 output
            tokens to fill four short fields that ``minimal`` filled identically
            in 3.3 s. Seven calls to an answer, so the choice is a 23-second reply
            or a 99-second one.
        request_timeout_s: Per-request timeout in seconds. Sized from what the
            calls here actually take rather than left at a round number: fast-tier
            calls land in 1-3 s, reading the problem in 3.4-5.0 s, and the report,
            the longest, in about 6 s. Thirty is roughly five times the slowest of
            those, and it is the number that bounds the *tail*. A timeout counts as
            transient, so a stalled call is retried ``max_retries`` times at the
            full timeout each: at sixty, one stuck call could hold a page for four
            minutes with only the node trail moving, and a prose answer makes about
            seven such calls in a row. Raise it if a legitimate call is ever cut
            off; the log records the attempt that failed.
        max_retries: Retries after a *transient* failure before a model call is
            allowed to fail. Permanent failures are never retried -- see
            :func:`src.agent.llm.is_transient`.
        requests_per_second: Client-side throttle on outbound model calls. The
            cheapest way to handle a rate limit is not to trip it: the free
            OpenRouter tier is strict, and an evaluation run fires far faster
            than a human ever would. Zero disables throttling.

            Four rather than one, because one was measured to be most of the
            wait: an answer makes about seven calls, and a one-per-second bucket
            adds six seconds of doing nothing to every question. The burst this
            guards against is an evaluation run, not one answer, which is why the
            setting stays and why :file:`Makefile` leaves it overridable.
        max_model_calls_per_run: Ceiling on model *attempts* in a single answer,
            retries included -- see :class:`src.agent.middleware.CallBudget`. It
            exists because a tool-calling agent that loops costs real money and the
            loop is always found after the bill.

            Sized above the honest worst case rather than near it. A prose answer
            can legitimately need eleven attempts before any retry: reading the
            problem, an intent tie, choosing a shelf, expanding the query, ordering
            the passages, grading and re-querying when a round keeps nothing, two
            rounds of tool consultation, the explanation, and the follow-ups. At
            twelve that run had room for a single retry in the whole campaign, so a
            rate-limited gateway -- the one condition that causes retries -- could
            exhaust the ceiling before the explanation call was made and degrade the
            answer with nothing on screen to say why. Twenty-four admits every
            honest call plus a retry on each, and still stops a genuine runaway far
            short of what an uncounted-retry ceiling allowed.
            The model-selection test pins it against the task list, so
            adding a call site that pushes the honest count up fails there rather
            than in somebody's answer.
        parallel_model_calls: Whether one answer may have several model calls in
            flight at once. The three at the front of the graph do not depend on
            each other, so on, the wait is the slowest of them rather than their
            sum. A switch rather than a decision because the sequential path makes
            a trace read top to bottom. It changes no answer, only when the calls
            were issued -- see :mod:`src.agent.prefetch`.
        eval_workers: How many evaluation cases ``make evals`` answers at once,
            four by default. It adds no load on the gateway: the throttle is
            process-wide (:func:`src.agent.llm._shared_limiter`), so what overlaps
            is the waiting, and past the point where workers keep the throttle
            busy the extra ones queue. Cases are independent and are put back into
            case order before grading, so the scorecard does not depend on this.
            ``EVAL_WORKERS=1 make evals`` runs them one at a time.
        checkpoint_path: SQLite file holding conversation state, relative to the
            project root. ``None`` keeps state in memory, which is what tests
            and one-shot scripts want.
        memory_path: JSONL file holding the agent's memory -- the recent turns it
            recalls and the ratings it learns a reading level from. ``None``
            disables memory entirely, which is a supported mode rather than a
            degraded one: see :mod:`src.agent.memory`. A plain text file because a
            person should be able to read everything the agent remembers about
            them without a database client.
        corpus_path: Directory of hand-written knowledge-base documents, read
            only by ingestion.
        promoted_path: Directory of notes fetched from an external source and
            admitted by the screen. Read by ingestion alongside ``corpus_path``.
            Setting it to ``None`` switches external material off entirely, which
            is the configuration to use when only reviewed text may be indexed.
        vector_store_path: Directory the Chroma index is persisted to.
        chroma_collection: Base name of the collection. Prefer
            :attr:`collection_name`, which qualifies it by embedding model.
        langsmith_api_key: Optional tracing credential.
        langsmith_project: Project name traces are filed under.
        langsmith_tracing: Master switch for tracing. Off by default so that a
            test run never posts to an external service.
    """

    # ``env_file`` is this project's dotenv step. There is deliberately no
    # ``load_dotenv()`` call anywhere: pydantic-settings reads ``.env`` itself, so a
    # second reader would be two mechanisms for one job -- and unlike load_dotenv it
    # validates as it reads and never mutates ``os.environ``.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # --- credentials -----------------------------------------------------
    openrouter_api_key: SecretStr
    openrouter_base_url: str = DEFAULT_BASE_URL

    # --- models ----------------------------------------------------------
    # Both defaults are available in the Basic and Advanced tiers alike, so a
    # tier change cannot silently strand the agent or the vector store.
    chat_model: str = DEFAULT_CHAT_MODEL
    guest_chat_model: str = GUEST_CHAT_MODEL
    fast_model: str = ""
    strong_model: str = ""
    embedding_model: str = "openai/text-embedding-3-small"
    # Both default to None, meaning "same gateway as chat". Overriding them is
    # how the retrieval half moves to another provider without the chat half
    # noticing -- and the reverse, which is why they are two fields.
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None

    # --- call behaviour --------------------------------------------------
    audience: Audience = DEFAULT_AUDIENCE
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=64, le=8192)
    reasoning_effort: ReasoningEffort = "minimal"
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    requests_per_second: float = Field(default=4.0, ge=0.0, le=100.0)
    max_model_calls_per_run: int = Field(default=24, ge=1, le=100)
    parallel_model_calls: bool = True
    eval_workers: int = Field(default=4, ge=1, le=16)

    # --- persistence -----------------------------------------------------
    checkpoint_path: str | None = ".checkpoints/quay.sqlite"
    memory_path: str | None = ".memory/quay.jsonl"

    # --- knowledge base --------------------------------------------------
    corpus_path: str = DEFAULT_CORPUS_PATH
    promoted_path: str | None = DEFAULT_PROMOTED_PATH
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH
    chroma_collection: str = DEFAULT_COLLECTION

    # --- observability ---------------------------------------------------
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "quay"
    langsmith_tracing: bool = False

    @field_validator("openrouter_api_key")
    @classmethod
    def _reject_a_blank_key(cls, value: SecretStr) -> SecretStr:
        """Fail on an empty credential instead of on a puzzling HTTP 401.

        An exported-but-empty variable is a common shell mistake and satisfies
        "the variable is set". Catching it here names the actual problem.

        Args:
            value: The credential as read from the environment.

        Returns:
            The credential, unchanged.

        Raises:
            ValueError: If the credential is empty or only whitespace.
        """
        if not value.get_secret_value().strip():
            raise ValueError("OPENROUTER_API_KEY is set but empty")
        return value

    @property
    def tracing_enabled(self) -> bool:
        """Whether LangSmith tracing is both switched on and credentialed."""
        return self.langsmith_tracing and self.langsmith_api_key is not None

    @property
    def collection_name(self) -> str:
        """Collection to read and write, with the embedding model in the name.

        Vectors are only comparable with vectors from the same embedding model.
        Point a new model at an existing collection and nothing breaks loudly --
        the search just returns nonsense, because it is measuring distances
        between coordinates in two unrelated spaces.

        Naming the model makes that mistake impossible rather than merely
        documented: a switch addresses a different, empty collection, so the next
        ingest rebuilds it and the symptom becomes "no results yet" instead of
        confident rubbish.

        Returns:
            A name Chroma accepts -- alphanumerics, hyphens and underscores,
            trimmed to :data:`CHROMA_MAX_COLLECTION_NAME` and never left ending
            in a separator.
        """
        suffix = re.sub(r"[^A-Za-z0-9]+", "-", self.embedding_model).strip("-").lower()
        return f"{self.chroma_collection}-{suffix}"[:CHROMA_MAX_COLLECTION_NAME].rstrip("-_")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructing them on first call.

    Cached so that ``.env`` is parsed once and every module observes the same
    configuration. Tests that need a different configuration should build a
    :class:`Settings` directly rather than clearing this cache.

    Returns:
        The validated settings.

    Raises:
        pydantic.ValidationError: If a required variable is missing or a value
            is out of range. Raised at startup, by design.
    """
    return Settings()  # type: ignore[call-arg]  # values come from the environment


def get_settings_or_none() -> Settings | None:
    """Return the process-wide settings, or ``None`` if they cannot be built.

    The offline mode this project documents needs exactly one of these. Every
    fallback in the agent is reached by discovering that no model is available, and
    the commonest way for none to be available is an absent or empty
    ``OPENROUTER_API_KEY`` -- which :class:`Settings` refuses, correctly, because a
    process that means to call a provider should fail at startup rather than
    halfway through a campaign.

    The trouble is that a campaign also reads settings for things that have nothing
    to do with credentials: which model slug a tier resolves to, how long to wait,
    how many passages to retrieve. Those calls sit inside graph nodes, so a missing
    key stopped the run at the second node with a validation error instead of
    taking the deterministic path the fallbacks exist to provide. The mode was
    written, tested, and unreachable.

    So the rule is: anything that *needs* a credential asks for the settings and is
    entitled to fail, and anything that merely needs a *preference* asks here and
    copes with ``None`` by using its own default.

    Returns:
        The settings, or ``None`` when they are invalid or incomplete.
    """
    try:
        return get_settings()
    except Exception:
        _logger.info(
            "settings_unavailable",
            extra={"detail": "falling back to defaults; a campaign will run offline"},
        )
        return None
