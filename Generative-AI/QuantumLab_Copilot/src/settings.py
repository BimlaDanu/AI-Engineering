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

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
"""OpenRouter's OpenAI-compatible endpoint.

OpenRouter implements the OpenAI wire protocol, which is why the project can
use ``langchain-openai`` against it with no vendor-specific client.
"""

Audience = Literal["beginner", "practitioner", "researcher"]
"""How much physics the reader already has.

Three levels rather than a smooth dial, because the differences that matter are
qualitative: whether a term needs unpacking, whether an equation helps or
interrupts, and whether the finite-size caveats are the interesting part or the
boring part. This changes *only* the prose. The number is the same at every level
and is cross-checked before any of this is consulted -- an explanation aimed at a
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
:class:`src.agent.setting.ModelSetting` starts its slider at the same value and two
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
:mod:`src.agent.setting` offers the same value as the settings knob's default,
and two independently written copies of a model slug are two things to forget to
change together. Anyone who wants the stronger model names it in the environment
or picks it from the knob -- see :data:`src.agent.setting.MODEL_CHOICES`.
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
        chat_model: Model backing the agent's reasoning and tool calls.
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
        temperature: Sampling temperature, mid-scale by default.

            It used to be zero, on the argument that variation is a liability
            here. Half of that argument was never true: **no number in an answer
            is sampled at all.** The energies, the gap, the magnetisation and
            every curve come from :mod:`src.physics`, and the one thing a model
            is asked to do with them is write the paragraph around them --
            :func:`src.agent.graph.build_answer` narrates what was already
            computed, and the deterministic text is right there beside it in
            :func:`src.agent.graph.plain_answer`. So the sampling temperature
            buys wording, and it cannot touch a result.

            Zero also cost something real. Greedy decoding on the *decisions* --
            routing, the next action, whether a passage is relevant -- makes a
            wrong first token unrecoverable: the same question takes the same
            wrong turn every time, and a run that reads well is not a run that
            reasoned. Mid-scale leaves the reasoning able to move.

            What keeps this honest is that nothing here rests on the wording. A
            claim is verified by two independent methods or it is caveated as
            unverified, and that gate is code -- see
            :mod:`src.verification.cross_check`. Set it to zero for a
            reproducible transcript; the numbers will be identical either way.
        max_output_tokens: Ceiling on the length of one reply. Mid-range rather
            than unset: a ceiling this high is not reached by an answer or by a
            code draft, and it is reached by a model that has started repeating
            itself, which is the runaway this bounds. See
            :data:`src.agent.drafting.MAX_CODE_CHARACTERS` for the same argument
            applied to a draft.
        reasoning_effort: How hard a reasoning model thinks before answering, or
            ``""`` to send no such parameter. Sent only to the models that accept
            it -- see :data:`src.agent.llm.REASONING_MODELS`.

            **The one parameter here deliberately left at the bottom of its
            range**, against the rule the other two now follow, because the cost
            was measured rather than guessed: at the middle of this scale the
            routing call took 14.6 seconds and 669 output tokens to fill four
            short fields, and 3.3 seconds to fill them identically at
            ``minimal``. Seven calls to an answer, so the choice is a 23-second
            reply or a 99-second one. Raise it if a decision looks badly made --
            it is one word here -- but raise it knowing that is the price.
        request_timeout_s: Per-request timeout in seconds.
        max_retries: Retries after a *transient* failure before a model call is
            allowed to fail. Permanent failures are never retried -- see
            :func:`src.agent.llm.is_transient`.
        requests_per_second: Client-side throttle on outbound model calls. The
            cheapest way to handle a rate limit is not to trip it: the free
            OpenRouter tier is strict, and an evaluation run fires far faster
            than a human ever would. Zero disables throttling.

            Four rather than one, because one was measured to be most of the wait.
            An answer makes about seven calls -- screen, route, decide, grade,
            compose, caveat, suggest -- and a one-per-second bucket adds six
            seconds of doing nothing to every question, roughly a third of the
            time a user spends watching the spinner. The calls are sequential
            anyway: this throttle only ever slows the *gaps* between them, so the
            burst it was written to prevent is not one a single answer can
            produce. An evaluation run still can, which is why the setting stays
            and why :file:`Makefile` leaves it overridable from the environment.
        max_model_calls_per_run: Ceiling on model calls in a single answer. A
            tool-calling agent that loops costs real money, and the loop is
            always discovered after the bill.
        eval_workers: How many evaluation cases ``make evals`` answers at once.
            Four by default. A case is about seven sequential model calls and
            there are twenty-six cases, so one at a time is a suite that takes
            minutes of mostly waiting.

            Concurrency here does not mean more load on the gateway. The
            throttle is process-wide and shared by every model this application
            builds (:func:`src.agent.llm._shared_limiter`), so the outbound rate
            stays at ``requests_per_second`` however many workers run; what
            overlaps is the time each worker spends waiting for a reply. That is
            why raising this is safe and why raising it far does nothing: past
            the point where the workers can keep the throttle busy, extra
            workers queue.

            The cases are independent -- each has its own conversation thread
            and its own throwaway memory file -- and results are put back into
            case order before grading, so the scorecard does not depend on this
            value. ``EVAL_WORKERS=1 make evals`` restores one at a time when a
            trajectory is easier to read in sequence.
        checkpoint_path: SQLite file holding conversation state, relative to the
            project root. ``None`` keeps state in memory, which is what tests
            and one-shot scripts want.
        memory_path: JSONL file holding the agent's memory -- the recent turns it
            recalls and the ratings it learns a reading level from. ``None``
            disables memory entirely, which is a supported mode rather than a
            degraded one: see :mod:`src.agent.memory`. A plain text file because a
            person should be able to read everything the agent remembers about
            them without a database client.
        corpus_path: Directory of knowledge-base documents, read only by
            ingestion.
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
    request_timeout_s: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    requests_per_second: float = Field(default=4.0, ge=0.0, le=100.0)
    max_model_calls_per_run: int = Field(default=12, ge=1, le=100)
    eval_workers: int = Field(default=4, ge=1, le=16)

    # --- persistence -----------------------------------------------------
    checkpoint_path: str | None = ".checkpoints/quantumlab.sqlite"
    memory_path: str | None = ".memory/quantumlab.jsonl"

    # --- knowledge base --------------------------------------------------
    corpus_path: str = DEFAULT_CORPUS_PATH
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH
    chroma_collection: str = DEFAULT_COLLECTION

    # --- observability ---------------------------------------------------
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "quantumlab-copilot"
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
