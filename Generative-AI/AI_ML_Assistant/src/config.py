"""Central configuration: paths, models, pricing, subjects, and tunable defaults."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "ai_ml_kb"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Embeddings -------------------------------------------------------------
# Two interchangeable backends: local sentence-transformers (free, offline, no API
# tokens) and an OpenAI-compatible API (Text Embedding 3, Qwen3) served via OpenRouter.
# Switching the backend or model changes the vector dimensionality, so the persisted
# index MUST be rebuilt (`make ingest`) after any change here.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # local fallback / experiments
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "api")  # "local" | "api"
API_EMBEDDING_MODEL = os.getenv("API_EMBEDDING_MODEL", "openai/text-embedding-3-large")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", OPENROUTER_BASE_URL)

# --- Chat models ------------------------------------------------------------
# The multi-model switcher (⚙️ Settings) and cost accounting are both driven by this single
# registry. Each entry carries the OpenRouter model id, a friendly picker label, its approximate
# USD price per 1M tokens (input, output), and a one-line positioning note. Ordered cheapest-ish
# first so the default (index 0) is an economical, capable choice.
#
# Prices are APPROXIMATE and static — a portfolio-grade estimate, not a billing source of truth
# (token counts are themselves heuristic; see src/utils.py). Update the figures here as vendor
# pricing moves. `MODEL_PRICES` and `OPENROUTER_MODELS` below are DERIVED from this list, so the
# whole app (estimate_cost, the token tool, A/B + evaluation pickers) stays in step automatically.


@dataclass(frozen=True)
class ModelSpec:
    """One selectable chat model: its id, display label, approximate pricing, and a note.

    ``price_in``/``price_out`` are USD per 1,000,000 tokens. ``id`` is the OpenRouter model
    slug used for the actual API call (the sole exception is the sentinel ``"gemini-native"``,
    routed directly to Google's API via langchain-google-genai when a ``GOOGLE_API_KEY`` is set).
    """

    id: str
    label: str
    price_in: float
    price_out: float
    note: str = ""

    @property
    def prices(self) -> tuple[float, float]:
        """(input, output) USD-per-1M pair, matching the legacy ``MODEL_PRICES`` value shape."""
        return (self.price_in, self.price_out)


# Drawn from the owner's OpenRouter subscription (chat models only). Prices are APPROXIMATE
# placeholders and the newer slugs (GPT-5.x, Claude Haiku 4.5, DeepSeek v4) are best-effort —
# verify against OpenRouter and adjust here as needed. Ordered roughly cheapest → priciest;
# index 0 is the default (a balanced, economical choice).
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "openai/gpt-4o-mini",
        "GPT-4o mini",
        0.15,
        0.60,
        "Cheap, fast, and capable — the balanced default for grounded RAG answers.",
    ),
    ModelSpec(
        "openai/gpt-4.1-nano",
        "GPT-4.1 Nano",
        0.10,
        0.40,
        "Ultra-cheap and quick; best for short, simple lookups.",
    ),
    ModelSpec(
        "openai/gpt-5-nano",
        "GPT-5 Nano",
        0.05,
        0.40,
        "Smallest GPT-5; very low cost with modern reasoning.",
    ),
    ModelSpec(
        "google/gemini-2.5-flash-lite",
        "Gemini 2.5 Flash Lite",
        0.10,
        0.40,
        "Google's cheapest — large context, great for long retrieved passages.",
    ),
    ModelSpec(
        "anthropic/claude-3-haiku",
        "Claude 3 Haiku",
        0.25,
        1.25,
        "Anthropic's cheapest — quick, concise, dependable for short answers.",
    ),
    ModelSpec(
        "openai/gpt-4.1-mini",
        "GPT-4.1 Mini",
        0.40,
        1.60,
        "Strong all-rounder; a step up from 4o-mini for harder questions.",
    ),
    ModelSpec(
        "openai/gpt-5-mini",
        "GPT-5 Mini",
        0.25,
        2.00,
        "Mid-tier GPT-5; good reasoning at a moderate price.",
    ),
    ModelSpec(
        "google/gemini-2.5-flash",
        "Gemini 2.5 Flash",
        0.30,
        2.50,
        "Fast Gemini with solid reasoning and a very large context window.",
    ),
    ModelSpec(
        "deepseek/deepseek-v4-flash",
        "DeepSeek v4 Flash",
        0.28,
        0.88,
        "Open-weight, low cost, strong step-by-step reasoning.",
    ),
    ModelSpec(
        "anthropic/claude-3.5-haiku",
        "Claude 3.5 Haiku",
        0.80,
        4.00,
        "Strong, well-structured writing; pricier output, worth it for quality.",
    ),
    ModelSpec(
        "anthropic/claude-haiku-4.5",
        "Claude Haiku 4.5",
        1.00,
        5.00,
        "Latest small Claude — best Haiku reasoning; higher output cost.",
    ),
    ModelSpec(
        "openai/gpt-5.2",
        "GPT-5.2",
        1.25,
        10.00,
        "Frontier-class reasoning for the hardest questions; highest cost.",
    ),
    ModelSpec(
        "openai/gpt-4o",
        "GPT-4o",
        2.50,
        10.00,
        "Very capable multimodal OpenAI model; premium pricing.",
    ),
    ModelSpec(
        "gemini-native",
        "Gemini 2.5 Flash (native)",
        0.30,
        2.50,
        "Direct Google API (needs GOOGLE_API_KEY) rather than via OpenRouter.",
    ),
)

MODEL_BY_ID: dict[str, ModelSpec] = {m.id: m for m in MODELS}

# Backward-compatible derived views: keep the historical shapes the rest of the app imports.
# `MODEL_PRICES` maps id -> (in, out); `OPENROUTER_MODELS` is the routable-via-OpenRouter subset.
MODEL_PRICES: dict[str, tuple[float, float]] = {m.id: m.prices for m in MODELS}
OPENROUTER_MODELS = [m.id for m in MODELS if m.id != "gemini-native"]

# Small, cheap model used only for classification (query routing + injection screening),
# independent of the user's chosen chat model. Temperature 0, structured-output responses.
# Override via .env for experiments; must be a model that supports json_schema responses.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "openai/gpt-4o-mini")

# UI label -> metadata `subject` values to retrieve from. Overlap docs are shared foundations
# (embeddings, optimization, evaluation) and are included in every single-subject view.
SUBJECTS: dict[str, list[str] | None] = {
    "All subjects": None,
    "Machine Learning": ["ml", "overlap"],
    "Deep Learning": ["dl", "overlap"],
    "AI Engineering": ["ai", "overlap"],
    "Overlap (shared foundations)": ["overlap"],
}

LEVELS = ["Beginner", "Practitioner", "Researcher"]

# Prompting technique -> extra system-prompt instruction (empty string = plain answering).
PROMPT_TECHNIQUES: dict[str, str] = {
    "Standard": "",
    "Chain-of-Thought": (
        "Reason step by step: break the explanation into small numbered steps that build "
        "on each other, then finish with a one-sentence takeaway."
    ),
    "Few-shot": (
        "Before answering, show one short worked example of a closely related concept, "
        "then answer the actual question in the same format."
    ),
    "Socratic": (
        "Teach Socratically: pose 2-3 short guiding questions (answering each yourself in "
        "one or two sentences) that lead the learner to the idea, then summarise."
    ),
    "Analogy-first": (
        "Open with a vivid real-world analogy, then map each part of the analogy onto the "
        "technical concept explicitly before giving the precise definition."
    ),
}

# Response-length preset -> extra system-prompt instruction.
RESPONSE_LENGTHS: dict[str, str] = {
    "Concise": (
        "Keep the whole answer under roughly 120 words: the core idea, one formula or "
        "example at most, no digressions."
    ),
    "Balanced": "",
    "Detailed": (
        "Give a thorough, structured answer: short section headings, the relevant maths, "
        "a worked example, and common pitfalls."
    ),
}


@dataclass
class RagSettings:
    """Tunable retrieval/generation knobs surfaced in the Developer tab."""

    top_k: int = 4
    hybrid_alpha: float = 0.7  # weight of vector similarity vs BM25 (1.0 = pure vector)
    rewrite_query: bool = True
    # Multi-query fusion (RAG-Fusion): off by default so the baseline single-query path is
    # unchanged. When on, the question is expanded into `multi_query_count` diverse search
    # queries (structured output), each retrieved independently, and the results fused by
    # Reciprocal Rank Fusion. Supersedes single-shot `rewrite_query` while active. Trades extra
    # LLM + retrieval calls for materially better recall on under-specified questions.
    multi_query: bool = False
    multi_query_count: int = 3
    # Listwise reranking (RAG stage 2): off by default so first-stage retrieval is unchanged.
    # When on, `retrieve` over-fetches `rerank_candidates` passages, an LLM reorders the whole
    # pool by relevance to the question (structured output), and the top `top_k` are kept.
    # Composes on top of multi-query fusion; trades one extra LLM call for sharper top-k
    # precision. Graceful: on any failure the first-stage order is kept (recall never drops).
    rerank: bool = False
    rerank_candidates: int = 12  # first-stage pool size fed to the reranker (cut to top_k)
    # Maximal Marginal Relevance (MMR) diversity re-selection: off by default so the top_k cut
    # is unchanged. When on, `retrieve` over-fetches `mmr_candidates` passages and greedily
    # selects top_k that balance relevance (the hybrid score) against novelty vs. already-picked
    # passages, so near-duplicate chunks don't crowd out the top_k. `mmr_lambda` weights that
    # trade-off: 1.0 = pure relevance (identical to the plain cut), 0.0 = pure diversity.
    # An alternative to `rerank`, not a companion — `rerank` takes precedence when both are on.
    # Graceful: if candidate embedding fails, falls back to the score-ordered top_k (no regress).
    mmr: bool = False
    mmr_candidates: int = 12  # pool size diversified down to top_k
    mmr_lambda: float = 0.5  # relevance↔diversity weight (1.0 = relevance only)
    filter_difficulty: bool = False
    # Knowledge-base metadata filters (topic / source / year). Empty by default so retrieval
    # is unfiltered and the baseline is unchanged. When any list is non-empty, `retrieve`
    # narrows both retrievers to chunks whose metadata is in the chosen set (see
    # src.rag.retriever.MetadataFilters). Values are drawn from the live index via
    # `KnowledgeBase.facets()`, so the UI only ever offers facets that actually exist.
    filter_topics: list[str] = field(default_factory=list)
    filter_sources: list[str] = field(default_factory=list)
    filter_years: list[str] = field(default_factory=list)
    compare_no_rag: bool = False
    # Corrective RAG: when the KB grades insufficient for an on-domain question,
    # fetch citable passages from external sources before generating. `min_similarity` is the
    # cheap floor — a best retrieval score below it is graded insufficient without spending an
    # LLM grader call. `augment_sources` selects providers from src.core.sources.SOURCE_REGISTRY.
    min_similarity: float = 0.15  # best-score floor: below this, retrieval is "insufficient"
    # Opt-in per invariant; baseline path stays local (no external calls) until toggled on.
    enable_augmentation: bool = False
    augment_sources: list[str] = field(default_factory=lambda: ["arxiv"])
    augment_k: int = 3  # passages to fetch per external source
    # Which orchestration engine runs the routed pipeline: "linear" (plain if/elif branching)
    # or "graph" (LangGraph StateGraph). Same steps, same behaviour — pick one to compare.
    pipeline_engine: str = "linear"
    # Bounded agent loop. Off by default so the four-route pipeline is
    # unchanged. When enabled, the router may send genuinely multi-step questions to a capped
    # plan->act->observe agent that can re-retrieve and call tools across up to
    # `agent_max_steps` iterations before writing a grounded, cited answer.
    enable_agent: bool = False
    agent_max_steps: int = 4
    # Remote MCP (Model Context Protocol) tools. When enabled, tools advertised by the
    # server at `mcp_server_url` are loaded (via langchain-mcp-adapters) and offered to the
    # model alongside the built-in tools. Off by default: it needs that optional dependency
    # and network access to the server. Default URL is DeepWiki — a public, no-auth,
    # read-only server that answers questions about public GitHub repositories.
    enable_mcp: bool = False
    mcp_server_url: str = "https://mcp.deepwiki.com/mcp"
    # LLM generation parameters (surfaced in ⚙️ Settings → Model & generation).
    temperature: float = 0.2
    top_p: float = 1.0  # nucleus sampling; 1.0 = disabled
    max_tokens: int = 0  # cap on the completion length; 0 = let the model decide
    # Per-session request rate limiting (medium optional #9). A token bucket caps how fast
    # one session can fire questions — a cheap guard against runaway API spend and accidental
    # loops, complementing the cost tracking. `burst` is the instantaneous allowance; questions
    # then refill at `per_min` per minute. On by default with a generous demo-friendly budget.
    rate_limit_enabled: bool = True
    rate_limit_burst: int = 8
    rate_limit_per_min: int = 30
    # Structured run logging (medium optional #10). When on, each answered/refused/failed
    # request appends one JSON line to a local run-log for observability (route, outcome,
    # tokens, cost, timings) — never any secret. Best-effort: a logging failure never affects
    # the answer. Path defaults to `logs/runs.jsonl` under the project root.
    enable_run_log: bool = True


# --- Per-visitor credential override ("Use a key" in the top bar) ---------------------------
# A visitor on a deployed Synapse may paste their own OpenRouter key so their questions are
# billed to them rather than to whoever deployed the app. The key itself lives in that browser
# session's server-side store (``src.auth.OWN_KEY_STATE``); what lives here is only the choice
# *the current script run* should use, published by :func:`src.auth.apply_runtime_keys` before
# any model is built.
#
# Thread-local rather than a plain module global, and that distinction is the whole point.
# ``src/core`` imports this module and must never import Streamlit, so the override cannot be a
# ``st.session_state`` read; but a module global is shared by every session the server is
# handling at once, which on a deployment would let one visitor's pasted key bill another
# visitor's question. Streamlit runs each session's script in its own thread, so a thread-local
# is scoped to exactly one run of one session.
#
# Set on every run, including to ``None`` when the visitor has no key of their own, so a thread
# reused for a later run never inherits the previous one's credential.
_RUNTIME_KEYS = threading.local()


def set_runtime_key(name: str, value: str | None) -> None:
    """Override ``name`` for this script run only, or clear the override when ``value`` is falsy.

    Args:
        name: Environment variable name, e.g. ``"OPENROUTER_API_KEY"``.
        value: The key to use for this run, or ``None``/``""`` to fall back to the environment.
    """
    overrides: dict[str, str] = getattr(_RUNTIME_KEYS, "overrides", None) or {}
    if value:
        overrides[name] = value
    else:
        overrides.pop(name, None)
    _RUNTIME_KEYS.overrides = overrides


def clear_runtime_keys() -> None:
    """Drop every override this thread is carrying, so a reused thread starts from nothing.

    The guarantee above — that no run inherits the previous one's credential — held only for
    keys some caller remembered to re-state every run, which was one of the two on offer.
    Clearing the table first makes it a property of the mechanism instead: a key added to
    :func:`src.auth.apply_runtime_keys` later cannot forget to opt in.
    """
    _RUNTIME_KEYS.overrides = {}


def _key(name: str) -> str | None:
    """Return the run's override for ``name`` if one is set, else the environment value."""
    overrides: dict[str, str] = getattr(_RUNTIME_KEYS, "overrides", None) or {}
    return overrides.get(name) or os.getenv(name)


def openrouter_api_key() -> str | None:
    """Return the OpenRouter API key: this run's pasted override, else the environment/.env."""
    return _key("OPENROUTER_API_KEY")


def google_api_key() -> str | None:
    """Return the Google (Gemini) API key: this run's pasted override, else environment/.env."""
    return _key("GOOGLE_API_KEY")
