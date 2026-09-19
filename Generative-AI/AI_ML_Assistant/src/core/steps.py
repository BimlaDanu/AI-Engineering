"""
Shared pipeline steps — the single source of truth for both orchestration
engines.

It provides the same routed pipeline through two orchestration styles:

* :mod:`src.core.linear` — a lightweight typed-router engine using plain Python
  ``if/elif`` branching.
* :mod:`src.core.graph` — a LangGraph ``StateGraph`` using the same steps as
  nodes with conditional edges.

Both engines use the step functions defined here, so their behaviour stays
identical; only the orchestration differs. Do not duplicate logic in engines —
add or modify behaviour in this module.

Each step follows ``(state, deps) -> dict`` and returns only the fields to merge
into the pipeline state. The state is plain data
(:class:`PipelineState`), while request-specific dependencies (LLM, knowledge
base, settings, router, timer) are provided through
:class:`PipelineDeps` instead of being stored in state.

Pipeline flow:

    START
      |
      v
    screen
      |
      +-----------------------------+
      |                             |
      v                             v
   refuse                         route
                                    |
          +-------------------------+-------------------------+-------------------------+
          |                         |                         |                         |
          v                         v                         v                         v
      retrieve()              tools_only()                meta()                  agent()
          |
          v
       grade()
          |
       +--+----------------+
       |                   |
       v                   v
  generate()          augment()
                           |
                           v
                       generate()

Each step has the signature ``(state, deps) -> dict`` and returns just the keys it wants
merged into the pipeline state. State is plain data (:class:`PipelineState`); the
per-request dependencies (LLM, knowledge base, settings, router, timer) live in
:class:`PipelineDeps` and are passed in rather than stored on the state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from src.core.agent import AgentStepRecord, run_agent_loop
from src.core.router import QueryRouter
from src.core.schemas import InjectionVerdict, RelevanceGrade, RouteDecision
from src.core.sources import build_sources
from src.generation import (
    AnswerResult,
    answer_meta,
    answer_question,
    answer_without_rag,
)
from src.rag.retriever import (
    KnowledgeBase,
    MetadataFilters,
    RetrievedChunk,
    expand_queries,
    reciprocal_rank_fusion,
    rerank_chunks,
    rewrite_query,
)
from src.utils import UsageMeter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.config import RagSettings
    from src.core.service import AnswerRequest

# Learner level -> chunk `difficulty` metadata value used for optional level filtering.
LEVEL_TO_DIFFICULTY: dict[str, str] = {
    "Beginner": "beginner",
    "Practitioner": "intermediate",
    "Researcher": "advanced",
}

REFUSAL = (
    "I'm a learning assistant for **machine learning, deep learning, and AI** topics, so "
    "I'll sincerely pass on that one. Ask me about models, training, transformers, RAG, "
    "evaluation, or the maths behind them — or try the Tools tab."
)
INJECTION_REFUSAL = (
    "⚠️ That message looks like a prompt-injection attempt, so I won't process it. "
    "Please ask a plain question about machine learning or AI."
)

# Route name -> the step that handles it. Shared by both engines' branching logic.
ROUTE_TO_STEP: dict[str, str] = {
    "knowledge": "retrieve",
    "tool": "tools_only",
    "meta": "meta",
    "off_topic": "refuse",
    "agent": "agent",
}


class PipelineState(TypedDict, total=False):
    """The typed data contract flowing through the pipeline — one shape for both engines.

    It is a ``TypedDict`` on purpose: every step returns a small dict of *only* the keys it
    writes, and both engines merge those in with last-write-wins — plain ``dict.update`` in the
    linear engine, LangGraph's default channel merge in the graph engine. That shared merge is
    exactly what keeps the two engines identical (see ``tests/test_engine_parity.py``).

    The fields fall into four groups, in the order the pipeline fills them: the **inputs**
    seeded once by :func:`initial_state`, the routing **decisions**, the retrieval/generation
    **products**, and the **terminal** answer. ``total=False`` means every field is optional —
    only the one route a question takes ever writes its own fields, so writes never collide.
    """

    # inputs (seeded by the service)
    question: str
    history: list[tuple[str, str]]
    subjects: list[str] | None
    level: str
    style: str
    # decisions
    verdict: InjectionVerdict
    decision: RouteDecision
    grade: RelevanceGrade
    # retrieval / generation products
    search_query: str
    search_queries: list[str]
    reranked: bool
    diversified: bool
    chunks: list[RetrievedChunk]
    augmented: bool
    augment_sources_used: list[str]
    result: AnswerResult
    no_rag: AnswerResult
    agent_trajectory: list[AgentStepRecord]
    # terminal
    text: str
    ok: bool


@dataclass
class PipelineDeps:
    """Per-request dependencies the steps use (kept out of the plain-data state)."""

    llm: BaseChatModel
    kb: KnowledgeBase | None
    settings: RagSettings
    router: QueryRouter
    req: AnswerRequest
    timer: Any  # src.utils.StageTimer (untyped here to avoid a hard import)
    # Tools advertised to the model this request: built-ins plus any remote MCP tools. None
    # lets `answer_question` fall back to the built-in ALL_TOOLS (used by offline tests).
    tools: list[Any] | None = None
    # Accumulates token usage from auxiliary main-model calls (query rewrite/expansion,
    # reranking) so the request total no longer undercounts them; folded in by the service.
    usage: UsageMeter = field(default_factory=UsageMeter)
    # Accumulates token usage from the classification calls (routing, injection screen,
    # relevance grading). These run on the separate, cheaper ROUTER_MODEL, so they are metered
    # apart from ``usage`` and priced at ROUTER_MODEL by the service — never mixed into the
    # answer model's price.
    router_usage: UsageMeter = field(default_factory=UsageMeter)


# state helpers
# Three small typed accessors so the steps talk to the state through one documented place,
# instead of scattering raw string keys and repeated ``or []`` defaults through every branch.
# `initial_state` is the *only* place a request becomes pipeline state (its four inputs); the
# reader helpers give the optional list fields a clean empty default in one spot.
def initial_state(
    *,
    question: str,
    history: list[tuple[str, str]],
    subjects: list[str] | None,
    level: str,
    style: str,
) -> PipelineState:
    """Seed the pipeline for one request — the single point where a request becomes state.

    Both engines start from exactly this dict, so keeping its construction here (rather than
    inline in the service) makes the pipeline's *inputs* one documented contract. Every other
    field in :class:`PipelineState` is produced by the steps as the request flows through.
    """
    return {
        "question": question,
        "history": history,
        "subjects": subjects,
        "level": level,
        "style": style,
    }


def history_of(state: PipelineState) -> list[tuple[str, str]]:
    """The prior conversation turns, or ``[]`` on the first turn (an optional-field read)."""
    return state.get("history") or []


def chunks_of(state: PipelineState) -> list[RetrievedChunk]:
    """The context passages gathered so far, or ``[]`` on routes that never retrieve."""
    return state.get("chunks") or []


# steps
def screen(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Screen the question for prompt injection (regex + LLM classifier)."""
    return {"verdict": deps.router.screen_injection(state["question"], meter=deps.router_usage)}


def route(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Classify the question into a handling route (the ``agent`` route is opt-in)."""
    return {
        "decision": deps.router.classify(
            state["question"],
            history_of(state),
            allow_agent=deps.settings.enable_agent,
            meter=deps.router_usage,
        )
    }


def retrieve(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Query translation + hybrid retrieval (knowledge path only).

    Two query-translation modes, both off-by-default-safe. When ``multi_query`` is on the
    question is expanded into several diverse queries (RAG-Fusion) that are each retrieved
    and fused by Reciprocal Rank Fusion; otherwise the single-shot ``rewrite_query`` path (or
    the raw question) runs exactly as before. ``search_queries`` records every query used so
    the trace can show the expansion.

    When ``rerank`` is on a wider candidate pool (``rerank_candidates``) is retrieved and an
    LLM reorders it by relevance to the original question before the top-``top_k`` cut; when
    off, exactly ``top_k`` chunks are retrieved as before. Both extra stages degrade to the
    baseline on failure, so first-stage recall is never reduced.

    Any user-chosen knowledge-base facets (topic/source/year) are passed to every ``kb.search``
    as a :class:`~src.rag.retriever.MetadataFilters`; an empty filter leaves retrieval
    unchanged, so the baseline is preserved when no facet is selected.
    """
    settings = deps.settings
    history = history_of(state)
    queries = [state["question"]]
    if deps.kb and settings.multi_query:
        with deps.timer.measure("query_expand"):
            queries = expand_queries(
                deps.llm,
                state["question"],
                history,
                n=settings.multi_query_count,
                meter=deps.usage,
            )
    elif settings.rewrite_query and deps.kb:
        with deps.timer.measure("query_rewrite"):
            queries = [rewrite_query(deps.llm, state["question"], history, meter=deps.usage)]

    chunks: list[RetrievedChunk] = []
    reranked = False
    diversified = False
    if deps.kb:
        # `.get` (not `[...]`) mirrors the defensive lookup in generation._level_style: a stale
        # session level or a non-UI caller passing an unmapped level yields no difficulty filter
        # rather than a KeyError deep inside retrieval.
        difficulty = LEVEL_TO_DIFFICULTY.get(deps.req.level) if settings.filter_difficulty else None
        filters = MetadataFilters(
            topics=tuple(settings.filter_topics),
            sources=tuple(settings.filter_sources),
            years=tuple(settings.filter_years),
        )
        # Over-fetch a wider pool only when a stage-2 selector (rerank or MMR) will trim it;
        # otherwise fetch exactly top_k so the baseline path is byte-for-byte unchanged.
        if settings.rerank:
            fetch_k = max(settings.rerank_candidates, settings.top_k)
        elif settings.mmr:
            fetch_k = max(settings.mmr_candidates, settings.top_k)
        else:
            fetch_k = settings.top_k
        with deps.timer.measure("retrieval"):
            ranked_lists = [
                deps.kb.search(
                    q,
                    subjects=state.get("subjects"),
                    difficulty=difficulty,
                    k=fetch_k,
                    alpha=settings.hybrid_alpha,
                    filters=filters,
                )
                for q in queries
            ]
            chunks = (
                reciprocal_rank_fusion(ranked_lists, limit=fetch_k)
                if len(ranked_lists) > 1
                else (ranked_lists[0] if ranked_lists else [])
            )
        # Stage-2 selectors are alternatives, not companions: rerank (LLM precision) takes
        # precedence over MMR (embedding diversity) when both are enabled; otherwise the plain
        # top_k cut. All three keep first-stage recall — they only reorder/trim the pool.
        if settings.rerank and len(chunks) > 1:
            with deps.timer.measure("rerank"):
                chunks = rerank_chunks(
                    deps.llm,
                    state["question"],
                    chunks,
                    top_n=settings.top_k,
                    meter=deps.usage,
                )
            reranked = True
        elif settings.mmr and len(chunks) > 1:
            with deps.timer.measure("mmr"):
                chunks = deps.kb.mmr_select(chunks, k=settings.top_k, lambda_=settings.mmr_lambda)
            diversified = True
        else:
            chunks = chunks[: settings.top_k]
    # ``search_query`` (singular) stays the primary query for the existing trace field.
    return {
        "search_query": queries[0],
        "search_queries": queries,
        "chunks": chunks,
        "reranked": reranked,
        "diversified": diversified,
    }


def grade(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Corrective-RAG sufficiency gate: are the retrieved chunks enough to answer?

    Cheap first, expensive only if needed. When augmentation is off the gate is a no-op
    (always sufficient). Otherwise a similarity *floor* (``settings.min_similarity``) rejects
    empty/weak retrievals without an LLM call; only borderline cases — passages that clear
    the floor but may still miss the concept — pay for the LLM grader (with an offline
    heuristic fallback). An insufficient verdict routes the flow through ``augment``.
    """
    settings = deps.settings
    if not settings.enable_augmentation:
        return {
            "grade": RelevanceGrade(
                sufficient=True, confidence=1.0, reason="Augmentation disabled."
            )
        }
    chunks = chunks_of(state)
    best = max((c.score for c in chunks), default=0.0)
    if not chunks or best < settings.min_similarity:
        return {
            "grade": RelevanceGrade(
                sufficient=False,
                confidence=0.9,
                reason=(
                    f"Best retrieval score {best:.2f} is below the "
                    f"{settings.min_similarity:.2f} similarity floor."
                ),
            )
        }
    with deps.timer.measure("grade"):
        verdict = deps.router.grade_relevance(state["question"], chunks, meter=deps.router_usage)
    return {"grade": verdict}


def augment(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Fetch citable passages from external sources and merge them into the context.

    Runs only when ``grade`` judged the KB insufficient. Sources are pluggable (arXiv now,
    web later) via :func:`src.core.sources.build_sources`; each is isolated so one failing
    provider never sinks the request. The extra chunks flow on exactly like KB chunks —
    numbered, cited, and shown in the trace. The step body is deliberately a single "fetch
    once" pass: Option C can later swap it for a bounded agentic sub-loop with no changes
    elsewhere.
    """
    settings = deps.settings
    query = state.get("search_query") or state["question"]
    extra: list[RetrievedChunk] = []
    used: list[str] = []
    with deps.timer.measure("augment"):
        for source in build_sources(settings.augment_sources):
            try:
                hits = source.fetch(query, k=settings.augment_k)
            except Exception:
                # A failing source must never sink the request, but silently dropping it
                # made a misconfigured source indistinguishable from one that simply had
                # no hits for the query.
                logger.warning(
                    "Augmentation source %r failed; skipping it.", source.name, exc_info=True
                )
                continue
            if hits:
                extra.extend(hits)
                used.append(source.name)
    merged = chunks_of(state) + extra
    return {
        "chunks": merged,
        "augmented": bool(extra),
        "augment_sources_used": used,
    }


def generate(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Level/technique-styled RAG generation, with optional no-RAG comparison."""
    with deps.timer.measure("generation"):
        result = answer_question(
            deps.llm,
            state["question"],
            chunks_of(state),
            history_of(state),
            level=deps.req.level,
            style=state["style"],
            tools=deps.tools,
        )
    out: dict[str, Any] = {"result": result, "text": result.text, "ok": True}
    if deps.settings.compare_no_rag:
        with deps.timer.measure("no_rag_generation"):
            out["no_rag"] = answer_without_rag(deps.llm, state["question"], deps.req.level)
    return out


def tools_only(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Tool-first path: skip retrieval and let the model call tools with no context."""
    with deps.timer.measure("generation"):
        result = answer_question(
            deps.llm,
            state["question"],
            [],
            history_of(state),
            level=deps.req.level,
            style=state["style"],
            tools=deps.tools,
        )
    return {"result": result, "text": result.text, "ok": True, "chunks": []}


def meta(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Answer a question about the assistant itself."""
    with deps.timer.measure("generation"):
        result = answer_meta(deps.llm, state["question"], history_of(state))
    return {"result": result, "text": result.text, "ok": True, "chunks": []}


def agent(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Bounded plan->act->observe agent route (Phase 6, Option C).

    Delegates to :func:`src.core.agent.run_agent_loop`, which runs a hard-capped loop that
    may re-retrieve and call tools before synthesising a grounded, cited answer. The gathered
    passages flow out as ``chunks`` (so citations and the Inspector work unchanged) and the
    step trajectory is exposed for the trace.
    """
    with deps.timer.measure("agent"):
        outcome = run_agent_loop(
            deps.llm,
            state["question"],
            kb=deps.kb,
            tools=deps.tools,
            settings=deps.settings,
            level=deps.req.level,
            style=state["style"],
            history=history_of(state),
            subjects=state.get("subjects"),
        )
    return {
        "result": outcome.result,
        "text": outcome.result.text,
        "ok": True,
        "chunks": outcome.chunks,
        "agent_trajectory": outcome.trajectory,
    }


def refuse(state: PipelineState, deps: PipelineDeps) -> dict[str, Any]:
    """Polite refusal: a distinct message for injection vs. off-topic."""
    verdict = state.get("verdict")
    text = INJECTION_REFUSAL if verdict and verdict.is_injection else REFUSAL
    return {"text": text, "ok": False, "chunks": []}


# --- branching predicates (state-only, shared by both engines) --------------
def after_screen(state: PipelineState) -> str:
    """Injection -> straight to refusal; otherwise carry on to routing."""
    verdict = state.get("verdict")
    return "refuse" if verdict and verdict.is_injection else "route"


def after_route(state: PipelineState) -> str:
    """Map the route decision onto the next step name.

    The ``agent`` route (Phase 6) maps to the ``agent`` step; any unknown route falls back
    to ``retrieve`` as a safe default.
    """
    decision = state.get("decision")
    return ROUTE_TO_STEP.get(decision.route if decision else "knowledge", "retrieve")


def after_grade(state: PipelineState) -> str:
    """Sufficient context -> generate; insufficient -> augment first (Corrective RAG)."""
    grade = state.get("grade")
    return "generate" if (grade is None or grade.sufficient) else "augment"
