"""
The assistant pipeline as a framework-agnostic service.

``AssistantService.answer`` executes the routed RAG workflow and returns a plain
:class:`AnswerBundle`. It validates input, creates the chat model, and delegates
execution to one of two interchangeable engines built on the same shared steps:
the default lightweight router
(:func:`~src.core.linear.run_linear`) or the LangGraph ``StateGraph``
(:func:`~src.core.graph.build_pipeline`), selected by
``settings.pipeline_engine``.

Both engines follow the same flow: injection screening, query routing, and
execution of only the required path (knowledge / tool / meta / off-topic).
The service does not depend on Streamlit, keeping it unit-testable, and manages
token/cost accounting and trace assembly around the selected engine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.config import ROUTER_MODEL, RagSettings
from src.core.graph import build_pipeline, run_streaming
from src.core.linear import run_linear
from src.core.mcp_client import load_mcp_tools
from src.core.router import QueryRouter
from src.core.schemas import InjectionVerdict, RelevanceGrade, RouteDecision
from src.core.steps import PipelineDeps, initial_state
from src.generation import compose_style
from src.llm import get_llm
from src.rag.retriever import KnowledgeBase
from src.runlog import log_run, now_iso
from src.security import validate_input
from src.tools import ALL_TOOLS
from src.utils import StageTimer, estimate_cost


@dataclass
class AnswerRequest:
    """Everything the pipeline needs to answer one question, free of any UI state."""

    question: str
    level: str
    model: str
    subjects: list[str] | None
    settings: RagSettings
    history: list[tuple[str, str]] = field(default_factory=list)
    technique: str = "Standard"
    length: str = "Balanced"
    extra_instructions: str = ""


@dataclass
class AnswerBundle:
    """The pipeline's result: the answer plus sources, tool calls, usage, and a trace.

    ``ok`` is False for the early-exit paths (invalid input, LLM unavailable, off-domain,
    injection, generation failure); in those cases ``text`` holds the user-facing message
    and the other fields stay at their empty defaults.
    """

    text: str
    ok: bool = True
    sources: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    # Full text of the retrieved (and any augmented) context passages, in order. Kept
    # separate from ``sources`` (which carries only metadata + scores for display) because
    # the evaluation harness needs the raw passage text to judge faithfulness and context
    # precision/recall. Empty on the non-knowledge routes.
    contexts: list[str] = field(default_factory=list)
    no_rag: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0


class AssistantService:
    """Runs the routed RAG + tool-calling pipeline over an optional knowledge base."""

    def __init__(self, kb: KnowledgeBase | None, router: QueryRouter | None = None) -> None:
        self._kb = kb
        self._router = router or QueryRouter()

    def answer(
        self, req: AnswerRequest, on_node: Callable[[str], None] | None = None
    ) -> AnswerBundle:
        """Execute the routed pipeline for one request and return an :class:`AnswerBundle`.

        ``on_node`` is an optional progress callback invoked with each graph node's name as it
        completes. It is honoured only on the LangGraph engine (``pipeline_engine == "graph"``),
        which can stream node updates; the linear engine and every early-exit path ignore it.
        A plain callable keeps this service free of any UI dependency — the Chat page passes a
        callback that ticks an ``st.status`` widget, but the core never imports Streamlit.
        """
        check = validate_input(req.question)
        if not check.ok:
            self._log_run(req, outcome="invalid", error=check.reason)
            return AnswerBundle(text=f"⚠️ {check.reason}", ok=False)

        timer = StageTimer()
        try:
            llm = get_llm(
                req.model,
                req.settings.temperature,
                top_p=req.settings.top_p,
                max_tokens=req.settings.max_tokens,
            )
        except Exception as exc:
            self._log_run(req, outcome="llm_unavailable", error=str(exc))
            return AnswerBundle(text=f"⚠️ LLM unavailable: {exc}", ok=False)

        style = compose_style(req.level, req.technique, req.length, req.extra_instructions)
        # Resolve the tool set once per request: built-ins plus any remote MCP tools (an
        # empty list unless MCP is enabled and reachable — load_mcp_tools never raises).
        mcp_tools = load_mcp_tools(req.settings)
        deps = PipelineDeps(
            llm=llm,
            kb=self._kb,
            settings=req.settings,
            router=self._router,
            req=req,
            timer=timer,
            tools=list(ALL_TOOLS) + mcp_tools,
        )
        # Seed the pipeline through the one documented entry point, so both engines start from
        # an identical state and steps.initial_state stays the single place a request becomes state.
        initial: dict[str, Any] = initial_state(
            question=req.question,
            history=req.history,
            subjects=req.subjects,
            level=req.level,
            style=style,
        )
        # Two interchangeable engines over the same shared steps: the lightweight linear
        # router (default) or the LangGraph StateGraph. Selectable for A/B comparison.
        engine = getattr(req.settings, "pipeline_engine", "linear")
        try:
            if engine == "graph":
                # Stream node-by-node progress when a callback is supplied, otherwise run the
                # graph in one shot. Both paths return an identical final state (see
                # :func:`~src.core.graph.run_streaming`).
                if on_node is not None:
                    final = run_streaming(deps, initial, on_node)
                else:
                    final = build_pipeline(deps).invoke(initial)
            else:
                final = run_linear(deps, initial)
        except Exception as exc:
            self._log_run(req, outcome="error", engine=engine, error=str(exc))
            # Any engine error degrades to a user-visible message rather than a crash (graded).
            # The wording no longer asserts it is specifically a billing problem — that misled on
            # non-provider errors; the exact exception is kept in the run log (outcome="error").
            return AnswerBundle(
                text=(
                    f"⚠️ The request could not be completed ({exc}). If this looks like an "
                    "authentication or rate-limit issue, check your API key and credit; "
                    "otherwise please try again."
                ),
                ok=False,
            )

        decision: RouteDecision | None = final.get("decision")
        verdict: InjectionVerdict | None = final.get("verdict")
        result = final.get("result")

        # Early-exit paths (injection / off-topic refusal, or a path that produced no
        # answer) carry their user-facing text and nothing else.
        if not final.get("ok", False) or result is None:
            self._log_run(
                req,
                outcome="refused",
                engine=engine,
                route=decision.route if decision else None,
                injection=verdict.is_injection if verdict else None,
                timings=timer.stages,
            )
            return AnswerBundle(
                text=final.get("text", ""),
                ok=False,
                meta=self._route_meta(req, decision, verdict),
            )

        chunks = final.get("chunks") or []
        no_rag = final.get("no_rag")
        # Cost is summed *per model*, never pooled at a single price. The answer-model tokens
        # are the final generation plus the auxiliary main-model calls (query rewrite/expansion,
        # reranking) captured on ``deps.usage`` — all on req.model. The routing, injection
        # screen, and relevance grader run on the separate, cheaper ROUTER_MODEL and are metered
        # apart on ``deps.router_usage``; pricing them at ROUTER_MODEL (rather than folding them
        # into req.model's price, which would misreport their cost) closes the known gap.
        answer_in = (
            result.input_tokens + (no_rag.input_tokens if no_rag else 0) + deps.usage.input_tokens
        )
        answer_out = (
            result.output_tokens
            + (no_rag.output_tokens if no_rag else 0)
            + deps.usage.output_tokens
        )
        router_in = deps.router_usage.input_tokens
        router_out = deps.router_usage.output_tokens
        answer_cost = estimate_cost(req.model, answer_in, answer_out)
        router_cost = estimate_cost(ROUTER_MODEL, router_in, router_out)
        cost = answer_cost + router_cost
        # Reported totals include the classification tokens so the count is honest; the split
        # by model is preserved in the trace breakdown below.
        tokens_in = answer_in + router_in
        tokens_out = answer_out + router_out

        cost_breakdown = {
            "answer": {
                "model": req.model,
                "input": answer_in,
                "output": answer_out,
                "cost_usd": round(answer_cost, 6),
            },
            "router": {
                "model": ROUTER_MODEL,
                "input": router_in,
                "output": router_out,
                "cost_usd": round(router_cost, 6),
            },
        }

        meta = self._route_meta(req, decision, verdict)
        meta.update({"tokens_in": tokens_in, "tokens_out": tokens_out, "cost": cost})

        self._log_run(
            req,
            outcome="answered",
            engine=engine,
            route=decision.route if decision else None,
            injection=verdict.is_injection if verdict else None,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            timings=timer.stages,
        )

        return AnswerBundle(
            text=result.text,
            ok=True,
            sources=[{"ref": i + 1, "score": c.score, **c.metadata} for i, c in enumerate(chunks)],
            contexts=[c.text for c in chunks],
            tools=[
                {"name": e.name, "args": e.args, "output": e.output} for e in result.tool_events
            ],
            no_rag=no_rag.text if no_rag else None,
            meta=meta,
            trace={
                "question": req.question,
                "engine": engine,
                "route": decision.route if decision else None,
                "route_reason": decision.reason if decision else None,
                "route_confidence": decision.confidence if decision else None,
                "injection": self._verdict_trace(verdict),
                "grade": self._grade_trace(final.get("grade")),
                "agent": self._agent_trace(final.get("agent_trajectory")),
                "augmented": bool(final.get("augmented")),
                "augment_sources_used": final.get("augment_sources_used") or [],
                "mcp": (
                    {
                        "enabled": True,
                        "server": req.settings.mcp_server_url,
                        "tools": [t.name for t in mcp_tools],
                    }
                    if req.settings.enable_mcp
                    else None
                ),
                "rewritten_query": final.get("search_query", req.question),
                "search_queries": final.get("search_queries")
                or [final.get("search_query", req.question)],
                "reranked": bool(final.get("reranked")),
                "diversified": bool(final.get("diversified")),
                "subject_filter": req.subjects or "all",
                "metadata_filters": {
                    "topics": list(req.settings.filter_topics),
                    "sources": list(req.settings.filter_sources),
                    "years": list(req.settings.filter_years),
                },
                "technique": req.technique,
                "response_length": req.length,
                "chunks": [
                    {
                        "source": c.metadata.get("source"),
                        "topic": c.metadata.get("topic"),
                        "difficulty": c.metadata.get("difficulty"),
                        "vector": c.vector_score,
                        "bm25": c.bm25_score,
                        "hybrid": c.score,
                    }
                    for c in chunks
                ],
                "timings_s": {k: round(v, 2) for k, v in timer.stages.items()},
                "tokens": {
                    "input": tokens_in,
                    "output": tokens_out,
                    "cost_usd": round(cost, 6),
                    "breakdown": cost_breakdown,
                },
                "model": req.model,
            },
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
        )

    @staticmethod
    def _log_run(
        req: AnswerRequest,
        *,
        outcome: str,
        engine: str | None = None,
        route: str | None = None,
        injection: bool | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
        timings: dict[str, float] | None = None,
        error: str | None = None,
    ) -> None:
        """Append one structured run record to the JSONL log (best-effort, secret-free).

        Gated by ``settings.enable_run_log``; the question is truncated and no credentials
        are ever included, so the log is safe to keep and share for observability.
        """
        log_run(
            {
                "ts": now_iso(),
                "question": req.question[:500],
                "model": req.model,
                "level": req.level,
                "engine": engine,
                "outcome": outcome,
                "route": route,
                "injection": injection,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(cost, 6),
                "timings_s": {k: round(v, 3) for k, v in (timings or {}).items()},
                "error": error,
            },
            enabled=getattr(req.settings, "enable_run_log", False),
        )

    @staticmethod
    def _route_meta(
        req: AnswerRequest,
        decision: RouteDecision | None,
        verdict: InjectionVerdict | None,
    ) -> dict[str, Any]:
        """Common ``meta`` fields shared by every return path."""
        return {
            "level": req.level,
            "model": req.model,
            "route": decision.route if decision else None,
            "injection": (verdict.is_injection if verdict else None),
        }

    @staticmethod
    def _verdict_trace(verdict: InjectionVerdict | None) -> dict[str, Any] | None:
        """Render the injection verdict for the trace, or None if screening didn't run."""
        if verdict is None:
            return None
        return {
            "is_injection": verdict.is_injection,
            "confidence": verdict.confidence,
            "reason": verdict.reason,
        }

    @staticmethod
    def _agent_trace(trajectory: list[Any] | None) -> list[dict[str, Any]] | None:
        """Render the bounded-agent trajectory for the trace, or None if the route didn't run."""
        if not trajectory:
            return None
        return [
            {
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
            }
            for step in trajectory
        ]

    @staticmethod
    def _grade_trace(grade: RelevanceGrade | None) -> dict[str, Any] | None:
        """Render the Corrective-RAG sufficiency grade for the trace, or None if it didn't run."""
        if grade is None:
            return None
        return {
            "sufficient": grade.sufficient,
            "confidence": grade.confidence,
            "reason": grade.reason,
        }
