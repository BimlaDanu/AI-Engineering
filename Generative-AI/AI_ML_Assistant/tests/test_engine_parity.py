"""Offline parity test: the linear and LangGraph engines must behave identically.

Phase 1 ships the routed pipeline as two orchestrations over the *same* shared steps
(:mod:`src.core.steps`): the lightweight ``if/elif`` engine (:mod:`src.core.linear`) and the
LangGraph ``StateGraph`` (:mod:`src.core.graph`). Their defining invariant — the whole point
of keeping both — is that, given identical steps and dependencies, they produce the same
final state on every route. This test pins that invariant down without a network by stubbing
the LLM-backed generation calls and the router so each route is deterministic; any divergence
must then come from the orchestration itself, which is exactly what we want to catch.
"""

from __future__ import annotations

import pytest

from src.config import RagSettings
from src.core import steps as steps_mod
from src.core.graph import build_pipeline, run_streaming
from src.core.linear import run_linear
from src.core.schemas import InjectionVerdict, RelevanceGrade, RouteDecision
from src.core.service import AnswerRequest
from src.core.steps import INJECTION_REFUSAL, REFUSAL, PipelineDeps
from src.generation import AnswerResult
from src.rag.retriever import RetrievedChunk
from src.utils import StageTimer

# A fixed answer the stubbed generation functions return, so both engines see identical step
# products and any difference in the final state must originate in the orchestration.
_STUB_ANSWER = AnswerResult(text="stub answer", input_tokens=11, output_tokens=7)

# A fixed external passage the stubbed augment source returns, so the Corrective-RAG augment
# path is deterministic and offline (no arXiv network call) in both engines.
_STUB_EXTERNAL = RetrievedChunk(
    text="external passage",
    metadata={"source": "arXiv:0000.00000", "title": "Stub paper"},
    vector_score=0.0,
    bm25_score=0.0,
    score=0.5,
)

# The observable outcome of a run; incidental/unset keys are compared as ``None`` on both.
# ``grade`` and ``augmented`` pin the Corrective-RAG branch behaviour across both engines.
_COMPARE_KEYS = (
    "text",
    "ok",
    "decision",
    "verdict",
    "grade",
    "search_query",
    "chunks",
    "augmented",
    "augment_sources_used",
    "result",
)


class _FakeSource:
    """Offline external source stub: fetch() returns one fixed passage, no network."""

    name = "stub"

    def fetch(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        return [_STUB_EXTERNAL]


class _FakeRouter:
    """Router stub with preset injection, route, and relevance verdicts (no LLM, no network)."""

    def __init__(self, *, is_injection: bool, route: str, sufficient: bool = False) -> None:
        self._verdict = InjectionVerdict(is_injection=is_injection, confidence=0.9, reason="stub")
        self._decision = RouteDecision(route=route, confidence=0.9, reason="stub")
        self._grade = RelevanceGrade(sufficient=sufficient, confidence=0.9, reason="stub")

    def screen_injection(self, question: str, *, meter: object = None) -> InjectionVerdict:
        return self._verdict

    def classify(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        *,
        allow_agent: bool = False,
        meter: object = None,
    ) -> RouteDecision:
        return self._decision

    def grade_relevance(
        self, question: str, chunks: list[RetrievedChunk], *, meter: object = None
    ) -> RelevanceGrade:
        return self._grade


def _make_deps(router: _FakeRouter, *, enable_augmentation: bool = True) -> PipelineDeps:
    """Build offline deps (no LLM, no KB) wired to the given router stub."""
    settings = RagSettings(enable_augmentation=enable_augmentation)
    req = AnswerRequest(
        question="q", level="Beginner", model="stub", subjects=None, settings=settings
    )
    return PipelineDeps(
        llm=None, kb=None, settings=settings, router=router, req=req, timer=StageTimer()
    )


def _initial() -> dict:
    """A fresh initial pipeline state, matching what the service seeds per request."""
    return {
        "question": "What is attention in transformers?",
        "history": [],
        "subjects": None,
        "level": "Beginner",
        "style": "",
    }


def _view(state: dict) -> dict:
    """Project a final state onto the keys that define its observable outcome."""
    return {key: state.get(key) for key in _COMPARE_KEYS}


@pytest.fixture(autouse=True)
def _stub_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the LLM-backed generation calls and the external source layer with stubs.

    Both engines run the *same* step functions from :mod:`src.core.steps`, which reference
    these names at module scope, so patching them here covers the linear and graph engines
    at once. Stubbing ``build_sources`` keeps the Corrective-RAG augment path offline.
    """
    monkeypatch.setattr(steps_mod, "answer_question", lambda *a, **k: _STUB_ANSWER)
    monkeypatch.setattr(steps_mod, "answer_meta", lambda *a, **k: _STUB_ANSWER)
    monkeypatch.setattr(steps_mod, "answer_without_rag", lambda *a, **k: _STUB_ANSWER)
    monkeypatch.setattr(steps_mod, "build_sources", lambda names: [_FakeSource()])


@pytest.mark.parametrize(
    ("is_injection", "route"),
    [
        (False, "knowledge"),  # retrieve -> grade -> augment -> generate
        (False, "tool"),  # tools_only
        (False, "meta"),  # about-the-app
        (False, "off_topic"),  # refuse (domain)
        (True, "knowledge"),  # injection short-circuits before the route is ever used
    ],
)
def test_engines_produce_identical_state(is_injection: bool, route: str) -> None:
    """Every route yields the same observable state from both orchestration engines."""
    linear_final = run_linear(
        _make_deps(_FakeRouter(is_injection=is_injection, route=route)), _initial()
    )
    graph_final = build_pipeline(
        _make_deps(_FakeRouter(is_injection=is_injection, route=route))
    ).invoke(_initial())
    assert _view(linear_final) == _view(graph_final)


def test_injection_short_circuits_to_refusal_in_both_engines() -> None:
    """An injection verdict skips routing/generation and refuses in both engines."""
    for run in (
        lambda deps: run_linear(deps, _initial()),
        lambda deps: build_pipeline(deps).invoke(_initial()),
    ):
        final = run(_make_deps(_FakeRouter(is_injection=True, route="knowledge")))
        assert final["ok"] is False
        assert final["text"] == INJECTION_REFUSAL
        assert final.get("result") is None  # generation never ran


def test_off_topic_refuses_in_both_engines() -> None:
    """An off-topic route produces the domain refusal (not the injection one) in both."""
    for run in (
        lambda deps: run_linear(deps, _initial()),
        lambda deps: build_pipeline(deps).invoke(_initial()),
    ):
        final = run(_make_deps(_FakeRouter(is_injection=False, route="off_topic")))
        assert final["ok"] is False
        assert final["text"] == REFUSAL


def test_insufficient_kb_augments_in_both_engines() -> None:
    """With no KB the cheap floor grades insufficient, so both engines fetch externally."""
    for run in (
        lambda deps: run_linear(deps, _initial()),
        lambda deps: build_pipeline(deps).invoke(_initial()),
    ):
        final = run(_make_deps(_FakeRouter(is_injection=False, route="knowledge")))
        assert final["ok"] is True
        assert final["grade"].sufficient is False
        assert final["augmented"] is True
        assert final["chunks"] == [_STUB_EXTERNAL]
        assert final["augment_sources_used"] == ["stub"]


def test_streaming_matches_invoke_and_reports_nodes_in_order() -> None:
    """The streamed run reports each node as it fires and returns the same final state.

    Track A adds ``run_streaming`` (``stream_mode="updates"``) as the graph engine's live-
    progress path. Its contract is that streaming changes only *when* progress is observed,
    never *what* is computed — so the accumulated final state must equal ``invoke()``'s, and
    the reported node sequence must trace the knowledge path (screen → route → retrieve →
    grade → augment → generate; augment appears because the empty KB grades insufficient).
    """
    seen: list[str] = []
    deps = _make_deps(_FakeRouter(is_injection=False, route="knowledge"))
    streamed = run_streaming(deps, _initial(), seen.append)

    invoked = build_pipeline(_make_deps(_FakeRouter(is_injection=False, route="knowledge"))).invoke(
        _initial()
    )
    assert _view(streamed) == _view(invoked)
    assert seen == ["screen", "route", "retrieve", "grade", "augment", "generate"]


def test_streaming_reports_only_the_refuse_node_on_injection() -> None:
    """An injection verdict short-circuits: streaming reports screen → refuse and nothing else."""
    seen: list[str] = []
    deps = _make_deps(_FakeRouter(is_injection=True, route="knowledge"))
    final = run_streaming(deps, _initial(), seen.append)
    assert seen == ["screen", "refuse"]
    assert final["text"] == INJECTION_REFUSAL


def test_augmentation_disabled_skips_augment_in_both_engines() -> None:
    """When augmentation is off the grade gate is a no-op: no external fetch in either engine."""
    for run in (
        lambda deps: run_linear(deps, _initial()),
        lambda deps: build_pipeline(deps).invoke(_initial()),
    ):
        final = run(
            _make_deps(
                _FakeRouter(is_injection=False, route="knowledge"),
                enable_augmentation=False,
            )
        )
        assert final["ok"] is True
        assert final["grade"].sufficient is True
        assert not final.get("augmented")
        assert final["chunks"] == []
