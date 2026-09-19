"""Offline unit tests for the Corrective-RAG layer: grade gate, augment, source registry.

These pin the Phase 2 logic that the engine-parity test only exercises indirectly:

* the ``grade`` step's cheap similarity floor and its escalation to the LLM grader,
* the ``augment`` step merging pluggable external passages (and tolerating a failing source),
* the router's offline sufficiency heuristic,
* the source registry / :func:`build_sources` factory.

Everything runs without a network: no LLM, no arXiv, only stubs.
"""

from __future__ import annotations

from src.config import RagSettings
from src.core.schemas import RelevanceGrade
from src.core.sources import (
    SOURCE_REGISTRY,
    ArxivSource,
    ExternalSource,
    WebSearchSource,
    _chunks_from_annotations,
    build_sources,
)
from src.core.steps import PipelineDeps, augment, grade
from src.rag.retriever import RetrievedChunk
from src.utils import StageTimer


def _chunk(score: float) -> RetrievedChunk:
    """A minimal retrieved chunk with the given hybrid score."""
    return RetrievedChunk(
        text="passage",
        metadata={"title": "t"},
        vector_score=score,
        bm25_score=0.0,
        score=score,
    )


class _GradeRouter:
    """Router stub recording whether the LLM grader was consulted."""

    def __init__(self, sufficient: bool) -> None:
        self.called = False
        self._grade = RelevanceGrade(sufficient=sufficient, confidence=0.8, reason="stub")

    def grade_relevance(
        self, question: str, chunks: list[RetrievedChunk], *, meter: object = None
    ) -> RelevanceGrade:
        self.called = True
        return self._grade


def _deps(router: object, settings: RagSettings) -> PipelineDeps:
    """Offline deps (no LLM, no KB) wired to the given router stub and settings."""
    return PipelineDeps(
        llm=None, kb=None, settings=settings, router=router, req=None, timer=StageTimer()
    )


# --- grade step -------------------------------------------------------------
def test_grade_disabled_is_sufficient_without_router() -> None:
    """With augmentation off the gate always passes and never consults the grader."""
    router = _GradeRouter(sufficient=False)
    deps = _deps(router, RagSettings(enable_augmentation=False))
    out = grade({"question": "q", "chunks": [_chunk(0.9)]}, deps)
    assert out["grade"].sufficient is True
    assert router.called is False


def test_grade_empty_chunks_below_floor_is_insufficient() -> None:
    """No retrieved chunks -> cheap floor rejects, LLM grader never runs."""
    router = _GradeRouter(sufficient=True)
    deps = _deps(router, RagSettings(enable_augmentation=True))
    out = grade({"question": "q", "chunks": []}, deps)
    assert out["grade"].sufficient is False
    assert router.called is False


def test_grade_weak_score_below_floor_is_insufficient() -> None:
    """A best score under min_similarity is rejected by the floor without the grader."""
    router = _GradeRouter(sufficient=True)
    deps = _deps(router, RagSettings(enable_augmentation=True, min_similarity=0.5))
    out = grade({"question": "q", "chunks": [_chunk(0.2)]}, deps)
    assert out["grade"].sufficient is False
    assert router.called is False


def test_grade_above_floor_escalates_to_router() -> None:
    """Chunks clearing the floor pay for the LLM grader, which decides sufficiency."""
    router = _GradeRouter(sufficient=True)
    deps = _deps(router, RagSettings(enable_augmentation=True, min_similarity=0.15))
    out = grade({"question": "q", "chunks": [_chunk(0.8)]}, deps)
    assert router.called is True
    assert out["grade"].sufficient is True


# --- augment step -----------------------------------------------------------
class _OneHitSource(ExternalSource):
    name = "one"

    def fetch(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        return [_chunk(0.5)]


class _BrokenSource(ExternalSource):
    name = "broken"

    def fetch(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        raise RuntimeError("boom")


def test_augment_merges_external_and_records_source(monkeypatch) -> None:
    """augment appends fetched passages to existing chunks and records which source fired."""
    import src.core.steps as steps_mod

    monkeypatch.setattr(steps_mod, "build_sources", lambda names: [_OneHitSource()])
    deps = _deps(object(), RagSettings())
    existing = _chunk(0.9)
    out = augment({"question": "q", "search_query": "q", "chunks": [existing]}, deps)
    assert len(out["chunks"]) == 2
    assert out["chunks"][0] is existing
    assert out["augmented"] is True
    assert out["augment_sources_used"] == ["one"]


def test_augment_tolerates_a_failing_source(monkeypatch) -> None:
    """A source that raises is skipped; the request still completes with no extra chunks."""
    import src.core.steps as steps_mod

    monkeypatch.setattr(steps_mod, "build_sources", lambda names: [_BrokenSource()])
    deps = _deps(object(), RagSettings())
    out = augment({"question": "q", "search_query": "q", "chunks": []}, deps)
    assert out["chunks"] == []
    assert out["augmented"] is False
    assert out["augment_sources_used"] == []


# --- router heuristic & source registry ------------------------------------
def test_heuristic_grade_needs_two_corroborating_passages() -> None:
    """Offline grader: >=2 passages past the floor is sufficient, a lone hit is not."""
    from src.core.router import QueryRouter

    assert QueryRouter._heuristic_grade([_chunk(0.5), _chunk(0.5)]).sufficient is True
    assert QueryRouter._heuristic_grade([_chunk(0.5)]).sufficient is False


def test_build_sources_resolves_known_names_only() -> None:
    """The registry maps 'arxiv' to ArxivSource and silently drops unknown names."""
    sources = build_sources(["arxiv", "does-not-exist"])
    assert len(sources) == 1
    assert isinstance(sources[0], ArxivSource)
    assert build_sources([]) == []


# --- web search source ------------------------------------------------------
def test_web_source_is_registered() -> None:
    """'web' resolves to WebSearchSource, so it appears in the Settings source picker."""
    assert SOURCE_REGISTRY["web"] is WebSearchSource
    assert isinstance(build_sources(["web"])[0], WebSearchSource)


def test_web_annotations_become_citable_chunks() -> None:
    """url_citation annotations map to chunks with url/title metadata; others are skipped."""
    annotations = [
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://ex.com/a",
                "title": "Alpha",
                "content": "alpha body",
            },
        },
        {"type": "url_citation", "url_citation": {"url": "https://ex.com/b", "title": "Beta"}},
        {"type": "file", "foo": 1},  # non-citation annotation is ignored
        {"type": "url_citation", "url_citation": {"title": "no url"}},  # no url -> skipped
    ]
    chunks = _chunks_from_annotations(annotations, k=5)
    assert len(chunks) == 2
    assert chunks[0].metadata["url"] == "https://ex.com/a"
    assert chunks[0].metadata["source"] == "web:ex.com"
    assert "alpha body" in chunks[0].text
    assert chunks[1].text == "Beta"  # no content -> title only
    # Rank-based scores are descending and positive.
    assert chunks[0].score > chunks[1].score > 0


def test_web_annotations_respect_k() -> None:
    """Parsing stops once k citations are collected."""
    annotations = [
        {"type": "url_citation", "url_citation": {"url": f"https://ex.com/{i}", "title": str(i)}}
        for i in range(5)
    ]
    assert len(_chunks_from_annotations(annotations, k=2)) == 2


def test_web_annotations_handle_empty() -> None:
    """None / empty annotations yield no chunks (no crash)."""
    assert _chunks_from_annotations(None, k=3) == []
    assert _chunks_from_annotations([], k=3) == []


def test_web_source_fetch_without_key_is_empty(monkeypatch) -> None:
    """With no OpenRouter key the web source stays offline and returns [] (never raises)."""
    import src.config as cfg

    monkeypatch.setattr(cfg, "openrouter_api_key", lambda: None)
    assert WebSearchSource().fetch("what is attention?") == []
