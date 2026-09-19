"""Offline tests for the listwise reranker (RAG stage 2): reorder a candidate pool by relevance.

The reranker over-fetches a wider first-stage pool, asks an LLM to reorder the whole pool by
relevance to the question (structured :class:`Reranking`), and keeps the top-k. These tests
pin the index-resolution maths, the graceful fallback to first-stage order, and the
off-by-default parity of the ``retrieve`` step — all without a network, an LLM, or embeddings.
"""

from __future__ import annotations

from src.config import RagSettings
from src.core.schemas import Reranking
from src.core.service import AnswerRequest
from src.core.steps import PipelineDeps, retrieve
from src.rag.retriever import RetrievedChunk, _resolve_ranking, rerank_chunks
from src.utils import StageTimer


def _chunk(text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={"source": "s.md"},
        vector_score=score,
        bm25_score=score,
        score=score,
    )


# --- _resolve_ranking: always a valid permutation of range(n) ---------------
def test_resolve_ranking_maps_one_based_to_zero_based() -> None:
    assert _resolve_ranking([3, 1, 2], 3) == [2, 0, 1]


def test_resolve_ranking_appends_omitted_indices_in_order() -> None:
    # Model ranked only #3; the rest are appended in their original order (recall never drops).
    assert _resolve_ranking([3], 4) == [2, 0, 1, 3]


def test_resolve_ranking_drops_out_of_range_and_duplicates() -> None:
    # 9 is out of range, the second 2 is a duplicate -> both ignored; 3 then appended.
    assert _resolve_ranking([2, 9, 2, 1], 3) == [1, 0, 2]


def test_resolve_ranking_empty_model_output_keeps_original_order() -> None:
    assert _resolve_ranking([], 3) == [0, 1, 2]


# --- rerank_chunks: reorder + graceful fallback -----------------------------
class _Structured:
    def __init__(self, reranking: Reranking) -> None:
        self._reranking = reranking

    def invoke(self, _messages: object) -> Reranking:
        return self._reranking


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content


class _RerankingLLM:
    """Returns a preset Reranking via structured output."""

    def __init__(self, ranking: list[int]) -> None:
        self._ranking = ranking

    def with_structured_output(self, schema: object, method: str | None = None) -> _Structured:
        return _Structured(Reranking(ranking=self._ranking))

    def invoke(self, _messages: object) -> _Reply:
        return _Reply("unused")


class _BrokenLLM:
    """Structured output raises -> reranker must keep the first-stage order."""

    def with_structured_output(self, schema: object, method: str | None = None) -> object:
        raise RuntimeError("structured output unavailable")

    def invoke(self, _messages: object) -> _Reply:
        return _Reply("unused")


def test_rerank_reorders_by_model_ranking() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    out = rerank_chunks(_RerankingLLM([3, 1, 2]), "q", [a, b, c])
    assert [x.text for x in out] == ["c", "a", "b"]


def test_rerank_caps_to_top_n() -> None:
    chunks = [_chunk(x) for x in ("a", "b", "c", "d")]
    out = rerank_chunks(_RerankingLLM([4, 3, 2, 1]), "q", chunks, top_n=2)
    assert [x.text for x in out] == ["d", "c"]


def test_rerank_falls_back_to_original_order_on_error() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    out = rerank_chunks(_BrokenLLM(), "q", [a, b, c], top_n=2)
    assert [x.text for x in out] == ["a", "b"]  # first-stage order preserved, then capped


def test_rerank_single_chunk_is_passthrough() -> None:
    # No LLM call is worthwhile for a pool of one; returns it unchanged.
    a = _chunk("a")
    assert rerank_chunks(_RerankingLLM([1]), "q", [a]) == [a]


def test_rerank_never_drops_candidates_when_model_omits_some() -> None:
    # Model ranks only #2; the reranker still returns all three (omitted ones appended).
    chunks = [_chunk(x) for x in ("a", "b", "c")]
    out = rerank_chunks(_RerankingLLM([2]), "q", chunks)
    assert {x.text for x in out} == {"a", "b", "c"}
    assert out[0].text == "b"


# --- retrieve step: off-by-default parity + over-fetch when on --------------
class _RecordingKB:
    """Records the ``k`` it is searched with and returns ``k`` distinct chunks."""

    def __init__(self) -> None:
        self.k_seen: list[int] = []

    def search(self, query, subjects=None, difficulty=None, k=4, alpha=0.7, *, filters=None):
        self.k_seen.append(k)
        return [_chunk(f"hit{i}", score=1.0 - i / 100) for i in range(k)]


def _deps(kb, settings, llm) -> PipelineDeps:
    req = AnswerRequest(
        question="how does attention work",
        level="Practitioner",
        model="m",
        subjects=None,
        settings=settings,
    )
    return PipelineDeps(llm=llm, kb=kb, settings=settings, router=None, req=req, timer=StageTimer())


def test_retrieve_does_not_overfetch_or_rerank_when_off() -> None:
    # Baseline: rerank off -> fetch exactly top_k, no reorder, reranked flag False.
    kb = _RecordingKB()
    settings = RagSettings(rerank=False, rewrite_query=False, top_k=4)
    out = retrieve({"question": "q", "history": []}, _deps(kb, settings, _RerankingLLM([])))
    assert kb.k_seen == [4]
    assert out["reranked"] is False
    assert len(out["chunks"]) == 4


def test_retrieve_overfetches_and_reranks_when_on() -> None:
    # rerank on -> over-fetch the wider pool, reorder it, keep top_k, set reranked flag.
    kb = _RecordingKB()
    settings = RagSettings(rerank=True, rerank_candidates=10, rewrite_query=False, top_k=3)
    # Reverse the pool: last candidate becomes first.
    llm = _RerankingLLM(list(range(10, 0, -1)))
    out = retrieve({"question": "q", "history": []}, _deps(kb, settings, llm))
    assert kb.k_seen == [10]  # over-fetched the candidate pool, not top_k
    assert out["reranked"] is True
    assert len(out["chunks"]) == 3
    assert out["chunks"][0].text == "hit9"  # the reversed pool's new leader
