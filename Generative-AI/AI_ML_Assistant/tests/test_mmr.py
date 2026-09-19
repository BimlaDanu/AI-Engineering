"""Offline tests for MMR diversity re-selection (RAG stage 2).

Maximal Marginal Relevance trims an over-fetched pool to top_k while balancing relevance
against novelty, so several near-identical passages don't all occupy the results. These pin
the pure ranking maths, the graceful embedding-failure fallback, and the off-by-default
wiring of the ``retrieve`` step — all without a real embedding backend or a vector store.
"""

from __future__ import annotations

from src.config import RagSettings
from src.core.service import AnswerRequest
from src.core.steps import PipelineDeps, retrieve
from src.rag.retriever import KnowledgeBase, RetrievedChunk, mmr_rank
from src.utils import StageTimer


def _chunk(text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={"source": "s.md"},
        vector_score=score,
        bm25_score=score,
        score=score,
    )


# --- pure MMR ranking -------------------------------------------------------
def test_mmr_empty_input() -> None:
    assert mmr_rank([], [], k=3) == []


def test_mmr_caps_at_k() -> None:
    vecs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert len(mmr_rank([0.9, 0.8, 0.7], vecs, k=2)) == 2


def test_mmr_lambda_one_is_pure_relevance_order() -> None:
    # λ=1.0 ignores diversity entirely, so the result is just relevance-sorted (ties by index).
    vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert mmr_rank([0.1, 0.9, 0.5], vecs, k=3, lambda_=1.0) == [1, 2, 0]


def test_mmr_prefers_a_novel_passage_over_a_near_duplicate() -> None:
    # item0 and item1 are identical vectors (near-duplicate); item2 is orthogonal (novel).
    # Pure relevance would pick 0 then 1; MMR (λ=0.5) picks the novel item2 second instead.
    vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    rels = [0.9, 0.8, 0.5]
    assert mmr_rank(rels, vecs, k=2, lambda_=0.5) == [0, 2]
    assert mmr_rank(rels, vecs, k=2, lambda_=1.0) == [0, 1]  # contrast: no diversity


# --- KnowledgeBase.mmr_select (embed + delegate + fallback) -----------------
class _FakeEmbeddings:
    """Returns a preset vector per text; raises if constructed to fail."""

    def __init__(self, vectors: dict[str, list[float]] | None = None, fail: bool = False) -> None:
        self._vectors = vectors or {}
        self._fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            raise RuntimeError("embedding backend down")
        return [self._vectors[t] for t in texts]


def _kb_with_embeddings(embeddings: _FakeEmbeddings) -> KnowledgeBase:
    kb = KnowledgeBase.__new__(KnowledgeBase)  # bypass Chroma/network in __init__
    kb._embeddings = embeddings
    return kb


def test_mmr_select_returns_pool_unchanged_when_not_larger_than_k() -> None:
    kb = _kb_with_embeddings(_FakeEmbeddings())
    chunks = [_chunk("a", 0.9), _chunk("b", 0.8)]
    assert kb.mmr_select(chunks, k=4) == chunks  # nothing to diversify, no embedding call


def test_mmr_select_diversifies_using_embeddings() -> None:
    a, b, c = _chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.5)
    kb = _kb_with_embeddings(_FakeEmbeddings({"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}))
    out = kb.mmr_select([a, b, c], k=2, lambda_=0.5)
    assert [x.text for x in out] == ["a", "c"]  # novel "c" beats near-duplicate "b"


def test_mmr_select_falls_back_to_score_order_on_embedding_failure() -> None:
    a, b, c = _chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.5)
    kb = _kb_with_embeddings(_FakeEmbeddings(fail=True))
    out = kb.mmr_select([a, b, c], k=2, lambda_=0.5)
    assert [x.text for x in out] == ["a", "b"]  # incoming (score) order, top-k, no crash


# --- retrieve step wiring: off by default, over-fetch + select when on ------
class _FakeKB:
    """Records search fetch_k and whether mmr_select ran; returns distinct scored chunks."""

    def __init__(self) -> None:
        self.fetch_k: int | None = None
        self.mmr_called = False

    def search(self, query, subjects=None, difficulty=None, k=4, alpha=0.7, *, filters=None):
        self.fetch_k = k
        return [_chunk(f"c{i}", 1.0 - i / 100) for i in range(k)]

    def mmr_select(self, chunks, *, k, lambda_=0.5):
        self.mmr_called = True
        return list(chunks[:k])


class _RerankLLM:
    """A minimal LLM whose structured output returns an identity reranking of the pool."""

    def with_structured_output(self, schema, method: str | None = None):
        from src.core.schemas import Reranking

        class _Structured:
            def invoke(self, _messages):
                # A permutation is resolved defensively downstream, so an empty ranking is fine
                # here — it keeps first-stage order. The point is only that rerank *ran*.
                return Reranking(ranking=[])

        return _Structured()


def _deps(kb, settings, llm=None) -> PipelineDeps:
    req = AnswerRequest(
        question="q", level="Practitioner", model="m", subjects=None, settings=settings
    )
    return PipelineDeps(
        llm=llm or object(), kb=kb, settings=settings, router=None, req=req, timer=StageTimer()
    )


def test_retrieve_does_not_diversify_by_default() -> None:
    kb, settings = _FakeKB(), RagSettings(top_k=4, rewrite_query=False)
    out = retrieve({"question": "q", "history": []}, _deps(kb, settings))
    assert kb.fetch_k == 4  # exact top_k fetch — baseline unchanged
    assert kb.mmr_called is False
    assert out["diversified"] is False


def test_retrieve_diversifies_when_mmr_on() -> None:
    kb = _FakeKB()
    settings = RagSettings(top_k=4, mmr=True, mmr_candidates=12, rewrite_query=False)
    out = retrieve({"question": "q", "history": []}, _deps(kb, settings))
    assert kb.fetch_k == 12  # over-fetched the wider pool
    assert kb.mmr_called is True
    assert out["diversified"] is True
    assert len(out["chunks"]) == 4


def test_retrieve_rerank_takes_precedence_over_mmr() -> None:
    # Both enabled: rerank wins and MMR is never invoked (alternatives, not companions).
    kb = _FakeKB()
    settings = RagSettings(top_k=4, mmr=True, rerank=True, rewrite_query=False)
    out = retrieve({"question": "q", "history": []}, _deps(kb, settings, llm=_RerankLLM()))
    assert out["reranked"] is True
    assert out["diversified"] is False
    assert kb.mmr_called is False
