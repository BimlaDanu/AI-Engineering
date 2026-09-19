"""Offline tests for multi-query fusion (RAG-Fusion): expansion + Reciprocal Rank Fusion.

Multi-query retrieval expands a question into several diverse search queries, retrieves each
independently, and fuses the rankings so a passage found by *several* variants outranks one
that ranks highly for a single phrasing. These tests pin the pure fusion maths, the graceful
expansion fallback chain, and the off-by-default parity of the ``retrieve`` step — all without
a network, an LLM, or embeddings.
"""

from __future__ import annotations

from src.config import RagSettings
from src.core.schemas import QueryExpansion
from src.core.service import AnswerRequest
from src.core.steps import PipelineDeps, retrieve
from src.rag.retriever import RetrievedChunk, expand_queries, reciprocal_rank_fusion
from src.utils import StageTimer


def _chunk(text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={"source": "s.md"},
        vector_score=score,
        bm25_score=score,
        score=score,
    )


# --- Reciprocal Rank Fusion -------------------------------------------------
def test_rrf_rewards_agreement_across_lists() -> None:
    # "b" is retrieved by both queries (rank 1 then rank 0); "a" tops one list only.
    # RRF should rank the cross-list agreement ("b") above the single-list leader ("a").
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    fused = reciprocal_rank_fusion([[a, b], [b, c]])
    assert [c.text for c in fused] == ["b", "a", "c"]


def test_rrf_dedupes_by_text() -> None:
    a1, a2 = _chunk("a", 0.3), _chunk("a", 0.9)
    fused = reciprocal_rank_fusion([[a1], [a2]])
    assert len(fused) == 1
    # The kept representative is the higher-scoring instance (for the trace's score column).
    assert fused[0].score == 0.9


def test_rrf_limit_caps_output() -> None:
    lists = [[_chunk(x) for x in ("a", "b", "c", "d", "e")]]
    assert len(reciprocal_rank_fusion(lists, limit=3)) == 3


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list_preserves_order() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    assert [x.text for x in reciprocal_rank_fusion([[a, b, c]])] == ["a", "b", "c"]


# --- query expansion + graceful fallback ------------------------------------
class _Structured:
    def __init__(self, expansion: QueryExpansion) -> None:
        self._expansion = expansion

    def invoke(self, _messages: object) -> QueryExpansion:
        return self._expansion


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content


class _ExpandingLLM:
    """Returns a preset QueryExpansion via structured output; plain invoke = rewrite fallback."""

    def __init__(self, queries: list[str]) -> None:
        self._queries = queries

    def with_structured_output(self, schema: object, method: str | None = None) -> _Structured:
        return _Structured(QueryExpansion(queries=self._queries))

    def invoke(self, _messages: object) -> _Reply:
        return _Reply("rewritten form")


class _BrokenLLM:
    """Structured output raises; plain invoke still works (drives the rewrite fallback)."""

    def with_structured_output(self, schema: object, method: str | None = None) -> object:
        raise RuntimeError("structured output unavailable")

    def invoke(self, _messages: object) -> _Reply:
        return _Reply("rewritten form")


def test_expand_queries_returns_diverse_variants() -> None:
    llm = _ExpandingLLM(["what is attention", "self-attention mechanism", "transformer attention"])
    out = expand_queries(llm, "how does attention work", [], n=3)
    assert out == ["what is attention", "self-attention mechanism", "transformer attention"]


def test_expand_queries_dedupes_and_caps_to_n() -> None:
    llm = _ExpandingLLM(["Attention", "attention", "self-attention", "keys and queries"])
    out = expand_queries(llm, "attention", [], n=2)
    assert out == ["Attention", "self-attention"]  # case-insensitive dedupe, then capped


def test_expand_queries_falls_back_to_rewrite_on_error() -> None:
    # Structured expansion fails -> single-shot rewrite keeps recall at the baseline.
    assert expand_queries(_BrokenLLM(), "attention", [], n=3) == ["rewritten form"]


def test_expand_queries_n_le_1_uses_single_rewrite() -> None:
    # n<=1 should not even attempt multi-query; it is just a rewrite.
    assert expand_queries(_ExpandingLLM(["a", "b"]), "q", [], n=1) == ["rewritten form"]


# --- retrieve step: off-by-default parity + multi-query wiring ---------------
class _RecordingKB:
    """Records every query it is searched with and returns one chunk echoing that query."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query, subjects=None, difficulty=None, k=4, alpha=0.7, *, filters=None):
        self.queries.append(query)
        return [_chunk(f"hit::{query}")]


def _deps(kb, settings, llm) -> PipelineDeps:
    req = AnswerRequest(
        question="how does attention work",
        level="Practitioner",
        model="m",
        subjects=None,
        settings=settings,
    )
    return PipelineDeps(llm=llm, kb=kb, settings=settings, router=None, req=req, timer=StageTimer())


def test_retrieve_single_query_when_multi_query_off() -> None:
    # Baseline: multi_query off -> exactly one retrieval on the single (rewritten) query.
    kb, settings = _RecordingKB(), RagSettings(multi_query=False, rewrite_query=True)
    state = {"question": "how does attention work", "history": []}
    out = retrieve(state, _deps(kb, settings, _ExpandingLLM([])))
    assert kb.queries == ["rewritten form"]
    assert out["search_queries"] == ["rewritten form"]
    assert len(out["chunks"]) == 1


def test_retrieve_fuses_when_multi_query_on() -> None:
    # multi_query on -> one retrieval per variant, then a fused result set.
    kb = _RecordingKB()
    settings = RagSettings(multi_query=True, multi_query_count=3)
    llm = _ExpandingLLM(["q1", "q2", "q3"])
    state = {"question": "how does attention work", "history": []}
    out = retrieve(state, _deps(kb, settings, llm))
    assert kb.queries == ["q1", "q2", "q3"]
    assert out["search_queries"] == ["q1", "q2", "q3"]
    assert out["search_query"] == "q1"  # primary query for the legacy trace field
    # Three disjoint single-hit lists fuse to three chunks (capped at top_k).
    assert {c.text for c in out["chunks"]} == {"hit::q1", "hit::q2", "hit::q3"}
