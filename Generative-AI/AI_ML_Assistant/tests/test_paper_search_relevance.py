"""Offline tests for the paper-search KB relevance gate.

The KB column of the tool playground must show a passage only when it is *about* the queried
topic — not when an off-domain query shares a word with the notes. That failure mode is a
high BM25 score with a ~0 vector score (e.g. the physics topic "spin chain metallic surfaces"
matching "chain rule" / loss "surfaces" in ML notes), so the gate keys on the vector score.
Numbers below are the real hybrid scores observed on the default embeddings.
"""

from __future__ import annotations

from src.rag.retriever import RetrievedChunk
from src.ui.pages.tools import _KB_MIN_VECTOR_SCORE, _relevant_kb_hits


def _chunk(vector_score: float, bm25_score: float, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text="x",
        metadata={"source": "s.md"},
        vector_score=vector_score,
        bm25_score=bm25_score,
        score=score,
    )


def test_keyword_coincidence_is_rejected() -> None:
    # "spin chain metallic surfaces": all score from BM25, vector is 0 -> not about the topic.
    off_domain = [
        _chunk(0.0, 1.0, 0.300),
        _chunk(0.0, 0.771, 0.231),
        _chunk(0.0, 0.757, 0.227),
    ]
    assert _relevant_kb_hits(off_domain) == []


def test_weak_semantic_off_domain_is_rejected() -> None:
    # "quantum monte carlo fermions": low vector, no keyword hits -> still off-domain.
    weak = [_chunk(0.221, 0.0, 0.155), _chunk(0.212, 0.0, 0.148), _chunk(0.188, 0.0, 0.132)]
    assert _relevant_kb_hits(weak) == []


def test_in_domain_hits_are_kept() -> None:
    # "attention transformer" — the weakest in-domain top hit observed still clears the gate.
    in_domain = [
        _chunk(0.351, 1.0, 0.546),
        _chunk(0.386, 0.855, 0.526),
        _chunk(0.312, 0.666, 0.418),
    ]
    assert len(_relevant_kb_hits(in_domain)) == 3


def test_gate_uses_vector_not_blended_score() -> None:
    # High blended score but zero vector: the old blended-score gate passed this; the new one
    # must not. This is the exact bug the fix targets.
    assert _relevant_kb_hits([_chunk(0.0, 1.0, 0.30)]) == []
    # And the mirror: real semantic match with a modest blended score is kept.
    assert len(_relevant_kb_hits([_chunk(0.45, 0.2, 0.29)])) == 1


def test_limit_caps_results() -> None:
    many = [_chunk(0.5, 0.5, 0.5) for _ in range(6)]
    assert len(_relevant_kb_hits(many)) == 4


def test_threshold_boundary() -> None:
    assert len(_relevant_kb_hits([_chunk(_KB_MIN_VECTOR_SCORE, 0.0, 0.2)])) == 1
    assert _relevant_kb_hits([_chunk(_KB_MIN_VECTOR_SCORE - 0.01, 0.0, 0.2)]) == []
