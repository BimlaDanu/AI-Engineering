"""Tests for BM25 scoring and front-matter parsing (no network, no embeddings)."""

from types import SimpleNamespace

import pytest

from src.rag.ingest import load_documents, parse_front_matter
from src.rag.retriever import BM25, KnowledgeBase


def _kb_without_chroma(texts, metas):
    """Build a KnowledgeBase with its in-memory indexes set but no Chroma store.

    Bypasses ``__init__`` (which would open a persisted collection) so hybrid-search fusion
    can be exercised fully offline by stubbing the vector-hit source per test.
    """
    kb = object.__new__(KnowledgeBase)
    kb._all_texts = list(texts)
    kb._all_metas = list(metas)
    kb._bm25 = BM25(texts)
    kb._bm25_index = {t: i for i, t in enumerate(texts)}
    return kb


def test_bm25_ranks_matching_doc_higher():
    corpus = [
        "Gradient descent updates weights using the loss gradient.",
        "Transformers use self-attention to mix token information.",
        "Support vector machines maximise the classification margin.",
    ]
    scores = BM25(corpus).scores("what is self-attention in transformers")
    assert scores[1] == max(scores)
    assert scores[1] > 0


def test_bm25_no_match_scores_zero():
    scores = BM25(["completely unrelated text"]).scores("quantum chromodynamics")
    assert scores == [0.0]


def test_bm25_empty_corpus():
    assert BM25([]).scores("anything") == []


def test_hybrid_union_surfaces_keyword_only_chunk(monkeypatch):
    # Regression: a chunk that matches strongly on keywords but falls outside the vector
    # top-k must still surface. The old vector-then-rerank design could never reach it.
    texts = [
        "Gradient descent updates weights using the loss gradient.",
        "Bayesian optimization tunes hyperparameters with a Gaussian process surrogate.",
    ]
    metas = [{"subject": "ml"}, {"subject": "ml"}]
    kb = _kb_without_chroma(texts, metas)

    # Vector search returns ONLY the unrelated first chunk (simulating the keyword chunk
    # being outside the vector top-fetch_k).
    vec_doc = SimpleNamespace(page_content=texts[0], metadata=metas[0])
    monkeypatch.setattr(kb, "_vector_hits", lambda *a, **k: [(vec_doc, 0.2)])

    results = kb.search("bayesian optimization gaussian process", k=2)
    found = {c.text for c in results}
    assert texts[1] in found  # keyword-only chunk now reachable
    keyword_chunk = next(c for c in results if c.text == texts[1])
    assert keyword_chunk.bm25_score > 0
    assert keyword_chunk.vector_score == 0.0  # it was not a vector hit


def test_hybrid_bm25_candidate_respects_subject_filter(monkeypatch):
    # A BM25-only candidate outside the requested subject must not leak past the filter.
    texts = [
        "Gradient descent updates weights using the loss gradient.",
        "Bayesian optimization tunes hyperparameters with a Gaussian process surrogate.",
    ]
    metas = [{"subject": "ml"}, {"subject": "dl"}]
    kb = _kb_without_chroma(texts, metas)
    vec_doc = SimpleNamespace(page_content=texts[0], metadata=metas[0])
    monkeypatch.setattr(kb, "_vector_hits", lambda *a, **k: [(vec_doc, 0.2)])

    results = kb.search("bayesian optimization gaussian process", subjects=["ml"], k=2)
    assert texts[1] not in {c.text for c in results}  # dl chunk filtered out


def test_search_dimension_mismatch_raises_actionable_error(monkeypatch):
    kb = _kb_without_chroma(["some ml text"], [{"subject": "ml"}])

    def _raise(*_a, **_k):
        raise ValueError("Collection expecting embedding with dimension of 384, got 1536")

    kb._store = SimpleNamespace(similarity_search_with_score=_raise)
    with pytest.raises(RuntimeError, match="make ingest"):
        kb.search("anything", k=2)


def test_parse_front_matter():
    text = "---\ntitle: Embeddings\nsubject: overlap\ndifficulty: beginner\n---\nBody text."
    meta, body = parse_front_matter(text)
    assert meta["subject"] == "overlap"
    assert meta["title"] == "Embeddings"
    assert body == "Body text."


def test_parse_no_front_matter_passthrough():
    meta, body = parse_front_matter("Just a document.")
    assert meta == {}
    assert body == "Just a document."


def test_parse_unclosed_front_matter_passthrough():
    text = "---\ntitle: Broken"
    meta, body = parse_front_matter(text)
    assert meta == {}
    assert body == text


def test_source_is_always_the_filename(tmp_path):
    # A front-matter `source:` line must not shadow the filename (used for stats and
    # the index-freshness check); it is preserved as `origin` instead.
    folder = tmp_path / "dl"
    folder.mkdir()
    (folder / "note.md").write_text(
        "---\ntitle: A Note\nsubject: dl\nsource: In-house study note\n---\nBody.",
        encoding="utf-8",
    )
    (doc,) = load_documents(tmp_path)
    assert doc.metadata["source"] == "note.md"
    assert doc.metadata["origin"] == "In-house study note"
    assert doc.metadata["subject"] == "dl"
