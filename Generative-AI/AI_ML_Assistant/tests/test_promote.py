"""Offline tests for KB promotion (:mod:`src.rag.promote`).

The promotion layer is pure — capture, id, Markdown serialisation, and dedup'd file writes,
no network or live index — so these exercise it directly and round-trip a promoted note back
through the ingest loader to prove the A (persist-to-source) half actually re-indexes with the
right subject and metadata.
"""

from __future__ import annotations

from src.rag.ingest import load_documents
from src.rag.promote import (
    EXTERNAL_MARKER,
    chunk_to_markdown,
    external_candidates,
    is_on_domain_passage,
    promote_chunks,
    promoted_dir,
    promotion_id,
)
from src.rag.retriever import RetrievedChunk


def _chunk(**meta) -> RetrievedChunk:
    return RetrievedChunk(
        text=meta.pop("text", "Some passage text."),
        metadata=meta,
        vector_score=0.0,
        bm25_score=0.0,
        score=0.0,
    )


def test_external_candidates_keeps_only_external_and_strips_display_fields() -> None:
    sources = [
        {"ref": 1, "score": 0.9, "source": "rag.md", "difficulty": "beginner"},
        {
            "ref": 2,
            "score": 0.4,
            "source": "arXiv:1234",
            "difficulty": EXTERNAL_MARKER,
            "origin": "https://arxiv.org/abs/1234",
            "title": "A paper",
        },
    ]
    contexts = ["KB chunk text", "External paper text"]
    cands = external_candidates(sources, contexts)
    assert len(cands) == 1
    only = cands[0]
    assert only.text == "External paper text"
    assert "ref" not in only.metadata and "score" not in only.metadata
    assert only.metadata["source"] == "arXiv:1234"


def test_external_candidates_rejects_misaligned_sources_and_contexts() -> None:
    """The sources/contexts alignment is an invariant: a length mismatch fails loudly rather
    than silently dropping the unpaired passages."""
    import pytest

    with pytest.raises(ValueError):
        external_candidates(
            [{"difficulty": EXTERNAL_MARKER}, {"difficulty": EXTERNAL_MARKER}],
            ["only one context"],
        )


def test_is_on_domain_passage_keeps_ml_and_flags_physics() -> None:
    """The promote panel's tidy-up filter: ML/AI passages pass, off-topic ones are flagged."""
    assert is_on_domain_passage(
        "Transformers use self-attention and gradient descent to train neural networks.",
        {"title": "Attention Is All You Need"},
    )
    # A particle-physics paper (the kind noisy arXiv ranking drags in) carries no ML/AI
    # vocabulary in title or body, so it's flagged off-topic.
    assert not is_on_domain_passage(
        "We measure the branching ratio of the Bs0 to mu+ mu- decay at the LHC.",
        {"title": "Observation of the rare Bs0 decay"},
    )


def test_is_on_domain_passage_matches_vocabulary_in_the_title_alone() -> None:
    """Domain signal in the title counts even when the body excerpt is generic."""
    assert is_on_domain_passage(
        "This paper presents new results on a benchmark task.",
        {"title": "A survey of reinforcement learning"},
    )


def test_is_on_domain_passage_tolerates_missing_title() -> None:
    assert is_on_domain_passage("An LLM fine-tuning recipe with LoRA.", {})
    assert not is_on_domain_passage("A note about medieval trade routes.", {})


def test_promotion_id_prefers_origin_so_same_domain_results_do_not_collide() -> None:
    a = _chunk(source="web:arxiv.org", origin="https://arxiv.org/abs/1111")
    b = _chunk(source="web:arxiv.org", origin="https://arxiv.org/abs/2222")
    assert promotion_id(a) != promotion_id(b)  # would collide if keyed on `source`


def test_promotion_id_is_a_bounded_safe_slug() -> None:
    pid = promotion_id(_chunk(origin="https://example.com/" + "x" * 500))
    assert pid and len(pid) <= 80
    assert all(c.islower() or c.isdigit() or c == "-" for c in pid)
    assert not pid.startswith("-") and not pid.endswith("-")


def test_chunk_to_markdown_has_ingest_compatible_front_matter() -> None:
    md = chunk_to_markdown(
        _chunk(
            text="Body.",
            title="Attention",
            topic="arXiv paper",
            origin="https://arxiv.org/abs/1706.03762",
            year="2017",
        ),
        subject="dl",
    )
    assert md.startswith("---")
    assert "subject: dl" in md
    assert f"difficulty: {EXTERNAL_MARKER}" in md
    assert "source: https://arxiv.org/abs/1706.03762" in md  # provenance -> `origin` on ingest
    assert md.rstrip().endswith("Body.")


def test_promote_chunks_writes_then_dedups(tmp_path) -> None:
    chunk = _chunk(source="arXiv:9", origin="https://arxiv.org/abs/9", title="T")
    first = promote_chunks([chunk], subject="ai", data_dir=tmp_path)
    assert first.n_written == 1 and first.n_skipped == 0
    assert (promoted_dir(tmp_path) / first.written[0]).exists()
    # Promoting the same passage again is a no-op (idempotent across sessions).
    second = promote_chunks([chunk], subject="ai", data_dir=tmp_path)
    assert second.n_written == 0 and second.n_skipped == 1


def test_promoted_note_round_trips_through_the_ingest_loader(tmp_path) -> None:
    promote_chunks(
        [
            _chunk(
                text="Transformers use self-attention.",
                title="Attn",
                origin="https://arxiv.org/abs/1706.03762",
            )
        ],
        subject="dl",
        data_dir=tmp_path,
    )
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["subject"] == "dl"  # front-matter subject respected over folder name
    assert doc.metadata["difficulty"] == EXTERNAL_MARKER
    assert doc.metadata["origin"] == "https://arxiv.org/abs/1706.03762"
    assert "self-attention" in doc.page_content
