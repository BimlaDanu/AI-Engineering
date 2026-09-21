"""Tests for the BM25 keyword half of retrieval.

Unlike the vector half, this one needs no model and no credential: BM25 is
arithmetic over the corpus, so the tests build small indexes by hand and assert
on the arithmetic. That is the point of writing it rather than installing it.

Three properties carry the file. A rare word must outweigh a common one, or the
keyword half is just word counting. Citation metadata must be searchable, because
an author's name lives in frontmatter and is never embedded -- that is the gap
this module exists to close. And a query sharing nothing with the corpus must
return *nothing*, so an off-topic question cannot be handed passages to refuse
with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from src.rag.lexical import Bm25, build_index, indexed_text, tokenise

# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


def chunk(identifier: str, text: str, **metadata: Any) -> Document:
    """A stored chunk, with the metadata ingestion attaches to every one."""
    defaults: dict[str, Any] = {
        "document": identifier.split("#")[0],
        "shelf": "physics-notes",
        "title": "Exact solution of the transverse-field Ising chain",
        "source": "Pfeuty, Annals of Physics 57, 79 (1970)",
        "arxiv": "",
        "section": "The gap",
    }
    defaults.update(metadata)
    return Document(id=identifier, page_content=text, metadata=defaults)


# A corpus small enough that every score below can be reasoned about by hand.
CHUNKS = [
    chunk("gap#000", "The gap closes linearly in the distance from the critical point."),
    chunk("jw#001", "The Jordan-Wigner transformation maps spins onto spinless fermions."),
    chunk("anneal#002", "Annealing must be slow compared with the inverse gap squared."),
]


# --------------------------------------------------------------------------
# Tokenising: what counts as a word here, and what does not
# --------------------------------------------------------------------------


def test_tokenising_keeps_digits_and_identifiers() -> None:
    # The opposite choice from content_words, and deliberately so: an identifier
    # is the whole of some queries, and a stemmed identifier is not one.
    assert tokenise("arXiv:1802.06002") == ["arxiv", "1802.06002"]


def test_tokenising_keeps_short_words_that_content_words_drops() -> None:
    # "h" carries no meaning, so the overlap test drops it. Here it is a token
    # like any other -- and BM25 will score it near zero on its own, which is the
    # right way to handle a word that appears everywhere.
    assert tokenise("h") == ["h"]


def test_tokenising_drops_stopwords() -> None:
    assert tokenise("what is the gap") == ["gap"]


def test_tokenising_does_not_stem() -> None:
    # Matching exactly is this half's job; paraphrase and morphology are the
    # vector half's. Stemming here would break the identifiers above.
    assert tokenise("closes closed") == ["closes", "closed"]


def test_tokenising_sees_through_an_invisible_character() -> None:
    # Normalised the same way the injection screen normalises, so a word hidden
    # from the screen cannot reappear as a searchable token.
    assert tokenise("Jordan​Wigner") == ["jordanwigner"]


def test_a_query_of_nothing_but_stopwords_has_no_tokens() -> None:
    assert tokenise("But why?") == []


# --------------------------------------------------------------------------
# The metadata is searchable, which is the whole reason this module exists
# --------------------------------------------------------------------------


def test_the_indexed_text_carries_the_citation_fields() -> None:
    text = indexed_text(CHUNKS[0])
    assert "Pfeuty" in text
    assert "Exact solution" in text


def test_an_author_is_searchable_although_the_prose_never_names_them() -> None:
    # The body says nothing about Pfeuty -- the citation does, in frontmatter that
    # the vector index never embeds. Without this, "which note cites Pfeuty?" is
    # unanswerable by any amount of similarity search.
    index = Bm25.of(CHUNKS)
    assert "pfeuty" not in CHUNKS[0].page_content.lower()
    hits = index.search("Pfeuty", k=3)
    assert hits
    assert all(document.metadata["source"].startswith("Pfeuty") for document, _ in hits)


# --------------------------------------------------------------------------
# The scoring: rarity is what makes this better than counting words
# --------------------------------------------------------------------------


def test_a_rare_word_outweighs_a_common_one() -> None:
    index = Bm25.of(CHUNKS)
    # "gap" is in two of three chunks; "annealing" is in one.
    assert index.weight("annealing") > index.weight("gap")


def test_a_word_in_every_chunk_is_worth_almost_nothing() -> None:
    index = Bm25.of([chunk(f"c#{n}", "the gap") for n in range(5)])
    assert index.weight("gap") < 0.2


def test_a_word_the_corpus_does_not_have_scores_zero() -> None:
    index = Bm25.of(CHUNKS)
    assert index.score(tokenise("pizza"), 0) == 0.0


def test_the_best_hit_for_a_distinctive_query_is_the_right_chunk() -> None:
    index = Bm25.of(CHUNKS)
    hits = index.search("Jordan-Wigner fermions", k=3)
    assert hits[0][0].id == "jw#001"


def test_search_returns_nothing_when_the_query_shares_no_vocabulary() -> None:
    # An empty result rather than three weak ones: the caller must be able to
    # refuse, and a list of near-zero scores is not a refusal.
    index = Bm25.of(CHUNKS)
    assert index.search("pizza in Vilnius", k=3) == []


def test_search_returns_nothing_for_a_query_of_only_stopwords() -> None:
    index = Bm25.of(CHUNKS)
    assert index.search("But why?", k=3) == []


def test_search_is_ordered_best_first_and_honours_the_limit() -> None:
    index = Bm25.of(CHUNKS)
    hits = index.search("gap", k=1)
    assert len(hits) == 1
    scores = [score for _, score in index.search("gap", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_an_empty_corpus_searches_without_dividing_by_zero() -> None:
    index = Bm25.of([])
    assert index.average_length == 0.0
    assert index.search("gap", k=3) == []


# --------------------------------------------------------------------------
# Building from the corpus on disk
# --------------------------------------------------------------------------


def test_the_index_is_built_from_the_real_corpus_without_a_credential() -> None:
    # No embedding model, no key, no Chroma: the keyword half is why retrieval can
    # degrade to something rather than to nothing when the index is missing.
    index = build_index(Path("data/corpus"))
    assert index is not None
    assert len(index.documents) > 1
    assert index.search("Jordan-Wigner", k=2)


def test_a_missing_corpus_disables_keyword_search_rather_than_raising(tmp_path: Path) -> None:
    # None, not an exception: the vector half can still answer, and a missing
    # corpus directory should cost recall rather than the question.
    assert build_index(tmp_path / "absent") is None


def test_any_failure_at_all_disables_keyword_search(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not just a missing directory. Reading the default corpus path means resolving
    # the settings, so a run with no credential arrives here as a validation error
    # -- which is how `ask` once turned a missing key into a crashed page instead
    # of a degraded search. Every exception is absorbed, so the test raises one
    # that is nothing like a filesystem error.
    def unresolvable(*args: object, **kwargs: object) -> None:
        raise ValueError("openrouter_api_key: field required")

    monkeypatch.setattr("src.rag.lexical.load_corpus", unresolvable)
    assert build_index() is None
