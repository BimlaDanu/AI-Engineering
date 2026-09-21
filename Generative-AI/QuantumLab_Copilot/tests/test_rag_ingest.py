"""Tests for building the retrieval index.

Everything here runs with no credential, no embedding model and no Chroma. That
is a property of the module rather than a trick played on it: chunking and id
derivation are pure, and the store is reached through a protocol, so the double
below is a legitimate implementation of it and not a monkeypatch.

The tests that matter most are the boring ones. Ingesting twice must not double
the collection, and a shortened document must not leave its old tail in the
index. Both are silent when they go wrong -- retrieval keeps working and simply
returns worse passages, which no assertion about the retriever would ever catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from src.rag.ingest import (
    CHUNK_CHARACTERS,
    SHELVES,
    CorpusError,
    IngestReport,
    Note,
    build_embeddings,
    chunk,
    chunk_corpus,
    chunk_id,
    ingest,
    load_corpus,
    parse_note,
    shelf_names,
    shelf_of,
    split_frontmatter,
    stale_ids,
    topics_of,
)
from src.settings import Settings

# --------------------------------------------------------------------------
# Fixtures and doubles
# --------------------------------------------------------------------------

NOTE = """---
title: Exact solution of the transverse-field Ising chain
source: "Pfeuty, Annals of Physics 57, 79 (1970)"
arxiv: null
topics: [exact-solution, free-fermions]
---

# Exact solution of the transverse-field Ising chain

The chain is one of the few interacting models solvable in closed form.

## The gap

The gap closes linearly in the distance from the critical point.

## Why it matters

Two solvers sharing no algebra are worth more than one careful solver.
"""
"""A complete note, shaped like the real corpus documents."""


def settings(**overrides: object) -> Settings:
    """Settings that never read ``.env``, for tests that must not touch it."""
    values: dict[str, object] = {"openrouter_api_key": "test-key"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


class FakeIndex:
    """An in-memory stand-in for the Chroma collection.

    Upserts by id, exactly as Chroma does, because that behaviour is the thing
    under test: a store that appended would make the duplicate test pass while
    the real one failed.

    Attributes:
        records: Id to ``(text, metadata)``.
        calls: The sequence of operations, so their *order* can be asserted.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.records: dict[str, tuple[str, dict[str, Any]]] = {}
        self.calls: list[str] = []

    def add_documents(self, documents: list[Any], **kwargs: Any) -> list[str]:
        """Insert or replace by id."""
        self.calls.append("add")
        ids = list(kwargs.get("ids") or [str(document.id) for document in documents])
        for identifier, document in zip(ids, documents, strict=True):
            self.records[identifier] = (document.page_content, dict(document.metadata))
        return ids

    def get(self, **kwargs: Any) -> dict[str, Any]:
        """Return every stored id and its metadata."""
        self.calls.append("get")
        ids = sorted(self.records)
        return {"ids": ids, "metadatas": [self.records[key][1] for key in ids]}

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:
        """Remove by id."""
        self.calls.append("delete")
        for identifier in ids or []:
            self.records.pop(identifier, None)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A one-document corpus on disk, laid out the way ``data/`` is."""
    notes = tmp_path / "physics-notes"
    notes.mkdir()
    (notes / "pfeuty-exact-solution.md").write_text(NOTE, encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------
# Frontmatter
# --------------------------------------------------------------------------


def test_frontmatter_is_separated_from_the_body() -> None:
    fields, body = split_frontmatter(NOTE)
    assert fields["title"].startswith("Exact solution")
    assert "---" not in body


def test_a_document_with_no_fence_yields_no_fields() -> None:
    fields, body = split_frontmatter("Just prose.")
    assert fields == {}
    assert body == "Just prose."


def test_an_unterminated_fence_is_not_treated_as_frontmatter() -> None:
    # Otherwise the whole document would be parsed as metadata and indexed as an
    # empty body -- a document that retrieves nothing and reports no error.
    fields, body = split_frontmatter("---\ntitle: A\nbody with no closing fence")
    assert fields == {}
    assert body.startswith("---")


def test_a_yaml_list_continuation_line_is_not_read_as_a_key() -> None:
    fields, _ = split_frontmatter("---\ntopics:\n  - one\n  - two\n---\nBody.")
    assert list(fields) == ["topics"]


def test_the_arxiv_null_placeholder_becomes_none() -> None:
    # "null" arrives as four characters because nothing parses the YAML.
    assert parse_note(Path("a.md"), NOTE).arxiv is None


def test_an_arxiv_id_survives() -> None:
    text = NOTE.replace("arxiv: null", "arxiv: 1234.5678")
    assert parse_note(Path("a.md"), text).arxiv == "1234.5678"


def test_topics_are_normalised_to_slugs() -> None:
    text = NOTE.replace("[exact-solution, free-fermions]", "[Free Fermions, free-fermions]")
    assert parse_note(Path("a.md"), text).topics == ("free-fermions",)


def test_a_document_without_frontmatter_is_refused() -> None:
    with pytest.raises(CorpusError, match="frontmatter fence"):
        parse_note(Path("a.md"), "# Title\n\nProse.")


@pytest.mark.parametrize("key", ["title", "source", "topics"])
def test_a_document_missing_a_citation_key_is_refused(key: str) -> None:
    # Refused rather than indexed with a gap: a passage that cannot be attributed
    # is a passage the answer has to quote anonymously.
    text = NOTE.replace(f"{key}:", f"x{key}:")
    with pytest.raises(CorpusError, match=key):
        parse_note(Path("a.md"), text)


def test_a_document_with_an_empty_topic_list_is_refused() -> None:
    with pytest.raises(CorpusError, match="topics"):
        parse_note(Path("a.md"), NOTE.replace("[exact-solution, free-fermions]", "[]"))


def test_topics_round_trip_through_metadata() -> None:
    note = parse_note(Path("a.md"), NOTE)
    assert topics_of(chunk(note)[0].metadata) == note.topics


def test_metadata_with_no_topics_reads_back_as_empty() -> None:
    assert topics_of({}) == ()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_a_corpus_directory_is_read(corpus: Path) -> None:
    notes = load_corpus(corpus)
    assert len(notes) == 1
    assert notes[0].slug == "pfeuty-exact-solution"


def test_the_notes_subdirectory_may_be_named_directly(corpus: Path) -> None:
    assert load_corpus(corpus / "physics-notes")


def test_a_missing_corpus_is_reported_as_missing(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="does not exist"):
        load_corpus(tmp_path / "nowhere")


def test_an_empty_corpus_is_reported_separately(tmp_path: Path) -> None:
    # A missing directory and an empty one are different mistakes with different
    # fixes, and one error message for both sends the reader to the wrong one.
    (tmp_path / "physics-notes").mkdir()
    with pytest.raises(CorpusError, match="no Markdown"):
        load_corpus(tmp_path)


def test_the_real_corpus_loads(tmp_path: Path) -> None:
    # An integration check against the committed documents: the parser and the
    # corpus contract are maintained separately and can drift apart.
    notes = load_corpus("data/corpus")
    assert len(notes) >= 3
    assert all(note.title and note.source and note.topics for note in notes)


def test_settings_are_only_consulted_when_no_directory_is_given(corpus: Path) -> None:
    # Reading a named directory must not require a credential, which is what
    # lets this whole file run without one.
    assert load_corpus(corpus, settings=None)


def test_the_configured_corpus_path_is_used_by_default(corpus: Path) -> None:
    assert load_corpus(settings=settings(corpus_path=str(corpus)))


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_a_note_is_split_on_its_headings() -> None:
    chunks = chunk(parse_note(Path("a.md"), NOTE))
    sections = [piece.metadata["section"] for piece in chunks]
    assert "The gap" in sections
    assert "Why it matters" in sections


def test_a_heading_stays_in_the_text_of_its_chunk() -> None:
    # The heading is often the most retrievable sentence in a section; stripping
    # it embeds the body without the phrase a user is most likely to type.
    chunks = chunk(parse_note(Path("a.md"), NOTE))
    gap = next(piece for piece in chunks if piece.metadata["section"] == "The gap")
    assert "The gap" in gap.page_content


def test_a_chunk_carries_everything_a_citation_needs() -> None:
    piece = chunk(parse_note(Path("notes/a.md"), NOTE))[0]
    assert piece.metadata["title"].startswith("Exact solution")
    assert "Pfeuty" in piece.metadata["source"]
    assert piece.metadata["path"] == "notes/a.md"
    assert piece.metadata["document"] == "a"


def test_every_metadata_value_is_a_scalar() -> None:
    # Chroma raises on a list value. Catching it here rather than at the first
    # real ingest, which is the run that costs money.
    for piece in chunk(parse_note(Path("a.md"), NOTE)):
        for key, value in piece.metadata.items():
            assert isinstance(value, (str, int, float, bool)), f"{key} is {type(value)}"


def test_chunk_ids_are_derived_from_the_document_and_position() -> None:
    note = parse_note(Path("a.md"), NOTE)
    assert chunk_id(note, 0) == "a#000"
    assert [piece.id for piece in chunk(note)] == [
        chunk_id(note, position) for position in range(len(chunk(note)))
    ]


def test_chunking_is_deterministic() -> None:
    note = parse_note(Path("a.md"), NOTE)
    first = [(piece.id, piece.page_content) for piece in chunk(note)]
    second = [(piece.id, piece.page_content) for piece in chunk(note)]
    assert first == second


def test_no_chunk_exceeds_the_size_bound() -> None:
    for piece in chunk_corpus(load_corpus("data/corpus")):
        assert len(piece.page_content) <= CHUNK_CHARACTERS


def test_the_real_corpus_produces_unique_ids() -> None:
    # A collision would make two chunks share a slot, so one of them would be
    # silently absent from the index.
    ids = [piece.id for piece in chunk_corpus(load_corpus("data/corpus"))]
    assert len(ids) == len(set(ids))


def test_a_long_section_is_split_further() -> None:
    long_note = Note(
        path=Path("long.md"),
        title="T",
        source="S",
        arxiv=None,
        topics=("a",),
        body="# Heading\n\n" + ("The gap closes linearly. " * 200),
    )
    chunks = chunk(long_note)
    assert len(chunks) > 1
    assert all(len(piece.page_content) <= CHUNK_CHARACTERS for piece in chunks)


def test_an_empty_note_produces_no_chunks() -> None:
    empty = Note(Path("e.md"), "T", "S", None, ("a",), "   \n")
    assert chunk(empty) == ()


# --------------------------------------------------------------------------
# Writing and reconciling
# --------------------------------------------------------------------------


def test_an_ingest_writes_every_chunk(corpus: Path) -> None:
    index = FakeIndex()
    report = ingest(index, root=corpus)
    assert report.notes == 1
    assert report.chunks == len(index.records) > 0


def test_ingesting_twice_does_not_double_the_index(corpus: Path) -> None:
    # The load-bearing test in this file. Duplicate chunks do not raise, they
    # just take retrieval slots the answer needed for something else.
    index = FakeIndex()
    first = ingest(index, root=corpus)
    after_first = dict(index.records)
    second = ingest(index, root=corpus)
    assert second.chunks == first.chunks
    assert index.records == after_first


def test_a_shortened_document_loses_its_old_tail(corpus: Path) -> None:
    index = FakeIndex()
    ingest(index, root=corpus)
    before = len(index.records)

    trimmed = NOTE.split("## Why it matters")[0]
    (corpus / "physics-notes" / "pfeuty-exact-solution.md").write_text(trimmed, encoding="utf-8")
    report = ingest(index, root=corpus)

    assert len(index.records) < before
    assert report.removed
    assert all(identifier not in index.records for identifier in report.removed)


def test_a_deleted_document_stops_being_retrievable(corpus: Path) -> None:
    index = FakeIndex()
    ingest(index, root=corpus)
    (corpus / "physics-notes" / "second.md").write_text(
        NOTE.replace("title: Exact", "title: Second"), encoding="utf-8"
    )
    ingest(index, root=corpus)
    documents = {metadata["document"] for _, metadata in index.records.values()}
    assert documents == {"pfeuty-exact-solution", "second"}

    (corpus / "physics-notes" / "second.md").unlink()
    ingest(index, root=corpus)
    documents = {metadata["document"] for _, metadata in index.records.values()}
    assert documents == {"pfeuty-exact-solution"}


def test_the_write_happens_before_the_delete(corpus: Path) -> None:
    # The other order empties the index first, and a failed ingest then leaves it
    # empty. Serving a stale index beats serving half of one.
    index = FakeIndex()
    ingest(index, root=corpus)
    assert index.calls.index("add") < index.calls.index("get")


def test_nothing_is_deleted_when_nothing_is_stale(corpus: Path) -> None:
    index = FakeIndex()
    ingest(index, root=corpus)
    assert "delete" not in index.calls


def test_stale_ids_ignores_what_is_being_kept() -> None:
    index = FakeIndex()
    index.records = {"a#000": ("text", {}), "b#000": ("text", {})}
    assert stale_ids(index, ["a#000"]) == ("b#000",)


def test_an_unparseable_document_stops_the_ingest(corpus: Path) -> None:
    # Rather than being skipped: an index missing one document looks complete and
    # answers with a hole in it.
    (corpus / "physics-notes" / "broken.md").write_text("no frontmatter", encoding="utf-8")
    with pytest.raises(CorpusError):
        ingest(FakeIndex(), root=corpus)


def test_a_corpus_of_empty_documents_is_refused(tmp_path: Path) -> None:
    notes = tmp_path / "physics-notes"
    notes.mkdir()
    (notes / "empty.md").write_text(NOTE.split("# Exact")[0], encoding="utf-8")
    with pytest.raises(CorpusError, match="no chunks"):
        ingest(FakeIndex(), root=tmp_path)


def test_a_computed_looking_number_is_reported_not_indexed_silently(corpus: Path) -> None:
    (corpus / "physics-notes" / "pfeuty-exact-solution.md").write_text(
        NOTE.replace("closes linearly", "closes at 0.49417"), encoding="utf-8"
    )
    report = ingest(FakeIndex(), root=corpus)
    assert report.suspect


def test_a_clean_corpus_reports_nothing_suspect(corpus: Path) -> None:
    assert ingest(FakeIndex(), root=corpus).suspect == ()


def test_the_report_names_every_count_including_the_zeroes() -> None:
    # "removed=0" is information; omitting it makes a stale index read like a
    # reconciled one.
    summary = IngestReport(notes=2, chunks=9, removed=(), suspect=()).summary()
    assert "2 notes" in summary
    assert "9 chunks" in summary
    assert "0 stale" in summary


# --------------------------------------------------------------------------
# The embedding endpoint
# --------------------------------------------------------------------------


def embeddings(**overrides: object) -> OpenAIEmbeddings:
    """Build the embedding client and assert the concrete type its fields need.

    ``build_embeddings`` is declared as returning the abstract ``Embeddings``, so
    the assertion is what lets these tests read ``openai_api_base`` -- and it
    doubles as a check that the return type has not quietly become something
    whose endpoint is configured differently.
    """
    client = build_embeddings(settings(**overrides))
    assert isinstance(client, OpenAIEmbeddings)
    return client


def test_embeddings_default_to_the_chat_gateway() -> None:
    assert embeddings().openai_api_base == "https://openrouter.ai/api/v1"


def test_the_embedding_endpoint_can_be_moved_independently() -> None:
    # The setting that exists because chat and embeddings need not be served by
    # the same provider.
    client = embeddings(
        embedding_base_url="https://api.example.invalid/v1",
        embedding_api_key="other-key",
    )
    assert client.openai_api_base == "https://api.example.invalid/v1"
    # The field is typed as a secret or a callable returning one; only the first
    # is what this project ever puts there.
    assert isinstance(client.openai_api_key, SecretStr)
    assert client.openai_api_key.get_secret_value() == "other-key"


def test_the_embedding_model_comes_from_settings() -> None:
    assert embeddings(embedding_model="a/b").model == "a/b"


# --------------------------------------------------------------------------
# Shelves: which knowledge base a note belongs to
# --------------------------------------------------------------------------


def test_every_chunk_carries_the_shelf_it_came_from() -> None:
    # The shelf is what makes "which knowledge base answered?" a question the
    # interface can answer, so it has to survive into the index.
    for piece in chunk_corpus(load_corpus("data/corpus")):
        assert shelf_of(piece.metadata) in shelf_names()


def test_a_note_takes_its_shelf_from_its_directory() -> None:
    note = parse_note(
        Path("data/corpus/quantum-computing/anything.md"),
        "---\ntitle: T\nsource: S\ntopics: [a]\n---\nBody.",
    )
    assert note.shelf == "quantum-computing"


def test_both_declared_shelves_are_loaded() -> None:
    shelves = {note.shelf for note in load_corpus("data/corpus")}
    assert shelves == set(shelf_names())


def test_a_root_pointing_at_one_shelf_loads_only_that_one() -> None:
    notes = load_corpus("data/corpus/quantum-computing")
    assert notes
    assert {note.shelf for note in notes} == {"quantum-computing"}


def test_an_undeclared_directory_of_notes_is_refused(tmp_path: Path) -> None:
    # Indexing it would give the agent a body of knowledge it cannot filter by or
    # cite consistently; ignoring it would leave notes nothing ever searches.
    (tmp_path / "physics-notes").mkdir()
    (tmp_path / "physics-notes" / "one.md").write_text(
        "---\ntitle: T\nsource: S\ntopics: [a]\n---\nBody.", encoding="utf-8"
    )
    (tmp_path / "lecture-slides").mkdir()
    (tmp_path / "lecture-slides" / "two.md").write_text("---\ntitle: T\n---\nx", encoding="utf-8")
    with pytest.raises(CorpusError, match="undeclared"):
        load_corpus(tmp_path)


def test_two_notes_with_the_same_filename_are_refused(tmp_path: Path) -> None:
    # Chunk ids are slug-derived, so this would have each document overwrite half
    # of the other's chunks in the index -- silently.
    body = "---\ntitle: T\nsource: S\ntopics: [a]\n---\nBody."
    for shelf in shelf_names():
        (tmp_path / shelf).mkdir()
        (tmp_path / shelf / "same-name.md").write_text(body, encoding="utf-8")
    with pytest.raises(CorpusError, match="share the filename"):
        load_corpus(tmp_path)


def test_a_chunk_from_an_older_index_reads_as_an_unknown_shelf() -> None:
    # Empty rather than a default: "unknown" is the truth about a chunk written
    # before shelves existed, and a confident wrong shelf is worse.
    assert shelf_of({}) == ""


def test_every_shelf_declares_what_it_covers() -> None:
    # The text is read by the routing prompt and by the capabilities answer, so an
    # empty one would silently weaken both.
    for shelf in SHELVES:
        assert shelf.name and shelf.title and len(shelf.covers) > 20
