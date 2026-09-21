"""Tests that hold the knowledge base to the contract in ``data/README.md``.

A corpus is data, so the instinct is that there is nothing here to test. But
retrieval quality is mostly decided by the corpus, and the two worst mistakes are
silent ones. A document with no frontmatter still retrieves -- it just answers with
an unattributed passage, and nobody can tell. A computed number written into a note
is retrievable and looks exactly like one the solver produced and verified, which
quietly voids the claim this whole project rests on.

Neither shows up in a test of the retriever, because in both cases the retriever
works perfectly. They have to be caught at the document.

The frontmatter reader below is hand-rolled rather than using ``yaml``: PyYAML
reaches us only as a transitive dependency of ``chromadb``, and importing it
directly would mean relying on a package :file:`pyproject.toml` never declares.
This frontmatter is simple enough that fifteen lines cover it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CORPUS = Path("data/corpus")
SHELVES = ("physics-notes", "quantum-computing", "applications")
"""The knowledge bases, written out rather than imported.

Deliberate duplication of :data:`src.rag.ingest.SHELVES`. These tests hold the
*repository* to its contract, so importing the declaration would make them agree
with whatever the code currently says -- including with a shelf somebody renamed
and forgot to move the notes for.
"""

NOTES = CORPUS / SHELVES[0]

REQUIRED_KEYS = ("title", "source", "topics")
"""Frontmatter keys ingestion attaches to every chunk as citable metadata."""

MIN_BODY_CHARACTERS = 400
"""Shortest body worth indexing.

Below roughly this length a document is smaller than one chunk, so it competes
with the real documents for a retrieval slot while carrying less than one of them.
"""

PRECISE_DECIMAL = re.compile(r"\d\.\d{3,}")
"""A decimal with three or more places, which is the shape of a computed result.

A proxy for "no computed numbers", and the one version of that rule a test can
actually check. Theory values in this field are written as exact fractions --
``1/4``, ``1/8`` -- whereas ``0.4941`` is something a solver produced. False
positives are possible; the fix is to write the value as a fraction or explain its
provenance in prose, not to loosen this pattern.
"""


def documents() -> list[Path]:
    """Every Markdown document in the committed corpus, across every shelf.

    Returns:
        The note files, sorted for a stable test order. Everything under
        ``data/`` is committed, so this is the same set on every checkout --
        which is what lets the tests below be parametrised over it at collection
        time. Fetched documents are cached outside ``data/`` and are deliberately
        not asserted about: they may be absent.
    """
    return sorted(path for shelf in SHELVES for path in (CORPUS / shelf).glob("*.md"))


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Separate YAML frontmatter from the body.

    Args:
        text: The full document.

    Returns:
        A ``(frontmatter, body)`` pair. The frontmatter is empty when the document
        does not open with a ``---`` fence, which is the failure the tests below
        are looking for. Values are left as raw strings -- these tests care
        whether a key is present and non-empty, not what it parses to.
    """
    if not text.startswith("---\n"):
        return {}, text
    _, _, rest = text.partition("---\n")
    block, fence, body = rest.partition("\n---")
    fields: dict[str, str] = {}
    if fence:
        for line in block.splitlines():
            key, colon, value = line.partition(":")
            if colon and not key.startswith((" ", "#")):
                fields[key.strip()] = value.strip()
    return fields, body


# --- the corpus exists ----------------------------------------------------


def test_the_corpus_directory_is_present() -> None:
    # Checked separately from its contents: a missing directory and an empty one
    # are different mistakes, and the parametrised tests below silently pass on
    # an empty collection.
    assert NOTES.is_dir(), f"{NOTES} is missing; see data/README.md"


def test_the_corpus_is_not_empty() -> None:
    assert documents(), f"no Markdown documents under {CORPUS}"


# --- every knowledge base exists and is populated -------------------------


@pytest.mark.parametrize("shelf", SHELVES)
def test_every_shelf_is_a_directory_with_notes_in_it(shelf: str) -> None:
    # A shelf the router can choose but nothing was filed on is worse than no
    # shelf: retrieval restricted to it returns nothing, and the widening that
    # rescues the answer hides the mistake.
    directory = CORPUS / shelf
    assert directory.is_dir(), f"{directory} is missing; see data/README.md"
    assert list(directory.glob("*.md")), f"{directory} holds no notes"


def test_no_notes_sit_outside_a_shelf() -> None:
    # `load_corpus` refuses an undeclared directory, so this catches the same
    # mistake one step earlier -- at the file, where it is obvious what to do.
    stray = [
        path
        for path in CORPUS.rglob("*.md")
        if path.parent.name not in SHELVES and path.parent != CORPUS
    ]
    assert not stray, f"notes in a directory no shelf declares: {stray}"


def test_no_two_notes_share_a_filename() -> None:
    # Chunk ids are derived from the filename, so a clash across shelves would
    # have each document overwrite half of the other's chunks in the index.
    names = [path.stem for path in documents()]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    assert not duplicated, f"filenames used on more than one shelf: {duplicated}"


def test_the_quantum_computing_shelf_covers_the_subjects_it_claims_to() -> None:
    # The shelf exists because the chain is the standard toy model of quantum
    # computing. These are the topics that claim makes, so a shelf missing one of
    # them is a shelf whose description over-promises.
    topics: set[str] = set()
    for path in (CORPUS / "quantum-computing").glob("*.md"):
        fields, _ = split_frontmatter(path.read_text(encoding="utf-8"))
        topics.update(
            topic.strip().strip("\"'") for topic in fields["topics"].strip("[]").split(",")
        )
    for claimed in ("vqe", "qaoa", "quantum-annealing", "quantum-circuit", "hardware"):
        assert claimed in topics, f"no note on the quantum-computing shelf declares {claimed!r}"


def test_the_applications_shelf_covers_the_subjects_it_claims_to() -> None:
    # The shelf exists so that "what is this good for in business?" is answered from
    # notes rather than refused, and the router's vocabulary now admits those
    # questions. Each topic below is a family of question the gate lets through, so a
    # missing one is a question admitted to a shelf that cannot answer it -- which is
    # worse than the refusal it replaced.
    topics: set[str] = set()
    for path in (CORPUS / "applications").glob("*.md"):
        fields, _ = split_frontmatter(path.read_text(encoding="utf-8"))
        topics.update(
            topic.strip().strip("\"'") for topic in fields["topics"].strip("[]").split(",")
        )
    for claimed in (
        "combinatorial-optimisation",
        "business-applications",
        "portfolio-optimisation",
        "np-hard",
        "machine-learning",
        "ising-machines",
        "quantum-algorithms",
        "research-and-development",
    ):
        assert claimed in topics, f"no note on the applications shelf declares {claimed!r}"


# --- every document is citable -------------------------------------------


@pytest.mark.parametrize("path", documents(), ids=lambda p: p.stem)
def test_a_document_carries_the_metadata_a_citation_needs(path: Path) -> None:
    fields, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    assert fields, f"{path} has no frontmatter fence"
    for key in REQUIRED_KEYS:
        assert fields.get(key), f"{path} is missing frontmatter key {key!r}"


@pytest.mark.parametrize("path", documents(), ids=lambda p: p.stem)
def test_a_document_declares_at_least_one_topic(path: Path) -> None:
    # Topics become filter metadata: a document with none can be found by
    # similarity but never narrowed to.
    fields, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    topics = fields["topics"].strip("[]").split(",")
    assert [topic for topic in topics if topic.strip()], f"{path} lists no topics"


@pytest.mark.parametrize("path", documents(), ids=lambda p: p.stem)
def test_a_document_has_enough_body_to_be_worth_indexing(path: Path) -> None:
    _, body = split_frontmatter(path.read_text(encoding="utf-8"))
    assert len(body.strip()) >= MIN_BODY_CHARACTERS, f"{path} is too short to index"


# --- no number can arrive by retrieval ------------------------------------


@pytest.mark.parametrize("path", documents(), ids=lambda p: p.stem)
def test_no_document_states_a_computed_number(path: Path) -> None:
    # The load-bearing test in this file. Every number the agent shows must be
    # traceable to a solver run that was independently verified; a figure sitting
    # in the corpus is retrievable and indistinguishable from a computed one.
    _, body = split_frontmatter(path.read_text(encoding="utf-8"))
    found = PRECISE_DECIMAL.findall(body)
    assert not found, (
        f"{path} contains what looks like a computed value: {found}. "
        "Numbers belong to the solver, not the corpus -- see data/README.md."
    )


# --- filenames --------------------------------------------------------------


@pytest.mark.parametrize("path", documents(), ids=lambda p: p.stem)
def test_a_filename_is_lowercase_and_hyphenated(path: Path) -> None:
    # The filename is shown as the source of a retrieved passage, so it is read
    # by users and not only by the loader.
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*\.md", path.name), path.name
