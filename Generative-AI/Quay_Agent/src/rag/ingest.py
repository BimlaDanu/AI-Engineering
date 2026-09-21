"""Building the retrieval index -- offline, deterministic, and re-runnable.

Run by ``make ingest``, never at question time. The corpus in ``data/`` is read
here and nowhere else: once the Chroma collection exists, the application answers
from the index, so a document edited after an ingest is invisible until the next
one. That is a property worth knowing rather than discovering.

**The corpus is shelved.** ``data/corpus/`` holds one directory per
:class:`Shelf` -- the physics of the chain, its uses in quantum computing, and what
it is applied to outside physics -- and every chunk carries the shelf it came from.
That turns "which body of knowledge answers this?" into a decision the router makes
and the interface shows, instead of an accident of which passage embedded closest. A
directory nobody declared is refused rather than indexed.

Three further decisions shape this module.

**Chunk on structure first, size second.** A Markdown heading is an author's own
statement about where one idea ends, and it is a better boundary than any
character count. Each section becomes a chunk; the character splitter only runs
on sections long enough to need it, as a safety net. The heading survives into
the chunk's metadata, which is what lets a retrieved passage be cited as
*document, section* instead of as an anonymous slice of text.

**Chunk identity is derived, not generated.** Every chunk gets a deterministic
id, so a second ingest *replaces* each chunk rather than appending a copy of it.
Without that, running ``make ingest`` twice doubles the collection and the
retriever starts returning the same passage two or three times -- crowding out
the passages that would have filled those slots. Duplicates in a vector store
degrade recall silently, which is the worst way for anything to degrade.

**The index is reconciled, not merely written to.** Ids that the corpus no longer
produces are deleted: shorten a document and its old tail would otherwise stay
retrievable forever, and a deleted document would keep answering questions. An
append-only index slowly stops matching its own source.

Embeddings are the one part that needs the network. They are injected -- the
whole chunking pipeline above is pure, and the tests exercise it with no
credential and no model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.logging_setup import get_logger
from src.settings import DEFAULT_CORPUS_PATH, Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from langchain_chroma import Chroma
    from langchain_core.embeddings import Embeddings

LOG = get_logger("rag.ingest")


@dataclass(frozen=True, slots=True)
class Shelf:
    """One knowledge base: a directory of notes with a subject of its own.

    Several shelves rather than one folder of everything, because *which* body of
    knowledge answers a question is a decision worth making and worth showing.
    The physics of the chain, the uses of the chain in quantum computing, and what
    the model is applied to outside physics are different literatures with
    different citation styles and different failure modes -- a hardware claim, a
    closed-form result and an industry pilot are not the same kind of statement,
    and a reader is owed the difference.

    The declaration is here and the directories follow it, never the other way
    round: an undeclared directory of notes is refused by :func:`load_corpus`
    rather than indexed with whatever conventions it happens to carry.

    Attributes:
        name: Slug, the directory name under ``data/corpus/`` and the value
            stored on every chunk.
        title: Human-readable name, shown in the interface.
        covers: One sentence on what belongs here. Read by
            :mod:`src.agent.router` when it asks a model which shelf to search,
            and by :mod:`src.agent.followups`, so the agent cannot suggest or
            describe a shelf that does not exist.
    """

    name: str
    title: str
    covers: str


SHELVES: tuple[Shelf, ...] = (
    Shelf(
        name="physics-notes",
        title="Physics of the chain",
        covers=(
            "the model itself: its exact solution, the free-fermion mapping, the critical "
            "point at h = J, critical exponents, and the practice and limits of exact "
            "diagonalisation"
        ),
    ),
    Shelf(
        name="quantum-computing",
        title="Quantum computing uses",
        covers=(
            "what the chain is used for as a toy model of quantum computing: the "
            "variational quantum eigensolver, QAOA and state preparation, quantum "
            "annealing and the minimum gap, Trotterised digital simulation, the hardware "
            "platforms that realise it, and how a solvable model verifies a device"
        ),
    ),
    Shelf(
        name="method-comparison",
        title="Choosing between the methods",
        covers=(
            "how the near-term methods compare and when to pick which: the variational "
            "quantum eigensolver against QAOA against variational imaginary time against "
            "quantum annealing, what each one is actually solving for, how a classical "
            "objective becomes a cost and a mixer, how a non-unitary imaginary-time step "
            "is approximated by a circuit, how deep a circuit should be before noise wins, "
            "and which real machines can run each method today"
        ),
    ),
    Shelf(
        name="applications",
        title="Applications, business and R&D",
        covers=(
            "what the model is used for outside physics: business and operational problems "
            "written as Ising cost functions, portfolio optimisation and finance, routing, "
            "scheduling and other NP-hard problems, the commercial Ising machines sold to "
            "solve them, machine learning both applied to the model and built out of it, "
            "which quantum speedups are actually proven, and how industrial R&D uses an "
            "exactly solvable instance"
        ),
    ),
)
"""The knowledge bases, in the order a reader should meet them.

Registration is an explicit tuple for the same reason the method registry is: a
directory nobody listed does not silently become part of what the agent knows.
"""

PROVENANCE_KEY = "provenance"
"""Frontmatter key recording how a note reached the library."""

CURATED = "curated"
"""A note written by hand and committed alongside the code.

Supplied when the frontmatter key is absent, so "no key" and "written by hand"
are one fact rather than two states every reader has to handle.
"""

PROMOTED = "promoted"
"""A note fetched from an external index, screened, and written to disk.

Carried on every chunk so that an answer can say which of its sources somebody
wrote and which arrived from a search nobody reviewed. The two are not equally
trustworthy and a citation that hides the difference is a citation that overstates
its own weight.
"""

PROVENANCES = (CURATED, PROMOTED)
"""Every value :data:`PROVENANCE_KEY` may take."""


def shelf_names() -> tuple[str, ...]:
    """Return every declared shelf name, in registration order.

    Returns:
        The slugs. This is the vocabulary the router is allowed to choose from
        and the only set of values the ``shelf`` metadata field ever holds.

    Examples:
        >>> shelf_names()
        ('physics-notes', 'quantum-computing', 'method-comparison', 'applications')
    """
    return tuple(shelf.name for shelf in SHELVES)


def get_shelf(name: str) -> Shelf | None:
    """Look up one shelf by name.

    Args:
        name: A shelf slug.

    Returns:
        The shelf, or ``None`` if nothing is registered under that name.
        ``None`` rather than an exception because the callers are a router
        validating a model's answer and an interface labelling a stored chunk,
        and neither should fail over an unknown string it did not choose.
    """
    for shelf in SHELVES:
        if shelf.name == name:
            return shelf
    return None


REQUIRED_KEYS = ("title", "source", "topics")
"""Frontmatter every document must carry.

Enforced here rather than only in a test: a document missing one of these is
refused at ingest, so a bad note cannot reach the index even on a machine that
never runs the suite. The ingest test covers the refusal.
"""

HEADINGS = (("#", "h1"), ("##", "h2"), ("###", "h3"))
"""Heading levels treated as chunk boundaries.

Three is enough for this corpus and deeper nesting than that is prose the author
did not intend to be read in isolation.
"""

CHUNK_CHARACTERS = 1200
"""Longest chunk the character splitter will leave alone.

Chosen against the corpus rather than as a round number: the note sections here
run 300-900 characters, so this bound is a backstop that almost never fires.
When it does fire, the section was long enough that a single embedding would
have averaged two ideas into one vector that matches neither.
"""

CHUNK_OVERLAP = 150
"""Characters repeated across a split boundary.

Only relevant when the backstop above fires. Overlap exists so a sentence cut in
half is still wholly present in one of the two pieces; a claim split across
chunks embeds as two half-claims.
"""

SEPARATORS = ("\n\n", "\n", ". ", " ", "")
"""Split points for the character splitter, strongest first.

Paragraph, then line, then sentence. The empty string is the last resort that
guarantees termination on text with no whitespace at all -- a long equation, in
this corpus.
"""

TOPIC_SEPARATOR = ","
"""How the topic list is flattened into chunk metadata.

Chroma metadata values must be scalars: a list raises. Topics are therefore
stored as ``"a,b,c"`` and read back with :func:`topics_of`, which is the only
place that encoding is known.
"""

PRECISE_DECIMAL = re.compile(
    r"(?:energy|\bE_?0\b|E\s*/\s*[NL]\b|eigenvalue)[^.\n]{0,80}?-?\d\.\d{3,}"
    r"|-?\d\.\d{3,}[^.\n]{0,80}?(?:per (?:site|spin|bond)|ground[- ]state energy)",
    re.IGNORECASE,
)
r"""The shape of a *retrievable ground-state energy*, which must never arrive by
retrieval.

The risk is specific and worth stating precisely: a figure sitting in the corpus is
retrievable, and once retrieved it is indistinguishable from one the solver produced
and cross-checked. The whole project rests on being able to tell those apart, so a
note that says "the ground-state energy is -1.2732 per site" is a note that can put
an uncomputed number into an answer.

**This was ``\d\.\d{3,}`` -- any decimal with three places -- and that made it
useless.** It fired on 90 of the corpus's 127 documents, because a corpus of arXiv
abstracts about quantum computing is *made* of numbers: gate fidelities of 99.9%,
error rates, qubit counts, exponents. Ninety-odd identical warnings scrolling past a
successful ``make ingest`` is not a control, it is weather -- and this repository has
a rule about it: a checker that cries wolf gets ignored on the run where it is
right.

Narrowed to a decimal within eighty characters of an energy word, in either order.
Against the current corpus that flags **nothing**, which is the correct answer and
was invisible underneath the false ones.

The gate is in the ingest test: the pattern is the only control on a
note that states an energy, so it needs a test rather than a warning nobody reads.
"""


class CorpusError(RuntimeError):
    """A document cannot be indexed, or there is nothing to index.

    Raised rather than logged: an ingest that skips a malformed document leaves
    an index that looks complete and answers with a gap in it.
    """


class EmbeddingUnreachableError(RuntimeError):
    """The embedding endpoint did not answer, so nothing could be indexed.

    Separate from :class:`CorpusError` because the two need opposite responses.
    A malformed document is a defect in this repository and the traceback is the
    useful part of it. An endpoint that times out is a fact about the network at
    that moment; the traceback is nine frames of somebody else's HTTP stack and
    says nothing a reader can act on. :func:`main` prints this one as a single
    line and exits non-zero.
    """


@dataclass(frozen=True, slots=True)
class Note:
    """One parsed corpus document, frontmatter separated from prose.

    Attributes:
        path: Where it was read from, kept for provenance and for reconciling
            the index against the corpus.
        title: Human-readable title, shown when a passage from it is cited.
        source: The citation -- author, journal, year -- precise enough to check
            the claim against the original.
        arxiv: arXiv id or DOI, or ``None`` for pre-arXiv work.
        topics: Declared topics, used to bias retrieval towards the right
            document.
        body: Everything after the frontmatter fence.
        provenance: :data:`CURATED` or :data:`PROMOTED`. Defaulted rather than
            required, because the hand-written notes predate the distinction and
            adding a key to every one of them would be a migration that changes
            no meaning.
    """

    path: Path
    title: str
    source: str
    arxiv: str | None
    topics: tuple[str, ...]
    body: str
    provenance: str = CURATED

    @property
    def slug(self) -> str:
        """Filename without its extension, used as the id prefix for its chunks."""
        return self.path.stem

    @property
    def shelf(self) -> str:
        """Which knowledge base this note belongs to.

        Returns:
            The name of the directory holding the note, which is a declared
            :class:`Shelf` name for anything :func:`load_corpus` produced.
            Derived rather than stored, so a note cannot claim a shelf it is not
            filed under -- the path is the single source of that fact.
        """
        return self.path.parent.name


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one ingest did, in enough detail to explain a surprising index.

    Attributes:
        notes: Documents read.
        chunks: Chunks written -- added or replaced in place.
        removed: Ids deleted because the corpus no longer produces them.
        suspect: Documents containing something shaped like a computed number.
        skipped: Ids dropped for carrying no prose -- headings and nothing else.
            Reported rather than silently discarded: a corpus producing fewer chunks
            than it has sections is a fact about the corpus, and the alternative --
            indexing them and rejecting them on every search -- is what this count
            exists to record the end of.
    """

    notes: int
    chunks: int
    removed: tuple[str, ...]
    suspect: tuple[str, ...]
    skipped: tuple[str, ...] = ()

    def summary(self) -> str:
        """One line for the log and the console.

        Returns:
            A summary naming every count, including the zeroes -- ``removed=0``
            is information, and omitting it makes a stale index look like a
            reconciled one.
        """
        return (
            f"{self.notes} notes, {self.chunks} chunks written, "
            f"{len(self.skipped)} headings-only chunks skipped, "
            f"{len(self.removed)} stale chunks removed, {len(self.suspect)} suspect documents"
        )


class Index(Protocol):
    """The slice of a vector store this module uses.

    A protocol rather than :class:`~langchain_chroma.Chroma` for one reason:
    every test here drives a plain in-memory double instead of a real store, so
    chunking, id derivation and reconciliation are tested without an embedding
    model, a credential or a temporary directory. The real Chroma satisfies this
    structurally.

    Arguments beyond the documents themselves are keyword-only and there is no
    ``**kwargs``, which makes the match strict in the useful direction: an
    implementation may accept more than this, but it cannot accept less.
    """

    def add_documents(self, documents: list[Document], *, ids: list[str]) -> list[str]:
        """Insert or replace documents by id."""
        ...

    def get(self, *, include: list[str] | None = None) -> dict[str, Any]:
        """Read stored ids and metadata."""
        ...

    def delete(self, *, ids: list[str] | None = None) -> None:
        """Remove documents by id."""
        ...


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    r"""Separate YAML frontmatter from the body of a note.

    Hand-rolled rather than using ``yaml``: PyYAML reaches this project only as a
    transitive dependency of ``chromadb``, and importing it directly would mean
    relying on a package :file:`pyproject.toml` never declares. This frontmatter
    is flat ``key: value`` lines, which is a dozen lines of parsing.

    Args:
        text: The full document.

    Returns:
        A ``(fields, body)`` pair. The fields are empty when the document does
        not open with a ``---`` fence, which :func:`parse_note` treats as an
        error rather than as a document with no metadata.

    Examples:
        >>> split_frontmatter("---\ntitle: A\n---\nBody.")
        ({'title': 'A'}, '\nBody.')
        >>> split_frontmatter("No fence here.")
        ({}, 'No fence here.')
    """
    if not text.startswith("---\n"):
        return {}, text
    _, _, rest = text.partition("---\n")
    block, fence, body = rest.partition("\n---")
    if not fence:
        return {}, text
    fields: dict[str, str] = {}
    for line in block.splitlines():
        key, colon, value = line.partition(":")
        if colon and not key.startswith((" ", "#", "-")):
            fields[key.strip()] = value.strip()
    return fields, body


def _topics(raw: str) -> tuple[str, ...]:
    """Read a frontmatter topic list into slugs.

    Args:
        raw: The raw value, written as ``[a, b, c]`` in the corpus.

    Returns:
        Deduplicated slugs in the order written. Quotes and brackets are
        stripped; anything that is not alphanumeric becomes a hyphen, so
        ``Free Fermions`` and ``free-fermions`` are one topic rather than two.
    """
    parts = (part.strip().strip("\"'") for part in raw.strip().strip("[]").split(","))
    slugs = (re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-") for part in parts)
    return tuple(dict.fromkeys(slug for slug in slugs if slug))


def topics_of(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Read the topics back out of chunk metadata.

    The inverse of the flattening described at :data:`TOPIC_SEPARATOR`. Kept
    beside it so the encoding is defined in one place; the retriever calls this
    rather than splitting the string itself.

    Args:
        metadata: A chunk's metadata as returned by the store.

    Returns:
        The topics, or an empty tuple if the chunk carries none.

    Examples:
        >>> topics_of({"topics": "exact-solution,free-fermions"})
        ('exact-solution', 'free-fermions')
        >>> topics_of({})
        ()
    """
    raw = metadata.get("topics")
    if not isinstance(raw, str) or not raw:
        return ()
    return tuple(topic for topic in raw.split(TOPIC_SEPARATOR) if topic)


def shelf_of(metadata: dict[str, Any]) -> str:
    """Read the shelf back out of chunk metadata.

    Args:
        metadata: A chunk's metadata as returned by the store.

    Returns:
        The shelf name, or ``""`` for a chunk written before shelves existed.
        Empty rather than a default shelf name: an index built by an older
        ingest should read as "unknown", not as a confident claim about which
        knowledge base a passage came from.

    Examples:
        >>> shelf_of({"shelf": "quantum-computing"})
        'quantum-computing'
        >>> shelf_of({})
        ''
    """
    raw = metadata.get("shelf")
    return raw if isinstance(raw, str) else ""


def parse_note(path: Path, text: str) -> Note:
    """Parse one document, refusing anything that cannot be cited.

    Args:
        path: Where the text came from, used in the error message and kept on
            the note.
        text: The document.

    Returns:
        The parsed note.

    Raises:
        CorpusError: If the frontmatter fence, a required key or the topic list
            is missing. Indexing an uncitable passage is worse than failing:
            the answer that quotes it cannot say where it came from.
    """
    fields, body = split_frontmatter(text)
    if not fields:
        raise CorpusError(
            f"{path} has no frontmatter fence; every note opens with a `---` block "
            f"carrying {', '.join(REQUIRED_KEYS)}"
        )
    missing = [key for key in REQUIRED_KEYS if not fields.get(key)]
    if missing:
        raise CorpusError(f"{path} is missing frontmatter {', '.join(missing)}")
    topics = _topics(fields["topics"])
    if not topics:
        raise CorpusError(f"{path} lists no topics")
    arxiv = fields.get("arxiv", "").strip().strip("\"'")
    declared = fields.get(PROVENANCE_KEY, "").strip().strip("\"'").lower()
    return Note(
        path=path,
        title=fields["title"].strip().strip("\"'"),
        source=fields["source"].strip().strip("\"'"),
        # "null" is how the corpus writes "pre-arXiv", and it arrives here as
        # the four-character string because nothing parsed the YAML.
        arxiv=None if arxiv in {"", "null", "none", "~"} else arxiv,
        topics=topics,
        body=body,
        provenance=declared if declared in PROVENANCES else CURATED,
    )


def shelf_directories(base: Path) -> tuple[Path, ...]:
    """Resolve a corpus root to the shelf directories to read.

    Args:
        base: The corpus root, or one shelf directory directly.

    Returns:
        The directories to read, in registration order. A root pointing straight
        at a shelf yields that shelf alone, which is how a fixture holding a
        single directory of notes is loaded.

    Raises:
        CorpusError: If no declared shelf exists under ``base``, or if some other
            subdirectory there holds Markdown. The second case is the one worth
            failing on: a directory of notes nobody declared is a body of
            knowledge the agent cannot cite, cannot filter by and will never
            search, and discovering that from an answer is far worse than
            discovering it here.
    """
    if base.name in shelf_names():
        return (base,)
    if not base.is_dir():
        raise CorpusError(f"corpus directory {base} does not exist")
    declared = tuple(base / name for name in shelf_names() if (base / name).is_dir())
    undeclared = sorted(
        child.name
        for child in base.iterdir()
        if child.is_dir() and child.name not in shelf_names() and any(child.glob("*.md"))
    )
    if undeclared:
        raise CorpusError(
            f"{base} holds undeclared note directories {undeclared}; add a Shelf for each "
            f"in src/rag/ingest.py or move the notes into one of {list(shelf_names())}"
        )
    if not declared:
        raise CorpusError(f"no shelf directory under {base}; expected one of {list(shelf_names())}")
    return declared


def _configured_corpus_path(settings: Settings | None) -> str:
    """Where the notes live, without requiring a credential to find out.

    Args:
        settings: Configuration, when the caller has it.

    Returns:
        The configured ``corpus_path``, or :data:`~src.settings.DEFAULT_CORPUS_PATH`
        when there is no configuration to read at all.

    Reading committed Markdown is not a privileged operation, and the default is the
    value the setting would have held anyway -- so a checkout with no credential
    still lists its knowledge base, which is what the README promises and what the
    Knowledge page had stopped doing. It is logged, because "the corpus is empty" and
    "the configuration could not be read" are different facts about a deployment.
    """
    if settings is not None:
        return settings.corpus_path
    try:
        return get_settings().corpus_path
    except Exception as error:
        LOG.info(
            "corpus_path_defaulted",
            extra={"error_type": type(error).__name__, "detail": DEFAULT_CORPUS_PATH},
        )
        return DEFAULT_CORPUS_PATH


def load_corpus(
    root: Path | str | None = None,
    settings: Settings | None = None,
) -> tuple[Note, ...]:
    """Read every note on every shelf.

    Args:
        root: Corpus directory. Defaults to the configured ``corpus_path``.
        settings: Configuration to read when ``root`` is not given. Settings are
            only consulted in that case, so reading a directory that was named
            explicitly needs no credential.

    Returns:
        The notes: shelves in registration order, and paths sorted within each,
        so that two ingests of the same corpus write the same ids in the same
        order.

    Raises:
        CorpusError: If the directory is missing or holds no Markdown, if two
            notes share a filename, or if any document fails :func:`parse_note`.
            A missing corpus and an empty one are reported separately: they are
            different mistakes.
    """
    if root is None:
        root = _configured_corpus_path(settings)
    base = Path(root)
    if not base.is_dir():
        raise CorpusError(f"corpus directory {base} does not exist")
    directories = shelf_directories(base)
    paths = [path for directory in directories for path in sorted(directory.glob("*.md"))]
    if not paths:
        raise CorpusError(f"no Markdown documents under {base}")
    notes = tuple(parse_note(path, path.read_text(encoding="utf-8")) for path in paths)
    refuse_duplicate_slugs(notes)
    return notes


def refuse_duplicate_slugs(notes: Sequence[Note]) -> None:
    """Fail if two notes would write over each other's chunks.

    Chunk ids are derived from the filename, so two notes sharing one -- on
    different shelves, or in different roots -- would each overwrite half of the
    other's chunks. Silently, with no error anywhere, and the damage shows up as
    a passage that answers the wrong question.

    Args:
        notes: The notes about to be chunked.

    Raises:
        CorpusError: If any filename appears twice, naming both paths.
    """
    seen: dict[str, Path] = {}
    for note in notes:
        clash = seen.get(note.slug)
        if clash is not None:
            raise CorpusError(f"{note.path} and {clash} share the filename {note.slug!r}")
        seen[note.slug] = note.path


def load_promoted(
    root: Path | str | None = None,
    settings: Settings | None = None,
) -> tuple[Note, ...]:
    """Read the notes that were fetched from an external source, if there are any.

    Tolerant where :func:`load_corpus` is strict, and deliberately so. The
    hand-written corpus is committed and a malformed note in it is somebody's
    mistake, worth stopping for. This directory is generated: a note in it can be
    malformed because an external index returned something unexpected, and that
    must not stop the whole library from being indexed. A bad promoted note is
    dropped with a warning; a bad curated note still fails the ingest.

    Args:
        root: Directory to read. Defaults to the configured ``promoted_path``.
        settings: Configuration to read when ``root`` is not given.

    Returns:
        The promoted notes, or nothing at all -- which is the ordinary state
        before anything has been fetched, and also the state when promotion is
        switched off.
    """
    if root is None:
        configured = settings.promoted_path if settings is not None else _configured_promoted_path()
        if not configured:
            return ()
        root = configured
    base = Path(root)
    if not base.is_dir() or not any(base.glob("*/*.md")):
        return ()
    try:
        return load_corpus(base, settings)
    except CorpusError as error:
        LOG.warning(
            "promoted_notes_unreadable",
            extra={"path": base.as_posix(), "detail": str(error)},
        )
        return ()


def _configured_promoted_path() -> str | None:
    """Where promoted notes live, without requiring a credential to find out.

    Returns:
        The configured path, or ``None`` when configuration cannot be read at
        all. ``None`` rather than a default here: promoted notes are optional,
        and a checkout that cannot read its configuration should index the
        committed corpus rather than guess about a generated directory.
    """
    try:
        return get_settings().promoted_path
    except Exception as error:
        LOG.info("promoted_path_unreadable", extra={"error_type": type(error).__name__})
        return None


def load_library(
    settings: Settings | None = None,
    *,
    root: Path | str | None = None,
    promoted: Path | str | None = None,
) -> tuple[Note, ...]:
    """Read everything the index should hold: the corpus, then what was promoted.

    Curated notes come first and win every collision. A fetch can legitimately
    turn up a work somebody has already written a note about, and when that
    happens the hand-written note is the better one -- it was read by a person
    and the other was not. Dropping the promoted copy is also what keeps a
    generated file from being able to break an ingest.

    Args:
        settings: Configuration to read.
        root: Corpus directory, overriding configuration.
        promoted: Promoted-notes directory, overriding configuration.

    Returns:
        Every note to index, curated before promoted.

    Raises:
        CorpusError: If the curated corpus is missing, empty or uncitable. A
            problem in the promoted directory is logged instead.
    """
    curated = load_corpus(root, settings)
    if promoted is None and root is not None and settings is None:
        # The caller named a corpus directly and gave no configuration, so there
        # is nothing to consult about a second one. Reading the process settings
        # here would make an explicitly scoped load depend on the environment.
        return curated
    extra = load_promoted(promoted, settings)
    known = {note.slug for note in curated}
    fresh = []
    for note in extra:
        if note.slug in known:
            LOG.info(
                "promoted_note_superseded",
                extra={"document": note.slug, "detail": "a curated note already covers it"},
            )
            continue
        known.add(note.slug)
        fresh.append(note)
    if fresh:
        LOG.info("promoted_notes_loaded", extra={"notes": len(fresh)})
    return (*curated, *fresh)


def _heading_path(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Collect the heading trail the Markdown splitter attached to a section.

    Args:
        metadata: Metadata from :class:`MarkdownHeaderTextSplitter`.

    Returns:
        The headings from outermost to innermost, skipping levels the section
        sits under but does not declare.
    """
    return tuple(str(metadata[key]) for _, key in HEADINGS if metadata.get(key))


def _sections(body: str) -> list[Document]:
    """Split a body on headings, then on size only where necessary.

    Args:
        body: The note's prose.

    Returns:
        Sections as documents carrying their heading metadata. Headings are kept
        in the text rather than stripped: the heading is often the most
        retrievable sentence in a section, and an embedding of the body alone
        loses it.
    """
    by_heading = MarkdownHeaderTextSplitter(
        headers_to_split_on=list(HEADINGS),
        strip_headers=False,
    ).split_text(body)
    by_size = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARACTERS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=list(SEPARATORS),
    )
    return by_size.split_documents(by_heading)


def chunk_id(note: Note, position: int) -> str:
    """Derive a chunk's stable identity.

    Args:
        note: The document the chunk came from.
        position: The chunk's index within that document.

    Returns:
        An id of the form ``slug#003``. Derived from the document and the
        position rather than from the content, so editing a paragraph updates
        that chunk in place instead of orphaning it and inserting a near
        duplicate beside it. Zero-padded so a listing sorts the way a reader
        expects.

    Examples:
        >>> note = Note(Path("pfeuty-exact-solution.md"), "T", "S", None, ("a",), "")
        >>> chunk_id(note, 3)
        'pfeuty-exact-solution#003'
    """
    return f"{note.slug}#{position:03d}"


def chunk(note: Note) -> tuple[Document, ...]:
    """Split one note into embeddable chunks with citable metadata.

    Args:
        note: The parsed document.

    Returns:
        The chunks, each carrying its id and everything a citation needs. Every
        metadata value is a scalar, because Chroma rejects anything else --
        see :data:`TOPIC_SEPARATOR`.
    """
    chunks: list[Document] = []
    for position, section in enumerate(_sections(note.body)):
        trail = _heading_path(section.metadata)
        chunks.append(
            Document(
                id=chunk_id(note, position),
                page_content=section.page_content.strip(),
                metadata={
                    "path": note.path.as_posix(),
                    "document": note.slug,
                    "shelf": note.shelf,
                    "title": note.title,
                    "source": note.source,
                    # Empty string rather than None: a null metadata value is
                    # not portable across store backends, and "" reads as
                    # "pre-arXiv" the same way the frontmatter does.
                    "arxiv": note.arxiv or "",
                    "topics": TOPIC_SEPARATOR.join(note.topics),
                    "section": trail[-1] if trail else note.title,
                    "position": position,
                    PROVENANCE_KEY: note.provenance,
                },
            )
        )
    return tuple(chunks)


def carries_prose(text: str) -> bool:
    r"""Whether a chunk contains at least one line that is not a heading.

    The same rule :func:`src.rag.retrieve.has_prose` applies at query time, stated
    here so it can be applied at *ingest* time as well. Deliberately duplicated
    rather than imported: :mod:`src.rag.ingest` must not depend on the retrieval
    layer, since the whole point of ingest is that it runs before anything searches.
    The ingest test holds the two definitions together so they cannot
    drift.

    Args:
        text: The chunk's content.

    Returns:
        True when some line has content and does not start with ``#``.

    Examples:
        >>> carries_prose("# A title\n## Abstract")
        False
        >>> carries_prose("## The gap\nIt closes linearly.")
        True
    """
    return any(line.strip() and not line.lstrip().startswith("#") for line in text.splitlines())


def chunk_corpus(notes: Iterable[Note]) -> tuple[Document, ...]:
    """Chunk every note, dropping the chunks retrieval would always refuse.

    **A chunk with no prose in it was indexed, retrieved and rejected on every
    search.** :func:`src.rag.retrieve.has_prose` refuses a headings-only chunk at
    query time -- correctly, since a list of headings can support no claim -- but the
    same rule was never applied when the index was built. So the store carried 32 of
    them out of 463, and each one was found again by every query, filled a slot in the
    candidate pool that a real passage could have used, was thrown away, and logged.
    With query expansion running four phrasings per question, that produced about
    thirty rejection lines for a single answer -- most of them the same few chunks
    over and over, which is how a log stops being read.

    Skipping them here fixes all three symptoms at once: a smaller index, a candidate
    pool made only of passages that can actually be cited, and a quiet log. The
    dropped count is returned to the caller rather than hidden, because "the corpus
    produced fewer chunks than it has sections" is a fact about the corpus that
    somebody should see once.

    Args:
        notes: The parsed corpus.

    Returns:
        Every chunk that carries prose, in corpus order.
    """
    return tuple(
        piece for note in notes for piece in chunk(note) if carries_prose(piece.page_content)
    )


def prose_less_chunks(notes: Iterable[Note]) -> tuple[str, ...]:
    """Identify the chunks :func:`chunk_corpus` drops.

    Args:
        notes: The parsed corpus.

    Returns:
        The ids of the headings-only chunks, so the ingest report can name how many
        were skipped and a test can assert the count.
    """
    return tuple(
        str(piece.id)
        for note in notes
        for piece in chunk(note)
        if not carries_prose(piece.page_content)
    )


def _stored(index: Index) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield every id currently in the index with its metadata.

    Args:
        index: The store to read.

    Yields:
        ``(id, metadata)`` pairs. Metadata is requested explicitly because the
        chunk text is not needed here and is by far the larger half of the read.
    """
    stored = index.get(include=["metadatas"])
    ids = stored.get("ids") or []
    metadatas = stored.get("metadatas") or []
    for position, identifier in enumerate(ids):
        metadata = metadatas[position] if position < len(metadatas) else None
        yield str(identifier), metadata if isinstance(metadata, dict) else {}


def stale_ids(index: Index, keeping: Sequence[str]) -> tuple[str, ...]:
    """Find chunks the corpus no longer produces.

    Args:
        index: The store to reconcile.
        keeping: Every id this ingest wrote.

    Returns:
        Ids to delete, sorted. Two cases, and both are silent corruption if
        skipped: a document that lost a section leaves its old tail retrievable,
        and a document deleted from the corpus keeps answering questions from
        beyond the grave.
    """
    keep = set(keeping)
    return tuple(sorted(identifier for identifier, _ in _stored(index) if identifier not in keep))


def _suspect(notes: Iterable[Note]) -> tuple[str, ...]:
    """Name documents stating an energy that could be mistaken for a computed one.

    Args:
        notes: The parsed corpus.

    Returns:
        The paths of the offenders, one warning each. Per-document rather than a
        count, because the point of the warning is to be actionable -- but see
        :data:`PRECISE_DECIMAL` for what happened when the pattern was broad enough
        to make that a flood.
    """
    return tuple(note.path.as_posix() for note in notes if PRECISE_DECIMAL.search(note.body))


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Construct the embedding client.

    Imported lazily so that :mod:`src.rag.ingest` can be imported -- and its
    chunking tested -- without ``langchain_openai`` being touched.

    A note on the endpoint. The chat model and the embedding model need not live
    behind the same gateway: OpenRouter's strength is chat completions, and its
    coverage of the ``/embeddings`` route is not something to assume. So
    ``embedding_base_url`` and ``embedding_api_key`` default to the OpenRouter
    values and can be pointed at any OpenAI-protocol endpoint that does serve
    embeddings, without the rest of the project noticing.

    Args:
        settings: Configuration to read. Defaults to the process settings.

    Returns:
        An embedding client for :attr:`Settings.embedding_model`.
    """
    from langchain_openai import OpenAIEmbeddings

    resolved = get_settings() if settings is None else settings
    return OpenAIEmbeddings(
        model=resolved.embedding_model,
        base_url=resolved.embedding_base_url or resolved.openrouter_base_url,
        api_key=resolved.embedding_api_key or resolved.openrouter_api_key,
        timeout=resolved.request_timeout_s,
        # Retries stay on for embeddings, unlike the chat model: there is no
        # middleware around an ingest to own them, and re-running the whole
        # ingest to recover from one dropped connection re-pays for every
        # embedding in the corpus.
        max_retries=resolved.max_retries,
        # The tokeniser knows OpenAI models; a slug from another provider makes
        # it guess. Counting tokens is only used to pre-split oversized inputs,
        # and this corpus is chunked well under any provider's limit.
        check_embedding_ctx_length=False,
    )


def open_index(
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> Chroma:
    """Open the persistent Chroma collection, creating it if it does not exist.

    The concrete store is returned rather than the :class:`Index` protocol, which
    covers writing only: retrieval needs the search half, and one function that
    opens the collection is better than two that could disagree about the
    distance metric.

    Args:
        embeddings: Embedding client. Built from settings when omitted, which is
            the point at which a credential becomes necessary.
        settings: Configuration to read.

    Returns:
        The collection named by :attr:`Settings.collection_name` -- which
        includes the embedding model, so switching models addresses a different,
        empty collection instead of comparing coordinates from two unrelated
        spaces.
    """
    from langchain_chroma import Chroma

    resolved = get_settings() if settings is None else settings
    return Chroma(
        collection_name=resolved.collection_name,
        embedding_function=build_embeddings(resolved) if embeddings is None else embeddings,
        persist_directory=resolved.vector_store_path,
        # Cosine, not the L2 default. Embedding models are trained so that
        # direction carries the meaning and magnitude carries little; with L2 a
        # long passage is penalised for being long.
        collection_metadata={"hnsw:space": "cosine"},
    )


PROBE_TIMEOUT_S = 10.0
"""Seconds the preflight probe waits before calling the endpoint unreachable.

Long enough that a slow-but-working gateway is not slandered, short enough that
the answer arrives while somebody is still watching. A probe that inherited the
configured sixty-second timeout and its three retries would take four minutes to
report a host that is simply not there.
"""

WRITE_BATCH = 64
"""Chunks per embedding request.

The whole corpus in one request is one bet: it either embeds everything or, on a
dropped connection, loses everything and reports nothing until the retries are
exhausted. Batching costs a few extra round trips and buys two things worth more
than they cost -- a log line per batch, so a slow ingest can be watched rather
than guessed at, and a failure that has already banked the batches before it.

Sixty-four keeps each request well inside any provider's payload limit while
holding the number of round trips for a corpus this size in the low twenties.
"""


def _write_in_batches(store: Index, chunks: list[Document], ids: list[str]) -> None:
    """Embed and write the chunks a batch at a time, logging progress.

    Args:
        store: The index to write to.
        chunks: Every chunk the corpus produced, in order.
        ids: Their deterministic ids, positionally aligned with ``chunks``.

    Raises:
        EmbeddingUnreachableError: If a batch's embedding request fails. The message
            names how many chunks were already written, because a re-run is
            idempotent -- ids are derived, so the banked batches are simply
            overwritten with the same content rather than duplicated.
    """
    total = len(chunks)
    written = 0
    for start in range(0, total, WRITE_BATCH):
        batch = chunks[start : start + WRITE_BATCH]
        try:
            store.add_documents(batch, ids=ids[start : start + WRITE_BATCH])
        except Exception as error:
            raise EmbeddingUnreachableError(
                f"embedding failed after {written} of {total} chunks: {error}"
            ) from error
        written += len(batch)
        LOG.info("ingest_progress", extra={"written": written, "total": total})


def ingest(
    index: Index | None = None,
    *,
    root: Path | str | None = None,
    settings: Settings | None = None,
) -> IngestReport:
    """Read the corpus, write its chunks, and reconcile what was there before.

    The order matters: write first, then delete what the write did not cover. The
    other order leaves a window in which the index is empty, and on a failed
    ingest that window never closes -- an application serving a half-built index
    answers worse than one serving a stale index.

    Args:
        index: Store to write to. Opened from settings when omitted.
        root: Corpus directory. Defaults to the configured ``corpus_path``.
        settings: Configuration to read.

    Returns:
        A report of what was written and removed.

    Raises:
        CorpusError: If the corpus is missing, empty, or holds a document that
            cannot be cited.
    """
    # Resolved only if something below actually needs it. A caller that names both
    # the store and the corpus directory -- every test in the ingest suite
    # does -- needs no credential, and reading one eagerly made a keyless machine
    # fail on a function that was never going to consult the configuration.
    resolved = settings
    if resolved is None and (root is None or index is None):
        resolved = get_settings()
    notes = load_library(resolved, root=root)
    chunks = chunk_corpus(notes)
    if not chunks:
        raise CorpusError("the corpus produced no chunks; every document is empty")

    store = open_index(settings=resolved) if index is None else index
    ids = [str(piece.id) for piece in chunks]
    _write_in_batches(store, list(chunks), ids)

    removed = stale_ids(store, ids)
    if removed:
        store.delete(ids=list(removed))

    suspect = _suspect(notes)
    for path in suspect:
        LOG.warning(
            "corpus_energy_suspected",
            extra={
                "path": path,
                "detail": "looks like a computed value; notes carry prose, not results",
            },
        )

    report = IngestReport(
        notes=len(notes),
        chunks=len(chunks),
        removed=removed,
        suspect=suspect,
        skipped=prose_less_chunks(notes),
    )
    LOG.info(
        "ingest_complete",
        extra={
            "notes": report.notes,
            "chunks": report.chunks,
            "removed": len(report.removed),
            # Unknown when the caller passed its own store and its own corpus: no
            # configuration was read, so naming a collection here would be a guess.
            "collection": resolved.collection_name if resolved is not None else "caller-supplied",
        },
    )
    return report


def check_endpoint(settings: Settings | None = None) -> None:
    """Embed one short string, to find out now whether the endpoint answers.

    Called before the corpus is read. Without it, an unreachable endpoint is
    discovered after loading every note, chunking it, and opening Chroma -- and
    then only once the first full batch has spent the request timeout and its
    retries. That is minutes of silence to learn something one round trip
    settles.

    The probe is one token, so it is free in any accounting that matters, and it
    separates the two failures a reader confuses: an endpoint that refuses the
    credential answers immediately with 401, while an endpoint that is not
    reachable at all does not answer.

    Args:
        settings: Configuration to read. Defaults to the process settings.

    Raises:
        EmbeddingUnreachableError: If the probe does not come back with a vector.
    """
    resolved = get_settings() if settings is None else settings
    endpoint = resolved.embedding_base_url or resolved.openrouter_base_url
    # The probe gets its own clock, and not the configured one. `request_timeout_s`
    # defaults to 60 with three retries, which is a sensible budget for a request
    # whose answer is wanted and an absurd one for a question whose whole purpose is
    # to be answered quickly: four minutes of silence to learn that a host is
    # unreachable is the latency this function exists to remove, so inheriting those
    # numbers would reintroduce it here.
    probing = resolved.model_copy(update={"request_timeout_s": PROBE_TIMEOUT_S, "max_retries": 0})
    try:
        vector = build_embeddings(probing).embed_query("ping")
    except Exception as error:
        raise EmbeddingUnreachableError(
            f"{resolved.embedding_model} at {endpoint} did not answer: {error}"
        ) from error
    if not vector:
        raise EmbeddingUnreachableError(
            f"{resolved.embedding_model} at {endpoint} returned an empty vector"
        )


def main() -> int:
    """Entry point for ``make ingest``.

    Logs the outcome rather than printing it -- ``print`` is banned project-wide
    and stdout belongs to the application.

    Two failures are reported differently on purpose. A corpus that cannot be
    indexed is a defect here and propagates with its traceback, because the
    frames are where the malformed document is. An endpoint that will not answer
    is a fact about the network, and its traceback is nine frames of an HTTP
    client library; that one becomes a single line and a non-zero exit.

    Returns:
        ``0`` on success, ``1`` if the embedding endpoint could not be reached.
    """
    from src.logging_setup import configure_logging

    settings = get_settings()
    configure_logging(secrets=[settings.openrouter_api_key.get_secret_value()])
    LOG.info("ingest_start", extra={"collection": settings.collection_name})
    try:
        check_endpoint(settings)
        report = ingest(settings=settings)
    except EmbeddingUnreachableError as error:
        LOG.error(
            "ingest_unreachable",
            extra={
                "detail": str(error),
                "remedy": (
                    "check the network, then that OPENROUTER_API_KEY is set; point "
                    "EMBEDDING_BASE_URL and EMBEDDING_API_KEY at another "
                    "OpenAI-protocol endpoint to use a different provider"
                ),
            },
        )
        return 1
    LOG.info("ingest_summary", extra={"summary": report.summary()})
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by `make ingest`
    raise SystemExit(main())
