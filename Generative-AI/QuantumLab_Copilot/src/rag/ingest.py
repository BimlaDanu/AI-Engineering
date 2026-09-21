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
            and by :func:`src.agent.graph.capabilities`, so the agent cannot
            describe a knowledge base that does not exist.
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

NOTES_SUBDIRECTORY = SHELVES[0].name
"""The first shelf's directory name.

Kept as a name because it is the default answer to "where does a note go?" and
because a corpus path pointing straight at one shelf -- which is how several tests
read a fixture -- still has to work.
"""


def shelf_names() -> tuple[str, ...]:
    """Return every declared shelf name, in registration order.

    Returns:
        The slugs. This is the vocabulary the router is allowed to choose from
        and the only set of values the ``shelf`` metadata field ever holds.

    Examples:
        >>> shelf_names()
        ('physics-notes', 'quantum-computing', 'applications')
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
"""Frontmatter every document must carry, per ``data/README.md``.

Enforced twice on purpose. ``tests/test_corpus.py`` catches a bad document in
CI; this module refuses to index one at all. The duplication is deliberate --
the test protects the repository, this protects the index, and only one of the
two runs on the same machine at ingest time.
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

PRECISE_DECIMAL = re.compile(r"\d\.\d{3,}")
"""The shape of a computed number, which must never arrive by retrieval.

Same pattern as ``tests/test_corpus.py`` and the same reasoning: a figure sitting
in the corpus is retrievable and indistinguishable from one the solver produced
and verified. Here it is a warning rather than a refusal -- the test is the gate,
and a corpus author who has written ``0.4941`` in prose about a published result
should be told, not blocked mid-ingest.
"""


class CorpusError(RuntimeError):
    """A document cannot be indexed, or there is nothing to index.

    Raised rather than logged: an ingest that skips a malformed document leaves
    an index that looks complete and answers with a gap in it.
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
    """

    path: Path
    title: str
    source: str
    arxiv: str | None
    topics: tuple[str, ...]
    body: str

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
    """

    notes: int
    chunks: int
    removed: tuple[str, ...]
    suspect: tuple[str, ...]

    def summary(self) -> str:
        """One line for the log and the console.

        Returns:
            A summary naming every count, including the zeroes -- ``removed=0``
            is information, and omitting it makes a stale index look like a
            reconciled one.
        """
        return (
            f"{self.notes} notes, {self.chunks} chunks written, "
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
        raise CorpusError(f"{path} has no frontmatter fence; see data/README.md")
    missing = [key for key in REQUIRED_KEYS if not fields.get(key)]
    if missing:
        raise CorpusError(f"{path} is missing frontmatter {', '.join(missing)}")
    topics = _topics(fields["topics"])
    if not topics:
        raise CorpusError(f"{path} lists no topics")
    arxiv = fields.get("arxiv", "").strip().strip("\"'")
    return Note(
        path=path,
        title=fields["title"].strip().strip("\"'"),
        source=fields["source"].strip().strip("\"'"),
        # "null" is how the corpus writes "pre-arXiv", and it arrives here as
        # the four-character string because nothing parsed the YAML.
        arxiv=None if arxiv in {"", "null", "none", "~"} else arxiv,
        topics=topics,
        body=body,
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
        raise CorpusError(f"corpus directory {base} does not exist; see data/README.md")
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
        raise CorpusError(
            f"no shelf directory under {base}; expected one of {list(shelf_names())} "
            "-- see data/README.md"
        )
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
        raise CorpusError(f"corpus directory {base} does not exist; see data/README.md")
    directories = shelf_directories(base)
    paths = [path for directory in directories for path in sorted(directory.glob("*.md"))]
    if not paths:
        raise CorpusError(f"no Markdown documents under {base}")
    notes = tuple(parse_note(path, path.read_text(encoding="utf-8")) for path in paths)
    # Chunk ids are derived from the filename, so two notes with the same name on
    # different shelves would write over each other's chunks -- half of each
    # document, silently, with no error anywhere. Cheaper to refuse.
    seen: dict[str, Path] = {}
    for note in notes:
        clash = seen.get(note.slug)
        if clash is not None:
            raise CorpusError(f"{note.path} and {clash} share the filename {note.slug!r}")
        seen[note.slug] = note.path
    return notes


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
                },
            )
        )
    return tuple(chunks)


def chunk_corpus(notes: Iterable[Note]) -> tuple[Document, ...]:
    """Chunk every note.

    Args:
        notes: The parsed corpus.

    Returns:
        Every chunk, in corpus order.
    """
    return tuple(piece for note in notes for piece in chunk(note))


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
    """Name documents that look like they state a computed number.

    Args:
        notes: The parsed corpus.

    Returns:
        The paths of the offenders, for the report and a warning. See
        :data:`PRECISE_DECIMAL` for why this is a warning here and a failing
        test in CI.
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
    # the store and the corpus directory -- every test in `tests/test_rag_ingest.py`
    # does -- needs no credential, and reading one eagerly made a keyless machine
    # fail on a function that was never going to consult the configuration.
    resolved = settings
    if resolved is None and (root is None or index is None):
        resolved = get_settings()
    notes = load_corpus(root, resolved)
    chunks = chunk_corpus(notes)
    if not chunks:
        raise CorpusError("the corpus produced no chunks; every document is empty")

    store = open_index(settings=resolved) if index is None else index
    ids = [str(piece.id) for piece in chunks]
    store.add_documents(list(chunks), ids=ids)

    removed = stale_ids(store, ids)
    if removed:
        store.delete(ids=list(removed))

    suspect = _suspect(notes)
    for path in suspect:
        LOG.warning(
            "corpus_number_suspected",
            extra={"path": path, "detail": "looks like a computed value; see data/README.md"},
        )

    report = IngestReport(notes=len(notes), chunks=len(chunks), removed=removed, suspect=suspect)
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


def main() -> None:
    """Entry point for ``make ingest``.

    Logs the outcome rather than printing it -- ``print`` is banned project-wide
    and stdout belongs to the application. A failure propagates: an ingest that
    reports success while having skipped a document is the failure mode this
    whole module is arranged against.
    """
    from src.logging_setup import configure_logging

    settings = get_settings()
    configure_logging(secrets=[settings.openrouter_api_key.get_secret_value()])
    LOG.info("ingest_start", extra={"collection": settings.collection_name})
    report = ingest(settings=settings)
    LOG.info("ingest_summary", extra={"summary": report.summary()})


if __name__ == "__main__":  # pragma: no cover - exercised by `make ingest`
    main()
