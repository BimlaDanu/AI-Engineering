"""BM25 keyword search over the corpus, to sit beside the vector search.

Embeddings are good at paraphrase and bad at exact tokens. "Which note cites
Pfeuty?" cannot be answered by the vector index at all, because the citation
lives in a note's frontmatter and only the body is embedded. So this module
indexes the body *and* the title, section, source and arXiv id, and scores it
with BM25 -- word overlap, weighted so that a rare word counts for more than a
common one.

Measured over the built corpus: 69 chunks, 1428 distinct tokens, and 785 of
those appear in exactly one chunk. That is what makes keyword search worth
having here. It is also why ``h`` and ``J`` are *not* the win they look like --
they appear in 24 and 19 chunks respectively, so BM25 scores them near zero,
which is correct.

No new dependency: ``rank-bm25`` is not installed and is not needed for 69
chunks. The index is rebuilt from ``data/corpus/`` rather than read back out of
Chroma, so it needs no credential and works when the vector index does not.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from src import security
from src.logging_setup import get_logger
from src.rag.ingest import chunk_corpus, load_corpus
from src.rag.retrieve import STOPWORDS
from src.settings import Settings

LOG = get_logger("rag.lexical")

K1 = 1.5
"""How fast a repeated word stops adding to the score. The usual BM25 value."""

B = 0.75
"""How much a long passage is penalised for its length. The usual BM25 value."""

TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")
"""A word or a number, keeping decimals and arXiv ids in one piece.

Digits are kept and short words are not dropped, which is the opposite of
:func:`src.rag.retrieve.content_words`. That function compares meaning, where
``h`` is noise; this one matches tokens, where an identifier is the whole query.
"""

INDEXED_FIELDS = ("title", "section", "source", "arxiv")
"""Metadata searched alongside the chunk text.

The vector index embeds the body only, so an author's name or an arXiv id is
unreachable by similarity search. Here it is just more words in the document.
"""


def tokenise(text: str) -> list[str]:
    """Split text into the tokens BM25 counts.

    Normalised through :func:`src.security.normalise` for the same reason the
    rest of retrieval is: a word disguised from the injection screen must not be
    able to reappear here.

    Args:
        text: A query, a passage, or a metadata field.

    Returns:
        Lower-cased tokens in order, with stopwords removed. Not stemmed --
        matching exactly is the point, and paraphrase is the vector half's job.

    Examples:
        >>> tokenise("Which note cites Pfeuty (1970)?")
        ['note', 'cites', 'pfeuty', '1970']
        >>> tokenise("arXiv:1802.06002")
        ['arxiv', '1802.06002']
    """
    return [word for word in TOKEN.findall(security.normalise(text)) if word not in STOPWORDS]


def indexed_text(document: Document) -> str:
    """Join a chunk's text with the metadata worth searching.

    Args:
        document: A chunk as :func:`src.rag.ingest.chunk` produced it.

    Returns:
        The body followed by the fields in :data:`INDEXED_FIELDS`.
    """
    metadata = document.metadata or {}
    extra = [str(metadata.get(field, "")) for field in INDEXED_FIELDS]
    return " ".join([document.page_content, *extra])


@dataclass(frozen=True, slots=True)
class Bm25:
    """A keyword index over the corpus chunks.

    Built once and reused. :meth:`search` deliberately mirrors the vector store's
    ``similarity_search_with_relevance_scores``, so :func:`src.rag.retrieve.search`
    can treat the two halves the same way.

    Attributes:
        documents: The chunks, in corpus order.
        counts: Token counts per chunk, aligned with ``documents``.
        frequency: How many chunks each token appears in.
        average_length: Mean chunk length in tokens.
    """

    documents: tuple[Document, ...]
    counts: tuple[Counter[str], ...]
    frequency: dict[str, int]
    average_length: float

    @classmethod
    def of(cls, documents: Sequence[Document]) -> Bm25:
        """Build an index from chunks.

        Args:
            documents: The chunks to index.

        Returns:
            The index. An empty corpus gives an index whose searches return
            nothing, rather than a division by zero.
        """
        counts = tuple(Counter(tokenise(indexed_text(piece))) for piece in documents)
        frequency: Counter[str] = Counter()
        for count in counts:
            frequency.update(count.keys())
        total = sum(sum(count.values()) for count in counts)
        return cls(
            documents=tuple(documents),
            counts=counts,
            frequency=dict(frequency),
            average_length=total / len(counts) if counts else 0.0,
        )

    def weight(self, token: str) -> float:
        """Score one token's rarity.

        Args:
            token: A query token.

        Returns:
            Its inverse document frequency. A token in every chunk is worth
            almost nothing; one in a single chunk is worth a lot.
        """
        total = len(self.documents)
        seen = self.frequency.get(token, 0)
        return math.log(1 + (total - seen + 0.5) / (seen + 0.5))

    def score(self, tokens: Sequence[str], position: int) -> float:
        """Score one chunk against a tokenised query.

        Args:
            tokens: The query tokens.
            position: Index of the chunk to score.

        Returns:
            The BM25 score. Zero when the chunk shares no token with the query.
        """
        count = self.counts[position]
        length = sum(count.values())
        total = 0.0
        for token in tokens:
            seen = count.get(token, 0)
            if not seen:
                continue
            norm = seen + K1 * (1 - B + B * length / (self.average_length or 1.0))
            total += self.weight(token) * seen * (K1 + 1) / norm
        return total

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Rank chunks against a query.

        Args:
            query: What to search for.
            k: How many to return.

        Returns:
            The best ``k`` chunks with their scores, best first. Chunks scoring
            zero are left out, so a query sharing no vocabulary with the corpus
            returns nothing at all.
        """
        tokens = tokenise(query)
        if not tokens:
            return []
        scored = [
            (self.documents[position], self.score(tokens, position))
            for position in range(len(self.documents))
        ]
        hits = [(document, value) for document, value in scored if value > 0]
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits[:k]


def build_index(
    root: Path | str | None = None,
    settings: Settings | None = None,
) -> Bm25 | None:
    """Read the corpus and index it.

    Args:
        root: Corpus directory. Defaults to the configured ``corpus_path``.
        settings: Configuration to read.

    Returns:
        The index, or ``None`` if the corpus could not be read. ``None`` rather
        than an exception because keyword search is an improvement on the vector
        search, not a precondition for it: a missing corpus should cost recall,
        not the answer.

        Every failure is absorbed, not a chosen few. Reading the corpus path means
        resolving the settings, so a run with no credential arrives here as a
        validation error rather than as a missing directory -- and a keyword index
        that cannot be built must never be the reason a question fails.
    """
    try:
        return Bm25.of(chunk_corpus(load_corpus(root, settings)))
    except Exception as error:
        LOG.warning(
            "lexical_index_unavailable",
            extra={"error_type": type(error).__name__, "detail": "keyword search is disabled"},
        )
        return None


@lru_cache(maxsize=1)
def default_index() -> Bm25 | None:
    """The process-wide index over the configured corpus.

    Cached because the corpus does not change while the application runs -- the
    same assumption the Chroma collection is built on. Tests pass their own index
    instead of calling this.

    Returns:
        The index, or ``None`` if the corpus could not be read.
    """
    return build_index()
