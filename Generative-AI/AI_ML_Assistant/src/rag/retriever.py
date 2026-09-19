"""Tools for searching documents using vector and keyword methods.

This module handles:
- Accessing the Chroma vector store.
- Searching documents using both BM25 keyword search and vector similarity search.
- Improving user queries with LLM-based query rewriting before retrieval.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import numpy as np
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import CHROMA_DIR, COLLECTION_NAME
from src.core.schemas import QueryExpansion, Reranking
from src.rag.embeddings import make_embeddings
from src.utils import UsageMeter

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric terms for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Minimal Okapi BM25 keyword scorer — pure Python, no extra dependency."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs = [_tokenize(doc) for doc in corpus]
        self._doc_len = [len(d) for d in self._docs]
        self._avg_len = (sum(self._doc_len) / len(self._docs)) if self._docs else 0.0
        self._df: dict[str, int] = {}
        for doc in self._docs:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1

    def scores(self, query: str) -> list[float]:
        """BM25 score of the query against every document in the corpus."""
        n = len(self._docs)
        out = [0.0] * n
        if not n:
            return out
        for term in _tokenize(query):
            df = self._df.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self._docs):
                tf = doc.count(term)
                if not tf:
                    continue
                norm = 1 - self._b + self._b * self._doc_len[i] / self._avg_len
                out[i] += idf * tf * (self._k1 + 1) / (tf + self._k1 * norm)
        return out


@dataclass
class RetrievedChunk:
    """One retrieved chunk with its vector, BM25, and blended hybrid scores."""

    text: str
    metadata: dict
    vector_score: float
    bm25_score: float
    score: float


@dataclass(frozen=True)
class MetadataFilters:
    """User-chosen knowledge-base facets that narrow retrieval (topic / source / year).

    Complements the ``subjects`` and ``difficulty`` filters (driven by the subject picker and
    the learner-level toggle) with the facets a researcher most often wants to scope by. Each
    facet is a set of allowed values; an empty tuple means "don't filter on this facet", so the
    default :class:`MetadataFilters` is inactive and leaves retrieval byte-for-byte unchanged.

    ``year`` values are compared as strings because that is how the ingest stores them (no
    numeric coercion); a chunk with no ``year`` never matches an active year filter. The same
    filter drives both retrievers: :meth:`conditions` is pushed into the Chroma ``where`` clause
    for the vector side, and :meth:`passes` is applied to BM25-only candidates so the two paths
    honour identical filtering.
    """

    topics: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    years: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        """True if any facet is set (and retrieval should therefore be narrowed)."""
        return bool(self.topics or self.sources or self.years)

    def conditions(self) -> list[dict]:
        """Chroma metadata predicates (one ``$in`` per active facet) for the vector side."""
        conditions: list[dict] = []
        if self.topics:
            conditions.append({"topic": {"$in": list(self.topics)}})
        if self.sources:
            conditions.append({"source": {"$in": list(self.sources)}})
        if self.years:
            conditions.append({"year": {"$in": list(self.years)}})
        return conditions

    def passes(self, meta: dict) -> bool:
        """Apply the same facet filter to a BM25-only candidate that Chroma applies to vectors."""
        if self.topics and meta.get("topic") not in self.topics:
            return False
        if self.sources and meta.get("source") not in self.sources:
            return False
        if self.years and str(meta.get("year")) not in self.years:
            return False
        return True


def compute_facets(metas: list[dict]) -> dict[str, list[str]]:
    """Distinct topic / source / year values across the index, for populating filter UIs.

    Pure over the chunk metadata so it is testable without a live store. Topics and sources are
    sorted alphabetically; years are sorted newest-first (and coerced to strings to match how
    they are stored). Missing/empty values are ignored so a facet only offers real choices.
    """
    topics: set[str] = set()
    sources: set[str] = set()
    years: set[str] = set()
    for meta in metas:
        topic = meta.get("topic")
        source = meta.get("source")
        year = meta.get("year")
        if topic:
            topics.add(str(topic))
        if source:
            sources.add(str(source))
        if year not in (None, ""):
            years.add(str(year))
    return {
        "topics": sorted(topics),
        "sources": sorted(sources),
        "years": sorted(years, reverse=True),
    }


class KnowledgeBase:
    """Loads the persisted Chroma collection and serves hybrid, filterable search."""

    def __init__(self, embeddings: Embeddings | None = None) -> None:
        self._embeddings = embeddings or make_embeddings()
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        data = self._store.get(include=["documents", "metadatas"])
        self._all_texts: list[str] = data.get("documents") or []
        self._all_metas: list[dict] = data.get("metadatas") or []
        self._bm25 = BM25(self._all_texts)
        self._bm25_index = {text: i for i, text in enumerate(self._all_texts)}

    @property
    def size(self) -> int:
        """Total number of indexed chunks (0 means the store needs `make ingest`)."""
        return len(self._all_texts)

    def facets(self) -> dict[str, list[str]]:
        """Available topic / source / year filter values across the index (for the filter UI)."""
        return compute_facets(self._all_metas)

    def stats(self) -> dict[str, dict[str, int]]:
        """Chunk counts grouped by subject and by source document (Knowledge Base tab)."""
        by_subject: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for meta in self._all_metas:
            subject = meta.get("subject", "?")
            source = meta.get("source", "?")
            by_subject[subject] = by_subject.get(subject, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1
        return {"by_subject": by_subject, "by_source": by_source}

    def add_chunks(self, chunks: list[RetrievedChunk]) -> int:
        """Add citable passages to the *live* index and return how many were newly added.

        Used to promote approved external passages (see :mod:`src.rag.promote`) so they are
        retrievable **immediately** — no re-ingest. Each passage is embedded and persisted to
        the Chroma collection, and the in-memory BM25 corpus is rebuilt so hybrid search sees
        it at once. Deduplicates by exact chunk text (also guarding against re-adding a
        passage already in the persisted store), so calling it twice is safe. Chroma rejects
        ``None`` metadata values, so those are dropped.
        """
        new = [c for c in chunks if c.text not in self._bm25_index]
        if not new:
            return 0
        self._store.add_texts(
            texts=[c.text for c in new],
            metadatas=[{k: v for k, v in c.metadata.items() if v is not None} for c in new],
        )
        for chunk in new:
            self._bm25_index[chunk.text] = len(self._all_texts)
            self._all_texts.append(chunk.text)
            self._all_metas.append(dict(chunk.metadata))
        self._bm25 = BM25(self._all_texts)  # corpus is small; a full rebuild is cheap
        return len(new)

    def search(
        self,
        query: str,
        subjects: list[str] | None = None,
        difficulty: str | None = None,
        k: int = 4,
        alpha: float = 0.7,
        *,
        filters: MetadataFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid search: a true union of vector and BM25 retrieval, blended by ``alpha``.

        Both retrievers contribute candidates independently — the top ``fetch_k`` vector hits
        *and* the top ``fetch_k`` BM25 hits (each honouring the metadata filter). Their union is
        scored ``alpha * vec + (1 - alpha) * bm25``; a candidate found by only one retriever
        scores 0 on the other component. This is what lets a strongly keyword-matching chunk
        surface even when it falls outside the vector top-``fetch_k`` — the earlier
        vector-then-rerank design could never reach it.

        ``alpha`` weights the vector score; ``1 - alpha`` weights BM25. Metadata filters
        restrict by subject (ml/dl/ai/overlap) and, optionally, chunk difficulty; ``filters``
        adds the user-chosen topic/source/year facets (see :class:`MetadataFilters`). Every
        active predicate is pushed into the Chroma ``where`` clause for the vector side and
        mirrored in ``_passes_filter`` for BM25-only candidates, so both retrievers filter
        identically.
        """
        if not self._all_texts:
            return []
        conditions: list[dict] = []
        if subjects:
            conditions.append({"subject": {"$in": subjects}})
        if difficulty:
            conditions.append({"difficulty": difficulty})
        if filters:
            conditions.extend(filters.conditions())
        where = (
            conditions[0]
            if len(conditions) == 1
            else ({"$and": conditions} if conditions else None)
        )

        def _passes_filter(meta: dict) -> bool:
            """Apply the same subject/difficulty/facet filter to BM25-only candidates as Chroma."""
            if subjects and meta.get("subject") not in subjects:
                return False
            if difficulty and meta.get("difficulty") != difficulty:
                return False
            if filters and not filters.passes(meta):
                return False
            return True

        fetch_k = max(k * 3, 10)
        hits = self._vector_hits(query, fetch_k, where)
        bm25_scores = self._bm25.scores(query)
        max_bm25 = max(bm25_scores, default=0.0)

        # Vector component per candidate text; BM25-only candidates default to 0.0.
        vec_by_text: dict[str, float] = {}
        for doc, distance in hits:
            vec_by_text[doc.page_content] = 1.0 - distance  # cosine distance -> similarity

        # BM25's own top candidates (filtered), so a keyword-only match can enter the pool.
        bm25_top: list[int] = []
        if max_bm25 > 0:
            ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            for i in ranked:
                if len(bm25_top) >= fetch_k:
                    break
                if bm25_scores[i] > 0 and _passes_filter(self._all_metas[i]):
                    bm25_top.append(i)

        # Union of the two candidate sets, keyed by chunk text.
        candidates: dict[str, dict] = {}
        for doc, _distance in hits:
            candidates[doc.page_content] = doc.metadata
        for i in bm25_top:
            candidates.setdefault(self._all_texts[i], self._all_metas[i])

        chunks: list[RetrievedChunk] = []
        for text, meta in candidates.items():
            vec = vec_by_text.get(text, 0.0)
            idx = self._bm25_index.get(text)
            bm = bm25_scores[idx] / max_bm25 if (idx is not None and max_bm25 > 0) else 0.0
            chunks.append(
                RetrievedChunk(
                    text=text,
                    metadata=meta,
                    vector_score=round(vec, 4),
                    bm25_score=round(bm, 4),
                    score=round(alpha * vec + (1 - alpha) * bm, 4),
                )
            )
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:k]

    def _vector_hits(self, query: str, fetch_k: int, where: dict | None) -> list:
        """Run Chroma similarity search, turning a dimension mismatch into a clear message.

        A stale index (built with a different embedding backend/model than the one now
        configured) makes Chroma raise a dimension error. Re-raise it as an actionable
        instruction rather than letting a cryptic vector-store error reach the user.
        """
        try:
            return self._store.similarity_search_with_score(query, k=fetch_k, filter=where)
        except Exception as exc:
            if "dimension" in str(exc).lower():
                raise RuntimeError(
                    "The vector index was built with a different embedding model than the one "
                    "now configured (see EMBEDDING_BACKEND / model in your settings). Re-run "
                    "`make ingest` to rebuild the index with the current embeddings."
                ) from exc
            raise

    def mmr_select(
        self, chunks: list[RetrievedChunk], *, k: int, lambda_: float = 0.5
    ) -> list[RetrievedChunk]:
        """Diversify a candidate pool down to ``k`` chunks by Maximal Marginal Relevance.

        Embeds the candidate texts once and reorders them with :func:`mmr_rank`, using each
        chunk's hybrid ``score`` as the relevance term. When there is nothing to diversify
        (``<= k`` candidates) the pool is returned unchanged. Graceful: if embedding the
        candidates fails, falls back to the incoming (score-ordered) top-``k`` so retrieval
        never regresses below the plain cut.
        """
        if len(chunks) <= k:
            return list(chunks)
        try:
            vectors = self._embeddings.embed_documents([c.text for c in chunks])
        except Exception:
            return list(chunks[:k])
        order = mmr_rank([c.score for c in chunks], vectors, k, lambda_)
        return [chunks[i] for i in order]


REWRITE_SYSTEM = (
    "You rewrite user questions into a single, self-contained search query for a "
    "machine-learning/AI knowledge base. Resolve pronouns using the conversation, "
    "expand acronyms once, and output ONLY the rewritten query — no explanations."
)


def rewrite_query(
    llm: BaseChatModel,
    question: str,
    history: list[tuple[str, str]],
    *,
    meter: UsageMeter | None = None,
) -> str:
    """LLM query translation; falls back to the original question on any failure.

    When a :class:`~src.utils.UsageMeter` is passed, the call's token usage is recorded on it
    (exact from the reply when the provider reports it, else estimated).
    """
    context = "\n".join(f"{role}: {text}" for role, text in history[-4:])
    prompt = (
        f"Conversation so far:\n{context}\n\nQuestion to rewrite: {question}"
        if context
        else question
    )
    try:
        reply = llm.invoke([SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=prompt)])
        rewritten = str(reply.content).strip().strip('"')
        if meter is not None:
            meter.record_call(reply, f"{REWRITE_SYSTEM}\n{prompt}", rewritten)
    except Exception:
        logger.warning("Query rewrite failed; using the question as asked.", exc_info=True)
        return question
    return rewritten or question


EXPAND_SYSTEM = (
    "You expand a user question into several diverse, self-contained search queries for a "
    "machine-learning/AI knowledge base. Resolve pronouns using the conversation and expand "
    "acronyms once. Vary the angle across the queries — a literal rephrasing, a broader "
    "framing, a narrower/technical framing, synonym variants — so together they retrieve "
    "complementary passages."
)


def expand_queries(
    llm: BaseChatModel,
    question: str,
    history: list[tuple[str, str]] | None = None,
    n: int = 3,
    *,
    meter: UsageMeter | None = None,
) -> list[str]:
    """Rewrite ``question`` into up to ``n`` diverse search queries (RAG-Fusion).

    Returns a de-duplicated, order-preserving list of at most ``n`` self-contained queries.
    Degrades gracefully in three stages so recall never drops below the single-query
    baseline: the structured multi-query LLM call first, then :func:`rewrite_query`, then the
    raw ``question`` — the returned list is therefore always non-empty. When a
    :class:`~src.utils.UsageMeter` is passed, whichever LLM call runs records its usage on it
    (estimated for the structured call, which does not report ``usage_metadata``).
    """
    if n <= 1:
        return [rewrite_query(llm, question, history or [], meter=meter)]
    context = "\n".join(f"{role}: {text}" for role, text in (history or [])[-4:])
    prompt = (
        f"Conversation so far:\n{context}\n\nQuestion to expand: {question}"
        if context
        else question
    )
    try:
        structured = llm.with_structured_output(QueryExpansion, method="json_schema")
        expansion = structured.invoke(
            [SystemMessage(content=EXPAND_SYSTEM), HumanMessage(content=prompt)]
        )
        if isinstance(expansion, QueryExpansion):
            queries = _dedupe_preserving_order(
                q.strip() for q in expansion.queries if q and q.strip()
            )
            if queries:
                if meter is not None:
                    meter.record_call(expansion, f"{EXPAND_SYSTEM}\n{prompt}", "\n".join(queries))
                return queries[:n]
    except Exception:
        logger.warning("Query expansion failed; falling back to a single query.", exc_info=True)
    return [rewrite_query(llm, question, history or [], meter=meter)]


def _dedupe_preserving_order(items: object) -> list[str]:
    """De-duplicate strings case-insensitively while keeping first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], *, rrf_k: int = 60, limit: int | None = None
) -> list[RetrievedChunk]:
    """Fuse several ranked chunk lists into one ranking by Reciprocal Rank Fusion.

    RRF scores each chunk by ``sum(1 / (rrf_k + rank))`` over the lists it appears in (rank
    0-based), so a passage retrieved by *several* query variants rises above one that ranks
    highly for a single phrasing — the point of multi-query retrieval. Chunks are keyed by
    text; the representative kept for each is the instance with the best hybrid ``score``
    (its per-retriever vector/BM25 scores stay intact for the trace). ``rrf_k`` damps the
    influence of top ranks (the standard constant is 60); ``limit`` caps the output.
    """
    fused: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            fused[chunk.text] = fused.get(chunk.text, 0.0) + 1.0 / (rrf_k + rank + 1)
            current = best_chunk.get(chunk.text)
            if current is None or chunk.score > current.score:
                best_chunk[chunk.text] = chunk
    ordered = sorted(fused, key=lambda text: fused[text], reverse=True)
    chunks = [best_chunk[text] for text in ordered]
    return chunks[:limit] if limit is not None else chunks


def mmr_rank(
    relevances: list[float], embeddings: list[list[float]], k: int, lambda_: float = 0.5
) -> list[int]:
    """Order candidates by Maximal Marginal Relevance, returning selected indices best-first.

    MMR greedily builds a set that balances *relevance* to the query against *novelty* versus
    what has already been picked, so several near-identical high-scoring passages don't all
    occupy the top_k. At each step it picks the candidate maximising::

        lambda_ * relevance[i] - (1 - lambda_) * max(sim(i, j) for j already selected)

    ``lambda_`` weights the trade-off: ``1.0`` reproduces the plain relevance order (no
    diversity), ``0.0`` maximises diversity. ``relevances`` is any per-candidate relevance
    score (here the hybrid retrieval score); ``embeddings`` must be **unit-normalised** so a
    dot product is cosine similarity — both project embedding backends normalise, so this holds.
    Pure and deterministic (ties broken by lower index): unit-testable without a real model.
    """
    n = len(relevances)
    if n == 0:
        return []
    k = min(k, n)
    rel = np.asarray(relevances, dtype=float)
    mat = np.asarray(embeddings, dtype=float)
    sim = mat @ mat.T  # cosine similarity between candidates (rows are unit vectors)

    selected: list[int] = []
    remaining = list(range(n))
    while len(selected) < k and remaining:
        if not selected:
            # Seed with the most relevant candidate (nothing to be diverse from yet).
            best = max(remaining, key=lambda i: (rel[i], -i))
        else:

            def score(i: int) -> float:
                penalty = max(sim[i][j] for j in selected)
                return lambda_ * rel[i] - (1.0 - lambda_) * penalty

            best = max(remaining, key=lambda i: (score(i), -i))
        selected.append(best)
        remaining.remove(best)
    return selected


RERANK_SYSTEM = (
    "You are a passage reranker for a machine-learning/AI knowledge base. Given a question "
    "and a numbered list of candidate passages, reorder the candidates from most to least "
    "relevant to answering that question. A passage that directly answers the question ranks "
    "above one that only mentions the topic in passing. Judge relevance only — not passage "
    "length or writing style — and return every candidate number exactly once, best first."
)


def rerank_chunks(
    llm: BaseChatModel,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    top_n: int | None = None,
    meter: UsageMeter | None = None,
) -> list[RetrievedChunk]:
    """LLM listwise rerank: reorder candidates by relevance, then keep the top ``top_n``.

    A second retrieval stage over the first-stage hybrid/fused pool. The candidate passages
    are shown to the model as one numbered list and it returns the numbers in best-first
    order (structured :class:`~src.core.schemas.Reranking`), so cross-passage comparisons
    refine the final ranking. Degrades gracefully: on any failure the original retrieval order
    is kept, and any candidate the model omits is appended in its original position — so
    reranking can only *reorder* the pool, never drop recall, before the ``top_n`` cut.

    When a :class:`~src.utils.UsageMeter` is passed, the call's usage is recorded on it
    (estimated — the candidate catalogue is the bulk of the input, so this is the request's
    largest previously-uncounted auxiliary call).
    """
    limit = top_n if top_n is not None else len(chunks)
    if len(chunks) <= 1:
        return chunks[:limit]
    catalogue = "\n\n".join(f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(chunks))
    prompt = f"Question: {question}\n\nCandidate passages:\n{catalogue}"
    try:
        structured = llm.with_structured_output(Reranking, method="json_schema")
        reranking = structured.invoke(
            [SystemMessage(content=RERANK_SYSTEM), HumanMessage(content=prompt)]
        )
        if isinstance(reranking, Reranking):
            if meter is not None:
                meter.record_call(reranking, f"{RERANK_SYSTEM}\n{prompt}", str(reranking.ranking))
            order = _resolve_ranking(reranking.ranking, len(chunks))
            return [chunks[i] for i in order][:limit]
    except Exception:
        logger.warning("LLM reranking failed; keeping the retrieval order.", exc_info=True)
    return chunks[:limit]


def _resolve_ranking(ranking: list[int], n: int) -> list[int]:
    """Turn 1-based model indices into a full, valid 0-based order over ``n`` chunks.

    Keeps the first occurrence of each in-range index and drops out-of-range/duplicate ones,
    then appends any index the model omitted in its original order — so the result is always a
    permutation of ``range(n)`` (every chunk retained, only reordered).
    """
    seen: set[int] = set()
    order: list[int] = []
    for value in ranking:
        idx = value - 1
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    order.extend(idx for idx in range(n) if idx not in seen)
    return order
