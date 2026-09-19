"""Promote approved external passages (arXiv / web) into the curated knowledge base.

Corrective-RAG augmentation (:mod:`src.core.sources`) retrieves useful passages
from external sources when the KB does not have enough information. These
passages are cited for one answer and then discarded. This module allows a
human to review and approve valuable passages, then promote them into the KB
for future use.

Two persistence layers are used intentionally (the A+B design):

* **Source of truth (A):** Approved passages are saved as curated Markdown
  notes under ``data/promoted/``. Front matter stores subject, topic, and
  provenance details, allowing these notes to be included again during the next
  ``make ingest`` and survive a complete index rebuild.

* **Live add (B):** The same approved passages are also passed to
  :meth:`~src.rag.retriever.KnowledgeBase.add_chunks`, making them searchable
  immediately in the current session without waiting for a new ingest.

This module is pure (filesystem operations + string formatting only). It does
not use network calls or Streamlit, so capture, saving, and duplicate-checking
logic can be tested offline. The live index update is handled by
:class:`~src.rag.retriever.KnowledgeBase`, while human approval is handled by
the UI. This keeps the knowledge base curated, which is the foundation of a
grounded assistant.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import DATA_DIR
from src.rag.retriever import RetrievedChunk
from src.security import has_domain_vocabulary

#: Sub-folder of ``data/`` that holds promoted notes. Not one of the four subject folders, so
#: a note's subject comes from its front matter (defaulting to ``overlap`` — shared, retrieved
#: in every subject view — when unset).
PROMOTED_SUBDIR = "promoted"

#: Metadata ``difficulty`` value that marks an externally-sourced passage (set by the sources
#: in :mod:`src.core.sources`); used to pick promotion candidates out of an answer's context.
EXTERNAL_MARKER = "external"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ID_MAX_LEN = 80


def promoted_dir(data_dir: Path = DATA_DIR) -> Path:
    """Return the directory promoted notes are written to (``<data_dir>/promoted``)."""
    return data_dir / PROMOTED_SUBDIR


def promotion_id(chunk: RetrievedChunk) -> str:
    """Stable, filesystem-safe id for a passage — the dedup key and note filename stem.

    Derived from the most *specific* provenance available so two different web results from
    the same domain never collide: the full ``origin``/``url`` first, then the ``source`` id,
    finally a prefix of the text. Slugged to ``[a-z0-9-]`` and length-capped.
    """
    meta = chunk.metadata
    raw = meta.get("origin") or meta.get("url") or meta.get("source") or chunk.text[:_ID_MAX_LEN]
    slug = _SLUG_RE.sub("-", str(raw).lower()).strip("-")
    return (slug[:_ID_MAX_LEN].rstrip("-")) or "passage"


def external_candidates(
    sources: Sequence[dict[str, Any]], contexts: Sequence[str]
) -> list[RetrievedChunk]:
    """Reconstruct the external passages from an answer's ``sources`` + ``contexts``.

    ``AnswerBundle.sources`` (metadata, incl. ``ref``/``score``) is index-aligned with
    ``AnswerBundle.contexts`` (the full passage text). Only passages marked ``external`` are
    promotable — KB chunks are already curated. Returns them as :class:`RetrievedChunk` so the
    rest of the promotion pipeline is uniform.
    """
    out: list[RetrievedChunk] = []
    # ``sources`` and ``contexts`` are built index-aligned from the same chunk list
    # (:class:`~src.core.service.AnswerBundle`), so ``strict=True`` documents that invariant
    # and turns a would-be silent drop of trailing passages into a loud failure — a length
    # mismatch here can only mean an upstream bug broke the alignment, never user input.
    for src_, text in zip(sources, contexts, strict=True):
        if src_.get("difficulty") != EXTERNAL_MARKER:
            continue
        meta = {k: v for k, v in src_.items() if k not in ("ref", "score")}
        out.append(
            RetrievedChunk(
                text=text,
                metadata=meta,
                vector_score=0.0,
                bm25_score=0.0,
                score=float(src_.get("score") or 0.0),
            )
        )
    return out


def is_on_domain_passage(text: str, metadata: Mapping[str, Any]) -> bool:
    """Does an external passage read as ML/AI — i.e. worth offering for KB promotion?

    Reuses the app's input-gate vocabulary check (:func:`src.security.has_domain_vocabulary`)
    over the passage's title *and* body, so "off-topic" here means exactly what it means at
    the question gate — one source of truth, no second keyword list to drift. Off-domain
    arXiv/web results (e.g. particle-physics papers surfaced by noisy arXiv relevance ranking)
    return ``False``, letting the promote panel tuck them away instead of cluttering the
    curated-KB candidate list. This is a heuristic filter for presentation only — nothing is
    dropped, and a flagged passage can still be promoted by hand if the human disagrees.
    """
    title = str(metadata.get("title") or "")
    return has_domain_vocabulary(f"{title} {text}")


def _front_matter_value(value: Any) -> str:
    """Render a front-matter value on one line (newlines would break the ``key: value`` block)."""
    return " ".join(str(value).split())


def chunk_to_markdown(chunk: RetrievedChunk, *, subject: str) -> str:
    """Serialise a passage as a curated Markdown note with ingest-compatible front matter.

    The provenance URL is written as ``source:`` — :func:`src.rag.ingest.parse_front_matter`
    maps that to the ``origin`` metadata (keeping ``source`` = filename), so citations still
    point back to the original paper/page. ``difficulty`` stays ``external`` so promoted
    content is visibly externally-sourced in stats and citations.
    """
    meta = chunk.metadata
    provenance = meta.get("origin") or meta.get("url") or meta.get("source") or ""
    fields = {
        "title": meta.get("title") or "Promoted external passage",
        "subject": subject,
        "topic": meta.get("topic") or "external",
        "difficulty": EXTERNAL_MARKER,
        "source": provenance,
        "year": meta.get("year") or "",
    }
    lines = ["---"]
    lines += [f"{k}: {_front_matter_value(v)}" for k, v in fields.items() if v]
    lines += ["---", "", chunk.text.strip(), ""]
    return "\n".join(lines)


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of a promotion batch: which notes were written and which were already present."""

    written: list[str]  # filenames newly created under data/promoted/
    skipped: list[str]  # filenames skipped because that passage was already promoted

    @property
    def n_written(self) -> int:
        """Number of new notes written."""
        return len(self.written)

    @property
    def n_skipped(self) -> int:
        """Number of duplicate passages skipped."""
        return len(self.skipped)


def promote_chunks(
    chunks: Iterable[RetrievedChunk], *, subject: str = "overlap", data_dir: Path = DATA_DIR
) -> PromotionResult:
    """Write each passage as a curated note under ``data/promoted/``; dedup by promotion id.

    Idempotent: a passage whose note already exists (same :func:`promotion_id`) is skipped, so
    promoting the same source twice — even across sessions — never creates duplicates. Returns
    a :class:`PromotionResult` summarising what changed. Does not touch the live index; the
    caller pairs this with :meth:`~src.rag.retriever.KnowledgeBase.add_chunks` for immediacy.
    """
    dest = promoted_dir(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: list[str] = []
    for chunk in chunks:
        path = dest / f"{promotion_id(chunk)}.md"
        if path.exists():
            skipped.append(path.name)
            continue
        path.write_text(chunk_to_markdown(chunk, subject=subject), encoding="utf-8")
        written.append(path.name)
    return PromotionResult(written=written, skipped=skipped)
