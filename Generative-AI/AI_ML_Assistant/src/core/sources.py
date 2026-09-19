"""
Pluggable external knowledge sources for Corrective-RAG augmentation.

When the ``grade`` step finds the knowledge base insufficient for an on-domain
query, the ``augment`` step retrieves additional citable passages from
configured sources and adds them to the context, keeping answers grounded
instead of relying on unverified parametric memory.

The layer uses a pluggable design: each provider implements the
:class:`ExternalSource` interface and registers in
:data:`SOURCE_REGISTRY`. ``RagSettings.augment_sources`` controls active
providers, and :func:`build_sources` creates their instances. Adding a new
provider only requires a new subclass and registry entry, without changing
steps or engines.

:class:`ArxivSource` is included first because it requires no additional
dependency or API key (the ``arxiv`` client is already used by
:mod:`src.tools.arxiv`). All sources fail gracefully: import or network errors
return an empty list so the pipeline continues normally.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from src.rag.retriever import RetrievedChunk

# Trailing arXiv version suffix, e.g. the "v1" in ".../abs/1903.10563v1".
_VERSION_SUFFIX = re.compile(r"v\d+$")


class ExternalSource(ABC):
    """A provider of external, citable passages for context augmentation.

    Concrete sources set a unique :attr:`name` (their registry key) and implement
    :meth:`fetch`, returning :class:`~src.rag.retriever.RetrievedChunk` objects so augmented
    passages flow through the rest of the pipeline exactly like KB chunks — numbered,
    cited, and shown in the trace. Implementations MUST NOT raise on network/parse errors;
    they return an empty list instead.
    """

    #: Short, stable key used in :data:`SOURCE_REGISTRY` and ``RagSettings.augment_sources``.
    name: str = ""

    @abstractmethod
    def fetch(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        """Return up to ``k`` citable passages relevant to ``query`` (never raises)."""
        raise NotImplementedError


class ArxivSource(ExternalSource):
    """External source backed by arXiv (title + abstract as citable passages).

    Reuses the already-present ``arxiv`` client — no new dependency, no API key. Each paper
    becomes a :class:`RetrievedChunk` whose text is ``title + abstract`` and whose metadata
    carries a stable ``source`` id and the abstract-page ``url`` for citation.
    """

    name = "arxiv"

    def fetch(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        """Search arXiv by relevance and return up to ``k`` passages; [] on any failure."""
        try:
            import arxiv  # deferred: network client, keeps import-time offline-safe
        except Exception:
            return []
        k = max(1, min(int(k), 10))
        try:
            search = arxiv.Search(query=query, max_results=k, sort_by=arxiv.SortCriterion.Relevance)
            results = list(arxiv.Client().results(search))
        except Exception:
            return []

        chunks: list[RetrievedChunk] = []
        for i, paper in enumerate(results):
            abs_url = _VERSION_SUFFIX.sub("", paper.entry_id.replace("http://", "https://"))
            short_id = abs_url.rsplit("/", 1)[-1]
            authors = ", ".join(a.name for a in paper.authors[:3])
            summary = " ".join((paper.summary or "").split())
            chunks.append(
                RetrievedChunk(
                    text=f"{paper.title}\n\n{summary}",
                    metadata={
                        "source": f"arXiv:{short_id}",
                        "title": paper.title,
                        "topic": "arXiv paper",
                        "difficulty": "external",
                        "origin": abs_url,
                        "url": abs_url,
                        "authors": authors,
                        "year": f"{paper.published:%Y}" if paper.published else "",
                    },
                    # External hits have no vector/BM25 score; expose a rank-based relevance
                    # so downstream sorting and the trace stay well-defined.
                    vector_score=0.0,
                    bm25_score=0.0,
                    score=round(max(0.1, 0.6 - 0.05 * i), 4),
                )
            )
        return chunks


# Prompt for the web-search helper call: keep it terse and grounded so the model spends its
# effort on retrieval (via the web plugin) rather than composing a long essay.
_WEB_SYSTEM = (
    "You are a web-search assistant for an AI/ML learning tool. Use the web to find "
    "current, factual information answering the user's question, and ground every claim in "
    "the sources you cite. Be concise."
)


def _ann_get(obj: Any, key: str) -> Any:
    """Read ``key`` from an annotation entry that may be a dict or a pydantic object."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _chunks_from_annotations(annotations: Any, k: int) -> list[RetrievedChunk]:
    """Turn OpenRouter ``url_citation`` annotations into up to ``k`` citable chunks.

    Each web citation carries a URL, a title, and (optionally) a snippet; we render it as a
    :class:`RetrievedChunk` so web results flow through the pipeline exactly like KB and
    arXiv passages — numbered, cited, and shown in the trace. Non-citation annotations and
    entries without a URL are skipped. Kept as a pure function so it is unit-testable
    offline, with no network or SDK involved.
    """
    chunks: list[RetrievedChunk] = []
    for i, ann in enumerate(annotations or []):
        if _ann_get(ann, "type") != "url_citation":
            continue
        cite = _ann_get(ann, "url_citation") or ann
        url = _ann_get(cite, "url")
        if not url:
            continue
        title = _ann_get(cite, "title") or url
        content = (_ann_get(cite, "content") or "").strip()
        chunks.append(
            RetrievedChunk(
                text=f"{title}\n\n{content}" if content else title,
                metadata={
                    "source": f"web:{urlparse(url).netloc or 'result'}",
                    "title": title,
                    "topic": "web result",
                    "difficulty": "external",
                    "origin": url,
                    "url": url,
                },
                # No vector/BM25 score for external hits; expose a rank-based relevance so
                # downstream sorting and the trace stay well-defined (arXiv uses the same idea).
                vector_score=0.0,
                bm25_score=0.0,
                score=round(max(0.1, 0.55 - 0.05 * i), 4),
            )
        )
        if len(chunks) >= k:
            break
    return chunks


class WebSearchSource(ExternalSource):
    """External source backed by OpenRouter's built-in web plugin (live web search).

    No new dependency and no new network host: it reuses the already-installed ``openai``
    SDK and talks only to OpenRouter (the provider the app already uses), enabling the
    ``web`` plugin on a cheap classification-tier model. The model's ``url_citation``
    annotations become citable passages. Costs a small amount per call (web-plugin fee +
    tokens), so it is opt-in via ``RagSettings.augment_sources`` rather than on by default.
    Degrades gracefully: missing key, missing SDK, or any network/parse error yields ``[]``.
    """

    name = "web"

    def fetch(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        """Search the web via OpenRouter and return up to ``k`` citable passages; [] on failure."""
        try:
            from openai import OpenAI  # already installed via langchain-openai
        except Exception:
            return []
        from src.config import OPENROUTER_BASE_URL, ROUTER_MODEL, openrouter_api_key

        key = openrouter_api_key()
        if not key:
            return []  # no credentials -> stay offline, let the pipeline answer without web
        k = max(1, min(int(k), 6))
        try:
            client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
            resp = client.chat.completions.create(
                model=ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": _WEB_SYSTEM},
                    {"role": "user", "content": query},
                ],
                extra_body={"plugins": [{"id": "web", "max_results": k}]},
                temperature=0.0,
            )
        except Exception:
            return []

        msg = resp.choices[0].message
        annotations = getattr(msg, "annotations", None)
        if annotations is None:  # SDK may stash unknown fields in model_extra
            annotations = (getattr(msg, "model_extra", None) or {}).get("annotations")
        chunks = _chunks_from_annotations(annotations, k)
        if chunks:
            return chunks
        # No structured citations came back: fall back to the answer text as one passage so
        # the augment step still has grounded material to work with.
        content = (getattr(msg, "content", "") or "").strip()
        if content:
            return [
                RetrievedChunk(
                    text=content,
                    metadata={
                        "source": "web:search",
                        "title": "Web search result",
                        "topic": "web result",
                        "difficulty": "external",
                        "origin": "",
                        "url": "",
                    },
                    vector_score=0.0,
                    bm25_score=0.0,
                    score=0.3,
                )
            ]
        return []


# Registry of available sources, keyed by :attr:`ExternalSource.name`. Extend by adding a
# subclass and one entry here; ``RagSettings.augment_sources`` chooses which are active.
SOURCE_REGISTRY: dict[str, type[ExternalSource]] = {
    ArxivSource.name: ArxivSource,
    WebSearchSource.name: WebSearchSource,
}


def build_sources(names: Iterable[str]) -> list[ExternalSource]:
    """Instantiate the registered sources named in ``names`` (unknown names are ignored)."""
    return [SOURCE_REGISTRY[n]() for n in names if n in SOURCE_REGISTRY]
