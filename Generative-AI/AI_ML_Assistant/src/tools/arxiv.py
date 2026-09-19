"""arXiv paper-search tool."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import tool

if TYPE_CHECKING:
    import arxiv

# Trailing arXiv version suffix, e.g. the "v1" in ".../abs/1903.10563v1".
_VERSION_SUFFIX = re.compile(r"v\d+$")
# An arXiv field prefix already present in the query (ti:, abs:, au:, all:, cat:, id:).
_HAS_FIELD_PREFIX = re.compile(r"\b(ti|abs|au|all|cat|id):", re.IGNORECASE)

# arXiv categories relevant to this app's AI/ML/DS domain, as {friendly label: cat code}.
# "" means "any category" (no ``cat:`` filter). Used to populate the playground's picker.
ARXIV_CATEGORIES: dict[str, str] = {
    "Any category": "",
    "Machine Learning (cs.LG)": "cs.LG",
    "Computation & Language / NLP (cs.CL)": "cs.CL",
    "Artificial Intelligence (cs.AI)": "cs.AI",
    "Computer Vision (cs.CV)": "cs.CV",
    "Neural & Evolutionary (cs.NE)": "cs.NE",
    "Information Retrieval (cs.IR)": "cs.IR",
    "Statistics · ML (stat.ML)": "stat.ML",
}

# Filler words that add nothing to an arXiv query but drag ranking toward unrelated papers
# (e.g. "arxiv"/"papers" in "find recent arxiv papers on X"). Stripped from free text.
_NOISE_WORDS = frozenset(
    {
        "arxiv",
        "paper",
        "papers",
        "article",
        "articles",
        "find",
        "show",
        "give",
        "get",
        "list",
        "search",
        "me",
        "please",
        "recent",
        "latest",
        "newest",
        "about",
        "on",
        "the",
        "a",
        "an",
        "of",
        "for",
        "and",
        "some",
        "any",
        "related",
        "regarding",
    }
)


def _clean_terms(text: str) -> list[str]:
    """Split free text into meaningful terms, dropping filler words that hurt ranking."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#-]*", text)
    return [w for w in words if w.lower() not in _NOISE_WORDS]


def _clean_freetext(query: str) -> str:
    """Filler-stripped free-text query; falls back to the original if nothing survives."""
    return " ".join(_clean_terms(query)) or query


def build_arxiv_query(
    keywords: str = "",
    author: str = "",
    title: str = "",
    category: str = "",
) -> str:
    """Compose a field-prefixed arXiv query from structured parts (pure, no network).

    Each supplied part is wrapped in its arXiv field prefix and ANDed with the rest, e.g.
    ``build_arxiv_query(keywords="diffusion models", author="Ho", category="cs.LG")`` ->
    ``all:diffusion AND all:models AND au:"Ho" AND cat:cs.LG``. Multi-word authors/titles are
    quoted so arXiv matches them as a phrase; keyword filler words are dropped. This is the
    deterministic alternative to feeding arXiv a raw sentence — which ranks poorly because a
    stray given name or the word "papers" matches unrelated work. Returns "" if all blank.
    """
    parts: list[str] = []
    if title.strip():
        t = title.strip()
        parts.append(f'ti:"{t}"' if " " in t else f"ti:{t}")
    if author.strip():
        a = author.strip()
        parts.append(f'au:"{a}"' if " " in a else f"au:{a}")
    parts.extend(f"all:{term}" for term in _clean_terms(keywords))
    if category.strip():
        parts.append(f"cat:{category.strip()}")
    return " AND ".join(parts)


def _run(query: str, criterion: arxiv.SortCriterion, max_results: int) -> list:
    """Execute one arXiv query and return the raw result list."""
    import arxiv

    search = arxiv.Search(query=query, max_results=max_results, sort_by=criterion)
    return list(arxiv.Client().results(search))


def _format(paper: arxiv.Result) -> str:
    """Render one paper as a markdown bullet with a clean, version-stripped abstract link."""
    authors = ", ".join(a.name for a in paper.authors[:3])
    more = " et al." if len(paper.authors) > 3 else ""
    abs_url = _VERSION_SUFFIX.sub("", paper.entry_id.replace("http://", "https://"))
    return f"- **[{paper.title}]({abs_url})** ({paper.published:%Y-%m}) — {authors}{more}"


@tool
def search_arxiv(
    query: str,
    max_results: int = 5,
    sort: Literal["relevance", "recent"] = "relevance",
) -> str:
    """Search arXiv for papers. Use for 'find/recommend papers about X' requests.

    Build a *structured* arXiv query rather than passing a whole natural-language
    question — arXiv ranks field-prefixed queries far better than a raw sentence.
    Available prefixes: ``au:`` (author), ``ti:`` (title), ``abs:`` (abstract),
    ``cat:`` (category, e.g. cs.LG), and ``all:``; combine with AND/OR.

    Examples:
        * "Giuseppe Carleo's most influential ML paper" -> ``au:Carleo cat:cs.LG``
        * "the transformer paper" -> ``ti:"attention is all you need"``
        * "recent papers on RAG" -> ``abs:"retrieval augmented generation"`` with sort='recent'

    Args:
        query: A structured arXiv query (prefer field prefixes above), a paper title,
            or plain keywords, e.g. 'machine learning and the physical sciences'.
        max_results: Number of papers to return (1-10).
        sort: 'relevance' ranks by how well papers match the query (the default, best for
            finding a specific paper or topic); 'recent' returns the newest submissions
            first (use only for a "what's new" feed).
    """
    import arxiv  # network client, deferred import

    max_results = max(1, min(int(max_results), 10))

    if sort == "recent":
        # Clean filler from a raw sentence so a "newest first" browse isn't dragged off by
        # stop-words matching unrelated recent submissions; leave field-prefixed queries as-is.
        recent_query = query if _HAS_FIELD_PREFIX.search(query) else _clean_freetext(query)
        papers = _run(recent_query, arxiv.SortCriterion.SubmittedDate, max_results)
    else:
        # arXiv's default all-fields relevance ranking is noisy: a multi-word query
        # like "machine learning and the physical sciences" matches "physical sciences"
        # in unrelated physics papers. So first try the query as an exact TITLE phrase —
        # that reliably surfaces a specific paper — and only fall back to the broad
        # all-fields search when the title match finds nothing (or the caller already
        # supplied their own field prefix).
        papers = []
        if _HAS_FIELD_PREFIX.search(query):
            papers = _run(query, arxiv.SortCriterion.Relevance, max_results)
        else:
            # Raw natural language: strip filler, try it as an exact title first (precise for
            # a specific paper), then fall back to a cleaned all-fields relevance search.
            cleaned = _clean_freetext(query)
            papers = _run(f'ti:"{cleaned}"', arxiv.SortCriterion.Relevance, max_results)
            if not papers:
                papers = _run(cleaned, arxiv.SortCriterion.Relevance, max_results)

    return "\n".join(_format(p) for p in papers) or "No papers found for that query."
