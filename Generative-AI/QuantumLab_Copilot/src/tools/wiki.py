"""Wikipedia, for the background the corpus does not carry.

The corpus is four notes written for this project, deep on the transverse-field
Ising chain and empty on everything a question can reasonably touch on the way
there -- what a Hamiltonian is, who Kramers and Wannier were, what a
Jordan-Wigner transformation is called in other fields. Refusing those is
correct but unhelpful when a free, citable encyclopedia answers them.

So this sits *between* the corpus and arXiv in what it is worth. Corpus passages
are evidence: written here, reviewed here, cited by name. arXiv results are
pointers nothing can check. An encyclopedia article is neither -- it is
background, attributable to a specific revision of a specific page. It is
labelled as unverified alongside arXiv, because from this application's point of
view the distinction that matters is whether *we* checked it, and we did not.

Two requests, both to Wikipedia's public API and neither needing a key: a search
for the best-matching title, then that page's summary. Two rather than one
because a search result gives a title and a snippet, and a snippet is not enough
to answer with -- while asking for a summary requires already knowing the exact
title, which a question rarely provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.logging_setup import get_logger
from src.tools.http import Fetcher, encode, get_json, sanitise_query

SEARCH_URL = "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=1&srsearch={query}"
"""Title search. Fixed host, fixed parameters, one interpolated query."""

SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
"""The REST summary endpoint: the lead paragraph, plus the canonical URL."""

EXTRACT_URL = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles={title}"
"""The full article as plain text, for a deep lookup.

``explaintext`` rather than HTML, because what reaches a prompt should be text: a
model given markup spends attention on the markup, and a stray tag is one more
thing an injection could hide in.
"""

SECTIONS_URL = (
    "https://en.wikipedia.org/w/api.php?action=parse&prop=sections&format=json&page={title}"
)
"""The article's section headings, which are its table of contents.

Cheap, and more useful than it looks: the headings tell the model what the article
covers, so it can say "the article has a section on quantum annealing" instead of
guessing from the lead paragraph whether it does.
"""

MAX_QUERY_CHARACTERS = 120
"""Longest search term sent. A question is not a search term."""

MAX_EXTRACT_CHARACTERS = 900
"""How much of the article a shallow lookup keeps.

The lead section of a long article is several paragraphs, and all of it would
crowd out the retrieved passages and the computed numbers in the prompt. The
first part of the lead is where an encyclopedia puts the definition.
"""

MAX_DEEP_CHARACTERS = 3200
"""How much a deep lookup keeps.

Enough for the lead plus the first substantive sections, which is where an
encyclopedia article stops defining and starts explaining. Still a cap: an article
on quantum mechanics runs to tens of thousands of characters and sending all of it
would push the verified numbers out of the model's attention, which is the one
thing this project cannot afford.
"""

MAX_SECTIONS = 12
"""Headings to keep. Enough to show the shape of an article, not its index."""

Depth = Literal["summary", "deep"]
"""How much of an article to fetch. See :func:`look_up`."""

LOG = get_logger("tools.wiki")


def clean_query(query: str) -> str:
    """Reduce a question to something worth searching for.

    Thin wrapper over :func:`src.tools.http.sanitise_query`, which is where the
    rule and the reasoning live: the term is chosen by a model, and the URL it
    lands in has structure that punctuation could otherwise reach. The name stays
    here because the limit is this service's, not a shared one.

    Args:
        query: What the model asked to look up.

    Returns:
        Letters, digits, spaces, hyphens and full stops only, collapsed and cut
        to :data:`MAX_QUERY_CHARACTERS`.

    Examples:
        >>> clean_query("Ising model&action=delete")
        'Ising model action delete'
        >>> clean_query("   ")
        ''
    """
    return sanitise_query(query, limit=MAX_QUERY_CHARACTERS)


@dataclass(frozen=True, slots=True)
class WikiArticle:
    """One encyclopedia article, reduced to what an answer can use.

    Attributes:
        title: The article's title, as Wikipedia gives it.
        extract: The article text kept, truncated to the depth's limit.
        url: The canonical page address, or ``""`` if the API omitted it.
        sections: The article's section headings, on a deep lookup only.
        depth: How much was fetched. See :data:`Depth`.
    """

    title: str
    extract: str
    url: str = ""
    sections: tuple[str, ...] = ()
    depth: Depth = "summary"

    @property
    def citation(self) -> str:
        """The article, attributed the way an encyclopedia should be."""
        suffix = f" — {self.url}" if self.url else ""
        return f"Wikipedia, “{self.title}”{suffix}"


@dataclass(frozen=True, slots=True)
class WikiLookup:
    """What one lookup produced, including the ways it produced nothing.

    Attributes:
        query: The cleaned search term actually sent.
        article: What came back, or ``None``.
        detail: Why there is nothing, when there is nothing.
    """

    query: str
    article: WikiArticle | None = None
    detail: str = ""

    @property
    def found(self) -> bool:
        """Whether an article came back."""
        return self.article is not None

    def context(self) -> str:
        """The article as material for a prompt.

        Returns:
            The title, the extract and a standing reminder of what it is worth.
            Framed as background rather than as evidence, because a model given
            an encyclopedia and a verified computation will otherwise weigh them
            the same.
        """
        if self.article is None:
            return ""
        parts = [
            "Background from Wikipedia (general reference, NOT verified by this "
            f"application and NOT part of its reviewed corpus):\n{self.article.title}: "
            f"{self.article.extract}"
        ]
        if self.article.sections:
            parts.append("Sections of that article: " + "; ".join(self.article.sections))
        return "\n".join(parts)

    def explain(self) -> str:
        """One line for the justification block.

        Returns:
            What was searched for and what came of it.

        Examples:
            >>> WikiLookup("ising model", detail="nothing matched").explain()
            'wikipedia "ising model": nothing matched'
        """
        if self.article is not None:
            depth = "" if self.article.depth == "summary" else " (deep)"
            return f'wikipedia{depth} "{self.query}": {self.article.title}'
        return f'wikipedia "{self.query}": {self.detail or "nothing found"}'


def _article_of(payload: Any) -> WikiArticle | None:
    """Read a summary response into an article.

    Args:
        payload: Whatever the endpoint returned.

    Returns:
        The article, or ``None`` if the response carried no usable extract --
        which is what a disambiguation page or a redirect loop looks like.
    """
    if not isinstance(payload, dict):
        return None
    title = str(payload.get("title", "")).strip()
    extract = str(payload.get("extract", "")).strip()
    if not title or not extract:
        return None
    pages = payload.get("content_urls")
    desktop = pages.get("desktop") if isinstance(pages, dict) else None
    url = desktop.get("page", "") if isinstance(desktop, dict) else ""
    return WikiArticle(
        title=title,
        extract=extract[:MAX_EXTRACT_CHARACTERS].strip(),
        url=str(url),
    )


def _best_title(payload: Any) -> str:
    """Read the first search hit's title out of a search response.

    Args:
        payload: Whatever the search endpoint returned.

    Returns:
        The title, or ``""`` when the response had no hits or an unexpected
        shape. A shape check rather than a chain of indexing, because this is a
        third party's JSON and it is allowed to change.
    """
    if not isinstance(payload, dict):
        return ""
    query = payload.get("query")
    hits = query.get("search") if isinstance(query, dict) else None
    if not isinstance(hits, list) or not hits:
        return ""
    first = hits[0]
    return str(first.get("title", "")).strip() if isinstance(first, dict) else ""


def _full_extract(payload: Any) -> str:
    """Read the plain-text extract out of an ``action=query`` response.

    The pages arrive in a dictionary keyed by page id, and the id is not something
    the caller knows, so the single value is taken rather than looked up. A
    negative id means "no such page", and that page carries no extract, which the
    emptiness check below catches without special-casing it.

    Args:
        payload: Whatever the endpoint returned.

    Returns:
        The article text, or ``""``.
    """
    if not isinstance(payload, dict):
        return ""
    query = payload.get("query")
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, dict):
        return ""
    for page in pages.values():
        if isinstance(page, dict):
            extract = str(page.get("extract", "")).strip()
            if extract:
                return extract
    return ""


def _sections_of(payload: Any) -> tuple[str, ...]:
    """Read the section headings out of an ``action=parse`` response.

    Args:
        payload: Whatever the endpoint returned.

    Returns:
        Up to :data:`MAX_SECTIONS` headings, in document order. Empty when the
        response had none, which is normal for a short article.
    """
    if not isinstance(payload, dict):
        return ()
    parse = payload.get("parse")
    sections = parse.get("sections") if isinstance(parse, dict) else None
    if not isinstance(sections, list):
        return ()
    headings = [
        str(section.get("line", "")).strip()
        for section in sections
        if isinstance(section, dict) and str(section.get("line", "")).strip()
    ]
    return tuple(headings[:MAX_SECTIONS])


def look_up(query: str, *, depth: Depth = "summary", fetch: Fetcher | None = None) -> WikiLookup:
    """Look one thing up in Wikipedia.

    Never raises. Every failure -- an empty query, no search hits, a page with no
    lead section, a network that is not there -- comes back as a
    :class:`WikiLookup` carrying the reason, because a tool that raises turns a
    verified answer into a traceback.

    Args:
        query: What to look up.
        depth: ``"summary"`` fetches the lead paragraph in two requests, which is
            enough to define a term. ``"deep"`` costs two further requests and
            returns the article text up to :data:`MAX_DEEP_CHARACTERS` plus its
            section headings -- worth it when the question is about the subject
            rather than about a word in it.
        fetch: How to fetch, injected by tests. Defaults to
            :func:`src.tools.http.get_json`.

    Returns:
        The lookup, found or not. A deep lookup whose extra requests fail
        degrades to the summary it already has rather than to nothing: a shorter
        article is a worse answer, and no article is a refusal.
    """
    get = get_json if fetch is None else fetch
    cleaned = clean_query(query)
    if not cleaned:
        return WikiLookup(query=cleaned, detail="the search term was empty once cleaned")

    title = _best_title(get(SEARCH_URL.format(query=encode(cleaned))))
    if not title:
        return WikiLookup(query=cleaned, detail="no article matched")

    encoded = encode(title.replace(" ", "_"))
    article = _article_of(get(SUMMARY_URL.format(title=encoded)))
    if article is None:
        return WikiLookup(query=cleaned, detail=f'"{title}" has no summary to quote')

    if depth == "deep":
        extract = _full_extract(get(EXTRACT_URL.format(title=encoded)))
        article = WikiArticle(
            title=article.title,
            extract=(extract or article.extract)[:MAX_DEEP_CHARACTERS].strip(),
            url=article.url,
            sections=_sections_of(get(SECTIONS_URL.format(title=encoded))),
            depth="deep",
        )

    LOG.info(
        "wikipedia_lookup",
        extra={"query": cleaned, "title": article.title, "depth": article.depth},
    )
    return WikiLookup(query=cleaned, article=article)
