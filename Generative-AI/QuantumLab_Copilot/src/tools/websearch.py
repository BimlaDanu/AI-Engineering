"""General web search, when it is configured -- and absent when it is not.

arXiv covers the literature and Wikipedia covers the vocabulary. Neither covers
the rest of the web: a lecture course, a hardware vendor's documentation, a blog
post that explains a derivation better than the paper did. This tool covers that,
and it is the least trustworthy thing in the project by a wide margin, which is
why it says so in three places -- here, in the tool description the model reads,
and in the label on every result that reaches the screen.

**No key, no tool.** Web search costs money, so it goes through whatever endpoint
the deployment configured, and there is no default. Unconfigured, the tool is not
offered to the model at all: it never appears in the schema list, so no question
can talk the agent into calling it. That is the same rule the arXiv toggle
follows, applied to configuration rather than to a checkbox.

The wire format is the widely-copied one -- ``POST`` a JSON body with ``query``,
receive ``{"results": [{"title", "url", "content"}]}`` -- so a Tavily-compatible
endpoint works unchanged and anything else needs a five-line adapter. Reading a
shape rather than importing a vendor's client is also what keeps this module
dependency-free.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src import security
from src.logging_setup import get_logger
from src.tools.http import post_json

ENDPOINT_VARIABLE = "WEB_SEARCH_URL"
"""Environment variable holding the search endpoint. No default."""

KEY_VARIABLE = "WEB_SEARCH_API_KEY"
"""Environment variable holding the search credential."""

MAX_RESULTS = 5
"""Most results to request. A prompt is not a search engine results page."""

MAX_QUERY_CHARACTERS = 200
"""Longest query sent. Long enough for a sentence, which search engines accept."""

MAX_SNIPPET_CHARACTERS = 400
"""How much of each result's text to keep."""

Search = Callable[[str, int], Any | None]
"""Something that runs a search and returns the parsed response, or ``None``."""

LOG = get_logger("tools.websearch")


@dataclass(frozen=True, slots=True)
class WebResult:
    """One search hit, reduced and neutralised.

    Attributes:
        title: The page's title.
        url: Where it is.
        snippet: The engine's extract, truncated.
    """

    title: str
    url: str
    snippet: str = ""

    @property
    def citation(self) -> str:
        """The hit, attributed -- and labelled for what it is worth."""
        return f"{self.title} — {self.url} (web, unverified)"


@dataclass(frozen=True, slots=True)
class WebSearch:
    """What one search produced, including the ways it produced nothing.

    Attributes:
        query: The cleaned query actually sent.
        results: What came back, in the engine's own order.
        detail: Why there is nothing, when there is nothing.
    """

    query: str
    results: tuple[WebResult, ...] = ()
    detail: str = ""

    @property
    def found(self) -> bool:
        """Whether anything came back."""
        return bool(self.results)

    def context(self) -> str:
        """The results as material for a prompt.

        Returns:
            The hits, under a standing warning. Blunter than the arXiv and
            Wikipedia equivalents on purpose: an arbitrary page has had no review
            of any kind, and a model that treats it as equal to a computed number
            is the failure this project exists to prevent.
        """
        if not self.results:
            return ""
        lines = [
            "Web search results (arbitrary pages, NOT verified, NOT reviewed, and NOT "
            "evidence for any number -- use them only as pointers):"
        ]
        lines.extend(
            f"- {result.title} ({result.url}): {result.snippet}" for result in self.results
        )
        return "\n".join(lines)

    def explain(self) -> str:
        """One line for the justification block.

        Returns:
            What was searched for and how much came back.

        Examples:
            >>> WebSearch("ising chain", detail="not configured").explain()
            'web "ising chain": not configured'
        """
        if self.results:
            return f'web "{self.query}": {len(self.results)} results (unverified)'
        return f'web "{self.query}": {self.detail or "nothing found"}'


def clean_query(query: str) -> str:
    r"""Reduce a query to plain words.

    A general search engine has no query syntax worth defending against in the way
    arXiv does, but the query still ends up in a JSON body and a log line, so
    control characters and runs of whitespace go.

    Args:
        query: What the model asked to search for.

    Returns:
        Printable text, collapsed and truncated.

    Examples:
        >>> clean_query("  ising\tchain   dynamics ")
        'ising chain dynamics'
    """
    printable = "".join(character if character.isprintable() else " " for character in query)
    return " ".join(printable.split())[:MAX_QUERY_CHARACTERS].strip()


def configuration() -> tuple[str, str] | None:
    """Read the endpoint and credential from the environment.

    Returns:
        The pair, or ``None`` when either is missing. Both are required: an
        endpoint with no key fails at the far end, and a key with no endpoint has
        nowhere to go.
    """
    endpoint = os.environ.get(ENDPOINT_VARIABLE, "").strip()
    key = os.environ.get(KEY_VARIABLE, "").strip()
    return (endpoint, key) if endpoint and key else None


def is_configured() -> bool:
    """Whether web search can run at all.

    Returns:
        ``True`` when both variables are set. Read by
        :class:`src.tools.calling.Toolbox` to decide whether to describe the tool,
        so an unconfigured deployment does not advertise something that cannot
        work.
    """
    return configuration() is not None


def live_search(query: str, limit: int) -> Any | None:
    """Run one search against the configured endpoint.

    The default :data:`Search`. The credential goes in a header, never in the URL:
    a URL reaches log files, proxy logs and error messages.

    Args:
        query: The cleaned query.
        limit: Most results to ask for.

    Returns:
        The parsed response, or ``None`` if there is no configuration or the call
        failed.
    """
    settings = configuration()
    if settings is None:
        return None
    endpoint, key = settings
    return post_json(
        endpoint,
        {"query": query, "max_results": limit},
        headers={"Authorization": f"Bearer {key}"},
    )


def _results_of(payload: Any, limit: int) -> tuple[WebResult, ...]:
    """Read hits out of a response, ignoring anything malformed.

    Every field is neutralised on the way in. This is the most hostile text the
    application handles -- an arbitrary page, chosen by a search engine, in
    response to a query written by a model -- and it is about to be put in front
    of that same model.

    Args:
        payload: Whatever the endpoint returned.
        limit: Most results to keep.

    Returns:
        The hits, dropping any without both a title and a URL.
    """
    if not isinstance(payload, dict):
        return ()
    raw = payload.get("results")
    if not isinstance(raw, list):
        return ()
    hits: list[WebResult] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        title = _clean_field(item.get("title"), MAX_SNIPPET_CHARACTERS)
        url = _clean_field(item.get("url"), MAX_SNIPPET_CHARACTERS)
        if not title or not url:
            continue
        snippet = _clean_field(item.get("content"), MAX_SNIPPET_CHARACTERS)
        hits.append(WebResult(title=title, url=url, snippet=snippet))
    return tuple(hits)


def _clean_field(value: object, limit: int) -> str:
    """Neutralise and truncate one field of third-party text.

    Args:
        value: Whatever the endpoint put there.
        limit: Longest string to keep.

    Returns:
        The text, safe to place in a prompt.
    """
    text = " ".join(security.neutralise(str(value or "")).split())
    return text[:limit].strip()


def search_web(query: str, limit: int = 3, *, search: Search | None = None) -> WebSearch:
    """Search the web, if the deployment configured somewhere to search.

    Never raises. An unconfigured endpoint, an empty query, a failed request and a
    response of the wrong shape all come back as an empty :class:`WebSearch`
    carrying the reason.

    Args:
        query: What to search for.
        limit: Most results to keep, clamped to :data:`MAX_RESULTS`.
        search: How to search, injected by tests. Defaults to
            :func:`live_search`.

    Returns:
        The search, found or not.

    Examples:
        >>> search_web("   ").detail
        'the query was empty once cleaned'
    """
    cleaned = clean_query(query)
    if not cleaned:
        return WebSearch(query="", detail="the query was empty once cleaned")
    if search is None and not is_configured():
        return WebSearch(
            query=cleaned,
            detail=(
                f"web search is not configured ({ENDPOINT_VARIABLE} and {KEY_VARIABLE} are unset)"
            ),
        )
    run = live_search if search is None else search
    wanted = max(1, min(limit, MAX_RESULTS))
    try:
        payload = run(cleaned, wanted)
    except Exception as error:
        LOG.warning("web_search_failed", extra={"error_type": type(error).__name__})
        return WebSearch(
            query=cleaned,
            detail=f"the web search failed ({type(error).__name__}); answered without it",
        )
    results = _results_of(payload, wanted)
    LOG.info("web_search", extra={"query": cleaned, "results": len(results)})
    if not results:
        return WebSearch(query=cleaned, detail="the search returned nothing usable")
    return WebSearch(
        query=cleaned,
        results=results,
        detail=f"{len(results)} web results, unverified and offered as pointers only",
    )
