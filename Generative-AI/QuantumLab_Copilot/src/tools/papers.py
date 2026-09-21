"""Looking papers up on arXiv: the one place this project reads the outside world.

Everything else the agent knows is either computed here or read from the small
committed corpus. This module is the exception, and it is treated as an exception
throughout:

* **The text is untrusted.** An abstract is a string written by somebody else and
  fetched over the network, so it is neutralised on the way in, exactly like a
  retrieved chunk, and truncated so a long one cannot crowd a prompt.
* **The results are never verified.** Nothing here is cross-checked against
  anything, so a paper is offered as *further reading* and labelled as such. A
  title that agrees with the computation is a coincidence until somebody reads
  the paper.
* **Failure is a result, not an exception.** No network, a rate limit, a
  malformed feed: all of them return an empty search carrying the reason. The
  agent is meant to answer without arXiv, so an outage must not be able to turn
  an answerable question into a traceback.

The search itself is domain-restricted and the query is sanitised, because the
query is written by a language model -- see :func:`clean_query` and
:data:`CATEGORY_FILTER`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from src import security
from src.logging_setup import get_logger
from src.tools.http import sanitise_query

LOG = get_logger("tools.papers")

MAX_PAPERS = 10
"""Most papers one lookup may return.

The ceiling is small on purpose. Ten titles is already more than an answer can
discuss, and every extra abstract is prompt budget spent on text nothing checked.
"""

MAX_QUERY_CHARACTERS = 120
"""Longest query sent to arXiv.

A model asked for keywords sometimes sends a paragraph, which matches nothing and
costs a round trip to find that out.
"""

MAX_SUMMARY_CHARACTERS = 600
"""Longest abstract kept, in characters.

Enough for the first few sentences, which is where an abstract says what the
paper does. The rest is method detail this agent cannot use and would only be
paying to include in a prompt.
"""

MAX_AUTHORS = 4
"""Authors listed before falling back to "et al."."""

CATEGORY_FILTER = "(cat:quant-ph OR cat:cond-mat.stat-mech OR cat:cond-mat.str-el)"
"""arXiv categories the search is confined to.

The three where the transverse-field Ising model actually lives. Without this the
tool is a general arXiv proxy: a model that mis-reads the question would happily
return economics preprints, and they would arrive looking exactly as
authoritative as a Sachdev paper.
"""

REQUEST_DELAY_S = 1.0
"""Seconds the client waits between pages, honouring arXiv's request that
automated clients pace themselves. One page is fetched per lookup, so this is
paid at most once."""

REQUEST_RETRIES = 1
"""Attempts per lookup. Retrying at length would make an interactive question sit
there; the tool is optional, so failing fast and saying so is the better trade."""


@dataclass(frozen=True, slots=True)
class Paper:
    """One arXiv result, reduced to the fields an answer can use.

    Attributes:
        identifier: The arXiv id, including its version suffix.
        title: The paper's title, neutralised.
        authors: Author names, truncated to :data:`MAX_AUTHORS`.
        published: Publication date as ``YYYY-MM-DD``, or empty if the feed
            omitted it.
        summary: The abstract, neutralised and truncated.
        url: Link to the abstract page.
    """

    identifier: str
    title: str
    authors: tuple[str, ...]
    published: str
    summary: str
    url: str

    @property
    def citation(self) -> str:
        """A one-line attribution, in the shape the corpus citations use.

        Returns:
            Authors, title and arXiv id, so a reader can find the paper without
            following the link.
        """
        who = ", ".join(self.authors) if self.authors else "unknown authors"
        year = self.published[:4] if self.published else "n.d."
        return f"{who} ({year}), {self.title} [arXiv:{self.identifier}]"


@dataclass(frozen=True, slots=True)
class PaperSearch:
    """What one lookup produced, including the case where it produced nothing.

    Attributes:
        query: The query as sent, after cleaning.
        papers: The results, best match first.
        detail: One sentence on what happened, always populated. This is what a
            user is shown when the list is empty, and "no papers found" is only
            useful next to why.
    """

    query: str
    papers: tuple[Paper, ...]
    detail: str

    @property
    def found(self) -> bool:
        """Whether the lookup returned anything."""
        return bool(self.papers)

    def context(self) -> str:
        """Render the results for a prompt, labelled as unverified.

        Returns:
            A block listing each paper with its abstract, headed by a sentence
            saying these were not checked. Empty when nothing was found, so a
            caller cannot accidentally send an encouraging empty frame.
        """
        if not self.papers:
            return ""
        body = "\n\n".join(
            f"[{position}] {paper.citation}\n{paper.summary}"
            for position, paper in enumerate(self.papers, start=1)
        )
        return (
            "The following arXiv results were fetched live. They are NOT part of "
            "the verified corpus and nothing has checked them. You may mention "
            "them as further reading. Do not take a number or a claim from them.\n"
            f"<<<ARXIV\n{body}\nARXIV>>>"
        )

    def explain(self) -> str:
        """Say what the lookup did, in one line.

        Returns:
            The query, the count and the reason, for the machinery layer and the
            log.
        """
        return f"arXiv {self.query!r}: {len(self.papers)} papers. {self.detail}"


Fetch = Callable[[str, int], Iterable[Any]]
"""Signature of the thing that actually talks to arXiv.

Takes the cleaned query and a limit, yields result objects. Injected so the tests
exercise the parsing, the truncation and the neutralisation without a network
call -- the network is the one part of this module that cannot be tested honestly
in a unit test, so it is the one part kept behind a seam.
"""


def clean_query(query: str) -> str:
    """Strip a model-written query down to something safe to send.

    arXiv's query language gives meaning to colons, parentheses, quotes and the
    words ``AND``/``OR``/``ANDNOT``. A query that keeps them can therefore change
    the shape of the whole expression -- including escaping the category filter
    this module wraps around it -- which is the same problem as SQL injection and
    is solved the same way: the untrusted part is not allowed to carry syntax.

    Args:
        query: What the model asked to search for.

    Returns:
        Words, digits and hyphens only, collapsed to single spaces and truncated
        to :data:`MAX_QUERY_CHARACTERS`. Empty if nothing survived.

    Examples:
        >>> clean_query('ising) OR cat:econ.GN AND all:"free money"')
        'ising OR cat econ.GN AND all free money'
        >>> clean_query("  transverse   field  ")
        'transverse field'
    """
    return sanitise_query(query, limit=MAX_QUERY_CHARACTERS)


def author_clause(author: str) -> str:
    """Build a clause that finds an author however their name is indexed.

    arXiv stores an author as a single string and different papers spell the same
    person differently: "Kadowaki, Tadashi", "Tadashi Kadowaki", "T. Kadowaki".
    Searching for one spelling finds one subset, which reads as "this person has
    written two papers" -- confidently, and wrongly. So a two-part name becomes
    three alternatives, and the surname alone is one of them, because a surname
    match with the given name absent is still the right person far more often than
    not.

    The quotes and the ``OR`` are added *here*, after :func:`clean_query` has
    removed any the caller supplied. That ordering is the whole safety argument:
    the syntax is ours and the words are theirs, and the two never swap places.

    Args:
        author: A name, in any order, cleaned or not.

    Returns:
        An arXiv clause, or ``""`` if nothing survived cleaning.

    Examples:
        >>> author_clause("Tadashi Kadowaki")
        '(au:"Kadowaki, Tadashi" OR au:"Tadashi Kadowaki" OR au:"Kadowaki")'
        >>> author_clause("Pfeuty")
        'au:"Pfeuty"'
        >>> author_clause("  ")
        ''
    """
    parts = clean_query(author).split()
    if not parts:
        return ""
    if len(parts) == 1:
        return f'au:"{parts[0]}"'
    *given, surname = parts
    given_names = " ".join(given)
    return f'(au:"{surname}, {given_names}" OR au:"{given_names} {surname}" OR au:"{surname}")'


def build_query(keywords: str = "", *, author: str = "", title: str = "") -> str:
    """Compose the arXiv expression from separately cleaned parts.

    Each part is searched in the field it belongs to -- keywords in the abstract,
    a title in the title, an author in the author list -- because arXiv ranks a
    field-scoped query far better than the same words thrown at every field.
    :data:`CATEGORY_FILTER` is ANDed on last and is not optional: this application
    answers about one model in one corner of physics, and a paper outside those
    categories is noise however well it matches.

    Args:
        keywords: Free text to match in the abstract.
        author: A name. See :func:`author_clause`.
        title: Words to match in the title.

    Returns:
        The expression to send, or ``""`` when every part was empty -- which the
        caller turns into a refusal rather than a search for the whole archive.

    Examples:
        >>> build_query("critical point", title="transverse field").split(" AND ")[:2]
        ['abs:(critical point)', 'ti:(transverse field)']
        >>> build_query(author="Pfeuty").startswith('au:"Pfeuty" AND (cat:')
        True
        >>> build_query("   ")
        ''
    """
    clauses: list[str] = []
    cleaned_keywords = clean_query(keywords)
    if cleaned_keywords:
        clauses.append(f"abs:({cleaned_keywords})")
    cleaned_title = clean_query(title)
    if cleaned_title:
        clauses.append(f"ti:({cleaned_title})")
    clause = author_clause(author)
    if clause:
        clauses.append(clause)
    if not clauses:
        return ""
    clauses.append(CATEGORY_FILTER)
    return " AND ".join(clauses)


def arxiv_results(expression: str, limit: int) -> list[Any]:
    """Fetch results from the live arXiv API.

    The default :data:`Fetch`. Imported lazily so that importing this module --
    which the whole test suite does -- does not pull in a network client.

    Args:
        expression: The built query expression. See :func:`build_query`.
        limit: Most results to return.

    Returns:
        The raw result objects, in relevance order.
    """
    import arxiv

    client = arxiv.Client(
        page_size=limit,
        delay_seconds=REQUEST_DELAY_S,
        num_retries=REQUEST_RETRIES,
    )
    search = arxiv.Search(
        query=expression,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(client.results(search))


def _text(value: object, limit: int) -> str:
    """Neutralise and truncate one field of third-party text."""
    cleaned = security.neutralise(str(value or "")).strip()
    cleaned = " ".join(cleaned.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _authors(names: Sequence[Any]) -> tuple[str, ...]:
    """Reduce an author list to a few neutralised names."""
    listed = [_text(getattr(name, "name", name), 80) for name in names[:MAX_AUTHORS]]
    kept = tuple(name for name in listed if name)
    if len(names) > MAX_AUTHORS and kept:
        return (*kept, "et al.")
    return kept


def paper_of(result: Any) -> Paper:
    """Reduce one arXiv result object to a :class:`Paper`.

    Reads every field defensively. The feed is somebody else's schema and this
    project pins a client, not a contract, so a missing or renamed attribute must
    degrade to an empty string rather than raise inside a tool call.

    Args:
        result: A result object from the arXiv client.

    Returns:
        The paper, with all text neutralised and truncated.
    """
    entry = str(getattr(result, "entry_id", "") or "")
    # Split on "/abs/" rather than on the last slash: a pre-2007 id carries its
    # archive as a path segment ("cond-mat/9804280v1"), and dropping it leaves a
    # number that resolves to nothing.
    _, _, tail = entry.partition("/abs/")
    identifier = tail or entry.rsplit("/", 1)[-1]
    published: Any = getattr(result, "published", None)
    date = published.date().isoformat() if hasattr(published, "date") else ""
    return Paper(
        identifier=_text(identifier, 40),
        title=_text(getattr(result, "title", ""), 200),
        authors=_authors(list(getattr(result, "authors", []) or [])),
        published=date,
        summary=_text(getattr(result, "summary", ""), MAX_SUMMARY_CHARACTERS),
        url=entry,
    )


def find_papers(
    query: str = "",
    limit: int = 3,
    *,
    author: str = "",
    title: str = "",
    fetch: Fetch | None = None,
) -> PaperSearch:
    """Look up papers on arXiv, by keyword, title, author, or any combination.

    Args:
        query: Keywords to search abstracts for. Cleaned before it is sent.
        limit: How many papers to return, clamped to :data:`MAX_PAPERS`.
        author: An author's name, matched in several spellings. See
            :func:`author_clause`.
        title: Words that should appear in the title.
        fetch: The thing that talks to arXiv. Defaults to the live client.

    Returns:
        The search, whatever happened. Never raises: an empty
        :class:`PaperSearch` carrying the reason is the failure mode, because the
        agent is expected to answer without this tool.

    Examples:
        A request with nothing searchable in it never reaches the network:

        >>> find_papers("   ").detail
        'the query was empty after cleaning, so nothing was searched'
    """
    expression = build_query(query, author=author, title=title)
    if not expression:
        return PaperSearch(
            query="",
            papers=(),
            detail="the query was empty after cleaning, so nothing was searched",
        )
    # What to show the user: the terms they would recognise, not the expression
    # with its category filter, which is machinery.
    asked = " ".join(part for part in (clean_query(query), clean_query(title), author) if part)
    wanted = max(1, min(limit, MAX_PAPERS))
    call = arxiv_results if fetch is None else fetch
    try:
        results = list(call(expression, wanted))
    except Exception as error:
        LOG.warning(
            "arxiv_lookup_failed",
            extra={"error_type": type(error).__name__, "query": asked},
        )
        return PaperSearch(
            query=asked,
            papers=(),
            detail=f"the arXiv lookup failed ({type(error).__name__}); answered without it",
        )
    papers = tuple(paper_of(result) for result in results[:wanted])
    kept = tuple(paper for paper in papers if paper.title)
    LOG.info("arxiv_lookup", extra={"query": asked, "papers": len(kept)})
    if not kept:
        return PaperSearch(
            query=asked,
            papers=(),
            detail=f"arXiv returned nothing for {asked!r} in {CATEGORY_FILTER}",
        )
    return PaperSearch(
        query=asked,
        papers=kept,
        detail=f"{len(kept)} results from arXiv, offered as further reading only",
    )
