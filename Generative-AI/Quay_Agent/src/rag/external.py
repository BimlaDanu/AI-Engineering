"""Reaching outside the corpus when the corpus does not cover the question.

The knowledge base is finite and the questions are not. When a search comes back
with nothing, there are three honest responses: say so, guess, or go and look. This
module is the third one -- a bounded search of an external index whose results are
screened, written down, and cited as what they are.

What a promoted note is
-----------------------

A note that arrived from outside, passed the admission screen, and was written to
:attr:`src.settings.Settings.promoted_path`. It is indexed alongside the
hand-written corpus and carries ``provenance: promoted`` in its frontmatter, so
every passage drawn from one can be labelled in an answer. A reader is owed the
difference between a note someone wrote and checked, and a paragraph an API
returned twenty seconds ago.

The directory is top-level and generated rather than committed, which is the
project's existing rule for anything a machine produces: whether a file was
authored or downloaded is answered by where it lives, not by opening it.

Three rules, and each one is load-bearing
-----------------------------------------

**Nothing is written until it has been screened.** Text fetched from an external
index is untrusted input that is about to be embedded, retrieved, and placed in a
prompt beside the instruction to trust it -- which is the whole shape of an
indirect prompt injection. A curated note that trips the screen is neutralised and
kept, because someone wrote it and it is probably a false positive. A fetched one
is *discarded*, because nobody has read it and there is nothing to lose. The
asymmetry is deliberate: the cheap response to an untrusted document behaving
oddly is not to have it.

**The body is the source's own words.** Nothing here summarises or paraphrases.
A retrieval corpus whose passages were rewritten by a language model is a corpus
of plausible sentences with real citations attached, and it is worse than no
corpus at all because the citations make it look checkable.

**The fetch is bounded and off the fast path.** It runs only when local retrieval
found nothing -- so a question the corpus already answers never waits for it -- and
it runs under a deadline. A question that would otherwise be refused is worth a
couple of seconds; a question that is already answered is not worth any.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.logging_setup import get_logger
from src.rag.ingest import PROMOTED, PROVENANCE_KEY, shelf_names
from src.security import screen
from src.settings import Settings, get_settings_or_none

_logger = get_logger("rag.external")

MAX_ABSTRACT_WORDS = 400
"""Longest abstract that will be accepted as a note.

Anything longer is a parsing accident rather than an abstract, and one runaway
document would take more of a retrieved context window than the four passages it
was competing with.
"""

MIN_ABSTRACT_WORDS = 30
"""Shortest text worth keeping.

Below this the record is a stub -- a title, a placeholder, a withdrawal notice --
and indexing it adds a passage that can match a query and then say nothing.
"""

DEFAULT_DEADLINE_S = 6.0
"""How long a live fetch may take before the caller gives up on it.

Chosen against what it is competing with. The alternative to fetching is telling
a person the corpus does not cover their question, which is instant; six seconds
is roughly the point past which they would rather have had the refusal. Nothing
here retries, for the same reason: a second attempt spends the budget twice to
answer the same way.
"""

DEFAULT_RESULTS = 4
"""How many external records one fetch keeps.

The same number the retriever returns, because these are candidates for exactly
those slots. Fetching more would cost time to produce passages that lose to the
ones already in hand.
"""

SLUG_TRIM = re.compile(r"[^a-z0-9]+")
"""Everything that is not a slug character, collapsed to a single hyphen."""

MAX_SLUG_LENGTH = 80
"""Longest filename stem a promoted note may have.

Filenames have to be unique across the whole library and are the basis of every
chunk id, so they are trimmed rather than left to a title that runs to a
paragraph.
"""


SUBJECT_TERMS: frozenset[str] = frozenset(
    {
        # the model and its neighbours
        "ising",
        "transverse field",
        "transverse-field",
        "spin chain",
        "spin glass",
        "spin model",
        "spin system",
        "spins",
        "hubbard",
        "heisenberg",
        "lattice",
        "hamiltonian",
        "ground state",
        "critical point",
        "quantum phase",
        "many-body",
        "quantum many-body",
        "magnetization",
        "magnetisation",
        "entanglement",
        "free fermion",
        "jordan-wigner",
        "fermionic",
        "spin-1/2",
        "correlated electron",
        "gutzwiller",
        # the methods
        "variational",
        "vqe",
        "qaoa",
        "annealing",
        "annealer",
        "adiabatic",
        "trotter",
        "quantum algorithm",
        "quantum simulation",
        "quantum simulator",
        "monte carlo",
        "tensor network",
        "dmrg",
        # the machines
        "qubit",
        "quantum computer",
        "quantum computing",
        "quantum circuit",
        "quantum hardware",
        "quantum device",
        "quantum processor",
        "quantum state",
        "quantum advantage",
        "nisq",
        "superconducting",
        "trapped ion",
        "rydberg",
        "error mitigation",
        "error correction",
        "quantum error",
        # the applications shelf
        "combinatorial optimization",
        "combinatorial optimisation",
        "optimization problem",
        "optimisation problem",
    }
)
"""Words a fetched abstract must use at least one of to count as on-subject.

**Why this exists at all.** An external index is searched with a query and returns
what its own ranking thinks best, which is not the same as what the query asked
for. A search for the critical point of a spin chain came back, in this project's
own logs, with four large collaboration papers -- gravitational-wave transients,
the ATLAS detector's expected performance, and a rare B-meson decay. Nothing in
:func:`admit` refused them, because length, shelf, slug and injection screening
have nothing to say about *subject*, and so they were written into the library.

**And why that mattered more than an untidy directory.** A promoted note's topic
tags are copied from the query that found it, not read out of the paper -- see
:attr:`Candidate.topics`, where that is a deliberate choice, because the tags are
what the retriever filters on. So an off-topic paper arrives tagged as though it
were on topic, becomes retrievable under those tags, and can be cited in an answer
about a lattice of magnets. Worse, retrieval accuracy is *scored* against those
same declared tags, so a mis-fetched note counts as a correct retrieval and the
measurement flatters itself.

**How the list is calibrated, and how that is kept honest.** Wide enough for every
note the four shelves cover, narrow enough to exclude a particle-physics abstract
that merely says "quantum". Generic words are deliberately absent: "quantum" alone
admits quantum chromodynamics, and "detector" or "measurement" admit most of
physics. As checked in the corpus test, every one of the committed
corpus's notes uses at least one of these and each of those four off-topic
abstracts uses none. The corpus test is the important half: it fails if this list
is ever tightened past what the project's own curated notes would survive, which
is the failure mode a relevance screen actually has.
"""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One external record, before anything has been decided about it.

    Attributes:
        title: The work's own title.
        body: Its abstract, verbatim.
        source: A citation line -- authors, identifier, date.
        identifier: The external index's own identifier for the record.
        shelf: Which knowledge base it would join.
        topics: Hyphenated slugs to tag it with, taken from the query that found
            it rather than guessed per record, because they are what the
            retriever filters on.
    """

    title: str
    body: str
    source: str
    identifier: str
    shelf: str
    topics: tuple[str, ...]

    @property
    def slug(self) -> str:
        """The filename stem this record would be written under."""
        return SLUG_TRIM.sub("-", self.title.lower()).strip("-")[:MAX_SLUG_LENGTH].strip("-")


@dataclass(frozen=True, slots=True)
class Admission:
    """Whether one candidate may be written down, and why.

    Attributes:
        admitted: Whether it passed.
        reason: One sentence, in language a person can act on.
        categories: The screening categories that fired, empty when it passed
            cleanly. Kept so a rejection can be reviewed rather than only
            counted -- a corpus that silently drops a third of what it fetches is
            a corpus with a bug nobody will find.
    """

    admitted: bool
    reason: str
    categories: tuple[str, ...] = ()


def on_subject(title: str, body: str) -> bool:
    """Say whether a record's own words place it inside this project's subject.

    Public rather than private because it is the one screening rule with a
    calibration that has to be re-checked whenever the corpus grows, and the test
    that re-checks it walks the committed corpus and calls this directly.

    Args:
        title: The work's own title.
        body: Its abstract.

    Returns:
        Whether at least one of :data:`SUBJECT_TERMS` appears. Substring matching
        rather than word matching, so "spins", "spin chains" and "Ising-like" all
        count -- a relevance gate that misses a plural is a gate that refuses good
        notes, which is the expensive direction to be wrong in.

    Examples:
        >>> on_subject("Critical scaling of the transverse-field Ising chain", "")
        True
        >>> on_subject("Expected Performance of the ATLAS Experiment", "trigger")
        False
    """
    haystack = f"{title}\n{body}".lower()
    return any(term in haystack for term in SUBJECT_TERMS)


def admit(candidate: Candidate) -> Admission:
    """Decide whether an external record may become a note.

    Five checks, cheapest first. Length bounds catch parsing accidents. The shelf
    check catches a caller asking for a knowledge base that does not exist. The
    subject check catches a record the index returned that is not about this
    project's subject at all -- see :data:`SUBJECT_TERMS` for what it cost to
    leave that one out. The injection screen catches text written to make a model
    do something, and unlike a curated note, a candidate that trips it is refused
    outright rather than neutralised and kept.

    Args:
        candidate: The record to judge.

    Returns:
        The decision, always. Nothing raises: a bad candidate is an ordinary
        outcome of searching an index nobody controls.
    """
    words = len(candidate.body.split())
    if words < MIN_ABSTRACT_WORDS:
        return Admission(False, f"only {words} words; too short to answer anything")
    if words > MAX_ABSTRACT_WORDS:
        return Admission(False, f"{words} words; longer than an abstract should be")
    if candidate.shelf not in shelf_names():
        return Admission(False, f"no knowledge base named {candidate.shelf!r}")
    if not candidate.slug:
        return Admission(False, "the title reduces to an empty filename")

    # Read from the record's own words, never from the tags it is about to be
    # given: those come from the query rather than from the paper, so screening a
    # rendered note against its own frontmatter would ask the claim to vouch for
    # itself. That circularity is not hypothetical -- it is what let four
    # off-topic abstracts through while carrying this project's topic tags.
    if not on_subject(candidate.title, candidate.body):
        return Admission(
            False,
            "the abstract never mentions this project's subject, so the index "
            "returned something the query did not ask for",
        )

    signals = screen(f"{candidate.title}\n{candidate.body}").signals
    if signals:
        categories = tuple(dict.fromkeys(signal.category for signal in signals))
        _logger.warning(
            "external_candidate_rejected",
            extra={
                "identifier": candidate.identifier,
                "categories": list(categories),
                "detail": "injection pattern in fetched text; nothing was written",
            },
        )
        return Admission(
            False,
            "the text carries instruction-like patterns and was not kept",
            categories,
        )
    return Admission(True, "clean")


def render_note(candidate: Candidate, provenance: str = PROMOTED) -> str:
    """Write a candidate out in the corpus's own note format.

    The body is the source's abstract and a citation line. The provenance key is
    what separates this file from a hand-written one everywhere downstream, and
    the closing sentence says the same thing in words for anyone reading the file
    rather than querying its metadata.

    Args:
        candidate: An admitted record.
        provenance: :data:`~src.rag.ingest.PROMOTED` for a note fetched during a
            run, or :data:`~src.rag.ingest.CURATED` for one being written into
            the committed corpus by a batch that a person chose the queries for
            and will commit the results of.

    Returns:
        The complete file contents, frontmatter included.
    """
    title = candidate.title.replace('"', "'")
    source = candidate.source.replace('"', "'")
    standing = (
        "Nobody has reviewed it."
        if provenance == PROMOTED
        else "It was fetched deliberately and committed alongside the code."
    )
    return (
        "---\n"
        f'title: "{title}"\n'
        f'source: "{source}"\n'
        f"arxiv: {candidate.identifier}\n"
        f"topics: [{', '.join(candidate.topics)}]\n"
        f"{PROVENANCE_KEY}: {provenance}\n"
        "---\n\n"
        f"# {candidate.title}\n\n"
        "## Abstract\n\n"
        f"{candidate.body}\n\n"
        "## Citation\n\n"
        f"{candidate.source}. Fetched from an external index and reproduced "
        f"without alteration. {standing}\n"
    )


def promoted_root(settings: Settings | None = None) -> Path | None:
    """Where promoted notes are written, or ``None`` if promotion is switched off.

    Args:
        settings: Configuration to read. Defaults to the process settings.

    Returns:
        The directory, or ``None`` when configuration disables external material
        entirely -- the setting to use when only reviewed text may be indexed, and
        also what an unreadable configuration gives. Reached from a graph node that
        promotes a fetched abstract, so raising here on a missing credential stopped
        an offline campaign at ``retrieve``; a path is a preference, and a campaign
        with no key was never going to fetch anything to promote anyway.
    """
    resolved = settings if settings is not None else get_settings_or_none()
    if resolved is None:
        return None
    return Path(resolved.promoted_path) if resolved.promoted_path else None


def promote(
    candidate: Candidate,
    settings: Settings | None = None,
    root: Path | None = None,
) -> Path | None:
    """Screen a candidate and, if it passes, write it where ingestion will find it.

    The only way a note enters the promoted library. Screening happens here rather
    than at the call site so that there is no path to disk that skips it.

    Args:
        candidate: The record to promote.
        settings: Configuration to read.
        root: Directory to write into, overriding configuration. Supplied by
            tests so a promotion can be exercised without touching the real one.

    Returns:
        The file written, or ``None`` if the candidate was refused, promotion is
        switched off, or an identical note already exists. An existing note is
        not overwritten: the copy on disk may have been edited by hand, and a
        refetch is not a reason to discard that.
    """
    base = root if root is not None else promoted_root(settings)
    if base is None:
        return None

    verdict = admit(candidate)
    if not verdict.admitted:
        _logger.info(
            "external_candidate_refused",
            extra={"identifier": candidate.identifier, "detail": verdict.reason},
        )
        return None

    shelf = base / candidate.shelf
    shelf.mkdir(parents=True, exist_ok=True)
    destination = shelf / f"{candidate.slug}.md"
    if destination.exists():
        return None

    destination.write_text(render_note(candidate), encoding="utf-8")
    _logger.info(
        "external_candidate_promoted",
        extra={"identifier": candidate.identifier, "shelf": candidate.shelf},
    )
    return destination


def _clean(text: str) -> str:
    """Collapse the line breaks an abstract arrives with, changing no words.

    Args:
        text: Raw text from an external record.

    Returns:
        The same words on one line.
    """
    return " ".join(text.split())


def as_candidate(paper: Any, shelf: str, topics: tuple[str, ...]) -> Candidate:
    """Read one external record into the shape this module works in.

    Public because the batch that builds the committed corpus reads the same
    records through the same client, and two readings of one record format is
    two places for a field to be misread.

    Args:
        paper: A result object from the external client.
        shelf: The knowledge base the query was aimed at.
        topics: Tags to attach.

    Returns:
        The candidate. Nothing is validated here -- that is :func:`admit`'s job,
        and keeping the two apart is what lets a rejected candidate still be
        logged with its identifier.
    """
    authors = ", ".join(author.name for author in paper.authors[:6])
    if len(paper.authors) > 6:
        authors += " et al."
    published = paper.published.date().isoformat() if paper.published else "unknown"
    identifier = paper.get_short_id()
    return Candidate(
        title=_clean(paper.title),
        body=_clean(paper.summary),
        source=f"{authors}, arXiv:{identifier} ({published})",
        identifier=identifier,
        shelf=shelf,
        topics=topics,
    )


def fetch(
    query: str,
    shelf: str,
    topics: tuple[str, ...] = (),
    limit: int = DEFAULT_RESULTS,
    deadline_s: float = DEFAULT_DEADLINE_S,
) -> tuple[Candidate, ...]:
    """Search an external index, under a deadline, and return what came back.

    Returns candidates rather than notes: nothing has been screened or written at
    this point, and separating the network call from the admission decision is
    what makes the admission decision testable without a network.

    Args:
        query: What to search for.
        shelf: Which knowledge base the results would join.
        topics: Tags to attach to whatever is found.
        limit: How many records to keep.
        deadline_s: Wall-clock budget. Records already read are kept when it
            expires; a partial result is worth more than nothing, and this is a
            supplement to an answer rather than the answer.

    Returns:
        The candidates found, possibly empty. Never raises -- an unreachable
        index, a malformed record and an expired deadline all mean the same
        thing to the caller, which is that there is nothing extra to offer.
    """
    started = time.monotonic()
    found: list[Candidate] = []
    try:
        import arxiv

        client = arxiv.Client(page_size=limit, delay_seconds=0.0, num_retries=1)
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for paper in client.results(search):
            found.append(as_candidate(paper, shelf, topics))
            if len(found) >= limit or time.monotonic() - started > deadline_s:
                break
    except Exception as error:
        _logger.warning(
            "external_fetch_failed",
            extra={
                "detail": type(error).__name__,
                "elapsed_s": round(time.monotonic() - started, 2),
            },
        )
        return tuple(found)

    _logger.info(
        "external_fetch",
        extra={
            "found": len(found),
            "shelf": shelf,
            "elapsed_s": round(time.monotonic() - started, 2),
        },
    )
    return tuple(found)


def fetch_and_promote(
    query: str,
    shelf: str,
    topics: tuple[str, ...] = (),
    limit: int = DEFAULT_RESULTS,
    deadline_s: float = DEFAULT_DEADLINE_S,
    settings: Settings | None = None,
    root: Path | None = None,
) -> tuple[Path, ...]:
    """Fetch, screen, and write whatever survives.

    The whole external route in one call, for a caller that wants the corpus
    extended and does not want to hold the intermediate state.

    Args:
        query: What to search for.
        shelf: Which knowledge base the results join.
        topics: Tags to attach.
        limit: How many records to consider.
        deadline_s: Wall-clock budget for the network half.
        settings: Configuration to read.
        root: Directory to write into, overriding configuration.

    Returns:
        The files written, in the order they were promoted. Empty is an ordinary
        outcome: the index may hold nothing, or everything it holds may have been
        refused.
    """
    written = [
        path
        for candidate in fetch(query, shelf, topics, limit, deadline_s)
        if (path := promote(candidate, settings, root)) is not None
    ]
    return tuple(written)
