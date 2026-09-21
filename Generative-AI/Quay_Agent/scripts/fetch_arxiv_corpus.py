"""Fetch abstracts from arXiv into the retrieval corpus.

The corpus that ships with this project is hand-written: nineteen notes across
three shelves, each one a paragraph someone actually composed. That is good for
precision and useless as evidence that the project can consume an **external
source**, which the brief asks for explicitly. This script closes that gap by
pulling real abstracts from the arXiv API and writing them as corpus notes in
the same frontmatter format ``src/rag/ingest.py`` already validates.

Two rules, and they are the whole design
----------------------------------------
1. **The body is the abstract, verbatim.** Nothing here summarises, paraphrases
   or "improves" what the authors wrote. A retrieval corpus whose passages were
   rewritten by a language model is a corpus of plausible sentences with real
   citations attached, which is worse than no corpus: the citation makes the
   invention checkable-looking without making it checked.
2. **Every note carries its arXiv identifier.** ``ingest.py`` refuses a note it
   cannot cite, and this is where that identifier comes from. An answer that
   quotes one of these can name the paper, and a reader can go and read it.

Existing files are never overwritten, so the hand-written notes are safe and a
second run is cheap. Run with ``make corpus``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import arxiv

from src.rag.external import admit, as_candidate, render_note
from src.rag.ingest import CURATED

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
"""The corpus root. One directory per shelf, exactly as ``ingest.py`` expects."""


@dataclass(frozen=True)
class Query:
    """One arXiv search, and where its results belong.

    Attributes:
        shelf: Corpus subdirectory. Must be a shelf ``ingest.py`` declares, or
            the note is refused at ingest rather than indexed.
        search: The arXiv query string.
        topics: Frontmatter topics attached to every note from this query. They
            are the retrieval filter, so they are set per query rather than
            guessed per paper.
        limit: How many results to keep.
    """

    shelf: str
    search: str
    topics: tuple[str, ...]
    limit: int = 6


QUERIES: tuple[Query, ...] = (
    Query(
        "physics-notes",
        'abs:"transverse field Ising" AND (abs:"exact solution" OR abs:"free fermion")',
        ("exact-solution", "free-fermions", "transverse-field-ising"),
    ),
    Query(
        "physics-notes",
        'abs:"transverse field Ising" AND (abs:"quantum critical" OR abs:"entanglement entropy")',
        ("criticality", "entanglement", "transverse-field-ising"),
    ),
    Query(
        "quantum-computing",
        'abs:"variational quantum eigensolver" AND abs:"Ising"',
        ("vqe", "variational", "transverse-field-ising"),
    ),
    Query(
        "quantum-computing",
        'abs:"quantum approximate optimization" AND (abs:"Ising" OR abs:"MaxCut")',
        ("qaoa", "variational", "optimisation"),
    ),
    Query(
        "quantum-computing",
        'abs:"barren plateau"',
        ("barren-plateaus", "variational", "trainability"),
    ),
    Query(
        "quantum-computing",
        'abs:"imaginary time evolution" AND abs:"quantum"',
        ("imaginary-time", "varqite", "variational"),
    ),
    Query(
        "quantum-computing",
        'abs:"shot noise" AND abs:"variational quantum"',
        ("shot-budget", "estimation", "variational"),
    ),
    Query(
        "applications",
        'abs:"Ising" AND (abs:"combinatorial optimization" OR abs:"Ising machine")',
        ("optimisation", "ising-machines", "applications"),
    ),
    # ── the classical competition ──────────────────────────────────────────────
    # The agent's whole job is deciding whether a quantum computer beats an
    # ordinary one, and until these were added it could describe the quantum side
    # in detail and had almost nothing to cite about the side it has to beat. A
    # one-dimensional chain is the classical method's best case, which is exactly
    # why an honest verdict has to be able to say so with a reference.
    Query(
        "physics-notes",
        'abs:"matrix product state" AND (abs:"ground state" OR abs:"spin chain")',
        ("tensor-networks", "classical-baseline", "matrix-product-states"),
    ),
    Query(
        "physics-notes",
        'abs:"density matrix renormalization group" AND abs:"one-dimensional"',
        ("dmrg", "classical-baseline", "tensor-networks"),
    ),
    Query(
        "physics-notes",
        'abs:"quantum Monte Carlo" AND abs:"transverse field Ising"',
        ("quantum-monte-carlo", "classical-baseline", "transverse-field-ising"),
    ),
    # ── when a quantum claim does not survive ──────────────────────────────────
    # Feasibility work is mostly the business of finding out that something does
    # not help. Without these the corpus could support an optimistic verdict far
    # better than a sober one, which is a bias built into the evidence itself.
    Query(
        "quantum-computing",
        'abs:"classical simulation" AND abs:"quantum advantage"',
        ("classical-simulation", "quantum-advantage", "verification"),
    ),
    Query(
        "quantum-computing",
        'abs:"quantum error mitigation" AND (abs:"overhead" OR abs:"sampling cost")',
        ("error-mitigation", "shot-budget", "hardware"),
    ),
    Query(
        "quantum-computing",
        'abs:"resource estimation" AND abs:"quantum algorithm"',
        ("resource-estimation", "quantum-advantage", "hardware"),
    ),
    # ── what the hardware actually costs ───────────────────────────────────────
    # The coherence and depth arithmetic the planner does needs somewhere to point
    # when it says a circuit is too deep to run.
    Query(
        "quantum-computing",
        'abs:"qubit routing" OR (abs:"transpilation" AND abs:"circuit depth")',
        ("transpilation", "connectivity", "hardware"),
    ),
    Query(
        "quantum-computing",
        'abs:"coherence time" AND abs:"two-qubit gate" AND abs:"superconducting"',
        ("coherence", "gate-error", "hardware"),
    ),
)
"""What to fetch, and onto which shelf.

Queries are narrow on purpose. A broad ``abs:"Ising"`` returns thousands of
papers with nothing in common, and a retrieval corpus of loosely related
abstracts retrieves loosely related answers.
"""


def fetch(query: Query, client: arxiv.Client) -> int:
    """Fetch one query and write what is new. Returns how many files were added."""
    shelf = CORPUS / query.shelf
    shelf.mkdir(parents=True, exist_ok=True)
    written = 0

    search = arxiv.Search(
        query=query.search,
        max_results=query.limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    for paper in client.results(search):
        candidate = as_candidate(paper, query.shelf, query.topics)
        # The same gate the live fetch uses. Notes written here are committed
        # rather than generated, but "somebody will read it eventually" is not a
        # control, and an abstract carrying an instruction should no more reach
        # the corpus than it should reach a prompt.
        verdict = admit(candidate)
        if not verdict.admitted:
            print(f"    refused ({verdict.reason}): {candidate.title[:60]}")
            continue
        destination = shelf / f"{candidate.slug}.md"
        if destination.exists():
            print(f"    already have: {destination.name}")
            continue
        destination.write_text(render_note(candidate, CURATED), encoding="utf-8")
        print(f"    wrote {query.shelf}/{destination.name}")
        written += 1
    return written


if __name__ == "__main__":
    client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=3)
    total = 0
    for query in QUERIES:
        print(f"{query.shelf}: {query.search}")
        try:
            total += fetch(query, client)
        # One failed query must not stop the rest: a network hiccup on note three
        # should not cost the five that would have followed it.
        except Exception as error:
            print(f"    FAILED: {error}")
        time.sleep(1.0)
    print(f"\n{total} new note(s). Run `make ingest` to re-embed the corpus.")
