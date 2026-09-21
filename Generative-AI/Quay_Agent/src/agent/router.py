"""Deciding whether to search, and which knowledge base to search first.

Retrieval answers *what is relevant*. This module answers the two questions
before it: is this in scope at all, and which shelf is likely to hold the answer.
They fail differently, which is why they are kept apart -- a question outside the
subject should be refused without spending an embedding call, while one inside it
sent to the wrong shelf should still be answered a round later by widening.

Routing sits on the hot path, so the decision is deterministic first and
escalated only when genuinely close:

1. The scope gate is a set intersection over the question's words. Microseconds,
   no network, and it stops an out-of-scope question reaching the index.
2. The shelf choice is a word-overlap score against each shelf's vocabulary.
   When one shelf wins clearly, that is the answer and no model is called.
3. A model is consulted only when the scores are tied or all zero, the case where
   the deterministic answer would be a coin toss.

:attr:`Routing.decided_by` records which, so the cost of a campaign is read off
its trace.

The corpus is four separate literatures: the physics of the chain, the circuits
it runs on, how the methods compare, and what the model is applied to outside
physics. A search that mixes them will answer a question about circuit depth out
of a note about portfolio optimisation. The choice orders results rather than
filtering them, so a wrong guess costs position and not the answer.

The vocabularies are checked against the shelves that exist at import, so a
renamed shelf fails immediately instead of producing a filter that matches
nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agent.model_selection import ModelPool
from src.agent.prompts import SHELF_CHOICE
from src.logging_setup import get_logger
from src.rag.ingest import SHELVES, shelf_names
from src.security import Screening, screen

_logger = get_logger("agent.router")

DecidedBy = Literal["heuristic", "model"]
"""Who chose the shelf.

``"heuristic"`` is the ordinary case and costs nothing. ``"model"`` means the word
scores could not separate two shelves and one call was spent breaking the tie.
"""

WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
"""What counts as a word when a question is scored.

Hyphens are kept inside a token because the corpus's own vocabulary is hyphenated
-- ``free-fermion``, ``transverse-field``, ``error-mitigation`` -- and splitting
those would score a question about free fermions against every note containing
the word "free".
"""

DOMAIN_TERMS: frozenset[str] = frozenset(
    {
        # the model and its physics
        "ising",
        "spin",
        "spins",
        "chain",
        "lattice",
        "hamiltonian",
        "magnetisation",
        "magnetization",
        "ferromagnetic",
        "antiferromagnetic",
        "transverse",
        "longitudinal",
        "field",
        "coupling",
        "ground",
        "energy",
        "gap",
        "critical",
        "criticality",
        "phase",
        "transition",
        "exponent",
        "correlation",
        "entanglement",
        "fermion",
        "fermions",
        "free-fermion",
        "jordan-wigner",
        "diagonalisation",
        "diagonalization",
        "eigenvalue",
        "eigenstate",
        "sigma",
        "pauli",
        # the quantum-computing half
        "quantum",
        "qubit",
        "qubits",
        "circuit",
        "gate",
        "gates",
        "ansatz",
        "variational",
        "vqe",
        "qaoa",
        "hva",
        "annealing",
        "annealer",
        "trotter",
        "trotterised",
        "trotterized",
        "adiabatic",
        "simulation",
        "simulator",
        "shot",
        "shots",
        "measurement",
        "coherence",
        "decoherence",
        "fidelity",
        "transpilation",
        "transpile",
        "barren",
        "plateau",
        "mitigation",
        "advantage",
        "speedup",
        "supremacy",
        # what it is applied to
        "optimisation",
        "optimization",
        "combinatorial",
        "maxcut",
        "qubo",
        "portfolio",
        "scheduling",
        "routing",
        "np-hard",
        # choosing between the methods
        "varqite",
        "mclachlan",
        "imaginary",
        "mixer",
        "depth",
        "layer",
        "layers",
        "expressibility",
        "trainability",
        "molecule",
        "molecules",
        "molecular",
        "chemistry",
        "anneal",
        "embedding",
        "hardware",
        "device",
        "machine",
        "machines",
        "platform",
        "superconducting",
        "ion",
        "ions",
        "rydberg",
        # What a run costs, in the words this project uses for it. `shot`, `shots`
        # and `measurement` were admitted; the plural was not.
        #
        # Bare "cost", "budget" and "price" are deliberately **not** here, though
        # they were tried. They admit *how much does a flight to Berlin cost* and
        # *what is the price of bitcoin*, and a gate that lets those through has
        # stopped being a gate. Cost questions about this project reach scope the
        # other way: either they name something physical too, or they refer back to
        # an answer already given -- see :data:`BACK_REFERENCES`.
        "measurements",
        "wall-clock",
        # the classical competition
        "classical",
        "baseline",
        "monte",
        "carlo",
        "metropolis",
        "dmrg",
        "tensor",
        "matrix",
        "product",
        "state",
        "mps",
        # The vocabulary of a two-dimensional lattice. "lattice", "square" and
        # "triangular" were already admitted by "ising" or "spin" appearing beside
        # them in most phrasings, but the shortest and most natural follow-ups do
        # not carry either word -- *does frustration change the answer?* named
        # nothing the gate knew and was refused as off-topic, underneath a report
        # that had just used the word. A follow-up the application itself offers
        # must never be refused by the application's own gate.
        "lattices",
        "square",
        "triangular",
        "frustrated",
        "frustration",
        "plaquette",
        "plaquettes",
        "geometry",
        "bipartite",
        "antiferromagnet",
        "coordination",
        "neighbours",
        "neighbors",
    }
)
"""The scope gate: a question sharing none of these words is refused.

A gate rather than a ranking hint, and the distinction is the whole point. A
question about the weather does not deserve a vector search, a grading pass and a
polite refusal three seconds later; it deserves an immediate one. Everything the
project can actually answer contains at least one of these words, because the
project answers questions about one spin chain and the machines people try to
solve it on.

The list is deliberately generous. A false positive costs one search that finds
nothing; a false negative refuses a question the corpus covers, which is the more
expensive mistake and the harder one to notice.
"""

DOMAIN_PHRASES: frozenset[str] = frozenset(
    {
        "business problem",
        "cost function",
        "use case",
        "error rate",
        "run time",
        "imaginary time",
        "circuit depth",
        "cost operator",
        "quantum computer",
        "quantum computers",
        # The plain-English names for the classical baseline. "classical" is in
        # DOMAIN_TERMS and was thought to cover this; it does not, because a person
        # asking whether the quantum route is worth it does not reach for the word
        # "classical". They reach for the words on the screen in front of them --
        # and "an ordinary computer" is this project's own phrase for the baseline,
        # written in the chat panel, the figures and the report. So the question the
        # entire verdict turns on, asked in the application's own vocabulary, was
        # refused as off-topic; `src/evals/retrieval_suite.py` is what found it.
        "ordinary computer",
        "normal computer",
        "regular computer",
        "conventional computer",
        "classical computer",
    }
)
"""Multi-word entries of the scope gate, matched against the question as text.

Separate from :data:`DOMAIN_TERMS` because they are matched differently, and
because the difference is load-bearing. ``business`` on its own would open the
gate to every question about a company; ``business problem`` names the thing this
corpus actually holds notes on, which is the practice of writing an operational
problem as a spin model. A single-word gate cannot express that distinction, and
a gate that cannot express it has to choose between admitting too much and
refusing a subject the corpus covers.
"""

SHELF_TERMS: dict[str, frozenset[str]] = {
    "physics-notes": frozenset(
        {
            "exact",
            "closed-form",
            "analytic",
            "free-fermion",
            "fermion",
            "fermions",
            "jordan-wigner",
            "bogoliubov",
            "diagonalisation",
            "diagonalization",
            "lanczos",
            "eigenvalue",
            "eigenstate",
            "spectrum",
            "gap",
            "critical",
            "criticality",
            "exponent",
            "universality",
            "correlation",
            "entanglement",
            "entropy",
            "transfer",
            "duality",
            "phase",
            "transition",
            "magnetisation",
            "magnetization",
            "ground",
            "hamiltonian",
            "dmrg",
            "tensor",
            "mps",
            "monte",
            "carlo",
            "metropolis",
            # The words a *derivation* question uses. Added with the notes that
            # answer them: "detail the mathematical detail of the quantum-to-
            # classical mapping" favoured no shelf at all and was searched against
            # the whole library, which is the right fallback and a worse answer than
            # the shelf that holds three notes on exactly that.
            "mapping",
            "quantum-to-classical",
            "suzuki",
            "trotter",
            "derivative",
            "derivatives",
            "susceptibility",
            "hellmann-feynman",
            "parity",
            "dispersion",
            "momentum",
            "momenta",
        }
    ),
    "quantum-computing": frozenset(
        {
            "qubit",
            "qubits",
            "circuit",
            "gate",
            "gates",
            "cnot",
            "ansatz",
            "variational",
            "vqe",
            "qaoa",
            "hva",
            "parameterised",
            "parameterized",
            "optimiser",
            "optimizer",
            "gradient",
            "barren",
            "plateau",
            "trotter",
            "trotterised",
            "trotterized",
            "adiabatic",
            "annealing",
            "annealer",
            "hardware",
            "device",
            "superconducting",
            "trapped",
            "ion",
            "coherence",
            "decoherence",
            "fidelity",
            "noise",
            "error",
            "mitigation",
            "correction",
            "transpilation",
            "transpile",
            "routing",
            "layout",
            "swap",
            "shot",
            "shots",
            "sampling",
            "estimator",
            "simulator",
            "advantage",
            "speedup",
            "supremacy",
            "resource",
        }
    ),
    "method-comparison": frozenset(
        {
            "compare",
            "comparison",
            "versus",
            "vs",
            "choose",
            "choice",
            "instead",
            "difference",
            "differ",
            "tradeoff",
            "trade-off",
            "alternative",
            "mixer",
            "cost-operator",
            "objective",
            "encoding",
            "encode",
            "embedding",
            "imaginary",
            "varqite",
            "mclachlan",
            "metric",
            "depth",
            "layer",
            "layers",
            "expressibility",
            "trainability",
            "molecule",
            "molecular",
            "chemistry",
            "annealing",
            "anneal",
            "platform",
            "vendor",
        }
    ),
    "applications": frozenset(
        {
            "optimisation",
            "optimization",
            "combinatorial",
            "maxcut",
            "qubo",
            "portfolio",
            "finance",
            "logistics",
            "scheduling",
            "np-hard",
            "business",
            "industry",
            "industrial",
            "commercial",
            "pilot",
            "vendor",
            "machine",
            "learning",
            "neural",
            "boltzmann",
            "annealer",
            "benchmark",
            "benchmarking",
            "application",
            "applications",
            "use-case",
            "roi",
        }
    ),
}
"""Vocabulary that marks a question as belonging to one shelf rather than another.

Broader than :data:`DOMAIN_TERMS`, because these two lists do different jobs. The
gate decides whether to search; these decide where to look *first* for a question
that is already in scope, and a word that is too general to admit a question can
still be a good hint about which literature it belongs to.

Overlap between shelves is intended. ``annealer`` belongs to the quantum-computing
literature and to the commercial one, and a question naming it scores on both --
which is exactly the tie that escalation exists to break.
"""

if set(SHELF_TERMS) != set(shelf_names()):  # pragma: no cover - import-time guard
    raise RuntimeError(
        "the router's shelf vocabularies do not match the shelves that exist: "
        f"vocabularies {sorted(SHELF_TERMS)}, shelves {sorted(shelf_names())}"
    )

TIE_MARGIN = 1
"""How far ahead a shelf must score before the deterministic choice is trusted.

One word. A question that hits two words of the circuit vocabulary and one of the
physics vocabulary is about circuits; a question that hits two of each is
genuinely both, and guessing between them is worse than asking. Raising this
would send more questions to a model and cost latency for ties that were not
close; lowering it to zero would resolve every draw by dictionary order, which is
a decision nobody made.
"""

MAX_SHELVES = 2
"""How many shelves a single search may be restricted to.

Two rather than one because the honest answer to some questions really is "both
literatures", and two rather than three because restricting to all three is the
same as not restricting at all -- at which point the routing decision has been
made and then thrown away.
"""


@dataclass(frozen=True, slots=True)
class Routing:
    """What was decided about a question before any searching happened.

    Frozen because a routing decision is evidence: it is written into the trace
    and read back when someone asks why a particular note was or was not
    consulted, and a record that a later stage can edit is not evidence.

    Attributes:
        question: The question as asked, unmodified. Kept so the record stands on
            its own.
        needs_retrieval: Whether to search at all. ``False`` means the question
            carries none of the project's vocabulary, and the caller should say
            so rather than return four irrelevant passages.
        shelves: Which knowledge bases to search first, best guess first. Empty
            means no shelf won and the whole library should be searched from the
            outset, which is the right behaviour when the question is in scope
            but names nothing shelf-specific.
        topics: Hyphenated topic slugs found in the question, passed downstream as
            a ranking bias rather than a filter.
        reason: One sentence a person can read. Shown when retrieval is skipped,
            because "nothing was searched" is only actionable if it says why.
        decided_by: Whether the shelf choice cost a model call.
    """

    question: str
    needs_retrieval: bool
    shelves: tuple[str, ...]
    topics: tuple[str, ...]
    reason: str
    decided_by: DecidedBy

    @property
    def searches_everything(self) -> bool:
        """Whether the first round is unrestricted.

        True when the question is in scope but no shelf stood out. Worth naming
        because it looks identical in a trace to a round whose shelf was preferred
        and answered nothing, and the two mean different things: this one never had
        a guess to spend.
        """
        return self.needs_retrieval and not self.shelves

    def describe(self) -> dict[str, Any]:
        """Render the decision as scalar fields for a log record or a trace.

        Returns:
            Primitives only, so the record can be written straight out without a
            custom encoder.
        """
        return {
            "needs_retrieval": self.needs_retrieval,
            "shelves": list(self.shelves),
            "topics": list(self.topics),
            "decided_by": self.decided_by,
            "reason": self.reason,
        }


class ShelfChoice(BaseModel):
    """A model's answer when the word scores could not separate two shelves.

    One field and a sentence. The model is being asked to break a tie between
    named options, not to design the search, and a schema that invited it to
    rewrite the query as well would blur which half of a bad decision came from
    where.
    """

    shelf: str = Field(description="The name of the single best knowledge base to search first.")
    reason: str = Field(description="One short sentence on why that knowledge base fits.")


def guard(question: str) -> Screening:
    """Screen a question before it is allowed to steer a search.

    Retrieval takes text from the user and puts it into a prompt beside passages
    the model is meant to trust. Screening first is what keeps a question from
    carrying instructions across that boundary.

    Args:
        question: The text as typed.

    Returns:
        The screening verdict. Callers refuse when it is blocked; nothing here
        rewrites the question, because a silently repaired injection attempt is
        an injection attempt nobody logged.
    """
    return screen(question)


def _words(text: str) -> frozenset[str]:
    """Extract the comparable words of a piece of text.

    Args:
        text: Any text.

    Returns:
        Lower-cased word tokens, deduplicated. Deduplicated because a question
        that repeats "circuit" four times is not four times more about circuits,
        and a score that says otherwise is a score a keyword-stuffed question can
        move.
    """
    return frozenset(WORD.findall(text.lower()))


def _gate_words(text: str) -> frozenset[str]:
    """The words of a question, plus the pieces of each hyphenated compound.

    :func:`_words` keeps a hyphenated token whole, and for shelf scoring that is
    right -- ``free-fermion`` should not score a question against every note
    containing "free". The scope gate wants the opposite, because it is asking a
    weaker question: does this sentence mention the subject at all?

    Keeping compounds whole there refused questions about the project's own
    subject. ``quantum-to-classical`` is one token, so neither ``quantum`` nor
    ``classical`` matched and *detail the mathematics of the quantum-to-classical
    mapping* was turned away at the door -- as were *what does the
    transverse-field term do* and *describe a spin-chain*, which name the model
    this project is about and nothing else. The gate's docstring already says a
    false negative is the failure that matters; this is what it costs.

    Splitting only here keeps that fix out of the shelf router, where the whole
    token is what does the work.

    Args:
        text: The question as asked.

    Returns:
        Every whole token, and additionally each hyphen-separated piece of any
        token that has one.

    Examples:
        >>> sorted(_gate_words("quantum-to-classical mapping"))
        ['classical', 'mapping', 'quantum', 'quantum-to-classical', 'to']
    """
    whole = _words(text)
    pieces = {piece for token in whole for piece in token.split("-") if piece}
    return whole | frozenset(pieces)


def _scores(words: frozenset[str]) -> dict[str, int]:
    """Score every shelf by how much of its vocabulary the question uses.

    Args:
        words: The question's words.

    Returns:
        One count per shelf, keyed by shelf name.
    """
    return {name: len(words & terms) for name, terms in SHELF_TERMS.items()}


def topics_in(question: str) -> tuple[str, ...]:
    """Pick out the hyphenated topic slugs a question names.

    The corpus tags every note with slugs of this shape, so a question that
    happens to use one is naming a tag directly and that is worth more than a
    word match. Returned separately from the shelf choice because a topic biases
    the ranking while a shelf restricts the search, and conflating the two would
    make a passing mention act as a filter.

    Args:
        question: The question as asked.

    Returns:
        The hyphenated tokens, in the order they appear, deduplicated.

    Examples:
        >>> topics_in("How do barren plateaus affect a free-fermion chain?")
        ('free-fermion',)
    """
    return tuple(dict.fromkeys(word for word in WORD.findall(question.lower()) if "-" in word))


def _ask_shelf(question: str, tied: list[str], pool: ModelPool) -> tuple[str, str] | None:
    """Spend one call breaking a tie between named shelves.

    Args:
        question: The question as asked.
        tied: The shelf names that scored equally, in registry order.
        pool: The campaign's models. Served at the fast tier, and counted against
            the run's call ceiling like every other call.

    Returns:
        The chosen shelf and the reason given, or ``None`` if the call failed or
        named a shelf that does not exist. ``None`` rather than a raise: an
        unusable answer here costs the precision of a filter, not the answer.
    """
    menu = "\n".join(f"- {shelf.name}: {shelf.covers}" for shelf in SHELVES if shelf.name in tied)
    choice = pool.invoke(
        "shelf_choice",
        ShelfChoice,
        SHELF_CHOICE,
        f"Question: {question}\n\nKnowledge bases:\n{menu}",
    )
    if choice is None or choice.shelf not in tied:
        return None
    return choice.shelf, choice.reason


CONTINUATION_MARKERS: tuple[str, ...] = (
    "and ",
    "what about",
    "how about",
    "what if",
    "same ",
    "instead",
    "now ",
    "then ",
    "also ",
    "but ",
    # Pointing back at the answer on screen rather than at the question that produced
    # it. These arrive constantly once a figure has been drawn -- "in the above, can we
    # plot the range 0 to 20?" -- and without them such a question is judged on its own
    # vocabulary, found to name no physics, and refused underneath the very picture it
    # is asking about.
    "in the above",
    "from the above",
    "for the above",
    "the above",
    "in that",
    "for that",
    "on that",
    "in this",
    "using the same",
    "with the same",
    "for the same",
)
"""Openings that mark a question as continuing the one before it.

A follow-up is elliptical by design -- "and with periodic boundary conditions?"
names no chain because the chain is already on the table -- so it cannot be judged
in scope on its own vocabulary. It has to inherit, and these are the words that say
it is entitled to.

Every one of them is *anaphoric*: it refers to something already said and means
nothing on its own. That is the property that makes the list safe to extend, and it
is why "can you" and "could we" are not on it -- they open a follow-up just as often,
and they also open "can you tell me the capital of France?", which would then inherit
scope for the rest of the session.

The alternative, letting *any* question inherit scope once a conversation has
started, was tried and is wrong: it puts "what is the capital of France?" back in
scope for the rest of the session, which is the exact failure the gate exists to
prevent, only now it is intermittent.
"""


BACK_REFERENCES: tuple[str, ...] = (
    "of this kind",
    "of that kind",
    "like this one",
    "like that one",
    "questions like this",
    "questions like that",
    "the same question",
    "this kind of question",
    "that kind of question",
)
"""Phrases that can only be pointing at something already asked.

Matched anywhere in the sentence, unlike :data:`CONTINUATION_MARKERS`, which are
weak enough to need the start position. These carry a back-reference outright, so
where they sit does not change what they mean.

They exist because the application proposes follow-ups and its scope gate then read
them cold. *What would a hundred questions of this kind cost?* names no physics and
opens with "what", so it was declined as off-topic on the same screen as the answer
that suggested it.
"""


def continues_earlier(question: str) -> bool:
    """Say whether a question reads as a follow-up to something already asked.

    Args:
        question: The question as asked.

    Returns:
        ``True`` when it opens with a continuation marker, or contains a phrase that
        points back at something already said.

    Continuation markers are matched at the *start* only, because they are weak on
    their own: "the coupling and the field" contains "and " and continues nothing.
    A back-reference is different -- "of this kind" cannot be read any way but as a
    reference to something earlier -- so those are matched anywhere.

    That distinction is what admits the application's own suggested follow-ups. *What
    would a hundred questions of this kind cost?* was proposed by the app underneath
    an answer and then declined by its own scope gate as off-topic, because it opens
    with "what" and names no physics. It is in scope, but only in the presence of the
    exchange it refers to -- which is exactly the condition the caller applies, since
    a back-reference with nothing to refer back to is not a question this can answer
    either.

    Examples:
        >>> continues_earlier("and what about a ring?")
        True
        >>> continues_earlier("What would a hundred questions of this kind cost?")
        True
        >>> continues_earlier("What is the best pizza in Vilnius?")
        False
    """
    lowered = question.strip().lower()
    if any(lowered.startswith(marker) for marker in CONTINUATION_MARKERS):
        return True
    return any(phrase in lowered for phrase in BACK_REFERENCES)


def in_domain(question: str) -> bool:
    """Say whether a question is about anything this project holds notes on.

    The scope gate, on its own, so that it can be asked *before* a question is turned
    into a Hamiltonian rather than only in front of a corpus search. Those are two
    different moments and only the earlier one is any use: by the time a search
    happens the question has already been formalised, and a question that named no
    physics has had a chain invented for it.

    Deliberately generous. A false positive costs one search that finds nothing; a
    false negative refuses a question the corpus covers, which is the failure a reader
    actually notices and complains about.

    Args:
        question: The question as asked.

    Returns:
        ``True`` when the question shares vocabulary with this project's subject.

    Examples:
        >>> in_domain("How deep must the ansatz circuit be?")
        True
        >>> in_domain("What is the best pizza in Vilnius?")
        False
    """
    lowered = question.lower()
    if any(phrase in lowered for phrase in DOMAIN_PHRASES):
        return True
    return bool(_gate_words(question) & DOMAIN_TERMS)


def route(
    question: str,
    pool: ModelPool | None = None,
    *,
    allow_model: bool = True,
) -> Routing:
    """Decide whether to search, and which knowledge base to search first.

    Deterministic in the ordinary case. A model is reached for only when two shelves
    score within :data:`TIE_MARGIN` of each other, the one situation where the cheap
    answer would be arbitrary.

    Args:
        question: The question as asked. Screen it with :func:`guard` first; this
            function does not, because refusing and routing are different decisions
            and a caller may want to log the refusal differently.
        pool: The campaign's models, consulted to break a tie. Omitted means a pool
            is built from configuration, and the tie is left to registry order if no
            model can be reached.
        allow_model: Set ``False`` to forbid the escalation outright, which is what a
            latency-sensitive caller passes when it would rather take an arbitrary
            tie-break than wait for a round trip.

    Returns:
        The decision, always. Nothing here raises: a question that cannot be routed
        is searched everywhere, which is slower but never wrong.

    Examples:
        An out-of-scope question is refused before anything is searched:

        >>> route("What is the best pizza in Vilnius?", allow_model=False).needs_retrieval
        False

        A question about circuits goes to the quantum-computing shelf:

        >>> route("How deep must the ansatz circuit be?", allow_model=False).shelves
        ('quantum-computing',)
    """
    words = _words(question)
    if not in_domain(question):
        _logger.info("routing_out_of_scope", extra={"words": len(words)})
        return Routing(
            question=question,
            needs_retrieval=False,
            shelves=(),
            topics=(),
            reason=(
                "the question names nothing this project holds notes on: the corpus "
                "covers one spin chain, the circuits people run it on, and what it is "
                "applied to"
            ),
            decided_by="heuristic",
        )

    topics = topics_in(question)
    scores = _scores(words)
    best = max(scores.values())

    if best == 0:
        return Routing(
            question=question,
            needs_retrieval=True,
            shelves=(),
            topics=topics,
            reason="in scope, but nothing in the question favours one knowledge base",
            decided_by="heuristic",
        )

    leaders = [name for name in shelf_names() if scores[name] >= best - TIE_MARGIN + 1]
    if len(leaders) == 1:
        return Routing(
            question=question,
            needs_retrieval=True,
            shelves=(leaders[0],),
            topics=topics,
            reason=f"the question uses {best} words specific to {leaders[0]}",
            decided_by="heuristic",
        )

    tied = [name for name in shelf_names() if scores[name] == best]
    if allow_model and len(tied) > 1:
        answer = _ask_shelf(question, tied, pool if pool is not None else ModelPool())
        if answer is not None:
            chosen, reason = answer
            _logger.info("routing_escalated", extra={"shelf": chosen, "tied": len(tied)})
            return Routing(
                question=question,
                needs_retrieval=True,
                shelves=(chosen,),
                topics=topics,
                reason=reason,
                decided_by="model",
            )

    ordered = sorted(leaders, key=lambda name: (-scores[name], shelf_names().index(name)))
    return Routing(
        question=question,
        needs_retrieval=True,
        shelves=tuple(ordered[:MAX_SHELVES]),
        topics=topics,
        reason=(
            f"{len(ordered)} knowledge bases score within a word of each other; "
            "searching the closest two"
        ),
        decided_by="heuristic",
    )
