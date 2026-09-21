"""Labelled questions for scoring the search half, and how they were labelled.

**Why this suite exists.** The retrieval layer is the largest single piece of
engineering in the project -- a hybrid of vector similarity and BM25, fused by
reciprocal rank, graded, re-queried once when a round keeps nothing, de-duplicated
and reordered for reading. Every part of that was measured only by unit tests
asserting that each stage does what it says. Nothing measured whether the whole of
it *finds the right thing*, which is the only question a reader of a
retrieval-augmented system actually has. Accuracy and honesty had suites and a
scorecard; the retrieval layer had a score of nothing at all.

**Where the labels come from, and why they are not circular.** Every note in the
corpus carries a curated ``topics`` tuple, written when the shelf was assembled and
stored in the note's frontmatter -- see :mod:`src.rag.ingest`. Retrieval never reads
those topics to *rank*: the vector half embeds the chunk body, and the keyword half
scores the body plus the title and citation line. So a topic tag is an
algorithm-independent relevance judgement that already existed, and a passage counts
as relevant here when its note declares one of the topics a good answer to the
question would have to come from.

That is the property a made-up label lacks. Scoring retrieval against keywords drawn
from the question would measure whether BM25 can find its own vocabulary, which it
can, and would say nothing.

**How a case was written.** Each question was written first, from what a person
using this application would plausibly ask; then the topics a correct answer must
draw on were named, from the question alone; then the suite was run and the result
recorded. The order matters and is the whole discipline of the thing -- labels
adjusted after seeing a score are no longer measuring anything. Two cases below are
deliberately hard and one is deliberately outside the corpus.

**The expected shelf is a second, independent label.** :attr:`Case.shelf` records
which of the four knowledge bases the question belongs to, which scores the router's
choice separately from the search's ranking. A question routed to the wrong shelf and
answered correctly anyway is a different result from one routed correctly and answered
badly, and the two have different fixes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Case:
    """One labelled question.

    Attributes:
        name: Short identifier, used in the scorecard's tables.
        question: The question, in the words somebody would actually type.
        topics: Curated topic tags a relevant note must declare at least one of.
            Empty means no note in the corpus is relevant, which is the label for
            a question the corpus does not cover -- see :data:`OFF_CORPUS`.
        shelf: The knowledge base the question belongs to, or ``""`` when no shelf
            covers it.
        why: What this case is here to catch. Written for a reader of the
            scorecard, who has no reason to know why one question about annealing
            is worth more than another.
    """

    name: str
    question: str
    topics: tuple[str, ...]
    shelf: str
    why: str

    @property
    def answerable(self) -> bool:
        """Whether any note in the corpus should be relevant to this question.

        Returns:
            True unless the case is labelled with no topics, which is how a
            question outside the corpus is written down.
        """
        return bool(self.topics)


CASES: tuple[Case, ...] = (
    Case(
        name="critical-point",
        question="What happens to the transverse-field Ising chain at h = J?",
        topics=("criticality", "quantum-criticality", "critical-exponents"),
        shelf="physics-notes",
        why=(
            "The plainest question the corpus covers, and the one the whole project "
            "is about. A suite where this fails is a suite scoring a broken index "
            "rather than a ranking."
        ),
    ),
    Case(
        name="exact-solution",
        question="How is the ground-state energy of the chain solved exactly?",
        topics=("exact-solution", "free-fermions", "jordan-wigner", "ground-state-energy"),
        shelf="physics-notes",
        why=(
            "Four tags, all naming the same route through the physics. Tests whether "
            "a question phrased in outcomes finds notes phrased in methods."
        ),
    ),
    Case(
        name="barren-plateaus",
        question="Why do the gradients vanish as the circuit gets deeper?",
        topics=("barren-plateaus",),
        shelf="quantum-computing",
        why=(
            "The hardest kind of case in the suite: the question shares not one "
            "content word with the term of art that names its answer. Pure "
            "keyword search cannot pass this, so it is where the vector half has "
            "to earn its place."
        ),
    ),
    Case(
        name="qaoa-versus-vqe",
        question="Should I use QAOA or VQE for this problem?",
        topics=("qaoa", "vqe", "variational"),
        shelf="method-comparison",
        why=(
            "A comparison question, which is the shape most of this application's "
            "traffic takes. Both named methods are in the corpus, so a failure here "
            "is a ranking failure rather than a coverage one."
        ),
    ),
    Case(
        name="annealing-gap",
        question="How slowly does a quantum annealer have to run near a phase transition?",
        topics=("quantum-annealing", "criticality"),
        shelf="quantum-computing",
        why=(
            "Two topics that meet in one physical fact -- the gap closes, so the "
            "schedule must slow -- held on notes that mostly discuss one or the "
            "other."
        ),
    ),
    Case(
        name="imaginary-time",
        question="Can imaginary-time evolution be run on a quantum circuit?",
        topics=("imaginary-time", "varqite"),
        shelf="method-comparison",
        why=(
            "The method this project implements classically, asked about in its "
            "quantum form. Checks that a term in the question does not simply "
            "return the notes about the classical version."
        ),
    ),
    Case(
        name="shot-budget",
        question="How many measurements does a variational energy estimate need?",
        topics=("shot-budget", "resource-estimation"),
        shelf="quantum-computing",
        why=(
            "The arithmetic the application refuses configurations on. A reader "
            "checking a refusal needs the note that justifies it, so this is a "
            "case where retrieval failing costs the product its citation."
        ),
    ),
    Case(
        name="hardware-limits",
        question="What stops a real device from running a deep circuit on many qubits?",
        topics=("hardware", "connectivity", "transpilation", "error-mitigation"),
        shelf="quantum-computing",
        why=(
            "Broadly labelled on purpose: four ways of answering the same question, "
            "and a good ranking should surface any of them."
        ),
    ),
    Case(
        name="classical-competition",
        question="Can an ordinary computer already do this better?",
        topics=("classical-baseline", "tensor-networks", "dmrg", "matrix-product-states"),
        shelf="method-comparison",
        why=(
            "The question the project's own verdict turns on. Asked in plain words "
            "with no method named, so it tests whether the search can reach four "
            "technical literatures from none of their vocabulary."
        ),
    ),
    Case(
        name="business-problems",
        question="Which business problems can be written as an Ising model?",
        topics=("applications", "qubo", "optimisation", "portfolio-optimisation"),
        shelf="applications",
        why=(
            "The smallest shelf, and the one a physics-heavy index is most likely "
            "to hide. Scores whether the router's shelf choice is doing real work."
        ),
    ),
    Case(
        name="advantage-claims",
        question="Is there a proven speedup for this, or only a hoped-for one?",
        topics=("quantum-advantage", "quantum-speedups", "benchmarking", "verification"),
        shelf="applications",
        why=(
            "A question about the strength of evidence rather than about physics. "
            "The application's central claim is a negative one, and it has to be "
            "able to cite the literature that supports being negative."
        ),
    ),
    Case(
        name="off-corpus",
        question="What is the boiling point of liquid helium at one atmosphere?",
        topics=(),
        shelf="",
        why=(
            "Nothing in the corpus answers this. Labelled with no topics, so the "
            "only way to score it is to return nothing -- which makes it the one "
            "case in the suite that a search returning its four best guesses for "
            "everything must fail. A retrieval score with no case like this is a "
            "score that rewards confident irrelevance."
        ),
    ),
)
"""The labelled set, in the order the scorecard prints them.

Twelve rather than fifty, and chosen rather than sampled. A hand-labelled set this
size is small enough that every label can be defended in one sentence -- the ``why``
field -- and a reader can check the labelling rather than take a pass rate on trust.
Fifty cases labelled by pattern-matching would give a smoother number and no way to
audit it.

One case, :data:`OFF_CORPUS`, is unanswerable by design.
"""

OFF_CORPUS: str = "off-corpus"
"""Name of the case no note in the corpus answers.

Named as a constant because two places need to treat it specially and a string
repeated in two files is a string that will disagree with itself.
"""

DEFAULT_TOP_K: int = 4
"""Passages a case keeps, matching what the application ships.

The score has to be the score of the thing that runs. Measuring at ten while the
product retrieves four would report a precision the product never achieves --
see :data:`src.ui.setting.DEFAULT_PASSAGES`, which is the same number.
"""


VECTOR_SHARE: float = 0.6
"""How much of the search is vector similarity when the suite runs hybrid.

Matches :data:`src.ui.setting.DEFAULT_VECTOR_SHARE`, which is the position the
interface's dial ships at and therefore the mix a visitor's question is actually
answered by. Deliberately *not* :data:`src.rag.retrieve.DEFAULT_VECTOR_SHARE`, which
is 0.5 and applies to callers that do not come through the interface -- a divergence
worth knowing about and worth a test, since a score measured at one mix and reported
as the other is not a score of this application. The retrieval-suite test
holds the two together.
"""


@dataclass(frozen=True, slots=True)
class Judged:
    """One case's result, after the passages it returned were judged.

    Attributes:
        case: The case that was run.
        returned: How many passages came back.
        relevant: How many of those declared one of the case's topics.
        first_relevant_rank: One-based rank of the first relevant passage, or
            ``None`` if none was relevant.
        shelves: Shelves the returned passages came from, in order.
        titles: Titles of the returned passages, for the scorecard's table.
        outcome: What the retrieval layer said it did -- ``grounded``,
            ``nothing_relevant``, ``store_unavailable``, or ``not_needed`` when
            the router declined to search at all.
        routed_to: Shelves the router chose, best guess first. Empty when the
            router refused the question or named none.
        seconds: Wall-clock time for the search.
        failure: Exception type, when the search raised.
    """

    case: Case
    returned: int
    relevant: int
    first_relevant_rank: int | None
    shelves: tuple[str, ...] = ()
    titles: tuple[str, ...] = ()
    outcome: str = ""
    routed_to: tuple[str, ...] = ()
    seconds: float = 0.0
    failure: str = ""

    @property
    def precision(self) -> float:
        """Share of returned passages that were relevant.

        Returns:
            Relevant over returned, and ``0.0`` when nothing was returned --
            except for an unanswerable case, where returning nothing is the
            correct answer and :attr:`passed` is what reads it.
        """
        return 0.0 if self.returned == 0 else self.relevant / self.returned

    @property
    def reciprocal_rank(self) -> float:
        """One over the rank of the first relevant passage.

        The metric that separates "the right note was fourth" from "the right note
        was first", which precision alone cannot see and which is what a reader
        actually experiences: the first citation is the one they read.

        Returns:
            The reciprocal rank, or ``0.0`` when nothing relevant came back.
        """
        return 0.0 if self.first_relevant_rank is None else 1.0 / self.first_relevant_rank

    @property
    def found_something(self) -> bool:
        """Whether at least one relevant passage came back."""
        return self.relevant > 0

    @property
    def routed_correctly(self) -> bool:
        """Whether the router sent this question to the shelf the case expects.

        The router's own choice, not the shelf the passages happened to come
        from. Those differ on purpose: retrieval honours the chosen shelf for one
        round and then widens to the whole library, so judging the router by where
        the passages ended up would mark a correct choice wrong whenever widening
        did its job.

        Returns:
            True when the case names a shelf and the router's best guess is it. An
            unanswerable case names no shelf and is not scored on routing.
        """
        if not self.case.shelf or not self.routed_to:
            return False
        return self.routed_to[0] == self.case.shelf

    @property
    def routed_nowhere(self) -> bool:
        """Whether the router named no shelf and left the whole library open.

        Reported separately from a wrong choice, because it is not one. An empty
        shelf list is the router saying it could not separate the four knowledge
        bases on this question's words, and the search that follows is unrestricted
        -- slower, never wrong. Counting that as a routing error would push the
        design towards guessing, which is the opposite of what the fallback is for.

        Returns:
            True when the case names a shelf and the router named none.
        """
        return bool(self.case.shelf) and not self.routed_to

    @property
    def passed(self) -> bool:
        """Whether this case counts as a pass.

        One binary per case, so the suite has a pass rate a reader can quote
        alongside the continuous metrics.

        Returns:
            For an answerable case, whether anything relevant came back at all.
            For the unanswerable one, whether the search correctly returned
            nothing -- the two conditions are opposite, which is exactly why the
            off-corpus case is in the suite.
        """
        if not self.case.answerable:
            return self.returned == 0
        return self.found_something
