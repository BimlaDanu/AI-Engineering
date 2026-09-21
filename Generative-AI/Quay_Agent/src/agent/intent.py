"""What kind of answer a question is asking for, decided before any work is done.

Without this seam, *write me a variational eigensolver*, *what is a barren
plateau* and *why is this chain exactly solvable* all receive a feasibility
campaign, a depth ladder and a verdict about something else. An answer to a
question nobody asked is worse than a refusal, because a refusal is legible.

:class:`Intent` is deliberately tiny, and the graph enforces the rule: an intent
with no branch behind it is a decision nothing can carry out.

The decision follows the same discipline as :mod:`src.agent.router`, since
nothing can start until it has decided:

1. Signal words and phrases score each intent -- microseconds, no network.
   Phrases count, because a bag of words loses the difference between *write me
   a* and *what would you write in the report*.
2. A clear winner is the answer, and no model is called. The ordinary case.
3. A model breaks a genuine tie, and only then.
   :attr:`Reading.decided_by` records which, so the cost is read off the trace.

Sentence position is not weighted. The words opening questions here are *what*,
*why*, *how* and *compare*, which open the application's own questions as often
as a request for prose.

Feasibility is the default rather than the fallback: it is what this application
is for, and the only branch producing a number graded against a reference. The
failure modes are not symmetric -- mistaking a feasibility question for an
explanation costs the reader the numbers and is obvious on sight, while the
reverse spends the shot budget and returns the wrong document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from src.agent.model_selection import ModelPool
from src.agent.reading import (
    asks_for_a_curve,
    asks_for_a_field_sweep,
    asks_to_compare_methods,
    read_count,
)
from src.logging_setup import get_logger

_logger = get_logger("agent.intent")

Intent = Literal["feasibility", "explain", "implement"]
"""What kind of answer the question wants.

``feasibility`` runs the campaign -- baseline, depth ladder, verdict, report.
``explain`` answers in prose from the notes, running no circuits and inventing no
verdict. ``implement`` writes code for the chain.

Three, because three is what the graph has branches for. This is not a taxonomy of
questions; it is the set of things this agent can actually do, and every member has a
node that carries it out.
"""

DecidedBy = Literal["heuristic", "model", "default"]
"""How the intent was arrived at, recorded so the trace can show it."""

WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
"""Words, keeping hyphenated and apostrophised forms whole.

Shared shape with :data:`src.agent.router.WORD` rather than shared code: these two
tokenise for different purposes and coupling them would mean a change made for shelf
matching silently altering which branch a question takes.
"""

MACHINE_TERMS: frozenset[str] = frozenset(
    {
        # The technologies, by name. A question that says "trapped-ion" is asking
        # about a kind of machine; a feasibility question says "hardware" and leaves
        # the kind unspecified, which is why the generic words stay where they are.
        "superconducting",
        "transmon",
        "trapped",
        "ion",
        "ions",
        "rydberg",
        "photonic",
        "annealer",
        "annealers",
        # The vendors. Naming one is naming a machine.
        "ibm",
        "ionq",
        "quantinuum",
        "rigetti",
        "quera",
        "pasqal",
        "google",
        # What a machine has rather than what it achieves.
        "topology",
        "connectivity",
        "processors",
        "qpu",
        "qpus",
        # Plurals only, deliberately, and that rule covers "processors" above too.
        # "Which quantum computers" is a request to be told about machines; "is a
        # quantum computer worth it" is a feasibility question and must keep scoring
        # as one. "Is this processor good enough" is the same trap one word along.
        "computers",
        "machines",
        "devices",
    }
)
"""Words naming a *particular* kind of quantum machine, scored as ``explain``.

Distinct from the generic hardware words, which stay in :data:`FEASIBILITY_TERMS` where
they belong: *hardware*, *device*, *coherence*, *fidelity* and *noise* are how somebody
asks whether a machine is good enough, and that question is answered by pricing
circuits. Naming a technology or a vendor is a different question -- *which* machine --
and it is answered from the notes, in prose, with citations.

The plurals are deliberate and the singulars are deliberately absent. "Which quantum
**computers** could run these methods" asks to be told about machines; "is a quantum
**computer** worth it here" is a feasibility question and has to keep scoring as one.
One letter carries that distinction in English and it is cheaper to honour it than to
write a rule that recovers it.
"""

MACHINE_PHRASES: tuple[str, ...] = (
    "which quantum computer",
    "which real quantum",
    "what quantum computer",
    "which machine",
    "which device",
    "which hardware",
    "what hardware",
    "which processor",
    "what kind of hardware",
    "run these methods",
    "real quantum computer",
    "on real hardware",
    "trapped ion",
    "neutral atom",
    "heavy-hex",
    "heavy hex",
    "d-wave",
)
"""Ways of asking *which* machine, rather than whether a machine is good enough.

The defect these exist for. *Which real quantum computers could run these methods?*
scored **zero on all three intents**, so the default applied, a two-spin chain was
invented for it and the campaign spent nine hundred million measurements returning a
verdict on whether quantum hardware is worth using -- to a reader who had asked which
machines exist. It survived only because :func:`src.agent.graph.after_converge` diverts
a chain-less feasibility reading to prose; add a length to the same sentence and the
guard cannot fire, and the full campaign runs.

Two starters were deleted for this exact failure -- see ``src.ui.starters.REMOVED`` --
and this one was still on the buttons. A guard downstream is not a substitute for
reading the question correctly.
"""

IMPLEMENT_TERMS: frozenset[str] = frozenset(
    {
        "code",
        "implement",
        "implementation",
        "script",
        "snippet",
        "program",
        "function",
        "notebook",
        "qiskit",
        "pennylane",
        "cirq",
        "numpy",
        "python",
    }
)
"""Words that say the reader wants something they can run.

A named framework counts on its own. Nobody writes *qiskit* into a question about
whether hardware is worth it; they write it when they want a file.
"""

IMPLEMENT_PHRASES: tuple[str, ...] = (
    "write me",
    "write a",
    "write the",
    "show me the code",
    "give me the code",
    "how do i code",
    "how would i code",
    "in python",
    "code up",
    "sample code",
    "example code",
)
"""Multi-word forms that a bag of words loses.

*Write a* is a request for a file; *write* alone appears in "what would you write in
the report", which is not.
"""

EXPLAIN_TERMS: frozenset[str] = frozenset(
    {
        "explain",
        "why",
        "what",
        "how",
        "meaning",
        "means",
        "intuition",
        "difference",
        "compare",
        "background",
        "derive",
        "derivation",
        "define",
        "definition",
        "concept",
        "understand",
        # Asking to be taught. None of these was here, and the omission cost the
        # application its own subject: *can you teach me the quantum-to-classical
        # mapping of the transverse-field Ising chain?* scored **zero on all three
        # intents** -- it contains no "what", no "why", no "how", no "explain" -- so
        # the default applied, a feasibility campaign ran on an invented chain, and a
        # request to be taught the mapping this whole project rests on came back as
        # "NO VERDICT".
        "teach",
        "learn",
        "mapping",
        "map",
        "maps",
        "tutorial",
        "introduction",
        "introduce",
        "walk",
        "overview",
        "summarise",
        "summarize",
        # Asking for a picture of how something got somewhere. These belong here and
        # not on the feasibility list, and the difference is not academic: "plot the
        # loss curve for these three methods" was being read as a feasibility question
        # and answered with a go/no-go verdict on a chain, which is a correct answer to
        # a question nobody asked. What such a question wants is the curve and a few
        # sentences about it.
        "plot",
        "plotted",
        "curve",
        "curves",
        "chart",
        "converge",
        "converges",
        "convergence",
        "epoch",
        "epochs",
        "fastest",
    }
)
"""Words that ask to be told something rather than shown a measurement.

``what`` and ``how`` are here despite appearing in feasibility questions too -- they
are weak on their own and are outscored by anything in
:data:`FEASIBILITY_TERMS`, which is the intended interaction rather than an accident
of the word lists.
"""

EXPLAIN_PHRASES: tuple[str, ...] = (
    "what is",
    "what are",
    "what does",
    "why is",
    "why does",
    "why do",
    "how does",
    "tell me about",
    "explain",
    "what do you mean",
    "in simple terms",
    "teach me",
    "show me how",
    "walk me through",
    "help me understand",
    "give me an overview",
    "i want to learn",
    "can you teach",
)
"""Openers that are almost never anything but a request for prose."""

LITERATURE_TERMS: frozenset[str] = frozenset(
    {
        # These three were on the explanation list, which is where the bug hid: they
        # scored towards prose, so `read` was right about them, and
        # `asks_for_literature` -- which reads this list -- said a question about
        # "what the literature says" was not about literature. One word on the wrong
        # list made the two functions disagree about the same sentence.
        "literature",
        "paper",
        "reference",
        "arxiv",
        "papers",
        "preprint",
        "preprints",
        "publication",
        "publications",
        "citation",
        "citations",
        "cite",
        "bibliography",
        "references",
        "authors",
        "survey",
    }
)
"""Words that ask for something to read rather than something to be told.

A separate list from :data:`EXPLAIN_TERMS`, scored with it, because it was added to
fix a specific failure and the fix should stay legible. *Can you provide some recent
arXiv papers of QAOA approximation on the quantum Ising chain?* scored **zero on all
three intents** -- ``paper`` and ``literature`` were on the explanation list, but the
question says *papers* and *arXiv*, and neither singular nor stem matching exists
here. With no signal at all the default applied, a feasibility campaign ran, and a
request for a reading list was answered with "NO VERDICT -- the campaign did not
reach one". That answer was truthful about the campaign and useless about the
question.

Plurals and singulars are both spelled out rather than stemmed. Stemming would be one
more thing on the hot path that can be wrong, and the words a reader uses to ask for
literature are a closed set of about a dozen.

These belong with the explanation signals rather than in a fourth intent: the
``explain`` branch already searches the notes, cites what it used, and reaches an
external index when they come back empty. That is exactly the answer such a question
wants, so the branch exists and only the routing was missing.
"""

LITERATURE_PHRASES: tuple[str, ...] = (
    "recent work",
    "recent papers",
    "recent progress",
    "recent literature",
    "prior work",
    "related work",
    "any papers",
    "any references",
    "point me to",
    "reading list",
    "state of the art",
    "in the literature",
    "who has studied",
    "has anyone",
)
"""How a reader actually asks for the reading list.

*Recent* on its own is not on the word list above and must not be: it appears in
questions about hardware just as often, and the phrase is what distinguishes
*recent work on QAOA* from *the most recent device*.
"""

FEASIBILITY_TERMS: frozenset[str] = frozenset(
    {
        "feasible",
        "feasibility",
        "worth",
        "advantage",
        "speedup",
        "beat",
        "better",
        "outperform",
        "viable",
        "practical",
        "budget",
        "shots",
        "cost",
        "runtime",
        "hardware",
        "device",
        "coherence",
        "fidelity",
        "noise",
        "should",
        "recommend",
        "verdict",
    }
)
"""Words that ask for a judgement backed by arithmetic.

These outrank the explanation words on purpose. *Why is quantum hardware not worth it
for this chain* asks a why-question whose answer is a feasibility assessment, and
sending it to the prose branch would drop the evidence that makes the answer worth
anything.
"""

FEASIBILITY_PHRASES: tuple[str, ...] = (
    "worth it",
    "worth using",
    "worth running",
    "is it worth",
    "quantum advantage",
    "beat a classical",
    "beat the classical",
    "better than classical",
    "how many shots",
    "how deep",
    "can we run",
    "could we run",
    "should we use",
)
"""Phrasings of the question this application was built to answer."""

WIN_MARGIN = 1
"""How far ahead of feasibility a non-default intent must score to take the branch.

One point, not zero, so a single incidental word cannot divert a question away from
the branch that produces evidence. A tie goes to a model, and then to feasibility.
"""


@dataclass(frozen=True, slots=True)
class Reading:
    """What kind of answer the question was read as wanting.

    Attributes:
        intent: The branch the graph will take.
        decided_by: Whether word signals settled it, a model was asked, or nothing
            in the question favoured anything and the default applied. Recorded
            because "the agent chose prose" and "the agent found no signal and fell
            through to its default" are different events that look identical in a
            transcript.
        reason: One sentence a reader can check the decision against. Written for
            the trace, so it names the evidence rather than restating the answer.
        scores: What each intent scored, kept for the same reason. A decision whose
            margin was one point is a decision worth looking at twice, and that is
            only visible if the margin survives.
    """

    intent: Intent = "feasibility"
    decided_by: DecidedBy = "default"
    reason: str = "nothing in the question favoured another kind of answer"
    scores: tuple[tuple[Intent, int], ...] = ()

    @property
    def runs_campaign(self) -> bool:
        """Whether this reading spends the shot budget.

        A property rather than ``intent == "feasibility"`` spelled out at each call
        site: the graph edge, the cost panel and the report composer all need the
        answer, and three spellings of one test is how one of them stops agreeing.
        """
        return self.intent == "feasibility"

    def explain(self) -> str:
        """Say what was decided and on what evidence, in one line for the trail."""
        return f"read the question as asking for {self.intent} ({self.decided_by}): {self.reason}"


class IntentChoice(BaseModel):
    """A model's answer when the word signals tied.

    Attributes:
        intent: Which of the three the question wants.
        reason: Why, in one short sentence, for the trace.
    """

    intent: Intent = Field(description="feasibility, explain, or implement")
    reason: str = Field(description="one short sentence naming the evidence in the question")


INTENT_SYSTEM = """You decide what kind of answer a question wants, for an agent \
that studies one physical system: the one-dimensional transverse-field Ising chain.

The agent can do exactly three things, and you must pick one:

- **feasibility** -- run the assessment. It solves the chain on an ordinary computer, \
proposes quantum circuits, prices each one against a real device's coherence time and \
a measurement budget, runs the ones that fit, and returns a verdict on whether quantum \
hardware is worth using. Pick this when the question asks whether something is worth \
doing, what it would cost, how deep a circuit can be, or how the two approaches compare.
- **explain** -- answer in prose from the project's notes, citing them. No circuits are \
run and no verdict is invented. Pick this when the question asks what something means, \
why something is true, or for background.
- **implement** -- write runnable code for the chain. Pick this when the reader wants \
a file they can run.

Pick feasibility when genuinely torn: it is the only branch that produces measured \
numbers, and returning prose where evidence was wanted is the cheaper mistake to \
recover from.

Answer with the intent and one short sentence naming the words in the question that \
decided it. Do not answer the question itself."""


def asks_for_literature(question: str) -> bool:
    """Whether the question is asking for something to read.

    A narrower question than :func:`read`, and asked for a different reason. The
    intent decides which *branch* runs; this decides whether the branch should reach
    outside the project's own notes. Those come apart in the case that produced this
    function: *give me recent arXiv papers on QAOA* routes to prose either way, and
    the notes will happily answer it with four curated passages from 2018 -- which is
    a truthful answer to a question about *recent* work only by accident.

    So a request for literature reaches the external index whether or not the corpus
    found something, while every other question reaches it only when the corpus came
    back empty. The asymmetry is the point: a network call on every question would be
    a network call on every question, and the corpus is what the project actually
    stands behind.

    Args:
        question: The question as asked.

    Returns:
        Whether any literature signal is present.

    Examples:
        >>> asks_for_literature("Any recent arXiv papers on QAOA?")
        True
        >>> asks_for_literature("Is a 12-spin chain worth running on hardware?")
        False
    """
    lowered = question.lower()
    if _words(question) & LITERATURE_TERMS:
        return True
    return any(phrase in lowered for phrase in LITERATURE_PHRASES)


def _words(text: str) -> frozenset[str]:
    """Lowercase word set, for scoring against the term lists.

    Args:
        text: The question as asked.

    Returns:
        Its distinct words. A set rather than a list because a question that says
        "code" four times is not four times more a request for code.
    """
    return frozenset(WORD.findall(text.lower()))


def _score(text: str, terms: frozenset[str], phrases: tuple[str, ...]) -> int:
    """Score one intent against the question.

    Args:
        text: The question as asked.
        terms: Single words that signal this intent.
        phrases: Multi-word forms, each worth as much as a word and checked against
            the raw sentence so that word order counts.

    Returns:
        The score: one per distinct matching word, one per matching phrase.
    """
    lowered = text.lower()
    return len(_words(text) & terms) + sum(1 for phrase in phrases if phrase in lowered)


def _ask(question: str, tied: list[Intent], pool: ModelPool) -> IntentChoice | None:
    """Break a genuine tie with a model.

    Args:
        question: The question as asked. Not wrapped as data: it is screened before
            it reaches this module, and the campaign's single funnel screens every
            prompt again on the way out -- see :meth:`ModelPool.invoke`.
        tied: The intents that scored equally, named in the prompt so the model is
            choosing between the live options rather than all three.
        pool: The campaign's models.

    Returns:
        The choice, or ``None`` when no model could be reached or it answered with
        an intent that was not actually tied -- which is a model ignoring the
        question, and taking it would be worse than falling through to the default.
    """
    options = ", ".join(tied)
    choice = pool.invoke(
        "intent_reading",
        IntentChoice,
        INTENT_SYSTEM,
        f"These scored equally: {options}. Question: {question}",
    )
    if choice is None or choice.intent not in tied:
        return None
    return choice


def read(question: str, pool: ModelPool | None = None, *, allow_model: bool = True) -> Reading:
    """Decide what kind of answer the question is asking for.

    Args:
        question: The question as asked. Screen it first -- this function does not,
            because refusing and reading are different decisions and a blocked
            question should never reach a branch at all.
        pool: The campaign's models, consulted only to break a tie. Omitted builds
            one from configuration.
        allow_model: Set ``False`` to forbid the escalation. Ties then fall to
            feasibility, which is what every offline run does.

    Returns:
        The reading, always. Nothing here raises: a question that cannot be read is a
        feasibility question, the branch producing the most evidence and the one
        whose mistake is most obvious to a reader.

    Examples:
        A request for a file goes to the code branch:

        >>> read("Write me a VQE implementation for a 10-spin chain", allow_model=False).intent
        'implement'

        A request for background goes to prose:

        >>> read("What is a barren plateau?", allow_model=False).intent
        'explain'

        So does a request for something to read:

        >>> read("Can you provide some recent arXiv papers of QAOA approximation "
        ...      "on the quantum Ising chain?", allow_model=False).intent
        'explain'

        The question this application is for keeps the campaign, even though it
        opens with a word on the explanation list:

        >>> read("Why is quantum hardware not worth it for this chain?",
        ...      allow_model=False).intent
        'feasibility'

        And so does anything the word lists have no opinion about:

        >>> read("A 12-spin chain at criticality.", allow_model=False).decided_by
        'default'
    """
    scored: dict[Intent, int] = {
        "feasibility": _score(question, FEASIBILITY_TERMS, FEASIBILITY_PHRASES),
        # Literature signals are scored with the explanation ones rather than
        # merged into the lists themselves, so that the union is visible at the one
        # place it decides anything. A request for papers is a request for prose
        # with citations, which is what this branch produces.
        "explain": _score(
            question,
            EXPLAIN_TERMS | LITERATURE_TERMS | MACHINE_TERMS,
            EXPLAIN_PHRASES + LITERATURE_PHRASES + MACHINE_PHRASES,
        ),
        "implement": _score(question, IMPLEMENT_TERMS, IMPLEMENT_PHRASES),
    }
    table = tuple(sorted(scored.items(), key=lambda pair: -pair[1]))

    # Decided before the word scores are compared, because on these sentences the
    # word scores are wrong and they are wrong for a structural reason: *plot the
    # ground-state energy and its first and second derivatives of the
    # transverse-field Ising model as a function of h/J* is dense in the vocabulary
    # of this application's own subject -- field, energy, ground, transverse, model,
    # chain -- so it outscored the explanation list and read as a request for a
    # feasibility verdict. Three of the five plotting questions a reader actually
    # asked read that way, and the campaign answered every one of them with a
    # go/no-go on quantum hardware. A question asking to be *shown the model's own
    # curves* is not asking whether a quantum computer is worth buying, and no
    # margin rule can recover that from counting nouns -- what recovers it is
    # reading the axis. See `src.agent.reading.asks_for_a_field_sweep`.
    if asks_for_a_field_sweep(question):
        _logger.info("intent_field_sweep", extra={"intent": "explain"})
        return Reading(
            intent="explain",
            decided_by="heuristic",
            reason=(
                "the question asks for an exact quantity traced against the field, "
                "which is answered by solving the model at every point rather than "
                "by pricing circuits against a machine"
            ),
            scores=table,
        )

    # And the mirror image, which is the correction the sentence above provoked. A
    # question that asks to watch three methods descend **and names the chain to
    # watch them on** is not a lesson: *Learning curve for VQE, QAOA and
    # VarQITE: 8 spins at criticality?* took `explain` because every plotting word
    # is on the explanation list, so it raced the methods and then answered in prose
    # -- and the depth ladder, the shot arithmetic, the classical baseline and the
    # verdict, which are the whole of what this application does, never ran on a
    # chain the reader had specified.
    #
    # The plotting words stay where they are, and that is deliberate: they were put
    # on the explanation list because *plot the loss curve for these three methods*
    # was being answered with a go/no-go verdict and no curve. What was missing is
    # the other half of that reading -- a curve question **with a chain in it** wants
    # both, and the graph can give it both, since `after_retrieve` sends every curve
    # question through the race whatever its intent and the verdict opens with what
    # the race found.
    #
    # The chain has to be named in the *sentence*, by `read_count`, and not taken off
    # the formalised model. A follow-up that inherits its length from memory -- *of
    # VQE, QAOA and VarQITE, which converges fastest?* -- is the memory layer's own
    # demonstration and must stay prose, and it is indistinguishable from a
    # chain-less question by anything except where the number came from.
    if (asks_for_a_curve(question) or asks_to_compare_methods(question)) and (
        read_count(question) is not None
    ):
        _logger.info("intent_named_chain_curve", extra={"intent": "feasibility"})
        return Reading(
            intent="feasibility",
            decided_by="heuristic",
            reason=(
                "the question asks to watch the methods on a chain it named, so the "
                "campaign runs the ladder and the baseline on that chain as well as "
                "racing them"
            ),
            scores=table,
        )

    best = max(scored.values())

    if best == 0:
        return Reading(scores=table)

    leaders: list[Intent] = [name for name, value in scored.items() if value == best]

    if len(leaders) == 1:
        winner = leaders[0]
        # A non-default branch has to clear feasibility by a margin, so that one
        # incidental word cannot divert a question away from the evidence.
        if winner != "feasibility" and best - scored["feasibility"] < WIN_MARGIN:
            return Reading(
                intent="feasibility",
                decided_by="heuristic",
                reason=(
                    f"{winner} scored {best} against feasibility's "
                    f"{scored['feasibility']}, which is too close to give up the "
                    "measured answer for"
                ),
                scores=table,
            )
        _logger.info("intent_heuristic", extra={"intent": winner, "score": best})
        return Reading(
            intent=winner,
            decided_by="heuristic",
            reason=f"the question uses {best} words specific to {winner}",
            scores=table,
        )

    if allow_model:
        choice = _ask(question, leaders, pool if pool is not None else ModelPool())
        if choice is not None:
            _logger.info("intent_escalated", extra={"intent": choice.intent})
            return Reading(
                intent=choice.intent,
                decided_by="model",
                reason=choice.reason,
                scores=table,
            )

    return Reading(
        decided_by="heuristic",
        reason=(
            f"{' and '.join(leaders)} scored equally at {best}, so the branch that "
            "produces measured numbers was kept"
        ),
        scores=table,
    )
