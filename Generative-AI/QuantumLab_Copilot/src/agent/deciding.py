"""What to do next, decided from what has been established so far.

The graph used to be a fixed pipeline: the router decided once, the nodes ran in
that order, and nothing could be revised. A search that came back thin was still
followed by exactly one round of tools and then an answer, however little there
was to answer from. This module is the seam that makes the run a **closed loop**
instead -- plan, act, observe, decide again -- bounded at :data:`MAX_STEPS`.

Two properties are worth stating, because both were requirements rather than
conveniences.

**The decision is about state, not about vocabulary.** :class:`Progress` carries
what happened -- were numbers computed, were passages kept, did a tool run -- and
the policy below reads only that. A rule keyed on the words in the question is a
per-question patch that has to be extended for the next question; a rule keyed on
"nothing has been established yet" generalises to questions nobody anticipated.

**The knowledge base comes first, and the outside world is a fallback.** The
policy will not spend a tool call while the corpus has not been searched, and it
reaches for the external tools only once retrieval has actually been tried and
come back without usable passages. That ordering is the whole reason the tools
are an *action* here rather than a node every run passes through.

Like :mod:`src.agent.grading` and :mod:`src.agent.rewriting`, this is a seam with
two implementations: a model decides when one is reachable, and a deterministic
policy decides when none is. The policy is not a degraded mode -- it is what every
test exercises, so the loop's behaviour is specified without a key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from src.agent.llm import ask_structured, chat_model_or_none
from src.logging_setup import get_logger
from src.rag.ingest import SHELVES
from src.settings import Settings

# Was `logging.getLogger(__name__)`, which put these records outside the
# `quantumlab.*` hierarchy the handler is attached to -- so every `decide_overridden`
# was dropped, and the one module whose logs explain the trajectory was the one
# module that logged nowhere. See :mod:`src.logging_setup`.
LOG = get_logger("agent.deciding")

Action = Literal["compute", "retrieve", "consult", "implement", "finish"]
"""What the agent may do next.

``compute`` solves the chain, ``retrieve`` searches the corpus, ``consult`` offers
the external tools, and ``finish`` composes the answer from what is already there.
Deliberately small: an action the graph has no node for would be a decision
nothing could carry out.
"""

MAX_SEARCHES = 2
"""How many separate retrieval actions one question may take.

Not the same thing as :data:`src.rag.retrieve.MAX_ROUNDS`, and the two are easy to
confuse. ``MAX_ROUNDS`` is the corrective retry *inside* one search: the same
question, rewritten, when the first phrasing found nothing. This is the number of
times the loop may search *at all*, each with a query it chose after reading what
the previous one returned. One is recovery from a vocabulary miss; the other is
noticing that half the question is still unanswered.

Two, because the second search is the one with new information behind it and the
third would be guessing.
"""

MAX_STEPS = 4
"""How many actions one question may take before the answer is composed.

A budget, not a target -- most questions finish in two or three, because the policy
stops as soon as it has what the route asked for. It is low on purpose: every step
past the first is another model call on a page that recomputes, and an unbounded
loop is the failure mode this project cannot afford. The cap is enforced in the
graph edge as well as here, so a decider that keeps asking for more work cannot
spend more than this.

Four rather than three, and the third step is what the fourth is for. ``finish`` is
itself an action, so a budget of three spent itself on *search, search again for the
gap, compose* and left ``consult`` unreachable: the loop could look for what it was
missing, or it could look outside the corpus, but never both. An action in
:data:`Action` that no trajectory can contain is not a capability, and a question
like *find recent arXiv papers on Trotter error* was answered out of the notes with
the paper tool sitting unused.
"""


@dataclass(frozen=True)
class Progress:
    """What the run has established, as the decider sees it.

    Attributes:
        wants_number: Whether the route asked for a computation.
        wants_prose: Whether the route asked for an explanation or a citation.
        wants_curve: Whether the question asked for a curve rather than a value.
            Solving the one chain in the settings cannot answer it -- a curve needs
            the sweep, which lives behind ``consult`` -- so this is the state that
            makes the tool step worth taking even when nothing is missing.
        has_numbers: Whether a chain has actually been solved and cross-checked.
        passages: How many graded passages survived retrieval.
        covered: What the kept passages are *about*, one short line each. A count
            says whether a search returned anything; it cannot say whether the
            material answers the question. Deciding "this covers the mapping but
            not the hardware limits" needs the subjects, so they are part of the
            state rather than something the decider is trusted to imagine.
        searches: How many retrieval actions have run, at most
            :data:`MAX_SEARCHES`. A count rather than a flag because "not tried",
            "tried once" and "tried twice" lead to three different decisions.
        corpus_available: Whether the knowledge base could be opened at all. Kept
            apart from ``passages == 0`` because the two call for opposite
            decisions: a corpus that was quiet is worth one more, narrower query,
            and a corpus that is not there returns the same nothing however the
            query is written.
        wants_code: Whether the question asked for something to run. Kept apart from
            ``wants_prose`` because the two are established by different actions --
            the corpus answers one and :mod:`src.agent.drafting` the other -- and a
            run that searched is not therefore a run that has written the code.
        drafted: Whether the drafting action has been *taken*, which is not the same
            as whether it produced anything. A draft can come back empty, and keying
            this on the code existing would let the loop choose to write it again,
            fail again, and spend the budget doing so.
        tools_run: Whether the external tools have been offered already.
        tools_available: Whether there is any source outside the corpus to reach
            for -- see :attr:`src.tools.calling.Toolbox.external_available`. False
            by default, which is the state in which composing is the only thing
            left to do: an agent cannot decide to consult sources it has not been
            given, and asking a model to weigh an action nothing can carry out is
            the latency this loop is careful about.
        taken: How many actions have been taken.
    """

    wants_number: bool = False
    wants_prose: bool = False
    wants_curve: bool = False
    wants_code: bool = False
    drafted: bool = False
    has_numbers: bool = False
    passages: int = 0
    covered: tuple[str, ...] = ()
    searches: int = 0
    corpus_available: bool = True
    tools_run: bool = False
    tools_available: bool = False
    taken: int = 0

    @property
    def searched(self) -> bool:
        """Whether retrieval has been attempted at all.

        Distinct from ``passages == 0``: "not tried" and "tried and found nothing"
        call for opposite decisions, and collapsing them is how a loop ends up
        searching twice or not at all.
        """
        return self.searches > 0

    @property
    def exhausted(self) -> bool:
        """Whether the step budget is spent."""
        return self.taken >= MAX_STEPS

    @property
    def established(self) -> bool:
        """Whether the run has anything at all to answer from.

        Returns:
            ``True`` once a cross-checked number or a kept passage exists. This is
            the floor under the opening decision: the model chooses what the
            question needs, but it may not choose to answer from nothing. A search
            that ran and kept nothing does not count -- an empty search establishes
            that the corpus is quiet on the subject, which is worth knowing and is
            not material.
        """
        return self.has_numbers or self.passages > 0

    @property
    def can_search_again(self) -> bool:
        """Whether another, more narrowly aimed search is still allowed.

        Returns:
            ``False`` once the searches are spent, and also when the knowledge base
            could not be opened. Rewriting the query is the remedy for a corpus that
            did not match; it is no remedy at all for one that is not there, and
            spending the second round on it costs a model call to reach the same
            nothing.
        """
        return self.corpus_available and self.searches < MAX_SEARCHES

    @property
    def can_reach_outside(self) -> bool:
        """Whether the corpus is spent and there is still somewhere else to look.

        The state in which "the material falls short" has exactly one remaining
        answer: the searches are used up, so the knowledge base has nothing further
        to give, and an outside source is offered and has not been tried.
        """
        return not self.can_search_again and not self.tools_run and self.tools_available

    def describe(self) -> str:
        """Say what stands, for a model that has to decide what is missing.

        Returns:
            One line per fact, phrased as a state rather than as an instruction.
            Written out here rather than assembled in the prompt so that the model
            and the offline policy are looking at exactly the same picture.
        """
        asked = " and ".join(
            part
            for part, wanted in (
                ("a number", self.wants_number),
                ("an explanation", self.wants_prose),
                ("code to run", self.wants_code),
            )
            if wanted
        )
        lines = [
            f"The question asks for: {asked or 'background'}.",
            f"Numbers computed and cross-checked: {'yes' if self.has_numbers else 'no'}.",
            (
                f"Knowledge base searched: {self.searches} of {MAX_SEARCHES} times"
                f", passages kept: {self.passages}."
            ),
        ]
        if not self.corpus_available:
            # Said plainly, because the line above it reads "0 of 2 searches, 0
            # passages kept" either way, and a model shown only that will quite
            # reasonably propose searching again.
            lines.append(
                "The knowledge base could not be opened, so it was never actually "
                "searched. Searching it again cannot help."
            )
        if self.covered:
            lines.append("What the kept passages cover:")
            lines.extend(f"- {subject}" for subject in self.covered)
        lines.append(f"External tools already used: {'yes' if self.tools_run else 'no'}.")
        lines.append(
            "Outside sources (arXiv, encyclopaedia, web) available: "
            f"{'yes' if self.tools_available else 'no'}."
        )
        lines.append(f"Actions taken so far: {self.taken} of {MAX_STEPS}.")
        return "\n".join(lines)


class Step(BaseModel):
    """One decision, with the reason it was made.

    Attributes:
        action: What to do next.
        reason: Why, in one plain sentence. Shown to the user: the trajectory is
            the part of an agentic run a reader most needs to see, because it is
            the difference between a pipeline and a decision.
        decided_by: ``model`` or ``policy``, never blank. A trajectory that does
            not say who chose cannot be audited.
        focus: What to search for, when the action is a *second* ``retrieve``.
            Empty means "the question as asked", which is right the first time and
            useless the second: repeating a query returns the passages already in
            hand. This field is what makes the follow-up search worth making --
            the loop names the part of the question the first search missed.
    """

    model_config = ConfigDict(frozen=True)

    action: Action = Field(description="One of compute, retrieve, consult, implement, finish.")
    reason: str = Field(description="One plain sentence.")
    decided_by: Literal["model", "policy"] = "model"
    focus: str = Field(
        default="",
        description="For a follow-up retrieve: the missing subject to search for, as a phrase.",
    )


SHELF_TITLES = "; ".join(shelf.title for shelf in SHELVES)
"""The knowledge bases, as the decider is told about them.

Built from :data:`src.rag.ingest.SHELVES` rather than written out, for the reason
:data:`src.agent.router.SHELF_MENU` is: this description had gone stale. It named
"the exact solution, the free-fermion mapping, criticality, and the model's use in
quantum computing" -- a true description of two shelves, written when there were
two. The applications shelf was invisible to the one component that decides
*whether to search at all*, so a question about a routing problem or a portfolio
could be sent outside to arXiv while the note that answered it sat unread. Titles
rather than the full ``covers`` text: the decider needs to know a shelf exists,
and the router is what chooses between them.
"""

DECIDE_SYSTEM = f"""You choose the next action for a physics agent that studies \
the 1D transverse-field Ising model. You do not answer the question and you never \
state a physics result.

Actions:
- compute: solve the chain and cross-check it with a second, independent method.
  The chain's parameters come from the user's settings, not from the question, so
  there is never a missing detail to ask about.
- retrieve: search this project's own knowledge base -- {len(SHELVES)} shelves:
  {SHELF_TITLES}. The third holds what the model is *used for* outside physics, so an
  optimisation or logistics problem written as an Ising cost function, an NP-hard
  problem, machine learning, or which quantum speedups are proven is covered here.
- consult: offer the tools. Two kinds live here. The project's own physics tools --
  a field sweep, which is the only way to produce a *curve* or a plot, and a
  comparison at another chain length -- and outside sources (arXiv, encyclopaedia,
  web). Choose it immediately when the question asks for a curve or a plot, since
  solving the single chain in the settings cannot answer that. Choose it for the
  outside sources only once the knowledge base has been searched, and then only when
  the question wants something the notes cannot hold however many passages they
  return -- recent or named publications, a date, an author's other work -- or when
  the searches are spent and the kept passages still do not cover the question.
- implement: write runnable code for the chain -- an implementation of a method, a
  script the reader can run and modify. Choose it when the question asks for code,
  and choose it *after* a search, so that what you write follows the ansatz and the
  gate decomposition the notes describe rather than being invented. It produces code
  and no numbers: it does not run anything.
- finish: compose the answer from what has been established.

You are asked first with nothing established, and that opening choice is yours: a
question asking only for a value should be solved, a question about the physics or
the literature should be searched, and a question the conversation has already
covered may need neither. You may not choose finish while nothing has been
established -- an answer grounded in nothing is worse than a slow one.

Rules:
- Search the knowledge base before reaching outside it.
- Compare what the kept passages cover against what the question asked. If part of
  the question is not covered and a search is still available, choose retrieve
  again and put the *missing* subject in the focus field as a short phrase -- for
  example "qubit connectivity and gate fidelity". Do not restate the whole
  question there: searching for it again returns what is already in hand.
- Choose finish once the material covers the question, or when a gap is one the
  knowledge base plainly would not hold.
- At most {MAX_SEARCHES} searches and {MAX_STEPS} actions per question.

Your reason field is shown to the user, so write one plain sentence."""


def policy_step(progress: Progress) -> Step:
    """Choose the next action with no model, from the state alone.

    The ordering is the policy, and each branch is a claim about which mistake is
    preferable:

    1. **Compute what was asked for**, first. A verified number is the one thing
       here nothing else can substitute for.
    2. **Search the corpus** when an explanation is wanted, or when nothing at all
       has been established yet -- an answer with no grounding is the failure this
       project exists to prevent, so the corpus is tried even for a question the
       router read as purely numeric.
    3. **Reach outside** only after retrieval has run and kept nothing. This is the
       fallback, and it is deliberately unreachable while the knowledge base is
       untried.
    4. **Finish.**

    Args:
        progress: What the run has established.

    Returns:
        The chosen step, attributed to the policy.
    """
    if progress.exhausted:
        return Step(
            action="finish",
            reason="The step budget for one question is spent.",
            decided_by="policy",
        )
    if progress.wants_number and not progress.has_numbers:
        return Step(
            action="compute",
            reason="The question asks for a value and nothing has been solved yet.",
            decided_by="policy",
        )
    if progress.wants_curve and not progress.tools_run:
        # Before the corpus, and that is not a violation of knowledge-base-first: the
        # sweep is this project's own physics layer, not an outside source. The rule
        # exists so a claim is grounded in the notes before it is grounded in the
        # web, and a curve is not a claim about the literature -- it is a
        # computation. Without this branch, "plot the magnetisation against the
        # field" solved the single point in the settings knob, searched the notes,
        # composed a paragraph about one number, and produced no figure at all.
        return Step(
            action="consult",
            reason="The question asks for a curve, which takes a sweep rather than a single solve.",
            decided_by="policy",
        )
    if not progress.searched and (progress.wants_prose or progress.passages == 0):
        return Step(
            action="retrieve",
            reason="The knowledge base has not been searched yet.",
            decided_by="policy",
        )
    if progress.wants_code and not progress.drafted:
        # After the retrieve branch above, so the draft follows the notes rather than
        # inventing an ansatz, and before reaching outside, because code the question
        # asked for outranks a paper it did not.
        return Step(
            action="implement",
            reason="The question asks for code and none has been written yet.",
            decided_by="policy",
        )
    if progress.searched and progress.passages == 0 and not progress.tools_run:
        return Step(
            action="consult",
            reason=(
                "The knowledge base held nothing usable, so the outside sources are worth a try."
            ),
            decided_by="policy",
        )
    return Step(
        action="finish",
        reason="What the question asked for has been established.",
        decided_by="policy",
    )


def instead_of(proposed: Step, progress: Progress) -> Step:
    """Choose the next action when the model's choice cannot be carried out.

    Falling straight back to :func:`policy_step` threw away the one thing the refused
    proposal established. *Search again* is not only a request for an action; it is a
    statement that the material in hand does not answer the question. When there is
    no search left, the policy's answer to that state is ``finish`` -- so a question
    like *find recent arXiv papers on Trotter error* was composed from the notes with
    the paper tool never offered, and the log showed the decider asking for more and
    being told to stop.

    So a refusal keeps the judgement and redirects the action: the corpus is spent,
    the material falls short, and the only remaining source is outside it.

    Args:
        proposed: What the model asked for, already refused.
        progress: What the run has established.

    Returns:
        ``consult`` when more material is wanted and outside sources are the only
        place left to find it, otherwise the policy's own choice. A refused
        ``compute`` is not a request for material -- the numbers are already in hand
        -- so it redirects nowhere and the policy decides.
    """
    wants_more = proposed.action == "retrieve"
    if wants_more and progress.can_reach_outside:
        return Step(
            action="consult",
            reason=(
                "The knowledge base is searched out and the question is still not "
                "covered, so the outside sources are the only place left to look."
            ),
            decided_by="policy",
        )
    return policy_step(progress)


def is_forced(progress: Progress) -> bool:
    """Whether the state leaves a model nothing to weigh.

    A loop that asks a model what to do next at every step spends a call per step,
    so this exists to skip the *consultation* where consulting cannot change the
    outcome. It is deliberately narrow, and it used to be much wider.

    **The opening move is a decision now, not a default.** Two earlier branches --
    "a number was asked for and nothing is solved" and "the corpus is untried" --
    made the first action of every single run a foregone conclusion. The trajectory
    read ``retrieve[policy] -> ...`` on every question in the application, which is
    a pipeline with a decision bolted to its second step: the run could revise a
    plan but never choose one. The visible symptom was that a purely numeric
    question searched the notes before solving anything, and a question the
    conversation had already answered searched again anyway.

    Removing them costs one model call per question and buys the thing the loop is
    for. What stops the model from skipping grounding is a refusal in :func:`decide`
    rather than a branch here -- "answer before anything is established" is
    overridden -- because a guard that blocks a bad *action* still leaves the good
    ones to be chosen between, whereas a guard that pre-empts the decision leaves
    nothing.

    Args:
        progress: What the run has established.

    Returns:
        ``True`` when the policy's answer is the only sensible one: the budget is
        spent, or the outside sources have already been offered and the corpus is
        searched out, so nothing further can be reached.

        The two open decisions left are the opening move -- *what does this question
        actually need?* -- and the state just after a search: *is this material
        enough?* Nothing else in this project can answer either. A count of passages
        cannot; an earlier version treated ``passages > 0`` as forced, and every
        ordinary question ran retrieve then finish with a partial answer looking
        exactly like a complete one.
    """
    if progress.exhausted:
        return True
    if progress.tools_run:
        return True
    if progress.can_search_again or progress.passages == 0:
        return False
    # Searched to the limit and something came back. Composing is defensible; so is
    # reaching outside the corpus, when there is anything out there to reach for --
    # *find recent arXiv papers on Trotter error* is a question the notes cannot
    # answer however many passages they return, because what it asks for is
    # recency. With no outside source configured there is nothing to weigh and
    # composing is all that is left.
    return not progress.tools_available


def decide(
    question: str,
    progress: Progress,
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> Step:
    """Decide the next action, with a model when one is reachable.

    Args:
        question: The question as asked, screened but otherwise raw.
        progress: What the run has established.
        model: An explicit chat model, normally supplied only by tests.
        settings: Configuration to build a model from.

    Returns:
        The step to take. No model is consulted when the state forces the answer
        (see :func:`is_forced`), so the common question costs no decision calls at
        all; the budget is checked before that, because a spent budget is not a
        decision either. A model that proposes an action the policy has already
        ruled out on grounding grounds is overridden, so the knowledge-base-first
        ordering cannot be talked out of.
    """
    fallback = policy_step(progress)
    if is_forced(progress):
        return fallback
    resolved = chat_model_or_none(model, settings)
    if resolved is None:
        return fallback
    proposed = ask_structured(
        resolved,
        Step,
        DECIDE_SYSTEM,
        f"{progress.describe()}\n\nQUESTION: {question}",
        purpose="decide",
    )
    if proposed is None:
        return fallback
    for detail, refused in (
        (
            # Not refused when a curve was asked for: the tool step is where the
            # sweep lives, and a sweep is a computation rather than an outside
            # source, so searching the notes first would only delay it.
            "the model asked for external tools before the corpus was searched",
            proposed.action == "consult" and not progress.searched and not progress.wants_curve,
        ),
        (
            "the model asked to search again with no focus, which repeats the query",
            proposed.action == "retrieve" and progress.searched and not proposed.focus.strip(),
        ),
        (
            "the model asked to search again with no search left",
            proposed.action == "retrieve" and not progress.can_search_again,
        ),
        (
            "the model asked to solve a chain that is already solved and cross-checked",
            proposed.action == "compute" and progress.has_numbers,
        ),
        (
            "the model asked to write code when the drafting action had already run",
            proposed.action == "implement" and progress.drafted,
        ),
        (
            # The mirror of the guard below on numbers, and it exists for the same
            # reason: retrieval is not a substitute for the thing that was asked for.
            # A question wanting code is answerable from the notes -- the ansatz and
            # the gate decomposition are in there -- so the "established" guard is
            # satisfied by passages alone, and the run would compose a paragraph
            # about how one *would* implement it and never write a line.
            "the model asked to answer without the code the question asked for",
            proposed.action == "finish" and progress.wants_code and not progress.drafted,
        ),
        (
            # The third of the same family, and the last one missing. The sweep is
            # the only thing here that produces a figure, and it lives behind
            # ``consult``; a run that finishes without it has answered a question
            # about a curve with a single number. The policy already knew that --
            # :func:`policy_step` sends a curve request to ``consult`` -- but the
            # policy only decides when no model is reachable, so with a key
            # configured the rule was not enforced at all. Found in use: *plot the
            # low-lying spectrum* solved one chain, searched the notes, and replied
            # "I cannot make plots here" with the sweep tool never offered.
            "the model asked to answer without the curve the question asked for",
            proposed.action == "finish" and progress.wants_curve and not progress.tools_run,
        ),
        (
            # The one guard that makes the opening move safe to hand over. Without
            # it, letting the model decide the first action means letting it decide
            # to answer from nothing -- and an ungrounded answer about physics is the
            # failure this project exists to prevent. With it, the model still
            # chooses *how* to establish something, which is the decision worth
            # having; it simply may not choose to establish nothing.
            "the model asked to answer before anything had been established",
            proposed.action == "finish" and not progress.established,
        ),
        (
            # The second guard the opening decision needs, found by the held-out
            # suite rather than by reasoning: on *why does the gap close, and what is
            # it at L = 8?* the model chose to search, read the passages, reached
            # outside, and composed -- a fluent answer to half a question, with the
            # solver never run. Passages are material, so the guard above was
            # satisfied; what it cannot see is that a number was asked for and is
            # still owed. Retrieval is not a substitute for a computation, and the
            # corpus is forbidden from containing computed numbers at all.
            "the model asked to answer without the number the question asked for",
            proposed.action == "finish" and progress.wants_number and not progress.has_numbers,
        ),
    ):
        if refused:
            LOG.info("decide_overridden", extra={"detail": detail})
            return instead_of(proposed, progress)
    return proposed.model_copy(update={"decided_by": "model"})
