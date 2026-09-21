"""The campaign loop: how a feasibility question gets answered.

Each node is a small function taking the campaign so far and returning only what
it changed; :func:`build_graph` wires them together. No physics and no prose is
written here -- both live in modules testable without a graph around them.

The shape, in words::

    screen -> formalise -> retrieve -+-> baseline --------------------+-> skeptic -> scribe
                                     |                                |
                                     +-> plan <-> solve -> analyse ---+
                                     |                                |
                                     +-> converge --------------------+

Six things about that picture are not obvious:

- The classical baseline hangs off the fan-out rather than sitting in the loop,
  and nothing can route around it. A quantum number with no classical number
  beside it is the characteristic failure of this field, so the verdict returns
  "no" when the baseline is missing.
- Plans are priced before they are run. One that does not fit the budget is
  recorded as rejected with its reason, and the loop tries something cheaper.
- The model proposes a depth; the code clamps it to what coherence allows and
  checks it against what has been tried.
- ``converge`` is a sibling of the loop, not a stage in it. It runs only when the
  question asked to watch something converge; inside the loop it would run once
  per depth for nothing. :func:`after_converge` decides where it rejoins.
- The first three model calls go out together, so the pause before any visible
  work is one round trip rather than three. Switchable, because the sequential
  order is easier to follow in a trace. See :mod:`src.agent.prefetch`.
- A verdict is only reached about a chain somebody named. Otherwise every
  chain-less question receives the same "no: this problem has a closed-form
  solution", which is true of the default chain and a non sequitur as an answer.
  See :attr:`~src.agent.state.FormalModel.length_was_given`.

Every node appends one line to the state's running log, which is what the run
view shows. Nothing here can reach an exact answer: the solvers that know one
live behind a wall this package cannot import, so the agent can measure, compare
and bound, but not peek.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.agent import reading, spelling
from src.agent.diagnosis import coefficient_bound, diagnose_run
from src.agent.drafting import draft as draft_code
from src.agent.explaining import explain as explain_in_prose
from src.agent.followups import propose
from src.agent.grading import model_expander, model_grader, model_reranker, model_rewriter
from src.agent.intent import Reading, asks_for_literature
from src.agent.intent import read as read_intent
from src.agent.llm import ToolCall, as_data
from src.agent.mathmarkup import clip
from src.agent.memory import Memory
from src.agent.model_selection import ModelPool
from src.agent.prefetch import Prefetch
from src.agent.prompts import (
    DEPTH_SUGGESTION,
    PROBLEM_READING,
    QUESTION_CLARIFYING,
    TOOL_CONSULTATION,
)
from src.agent.report import compose
from src.agent.router import Routing, continues_earlier, in_domain, route
from src.agent.state import (
    Answer,
    BaselineResult,
    Belief,
    CampaignState,
    Citation,
    CoherenceBudget,
    Draft,
    FormalModel,
    PlannedRun,
    Request,
    RuledOut,
    RunRecord,
    SearchRound,
    new_campaign,
)
from src.agent.tools import TOOLS, FieldSweepBench, field_sweep_tool
from src.agent.tracing import configure_tracing
from src.agent.verdict import judge
from src.hardware.devices import Device
from src.hardware.fidelity import coherence_budget, estimate
from src.hardware.transpile import Transpiled, fits, transpile
from src.logging_setup import get_logger
from src.physics.lattice import MIN_SIDE, Lattice
from src.physics.model import DEFAULT_SITES, MAX_SITES_STATEVECTOR, fields_in_words
from src.physics.quantum.ansatz import (
    DEFAULT_DEPTH as DEFAULT_CIRCUIT_DEPTH,
)
from src.physics.quantum.ansatz import AnsatzSpec, interpolate_to_depth
from src.physics.quantum.circuit_algebra import circuit_mathematics, state_preparation_latex
from src.physics.quantum.hamiltonians import PauliSum, ising_chain
from src.physics.quantum.method_race import METHOD_NAMES, race_methods, unsupported_reason
from src.physics.quantum.quantum_approximate_optimisation import scan_ramp_time
from src.physics.quantum.statevector import (
    FloatArray,
    diagonal_energies,
    evolve,
    measurement_deviation,
)
from src.physics.quantum.variational_eigensolver import VqeResult, solve
from src.security import screen as screen_text
from src.settings import DEFAULT_AUDIENCE, Audience, get_settings

MEMORY_SUMMARY_CHARS = 400
"""How much of a prose answer is carried into the next question's context.

Enough to say what the answer was about and not so much that one turn fills the recall
window. A follow-up needs the subject of the conversation, not a transcript of it.
"""

RACE_DEPTH = 3
"""Circuit layers every method gets in the comparison when the question names none.

Three is deep enough that the methods have separated -- at one layer the adiabatic
start and the near-identity start are still in the same place -- and shallow enough
that all three reach the bottom in a second or two, which is what a page can wait for.
A question that names its own depth overrides this, and the caption says which of the
two supplied the number.
"""

DEPTH_LADDER: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
"""Depths to try, in order, when nothing better is suggested.

Roughly doubling, because each added layer buys less than the one before it and
walking up in single steps would spend most of the budget distinguishing depths
that differ by less than the accuracy anyone cares about. Stopping at eight is a
budget decision and not a physical limit -- by then the marginal gain on this chain
is generally smaller than the error bar the classical baseline carries.

Three is on the ladder because it is where the tolerance is crossed. On the critical
chain the error per spin runs 0.50, 0.018, 0.0086, 0.0044 at depths one to four, so
the step from two to three is the one that gets under :data:`TARGET_ENERGY_ERROR`
and a ladder jumping from two to four asked for twice the measurements to buy an
accuracy it could already afford. A geometric ladder is right about the shape of the
returns and cannot know where the threshold falls.
"""

TARGET_ENERGY_ERROR = 0.01
"""Accuracy an energy reading is priced for, in units of the coupling :math:`J`.

Shots scale as the inverse square of this, so it is the single most expensive
number in the campaign: asking for ten times the accuracy costs a hundred times as
much. One per cent of :math:`J` is chosen to be comparable with the uncertainty the
classical baseline reports, since resolving the quantum answer far more finely than
the number it is being compared against buys nothing.
"""

DEFAULT_SHOT_BUDGET = 10**9
r"""Measurements a campaign may spend when the caller does not say.

The same figure the interface ships as its default, and it is one number rather than
two on purpose. It used to be fifty million here and a billion there, and the
docstring on this side claimed the default was "generous enough that a small chain
finishes its depth ladder" -- which was false and had been for as long as the
arithmetic had stood. One energy reading to a total accuracy of
:data:`TARGET_ENERGY_ERROR` costs :math:`(\sum_\alpha |c_\alpha| / \epsilon)^2`
shots, and :math:`\sum_\alpha |c_\alpha|` grows with the chain, so fifty million
covered a six-spin chain and refused every rung of the ladder for the ten-spin chain
this application uses as its own headline example. A default that silently turns the
flagship demonstration into "nothing ran" is worse than no default.

A billion is a plausible thing to book, not a generous one: a superconducting device
sampling at roughly ten kilohertz spends it in about a day of wall-clock time.

Note what this does *not* fix. A budget large enough to run the ladder does not make
the ladder cheap, and the refusals a smaller budget produces are arithmetic rather
than pessimism. That is the finding, and raising the default puts it in the report
where a reader can see it instead of in an empty table.
"""

MAX_LISTED_SOURCES = 8
"""How many sources are listed under a prose or code answer.

Eight is more than retrieval normally keeps, so in practice the list is complete and
the cap is a guard against a future retrieval setting rather than a routine
truncation. A list that silently stopped at its cap would read as "these are the
sources", which is a different claim.
"""

EVALUATIONS_PER_LAYER = 40
"""Energy readings charged per ansatz layer when pricing a plan before it runs.

Deliberately generous, because this multiplies a budget that refuses. What the
optimiser actually asks for is now recorded beside it as
:attr:`~src.agent.state.RunRecord.energy_evaluations`, and measured on a ten-spin
critical chain the real counts are 10, 12, 25, 58 and 236 readings at depths one,
two, three, four and eight -- convex in depth, where this is linear, so the estimate
over-prices a shallow circuit about fourfold and converges by depth eight. Both
errors point the same way inside the ladder this project climbs, which is the side a
refusal should err on. Straightening the shape would need a fit, and a fit to five
points on one chain is not better than a bound that is known to be loose.
"""

MAX_PLANNING_ROUNDS = 6
"""How many times the loop may propose a configuration before it must conclude.

A hard stop, not a target. The loop's natural exits are running out of budget or
running out of depths worth trying; this exists so that a planner which keeps
proposing configurations that are rejected cannot spin, and so a campaign has a
bounded cost even when something upstream is behaving oddly.
"""

AUTO_MODEL: Literal["auto"] = "auto"
"""Ask the settings for a language model, rather than being handed one.

The default, and what production uses. The alternative is to pass a model
explicitly -- including passing ``None``, which means *run with no model at all*
and is a different instruction from "work out whether one is available". Tests
pass ``None`` for exactly that reason: a suite that reaches a gateway is a suite
whose results depend on the network, and the deterministic path is the one that
has to keep working anyway.
"""

SEARCH_KEY = "search_corpus"
"""Whether this run may search the corpus, in the graph's runtime configuration.

The second external service a campaign can reach, and it gets its own switch for
the same reason the model does: a test that searches needs an index and an
embedding endpoint, and a suite whose results depend on either is a suite that
fails for reasons that have nothing to do with the code. Absent means "search";
``False`` means "do not", and the campaign then says in its notes that its verdict
rests on measurement alone.
"""

FOLLOWUPS_KEY = "suggest_followups"
"""Whether the agent proposes what to ask next.

A dial rather than a constant because a suggestion is a model call, and somebody
metering what a campaign costs should be able to read the number without one.
"""


AUDIENCE_KEY = "audience"
"""Configuration key carrying who the prose is written for.

On the configuration channel rather than in :class:`CampaignState`, for the same
reason the retrieval dials are: it is an input to the run, not something the run
establishes, and a state field would put it into every checkpoint and every reducer
for no benefit.
"""

PRECISION_KEY = "precision_per_site"
"""Configuration key carrying the target accuracy per magnet.

Per magnet rather than per chain, because the chain length is read out of the
question and the interface does not know it in advance. The shot arithmetic
multiplies by the length it actually formalised.
"""

RETRIEVAL_ROUNDS = 2
"""Searches one question gets by default, the first included.

Restated here rather than imported so that :mod:`src.rag.retrieve` -- which pulls in
the vector store -- stays a lazy import inside the one function that searches, and a
campaign that never retrieves never pays for it. The graph test pins
this against :data:`src.rag.retrieve.MAX_ROUNDS`, so the copy cannot drift.
"""


@dataclass(frozen=True, slots=True)
class SearchTuning:
    """How hard to search, as one value the interface can hand the graph.

    These were constants, and two of them were the wrong kind of constant: how many
    passages to keep and which shelf to look on are judgements about *this* question,
    not properties of the application. A reader who wanted to know what the agent
    would say from one shelf, or from twice as much material, had no way to ask.

    Attributes:
        passages: How many passages a search keeps.
        vector_share: How much of the search is meaning-matching rather than
            keyword-matching. The keyword half is what finds a passage naming an
            author or an arXiv number, so this is not a dial to take to one.
        shelves: Which knowledge bases to search first. Empty leaves the choice to
            the router, which reads it off the question's vocabulary and widens on
            the next round -- better than a person guessing once, which is why it is
            the default.
        rounds: How many searches one question gets, the first included. The second
            round is what turns a failed search into a reformulated one, graded
            against the same bar; setting this to one is how a reader sees the
            difference the corrective loop makes rather than taking it on trust.
    """

    passages: int = 4
    vector_share: float = 0.6
    shelves: tuple[str, ...] = ()
    rounds: int = RETRIEVAL_ROUNDS


SOURCES_WANTED = 4
"""How many background sources a campaign looks for.

Enough that a claim can be supported from more than one place, few enough that
the reference list stays something a reader will actually check. It is also the
number fetched from outside when the notes are silent, so the two routes offer
the same amount of evidence and neither looks better merely for being longer.
"""

SNIPPET_CHARACTERS = 400
"""How much of a source is kept on a citation.

Cut by :func:`src.agent.mathmarkup.clip` rather than by a slice. A slice ends
mid-word, which reads as a broken file rather than a quotation, and it ends inside
``$...$`` often enough to matter -- an unmatched dollar makes a markdown renderer
swallow everything up to the next one, so a stray delimiter in a citation can hide a
paragraph of the answer above it.

Enough to see that the passage says what it is cited for. The whole passage is in
the index and can be looked up; carrying it in the campaign state would put it in
every checkpoint and every log line as well.
"""

DEFAULT_FETCH_SHELF = "quantum-computing"
"""Where an externally fetched note is filed when routing chose no shelf.

The shelf that covers running a chain on a machine, which is what a feasibility
question is about. A note filed on the wrong shelf is still found -- the search
widens after the first round -- so this is a default rather than a decision.
"""

SEARCH_TUNING_KEY = "search_tuning"
"""Where the retrieval dials travel, as one key rather than four.

One key because they move together and are read in one place. Four keys would each
need a default at the read site, and four defaults in four places is how a dial ends
up doing nothing on one path and something on another.
"""

FETCH_KEY = "fetch_external"
"""Whether this run may reach an external index, in the runtime configuration.

The third external service a campaign can touch, and it gets its own switch for
the reason the other two do: it is a network call, and a test that makes one is a
test that fails for reasons unrelated to the code.
"""

POOL_KEY = "model_pool"
"""Where a prepared set of models sits in the graph's runtime configuration.

Takes precedence over :data:`MODEL_KEY`. This is the channel a person's model
selection travels down: the interface builds a pool with the tiers it was told
to use, and the graph runs against it without any node knowing a choice was made.
"""

MODEL_KEY = "chat_model"
"""Where the language model sits in the graph's runtime configuration.

Injected through the configuration channel rather than stored in the state, because
the state is checkpointed and a live client is not something that can be written to
a database and read back.
"""


RECALL_KEY = "recalled_turns"
"""Configuration key carrying what the agent remembers of this conversation.

A rendered string rather than a memory object, because the graph does no file
access: recall happens once at the campaign boundary, and what reaches a node is
text it can put in a prompt. That keeps every node runnable with no disk, which is
why the whole graph can be tested without one.
"""

MEMORY_KEY = "memory"
"""Configuration key carrying this conversation's memory.

The memory object rather than a rendered string, now that recalling and remembering
are nodes: a node that has to write has to hold the thing that writes. It stays in
the configuration rather than the state for the same reason the model does -- state
is checkpointed, and a file handle is not something to write to a database and read
back.
"""

CLARIFY_CALL = "clarify"
INTENT_CALL = "intent"
PROPOSAL_CALL = "problem_reading"
"""The three front-of-graph model calls, named so that one node can start what another
collects.

Constants rather than literals at the six call sites, because a name typed differently in
the ``start`` and the ``take`` is not an error anywhere: the collector simply finds
nothing started, makes the call itself, and the campaign is correct and exactly as slow
as it was before. A silent loss of the entire optimisation is the failure mode worth
spending three constants to make impossible.
"""

PROSE_SINK_KEY = "prose_sink"
"""Where a caller leaves somewhere to send the explanation as it is written.

The explanation is the longest call the graph makes and the one whose output *is*
the answer, so it is the one worth streaming: a reader can be reading the first
paragraph while the last is still arriving. Passed through the configuration rather
than as a node argument because it belongs to the caller's rendering, not to the
campaign -- and a campaign with nowhere to send it behaves exactly as it did before,
which is what every test and every script relies on.
"""

PREFETCH_KEY = "prefetch"
"""Configuration key carrying the campaign's head start on its own model calls.

In the configuration rather than the state for the reason every live object here is:
a thread pool is not something to checkpoint. What it holds is not a finding either --
it is the same three answers the same three nodes would have computed, only requested
earlier. See :mod:`src.agent.prefetch` for what that does and does not change.
"""

CIRCUIT_DEPTH_KEY = "circuit_depth"
"""Configuration key: how many layers to assume when a question names none.

The interface passes the depth its own knob is set to, so the equations above a
drafted program describe the circuit drawn under it. Absent -- a script, a test, a
question asked through the MCP server -- falls back to
:data:`~src.physics.quantum.ansatz.DEFAULT_DEPTH`.
"""

BENCH_KEY = "reference_bench"
"""Configuration key carrying the exact field sweep, when a host lends one.

The seam described on :func:`src.physics.registry.field_sweep_bench`. The solver is
sealed from everything under ``src/agent/``, so it cannot be imported here and is
handed in instead: a host that has it -- the interface, an eval, a script -- puts
the callable in the configuration, and :func:`answer_toolkit` turns it into one
extra tool for the consultation node. Absent, which is what a bare
:func:`run_campaign` gets, means the agent answers curve questions from the notes
and says that it did.

In the configuration rather than the state for the same reason the device is: it is
a fact about the world the campaign runs in, not something the campaign discovers.
"""

DEVICE_KEY = "device"
"""Configuration key carrying the machine a plan is priced against.

In the configuration rather than the state for the same reason the model is: a
device is a fixed description of the world the campaign runs in, not something the
campaign discovers and updates. Absent means price the circuit in the abstract,
which is what a question about the algorithm rather than about hardware wants.
"""

DIALS_KEY = "chain_defaults"
"""Configuration key carrying the chain the caller already has on screen.

A host with visible controls for the chain -- the interface has five: length,
boundary, coupling, transverse field and longitudinal field -- puts their positions
here, and every term the question and the conversation left unsaid is taken from them
instead of from a constant. Without it the interface could show a ten-spin ring while
the answer under it read *TFIM L=6 (open)*, which is a dial that changes nothing and
a report that contradicts the picture above it.

It supplies values and never a *finding*. Whether the length was given still depends
on the words, because that is what decides if a verdict may be reached
(:attr:`~src.agent.state.FormalModel.length_was_given`), and a dial somebody set once
is not a statement about the question they typed afterwards. So a question naming no
chain is still consulted rather than assessed -- it is now consulted about the chain
on screen.

In the configuration rather than the state for the same reason the device is: it
describes the world the campaign runs in, not something the campaign discovers.
"""

DEFAULT_LATTICE_SITES = 9
"""Spins to assume for a lattice nobody sized.

Nine rather than :data:`~src.physics.model.DEFAULT_SITES`, and the difference is not
a preference. Six spins on a square is a 2x3 strip, which has one interior bond and
looks far more like a chain than like a grid; nine is 3x3, the smallest size with a
site that has four neighbours, and it is therefore the smallest size at which the
answer is about a lattice at all. It also stays well inside
:data:`~src.physics.model.MAX_SITES_STATEVECTOR`.
"""

_logger = get_logger("agent.graph")


class ModelProposal(BaseModel):
    """The Hamiltonian a language model reads out of the request.

    Deliberately small. Everything here is a number the request either states or
    does not; asking a model for judgement about the physics as well would mix a
    reading task with a modelling one, and make a wrong answer impossible to
    attribute to either.
    """

    n_sites: int = Field(description="How many interacting elements the problem describes.")
    coupling: float = Field(
        default=1.0, description="How strongly neighbours pull on each other, written J."
    )
    transverse_field: float = Field(
        default=1.0, description="The competing sideways influence, written h."
    )
    longitudinal_field: float = Field(
        default=0.0,
        description=(
            "Any bias pulling elements along the same direction their neighbours do, "
            "written g. Zero unless the request describes one."
        ),
    )
    boundary: Literal["periodic", "open"] = Field(
        default="open",
        description="'periodic' if the chain closes into a ring, 'open' if it has two ends.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Every choice made that the request did not state, one short sentence each, "
            "in plain language."
        ),
    )


class DepthProposal(BaseModel):
    """A language model's suggestion for the next configuration to try."""

    depth: int = Field(description="How many layers the next circuit should have.")
    reason: str = Field(description="One sentence on why this depth is the one worth trying next.")


# ── nodes ──────────────────────────────────────────────────────────────────────


def screen_request(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Check the incoming text for attempts to redirect the agent, and restate it.

    Two things happen here and neither replaces the request. The screen's finding is
    stored beside the text, and a blocked request still gets a campaign -- one that
    stops at the verdict with nothing formalised -- because silently discarding input
    is worse than reporting that it was refused.

    The second is a plain restatement of the question, stored in
    :attr:`~src.agent.state.Request.clarified`. It exists because a vague or elliptical
    question makes a poor search query, and because a reader is owed a view of what
    was understood before seeing what was concluded. It is **not** allowed to become
    the question: replacing the text would normalise away the framing, and measuring
    whether a verdict moves with the wording is only possible if the wording survives
    the trip.

    A blocked question is never restated. Rewriting an injection attempt produces a
    tidier injection attempt, and the tidier one is the more dangerous of the two.

    Args:
        state: The opening campaign.
        config: The graph's runtime configuration, which is where a language model is
            injected. With no model the restatement is simply absent, which is honest
            -- the campaign then searches with the original wording.

    Returns:
        The request with its screening note and, when one was made, its restatement.
    """
    request = state["request"]
    screening = screen_text(request.text)
    note = screening.explain()
    # Spelling is repaired before anything reads the words, because everything that
    # follows reads them by matching: the scope gate, the intent reader, the shelf
    # router and the search query. One mistyped noun therefore failed all four at
    # once -- "what is the phys of the quntum isng mdl?" shares no word with the
    # scope vocabulary, so it was refused as out of scope, no chain was formalised
    # and nothing was searched. See `src.agent.spelling`.
    repaired = spelling.repair(request.text)
    fixed = spelling.corrections(request.text)
    # The scope gate belongs here rather than in front of the corpus search, which is
    # where it used to be. Down there the question had already been turned into a
    # Hamiltonian, so a question that named no physics at all got a chain invented for
    # it and then a full feasibility campaign run on the invention. Asked here, on the
    # words the user actually typed, it costs one set intersection and refuses in
    # three milliseconds.
    # A follow-up inherits the conversation's scope. "And with periodic boundary
    # conditions?" carries none of this project's vocabulary and is plainly in scope
    # when it arrives after a question about a spin chain -- judging it on its own
    # words would decline exactly the questions that prove the memory layer works.
    in_scope = in_domain(repaired) or (continues_earlier(repaired) and bool(_recalled(config)))

    # Three model calls, started together. The gate above is the only thing any of them
    # has to wait for, and it is a set intersection -- so from here the restatement, the
    # intent reading and the reading of the chain's numbers are all in flight at once
    # instead of one after another across three nodes. The two nodes downstream collect
    # them; neither can tell that the call was issued here rather than there.
    pool = _pool_from(config)
    ahead = _prefetch_from(config)
    admitted = not screening.blocked and in_scope
    if admitted:
        recalled = _recalled(config)
        ahead.start(CLARIFY_CALL, lambda: _clarify(request.text, pool))
        ahead.start(
            INTENT_CALL, lambda: read_intent(request.text, pool, allow_model=not pool.offline)
        )
        ahead.start(PROPOSAL_CALL, lambda: _propose_model(request.text, pool, recalled))

    clarified = ahead.take(CLARIFY_CALL, lambda: _clarify(request.text, pool)) if admitted else ""
    # The repair is a restatement, and the honest one to show when no model wrote a
    # better one: it is the question, in the words the agent went on to act on.
    if not clarified and fixed:
        clarified = repaired
    _logger.info(
        "campaign_screen",
        extra={
            "blocked": screening.blocked,
            "categories": list(screening.categories),
            "restated": bool(clarified),
        },
    )
    notes = [f"screened the request: {note.splitlines()[0]}"]
    if not in_scope:
        notes.append(
            "the question names nothing this project covers, so no chain was invented for it"
        )
    if fixed:
        notes.append(
            "corrected the spelling before reading it: "
            + ", ".join(f"{typed} \u2192 {read}" for typed, read in fixed)
        )
    if clarified:
        notes.append(f"read the question as: {clarified}")
    return {
        "request": Request(
            text=request.text,
            framing=request.framing,
            screening_note=note,
            clarified=clarified,
            in_scope=in_scope,
        ),
        "notes": tuple(notes),
    }


class Restatement(BaseModel):
    """One plain sentence restating the question, for a search query and for a reader."""

    question: str = Field(description="The question restated in one plain sentence.")


def _clarify(text: str, pool: ModelPool) -> str:
    """Ask a model to restate the question plainly.

    Args:
        text: The question as it arrived.
        pool: The campaign's models.

    Returns:
        The restatement, or the empty string when no model answered or the
        restatement came back the same as the original. Returning nothing rather than
        a copy keeps "was restated" a meaningful thing for a trace to report.
    """
    answer = pool.invoke("query_rewrite", Restatement, QUESTION_CLARIFYING, as_data(text))
    if answer is None:
        # No model, so the only restatement available is the spelling repair -- and
        # it is the right one. What stood here before was a description of the
        # *chain* the pattern reader found, which is a fine thing to report and a
        # wrong thing to call a restatement of the question: on a question about
        # barren plateaus it read "The ground state of 10 spins.", and
        # `Request.for_search` then searched the corpus for that. The chain reading
        # is still reported, by `formalise`, where it belongs.
        return ""
    restated = answer.question.strip()
    return "" if not restated or restated == text.strip() else restated


def interpret_request(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Decide what kind of answer the question is asking for.

    A node of its own, before any work is done, because it chooses which work gets
    done -- without it a request for code or for background comes back as a verdict
    on whether quantum hardware is worth using.

    Placed after screening and before formalising, the only interval where it is both
    safe and useful. Before the screen it would read text not yet checked for an
    injected instruction; after formalising, a chain has been invented and a search
    already aimed at it.

    A blocked or out-of-scope request is not read at all. There is no branch to
    choose for a question the campaign is about to refuse.

    Args:
        state: The campaign, just after screening.
        config: The graph's runtime configuration, for the models.

    Returns:
        The reading, and a line for the trail naming what decided it.
    """
    request = state["request"]
    if request.blocked or not request.in_scope:
        return {"notes": ("did not read an intent: the request was refused at the door",)}
    pool = _pool_from(config)
    reading = _prefetch_from(config).take(
        INTENT_CALL, lambda: read_intent(request.text, pool, allow_model=not pool.offline)
    )
    _logger.info(
        "campaign_intent",
        extra={"intent": reading.intent, "by": reading.decided_by},
    )
    return {"intent": reading, "notes": (reading.explain(),)}


def formalise(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Turn the request into a specific Hamiltonian, and record what that assumed.

    The step where a feasibility study most often goes quietly wrong, because
    whatever comes out of it is treated as the problem from here on. A language
    model reads the numbers out of the request; if there is no model available, or
    it returns nothing usable, the campaign falls back to a default chain and says
    so in the assumptions. Both paths record their assumptions the same way, so a
    reader can tell what was assumed without being able to tell which path produced it.

    Args:
        state: The campaign, with the request screened.
        config: The graph's runtime configuration, which is where a language model
            is injected. Absent or empty means fall back to the settings.

    Returns:
        The formal model, or nothing if the request was refused by the screen.
    """
    request = state["request"]
    if request.blocked:
        _logger.warning("campaign_refused", extra={"reason": "screening"})
        return {"notes": ("refused the request: the input screen rejected it",)}
    if not request.in_scope:
        # Declining costs nothing and is the honest answer. Formalising anyway would
        # produce a confident feasibility assessment of a problem the asker never
        # mentioned, which is worse than saying no in every way that matters.
        _logger.info("campaign_refused", extra={"reason": "out_of_scope"})
        return {
            "notes": (
                "declined the request: it is not a question about a lattice of "
                "interacting spins, which is the only thing this assessment covers",
            )
        }

    recalled = _recalled(config)
    # The conversation is read too, and this question is laid over it. Without that a
    # follow-up naming only a boundary would be answered about a default chain rather
    # than about the one under discussion -- which is the offline half of what the
    # memory layer is for.
    read_by_pattern = reading.read(request.text).under(reading.read(recalled))

    # A length that was named and cannot be served is refused here, by name. Found by
    # running `make live-check`: *is a quantum computer worth using for a chain of 200
    # magnets?* used to come back as a verdict about a **six**-element chain, with the
    # assumption "the request named no chain" printed underneath it -- because the
    # reader returned `None` both for a sentence that named no length and for one that
    # named 200, and the fallback could not tell the two apart. The verdict was
    # internally consistent, fluent, and about a problem thirty-three times smaller
    # than the one asked about, which is the exact failure mode this project exists to
    # argue against.
    if read_by_pattern.named_too_long:
        too_long = read_by_pattern.too_long
        _logger.info("campaign_refused", extra={"reason": "chain_too_long", "n_sites": too_long})
        return {
            "notes": (
                f"declined the request: it names a chain of {too_long}, and the "
                f"longest this can price is {reading.MAX_SITES}. Nothing was assumed "
                "in its place -- answering about a shorter chain would be a verdict "
                "about a problem nobody asked about",
            ),
            "beliefs": (
                Belief(
                    claim=(
                        f"a chain of {too_long} elements is past every method and "
                        f"every device modelled here, which stop at "
                        f"{reading.MAX_SITES} elements"
                    ),
                    support=(
                        "the state vector these solvers carry doubles with every "
                        "element added, so the memory alone rules it out",
                        "no device in this project's catalogue has that many qubits, "
                        "so there is nothing to place the circuit on either",
                    ),
                ),
            ),
        }

    pool = _pool_from(config)
    proposal = _prefetch_from(config).take(
        PROPOSAL_CALL, lambda: _propose_model(request.text, pool, recalled)
    )
    model = _clamp_model(proposal, read_by_pattern, _dials_from(config))
    _logger.info(
        "campaign_formalise",
        extra={"model": model.label(), "read_by_pattern": read_by_pattern.found_anything},
    )
    notes = [f"read the request as {model.label()}"]
    if proposal is None and read_by_pattern.found_anything:
        notes.append(
            "read it by pattern, with no language model: " + ", ".join(read_by_pattern.describe())
        )
    if recalled:
        # Said out loud because a reading that only makes sense given an earlier
        # question is a reading the reader cannot check without being told there
        # was one.
        notes.append("read it in the light of what was asked earlier in this conversation")
    return {"model": model, "notes": tuple(notes)}


def _lattice_for(model: FormalModel) -> Lattice | None:
    """The shape to hand a solver, or ``None`` when the problem is a chain.

    ``None`` rather than ``Lattice("chain", ...)`` because that is what every solver's
    default already means, and a chain that travelled as an explicit object would take
    a different code path through the physics layer than the one the whole test suite
    exercises. One shape, one path.

    Args:
        model: The formalised problem.

    Returns:
        The lattice, or ``None`` for a chain.
    """
    return None if model.geometry == "chain" else model.lattice


def _with_dials(
    asked: reading.Reading, dials: reading.Reading | None
) -> tuple[reading.Reading, tuple[str, ...]]:
    """Fill what the request left unsaid from the controls the caller has on screen.

    Args:
        asked: What the request and the conversation before it yielded.
        dials: The chain on screen, or ``None`` when the caller has no controls --
            every script and every test, which keeps the project defaults.

    Returns:
        The reading to build the model from, and one assumption naming every term
        the dials supplied. One line rather than one per term: what a reader needs
        to know is that the panel was read, not to have it read back to them a
        number at a time.
    """
    if dials is None:
        return asked, ()
    # The two strengths are left alone when the sentence fixed their *ratio*. "At the
    # critical point" means h equals J and the reader gets h back from the pattern
    # reader with no J beside it; taking J off the panel would then answer about a
    # chain away from the transition, which is the one thing that sentence ruled out.
    ratio_is_fixed = asked.critical or asked.strengths_from_ratio
    supplied = reading.Reading(
        n_sites=None if asked.n_sites is not None else dials.n_sites,
        boundary=None if asked.boundary is not None else dials.boundary,
        coupling=None if ratio_is_fixed or asked.coupling is not None else dials.coupling,
        field=None if ratio_is_fixed or asked.field is not None else dials.field,
        longitudinal=None if asked.longitudinal is not None else dials.longitudinal,
    )
    described = supplied.describe()
    if not described:
        return asked, ()
    return asked.under(supplied), (
        "the request did not give the whole chain, so the settings on screen "
        "supplied the rest: " + ", ".join(described),
    )


def _from_pattern(found: reading.Reading, dials: reading.Reading | None = None) -> FormalModel:
    """Build a model from what could be read without a language model.

    Every field the sentence was silent about is recorded as an assumption rather
    than filled in quietly. That is the whole difference between a default and a
    finding, and a reader who cannot tell them apart cannot judge the verdict.

    Args:
        found: What the sentence yielded.
        dials: The chain the caller has on screen, which fills what the sentence
            left unsaid. See :data:`DIALS_KEY`.

    Returns:
        The model, with one assumption per thing that had to be supplied.
    """
    filled, dial_notes = _with_dials(found, dials)
    assumptions: list[str] = []
    if not found.found_anything:
        assumptions.append(
            "no language model was available and the request named no chain, so the "
            "chain set on screen was assessed"
            if dial_notes
            else "no language model was available and the request named no chain, so a "
            "six-element chain with equal neighbour and field strengths was assumed "
            "as a worked default"
        )
    else:
        assumptions.append(
            "no language model was available, so the request was read by pattern: "
            + ", ".join(found.describe())
        )
    assumptions.extend(dial_notes)
    # Every note below fires on what is *still* missing after the dials, so a term
    # the panel supplied is reported once, by the line above, rather than twice with
    # two different values in it.
    #
    # The length is not among them: `_shape_from` says the same thing, and this
    # function used to say it as well -- so a question naming a field and no length
    # printed "the request named no chain length, so six elements were assumed"
    # twice, and a question naming a square lattice and no size printed it once
    # beside "so a 3x3 one was assumed", which is two different sizes in one list.
    if filled.boundary is None:
        assumptions.append(
            "the request did not say whether the chain is a line or a ring, so a line "
            "was assumed -- which is what hardware looks like"
        )
    if filled.field is None:
        assumptions.append(
            "the request gave no field strength, so it was taken equal to the coupling, "
            "which is the hardest case and therefore the cautious one"
        )
    coupling, coupling_note = _usable_coupling(filled.coupling)
    if coupling_note is not None:
        assumptions.append(coupling_note)
    n_sites, rows, shape_notes = _shape_from(filled)
    assumptions.extend(shape_notes)
    return FormalModel(
        n_sites=n_sites,
        coupling=coupling,
        transverse_field=filled.field if filled.field is not None else 1.0,
        longitudinal_field=filled.longitudinal or 0.0,
        boundary=filled.boundary or "open",
        geometry=filled.geometry or "chain",
        rows=rows,
        assumptions=tuple(assumptions),
        # From the sentence, never from the panel. This is what decides whether a
        # verdict may be reached at all, and a dial somebody set once is not a
        # statement about the question they typed afterwards.
        length_was_given=found.n_sites is not None,
    )


def _shape_from(found: reading.Reading) -> tuple[int, int, list[str]]:
    """Settle how many spins there are and how they are arranged.

    Three cases, and the middle one is why this is a function rather than two
    expressions. A sentence can name a shape *and* a size (``4x4``), a shape and a
    count but no arrangement (*a square lattice of 16 spins*), or no shape at all.
    Only the last is a chain, and the middle one requires a choice -- 16 spins on a
    square is 4x4 or 2x8, which are different problems -- so the choice is made
    here, once, and written down where a reader can reject it.

    Args:
        found: What the sentence yielded.

    Returns:
        ``(n_sites, rows, assumptions)``. The site count may be adjusted downwards
        from what was asked, because a lattice has to be a rectangle and an
        arrangement is only possible for a count that factorises.
    """
    notes: list[str] = []
    if found.geometry is None or found.geometry == "chain":
        if found.n_sites is None and found.found_anything:
            notes.append("the request named no chain length, so six elements were assumed")
        return found.n_sites or DEFAULT_SITES, 1, notes

    if found.rows is not None:
        stated = found.n_sites or DEFAULT_SITES
        if stated % found.rows == 0:
            return stated, found.rows, notes
        # The rows came from the sentence and the count did not survive it. A
        # question naming a 6x6 lattice of 36 spins is read correctly as 36 in 6
        # rows, and then `_clamp_model` lowers the count to the largest circuit
        # this project can simulate -- leaving six rows over sixteen spins, which
        # is not a rectangle. `FormalModel.lattice` refuses it, and it did so from
        # inside the f-string of the baseline node's *explanation* that a
        # two-dimensional lattice has no classical baseline yet, so the campaign
        # died while saying why it could not finish.
        #
        # The row count is what gets dropped rather than the site count, because
        # the site count has already been clamped for a stated reason and a second
        # adjustment on top of it would answer about a size nobody chose. Falling
        # through re-derives the arrangement from the count that survived, which is
        # the branch below and already handles the awkward cases.
        notes.append(
            f"the request implied {found.rows} rows, which do not divide the "
            f"{stated} elements this assessment can carry into a rectangle, so the "
            "arrangement was chosen from the element count instead"
        )
        found = found.without_rows()

    # A shape with no arrangement. The most nearly square factorisation is chosen,
    # because it is the arrangement the word "lattice" is nearly always used for and
    # because the alternative -- a 2 x 8 strip -- is a chain in all but name, so
    # guessing it would quietly answer the easier question.
    count = found.n_sites or DEFAULT_LATTICE_SITES
    rows = max(
        (side for side in range(MIN_SIDE, int(count**0.5) + 1) if count % side == 0),
        default=1,
    )
    if rows == 1:
        # A prime count cannot be a rectangle at all. Moving to the nearest count that
        # can is honest as long as it is said; refusing outright would turn a
        # reasonable question into an error over an arithmetic accident.
        #
        # Downwards where there is anywhere to go, and upwards from two and three,
        # where there is not: `_factorises` is false below four, so the search below
        # `count` comes back empty for exactly those two sizes and this used to raise
        # `max() arg is an empty sequence` out of the formalise node -- taking the
        # whole campaign with it, since nothing between here and the interface
        # catches it. *Is a 3-spin square lattice worth it?* is a fair question with
        # an arithmetic answer: three spins make no rectangle, four make the only
        # small one there is.
        adjusted = max(
            (side for side in range(MIN_SIDE, count + 1) if _factorises(side)),
            default=SMALLEST_RECTANGLE,
        )
        moved = "so a" if adjusted < count else "so the smallest one that can, a"
        notes.append(
            f"{count} spins cannot be arranged as a rectangle, {moved} "
            f"{adjusted}-spin lattice was assessed instead"
        )
        count = adjusted
        rows = max(side for side in range(MIN_SIDE, int(count**0.5) + 1) if count % side == 0)
    if found.n_sites is None:
        notes.append(
            f"the request named a {found.geometry} lattice but no size, so a "
            f"{rows}x{count // rows} one was assumed"
        )
    else:
        notes.append(
            f"the request named {count} spins on a {found.geometry} lattice but not how "
            f"they are arranged, so a {rows}x{count // rows} grid was assumed -- a "
            f"{1}x{count} strip would be a chain wearing another shape's name"
        )
    return count, rows, notes


SMALLEST_RECTANGLE = MIN_SIDE * MIN_SIDE
"""Fewest spins that can be laid out as a rectangle with both sides above one.

Four, and it is written as the product because that is where it comes from: a
two-dimensional lattice needs :data:`src.physics.lattice.MIN_SIDE` in each
direction. A question naming two or three spins on a square lattice has named a
shape that does not exist at that size, and this is what it is moved to.
"""


def _factorises(count: int) -> bool:
    """Whether a site count can be laid out as a rectangle with both sides above one.

    Args:
        count: How many spins.

    Returns:
        True when a rectangle exists, which is false only for primes and for counts
        below four.
    """
    return any(count % side == 0 for side in range(MIN_SIDE, int(count**0.5) + 1))


def retrieve_background(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Look up prior work on this kind of problem, and keep what can be cited.

    Background is not decoration here. A feasibility verdict that cites nothing is
    one nobody can follow up, and the retrieval layer is what lets the report point
    at where a claim came from. A search that finds nothing is not an error -- the
    campaign continues, with a note saying the claim rests on its own measurements
    alone.

    Args:
        state: The campaign, with a model if one was formalised.
        config: The graph's runtime configuration, which is where the search can be
            switched off. Absent means search.

    Returns:
        Whatever sources were found, reduced to what a citation needs.
    """
    model = state["model"]
    if model is None:
        return {"notes": ("skipped the background search: no problem to search for",)}
    if not _search_allowed(config):
        return {"notes": ("background search was switched off for this run",)}

    pool = _pool_from(config)
    query = _background_query(model, state["request"], state["intent"].intent)
    routing = route(query, pool, allow_model=not pool.offline)
    citations, rounds = _search(query, routing, pool, _tuning_from(config))

    # A request for papers reaches arXiv even when the notes answered, because the
    # notes cannot answer it: the corpus is a fixed set of abstracts, so "what is
    # the recent work on this" gets four passages from whenever the corpus was last
    # built and no indication that this is what happened. Every other question keeps
    # the old rule and touches the network only when the corpus is silent.
    wants_papers = asks_for_literature(state["request"].text)
    if wants_papers and _fetch_allowed(config):
        outside = _fetch_background(query, routing, config)
        if outside:
            _logger.info(
                "campaign_retrieve_literature",
                extra={"corpus": len(citations), "fetched": len(outside)},
            )
            return {
                "citations": citations + outside,
                "searches": rounds,
                "notes": (
                    f"the question asked for papers, so arXiv was searched as well as "
                    f"the notes: {len(citations)} from the corpus and {len(outside)} "
                    "fetched, the fetched ones cited as unreviewed",
                    _searched_for(rounds),
                ),
            }

    if citations:
        _logger.info(
            "campaign_retrieve",
            extra={"found": len(citations), "shelves": list(routing.shelves)},
        )
        return {
            "citations": citations,
            "searches": rounds,
            "notes": (
                f"searched {routing.reason}; found {len(citations)} background sources"
                + ("; arXiv was searched too and returned nothing usable" if wants_papers else ""),
                _searched_for(rounds),
            ),
        }

    fetched = _fetch_background(query, routing, config)
    if fetched:
        _logger.info("campaign_retrieve_external", extra={"found": len(fetched)})
        return {
            "citations": fetched,
            "searches": rounds,
            "notes": (
                f"the notes held nothing, so {len(fetched)} sources were fetched from an "
                "external index; they are cited as unreviewed",
                _searched_for(rounds),
            ),
        }
    return {
        "searches": rounds,
        "notes": (
            "background search found nothing; the verdict rests on measurement",
            _searched_for(rounds),
        ),
    }


def _searched_for(rounds: tuple[SearchRound, ...]) -> str:
    """Write the search trail as one line of the campaign's own trail.

    The trail is what decides whether an answer was earned, and until now it said
    *how many* sources a search found and never *what it searched
    for*. Those are different claims, and only the second one lets a reader tell a
    corpus that is silent on the subject from a query that never mentioned it.

    Args:
        rounds: Every search that ran, in order.

    Returns:
        One sentence naming each query and what it kept, with the rewrite
        attributed. Never empty: "the index could not be opened" and "the search
        was run and kept nothing" look identical on a page that prints neither.
    """
    if not rounds:
        return "no search was run: the notes could not be opened"
    parts = [
        f"{'rewrote to' if item.kind == 'rewrite' else 'searched'} {item.query!r} "
        f"in {item.where()} -- {item.found} found, {item.kept} kept"
        + (f", query written by {item.rewritten_by}" if item.rewritten_by else "")
        for item in rounds
    ]
    return "; ".join(parts)


def run_classical_baseline(state: CampaignState) -> dict[str, Any]:
    """Run the classical method, unconditionally, whatever the quantum arm is doing.

    This node is on its own branch precisely so that nothing can decide to skip it.
    It is also the only number in the campaign that comes with a genuine statistical
    error bar, which is what makes the eventual comparison a comparison rather than
    two figures side by side.

    Args:
        state: The campaign, with a model.

    Returns:
        The classical result, or a note explaining why none could be produced.
    """
    model = state["model"]
    if model is None:
        return {"notes": ("skipped the classical baseline: no problem to solve",)}

    if model.geometry != "chain":
        # Refused, not approximated. The classical method maps the chain onto a
        # two-dimensional classical lattice -- one row of spins times imaginary time
        # -- and a *two*-dimensional quantum problem would need a three-dimensional
        # one, which `src.physics.classical.dual_lattice` does not implement. Running
        # the chain sampler anyway would return a number for a different problem, and
        # the reader would compare the quantum lattice result against it.
        #
        # Leaving the baseline as `None` is what makes the refusal safe: the
        # `no_classical_comparison` screen then fires and no comparative verdict can
        # be reached. That is the honest outcome and it is the point of the branch.
        reason = (
            f"there is no classical baseline for a {model.lattice.describe()} yet -- the "
            "sampler implements a chain in imaginary time, and a two-dimensional "
            "lattice needs one more dimension than it has"
        )
        _logger.warning("campaign_baseline_unavailable", extra={"reason": reason})
        return {
            "notes": (f"no classical baseline: {reason}",),
            "beliefs": (
                Belief(
                    claim=reason,
                    support=(
                        "the quantum arm ran on the lattice that was asked about, so "
                        "its energy is a real number for the real problem; what is "
                        "missing is something to compare it against, and a verdict "
                        "without one would be a quantum number on its own",
                    ),
                ),
            ),
        }

    from src.physics.classical.regime import BaselineUnavailableError
    from src.physics.classical.variational_imaginary_time import ground_state_energy
    from src.physics.model import TFIMSpec

    spec = TFIMSpec(
        n_sites=model.n_sites,
        coupling=model.coupling,
        field=model.transverse_field,
        boundary=model.boundary,
    )
    try:
        result = ground_state_energy(
            spec,
            depth=2,
            longitudinal_field=model.longitudinal_field,
            seed=0,
        )
    except BaselineUnavailableError as refused:
        # Left as `None` on purpose, which is what makes the refusal safe. The
        # `no_classical_comparison` screen then fires and the campaign cannot
        # reach a comparison at all -- whereas storing a `BaselineResult` with a
        # caveat attached would put a number in front of a reader who would
        # reasonably compare against it.
        _logger.warning("campaign_baseline_unavailable", extra={"reason": refused.regime.reason})
        return {
            "notes": (f"no classical baseline: {refused.regime.reason}",),
            "beliefs": (
                Belief(
                    claim=refused.regime.reason,
                    support=(refused.regime.detail,),
                ),
            ),
        }
    baseline = BaselineResult(
        method="variational_imaginary_time",
        energy_per_site=result.energy,
        energy_error=result.energy_error,
        n_measurements=result.n_measurements,
        confidence=result.regime.confidence,
        caveat="" if result.regime.is_trustworthy else result.regime.detail,
    )
    _logger.info(
        "campaign_baseline",
        extra={"energy_per_site": baseline.energy_per_site, "error": baseline.energy_error},
    )
    return {
        "classical": baseline,
        "beliefs": (
            Belief(
                claim=(
                    f"an ordinary computer reaches {baseline.energy_per_site:.6f} per spin "
                    f"on this problem, give or take {baseline.energy_error:.6f}"
                ),
                support=(f"variational_imaginary_time over {baseline.n_measurements:,} samples",),
            ),
        ),
        "notes": (
            f"classical baseline: {baseline.energy_per_site:.6f} "
            f"± {baseline.energy_error:.6f} per spin",
        ),
    }


def converge_methods(state: CampaignState) -> dict[str, Any]:
    """Race the three near-term methods on this campaign's own chain and keep the curves.

    Reached only when the question asked to *watch* something converge rather than to be
    told where it ended -- see :func:`src.agent.reading.asks_for_a_curve`. Three methods
    on a twelve-spin chain is a second or two of real compute, which is cheap enough to
    do on request and far too expensive to do on every question as a matter of course.

    The chain is this campaign's rather than a stock one. All three methods run at
    whatever :math:`J`, :math:`h` and :math:`g` the question was formalised into, so
    the curve underneath the answer belongs to the problem that was asked about. A
    figure drawn from a fixed example would be a decoration; this is evidence.

    Every energy here is an exact expectation value from a simulated state vector, so
    the curves compare the methods on a perfect device and say nothing about noise.
    The belief this node records says as much in its own words: a comparison that
    quietly drops its conditions is how a feasibility study reaches a confident wrong
    verdict.

    Args:
        state: The campaign, with a model.

    Returns:
        The race, a belief stating what it showed, and a note -- or a note alone
        explaining why no race was possible, which is a finding rather than a failure.
    """
    model = state["model"]
    if model is None:
        return {"notes": ("skipped the method race: no problem to solve",)}

    depth = reading.read_depth(state["request"].text) or RACE_DEPTH
    reason = unsupported_reason(model.n_sites, depth)
    if reason is not None:
        return {"notes": (f"could not race the methods: {reason}",)}

    race = race_methods(
        n_sites=model.n_sites,
        depth=depth,
        coupling=model.coupling,
        transverse_field=model.transverse_field,
        longitudinal_field=model.longitudinal_field,
        boundary=model.boundary,
        lattice=_lattice_for(model),
    )
    winner = race.best()
    _logger.info(
        "campaign_method_race",
        extra={
            "chain": race.label(),
            "winner": None if winner is None else winner.method,
            "steps": {run.method: run.n_steps for run in race.runs},
        },
    )
    if winner is None:
        return {"race": race, "notes": ("raced no methods: none were named",)}

    # Two claims, and they are different claims. Where they *land* is a statement about
    # the circuit, which all three share, so agreement there is expected and a
    # disagreement is the interesting case. How many steps it took is a statement about
    # the method, and it is the only one of the two that a shot budget cares about.
    spread = max(run.energy for run in race.runs) - min(run.energy for run in race.runs)
    settled = "reach the same energy" if spread < 1e-6 else "do not all reach the same energy"
    return {
        "race": race,
        "beliefs": (
            Belief(
                claim=(
                    f"on this chain at depth {depth} the three methods {settled}; "
                    f"{winner.method} got lowest, in {winner.n_steps} steps and "
                    f"{winner.n_energy_evaluations} energy evaluations"
                ),
                support=(
                    f"method_race over {', '.join(METHOD_NAMES)} on {race.label()}, "
                    "exact state-vector energies with no shot noise and no gate error",
                ),
            ),
        ),
        "notes": (
            f"raced {len(race.runs)} methods at depth {depth}: "
            + ", ".join(
                f"{run.method} {run.energy_per_site:.6f} per spin in {run.n_steps} steps"
                for run in race.runs
            ),
        ),
    }


def plan_configuration(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Choose the next configuration to try, price it, and accept or reject it.

    Three things happen here and the order matters. A depth is proposed, by a
    language model where one is available and by the depth ladder otherwise. That
    depth is then priced -- circuit depth from the layer arithmetic, measurements
    from the accuracy wanted -- which costs a handful of multiplications and no
    execution. Only then is it checked against the two budgets.

    A configuration that fails either check is recorded as ruled out, with the
    budget's own explanation of why, and the loop gets another turn to propose
    something cheaper. Nothing is spent on finding this out.

    Args:
        state: The campaign so far.
        config: The graph's runtime configuration, which is where a language model
            is injected. Absent or empty means fall back to the settings.

    Returns:
        A priced plan ready to run, or a rejection and no plan.
    """
    model = state["model"]
    if model is None:
        return {"pending": None, "notes": ("cannot plan: no problem was formalised",)}

    depth = _next_depth(state, _pool_from(config))
    if depth is None:
        return {"pending": None, "notes": ("no untried depth left worth proposing",)}

    ansatz = AnsatzSpec(
        n_qubits=model.n_sites,
        depth=depth,
        boundary=model.boundary,
        family="hva",
        longitudinal=model.longitudinal_field != 0.0,
        lattice=_lattice_for(model),
    )
    label = f"hva-p{depth}"

    two_qubit_depth, routing_note = _depth_on_device(ansatz, _device_from(config))
    if routing_note is not None and two_qubit_depth < 0:
        _logger.info("campaign_plan_rejected", extra={"label": label, "reason": "layout"})
        return {
            "pending": None,
            "ruled_out": (RuledOut(label=label, reason=routing_note),),
            "notes": (f"rejected {label} before running: {routing_note}",),
        }

    compiled = _compiled_for(ansatz, _device_from(config))
    inflation = 1.0 if compiled is None else estimate(compiled).shot_inflation()
    if inflation == math.inf:
        exhausted = (
            f"no signal survives {depth} layers on this machine, so no number of "
            "measurements recovers the energy"
        )
        _logger.info("campaign_plan_rejected", extra={"label": label, "reason": "fidelity"})
        return {
            "pending": None,
            "ruled_out": (RuledOut(label=label, reason=exhausted),),
            "notes": (f"rejected {label} before running: {exhausted}",),
        }
    shots = _price_shots(state, depth, _target_error_for(state, config), inflation)

    budget = state["coherence"]
    too_deep = (
        budget.refusal(two_qubit_depth)
        if compiled is None
        else budget.duration_refusal(compiled.duration_ns)
    )
    if too_deep is not None:
        _logger.info("campaign_plan_rejected", extra={"label": label, "reason": "coherence"})
        # The routing note travels with this refusal because it is usually its cause:
        # a plan rejected for duration on a machine that had to route wants to say
        # the wiring did it, not that the algorithm asked for too many layers.
        #
        # And it travels in the *reason*, not only in the trail. The trail is a page
        # away; the reason is what the answer prints under "what it refused", and on
        # a machine wired in a line a ring is refused at every depth and every size.
        # A reader given only the duration reads that as the algorithm asking for too
        # much, tries a shallower circuit, and is refused again with the same
        # sentence -- while the number that explains it sits on another page.
        rejected = [f"rejected {label} before running: {too_deep}"]
        if routing_note is not None:
            rejected.append(routing_note)
        return {
            "pending": None,
            "ruled_out": (
                RuledOut(
                    label=label,
                    reason=too_deep if routing_note is None else f"{too_deep}. {routing_note}",
                ),
            ),
            "notes": tuple(rejected),
        }

    too_costly = state["shots"].refusal(shots)
    if too_costly is not None:
        _logger.info("campaign_plan_rejected", extra={"label": label, "reason": "shots"})
        return {
            "pending": None,
            "ruled_out": (RuledOut(label=label, reason=too_costly),),
            "notes": (f"rejected {label} before running: {too_costly}",),
        }

    plan = PlannedRun(
        label=label,
        family="hva",
        depth=depth,
        two_qubit_depth=two_qubit_depth,
        shots=shots,
        reason=_plan_reason(state, depth),
    )
    _logger.info(
        "campaign_plan",
        extra={
            "label": label,
            "depth": depth,
            "shots": shots,
            "two_qubit_depth": two_qubit_depth,
        },
    )
    planned = f"planned {label}: {two_qubit_depth} two-qubit layers, {shots:,} shots"
    notes = [f"{planned} -- {plan.reason}"]
    if routing_note is not None:
        notes.append(routing_note)
    return {"pending": plan, "notes": tuple(notes)}


def _depth_on_device(ansatz: AnsatzSpec, device: Device | None) -> tuple[int, str | None]:
    """Price one circuit's depth against a machine, or in the abstract.

    The abstract depth is what the ansatz costs if every pair of spins can interact
    directly. On a real machine some of them cannot, and the difference is paid in
    SWAPs that do no physics. Pricing the plan against the machine rather than
    against the algorithm is what lets the coherence budget refuse a configuration
    for the right reason -- and lets the trail say that the reason was the wiring
    rather than the depth.

    Args:
        ansatz: The circuit specification.
        device: The machine, or ``None`` to price it in the abstract.

    Returns:
        The two-qubit depth and a note for the trail, or ``-1`` and a refusal when
        the chain does not fit on the machine at all.
    """
    if device is None:
        return ansatz.two_qubit_depth, None
    refusal = fits(ansatz.n_qubits, device)
    if refusal is not None:
        return -1, refusal
    compiled = transpile(ansatz, device)
    if compiled.swaps == 0:
        return compiled.two_qubit_depth, None
    return compiled.two_qubit_depth, (
        f"on {device.name} the wiring does not carry every interaction directly: "
        f"{compiled.swaps} routing moves per circuit take the depth from "
        f"{ansatz.two_qubit_depth} to {compiled.two_qubit_depth}"
    )


def _compiled_for(ansatz: AnsatzSpec, device: Device | None) -> Transpiled | None:
    """The circuit as one machine would receive it, placed, routed and scheduled.

    A plan that has this knows the two things the abstract circuit cannot say: how
    long the whole schedule occupies the machine, and how much signal survives it.
    The compile is memoised, so asking for it beside :func:`_depth_on_device` costs
    a dictionary lookup.

    Args:
        ansatz: The circuit specification.
        device: The machine, or ``None`` when the campaign is reasoning about the
            algorithm rather than about hardware.

    Returns:
        The compiled circuit, or ``None`` when there is no machine to compile for or
        the register is too small to hold the problem.
    """
    if device is None or fits(ansatz.n_qubits, device) is not None:
        return None
    return transpile(ansatz, device)


def run_configuration(state: CampaignState) -> dict[str, Any]:
    """Run the planned configuration, charge it to the budget, and judge the result.

    The energy comes from an exact simulation of the circuit, so no measurements are
    literally taken. The budget is still charged what the run would have cost on
    hardware, because the question being answered is whether this is affordable on a
    device -- and a campaign that simulated for free and reported the result as
    affordable would answer a different question than the one asked. The readings are
    then re-priced against the state the run actually prepared, so the report can say
    how loose the worst case the plan was approved on turned out to be.

    The starting schedule is searched rather than assumed. Every angle in this circuit
    family is fixed by one number, the time the schedule is stretched over, so the
    optimiser's starting point comes from scanning that number instead of from a
    default. It is worth the scan: on the same circuit with the same optimiser, two
    starting schedules land in different basins and return energies differing in the
    first significant figure, and a run that skipped it would report whichever basin
    the default fell into with nothing in the result to show a better one existed.

    Args:
        state: The campaign, with a plan pending.

    Returns:
        The run record, the debited ledger, and the plan cleared.
    """
    model = state["model"]
    plan = state["pending"]
    if model is None or plan is None:
        return {"pending": None}

    warm_start = _warm_start_for(state, plan.depth)
    if warm_start is None:
        # Nothing has run yet, so the basin has to be found rather than inherited.
        ramp_time, result = scan_ramp_time(
            n_sites=model.n_sites,
            depth=plan.depth,
            coupling=model.coupling,
            transverse_field=model.transverse_field,
            longitudinal_field=model.longitudinal_field,
            boundary=model.boundary,
            lattice=_lattice_for(model),
        )
    else:
        ramp_time = 0.0
        result = solve(
            n_sites=model.n_sites,
            depth=plan.depth,
            coupling=model.coupling,
            transverse_field=model.transverse_field,
            longitudinal_field=model.longitudinal_field,
            boundary=model.boundary,
            family="hva",
            initial_parameters=warm_start,
            lattice=_lattice_for(model),
        )
    bound = coefficient_bound(
        n_sites=model.n_sites,
        coupling=model.coupling,
        transverse_field=model.transverse_field,
        longitudinal_field=model.longitudinal_field,
        boundary=model.boundary,
        lattice=_lattice_for(model),
    )
    diagnosis = diagnose_run(result, bound)
    record = RunRecord(
        label=plan.label,
        family=plan.family,
        depth=plan.depth,
        two_qubit_depth=plan.two_qubit_depth,
        energy=result.energy,
        energy_per_site=result.energy_per_site,
        shots_spent=plan.shots,
        diagnosis=diagnosis,
        energy_evaluations=result.n_energy_evaluations,
        shots_at_true_variance=_shots_at_true_variance(model, result, plan.shots),
        parameters=tuple(float(angle) for angle in result.parameters),
        energy_history=result.energy_history,
    )
    _logger.info(
        "campaign_run",
        extra={
            "label": plan.label,
            "energy_per_site": record.energy_per_site,
            "signal": diagnosis.signal,
            "evaluations": result.n_energy_evaluations,
            "ramp_time": ramp_time,
        },
    )
    return {
        "runs": (record,),
        "shots": state["shots"].debit(plan.shots),
        "pending": None,
        "notes": (
            f"ran {plan.label}: {record.energy_per_site:.6f} per spin, "
            f"{diagnosis.signal.replace('_', ' ')}",
            f"charged {plan.shots:,} measurements against the worst case; the state it "
            f"prepared would have resolved the same energy in "
            f"{record.shots_at_true_variance:,}",
        ),
    }


def analyse_runs(state: CampaignState) -> dict[str, Any]:
    """Turn the latest run into a belief, and note whether another is worth doing.

    Deciding whether to continue is left to the edge that follows; what happens
    here is the recording of what was learnt, so that a campaign which stops has
    stated its reasons rather than simply having run out.

    Args:
        state: The campaign, with at least one run.

    Returns:
        A belief drawn from the most recent run.
    """
    runs = state["runs"]
    if not runs:
        return {"notes": ("nothing to analyse: no run completed",)}

    latest = runs[-1]
    gain = _marginal_gain(state)
    claim = (
        f"{latest.label} reaches {latest.energy_per_site:.6f} per spin at "
        f"{latest.two_qubit_depth} two-qubit layers"
    )
    if gain is not None:
        claim += f", an improvement of {gain:.6f} over the previous depth"

    _logger.info(
        "campaign_analyse",
        extra={"label": latest.label, "signal": latest.diagnosis.signal, "gain": gain},
    )
    return {
        "beliefs": (Belief(claim=claim, support=(latest.diagnosis.evidence,)),),
        "notes": (f"analysed {latest.label}: {latest.diagnosis.evidence}",),
    }


def apply_skeptic(state: CampaignState) -> dict[str, Any]:
    """Reach the verdict by rule, and record which rule reached it.

    The call is made by the screens and not here, and not by a language model. This
    node's job is to run them and to write the outcome into the state, which keeps
    the decision in one testable place and makes removing the skeptic a matter of
    routing around this node rather than of editing what the verdict means.

    Args:
        state: The finished campaign.

    Returns:
        The verdict and a note naming the rule behind it, or the note alone. A
        campaign that measured a chain the question never named reaches no verdict:
        at ``g = 0`` every unnamed chain gets the same "no, this has a closed-form
        solution", which is true of the default and an answer to nobody's question.
        What it measured is still reported.
    """
    model = state["model"]
    if model is not None and not model.length_was_given:
        _logger.info("campaign_verdict_withheld", extra={"reason": "no_chain_named"})
        return {
            "notes": (
                "no verdict: the question named no chain, so the runs describe the "
                f"{model.label()} assumed for them and not a problem anybody posed",
            ),
        }
    verdict, decided_by = judge(state)
    _logger.info(
        "campaign_verdict",
        extra={"call": verdict.call, "decided_by": decided_by, "framing": state["request"].framing},
    )
    return {
        "verdict": verdict,
        "notes": (f"verdict: {verdict.call}, decided by the {decided_by} rule",),
    }


def _chain_the_question_named(state: CampaignState) -> FormalModel | None:
    """The chain, but only when the question actually named one.

    ``formalise`` always produces a model, filling anything the question left out
    from defaults and recording each fill as an assumption. That is right for the
    feasibility branch, where a stated default is better than a refusal and the
    assumptions are printed above the verdict.

    It is wrong for the other two. Asked *what is a barren plateau*, the reader named
    no chain, formalising invented a two-spin chain with the transverse field at
    zero -- which is not a transverse-field Ising chain at all -- and the explanation
    came back describing barren plateaus "for the TFIM chain with L=2, h=0". Every
    word of it was about a system nobody had asked about, and the assumptions list
    that would have caught it is not shown beside a prose answer.

    Args:
        state: The campaign, after formalising.

    Returns:
        The model when the question named a chain length, and ``None`` when it did
        not. Keyed on the length because that is what a question naming a chain
        always carries; the other parameters have defaults a reader would accept.
    """
    if reading.read(state["request"].text).n_sites is None:
        return None
    return state["model"]


CIRCUIT_WORDS: tuple[str, ...] = (
    "circuit",
    "ansatz",
    "layer",
    "depth",
    "gate",
    "qaoa",
    "hva",
    "variational",
    "eigensolver",
    "transpile",
    "entangl",
)
"""Words that mean the asker is picturing a program, not only a physical system.

Checked alongside a chain length, and both are required. *What is a barren plateau*
is about circuits and names no chain; *is a 12-spin chain hard* names a chain and no
circuit. Neither describes a specific diagram, and drawing one anyway would be the
interface inventing a picture the question did not ask for.
"""


MEASURED_QUANTITY_WORDS: tuple[str, ...] = (
    "how deep",
    "how many layers",
    "how many shots",
    "how many measurements",
    "noise wins",
    "shot budget",
    "how accurate",
    "how close",
    "how long does it take",
)
"""Phrases naming something the depth ladder measures and prose cannot.

Read only for a feasibility question that named no chain, and only to decide whether
the loop runs -- never to decide a verdict, which stays withheld either way. *How deep
should the circuit be before noise wins?* is answered by climbing the ladder on a
stated default chain and reporting where the error stops falling; answering it from
the notes returns somebody else's number for somebody else's device.
"""


def asks_about_a_measured_quantity(text: str) -> bool:
    """Whether the question asks for a figure the campaign's loop produces.

    Args:
        text: The question as it was asked.

    Returns:
        Whether it names a quantity the ladder measures. Both a circuit word and a
        measuring phrase are required: *what is a barren plateau* is about circuits
        and wants prose, and *how long does the delivery take* measures nothing here.

    Examples:
        >>> asks_about_a_measured_quantity("How deep should the circuit be before noise wins?")
        True
        >>> asks_about_a_measured_quantity("What is a barren plateau?")
        False
    """
    lowered = text.lower()
    return any(word in lowered for word in CIRCUIT_WORDS) and any(
        phrase in lowered for phrase in MEASURED_QUANTITY_WORDS
    )


def circuit_the_question_named(
    state: CampaignState, fallback_depth: int = DEFAULT_CIRCUIT_DEPTH
) -> AnsatzSpec | None:
    """Work out which circuit, if any, a question described precisely enough to draw.

    Read here, in the agent, rather than in the interface that draws it. Two surfaces
    need the answer -- the diagram on the Chat page and the equations the code branch
    puts above its program -- and a second reading of the same question is a second
    thing that can decide on a different depth, which would put a picture of one
    circuit beside the algebra of another.

    Separated from the drawing because this is the half that can be quietly wrong.
    Whether a diagram appears at all, and what is in it, is decided here from the
    question's own words; :func:`src.ui.panels.circuit_asked_for` only puts the
    result on the page.

    A length and a circuit are both required, and neither alone will do. *What is a
    barren plateau* is about circuits and names no chain; *is a 12-spin chain hard*
    names a chain and no circuit. Drawing for either would be the interface inventing
    a picture nobody asked for, which is the mistake :func:`_chain_the_question_named`
    prevents on the numbers.

    Args:
        state: The finished campaign.
        fallback_depth: How many layers to draw when the question named no depth.
            Passed in rather than read from a settings knob, so that this function
            can be exercised without a session and so that the interface and the
            agent are looking at the same number.

    Returns:
        The specification to draw, or ``None`` when the question did not describe one.
    """
    request = state["request"]
    # A question the screen rejected, or one ruled out of scope, was never answered.
    # Putting a competent-looking diagram under a refusal would undo the refusal.
    if request.blocked or not request.in_scope:
        return None
    # One picture per question. A question that asked to *watch* three methods
    # converge names QAOA in passing and would otherwise also get a circuit diagram --
    # a second figure nobody asked for, drawn at the sidebar's depth rather than the
    # one the race ran at, so the screen showed two different depths for one question.
    # The curve is what was asked for; the circuit is the answer to a different
    # question, and it is one press away on the Quay Lab.
    if reading.asks_for_a_curve(request.text):
        return None
    found = reading.read(request.text)
    lowered = request.text.lower()
    if found.n_sites is None or not any(word in lowered for word in CIRCUIT_WORDS):
        return None
    model = state["model"]
    # A depth is a choice about the program, so a question that omits one has not made
    # a mistake -- it has left the choice to the reader, and the settings knob is where
    # the reader makes it. Which of the two supplied it is said in the caption, because
    # a number nobody typed must never look like one somebody did.
    return AnsatzSpec(
        n_qubits=found.n_sites,
        depth=max(found.depth or fallback_depth, 1),
        boundary=found.boundary or (model.boundary if model is not None else "open"),
        family="qaoa" if "qaoa" in lowered else "hva",
        longitudinal=model is not None and model.longitudinal_field != 0.0,
    )


# The tools the answering branches may reach through the model, in the order the
# prompt introduces them. Every tool in `src.agent.tools`, deliberately: the
# question "what can this agent do?" already has one answer, and a second, shorter
# list here would make it two answers that disagree. What keeps the loop cheap is
# the round limit and the prompt's instruction to call nothing when nothing helps,
# not a hidden shortlist.
ANSWER_TOOLKIT: tuple[BaseTool, ...] = TOOLS
"""What the consultation node may call before a host lends anything.

An alias rather than a subset, so that a tool added to :data:`src.agent.tools.TOOLS`
is reachable by the agent as well as by the MCP server. The two used to differ by
everything: the registry was exposed to external clients only, and the agent -- the
thing the registry is named for -- could call none of it.

One tool is *not* here and cannot be: the exact field sweep is built around a solver
this package may not import. :func:`answer_toolkit` adds it when the host lent one.
"""


def answer_toolkit(config: RunnableConfig) -> tuple[BaseTool, ...]:
    """The tools this run may call, including any the host lent it.

    Built per run rather than read from a constant, and that is the whole mechanism
    by which the agent can plot an exact solution it is not allowed to import. When
    the configuration carries a field-sweep bench, this returns the standing toolkit
    plus a tool constructed around it; when it does not, the extra tool does not
    exist -- there is no disabled stub to reason about and nothing to switch off.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The toolkit, standing tools first and the lent one last, which is the order
        the consultation prompt introduces them in.
    """
    bench = _bench_from(config)
    return ANSWER_TOOLKIT if bench is None else (*ANSWER_TOOLKIT, field_sweep_tool(bench))


SWEEP_TOOL_NAME = "exact_field_sweep"
"""The lent tool's name, as the model calls it.

Written here rather than taken off the tool object, because the tool is *built* when
a host lends the solver behind it and there is no module-level object to read a name
from until then. Held to the built tool's own name by a test.
"""

DIVERTED_NOTE = (
    "no verdict will be reached: the question asked whether something is worth doing "
    "but named no chain to decide it about, so it is answered in prose rather than "
    "with a verdict about an invented one"
)
"""The trail line for a feasibility question answered as an explanation.

Said out loud because the alternative is a reader seeing an explanation where they
expected a verdict, with nothing anywhere saying why -- which is the same complaint
as the badge, one step further on.
"""


def _diverted_to_prose(state: CampaignState) -> dict[str, Any]:
    """Record that a feasibility question is being answered in prose after all.

    A question can be read as feasibility, name no chain, and be sent to the prose
    branch by :func:`after_converge` -- *can you teach me the quantum-to-classical
    mapping of the transverse-field Ising chain?* does exactly that, and the diversion
    is right. What is left behind is the reading, which still says ``feasibility``.
    The interface badges an answer from the reading, so a perfectly good explanation
    arrives under a NO VERDICT badge and a trail describing a campaign that was
    abandoned. The correction has to happen here rather than on the edge, because a
    conditional edge returns a destination and cannot write state, and because every
    diverted question passes through this node: a feasibility reading that arrives
    here is by definition the diverted case, since nothing else is sent this way.

    Args:
        state: The campaign, on its way into the prose branch.

    Returns:
        A state fragment replacing the reading, or an empty one when there is nothing
        to correct. Merged into whatever else the node returns, so the correction
        cannot be lost down one of its five exits -- there are five, and two of them
        dropped it until an audit found them. It carries no ``notes`` key of its
        own: ``notes`` accumulates across nodes but *within* one returned dictionary
        a second spelling of the key simply replaces the first, so a note added here
        would be silently dropped by every exit that also writes one. The sentence
        for the trail is :data:`DIVERTED_NOTE`, added beside it.
    """
    reading_so_far = state["intent"]
    if not reading_so_far.runs_campaign:
        return {}
    model = state["model"]
    named = model is not None and model.length_was_given
    return {
        "intent": Reading(
            intent="explain",
            decided_by=reading_so_far.decided_by,
            reason=(
                "read as a feasibility question, but "
                + (
                    "no chain was named, so there is no specific problem to reach a "
                    "verdict about; answered in prose instead"
                    if not named
                    else "diverted to prose"
                )
            ),
            scores=reading_so_far.scores,
        ),
    }


def consult_tools(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Let the model call tools before the answer is written.

    The one node where the *model* decides how much work happens, and the reason it
    is a node rather than a few lines inside the answering branches: a reader needs
    to see it on the trace, with what was called and what came back, and a step that
    can spend four model calls should be visible in the graph that spends them.

    Placed before ``explain`` and ``implement`` and not on the feasibility branch.
    Feasibility already computes everything it claims -- baseline, depth ladder, shot
    arithmetic, verdict -- through direct calls that need no model to choose them, and
    putting a tool loop in front of that would add cost and latency to the branch that
    least needs it. The two prose branches are the opposite case: they stand on
    retrieved text, and a request for recent papers or an exact derivative is
    something the corpus structurally cannot answer.

    Skipped without a model, which is not a degradation -- every branch here has a
    path that needs no model, and this one's is "write the answer from the notes".

    Args:
        state: The campaign, after retrieval.
        config: The graph's runtime configuration, for the models.

    Returns:
        The tool calls made and a line for the trail. Never an ``answer``: this node
        gathers material and writes nothing a reader sees, which is what stops the
        same question being answered twice by two different calls.
    """
    diverted = _diverted_to_prose(state)

    pool = _pool_from(config)
    extra = (DIVERTED_NOTE,) if diverted else ()
    if pool.offline:
        by_rule = _sweep_by_rule(state, config)
        if by_rule is None:
            return {**diverted, "notes": (*extra, "no model, so no tools were offered")}
        return {
            **diverted,
            "tool_calls": (by_rule,),
            "notes": (
                *extra,
                "no model, so the exact field sweep was chosen by rule rather than "
                f"by a model and called directly -- {by_rule.summarise(90)}",
            ),
        }

    consultation = pool.consult(
        "tool_consultation",
        TOOL_CONSULTATION,
        _consultation_brief(state),
        answer_toolkit(config),
    )
    if consultation is None:
        by_rule = _sweep_by_rule(state, config)
        if by_rule is None:
            # `**diverted` and `*extra` on this exit too. They were missing from this
            # one and from the no-tools-called exit below, and the consequence was a
            # paid-for answer thrown away: a question read as `feasibility` but
            # diverted to prose lost the correction here, so `write_report` took the
            # verdict path and printed "The campaign did not reach a verdict." with
            # empty sections -- over the top of prose that had already been written
            # and billed. `_diverted_to_prose` claims the correction "cannot be lost
            # down one of its four exits"; there are five.
            return {
                **diverted,
                "notes": (
                    *extra,
                    "the tool consultation could not be made; no tool was called",
                ),
            }
        return {
            **diverted,
            "tool_calls": (by_rule,),
            "notes": (
                *extra,
                "the tool consultation could not be made, so the exact field sweep "
                f"was called by rule -- {by_rule.summarise(90)}",
            ),
        }
    calls = _with_the_curve_that_was_asked_for(state, config, consultation.calls)
    if not calls:
        # The ordinary case, and the one where losing `diverted` hurt most: a model
        # offered tools and calling none is not a failure, so this exit is taken on
        # perfectly healthy runs.
        return {
            **diverted,
            "notes": (
                *extra,
                f"offered the tools and the model called none: {consultation.stopped_because}",
            ),
        }

    _logger.info(
        "campaign_consulted",
        extra={
            "calls": len(calls),
            "rounds": consultation.rounds,
            "tools": [call.name for call in calls],
        },
    )
    called = ", ".join(dict.fromkeys(call.name for call in calls))
    failures = sum(1 for call in calls if call.failed)
    return {
        **diverted,
        "tool_calls": calls,
        "notes": (
            *extra,
            f"called {len(calls)} tool{'' if len(calls) == 1 else 's'} in "
            f"{consultation.rounds} round{'' if consultation.rounds == 1 else 's'}"
            f" -- {called}" + (f", {failures} of which returned an error" if failures else ""),
        ),
    }


def _with_the_curve_that_was_asked_for(
    state: CampaignState,
    config: RunnableConfig,
    calls: tuple[ToolCall, ...],
) -> tuple[ToolCall, ...]:
    """Add the field sweep to the consultation when the model did not call it.

    The figure is drawn from a recorded tool call, so a model that declines the tool
    produces an answer with no figure in it. *Plot the exact energy levels as the
    field is turned up* did exactly that: a fluent answer carrying the closed-form
    dispersion, the elliptic integral for the energy density, the logarithmic
    divergence of the susceptibility and the linear gap closing, closing on "the exact
    energy levels as the field is turned up can be computed and plotted", with nothing
    anywhere on the page computing or plotting them.

    Declining a tool is ordinarily the model's call to make, and usually right. It is
    not right on this branch, because the reading already established that the
    question asked for a curve in the field -- that is what routed it here instead of
    to the method race. Leaving the fetch to the model's discretion lets a decision
    this project has already made be reversed by a call that happened not to happen.
    The curves are computed by code either way; the model contributes the prose and
    the choice of which curves, and only the second is at stake here.

    So the model's own call wins when it made one, and otherwise the sweep is called
    with the curves read out of the sentence, and the trail says which happened.

    Args:
        state: The campaign, after retrieval.
        config: The graph's runtime configuration, for the lent bench.
        calls: What the consultation actually called.

    Returns:
        The calls, with a sweep appended when the question asked for one, no sweep
        succeeded, and a host lent the solver. Unchanged otherwise -- including when
        the model's own sweep call *failed*, since a second identical attempt would
        fail identically and the reader is better served by seeing the error.
    """
    if not reading.asks_for_a_field_sweep(state["request"].text):
        return calls
    if any(call.name == SWEEP_TOOL_NAME for call in calls):
        return calls
    by_rule = _sweep_by_rule(state, config)
    return calls if by_rule is None else (*calls, by_rule)


def _sweep_by_rule(state: CampaignState, config: RunnableConfig) -> ToolCall | None:
    """Call the exact field sweep without a model, when the question asked for one.

    The offline path of the consultation node, and it exists for the same reason
    every other branch here has one: this application has to work with no key, no
    account and no network, and *plot the low-lying spectrum against the field* is a
    question with a right answer that needs no language model to fetch.

    What a model contributes when there is one is judgement about **which** curves
    and **which** chain, and that is exactly what is degraded here rather than
    faked: :func:`src.agent.reading.curves_asked_for` reads the curves out of the
    sentence by rule, the chain is the one the campaign formalised, and the trail
    says the choice was made by rule. A worse choice, honestly labelled, and a very
    much better outcome than an empty panel under a question about a curve.

    Args:
        state: The campaign, after retrieval.
        config: The graph's runtime configuration, for the lent bench.

    Returns:
        The recorded call, or ``None`` when there is nothing to call -- no bench was
        lent, or the question was not about a curve in the field.
    """
    bench = _bench_from(config)
    question = state["request"].text
    if bench is None or not reading.asks_for_a_field_sweep(question):
        return None
    model = state["model"]
    arguments: dict[str, Any] = {
        "n_sites": model.n_sites if model is not None else DEFAULT_SITES,
        "curves": list(reading.curves_asked_for(question)),
        "coupling": model.coupling if model is not None else 1.0,
        # The boundary the *question* named, read from its own words rather than
        # taken off the formalised chain -- which defaults to a segment, so every
        # question that mentioned no boundary at all was refused by a closed form
        # that needs a ring. A question that does name one gets it, and a segment
        # then comes back as a refusal naming the reason: substituting a ring for a
        # chain the reader described would draw the right curve for a different
        # problem under their own words.
        "boundary": reading.read_boundary(question) or "periodic",
    }
    # The shape comes off the formalised model, which is where the reader's own
    # words about a square or triangular lattice ended up. Left out, a question
    # about a lattice's curve was swept as a *line* of the same size and drawn
    # under the lattice's name -- the same failure as substituting a ring for a
    # segment above, one dimension up, and a larger error: at sixteen sites the
    # two energies differ by nearly a factor of two per site.
    if model is not None and model.lattice.geometry != "chain":
        arguments["geometry"] = model.geometry
        arguments["rows"] = model.rows
        # A lattice has no closed form, so a ring is not the better default there:
        # both boundaries cost one diagonalisation per point, and the reader's own
        # words should decide.
        arguments["boundary"] = reading.read_boundary(question) or model.boundary
    instrument = field_sweep_tool(bench)
    try:
        outcome = str(instrument.invoke(arguments))
    except Exception as error:  # a failed sweep must not lose the answer
        _logger.warning("sweep_by_rule_failed", extra={"error_type": type(error).__name__})
        return None
    return ToolCall(
        name=instrument.name,
        arguments=arguments,
        result=outcome,
        failed='"error"' in outcome[:24] or outcome.startswith("{'error'"),
    )


def _consultation_brief(state: CampaignState) -> str:
    """Write what the consulting model needs to know, and nothing more.

    The question, the chain if one was named, and how the corpus did. The passages
    themselves are deliberately *not* included: this call decides what to fetch, and
    handing it four abstracts invites it to answer from them instead -- which is the
    downstream call's job and is done better there, with the audience setting applied.

    Args:
        state: The campaign, after retrieval.

    Returns:
        The brief. The question is wrapped as data by the funnel it goes through.
    """
    model = state["model"]
    lines = [state["request"].text]
    if model is not None:
        lines.append(f"\nThe chain in question: {model.label()}.")
    else:
        lines.append("\nNo specific chain was named.")
    if reading.asks_for_a_field_sweep(state["request"].text):
        # Said explicitly because the reading already knows it, and a model that has
        # to infer it from the sentence sometimes does not: the live answer to *plot
        # the exact energy levels as the field is turned up* called no tool at all and
        # ended by telling the reader that the levels could be computed and plotted.
        wanted = ", ".join(reading.curves_asked_for(state["request"].text))
        lines.append(
            f"\nThis question asks for a quantity traced against the field, and the "
            f"reader is shown it as a figure beneath the answer. Call "
            f"`{SWEEP_TOOL_NAME}` for it -- it solves the model exactly at every "
            f"field in the range and nothing else here can. Reading the sentence by "
            f"rule, the curves it asks for are: {wanted}. Use your own judgement if "
            f"you read it differently, but do call it."
        )
    kept = len(state["citations"])
    lines.append(
        f"This project's own notes were searched and returned {kept} passage(s)."
        + (
            " They are a fixed set of abstracts and cannot tell you what is recent."
            if kept
            else " Nothing usable came back, so anything you do not fetch will be absent."
        )
    )
    return "\n".join(lines)


def rendered_tool_calls(state: CampaignState) -> str:
    """Render what the tools returned, for the call that writes the answer.

    Failed calls are named but their output is dropped. A model that reads its own
    error messages back tends to explain them to the reader, and "search_arxiv
    returned an error" is not an answer to anything -- while the *fact* that a search
    was attempted and failed does belong in the trail, which is where it goes.

    Args:
        state: The campaign, after the consultation.

    Returns:
        The block, or the empty string when nothing succeeded.
    """
    useful = [call for call in state["tool_calls"] if not call.failed]
    if not useful:
        return ""
    return "\n\n".join(call.summarise(width=MAX_RENDERED_TOOL_RESULT) for call in useful)


MAX_RENDERED_TOOL_RESULT = 3200
"""How much of one tool result is passed on to the writing call, in characters.

Wider than the trail's excerpt and narrower than the raw result. An arXiv search
returns four abstracts and the writer needs all four; a classical baseline returns a
dozen numbers and the writer needs them exactly. What is cut is the long tail of a
tool that returned more than anyone asked for.

Raised from 2400 when the exact field sweep arrived, which returns a table of a
whole curve plus the features on it named by number. Two things had to be true for
that to be safe rather than a habit of widening the window whenever something does
not fit: the table thins its own rows before returning, and it puts the named
features *above* the rows -- so what a clip removes is the tail of the evidence
rather than the head of the answer. See
:meth:`src.physics.reference.field_sweep.FieldSweep.table`.
"""


def answer_in_prose(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Answer a question that asked to be told something, not shown a measurement.

    Runs no circuit, spends no shots and reaches no verdict, and the state it returns
    says so by omission: ``verdict`` stays ``None``, ``runs`` stays empty. The
    interface reads those rather than being told separately, so there is no way for
    this branch to leave a feasibility call on the screen that nothing computed.

    Args:
        state: The campaign, with whatever retrieval found.
        config: The graph's runtime configuration, for the models.

    Returns:
        The prose answer and a line for the trail.
    """
    written = explain_in_prose(
        state["request"].text,
        _chain_the_question_named(state),
        state["citations"],
        _pool_from(config),
        _audience_from(config),
        # Everything else the campaign established by the time it got here. The
        # branch answers in one woven piece rather than by touring the shelves, and a
        # narrator can only weave what it was handed: without the recall it answers
        # the turn before it as though nobody had asked, and without the search trail
        # it cannot tell a corpus that is silent on the subject from a query that
        # never named it.
        recalled=_recalled(config),
        searches=state["searches"],
        # What this campaign computed for this question, when it computed anything.
        # Without it a question that asked to watch three methods race was answered
        # "the notes do not cover this" with the race drawn directly underneath.
        measured=_measured(state),
        # What the tools actually returned. The reason a request for recent papers
        # can now be answered with recent papers rather than with whatever the
        # corpus happened to be built from.
        consulted=rendered_tool_calls(state),
        # Where to send the answer as it is written, when the caller offered
        # somewhere. The returned answer is still the one of record -- this is a
        # preview, raw and unrepaired -- so nothing downstream reads it.
        sink=_prose_sink_from(config),
    )
    _logger.info(
        "campaign_answer",
        extra={"by": written.written_by, "cited": written.cited},
    )
    return {"answer": written, "notes": (written.explain(),)}


def _measured(state: CampaignState) -> str:
    """Render what this campaign computed for the question, for the branch that writes.

    Only the method race so far, because it is the only thing the non-feasibility
    branches produce a number from. Kept out of :mod:`src.agent.explaining` so that the
    prose module stays a writer: it is handed material and does not go looking for it.

    Args:
        state: The campaign, after any race.

    Returns:
        A short markdown block, or the empty string when nothing was computed.
    """
    race = state["race"]
    if race is None or not race.runs:
        return ""
    winner = race.best()
    lines = [
        f"**{race.n_sites} spins, {race.depth} circuit layers, "
        + fields_in_words(race.coupling, race.transverse_field, race.longitudinal_field)
        + (" -- the critical point.**" if race.at_criticality else ".**"),
        "",
        "| method | energy reached | epochs | energy evaluations | stopped because |",
        "|---|---|---|---|---|",
        *(
            f"| {run.method} | {run.energy:.6f} | {run.n_steps} | "
            f"{run.n_energy_evaluations} | {run.stop_reason.replace('_', ' ')} |"
            for run in race.runs
        ),
    ]
    if winner is not None:
        lines += [
            "",
            f"Lowest energy: **{winner.method}**. Every energy here is exact -- no shot "
            "noise and no gate error -- so this compares the methods on a perfect "
            "device and says nothing about robustness to noise.",
        ]
    return "\n".join(lines)


def write_code(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Write runnable code for the chain, when that is what was asked for.

    Nothing written here is executed -- see :mod:`src.agent.drafting`. The draft
    carries its own reason when it is empty, so a reader who gets no code is told
    which of the three reasons applied rather than being left to assume the feature
    is broken.

    Args:
        state: The campaign, with the chain and whatever retrieval found.
        config: The graph's runtime configuration, for the models.

    Returns:
        The draft and a line for the trail.
    """
    passages = "\n\n".join(f"{c.title}: {c.snippet}" for c in state["citations"][:4])
    # Only when the question named one. A request for code that names no size
    # should get a runnable example at whatever size the writer picks, not a
    # program built around a two-spin chain the reader never mentioned.
    named = _chain_the_question_named(state)
    # The equations are composed here, from integers, and are the one part of a code
    # answer this project can stand behind -- the program itself is text nobody ran.
    # They are also given to the writer, so the file and the algebra above it use the
    # same letters in the same order and a reader can actually check one against the
    # other.
    spec = circuit_the_question_named(state, _circuit_depth_from(config))
    mathematics = (
        circuit_mathematics(
            spec,
            named.coupling if named is not None else None,
            named.transverse_field if named is not None else None,
            named.longitudinal_field if named is not None else 0.0,
        )
        if spec is not None
        else ""
    )
    written = draft_code(
        state["request"].text,
        named,
        passages,
        _pool_from(config),
        mathematics,
        state_preparation_latex(spec) if spec is not None else "",
    )
    _logger.info(
        "campaign_code",
        extra={"written": written.written, "equations": bool(mathematics)},
    )
    return {"draft": written, "notes": (written.explain(),)}


def write_report(state: CampaignState) -> dict[str, Any]:
    """Compose the written answer, in the shape the question asked for.

    One node for three branches rather than three, because what changes between them
    is the document and not the step: every branch ends by handing the interface one
    piece of markdown under ``report``, and a reader of the trace should see the same
    node close every run.

    The feasibility branch is the only one that reaches a verdict, and it is the only
    one that calls :func:`judge`. That is deliberate and load-bearing. Judging on the
    other branches would put a confident *no, quantum hardware is not worth it* under
    an answer to *what is a barren plateau* -- decided by a rule that read a chain
    nobody asked about, from a campaign that ran nothing.

    Args:
        state: The finished campaign.

    Returns:
        The document, and a line for the trail naming which kind was written.
    """
    intent = state["intent"].intent

    if intent == "explain":
        written = state["answer"]
        return {
            "report": _prose_document(state, written),
            "notes": ("wrote the explanation",),
        }

    if intent == "implement":
        return {
            "report": _code_document(state, state["draft"]),
            "notes": ("wrote the code answer",),
        }

    verdict, decided_by = judge(state)
    report = compose(state, decided_by)
    _logger.info("campaign_report", extra={"characters": len(report), "call": verdict.call})
    return {"report": report, "notes": ("wrote the feasibility report",)}


def _sources_section(state: CampaignState) -> str:
    """List what was cited, in the numbering the answer's markers refer to.

    Args:
        state: The campaign, with whatever retrieval kept.

    Returns:
        A markdown list, or a line saying the notes held nothing. Never empty: an
        answer that silently omits its sources looks identical to one that had none.
    """
    citations = state["citations"]
    if not citations:
        return (
            "**Sources.** The notes returned nothing for this question, so the answer "
            "above rests on general knowledge alone and should be treated accordingly."
        )
    # One line per document, not one per passage. Retrieval works in chunks, so a
    # document that answers the question well is returned several times over -- and
    # a list that printed each of them showed the same title four times, which reads
    # as four independent sources agreeing. That is the opposite of what happened.
    seen: dict[str, Citation] = {}
    for citation in citations:
        seen.setdefault(citation.identifier, citation)
    lines = ["**Sources.** Numbered as the answer refers to them.", ""]
    lines += [
        f"{index}. {c.title} (`{c.identifier}`)"
        + ("" if c.reviewed else " — fetched from an external index, not reviewed")
        for index, c in enumerate(list(seen.values())[:MAX_LISTED_SOURCES], start=1)
    ]
    return "\n".join(lines)


def _prose_document(state: CampaignState, written: Answer | None) -> str:
    """Assemble the explanation branch's answer into one piece of markdown.

    Args:
        state: The finished campaign.
        written: What the explaining step produced.

    Returns:
        The document. It opens by saying what kind of answer this is, because a
        reader arriving from a feasibility report needs to know within one line that
        this one ran nothing and concluded nothing about hardware.
    """
    if written is None or not written.written:
        return (
            "## Nothing was written\n\nThe question was read as asking for an "
            "explanation, but no answer was composed. This is a fault rather than a "
            "finding; the campaign trail on the Pipeline trace page says where it stopped."
        )
    parts = [
        "## Explanation",
        "",
        "*This answer explains. It ran no circuit, spent none of the measurement "
        "budget and reached no verdict on whether quantum hardware is worth using "
        "here -- that question has its own assessment, and guessing at it from prose "
        "would look the same on the page and mean nothing.*",
        "",
        written.text,
    ]
    if written.rests_on:
        parts += ["", f"**How to check this.** {written.rests_on}"]
    parts += ["", _sources_section(state)]
    return "\n".join(parts)


def _code_document(state: CampaignState, written: Draft | None) -> str:
    """Assemble the code branch's answer into one piece of markdown.

    Args:
        state: The finished campaign.
        written: What the drafting step produced.

    Returns:
        The document, with the caveat attached to the code rather than filed at the
        end. A reader who scrolls to the block and copies it should have already
        passed the sentence saying nobody ran it.
    """
    model = state["model"]
    described = f" for {model.label()}" if model is not None else ""
    if written is None or not written.written:
        reason = written.note if written is not None else "the drafting step did not run"
        return (
            f"## No code was written\n\n{reason.capitalize()}.\n\n"
            "Nothing is being offered in its place. A feasibility report is not an "
            "answer to a request for a file, and substituting one is the behaviour "
            "this branch exists to stop."
        )
    parts = [
        f"## Code{described}",
        "",
        "*Written by a language model and **not run by this application**. Every other "
        "number here is one two methods sharing no algebra agreed on; this has no such "
        "backing, and executing model-written code to find out is not a trade this "
        "project makes. The program ends by checking itself against a known result, "
        "which is the check you can run.*",
        "",
    ]
    if written.preamble:
        parts += [written.preamble, ""]
    # Above the code, in the report as on the page, and for the same reason: this is
    # the half of the answer that was composed rather than generated.
    if written.mathematics:
        parts += [written.mathematics, ""]
    parts += [f"```{written.language}", written.code, "```", "", _sources_section(state)]
    return "\n".join(parts)


def suggest_next(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Propose what to ask next, from what this campaign established.

    A node rather than something the interface works out for itself, and that is the
    whole point: a suggestion is a question that will be asked, so it belongs to the
    agent that would have to answer it. Putting it here means it is recorded in the
    campaign, visible in the trace, screened by the same guard as anything typed, and
    charged to the same call budget -- none of which is true of a list an interface
    composes on the side.

    It runs after the report rather than before, because the most useful follow-up
    depends on what the verdict turned out to be.

    Args:
        state: The finished campaign.
        config: The graph's runtime configuration, for the models.

    Returns:
        The suggestions, and a note saying who wrote them.
    """
    proposed = propose(state, _pool_from(config), enabled=_followups_allowed(config))
    _logger.info(
        "campaign_followups",
        extra={"count": len(proposed.suggestions), "by": proposed.proposed_by},
    )
    return {"followups": proposed, "notes": (proposed.explain(),)}


# ── edges ──────────────────────────────────────────────────────────────────────


def recall_conversation(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Read what was said earlier in this conversation, if anything was.

    A node rather than a step folded into the campaign's entry point, because it is a
    decision with a visible consequence: a follow-up that names no chain is answered
    from what came before, and a reader looking at the trace should be able to see
    that happening rather than infer it. It also means the whole graph, memory
    included, runs from one call.

    Args:
        state: The opening campaign.
        config: The graph's runtime configuration, which is where the memory is
            injected. Absent means this conversation has no history, which is the
            default and what every test uses.

    Returns:
        Only notes. What was recalled travels in the configuration, so that nodes do
        no file access and the state stays a record of this campaign rather than of
        the ones before it.
    """
    recalled = _recalled(config)
    if not recalled:
        return {"notes": ("no earlier conversation to draw on",)}
    turns = recalled.count("asked:")
    _logger.info("campaign_recall", extra={"turns": turns})
    return {
        "notes": (
            f"recalled {turns} earlier exchange{'s' if turns != 1 else ''} in this "
            "conversation, which is what lets a follow-up name only what changed",
        )
    }


def remember_exchange(state: CampaignState, config: RunnableConfig) -> dict[str, Any]:
    """Record this exchange, so the next question can be read in its light.

    A short summary is stored rather than the report. A report is thousands of words
    written for a reader; what a follow-up needs is the sentence saying what the answer
    was, and storing the long form would spend the whole recall window on one turn.

    Every branch is remembered, not only the one that reaches a verdict. Writing only
    when ``state["verdict"]`` is set leaves an explanation and a piece of code with no
    trace, so a follow-up to either is judged on its own words, found to name no
    physics, and declined. *Plot the loss curve for these three methods* followed by
    *can we see epochs 0 to 20?* fails there: the second question means nothing on its
    own, and carrying it is what a memory is for.

    Args:
        state: The finished campaign.
        config: The graph's runtime configuration, which is where the memory is
            injected.

    Returns:
        Only notes. Nothing about the campaign changes by being written down.
    """
    memory = _memory_from(config)
    summary, call = _worth_remembering(state)
    if memory is None or not memory.enabled or not summary:
        return {"notes": ("nothing was remembered from this exchange",)}
    written = memory.remember_turn(state["request"].text, summary, call)
    if written is None:
        return {"notes": ("declined to remember this exchange: the input screen refused it",)}
    return {"notes": ("remembered this exchange, so a follow-up can build on it",)}


def _worth_remembering(state: CampaignState) -> tuple[str, str]:
    """Reduce whatever a branch produced to the one line a follow-up needs.

    Ordered by how much the answer settled. A verdict is the strongest thing a campaign
    can conclude and is stored as such; prose and code are recorded by what they were
    about, because a follow-up needs to know *which chain was under discussion* far
    more than it needs the words that were said about it. Failing all three, the
    formalised chain alone is still worth a line -- a question that was read and then
    answered badly is still a question that named a chain.

    Args:
        state: The finished campaign.

    Returns:
        The summary and the verdict word, either of which may be empty. An empty
        summary means there was genuinely nothing here to carry forward.
    """
    verdict = state["verdict"]
    if verdict is not None:
        return verdict.summary, verdict.call
    model = state["model"]
    subject = f" about {model.label()}" if model is not None else ""
    answer = state["answer"]
    if answer is not None and answer.written:
        return f"answered in prose{subject}: {clip(answer.text, MEMORY_SUMMARY_CHARS)}", ""
    draft = state["draft"]
    if draft is not None and draft.written:
        return f"wrote {draft.language} code{subject}", ""
    return (f"read the question as{subject}" if subject else ""), ""


def _memory_from(config: RunnableConfig) -> Memory | None:
    """Find this conversation's memory, if there is one.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The memory, or ``None`` when the campaign is running without one.
    """
    configurable = config.get("configurable") or {}
    held = configurable.get(MEMORY_KEY)
    return held if isinstance(held, Memory) else None


def after_formalise(state: CampaignState) -> Literal["retrieve", "conclude"]:
    """Decide whether there is a problem here worth doing any work on.

    A refused or declined question stops here rather than walking the whole graph.
    Searching a corpus for a problem that does not exist, running a baseline on a
    chain nobody asked about and proposing configurations for it are all no-ops that
    produce notes, and they leave the pipeline trace showing a full route for a
    question thrown out at the door.

    Args:
        state: The campaign, just after formalising.

    Returns:
        ``"retrieve"`` when there is a Hamiltonian to work on, ``"conclude"`` when
        there is not.
    """
    return "retrieve" if state["model"] is not None else "conclude"


def after_plan(state: CampaignState) -> Literal["solve", "plan", "conclude"]:
    """Decide what follows a planning round.

    Three outcomes, and the middle one is why this is a loop at all: a plan that was
    priced and rejected sends the campaign back to propose something cheaper, which
    is what a person does when told the thing they wanted is out of budget.

    Going round again is only allowed when the next turn could do something
    different. A planner with nothing left to propose that is sent back to plan will
    propose nothing, record nothing, and return here unchanged, so the exhausted case
    has to be a stop rather than a retry.

    Args:
        state: The campaign, just after planning.

    Returns:
        ``"solve"`` when a plan is pending, ``"plan"`` to try something cheaper,
        ``"conclude"`` when nothing was formalised, when the loop has had enough
        turns, or when it has nothing left to try.
    """
    if state["pending"] is not None:
        return "solve"
    # Nothing was formalised, so there is nothing to plan and no round will ever
    # record anything. Without this the loop spins until the graph's own recursion
    # limit stops it -- which it did, for every refused and every out-of-scope
    # request, because the planner returns "nothing pending" and the guards below
    # both look at evidence a planner with no model never produces.
    if state["model"] is None:
        return "conclude"
    if _rounds_used(state) >= MAX_PLANNING_ROUNDS:
        return "conclude"
    # Asking the same question the planner just asked. If there is still a depth it
    # has neither tried nor rejected, a cheaper attempt is worth one more turn; if
    # there is not, going round again would propose nothing, record nothing, and
    # arrive back here in exactly this state -- which is a loop with no exit.
    if _next_depth(state) is None:
        return "conclude"
    return "plan"


def after_analyse(state: CampaignState) -> Literal["plan", "conclude"]:
    """Decide whether the campaign should try another configuration.

    Stops on any of three conditions: the budget is exhausted, the round limit is
    reached, or the last run reported something no further attempt can fix. The
    third matters most -- a campaign that keeps trying after being told the result
    is impossible is burning budget to reproduce a bug.

    Args:
        state: The campaign, just after analysis.

    Returns:
        ``"plan"`` to go round again, ``"conclude"`` to stop.
    """
    if _rounds_used(state) >= MAX_PLANNING_ROUNDS:
        return "conclude"
    if state["runs"] and not state["runs"][-1].diagnosis.is_actionable:
        return "conclude"
    if state["shots"].remaining <= 0:
        return "conclude"
    if _next_depth(state) is None:
        return "conclude"
    return "plan"


# ── wiring ─────────────────────────────────────────────────────────────────────


def after_retrieve(state: CampaignState) -> list[str]:
    """Send the question down the branch its reading asked for.

    The one place the intent has a consequence. Everything before it is shared --
    every question is screened, read, formalised and searched -- and everything after
    it differs.

    A feasibility reading returns two destinations, which is how the fan-out on
    :func:`build_graph` survives being put behind a condition: the classical baseline
    and the planning loop start together, so nothing on the quantum side can route
    around the number it has to beat. The other two readings return one node each and
    rejoin at the scribe.

    ``converge`` is added to whichever branch was chosen rather than replacing it. A
    question asking to watch the methods race is still an explanation, or still a
    feasibility study, and wants the picture in addition.

    Two shapes of question reach the race: *plot the loss against epoch*, and *VQE,
    QAOA or imaginary time?*, which names three methods and asks which with no
    plotting word in it. One shape must not -- *plot the low-lying spectrum against
    the field* contains a plotting word and is not a question about an optimiser, and
    on the race it returns three agreeing energies for a chain nobody named.
    :func:`src.agent.reading.asks_for_a_field_sweep` reads the axis and keeps those
    off the race; :func:`answer_toolkit` solves the model exactly at every point
    instead.

    Args:
        state: The campaign, just after retrieval.

    Returns:
        The nodes to run next. A list rather than a key into a mapping, because a
        mapping cannot express "both of these at once".
    """
    question = state["request"].text
    # A curve whose axis is the *field* is a question about the model, not about an
    # optimiser, and racing three approximate methods to answer it would answer past
    # it. Checked first because both of the conditions below also match those
    # sentences -- "plot" is a plotting word wherever it appears.
    if reading.asks_for_a_field_sweep(question):
        return after_converge(state)
    # Two different ways of asking for the same three runs. One asks to watch the
    # descent; the other names the methods and asks which. Both are answered by racing
    # them, and neither is answered by prose alone -- see the two functions' docstrings.
    if reading.asks_for_a_curve(question) or reading.asks_to_compare_methods(question):
        return ["converge"]
    return after_converge(state)


def after_converge(state: CampaignState) -> list[str]:
    """Send the question down the branch its reading asked for, race or no race.

    The same decision as :func:`after_retrieve` makes when no race is wanted, and
    deliberately the same function rather than a copy of it: two lists of branch names
    that have to agree is one list that will eventually not.

    The race runs before the answer rather than beside it, which is the reason this
    edge exists. Racing in parallel puts the answering node in the same superstep,
    where it cannot see the result, so the prose gets written as though nothing had
    been computed while the computed figure sits directly beneath it. One superstep of
    latency buys an answer that can quote its own measurement.

    Args:
        state: The campaign, after retrieval and after any race.

    A feasibility reading is only honoured when a chain was actually named -- see the
    comment in the body. Everything else is answered in prose.

    Returns:
        The nodes to run next.
    """
    intent = state["intent"].intent
    # Both prose readings go to the same place, and which of the two runs is decided
    # by `after_consult` on the far side of it. Naming the branch here as well would
    # be the same decision written twice, and the copy that goes stale is always the
    # one nobody is looking at.
    if intent in ("explain", "implement"):
        return ["consult"]
    model = state["model"]
    if model is not None and not model.length_was_given:
        # A verdict is a statement about one specific problem, and a question that
        # named no chain cannot receive one -- `apply_skeptic` withholds it. Whether
        # the *loop* should still run is a separate question, and the answer depends
        # on what was asked. *How deep should the circuit be before noise wins?* is
        # answered by the depth ladder and by nothing else; *which real machines could
        # run these methods?* is answered from the notes, and running a ladder for it
        # would spend a minute to print a number nobody asked for.
        if not asks_about_a_measured_quantity(state["request"].text):
            return ["consult"]
    return ["baseline", "plan"]


def after_consult(state: CampaignState) -> str:
    """Choose which of the two prose branches writes the answer.

    Split from :func:`after_converge` rather than folded into it, because the two
    decide different things: that one decides *whether* prose is the answer at all,
    this one decides which kind of prose. Both read the same intent, and neither can
    disagree with the other about a question it was not asked.

    Args:
        state: The campaign, after the consultation.

    Returns:
        ``"implement"`` for a request for code, ``"explain"`` for everything else.
        Not a list: exactly one of these writes, and returning a list here would
        make it expressible for both to.
    """
    return "implement" if state["intent"].intent == "implement" else "explain"


def build_graph() -> CompiledStateGraph:
    """Wire the nodes together and compile the campaign.

    The fan-out after retrieval is the structural commitment: the classical
    baseline and the quantum loop are siblings, and both must finish before the
    verdict is reached. Nothing on the quantum side can route around the classical
    branch, because it is not downstream of anything the planner controls.

    The verdict node is deferred so that "both must finish" is true rather than
    merely intended -- see the comment beside it.

    Returns:
        The compiled graph, ready to invoke with an opening state.
    """
    builder: StateGraph[CampaignState, None, CampaignState, CampaignState] = StateGraph(
        CampaignState
    )

    builder.add_node("recall", recall_conversation)
    builder.add_node("screen", screen_request)
    builder.add_node("interpret", interpret_request)
    builder.add_node("formalise", formalise)
    builder.add_node("retrieve", retrieve_background)
    builder.add_node("baseline", run_classical_baseline)
    # Named for what a reader watches it do rather than for the module it calls. It is a
    # sibling of the branch that answers the question, not a step inside it: see
    # `after_retrieve` for why it is added to a branch rather than substituted for one.
    builder.add_node("converge", converge_methods)
    builder.add_node("plan", plan_configuration)
    # Named for the job rather than for the mechanics. It was ``run``, and on the
    # trace page that produced a graph reading plan -> run -> analyse, in which a
    # reader could not see where the problem was actually solved -- the one step
    # every other step exists to serve. This is the node that calls the variational
    # eigensolver.
    builder.add_node("solve", run_configuration)
    builder.add_node("analyse", analyse_runs)
    # The two branches that are not the feasibility campaign. Each is one node,
    # which is the honest shape: neither runs a circuit, so neither needs a loop.
    builder.add_node("consult", consult_tools)
    builder.add_node("explain", answer_in_prose)
    builder.add_node("implement", write_code)
    # Deferred, and this is the whole reason the fan-in works. The two branches are
    # of very different lengths -- the classical baseline is one node, the quantum
    # campaign is a loop -- so without this the verdict would be reached the moment
    # the shorter branch finished, on a campaign whose quantum runs had not happened
    # yet, and then reached again at the end. Deferring holds it until nothing else
    # is left to run.
    builder.add_node("skeptic", apply_skeptic, defer=True)
    builder.add_node("scribe", write_report)
    builder.add_node("suggest", suggest_next)
    builder.add_node("remember", remember_exchange)

    builder.add_edge(START, "recall")
    builder.add_edge("recall", "screen")
    builder.add_edge("screen", "interpret")
    builder.add_edge("interpret", "formalise")
    builder.add_conditional_edges(
        "formalise",
        after_formalise,
        {"retrieve": "retrieve", "conclude": "skeptic"},
    )

    # The branch point. A feasibility question fans out to the baseline and the
    # planning loop at once -- both start here and both must reach the skeptic. The
    # other two readings take one node each and rejoin at the scribe, skipping the
    # skeptic, which has nothing to argue against where no verdict is reached.
    builder.add_conditional_edges(
        "retrieve",
        after_retrieve,
        ["baseline", "plan", "consult", "converge"],
    )
    # The two prose branches are reached through the consultation, so the model gets
    # a chance to fetch what the corpus could not supply before anything is written.
    builder.add_conditional_edges(
        "consult",
        after_consult,
        {"explain": "explain", "implement": "implement"},
    )
    builder.add_edge("explain", "scribe")
    builder.add_edge("implement", "scribe")
    builder.add_conditional_edges(
        "converge",
        after_converge,
        ["baseline", "plan", "consult"],
    )

    builder.add_conditional_edges(
        "plan",
        after_plan,
        {"solve": "solve", "plan": "plan", "conclude": "skeptic"},
    )
    builder.add_edge("solve", "analyse")
    builder.add_conditional_edges(
        "analyse",
        after_analyse,
        {"plan": "plan", "conclude": "skeptic"},
    )
    builder.add_edge("baseline", "skeptic")

    builder.add_edge("skeptic", "scribe")
    builder.add_edge("scribe", "suggest")
    builder.add_edge("suggest", "remember")
    builder.add_edge("remember", END)

    return builder.compile()


def pipeline_mermaid(visited: Sequence[str] = ()) -> str:
    """Draw the compiled graph, highlighting the nodes a run went through.

    The diagram is generated by LangGraph from the graph object rather than
    maintained beside it, so it cannot describe a pipeline other than the one that
    runs. A node added to the agent appears here with no work, and a diagram that has
    drifted from the code is not a failure this can have.

    Args:
        visited: Node names that ran, in any order. Repeats are ignored -- the count
            belongs in a table, not on a box.

    Returns:
        Mermaid source, with a style line per visited node appended.
    """
    drawn = build_graph().get_graph().draw_mermaid()
    highlights = [
        f"    style {name} fill:#5B6CFF,stroke:#3D4CCC,color:#ffffff"
        for name in dict.fromkeys(visited)
        if not name.startswith("__")
    ]
    return drawn + "\n".join(highlights) + ("\n" if highlights else "")


NODE_RAN = "#5B6CFF"
NODE_LAST = "#2B3A67"
NODE_IDLE = "#e8eaed"
"""Three node states, because two of them were being confused.

A node that ran and a node that ran *last* look the same in a two-colour diagram,
and the difference is the most informative thing on the page: where a campaign
stopped says whether it concluded or was cut off. The third is everything the graph
offers and this question did not need, which is what makes the route a route rather
than a picture of the whole graph.
"""


TERMINAL_FILL = "#f1f3f4"
"""Fill for LangGraph's own ``__start__`` and ``__end__`` markers.

Neutral and distinct from all three node colours, because they are not steps the
agent runs and colouring them as visited or idle would say they were.
"""

TERMINAL_LABELS = {"__start__": "START", "__end__": "END"}
"""What LangGraph's markers are called on the page.

Drawn rather than dropped. They were filtered out, which left a diagram with no
visible entry point and no visible exit -- so a reader could not tell where a
question comes in, and the two nodes that end every run looked like ordinary
middles. Every LangGraph diagram in the documentation draws them, and a trace page
whose whole claim is *this is the graph itself* should not quietly edit it.
"""


# A node's box carries its name and nothing else. It has been three things: `plan\n6x`,
# which reads as a multiplier and so hid the loop count on a diagram whose subject is a
# loop; then `plan\nran 6 times`, which said it plainly but put a second line of text on
# every repeated box; and now the name alone. The colour already says whether a node ran,
# and the exact count for every node is one line below in the table -- which is where a
# reader who wants a number goes, and where it does not compete with the shape of the
# graph for a reader who wants the shape.


def pipeline_dot(visited: Sequence[str] = ()) -> str:
    """Draw the compiled graph as Graphviz DOT, highlighting the nodes a run used.

    The same graph object as :func:`pipeline_mermaid` and for the same reason -- it is
    generated rather than maintained, so it cannot describe a pipeline other than the
    one that runs. DOT rather than Mermaid because Graphviz renders natively in the
    interface: Mermaid has to be fetched from a content delivery network by the
    reader's browser, and a diagram that needs the internet to appear is a diagram
    that is sometimes just missing. The Mermaid source is still offered beside it,
    for pasting into anything that reads it.

    Everything drawn is read off the compiled graph -- the two terminal markers, the
    dashed conditional edges, the branch labels on them -- so what appears is the
    standard state-graph picture rather than a house style. A reader who knows
    LangGraph needs no explanation, and one who does not can be sent to its
    documentation.

    Args:
        visited: Node names that ran, in the order they ran. Repeats decide a box's
            colour and nothing else; the count itself belongs to the table under the
            diagram, for the reason given above this function.

    Returns:
        A DOT digraph.
    """
    drawn = build_graph().get_graph()
    counts: dict[str, int] = {}
    for name in visited:
        counts[name] = counts.get(name, 0) + 1
    lines = [
        "digraph {",
        '  rankdir=LR; bgcolor="transparent"; pad=0.2;',
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10 penwidth=0];',
        '  edge [color="#9aa0a6" fontname="Helvetica" fontsize=8];',
    ]
    for marker, label in TERMINAL_LABELS.items():
        if marker in drawn.nodes:
            lines.append(
                f'  "{marker}" [label="{label}" shape=oval fillcolor="{TERMINAL_FILL}" '
                'fontcolor="#5f6368" fontsize=9]'
            )
    ended_at = next((name for name in reversed(list(visited)) if not name.startswith("__")), "")
    for name in pipeline_nodes():
        ran = counts.get(name, 0)
        if name == ended_at:
            fill, text = NODE_LAST, "#ffffff"
        elif ran:
            fill, text = NODE_RAN, "#ffffff"
        else:
            fill, text = NODE_IDLE, "#5f6368"
        lines.append(f'  "{name}" [label="{name}" fillcolor="{fill}" fontcolor="{text}"]')
    for edge in drawn.edges:
        # A conditional edge is where the routing decision is, so it is dashed, and
        # where LangGraph carries a name for the branch that name is drawn on it.
        # Without the labels the three edges leaving `plan` are indistinguishable,
        # and they are the difference between running, retrying and giving up.
        style = ["style=dashed"] if edge.conditional else []
        if edge.data:
            style.append(f'label=" {edge.data} "')
        decoration = f" [{' '.join(style)}]" if style else ""
        lines.append(f'  "{edge.source}" -> "{edge.target}"{decoration}')
    lines.append("}")
    return "\n".join(lines)


def pipeline_nodes() -> tuple[str, ...]:
    """List the agent's own nodes, without LangGraph's start and end markers.

    Returns:
        The node names, in the order the graph declares them.
    """
    return tuple(name for name in build_graph().get_graph().nodes if not name.startswith("__"))


def _configure_tracing_quietly() -> bool:
    """Point LangSmith at the configured project, and never fail an answer for it.

    Observability is not allowed to cost a result -- the same rule
    :func:`src.logging_setup.observing_calls` applies to the token meter. An
    unconfigured process has no settings to read at all, which is a supported way
    to run this and must not raise here.

    Returns:
        Whether tracing is now on.
    """
    try:
        return configure_tracing()
    except Exception:  # pragma: no cover - defensive; settings validate at startup
        get_logger("tracing").warning("tracing_not_configured", extra={"reason": "no_settings"})
        return False


def run_campaign(
    question: str,
    framing: Literal["neutral", "vendor", "skeptical"] = "neutral",
    shot_budget: int = DEFAULT_SHOT_BUDGET,
    coherence: CoherenceBudget | None = None,
    device: Device | None = None,
    chat_model: BaseChatModel | Literal["auto"] | None = AUTO_MODEL,
    models: ModelPool | None = None,
    search_corpus: bool = True,
    fetch_external: bool = True,
    search_tuning: SearchTuning | None = None,
    suggest_followups: bool = True,
    audience: Audience = DEFAULT_AUDIENCE,
    precision_per_site: float | None = None,
    circuit_depth: int | None = None,
    chain_defaults: reading.Reading | None = None,
    memory: Memory | None = None,
    parallel_model_calls: bool | None = None,
    reference_bench: FieldSweepBench | None = None,
    visited: list[str] | None = None,
    on_node: Callable[[str, CampaignState], None] | None = None,
    prose_sink: Callable[[str], None] | None = None,
) -> CampaignState:
    """Run one campaign from a question to a written verdict.

    Args:
        question: The problem, in the asker's own words.
        framing: The voice it was asked in. Recorded, never acted on.
        shot_budget: Total measurements the campaign may spend.
        coherence: The depth limit to refuse against. Defaults to the machine's own
            clock when ``device`` is given, and to the nominal figures otherwise.
        device: The machine to price circuits against. When given, plans are placed
            and routed on its wiring, so one can be refused for connectivity as well
            as for depth. ``None`` prices circuits in the abstract.
        chat_model: The model for every call. ``"auto"`` takes whatever the settings
            provide; ``None`` runs the deterministic paths only. Ignored when
            ``models`` is given.
        models: A prepared model per tier -- see :mod:`src.agent.model_selection`.
        search_corpus: Whether to look for background sources. Off costs the report
            its citations and nothing else.
        search_tuning: How hard to search. ``None`` takes the defaults.
        suggest_followups: Whether to propose what to ask next. A dial because a
            suggestion is a model call.
        audience: Who the prose is written for. Changes wording only; every number,
            refusal and verdict is identical at every level.
        precision_per_site: Target energy accuracy per magnet. ``None`` prices
            against :data:`TARGET_ENERGY_ERROR` as a total. This one moves numbers:
            ten times the accuracy costs a hundred times the measurements.
        circuit_depth: Layers to assume when a question describes a circuit but
            names no depth. ``None`` falls back to
            :data:`~src.physics.quantum.ansatz.DEFAULT_DEPTH`.
        chain_defaults: The chain a caller already has on screen, which fills any
            term the question and the conversation left unsaid -- so a host with
            visible controls answers about the chain those controls describe rather
            than about a constant. ``None`` keeps the project defaults, which is
            what a script or an eval wants. It supplies values only: whether the
            length was *given* still depends on the words, because that is what
            decides whether a verdict may be reached.
        fetch_external: Whether a search that finds nothing locally may reach an
            external index. Only reached on that path.
        reference_bench: The exact field sweep, lent by a host that has it, so a
            question about a curve gets the real answer. Sealed from ``src/agent/``,
            hence a callable rather than an import -- see :data:`BENCH_KEY`. ``None``
            leaves the tool out and the agent answers from the notes.
        visited: A list to append each node's name to as it runs. Given one, the
            graph is streamed rather than invoked; the result is identical. Owned by
            the caller so the path survives a campaign that raises partway.
        on_node: Called with each node's name and the campaign after it, as each
            finishes. Drives progress display and lets a result be drawn before the
            last node ends. Never given a name the graph did not run.
        prose_sink: Receives the explanation as the model writes it. Raw -- sources
            still attached, mathematics unrepaired -- so it is a progress signal, not
            a draft to display. ``None`` makes the call in one piece.
        memory: What the agent remembers of this conversation. ``None`` runs with no
            history and leaves nothing behind.
        parallel_model_calls: Whether the three opening model calls are issued
            together. ``None`` takes the configured default. It moves no decision and
            changes no answer, only the wait. Off keeps the trace sequential.

    Returns:
        The finished campaign, carrying the verdict, the report, and every run and
        rejection that led to them.
    """
    # Tracing is configured here because this is where a campaign actually begins.
    # It was written to be called from the entry point and never was, so
    # `langsmith_tracing` was a switch that set nothing: LANGSMITH_TRACING never
    # reached the environment LangChain reads it from, and a deployment with a key
    # and the switch on still traced nothing. Idempotent, so calling it per
    # campaign is the same work as calling it once.
    _configure_tracing_quietly()
    graph = build_graph()
    if coherence is None and device is not None:
        coherence = coherence_budget(device)
    opening = new_campaign(
        request=Request(text=question, framing=framing),
        shot_budget=shot_budget,
        coherence=coherence,
    )
    configurable: dict[str, Any] = {
        SEARCH_KEY: search_corpus,
        FETCH_KEY: fetch_external,
        SEARCH_TUNING_KEY: search_tuning or SearchTuning(),
        FOLLOWUPS_KEY: suggest_followups,
        AUDIENCE_KEY: audience,
    }
    if precision_per_site is not None:
        configurable[PRECISION_KEY] = float(precision_per_site)
    if circuit_depth is not None:
        configurable[CIRCUIT_DEPTH_KEY] = int(circuit_depth)
    if chain_defaults is not None:
        configurable[DIALS_KEY] = chain_defaults
    if device is not None:
        configurable[DEVICE_KEY] = device
    if reference_bench is not None:
        configurable[BENCH_KEY] = reference_bench
    if models is not None:
        configurable[POOL_KEY] = models
    elif chat_model != AUTO_MODEL:
        configurable[MODEL_KEY] = chat_model
    if memory is not None:
        configurable[RECALL_KEY] = memory.as_context()
        configurable[MEMORY_KEY] = memory
    ahead = Prefetch(
        enabled=_parallel_by_default() if parallel_model_calls is None else parallel_model_calls
    )
    configurable[PREFETCH_KEY] = ahead
    if prose_sink is not None:
        configurable[PROSE_SINK_KEY] = prose_sink
    config: RunnableConfig = {"configurable": configurable}
    try:
        if visited is None and on_node is None:
            final: CampaignState = graph.invoke(opening, config=config)  # type: ignore[assignment]
        else:
            final = _streamed(graph, opening, config, visited, on_node)
    finally:
        # In a `finally` because a campaign that raises has still left calls in flight,
        # and a thread pool nobody shuts down keeps a non-daemon thread -- and the
        # process behind it -- alive after the traceback has already been printed.
        ahead.close()
    return final


def _parallel_by_default() -> bool:
    """Whether to overlap the front-of-graph calls when the caller did not say.

    Read from configuration rather than defaulted in the signature, so the answer is
    one a deployment set rather than one hidden in a keyword argument. A configuration
    that cannot be read at all -- no credential, which is every offline test -- gets
    the overlap, because with no model to call there is nothing to overlap and the
    answer is the same either way.

    Returns:
        The configured default.
    """
    try:
        return bool(get_settings().parallel_model_calls)
    except Exception:
        return True


def _streamed(
    graph: CompiledStateGraph,
    opening: CampaignState,
    config: RunnableConfig,
    visited: list[str] | None = None,
    on_node: Callable[[str, CampaignState], None] | None = None,
) -> CampaignState:
    """Run the graph one node at a time, recording which nodes ran.

    LangGraph can emit both the per-node updates and the accumulated state as it
    goes. Both are asked for here rather than just the updates, because rebuilding
    the state from the updates would mean reimplementing the reducers -- and a second
    implementation of a merge rule is a second thing that can disagree with the
    first.

    Args:
        graph: The compiled graph.
        opening: The campaign as it starts.
        config: The runtime configuration.
        visited: A list to append node names to, in the order they ran, or ``None``
            to record nothing.
        on_node: Called with each node's name and the state after it, or ``None``.
            An exception from it is deliberately not caught: a progress display that
            raises is a bug in the display, and swallowing it here would leave a
            campaign that looks stuck for the rest of its run with nothing in the log
            to say why.

    Returns:
        The finished campaign, identical to what ``invoke`` would have returned.
    """
    final: CampaignState = opening
    # The two streams interleave in a fixed order: the names of the nodes that just
    # ran, then the state those nodes produced. So a name is held until the state
    # arrives and the pair is reported together -- otherwise a caller drawing a
    # result on the strength of a node's name would draw it from the state as it was
    # *before* that node ran, which is the one state where the result is missing.
    just_ran: list[str] = []
    for mode, payload in graph.stream(opening, config=config, stream_mode=["updates", "values"]):
        if mode == "updates" and isinstance(payload, dict):
            just_ran.extend(payload)
            if visited is not None:
                visited.extend(payload)
        elif mode == "values":
            final = payload  # type: ignore[assignment]
            if on_node is not None:
                for node in just_ran:
                    on_node(node, final)
            just_ran.clear()
    return final


def _tuning_from(config: RunnableConfig) -> SearchTuning:
    """Read the retrieval dials off the graph's configuration.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The dials, or the defaults when nothing set them -- which is every caller
        that does not care, including every test that is about something else.
    """
    configurable = config.get("configurable") or {}
    held = configurable.get(SEARCH_TUNING_KEY)
    return held if isinstance(held, SearchTuning) else SearchTuning()


def _audience_from(config: RunnableConfig) -> Audience:
    """Read who the answer is being written for off the graph's configuration.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The reading level, or the project default when nothing set one.
    """
    configurable = config.get("configurable") or {}
    held = configurable.get(AUDIENCE_KEY)
    return cast("Audience", held) if held in get_args(Audience) else DEFAULT_AUDIENCE


def _target_error_for(state: CampaignState, config: RunnableConfig) -> float:
    """The total energy accuracy the shot arithmetic should be priced against.

    The interface asks for an accuracy *per magnet*, because that is the quantity a
    reader can compare between chains of different lengths -- a total error of
    ``0.01`` means something quite different on four magnets and on twenty. The
    chain length is not known when the dial is set, so the multiplication happens
    here, against the chain the campaign actually formalised.

    Args:
        state: The campaign, carrying the formalised chain.
        config: The graph's runtime configuration.

    Returns:
        The absolute energy error to price against. Falls back to
        :data:`TARGET_ENERGY_ERROR` when nothing set a dial, which is what every
        caller that predates it passes.
    """
    configurable = config.get("configurable") or {}
    per_site = configurable.get(PRECISION_KEY)
    model = state["model"]
    if not isinstance(per_site, float) or per_site <= 0.0 or model is None:
        return TARGET_ENERGY_ERROR
    return per_site * model.n_sites


def _followups_allowed(config: RunnableConfig) -> bool:
    """Whether the agent may propose what to ask next.

    Args:
        config: The graph's runtime configuration.

    Returns:
        ``True`` unless something switched it off. Defaulting to on means a caller
        that predates the dial keeps the behaviour it had.
    """
    configurable = config.get("configurable") or {}
    return bool(configurable.get(FOLLOWUPS_KEY, True))


def _search_allowed(config: RunnableConfig) -> bool:
    """Whether this run may reach the corpus index.

    Args:
        config: The graph's runtime configuration.

    Returns:
        ``True`` unless a caller explicitly switched the search off.
    """
    configurable = config.get("configurable") or {}
    return bool(configurable.get(SEARCH_KEY, True))


def _recalled(config: RunnableConfig) -> str:
    """Read what the caller remembered of this conversation.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The rendered history, or the empty string when the campaign was run
        without a memory -- which is the default and the mode every test uses.
    """
    configurable = config.get("configurable") or {}
    recalled = configurable.get(RECALL_KEY, "")
    return recalled if isinstance(recalled, str) else ""


def _dials_from(config: RunnableConfig) -> reading.Reading | None:
    """Read the chain the caller has on screen, if the caller has one.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The dial positions as a reading, or ``None`` when nothing was offered --
        which is every campaign run from a script or a test, and leaves the project
        defaults in place. See :data:`DIALS_KEY`.
    """
    configurable = config.get("configurable") or {}
    dials = configurable.get(DIALS_KEY)
    return dials if isinstance(dials, reading.Reading) else None


def _bench_from(config: RunnableConfig) -> FieldSweepBench | None:
    """Find the exact field sweep a host lent this run, if it lent one.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The callable, or ``None``. Checked for callability rather than for a type,
        because the point of the arrangement is that this package cannot name the
        module the real one comes from -- see :data:`BENCH_KEY`.
    """
    configurable = config.get("configurable") or {}
    lent = configurable.get(BENCH_KEY)
    return cast("FieldSweepBench", lent) if callable(lent) else None


def _device_from(config: RunnableConfig) -> Device | None:
    """Find the machine this campaign is pricing against, if there is one.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The device, or ``None`` when the campaign is reasoning about the algorithm
        rather than about a machine.
    """
    configurable = config.get("configurable") or {}
    device = configurable.get(DEVICE_KEY)
    return device if isinstance(device, Device) else None


def _circuit_depth_from(config: RunnableConfig) -> int:
    """Find the depth to assume for a question that named no layers.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The interface's depth knob when one was passed, and
        :data:`~src.physics.quantum.ansatz.DEFAULT_DEPTH` otherwise. Never zero: a
        question describing a circuit has described at least one layer.
    """
    configurable = config.get("configurable") or {}
    depth = configurable.get(CIRCUIT_DEPTH_KEY)
    if isinstance(depth, int) and depth > 0:
        return depth
    return DEFAULT_CIRCUIT_DEPTH


def _prose_sink_from(config: RunnableConfig) -> Callable[[str], None] | None:
    """Read the caller's somewhere-to-stream-the-answer off the configuration.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The sink, or ``None`` when the caller offered none -- which is every caller
        that is not a live interface, and the path every test takes.
    """
    configurable = config.get("configurable") or {}
    held = configurable.get(PROSE_SINK_KEY)
    return held if callable(held) else None


def _prefetch_from(config: RunnableConfig) -> Prefetch:
    """Find this run's head start, or a disabled one that changes nothing.

    A campaign run outside :func:`run_campaign` -- a node exercised on its own in a
    test, a graph invoked directly -- carries no carrier, and gets one that runs every
    call in place. That is the sequential behaviour this project had before, reached by
    the same code, which is what makes the switch honest.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The carrier. Always one, never ``None``.
    """
    configurable = config.get("configurable") or {}
    existing = configurable.get(PREFETCH_KEY)
    return existing if isinstance(existing, Prefetch) else Prefetch(enabled=False)


def _pool_from(config: RunnableConfig) -> ModelPool:
    """Find the models this run may call, as one pool.

    Four cases, and they are genuinely four. A pool was handed in and is used --
    which is how a person's choice of model reaches the graph. A single model was
    handed in, and every task is served by it. ``None`` was handed in, meaning run
    without any model, and no attempt is made to find one. Nothing was handed in,
    meaning work it out from configuration, which is what production does and what
    a test must never do, since it would put a network call inside the suite.

    Args:
        config: The graph's runtime configuration.

    Returns:
        The pool. Always a pool, never ``None``: an offline pool answers every
        request with ``None`` and so a caller has one shape to handle rather than
        two.
    """
    configurable = config.get("configurable") or {}
    existing = configurable.get(POOL_KEY)
    if isinstance(existing, ModelPool):
        return existing
    if MODEL_KEY in configurable:
        supplied: BaseChatModel | None = configurable[MODEL_KEY]
        return ModelPool(offline=supplied is None, pinned=supplied)
    return ModelPool()


def _propose_model(text: str, pool: ModelPool, recalled: str = "") -> ModelProposal | None:
    """Ask a language model to read the problem's numbers out of the request.

    History is included when there is any, and only then. "And at twice the field?"
    is a complete request to a person who heard the previous one and an unreadable
    one to a model that did not, which is the whole case for memory -- but an empty
    history section still costs tokens and still invites the model to explain that
    there is no history, so the section is omitted rather than left blank.

    Both parts are wrapped as data. The history is text this application wrote, but
    it is text *derived from* what a user typed, and a boundary that only holds for
    material arriving from outside is a boundary an attacker crosses by waiting one
    turn.

    Args:
        text: The request, verbatim.
        pool: The campaign's models.
        recalled: Earlier turns in this conversation, already rendered. Empty when
            there is no memory or nothing to recall.

    Returns:
        The proposal, or ``None`` when there is no model or the reply was unusable.
        Both are the same outcome for the caller: fall back.
    """
    payload = as_data(text)
    if recalled:
        payload = f"Earlier in this conversation:\n{as_data(recalled)}\n\nThe request:\n{payload}"
    return pool.invoke("problem_reading", ModelProposal, PROBLEM_READING, payload)


def _usable_coupling(coupling: float | None) -> tuple[float, str | None]:
    """The interaction strength a solver will accept, and what settling on it assumed.

    Every solver in the project requires :math:`J > 0`: the closed form, the sparse
    diagonalisation and the Pauli construction all refuse anything else, and they are
    right to -- :math:`J = 0` is not a lattice of interacting spins at all.

    So the reading has to be brought inside that range *before* a model is built, and
    this is the one function that does it. It exists because the same rule was written
    twice and then only maintained once: the path with a language model on it clamped
    the coupling, the offline pattern-reading path passed it straight through, and *a
    6-spin chain with J=0* therefore travelled four nodes into the campaign before
    raising ``ValueError: coupling J must be strictly positive`` out of the depth
    ladder -- past the point where anything could still answer the question.

    Args:
        coupling: What the sentence or the model gave, or ``None`` when neither named
            one.

    Returns:
        The coupling to use, and the assumption to record beside it, or ``None`` when
        nothing had to be assumed. A note here is the whole value of the function: a
        number quietly changed on the way to a solver is a verdict about a problem
        nobody asked about.
    """
    if coupling is None or coupling > 0.0:
        return (1.0 if coupling is None else coupling), None
    if coupling == 0.0:
        return 1.0, (
            "the request put the neighbour interaction at zero, which leaves the spins "
            "independent rather than a lattice, so a unit interaction was assumed -- an "
            "assessment of uncoupled spins would say nothing about the question asked"
        )
    return abs(coupling), (
        f"the request gave a neighbour interaction of {coupling:g} and its magnitude "
        f"{abs(coupling):g} was assessed. On a shape that two-colours -- a chain, a "
        "square lattice -- the sign is a relabelling of one sublattice and the energies "
        "are unchanged, so this is the same problem; on a shape that does not, it "
        "describes a frustrated problem that nothing here solves"
    )


def _clamp_model(
    proposal: ModelProposal | None,
    found: reading.Reading | None = None,
    dials: reading.Reading | None = None,
) -> FormalModel:
    """Turn a proposal into a model the rest of the project can actually run.

    Every bound applied here is recorded as an assumption rather than applied
    silently. A chain quietly shortened from a hundred sites to twelve would make
    every number downstream answer a different question than the one asked, and the
    only defence against that is saying so where the reader will see it.

    Args:
        proposal: What the model suggested, or ``None`` if there was no model.
        found: What :mod:`src.agent.reading` could read out of the sentence by
            pattern. Used when there was no model, so that an offline campaign is
            about the chain that was actually asked about rather than about a fixed
            default -- which it was, until this argument existed, and answering a
            question nobody asked is a worse failure than answering vaguely.
        dials: The chain the caller has on screen. Every term the sentence left
            unsaid is taken from it in preference to anything the model proposed --
            the same rule this function already applies to the length, the geometry
            and the boundary, and for the same reason: a value nobody stated is not
            the model's to invent while somebody has it set in front of them. See
            :data:`DIALS_KEY`.

    Returns:
        A runnable model, with its assumptions attached.
    """
    if proposal is None:
        return _from_pattern(found or reading.Reading(), dials)

    asked = found or reading.Reading()
    filled, dial_notes = _with_dials(asked, dials)
    assumptions = list(proposal.assumptions)
    assumptions.extend(dial_notes)
    n_sites = proposal.n_sites
    if n_sites > MAX_SITES_STATEVECTOR:
        assumptions.append(
            f"the request describes {n_sites} elements; this assessment simulates "
            f"{MAX_SITES_STATEVECTOR}, the largest circuit this can carry, so the "
            "results describe a smaller version of the problem than the one asked about"
        )
        n_sites = MAX_SITES_STATEVECTOR
    if n_sites < 2:
        assumptions.append(
            f"the request implied {proposal.n_sites} elements, which is not a chain; "
            "two was used as the smallest meaningful size"
        )
        n_sites = 2

    # A length the sentence did not state is not the model's to invent. Found live:
    # "How can a circuit run imaginary time, which is not unitary?" -- a question
    # about a method, naming no chain -- came back formalised as **two** magnets,
    # because the model had nothing to read and returned something small that the
    # floor above then rounded up. A two-spin chain has one bond and no interior, and
    # printing "read the request as TFIM L=2" under an answer about an algorithm is
    # both wrong and silly. The offline reader has always used the project default
    # here, so this is the same rule applied on the path that has a model on it -- the
    # third such case in this function, after geometry and a stated ratio.
    if asked.n_sites is None:
        if filled.n_sites is not None:
            # The dials named a length, and the note saying so is already in the list.
            n_sites = filled.n_sites
        else:
            if n_sites != DEFAULT_SITES:
                # Word for word what the offline path says, so the two cannot be told
                # apart by their assumptions -- the promise `formalise` makes.
                assumptions.append(
                    "the request named no chain length, so six elements were assumed"
                )
            n_sites = DEFAULT_SITES

    coupling, coupling_note = _usable_coupling(
        filled.coupling
        if asked.coupling is None and filled.coupling is not None
        else proposal.coupling
    )
    if coupling_note is not None:
        assumptions.append(coupling_note)

    # A transverse field of zero is not a transverse-field Ising model. Every term
    # commutes, the ground state is a single arrangement of magnets, and there is
    # nothing quantum left to assess -- so a campaign run on it produces a confident,
    # correct, entirely vacuous verdict. It reached here because a question that named
    # no field left the model free to propose one, and "0" is a plausible-looking
    # number to propose. The offline reader has always taken a missing field as equal
    # to the coupling, which is the hardest case and so the cautious one; this is the
    # same rule, applied on the path that has a language model on it.
    transverse_field = abs(
        filled.field
        if asked.field is None and filled.field is not None
        else proposal.transverse_field
    )
    if transverse_field == 0.0:
        assumptions.append(
            "no sideways field strength was stated, so it was taken equal to the "
            "neighbour interaction -- at zero the problem is not a quantum one at all, "
            "and an assessment of it would say nothing about the question asked"
        )
        transverse_field = coupling

    # A sentence that states only a *ratio* between the two strengths is read by
    # pattern too, and for the same reason as geometry: the two paths disagreed. An
    # energy is not a pure number until the unit is fixed, and "five times" fixes no
    # unit -- so J=5,h=1 and J=1,h=0.2 are the same physics and different energies.
    # A language model picks either, and the run then answers a question graded at
    # the other scale: `weak-field-6` came back at -2.83 per magnet against an exact
    # -0.85, which reads as a variational bound violated rather than as two answers
    # about two scales. The pattern reader normalises to J=1, which is the
    # literature's convention, so it wins here.
    found_reading = filled
    if (
        found_reading.strengths_from_ratio
        and found_reading.coupling is not None
        and found_reading.field is not None
    ):
        stated = (found_reading.coupling, found_reading.field)
        if (coupling, transverse_field) != stated:
            assumptions.append(
                "the request compared the two strengths without giving either a "
                f"number, so the neighbour interaction was set to {stated[0]:g} and "
                f"the sideways field to {stated[1]:g} -- the ratio is what was "
                "asked about, and this is the scale it is conventionally written on"
            )
        coupling, transverse_field = stated

    # The boundary is the pattern reader's too, and this was the one field in this
    # function taken from the proposal unchecked. A ring is the marked case -- a
    # sentence that means one says so -- and *a 10-spin critical chain* does not, so
    # the flagship starter was being formalised as a ring. On the default machine, a
    # linear one, closing a ring costs sixteen routing moves per circuit: the shallowest
    # rung on the ladder came out at 11.2 microseconds against a 10-microsecond window,
    # was refused on coherence, and set a ceiling nothing below could satisfy. The
    # planning loop then had nothing left to propose and concluded without running a
    # single configuration, so `plan -> solve -> analyse` never happened for the one
    # question that exists to demonstrate it.
    boundary = found_reading.boundary
    if boundary is None:
        boundary = "open"
        # Said whenever the sentence was silent, rather than only when the model's
        # guess had to be overridden: what a reader needs to know is that a line was
        # assumed, not that a proposal disagreed. Word for word what the offline path
        # says, so the two readings cannot be told apart by their assumptions.
        assumptions.append(
            "the request did not say whether the chain is a line or a ring, so a "
            "line was assumed -- which is what hardware looks like"
        )

    # Geometry comes from the pattern reader alone, for the same reason
    # `length_was_given` does: `ModelProposal` has no shape field, so a language model
    # cannot report one. Without this the two paths disagreed -- an offline campaign
    # assessed the 4x4 square that was asked about and an online one assessed a
    # sixteen-element chain, which is the same failure as answering about the wrong
    # length and harder to spot, because the site count looks right.
    n_sites, rows, shape_notes = _shape_from(found_reading.with_sites(n_sites))
    assumptions.extend(shape_notes)

    return FormalModel(
        n_sites=n_sites,
        coupling=coupling,
        transverse_field=transverse_field,
        longitudinal_field=(
            filled.longitudinal
            if asked.longitudinal is None and filled.longitudinal is not None
            else proposal.longitudinal_field
        ),
        boundary=boundary,
        geometry=filled.geometry or "chain",
        rows=rows,
        assumptions=tuple(assumptions),
        # Taken from the pattern reader rather than from the proposal, because a
        # language model always returns a length and so can never report that the
        # question did not carry one. The reader is laid over the conversation first,
        # so a follow-up inherits a length the same way a person would.
        length_was_given=found is not None and found.n_sites is not None,
    )


def _search(
    query: str,
    routing: Routing,
    pool: ModelPool,
    tuning: SearchTuning | None = None,
) -> tuple[tuple[Citation, ...], tuple[SearchRound, ...]]:
    """Search the corpus the way the router decided, and cite what survives grading.

    Every stage the retrieval layer offers is used here, and each is cheap until
    it is needed. The router's shelf narrows the first round. The keyword index is
    fused with the vector search so a passage naming an author is findable even
    when nothing paraphrases the question. The grader and the rewriter escalate to
    a model only when a round kept nothing, so a question the notes already answer
    pays for none of it.

    Args:
        query: What to search for.
        routing: What was decided about where to look.
        pool: The campaign's models, for the stages that may escalate.
        tuning: How hard to search. A shelf named here overrides the router's
            choice, which is the point of naming one: the router is a good guess
            and this is somebody who knows better.

    Returns:
        Two things: the passages found reduced to citations, and the record of
        every round that was run to get them. The citations are empty when the
        notes do not cover the question, and equally empty when there is no index
        at all -- a distinction the campaign does not act on differently, since
        both leave the verdict resting on its own measurements.

        The rounds are returned rather than logged because they are the only
        surviving evidence of a search that kept nothing: a round that found six
        passages and threw all six out leaves no citation behind it, and it is
        precisely the round a reader wanting to know why an answer is thin needs
        to see. Discarding them here is what made the query rewriting invisible
        everywhere in the interface, despite the loop having performed it.
    """
    if not routing.needs_retrieval:
        return (), ()
    try:
        from src.rag.lexical import default_index
        from src.rag.retrieve import retrieve

        dials = tuning or SearchTuning()
        found = retrieve(
            query,
            topics=routing.topics,
            shelves=dials.shelves or routing.shelves,
            grader=model_grader(pool),
            rewriter=model_rewriter(pool),
            # Four model-backed stages, and only one of them runs on a question
            # that goes well. Expansion runs first and always, because a
            # vocabulary gap cannot be detected without searching; reranking runs
            # on a set worth ordering; grading and rewriting escalate only when a
            # round kept nothing. All four answer with identifiers or short
            # strings, all four fall back to a deterministic route, and a campaign
            # with no credential at all still retrieves -- which is what the
            # offline tests exercise.
            expander=model_expander(pool),
            reranker=model_reranker(pool),
            lexical=default_index(),
            limit=dials.passages,
            vector_share=dials.vector_share,
            rounds=dials.rounds,
        )
    except Exception as error:
        _logger.warning("campaign_retrieve_unavailable", extra={"detail": type(error).__name__})
        return (), ()
    citations = tuple(
        Citation(
            identifier=passage.arxiv or passage.identifier,
            title=passage.title,
            snippet=clip(passage.text, SNIPPET_CHARACTERS),
            shelf=passage.shelf,
        )
        for passage in found.passages
    )
    rounds = tuple(
        SearchRound(
            query=attempt.query,
            kind=attempt.kind,
            found=attempt.found,
            kept=attempt.kept,
            reason=attempt.reason,
            shelves=attempt.shelves,
            rewritten_by=attempt.rewritten_by,
        )
        for attempt in found.attempts
    )
    return citations, rounds


def _fetch_background(
    query: str,
    routing: Routing,
    config: RunnableConfig,
) -> tuple[Citation, ...]:
    """Look outside the corpus, once, when the corpus held nothing.

    Reached only after a local search came back empty, which is what keeps it off
    the path of every question the notes already answer. What comes back is
    screened before it is written down and cited as unreviewed, because it is:
    the text is real and nobody has read it.

    The candidates serve the current campaign directly from memory rather than
    being searched for again. They are already in hand, and a round trip through
    the index would buy nothing except the wait for an embedding call.

    Args:
        query: The search that found nothing locally.
        routing: What was decided about where to look. Its shelf is where a
            promoted note is filed.
        config: The graph's runtime configuration, which is where the external
            reach can be switched off.

    Returns:
        Citations for whatever was admitted, empty if nothing was.
    """
    if not _fetch_allowed(config):
        return ()
    try:
        from src.rag.external import admit, fetch, promote

        shelf = routing.shelves[0] if routing.shelves else DEFAULT_FETCH_SHELF
        candidates = fetch(query, shelf, routing.topics, limit=SOURCES_WANTED)
    except Exception as error:
        _logger.warning("campaign_fetch_unavailable", extra={"detail": type(error).__name__})
        return ()

    citations: list[Citation] = []
    for candidate in candidates:
        if not admit(candidate).admitted:
            continue
        # Written to disk as well as used here, so the next campaign asking a
        # question like this one finds it in the index rather than fetching again.
        promote(candidate)
        citations.append(
            Citation(
                identifier=candidate.identifier,
                title=candidate.title,
                snippet=clip(candidate.body, SNIPPET_CHARACTERS),
                shelf=candidate.shelf,
                reviewed=False,
            )
        )
    return tuple(citations)


def _fetch_allowed(config: RunnableConfig) -> bool:
    """Whether this run may reach an external index when the corpus is silent.

    Args:
        config: The graph's runtime configuration.

    Returns:
        ``True`` unless a caller switched it off. Off by default in any run that
        also switched the corpus search off, because a caller who did not want an
        index consulted did not want a network call either.
    """
    configurable = config.get("configurable") or {}
    if not _search_allowed(config):
        return False
    return bool(configurable.get(FETCH_KEY, True))


def _next_depth(state: CampaignState, pool: ModelPool | None = None) -> int | None:
    """Choose the next depth to try, preferring a language model's suggestion.

    The suggestion is advisory in the strict sense: it is discarded unless it names
    a depth that is positive, untried, and not already ruled out. A planner whose
    suggestions are always accepted is not being guarded by anything.

    Args:
        state: The campaign so far.
        pool: The models to consult, or ``None`` to walk the ladder instead.

    Returns:
        The depth, or ``None`` when every candidate has been tried or rejected.
    """
    used = {run.depth for run in state["runs"]}
    used |= {_depth_from_label(item.label) for item in state["ruled_out"]}
    ceiling = _refused_above(state)
    candidates = [depth for depth in DEPTH_LADDER if depth not in used and depth < ceiling]
    # Asked *after* the candidates are known, and only when there are some. A ladder
    # with nothing left below the ceiling has one honest answer -- stop -- and asking
    # a model to propose a depth that will be refused on arithmetic the moment it
    # arrives costs a round trip with a person waiting and cannot change the verdict.
    if not candidates:
        return None
    suggested = _suggest_depth(state, pool) if pool is not None else None
    if suggested is not None and 0 < suggested < ceiling and suggested not in used:
        return suggested
    return candidates[0]


def _refused_above(state: CampaignState) -> int:
    """The depth at or above which nothing can be afforded any more.

    Every reason a configuration is refused here grows with depth. A circuit that
    does not fit the machine's wiring does not fit it at any depth; one already past
    the coherence limit is further past it with another layer; and the measurements
    an optimisation costs rise with the number of parameters. So the shallowest
    refusal is a ceiling: nothing at or above it can succeed, and a ladder that keeps
    climbing past one is proposing configurations whose refusal is already known.

    This is a correctness fix that happens to be a latency fix. Before it, a run
    whose second rung was refused on cost went on to propose the third, the fourth and
    the fifth -- each one a model call, each one refused for the same arithmetic --
    and the trail filled with rejections that told a reader nothing except that the
    loop had not noticed. Now the loop stops at the first one and says so.

    Args:
        state: The campaign so far.

    Returns:
        The shallowest refused depth, or a ceiling above every rung of the ladder
        when nothing has been refused yet.
    """
    refused = [
        depth for depth in (_depth_from_label(item.label) for item in state["ruled_out"]) if depth
    ]
    return min(refused) if refused else max(DEPTH_LADDER) + 1


def _suggest_depth(state: CampaignState, pool: ModelPool) -> int | None:
    """Ask a language model which depth to try next, given what has happened.

    Args:
        state: The campaign so far.
        pool: The campaign's models.

    Returns:
        The suggested depth, or ``None`` when there is no model, the reply was
        unusable, or there is nothing yet to reason from.
    """
    if not state["runs"]:
        return None

    history = "\n".join(
        f"{run.label}: {run.depth} layers, energy per element "
        f"{run.energy_per_site:.6f}, outcome {run.diagnosis.signal}"
        for run in state["runs"]
    )
    rejected = "\n".join(
        f"{item.label}: rejected because {item.reason}" for item in state["ruled_out"]
    )
    proposal = pool.invoke(
        "depth_suggestion",
        DepthProposal,
        DEPTH_SUGGESTION,
        f"Tried so far:\n{history}\n\nRejected:\n{rejected or 'nothing'}",
    )
    return None if proposal is None else proposal.depth


def _plan_reason(state: CampaignState, depth: int) -> str:
    """Say why this depth is the next one, in one sentence.

    Args:
        state: The campaign so far.
        depth: The depth chosen.

    Returns:
        The reason, which is the first attempt's rationale if nothing has run yet
        and a comparison against the last run otherwise.
    """
    if not state["runs"]:
        return "the shallowest circuit worth running, to establish what one layer achieves"
    previous = state["runs"][-1]
    return (
        f"{previous.label} reached {previous.energy_per_site:.6f} per element; "
        f"{depth} layers is the next step up the ladder"
    )


def _price_shots(
    state: CampaignState, depth: int, target_error: float, noise_inflation: float = 1.0
) -> int:
    r"""Work out what one optimisation at this depth would cost in measurements.

    Three factors multiplied. One energy reading to accuracy :math:`\epsilon` costs
    :math:`(\sum_\alpha |c_\alpha| / \epsilon)^2` shots, because the error of an
    average falls as the square root of the sample count and so the cost rises as
    its square. An optimisation needs many readings, estimated from the depth. Noise
    damps the reading towards zero, so undoing it divides the error by the surviving
    fidelity and multiplies the count by :math:`1/F^2`.

    The operator is built on the problem's own lattice. A square or triangular
    cluster carries more bonds than a chain of the same site count, so
    :math:`\sum_\alpha |c_\alpha|` is larger and the shots go up as its square:
    pricing a 3x3 triangular cluster as a nine-site chain understates it by a factor
    of two.

    Args:
        state: The campaign, carrying the model.
        depth: Layers in the proposed circuit.
        target_error: The total energy accuracy to price against, in the same units
            as the energy. The interface sets it per magnet and
            :func:`_target_error_for` multiplies by the chain length.
        noise_inflation: How many times more shots the machine's noise costs, or
            ``1.0`` when the campaign is pricing the algorithm rather than a device.

    Returns:
        The estimated shot cost, rounded up -- a budget that rounded a requirement
        down would buy an accuracy nobody asked for.
    """
    model = state["model"]
    assert model is not None  # the caller checks this before pricing
    per_reading = math.ceil((_operator_for(model).coefficient_l1() / target_error) ** 2)
    readings = EVALUATIONS_PER_LAYER * depth
    return math.ceil(per_reading * readings * noise_inflation)


def _operator_for(model: FormalModel) -> PauliSum:
    """Assemble the problem's Hamiltonian on its own geometry.

    One place, because pricing a plan and re-pricing the run that came of it have to
    agree on the term list: a lattice carries more bonds than a chain of the same
    site count, and the two figures are only comparable if both counted them.

    Args:
        model: The formalised problem.

    Returns:
        The Pauli sum.
    """
    return ising_chain(
        n_sites=model.n_sites,
        coupling=model.coupling,
        transverse_field=model.transverse_field,
        longitudinal_field=model.longitudinal_field,
        boundary=model.boundary,
        lattice=_lattice_for(model),
    )


def _shots_at_true_variance(model: FormalModel, result: VqeResult, priced: int) -> int:
    r"""Re-price a finished run's readings against the state it actually prepared.

    The budget was approved on :math:`\sum_\alpha |c_\alpha|`, which charges every
    term the largest variance its :math:`\pm 1` spectrum allows. The prepared state
    has a variance of its own, :math:`\sum_g \sigma_g` over the measurement
    settings, and the shot count goes as the square of either. Scaling the priced
    figure by that ratio rather than counting afresh keeps the readings, the target
    accuracy and any device premium already in the price.

    Args:
        model: The formalised problem.
        result: The finished run, carrying the angles it ended on.
        priced: What the plan was charged before it ran.

    Returns:
        The re-priced count, never above ``priced``. Falls back to ``priced`` if the
        state turns out to be sharp in both settings, which no circuit here prepares.
    """
    lattice = _lattice_for(model)
    diagonal = diagonal_energies(
        n_sites=model.n_sites,
        coupling=model.coupling,
        longitudinal_field=model.longitudinal_field,
        boundary=model.boundary,
        lattice=lattice,
    )
    prepared = evolve(result.parameters, result.spec, diagonal, model.transverse_field)
    deviation = measurement_deviation(prepared, diagonal, model.transverse_field)
    bound = _operator_for(model).coefficient_l1()
    if deviation <= 0.0 or bound <= 0.0:
        return priced
    return math.ceil(priced * min(1.0, (deviation / bound) ** 2))


def _marginal_gain(state: CampaignState) -> float | None:
    """How much the latest run improved on the best energy before it.

    Args:
        state: The campaign so far.

    Returns:
        The improvement, positive when the latest run is better. ``None`` when
        there is nothing to compare against.
    """
    runs = state["runs"]
    if len(runs) < 2:
        return None
    return min(run.energy_per_site for run in runs[:-1]) - runs[-1].energy_per_site


def _rounds_used(state: CampaignState) -> int:
    """How many planning turns the campaign has taken.

    Args:
        state: The campaign so far.

    Returns:
        Runs completed plus configurations rejected, which together count every
        turn the planning loop has had.
    """
    return len(state["runs"]) + len(state["ruled_out"])


def _depth_from_label(label: str) -> int:
    """Recover the depth from a configuration label such as ``hva-p4``.

    Args:
        label: The configuration label.

    Returns:
        The depth, or zero when the label does not carry one. Zero is never a
        candidate depth, so an unparseable label simply fails to exclude anything
        rather than excluding the wrong thing.
    """
    _, _, tail = label.partition("-p")
    return int(tail) if tail.isdigit() else 0


def _warm_start_for(state: CampaignState, depth: int) -> FloatArray | None:
    """Stretch the best schedule found so far onto a deeper circuit.

    A deeper circuit contains every shallower one, so the angles that worked at
    the depth below are already a good answer at this depth -- stretched over more
    layers, they describe the same schedule sampled more finely. Starting there
    instead of from scratch means each depth after the first costs one optimisation
    rather than a fresh scan over starting schedules, and it starts in a basin that
    is known to be good rather than in whichever one the scan happens to find.

    Args:
        state: The campaign so far.
        depth: The depth about to be run.

    Returns:
        Angles to start from, or ``None`` when nothing has run yet or nothing that
        ran can be trusted -- in which case the basin has to be searched for.
    """
    usable = [
        run
        for run in state["runs"]
        if run.parameters and run.diagnosis.signal != "below_variational_bound"
    ]
    if not usable:
        return None
    best = min(usable, key=lambda run: run.energy)
    if best.depth > depth:
        return None
    return interpolate_to_depth(np.asarray(best.parameters, dtype=float), depth)


def _background_query(model: FormalModel, request: Request, intent: str) -> str:
    """Build the search that will find background worth citing for this problem.

    Two halves, and the second is the one that is easy to leave out. The first asks
    about the quantum method being proposed. The second asks about the classical
    methods that already solve problems of this shape, because a search that only
    looks for quantum work returns a reading list in which quantum always looks
    promising -- the bias then sits in the evidence rather than in the reasoning,
    where no amount of care further down will find it.

    The problem's own character steers the wording: a chain with a closed-form
    solution and one without are different questions, and asking the same thing for
    both wastes the retrieval.

    The branch decides whose words are searched. On ``explain`` and ``implement`` the
    subject is whatever was asked, so the query is the question itself, taken from
    :attr:`~src.agent.state.Request.for_search` -- the restatement where one was made
    and the original otherwise. Without that, the interface tells readers the
    restatement "is used to build a search query" while the search ignores it, and a
    question about barren plateaus gets answered out of passages retrieved for a query
    about circuit depth, because the templated query never mentions the subject.

    The feasibility branch keeps the templated query and deliberately does **not**
    prepend the question. The template is what makes the reading list balanced, and
    the same problem arrives here worded as a vendor's pitch or a skeptic's
    challenge; letting either steer the search would put the framing bias in the
    evidence, which is the one place no amount of care further down can remove it.

    Args:
        model: The formalised problem.
        request: The question as asked, for the branches it is the subject of.
        intent: Which branch is running.

    Returns:
        The query string.
    """
    if intent in ("explain", "implement"):
        return request.for_search

    quantum = (
        "variational quantum eigensolver and QAOA circuit depth, shot cost and "
        "trainability for a transverse-field Ising chain"
    )
    classical = (
        "matrix product states, DMRG and quantum Monte Carlo as the classical "
        "baseline for one-dimensional spin chains"
    )
    character = (
        "exactly solvable free-fermion chain"
        if model.is_exactly_solvable
        else "non-integrable chain with a longitudinal field"
    )
    return f"{quantum}; {classical}; {character}"
