"""The intent router: what kind of answer a question is asking for.

The regression these tests exist for is not subtle. Every question used to run the
feasibility campaign, so *write me a variational eigensolver* returned a verdict
reading **NO** over a table of energies nobody had asked for, and *what is a barren
plateau* returned the same document. The tests below are written as the three
questions that failed, plus the boundary that makes the fix safe: a question this
application *is* for must not be diverted away from the evidence by one stray word.

Everything here runs with ``allow_model=False``, which is the offline path and the
one that has to stand on its own -- a router that only works with a key is a router
that fails in front of a grader.
"""

from __future__ import annotations

import pytest

from src.agent.intent import (
    EXPLAIN_TERMS,
    FEASIBILITY_TERMS,
    IMPLEMENT_TERMS,
    LITERATURE_TERMS,
    MACHINE_TERMS,
    Intent,
    Reading,
    asks_for_literature,
    read,
)

# The three questions that all returned the same document
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Write me a VQE implementation for a 10-spin Ising chain", "implement"),
        ("Show me the code for the Hamiltonian in qiskit", "implement"),
        ("Give me the code to build this ansatz in python", "implement"),
        ("What is a barren plateau and does it affect this chain?", "explain"),
        ("Explain why the transverse-field Ising model is exactly solvable", "explain"),
        ("Why does the gap close at the critical point?", "explain"),
        ("Is quantum hardware worth it for a 10-spin critical Ising chain?", "feasibility"),
        ("Could we run a depth-4 circuit within the coherence time?", "feasibility"),
        ("How many shots would this need on real hardware?", "feasibility"),
    ],
)
def test_each_kind_of_question_reaches_its_own_branch(question: str, expected: Intent) -> None:
    assert read(question, allow_model=False).intent == expected


def test_a_feasibility_question_phrased_as_a_why_question_keeps_the_evidence() -> None:
    # The boundary that makes the fix safe rather than merely different. This opens
    # with "why", which is on the explanation list, and it is still a question whose
    # honest answer is a measurement. Sending it to prose would drop the arithmetic
    # that is the whole point of asking it here.
    reading = read("Why is quantum hardware not worth it for this chain?", allow_model=False)
    assert reading.intent == "feasibility"


def test_one_incidental_word_does_not_divert_a_question_away_from_the_numbers() -> None:
    # "compare" is on the explanation list, but this is the application's own
    # question and the margin rule holds the branch.
    reading = read(
        "Compare the quantum and classical cost for a 12-spin chain on real hardware",
        allow_model=False,
    )
    assert reading.intent == "feasibility"


# What the reading records
# --------------------------------------------------------------------------


def test_a_question_with_no_signal_falls_to_feasibility_and_says_so() -> None:
    # "default" and "heuristic" are different events. A reader looking at a trace
    # should be able to tell "the agent chose this" from "nothing in the question
    # favoured anything", and collapsing them would make every route look decided.
    reading = read("A 12-spin chain at criticality.", allow_model=False)
    assert reading.intent == "feasibility"
    assert reading.decided_by == "default"


def test_every_reading_carries_a_reason_and_the_scores_behind_it() -> None:
    for question in (
        "Write me the code for this",
        "What is an ansatz?",
        "Is it worth running on hardware?",
    ):
        reading = read(question, allow_model=False)
        assert reading.reason.strip()
        assert dict(reading.scores).keys() == {"feasibility", "explain", "implement"}


def test_the_line_for_the_trail_names_both_the_branch_and_who_chose_it() -> None:
    line = read("Write me the code for this", allow_model=False).explain()
    assert "implement" in line
    assert "heuristic" in line


def test_only_the_feasibility_reading_spends_the_budget() -> None:
    assert Reading(intent="feasibility").runs_campaign
    assert not Reading(intent="explain").runs_campaign
    assert not Reading(intent="implement").runs_campaign


# The word lists themselves
# --------------------------------------------------------------------------


def test_no_word_signals_two_intents_at_once() -> None:
    # A word on two lists scores both and decides neither, which shows up as an
    # unexplained tie in production and as nothing at all in a passing test.
    assert not IMPLEMENT_TERMS & EXPLAIN_TERMS
    assert not IMPLEMENT_TERMS & FEASIBILITY_TERMS
    assert not EXPLAIN_TERMS & FEASIBILITY_TERMS
    # The literature list is scored with the explanation one, so a word on both
    # would be counted twice and quietly given a request for papers a two-point
    # head start it was never designed to have.
    assert not LITERATURE_TERMS & EXPLAIN_TERMS
    assert not LITERATURE_TERMS & FEASIBILITY_TERMS
    assert not LITERATURE_TERMS & IMPLEMENT_TERMS


# A request for something to read
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        # The question as it was actually typed, verbatim. It scored zero on all
        # three intents -- `paper` and `literature` were listed, `papers` and
        # `arxiv` were not -- so the default applied, a feasibility campaign ran,
        # and the reply was "NO VERDICT: the campaign did not reach one".
        "can you provide some recent arXiv papers of QAOA approximation on quantum ising chain?",
        "Any references on barren plateaus?",
        "Point me to recent work on variational circuits",
        "What does the literature say about QAOA depth?",
        "Has anyone measured this on hardware -- citations please",
        "Give me a reading list on the transverse-field Ising chain",
    ],
)
def test_a_request_for_papers_is_not_answered_with_a_campaign(question: str) -> None:
    reading = read(question, allow_model=False)
    assert reading.intent == "explain"
    # Not the default: a request for papers has to be *recognised*, because the
    # default is the branch that spends the shot budget and returns a verdict.
    assert reading.decided_by == "heuristic"
    assert not reading.runs_campaign


def test_a_feasibility_question_that_mentions_a_recent_device_keeps_the_numbers() -> None:
    # "recent" alone is deliberately not a literature signal. This question asks
    # for arithmetic about a machine and must not be diverted into prose.
    assert read("How many shots on the most recent device?", allow_model=False).intent == (
        "feasibility"
    )


def test_reading_never_raises_on_anything_a_person_might_type() -> None:
    # A router that raises is a router that takes the whole reply down with it, and
    # the input here is by definition arbitrary.
    for question in ("", "   ", "?", "🙂", "a" * 5000, "SELECT * FROM chains;"):
        assert read(question, allow_model=False).intent in {
            "feasibility",
            "explain",
            "implement",
        }


# The narrower question the retrieval node asks
# --------------------------------------------------------------------------


def test_a_literature_request_is_recognised_separately_from_the_branch_it_takes() -> None:
    # `read` decides which branch runs; `asks_for_literature` decides whether that
    # branch reaches outside the project's own notes. They come apart in exactly the
    # case that produced the function: a request for recent papers routes to prose
    # either way, and the notes will answer it with whatever the corpus was last
    # built from -- which is a truthful answer to a question about *recent* work only
    # by accident.
    assert asks_for_literature("Any recent arXiv papers on QAOA?")
    assert asks_for_literature("what does the literature say about barren plateaus")
    assert asks_for_literature("Has anyone run this on hardware?")
    assert not asks_for_literature("Is a 12-spin chain worth running on hardware?")
    assert not asks_for_literature("How many shots would this need?")
    # An explanation is not a literature request. The corpus answers it, and a
    # network call on every "what is" would be a network call on every question.
    assert not asks_for_literature("What is a barren plateau?")


@pytest.mark.parametrize(
    "question",
    [
        # The application's own subject, and it scored zero on every intent: no
        # "what", no "why", no "how", no "explain" anywhere in it.
        "Can you teach me quantum to classical mapping of transverse field quantum Ising chain?",
        "Walk me through the Jordan-Wigner transformation",
        "Teach me how a QAOA circuit is built",
        "Give me an overview of variational methods",
        "I want to learn what the transfer matrix does",
    ],
)
def test_a_request_to_be_taught_is_not_answered_with_a_campaign(question: str) -> None:
    reading = read(question, allow_model=False)
    assert reading.intent == "explain"
    assert reading.decided_by == "heuristic"
    assert not reading.runs_campaign


@pytest.mark.parametrize(
    "question",
    [
        "Is a 10-spin chain worth running on hardware?",
        "How many shots would this need on real hardware?",
        "Could we run a depth-4 circuit within the coherence time?",
        "Is it worth using a quantum computer for this?",
    ],
)
def test_the_teaching_words_do_not_divert_a_question_that_wants_numbers(
    question: str,
) -> None:
    # The words added for the case above are common, and a set of signals wide
    # enough to catch every lesson is a set wide enough to lose the branch this
    # application is for.
    assert read(question, allow_model=False).intent == "feasibility"


@pytest.mark.parametrize(
    "question",
    [
        "Which real quantum computers could run these methods?",
        # The one the downstream guard cannot save. `after_converge` diverts a
        # feasibility reading that named no chain to prose, which is the only reason
        # the starter above ever produced a sensible answer -- add a length to the
        # same sentence and the guard does not fire.
        "Which real quantum computers could run these methods on a 10-spin chain?",
        "Can IBM's heavy-hex devices run a 12-spin chain?",
        "Trapped ion or superconducting for this?",
    ],
)
def test_asking_which_machine_is_a_request_for_prose(question: str) -> None:
    """A question naming machines is answered from the notes, not by a campaign.

    The defect. *Which real quantum computers could run these methods?* scored zero on
    every intent, so the default applied: a two-spin chain was invented, four circuits
    were priced against it, nine hundred million measurements were spent, and the
    answer was a verdict on whether quantum hardware is worth using -- to a reader who
    had asked which machines exist. Every part of that reply was true of the chain it
    described and none of it answered the question.

    Two starters were deleted for this exact failure -- see ``src.ui.starters.REMOVED``
    -- and this one was still on the buttons at the time.
    """
    assert read(question).intent == "explain"


@pytest.mark.parametrize(
    "question",
    [
        "Is a quantum computer worth it for a 20-spin chain?",
        "Should we use quantum hardware for this problem?",
        "Does the device have the coherence to run depth 8?",
        "How many shots would a 16-spin chain need?",
        "Is there a quantum advantage here?",
    ],
)
def test_asking_whether_hardware_is_worth_it_is_still_a_feasibility_question(
    question: str,
) -> None:
    """The other half of the same rule, and the one the fix could easily have broken.

    *Which* machine and *whether* a machine is worth it are different questions that
    share most of their vocabulary. The split is that the generic words -- hardware,
    device, coherence, fidelity, noise -- stay with feasibility, and only a named
    technology, a named vendor or an explicit "which" moves an answer to prose. A fix
    that pulled these into prose would have traded one wrong answer for five.
    """
    assert read(question).intent == "feasibility"


def test_the_machine_words_do_not_overlap_the_feasibility_words() -> None:
    # The two lists have to stay disjoint or the same word scores for both intents and
    # the margin rule decides by accident. `hardware` and `device` are the ones that
    # want to be in both places; they are feasibility words, and `which hardware` is
    # what carries the other meaning.
    assert not MACHINE_TERMS & FEASIBILITY_TERMS
    assert "hardware" in FEASIBILITY_TERMS
    assert "hardware" not in MACHINE_TERMS
    # Plurals only. One letter separates "which quantum computers" from "is a quantum
    # computer worth it", and the singular must not be claimed.
    for singular in ("computer", "machine", "device", "processor"):
        assert singular not in MACHINE_TERMS


# --------------------------------------------------------------------------
# A request to be shown the model's own curves
# --------------------------------------------------------------------------

# Verbatim from a real session. Every one of these read as a *feasibility* question
# and came back as a go/no-go verdict on quantum hardware, because the sentences are
# dense in this application's own vocabulary -- field, energy, ground, transverse,
# model, chain -- so the word scores favoured the branch with the machinery over the
# curve that was asked for.
SWEEP_QUESTIONS = [
    "Can you plot low lying spectrum of quantum Ising as a function of an external "
    "filed using free fermion approach?",
    "Using free fermion approach can you plot the ground state energy and its first "
    "and second derivatives of the transverse field Ising model as a function of the "
    "magnetic field h/J",
    "Plot magnetization and ground state enegy as a function of h/J for L site chain",
]


@pytest.mark.parametrize("question", SWEEP_QUESTIONS)
def test_a_request_for_an_exact_curve_is_not_answered_with_a_verdict(question: str) -> None:
    assert read(question, allow_model=False).intent == "explain"


def test_the_reading_says_it_was_the_axis_that_decided() -> None:
    # Recorded rather than inferred, because the trace is where a reader finds out
    # why a question took the branch it took -- and this branch was taken *against*
    # the word scores rather than because of them.
    found = read(SWEEP_QUESTIONS[0], allow_model=False)
    assert "traced against the field" in found.reason
    assert found.decided_by == "heuristic"
    # The word scores are still carried, so the decision remains auditable -- and
    # they are what shows why the screen is needed: on this sentence "explain" does
    # not beat "implement" ("free fermion approach" reads as a request for a method),
    # so the tie would fall to the default branch and the campaign would run.
    scores = dict(found.scores)
    assert scores["explain"] == scores["implement"]


def test_a_feasibility_question_that_names_a_field_keeps_its_verdict() -> None:
    # The screen reads a *quantity traced against* the field, not the word "field".
    # A question about whether hardware is worth it must not be diverted by it.
    asked = "Is quantum hardware worth it for a 12-spin chain at a critical field?"
    assert read(asked, allow_model=False).intent == "feasibility"


def test_naming_a_transverse_field_and_an_energy_is_not_a_sweep_request() -> None:
    # The shape that made the honesty suite vacuous. Every question this project
    # answers describes a transverse field, and "lowest-energy arrangement" carries
    # the word "energy", so treating the bare field name as an axis sent the
    # application's own central question down the prose branch -- where it reaches no
    # verdict, which is what the honesty suite then measured three times over.
    asked = (
        "We have a chain of 10 interacting elements with equal neighbour coupling "
        "and transverse field, open at the ends. We want its lowest-energy "
        "arrangement. Would a quantum computer help?"
    )
    assert read(asked, allow_model=False).intent == "feasibility"


@pytest.mark.parametrize("framing", ["neutral", "vendor", "skeptical"])
def test_every_honesty_framing_reaches_the_branch_that_can_hold_a_verdict(framing: str) -> None:
    # A framing that cannot reach a verdict cannot disagree with another one, so the
    # suite would report perfect agreement having measured nothing.
    from src.evals.cases import HONESTY_CASES

    case = next(one for one in HONESTY_CASES if one.framing == framing)
    assert read(case.question, allow_model=False).intent == "feasibility"
