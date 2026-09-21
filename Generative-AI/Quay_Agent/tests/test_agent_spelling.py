"""Typos, and whether the agent reads through them.

The failure this guards is quiet and total. Everything downstream of the question
works by matching words -- the scope gate, the intent reader, the shelf router and
the retrieval query -- so one mistyped noun fails all four at once. *what is the phys
of the quntum isng mdl?* shares no word with the scope vocabulary, and the reader was
told the request could not be turned into a physics problem. It could; three
characters were wrong.

The tests come in two halves and the second is the important one. Correcting a word
the user meant is worse than leaving one they mistyped: the first silently answers a
different question, and the second at worst leaves the reader where they were. So
about half of what follows asserts that ordinary English is left completely alone.
"""

from __future__ import annotations

import pytest

from src.agent.graph import run_campaign
from src.agent.router import in_domain
from src.agent.spelling import CUTOFF, LENGTH_SLACK, SHORTHAND, corrections, repair
from src.agent.state import CampaignState

# --------------------------------------------------------------------------
# What it repairs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "read"),
    [
        ("quntum", "quantum"),
        ("isng", "ising"),
        ("baren", "barren"),
        ("plateu", "plateau"),
        ("varational", "variational"),
        ("hamiltonain", "hamiltonian"),
        ("entaglement", "entanglement"),
    ],
)
def test_a_dropped_or_doubled_letter_is_repaired(typed: str, read: str) -> None:
    assert repair(typed) == read


@pytest.mark.parametrize(("typed", "read"), [("feild", "field"), ("chian", "chain")])
def test_a_transposition_is_repaired(typed: str, read: str) -> None:
    """The commonest typo, and the one edit distance is worst at.

    ``feild`` and ``field`` share every letter and score 0.60 against each other,
    because :mod:`difflib` measures matching *blocks* and a swap breaks the word
    into three of them -- well under the cutoff. Sorted letters catch it exactly.
    """
    assert repair(typed) == read


@pytest.mark.parametrize("short", sorted(SHORTHAND))
def test_every_shorthand_expands_to_something_longer(short: str) -> None:
    # An abbreviation is not a misspelling: `mdl` scores 0.75 against `model`, below
    # any cutoff that leaves ordinary words alone. They are listed rather than
    # inferred, so the list is only as correct as its entries.
    assert repair(short) == SHORTHAND[short]
    assert len(SHORTHAND[short]) > len(short)


def test_the_whole_question_survives_the_repair() -> None:
    # Punctuation, spacing and casing all stay put: the repaired question is shown to
    # the reader as what the agent understood, and a restatement that has quietly been
    # reformatted is harder to compare, not easier. A corrected word keeps the capital
    # it was typed with, because a lowercased first word merges two sentences into one
    # for every reader downstream that looks for the boundary.
    assert repair("what is Phys of quntum isng mdl?") == "what is Physics of quantum ising model?"
    assert repair("Quntum hardware. Isng chains.") == "Quantum hardware. Ising chains."


def test_a_correct_singular_is_not_pluralised() -> None:
    # The vocabulary is assembled from matching lists and carries whichever form the
    # matching wanted, so it holds "neighbours" and not "neighbour". The singular is a
    # correctly spelled word and was being rewritten into the plural mid-sentence,
    # which put "the neighbours interaction" in front of the person who had typed the
    # sentence correctly.
    assert repair("the neighbour interaction") == "the neighbour interaction"
    assert repair("Neighbour coupling") == "Neighbour coupling"
    assert corrections("the neighbour interaction") == ()


def test_the_corrections_are_reported_and_not_only_applied() -> None:
    # A correction is a guess about what somebody meant, and a guess nobody can see
    # is the kind that gets believed.
    assert corrections("what is quntum isng?") == (("quntum", "quantum"), ("isng", "ising"))


def test_nothing_is_reported_when_nothing_changed() -> None:
    assert corrections("How deep must the ansatz circuit be?") == ()


# --------------------------------------------------------------------------
# What it must never touch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What is the best pizza in Vilnius?",
        "I put the plate on the table and ate three states of pizza",
        "Which of these things should I read first, and why?",
        "Tell me about the different phases of the moon",
        "How deep must the ansatz circuit be?",
        "Is quantum hardware worth it for a 10-spin critical Ising chain?",
        "Write me a variational eigensolver for a 6-spin chain",
    ],
)
def test_ordinary_english_is_left_exactly_as_it_was(question: str) -> None:
    assert repair(question) == question


def test_a_plural_is_not_quietly_made_singular() -> None:
    # `states` is not in the vocabulary and `state` is, so without a check on the
    # stem the rule "corrects" the plural -- a change that is both wrong and
    # invisible, since the result is a real word in the right place.
    assert repair("how many states are there") == "how many states are there"


def test_a_short_word_is_never_guessed_at() -> None:
    # Below four characters an edit distance says almost nothing: `its`, `gap` and
    # `sim` are each within one edit of several unrelated terms.
    assert repair("is it in the set of ten") == "is it in the set of ten"


def test_the_rule_will_not_reach_across_a_real_gap() -> None:
    # `plate` scores 0.83 against `plateau`, over the cutoff, and is two characters
    # short of it. A typo is a slip of one or two keys and almost never changes a
    # word's length by more than one, which is what stops this.
    assert CUTOFF < 0.85
    assert LENGTH_SLACK == 1
    assert repair("plate") == "plate"


# --------------------------------------------------------------------------
# The campaign, end to end
# --------------------------------------------------------------------------

MISSPELT = "what is Phys of quntum isng mdl?"


def test_a_misspelt_question_is_out_of_scope_until_it_is_repaired() -> None:
    # The bug in one line. Both sentences ask the same thing and only one of them
    # used to be answerable.
    assert not in_domain(MISSPELT)
    assert in_domain(repair(MISSPELT))


@pytest.fixture(scope="module")
def misspelt() -> CampaignState:
    """One offline campaign run on a question with four typos in it."""
    return run_campaign(
        MISSPELT,
        shot_budget=1,
        chat_model=None,
        search_corpus=True,
        fetch_external=False,
    )


def test_the_campaign_reads_through_the_typos(misspelt: CampaignState) -> None:
    assert misspelt["request"].in_scope, "a question with typos in it was refused as off topic"
    assert misspelt["intent"].intent == "explain"


def test_the_question_itself_is_never_overwritten(misspelt: CampaignState) -> None:
    # The repair is stored beside the question, never in place of it. Replacing the
    # text would launder away the wording, and measuring whether a verdict moves
    # with the wording is only possible if the wording survives the trip.
    assert misspelt["request"].text == MISSPELT
    assert misspelt["request"].clarified != MISSPELT


def test_the_repaired_wording_is_what_reaches_the_index(misspelt: CampaignState) -> None:
    # The whole point. A corrected question that is not what gets searched for has
    # corrected nothing.
    searched = [item.query for item in misspelt["searches"]]
    assert searched, "nothing was searched for"
    assert "quantum" in searched[0] and "ising" in searched[0]
    assert "quntum" not in searched[0]


def test_the_trail_says_which_words_were_changed(misspelt: CampaignState) -> None:
    trail = " ".join(misspelt["notes"])
    assert "corrected the spelling" in trail
    assert "quntum" in trail and "quantum" in trail


@pytest.mark.parametrize(
    "word",
    ["plot", "plots", "plotting", "graph", "chart", "curve", "loss", "epoch", "versus", "axis"],
)
def test_the_vocabulary_of_asking_for_a_picture_is_left_alone(word: str) -> None:
    # `plot` scored 0.89 against `pilot`, a real word this project's hardware pages
    # use, so "plot the loss against epoch" was being restated as "pilot the loss" --
    # underneath a figure the reader was asking about. The nearest-match rule is only
    # safe where the words it could reach are enumerated, and these were not.
    assert repair(word) == word


def test_a_request_for_a_curve_survives_the_speller_intact() -> None:
    asked = "Can you plot the loss versus optimization epoch?"
    assert repair(asked) == asked


def test_the_real_typos_are_still_repaired() -> None:
    # The guard above must not have been bought by switching the rule off.
    assert repair("what is the phys of the quntum isng mdl?") == (
        "what is the physics of the quantum ising model?"
    )
