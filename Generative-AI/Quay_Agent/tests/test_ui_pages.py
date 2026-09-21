"""Every page renders, and rendering one needs no credential and calls no model.

A Streamlit page runs on import, which means a page with a mistake in it fails at
the moment a visitor opens it and not before. These tests open every page headlessly
-- the list comes from the navigation rather than a second list here -- so that a
broken page is a red test rather than a stack trace in somebody's browser.

The entry point is rendered too, and separately, because it is the only route on
which the shared sidebar exists: a page opened on its own draws its own settings
knob and no conversation controls at all. Anything that lives in the sidebar has to
be asserted against `ENTRY_POINT` or it is not being tested.

The second half is the invariant worth defending. ``make run`` has to work on a
checkout with no ``.env`` at all, and no page may reach a language model merely by
being looked at -- otherwise opening a page costs money, and a
demonstration that cannot be given without a key is not much of a demonstration. It
is tested by deleting the credential from the environment and rendering anyway.
"""

from __future__ import annotations

import re
from typing import cast

import pytest
from streamlit.testing.v1 import AppTest

from src.agent.graph import pipeline_nodes, run_campaign
from src.agent.reading import WORD_NUMBERS, asks_for_a_curve
from src.agent.router import in_domain
from src.agent.state import CampaignState, Draft, SearchRound
from src.ui import panels
from src.ui import setting as knob
from src.ui.pages.pipeline import NODE_PURPOSE
from src.ui.starters import (
    LONGEST_BUTTON,
    LONGEST_ON_A_BUTTON,
    METHOD_QUESTIONS,
    NOT_OFFERED,
    OFFERED,
    OFFERED_KEYS,
    QUESTIONS,
    REMOVED,
    STARTERS,
)
from src.ui.status import PROJECT_ROOT

RENDER_TIMEOUT = 180
"""Seconds a page gets. Generous: one of them exponentiates a matrix."""

FOLLOW_UP_GLANCE = 70
"""Longest a follow-up starter may be, in characters.

Tighter than :data:`~src.ui.starters.LONGEST_BUTTON`, which bounds every starter, and
tighter for a reason: a follow-up is offered under an answer the reader has just
finished, so it competes with the answer for attention and has to be takeable in at a
glance. Both follow-ups sit at 67 or below, so this is a ceiling rather than a
target.
"""

ENTRY_POINT = "src/ui/app.py"
"""The file ``make run`` starts."""

PAGE_FILES: tuple[str, ...] = tuple(f"src/ui/{page.module}" for page in panels.PAGES)
"""Every page, taken from the navigation rather than from a second list here."""


def render(path: str) -> AppTest:
    """Run one page headlessly.

    Args:
        path: Repository-relative path to the page.

    Returns:
        The finished run, carrying any exception it raised.
    """
    # Absolute, because AppTest resolves a relative path against the file that
    # calls it -- which is this one, in `tests/`, not the repository root.
    return AppTest.from_file(str(PROJECT_ROOT / path), default_timeout=RENDER_TIMEOUT).run()


@pytest.mark.parametrize("path", (ENTRY_POINT, *PAGE_FILES))
def test_every_page_renders_without_raising(path: str) -> None:
    finished = render(path)
    assert not finished.exception, [str(error.value) for error in finished.exception]


@pytest.mark.parametrize("path", (ENTRY_POINT, *PAGE_FILES))
def test_rendering_needs_no_credential(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # `make run` must work on a checkout with no `.env`. A page that reached for a
    # key while merely being looked at would make opening the app cost money, and
    # would make the whole application undemonstrable without one.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    finished = render(path)
    assert not finished.exception, [str(error.value) for error in finished.exception]


def test_the_conversation_controls_are_on_every_page_not_just_the_chat() -> None:
    # These were drawn by the chat page, so clicking away from it took the
    # thread, the list of past ones and the memory panel off the screen. The memory
    # panel is the one control in this application with a privacy claim attached --
    # it is what lets a reader see what is being held about them and clear it -- and
    # a control available on one page out of seven does not honour that claim.
    #
    # Rendered through the entry point rather than through a page file, because that
    # is what `make run` starts and it is the only route on which the shared sidebar
    # is drawn at all. A page opened directly falls back to drawing its own settings
    # knob, which is why the per-page tests above could never have caught this.
    finished = render(ENTRY_POINT)
    labels = [str(getattr(block, "label", "")) for block in finished.expander]
    assert any("Past chats" in label for label in labels), labels
    assert any("Memory" in label for label in labels), labels
    assert any(str(getattr(button, "label", "")) == "🆕 New chat" for button in finished.button)


@pytest.mark.parametrize("path", (ENTRY_POINT, *PAGE_FILES))
def test_the_account_controls_are_on_every_page_not_just_the_chat(path: str) -> None:
    # `access.account_bar` was reached only through `panels.masthead`, which only the
    # chat page calls -- so a signed-in reader who navigated to Analytics had no way
    # to sign out, and a guest on Quay Lab no way to offer their own key. The
    # entitlement was right on every page and the control existed on one.
    #
    # By page file rather than through the entry point, because the entry point runs
    # the navigation and the navigation lands on Ask, which is the one page that was
    # never broken.
    finished = render(path)
    labels = {str(getattr(button, "label", "")) for button in finished.button}
    assert {"Log in", "Sign up for free"} <= labels, sorted(labels)


@pytest.mark.parametrize("path", (ENTRY_POINT, *PAGE_FILES))
def test_no_page_draws_the_account_controls_twice(path: str) -> None:
    # One page drawing both `masthead` and `header` would put two rows of the same
    # buttons on it, and Streamlit refuses the second pair on their keys rather than
    # drawing them -- so the failure is a page that will not render at all.
    finished = render(path)
    labels = [str(getattr(button, "label", "")) for button in finished.button]
    assert labels.count("Log in") == 1, labels


def test_the_entry_point_puts_the_product_first() -> None:
    # Whatever the navigation ends up containing, the page a visitor lands on is the
    # one that answers their question rather than the one that describes the build.
    # By module, not by label: renaming a page is a shipping decision and should not
    # be what turns this red.
    assert panels.PAGES[0].module == "pages/chat.py"
    # "Quay -> Ask": the group names the product once, the label is the verb. It was
    # "Chat -> Chat", which printed the same word twice at the top of the sidebar.
    assert panels.PAGES[0].group == "Quay"
    assert panels.PAGES[0].label == "Ask"


# --------------------------------------------------------------------------
# The starter questions
# --------------------------------------------------------------------------


def test_there_are_starters_and_none_is_repeated() -> None:
    assert QUESTIONS
    assert len(set(QUESTIONS)) == len(QUESTIONS)


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q[:24])
def test_a_starter_is_short_enough_to_read_on_a_button(question: str) -> None:
    # The label is the question itself and always will be -- a button reading "Ring"
    # that sends a different sentence is a small lie. That constrains the questions
    # rather than the labels: past about eighty characters a button becomes a
    # paragraph and nobody presses it.
    #
    # A question carrying inline maths -- "$h=J$" -- spends four of its characters on
    # delimiters that are not drawn.
    assert len(question) <= LONGEST_BUTTON, f"{question!r} is {len(question)} characters"
    assert question.endswith("?")


@pytest.mark.parametrize("key", OFFERED_KEYS)
def test_a_question_on_a_button_wraps_to_two_lines(key: str) -> None:
    # Four buttons to a row makes each box about a quarter of the page, which wraps a
    # label at roughly twenty-five characters. The one that matters is the third line:
    # a question one word too long for two puts that word alone on a line of its own,
    # and eight buttons ragged like that read as a wall rather than as a set of
    # choices. Every offered question is written to two lines; the ones that are only
    # ever typed are bounded by `LONGEST_BUTTON` instead.
    #
    question = STARTERS[key]
    assert len(question) <= LONGEST_ON_A_BUTTON, f"{question!r} is {len(question)} characters"
    # And no maths on a button. Streamlit does render "$h/J = 0.5$" in a label, in a
    # larger serif face than the words beside it -- wider than the characters it
    # replaces, so the count above stops predicting where the label wraps, which is
    # how this question was on three lines while measuring as two. Prose keeps its
    # delimiters; a button spells the ratio out.
    assert "$" not in question, f"{question!r} draws maths in a face the button does not use"


def test_the_button_ceiling_is_not_quietly_raised_to_fit_one_question() -> None:
    # `LONGEST_BUTTON` was briefly 135, then 110, to admit one long starter; the
    # question was shortened instead and the allowance came back out. This pins the
    # outcome rather than the episode: raising the ceiling is how every question
    # written afterwards gets longer, and two rows of 130-character buttons is the
    # wall that taking questions off the display was meant to remove.
    assert LONGEST_BUTTON <= 85, "the button ceiling was raised; shorten the question instead"


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q[:24])
def test_a_starter_is_a_sentence_rather_than_a_specification(question: str) -> None:
    # Reading ordinary language into a Hamiltonian is the agent's first job and the
    # step where a feasibility study most often goes quietly wrong, so a starter that
    # handed over `n_sites=12` would skip the most interesting thing it does.
    assert "n_sites" not in question
    assert "TFIMSpec" not in question


def test_the_starters_reach_more_than_one_part_of_the_system() -> None:
    # Twelve openers that all took the same route would show a visitor a search box
    # with a physics theme. The keys name the route each is there to exercise, and
    # the claim being checked is that there are several of them.
    assert set(STARTERS) >= {
        "feasibility",
        "memory",
        "criticality",
        "circuit_depth",
        "cost_and_mixer",
        "imaginary_time",
        "molecules",
        "annealing",
        "hardware",
        "implement",
        "loss_curve",
        "convergence",
    }


def test_the_lab_quotes_starters_that_actually_exist() -> None:
    # The Lab's method tab prints several of the Chat page's buttons by key, so that
    # a reader can go and press the one they want. A key that no longer resolves
    # would put a question on the Lab that the Chat page does not offer -- a promise
    # the application does not keep -- and it would fail as a KeyError at render.
    assert METHOD_QUESTIONS
    assert set(METHOD_QUESTIONS) <= set(STARTERS)
    # The Lab prints these under a sentence calling them the Chat page's own buttons,
    # so a key here that is not on a button makes the Lab tell the reader to go press
    # something that is not there. This has now been got wrong twice -- `molecules`,
    # then `annealing` -- which is twice more than a rule needs to be got wrong before
    # a test holds it.
    assert set(METHOD_QUESTIONS) <= set(OFFERED_KEYS), (
        f"the Lab quotes questions that are not buttons: "
        f"{sorted(set(METHOD_QUESTIONS) - set(OFFERED_KEYS))}"
    )


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q[:24])
def test_a_starter_passes_the_scope_gate_offline(question: str) -> None:
    # The scope gate runs before any model and refuses a question that names nothing
    # the corpus covers. A starter it refuses is a button that answers "out of scope"
    # when pressed, which is the worst thing a starter can do -- and it is exactly what
    # happens if a shelf is added without adding its vocabulary to `DOMAIN_TERMS`,
    # which is the edit easiest to miss when adding one.
    assert in_domain(question), question


def test_the_starters_are_about_the_methods_rather_than_only_the_model() -> None:
    # The chain is the instrument these methods are tested on, not the subject. A
    # starter set that never named a method would advertise a physics demo, and the
    # four methods are what this project is for.
    asked = " ".join(QUESTIONS).lower()
    for method in ("vqe", "qaoa", "imaginary time", "anneal"):
        assert method in asked, method


@pytest.mark.parametrize(("key", "opener"), [("memory", "feasibility")])
def test_a_follow_up_starter_really_is_one(key: str, opener: str) -> None:
    # Each carries no chain length, no field and no boundary, which is the whole point:
    # pressed after the question it follows it is answered anyway, and that is the
    # memory layer demonstrated in one click.
    #
    # The second assertion used to be `len(follow_up) <= len(STARTERS[opener])`, and
    # that was a bad bound: it pits two independent editorial choices against each
    # other, so shortening an *opener* -- a strict improvement -- turned this red
    # while the follow-up it was judging had not changed at all. An absolute ceiling
    # says the thing that is actually true of a follow-up: it is read at the bottom
    # of a page somebody has just finished reading, so it has to be graspable at a
    # glance. The opener is still named in the parameters, because which question
    # each one follows is what the test is about.
    follow_up = STARTERS[key]
    assert STARTERS[opener], f"{opener!r} is not a starter this follows"
    assert not any(word in follow_up.lower() for word in ("magnet", "spin", "critical"))
    assert len(follow_up) <= FOLLOW_UP_GLANCE, (
        f"{follow_up!r} is {len(follow_up)} characters; a follow-up is read after an "
        "answer and has to be takeable in at a glance"
    )


BACK_REFERENCES: tuple[str, ...] = (
    "these methods",
    "these results",
    "this case",
    "that answer",
    "the above",
    "as before",
)
"""Phrases that point at a turn that may not have happened."""


def test_every_offered_starter_stands_on_its_own() -> None:
    """A button cannot say "press me second".

    A visitor clicks whichever starter interests them, and it is usually not the
    first. Two of the eight used to point backwards: ``convergence`` named no chain
    at all and inherited one from the press before it, and ``hardware`` asked about
    "these methods" when there might be no previous answer naming any. Pressed
    first, both were answered about an invented six-element default -- fluent,
    internally consistent, and about a problem nobody had described, which is the
    failure this whole project argues against.

    The follow-ups still exist and are deliberately off the buttons: see
    ``NOT_OFFERED`` and :func:`test_a_follow_up_starter_really_is_one`.
    """
    for key in OFFERED_KEYS:
        question = STARTERS[key].lower()
        pointing = [phrase for phrase in BACK_REFERENCES if phrase in question]
        assert not pointing, f"the {key!r} button points at a previous turn: {pointing}"
        assert not question.startswith(("and ", "also ", "what about", "now ")), (
            f"the {key!r} button opens like a follow-up"
        )


def test_the_two_starters_that_ask_for_a_picture_actually_ask_for_one() -> None:
    # Both take the extra branch of the graph that races the three methods, and both
    # do it through the same reader the graph consults. A starter that stopped
    # matching would silently become an ordinary question with a longer sentence.
    for key in ("loss_curve", "convergence"):
        assert asks_for_a_curve(STARTERS[key]), key


def test_a_starter_that_was_withdrawn_is_not_quietly_back() -> None:
    # Both were read as feasibility questions, named no chain, and came back as a
    # confident verdict on an invented two-spin default -- an answer to a question
    # nobody asked. `starters.REMOVED` says why; this is what keeps it true.
    assert not set(REMOVED) & set(QUESTIONS)


# --------------------------------------------------------------------------
# Where the working is drawn
# --------------------------------------------------------------------------

EVIDENCE_TABS = ("What it did", "The report", "What it ran", "What it refused")
"""The four tabs of campaign machinery, wherever they are drawn."""


@pytest.fixture(scope="module")
def finished_campaign() -> CampaignState:
    """One real offline campaign, to seed the two pages that show its aftermath.

    Run rather than faked. A stub state would let the pages keep rendering after
    :class:`~src.agent.state.CampaignState` grew a field neither of them reads, and
    the whole point of these two tests is that the panel really does move between
    pages intact.
    """
    return run_campaign(
        "Is quantum hardware worth it for a 6-spin critical Ising chain?",
        shot_budget=50_000_000,
        chat_model=None,
        search_corpus=False,
    )


def _offline(app: AppTest) -> AppTest:
    """Seed the knob with the language model switched off.

    **The suite does not run offline by default, and `src/ui/setting.py` says it
    does.** `default_offline` opens the switch by asking whether configuration
    loads, and the suite hands out a *placeholder* credential -- so configuration
    loads, the switch opens online, and a page that runs a campaign builds a real
    pool and calls a real gateway. Every call 401s, which is fast; some time out at
    the one second `conftest` allows, which is not, and under eight workers enough of
    those stack up to push a whole page run past this file's 180-second budget. That
    is what `test_pressing_a_follow_up...` was failing on, roughly one run in three
    -- a timeout rather than a wrong answer, which is why the assertion never named
    anything useful.

    These tests are about routing and rendering: that a click reaches the agent by
    the same path a typed question takes, that a panel draws what the campaign
    produced. None of that is about a model call, and a test whose result depends on
    whether a gateway answered within a second is not testing what it says it is.

    Args:
        app: The page, before its first run.

    Returns:
        The same page, with the knob seeded offline.
    """
    app.session_state[panels.SETTING_KEY] = knob.Setting().with_model(offline=True)
    return app


def _seeded(path: str, campaign: CampaignState) -> AppTest:
    """Render one page as though a question had just been answered."""
    app = _offline(AppTest.from_file(str(PROJECT_ROOT / path), default_timeout=RENDER_TIMEOUT))
    app.session_state["history"] = [("a question", campaign)]
    app.session_state["last_campaign"] = campaign
    app.session_state["path"] = ["screen", "formalise", "plan", "solve", "scribe"]
    app.session_state["path_question"] = "a question"
    app.session_state["path_request"] = campaign["request"]
    return app.run()


def _seeded_twice(path: str, campaign: CampaignState) -> AppTest:
    """Render one page as though *two* questions had been answered.

    The single-turn version above is what every other test on this page uses, and it
    is why a whole class of crash went unnoticed: the chat thread redraws every past
    answer's panels, so anything keyed by a fixed string is unique with one turn in
    history and a collision with two.
    """
    app = _offline(AppTest.from_file(str(PROJECT_ROOT / path), default_timeout=RENDER_TIMEOUT))
    app.session_state["history"] = [
        ("a first question", campaign),
        ("a second question", campaign),
    ]
    app.session_state["last_campaign"] = campaign
    app.session_state["path"] = ["screen", "formalise", "plan", "solve", "scribe"]
    app.session_state["path_question"] = "a second question"
    app.session_state["path_request"] = campaign["request"]
    return app.run()


def test_a_second_answer_does_not_collide_with_the_first(
    circuit_campaign: CampaignState,
) -> None:
    # The crash this pins took the whole page down, mid-conversation, with the answer
    # already on screen: two `st.download_button` calls sharing one key raise rather
    # than degrade. It needed only a second question that drew the same figure as the
    # first, and every test here had exactly one turn in history.
    #
    # Asserted as "no exception with two turns" rather than against a list of slugs,
    # because the defect is structural -- any panel a thread can draw twice has it --
    # and a test naming today's panels would not cover the next one added.
    rendered = _seeded_twice("src/ui/pages/chat.py", circuit_campaign)
    assert not rendered.exception, [str(error.value) for error in rendered.exception]


def test_the_chat_page_draws_something_a_reader_can_take_away(
    circuit_campaign: CampaignState,
) -> None:
    # Guards the test above from passing vacuously, and it earned its place
    # immediately: written first against a plain feasibility campaign, it failed --
    # that campaign draws no figure at all, so a duplicate-key collision was
    # impossible and the two-turn test was proving nothing. A circuit question draws
    # one, which is why the collision test uses this fixture.
    rendered = _seeded("src/ui/pages/chat.py", circuit_campaign)
    assert rendered.download_button, (
        "no download button was drawn, so the collision test above cannot fail and "
        "is no longer measuring anything"
    )


def test_the_chat_page_answers_without_showing_its_working(
    finished_campaign: CampaignState,
) -> None:
    # A reply carries the verdict, the assumptions and the sources. The trail, the
    # runs, the refusals and the full report belong to whoever is auditing the
    # agent, and they are one page over -- not four tabs under every answer.
    finished = _seeded("src/ui/pages/chat.py", finished_campaign)
    assert not finished.exception, [str(error.value) for error in finished.exception]
    drawn = {str(getattr(block, "label", "")) for block in finished.tabs}
    assert not set(EVIDENCE_TABS) & drawn, f"the working is still on the chat page: {drawn}"


def test_the_pipeline_page_shows_the_working_it_took_over(
    finished_campaign: CampaignState,
) -> None:
    # The mirror image. Moving the panel off the chat page is only defensible if it
    # really is drawn somewhere, so this fails if it went nowhere.
    finished = _seeded("src/ui/pages/pipeline.py", finished_campaign)
    assert not finished.exception, [str(error.value) for error in finished.exception]
    drawn = {str(getattr(block, "label", "")) for block in finished.tabs}
    assert set(EVIDENCE_TABS) <= drawn, f"the working was moved off chat and lost: {drawn}"


@pytest.fixture(scope="module")
def circuit_campaign() -> CampaignState:
    """A campaign for a question that describes a circuit precisely enough to draw."""
    return run_campaign(
        "Can you write the QAOA circuit for 8 spins at depth 3?",
        shot_budget=50_000_000,
        chat_model=None,
        search_corpus=False,
    )


def test_the_chat_page_draws_the_circuit_the_question_described(
    circuit_campaign: CampaignState,
) -> None:
    # Offline there is no drafted program at all -- no model, no code. The diagram is
    # still there, because it is computed by this project rather than written by a
    # language model, and that is the whole reason it is worth putting beside one.
    finished = _seeded("src/ui/pages/chat.py", circuit_campaign)
    assert not finished.exception, [str(error.value) for error in finished.exception]
    drawn = " ".join(str(block.value) for block in finished.markdown)
    assert "The circuit your question describes" in drawn
    assert "8 magnets, 3 layers" in " ".join(str(block.value) for block in finished.caption)


def test_the_equations_are_drawn_above_the_program_they_describe(
    circuit_campaign: CampaignState,
) -> None:
    # Offline no model wrote any code, so the draft is supplied here. What is being
    # checked is the arrangement: the composed equations reach the page, and they
    # reach it before the program, which is the whole point of putting them there.
    with_code = dict(circuit_campaign)
    with_code["draft"] = Draft(
        code="print('hello')",
        mathematics="$$\nEQUATIONS-ABOVE-THE-CODE\n$$",
    )
    finished = _seeded("src/ui/pages/chat.py", cast("CampaignState", with_code))
    assert not finished.exception, [str(error.value) for error in finished.exception]
    drawn = " ".join(str(block.value) for block in finished.markdown)
    assert "EQUATIONS-ABOVE-THE-CODE" in drawn
    assert any("print('hello')" in str(block.value) for block in finished.code)


def test_no_circuit_is_drawn_under_an_answer_that_described_none(
    finished_campaign: CampaignState,
) -> None:
    # The mirror image, on a real campaign that names a chain and no circuit. Without
    # this the panel could quietly become "draw one under everything", which is the
    # failure the decision function exists to prevent.
    finished = _seeded("src/ui/pages/chat.py", finished_campaign)
    drawn = " ".join(str(block.value) for block in finished.markdown)
    assert "The circuit your question describes" not in drawn


@pytest.fixture(scope="module")
def raced_campaign() -> CampaignState:
    """A campaign for a question that asked to watch three methods converge."""
    return run_campaign(
        "Loss/learning curve for VQE, QAOA and VarQITE: 6 spins at criticality?",
        shot_budget=50_000_000,
        chat_model=None,
        search_corpus=False,
    )


def test_a_question_that_asked_to_watch_a_descent_gets_one_drawn(
    raced_campaign: CampaignState,
) -> None:
    # Offline, with no model anywhere: the three methods are run by this project's own
    # physics layer on the chain the campaign formalised, which is why the picture is
    # there at all on the path a first-time reader meets.
    assert raced_campaign["race"] is not None
    finished = _seeded("src/ui/pages/chat.py", raced_campaign)
    assert not finished.exception, [str(error.value) for error in finished.exception]
    drawn = " ".join(str(block.value) for block in finished.markdown)
    assert "How each method got there" in drawn
    for method in ("VQE", "QAOA", "VarQITE"):
        assert method in drawn, method


def test_the_drawn_race_says_it_shows_nothing_about_noise(
    raced_campaign: CampaignState,
) -> None:
    # The one thing the figure cannot show, stated in the caption rather than in an
    # expander. A reader who takes it as a statement about robustness to noise has
    # taken the opposite of what it says.
    finished = _seeded("src/ui/pages/chat.py", raced_campaign)
    said = " ".join(str(block.value) for block in finished.caption)
    assert "No noise anywhere in this figure" in said


def test_no_race_is_drawn_under_an_answer_that_asked_for_none(
    finished_campaign: CampaignState,
) -> None:
    finished = _seeded("src/ui/pages/chat.py", finished_campaign)
    drawn = " ".join(str(block.value) for block in finished.markdown)
    assert "How each method got there" not in drawn


@pytest.fixture(scope="module")
def searched_campaign() -> CampaignState:
    """A campaign that really touched the index, so there is a search trail to draw."""
    return run_campaign(
        "What is a barren plateau, and does it affect this chain?",
        shot_budget=1,
        chat_model=None,
        search_corpus=True,
        fetch_external=False,
    )


def test_the_pipeline_page_shows_the_query_that_actually_went_to_the_index(
    searched_campaign: CampaignState,
) -> None:
    """The rewriting was happening and nothing in the interface said so.

    Retrieval is a loop -- search, grade, rewrite when a round keeps nothing --
    and the rounds were discarded the moment the citations were extracted. So the
    page showed the question and then the citations, and a reader could not tell a
    corpus that is silent on the subject from a query that never named it.
    """
    finished = _seeded("src/ui/pages/pipeline.py", searched_campaign)
    assert not finished.exception, [str(error.value) for error in finished.exception]
    assert searched_campaign["searches"], "the fixture searched nothing, so this proves nothing"

    shown = [block.value for block in finished.code]
    assert searched_campaign["request"].text in shown, "the question as typed is not on the page"
    assert searched_campaign["searches"][-1].query in shown, "the query searched is not on the page"


def test_the_page_says_so_out_loud_when_no_query_was_rewritten(
    searched_campaign: CampaignState,
) -> None:
    # Showing the searched query only when a rewrite happened means the common case
    # displays a question and then silence, and a reader cannot tell from the screen
    # whether the query was left alone or merely not reported.
    finished = _seeded("src/ui/pages/pipeline.py", searched_campaign)
    captions = " ".join(str(block.value) for block in finished.caption)
    rewritten = any(item.kind == "rewrite" for item in searched_campaign["searches"])

    assert ("rewritten by" in captions) is rewritten
    assert ("no reformulation" in captions) is not rewritten


def test_a_rewritten_query_is_shown_as_a_rewrite_and_attributed(
    searched_campaign: CampaignState,
) -> None:
    """The case the panel exists for, and the one an offline run rarely produces.

    A rewrite only happens when a round keeps nothing, and the deterministic grader
    is generous enough that the corpus usually answers on the first try. So the
    rounds are set directly here: the alternative is a test that passes because the
    branch never ran.
    """
    rewritten = dict(searched_campaign)
    rewritten["searches"] = (
        SearchRound(
            query="thermal transport", kind="question", found=6, kept=0, reason="off topic"
        ),
        SearchRound(
            query="Bethe ansatz spin chain transport",
            kind="rewrite",
            found=4,
            kept=3,
            reason="closer",
            rewritten_by="grader",
        ),
    )
    finished = _seeded("src/ui/pages/pipeline.py", cast(CampaignState, rewritten))
    assert not finished.exception, [str(error.value) for error in finished.exception]

    shown = [block.value for block in finished.code]
    captions = " ".join(str(block.value) for block in finished.caption)

    # The rewritten query is what the answer rests on, so it is the one on the page.
    assert "Bethe ansatz spin chain transport" in shown
    assert "the relevance grader" in captions, "the rewrite was shown but not attributed"
    # And the round that kept nothing is still there -- it is the whole reason the
    # query changed, and a trace that shows only the round that worked explains
    # nothing about why it was needed.
    queried = list(finished.dataframe[0].value["query"])
    assert "thermal transport" in queried


def test_every_node_in_the_graph_has_its_purpose_stated() -> None:
    # The route table joins the compiled graph's nodes against NODE_PURPOSE, so a
    # node added to the agent appears there with a blank description until somebody
    # writes one. A blank cell in a table explaining the pipeline is worse than no
    # table, and this is what catches it.
    missing = [name for name in pipeline_nodes() if not NODE_PURPOSE.get(name)]
    assert not missing, f"nodes drawn with no explanation: {missing}"


def test_the_chat_page_offers_the_agents_follow_up_questions(
    finished_campaign: CampaignState,
) -> None:
    # The suggestions come from a node in the graph, so this checks the interface
    # actually surfaces them rather than that they exist.
    finished = _seeded("src/ui/pages/chat.py", finished_campaign)
    labels = {button.label for button in finished.button}
    offered = {suggestion.question for suggestion in finished_campaign["followups"].suggestions}
    assert offered, "the seeded campaign proposed nothing, so this proves nothing"
    assert offered <= labels, f"suggestions were not offered as buttons: {sorted(labels)}"


def test_pressing_a_follow_up_asks_it_by_the_same_path_a_typed_question_takes(
    finished_campaign: CampaignState,
) -> None:
    # A click and a typed question must reach the agent identically -- including
    # the input screen. Any second route would be a route with no guard on it.
    finished = _seeded("src/ui/pages/chat.py", finished_campaign)
    wanted = finished_campaign["followups"].suggestions[0].question
    pressed = next(button for button in finished.button if button.label == wanted)
    after = pressed.click().run()
    assert not after.exception, [str(error.value) for error in after.exception]


# --------------------------------------------------------------------------
# The Lab page, at knob positions the knob can actually reach
# --------------------------------------------------------------------------

LAB = "src/ui/pages/lab.py"
"""The page whose panels each depend on a different method's size ceiling."""

KNOB_POSITIONS: tuple[tuple[str, dict[str, object]], ...] = (
    ("the shortest chain the knob offers", {"n_sites": 2}),
    ("a chain past the picture ceiling", {"n_sites": 10}),
    ("a chain past the exact-answer ceiling", {"n_sites": 16}),
    ("the longest chain the knob offers", {"n_sites": knob.MAX_SITES, "boundary": "periodic"}),
    ("an odd ring, which needs three rounds", {"n_sites": 7, "boundary": "periodic"}),
    ("no push at all, the classical limit", {"n_sites": 6, "field": 0.0}),
    ("no circuit layers", {"depth": 0}),
    ("a machine with no wiring problems", {"device": "ideal"}),
    ("a machine with plenty", {"device": "heavy-hex-27", "n_sites": 6}),
    ("the reserved longitudinal knob", {"longitudinal": 0.3}),
)
"""Every position that used to reach a different branch, or a different refusal.

Four of these once produced a stack trace in the browser rather than a page. The
knob offers chains up to twenty magnets and the exact solver refuses above twelve,
so a page that called it unguarded died the moment somebody dragged the slider --
and dragging the slider is the first thing anybody does. The ceilings are real and
each panel is entitled to refuse; what it may not do is take the page down with it.
"""


LAB_TABS: tuple[str, ...] = (
    "1 · What it is",
    "2 · What the machine does",
    "3 · Which method",
    "4 · The circuit",
    "5 · Tuning it",
    "6 · The true answer",
    "7 · Why an ordinary computer competes",
    "Report",
)
"""The Lab's tabs, in the order a reader needs them rather than the order they compute.

Asserted as a sequence and not as a set. The order *is* the argument: what the thing
is, then how an algorithm of this kind works at all, then which of the four such
algorithms to reach for, then what gets built, then the search, then the answer it is
chasing, then why an ordinary computer is a serious competitor -- and only last the
block of numbers, which is the one part a reader with no physics has no use for.

Numbered on the label, and the numbers are asserted here rather than left as
decoration. Eight unnumbered tabs offer a reader with no physics no way to find the
reading order except by opening each one, and an order that only the source code
knows about is an order the interface does not have.
"""


@pytest.mark.parametrize(("description", "physics"), KNOB_POSITIONS, ids=lambda value: str(value))
def test_the_lab_survives_every_position_of_the_knob(
    description: str, physics: dict[str, object]
) -> None:
    from src.ui import setting as knob

    app = AppTest.from_file(str(PROJECT_ROOT / LAB), default_timeout=RENDER_TIMEOUT)
    app.session_state[panels.SETTING_KEY] = knob.Setting().with_physics(**physics)
    finished = app.run()
    assert not finished.exception, (
        f"the Lab page fell over at {description}: "
        f"{[str(error.value) for error in finished.exception]}"
    )


def test_the_lab_leads_with_the_physics_and_ends_with_the_numbers() -> None:
    # The order is the argument. Somebody arriving from the Chat page has been given
    # a verdict about a chain and has never been told what a chain is, so the first
    # tab is pictures and the last one is the block of figures to paste elsewhere.
    # Sliced off the front, because the shared sidebar draws tabs of its own and
    # they land in the same list.
    finished = render(LAB)
    drawn = [str(getattr(block, "label", "")) for block in finished.tabs]
    assert drawn[: len(LAB_TABS)] == list(LAB_TABS)


def test_every_answer_reaches_the_page_through_the_maths_repair() -> None:
    # `src.agent.mathmarkup` exists to rewrite the LaTeX delimiters Streamlit does
    # not render into the two it does, and for a long while it had no caller at all:
    # the prompt asked models to use dollar signs and nothing enforced it, which is
    # a request rather than a guarantee. `panels.prose` is the enforcement, applied
    # at the last possible moment so no branch can bypass it by composing its answer
    # somewhere new. This asserts the wire rather than the intention.
    drawn: list[str] = []
    original = panels.st.markdown
    panels.st.markdown = lambda text, *args, **kwargs: drawn.append(str(text))  # type: ignore[assignment]
    try:
        panels.prose(r"the energy is \(E_0\) per site")
    finally:
        panels.st.markdown = original
    assert drawn == ["the energy is $E_0$ per site"]


def test_the_maths_repair_leaves_correct_notation_alone() -> None:
    # Idempotent, which is what makes it safe to apply to everything on the way out
    # rather than only to text a model wrote.
    already = "the gap is $2|J - h|$ and closes at $h = J$"
    drawn: list[str] = []
    original = panels.st.markdown
    panels.st.markdown = lambda text, *args, **kwargs: drawn.append(str(text))  # type: ignore[assignment]
    try:
        panels.prose(already)
    finally:
        panels.st.markdown = original
    assert drawn == [already]


# --------------------------------------------------------------------------
# What the answer looks like when it reaches the page
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("2111.05176", "[arXiv:2111.05176](https://arxiv.org/abs/2111.05176)"),
        ("2011.12245v2", "[arXiv:2011.12245v2](https://arxiv.org/abs/2011.12245v2)"),
        ("cond-mat/9804280", "[arXiv:cond-mat/9804280](https://arxiv.org/abs/cond-mat/9804280)"),
    ],
)
def test_an_arxiv_citation_is_offered_as_a_link(identifier: str, expected: str) -> None:
    # A citation nobody can follow is a citation the reader has to take on trust,
    # which is the opposite of the reason it was retrieved.
    assert panels.source_link(identifier) == expected


@pytest.mark.parametrize("identifier", ["pfeuty-notes", "", "chapter 4", "10.1103/PhysRev.65.117"])
def test_anything_that_is_not_an_arxiv_identifier_is_not_linked(identifier: str) -> None:
    # A link built from an identifier that is not one goes to a 404, and a broken
    # link is worse than no link because it looks checkable.
    assert "arxiv.org" not in panels.source_link(identifier)


def test_the_no_model_answer_leads_with_the_passages_rather_than_reprinting_them() -> None:
    """Offline, the retrieved passages *are* the answer -- but not all of them at once.

    The branch used to print every chunk in full, one blockquote after another, and
    the result was several screens of quotation with the answer somewhere inside
    it. That is honest and unreadable, which is its own kind of dishonesty: material
    a reader scrolls past has not been communicated. The full text is not withheld
    -- it is under **Sources** -- so this asserts the lead is shorter than the
    passage it leads with, and that nothing was silently dropped.
    """
    from src.agent.explaining import explain
    from src.agent.model_selection import ModelPool
    from src.agent.state import Citation

    passage = "## The gap\n\n" + "It closes linearly at the critical point. " * 30
    citations = tuple(
        Citation(
            identifier=f"note-{index}",
            title=f"Note {index}",
            snippet=passage,
            shelf="physics-notes",
        )
        for index in range(3)
    )
    # `offline=True`, not `pool=None`. Omitting the pool *builds* one from
    # configuration, so the test the docstring describes as offline was in fact
    # constructing a client and waiting on a gateway -- 100 seconds of the suite,
    # for a branch that is supposed to touch no model at all.
    written = explain("what closes the gap?", None, citations, ModelPool(offline=True))

    assert written.cited == 3
    for index in (1, 2, 3):
        assert f"[{index}]" in written.text, "a passage lost its number"
    assert len(written.text) < len(passage) * 3, "the lead is no shorter than the passages"
    assert written.text.count(passage) == 0, "a passage was reprinted in full"
    assert "Sources" in written.text, "the reader is not told where the full text is"


ANALYTICS = "src/ui/pages/analytics.py"
EVALUATIONS = "src/ui/pages/evaluations.py"

ANALYTICS_TABS: tuple[str, ...] = (
    "Model spend",
    "Model bake-off",
    "Hardware cost surface",
    "Models",
)
"""The Analytics tabs, and the order is again the argument.

Named for their currency, not for the word "cost". The page carries two budgets that
are not interchangeable -- money for model calls, and measurements on a quantum
device -- and it once called one tab *This session* and the next *Cost surface*,
which left the reader to work out that the second was not more of the first.

The session's own evidence comes first because it is the part a reader cannot get
from a screenshot: unequal bars over the graph's nodes are what separate a branching
pipeline from a single prompt behind a text box. The bake-off comes second because
it is the experiment the project's central claim invites -- swap the model and the
numbers must not move. The physics cost surface and the model inventory are
reference, and reference goes last.
"""

EVALUATION_TABS: tuple[str, ...] = ("Results", "Try it yourself", "What is measured")
"""The Evaluations tabs.

The result before the methodology, deliberately. A reader who wants the finding
should not have to read the method first, and a reader who doubts the finding should
not have to hunt for the method.
"""


def test_the_analytics_page_leads_with_this_sessions_own_evidence() -> None:
    finished = render(ANALYTICS)
    drawn = [str(getattr(block, "label", "")) for block in finished.tabs]
    assert drawn[: len(ANALYTICS_TABS)] == list(ANALYTICS_TABS)


def test_the_evaluations_page_leads_with_the_result_and_not_the_method() -> None:
    finished = render(EVALUATIONS)
    drawn = [str(getattr(block, "label", "")) for block in finished.tabs]
    assert drawn[: len(EVALUATION_TABS)] == list(EVALUATION_TABS)


def test_the_analytics_page_draws_its_sliders_once() -> None:
    # It reads the knob before the tabs rather than inside two of them. Streamlit
    # rejects two widgets built from identical parameters, so a second
    # `current_setting()` on a page opened on its own takes the whole page down --
    # and it takes it down only on the path a test or a bookmark uses, never on the
    # path through the navigation, which is the worst kind of breakage to own.
    finished = render(ANALYTICS)
    assert not finished.exception
    labels = [str(getattr(block, "label", "")) for block in finished.slider]
    assert len(labels) == len(set(labels)), f"a slider was drawn twice: {labels}"


def test_the_bakeoff_says_what_it_needs_rather_than_offering_a_dead_button() -> None:
    # The one feature in this application that cannot run without a credential. A
    # dial that does nothing is worse than no dial, so with no key the tab draws the
    # reason instead of the controls.
    from src.ui import setting as knob

    app = AppTest.from_file(str(PROJECT_ROOT / ANALYTICS), default_timeout=RENDER_TIMEOUT)
    app.run()
    body = " ".join(str(block.value) for block in app.markdown)
    if knob.credential_available():
        assert "Run the bake-off" in " ".join(str(block.label) for block in app.button)
    else:
        assert "OPENROUTER_API_KEY" in body


def test_the_buttons_and_the_hidden_questions_account_for_every_starter() -> None:
    # Two hand-written lists, and the failure they guard against is silent: a
    # question left off both simply stops being offered, with nothing anywhere
    # saying it was dropped.
    assert set(OFFERED_KEYS) | set(NOT_OFFERED) == set(STARTERS)
    assert not set(OFFERED_KEYS) & set(NOT_OFFERED)
    assert len(OFFERED_KEYS) == len(set(OFFERED_KEYS))


def test_the_buttons_fill_whole_rows() -> None:
    # The reason four were taken off the display. Eight is two full rows of four; a
    # count that is not a multiple of the row width leaves a ragged last row, which
    # reads as a question missing rather than as a set ending.
    assert len(OFFERED) % 4 == 0


def test_the_request_for_code_is_the_last_button_of_the_first_row() -> None:
    # Placed rather than inherited from the dictionary's order, so it is worth
    # pinning: a reader who has just seen the three methods compared is the one most
    # likely to want the circuit, and this is the last thing read before the eye
    # drops to the second row.
    assert OFFERED[3] == STARTERS["implement"]


def test_the_page_count_in_prose_is_the_number_of_pages_there_are() -> None:
    # Three places said "one page out of eight" or "out of seven" about a
    # navigation holding seven. A count written by hand beside a list that grows
    # is a count that goes stale the first time the list does.
    # Spelled out in the prose, so the count is read back through the same word
    # table the agent parses questions with rather than a second one written here.
    for path in (
        PROJECT_ROOT / "src/ui/app.py",
        PROJECT_ROOT / "src/ui/panels.py",
        PROJECT_ROOT / "README.md",
    ):
        written = re.findall(r"page out of (\w+)", path.read_text(encoding="utf-8"))
        assert written, f"{path} lost the sentence this test guards"
        for word in written:
            assert WORD_NUMBERS.get(word) == len(panels.PAGES), (
                f"{path} says 'page out of {word}' and the navigation holds {len(panels.PAGES)}"
            )


def test_the_only_control_that_erases_everything_says_so_and_asks_twice() -> None:
    # It read "Forget this conversation" and called Memory.forget(), which erases
    # every thread the user has -- so the most destructive control in the app
    # claimed the smallest scope, and did it on one press while deleting a single
    # chat asked for two.
    source = (PROJECT_ROOT / "src/ui/panels.py").read_text(encoding="utf-8")
    handler = source[source.index("def _forget_everything_button") :]
    handler = handler[: handler.index("\n\nVECTOR_SHARE_LABEL")]

    # The label on the widget, not the word anywhere in the file: this function's
    # own docstring names the old label to record what went wrong.
    assert 'st.button("Forget this conversation"' not in source, "the misleading label is back"
    assert '"Forget everything"' in handler
    assert "memory_forget_armed" in handler, "it no longer arms before erasing"
    # The erase sits under the armed branch, above the press that arms it.
    armed = handler[handler.index('st.session_state.get("memory_forget_armed")') :]
    arms = 'st.session_state["memory_forget_armed"] = True'
    assert armed.index("store.forget()") < armed.index(arms), "it erases before it asks"


def test_clearing_the_whole_chat_list_asks_twice_and_keeps_the_ratings() -> None:
    # The listing grew a *Delete all chats* control beside the per-row bins. It has
    # to arm before it writes, like every other erasure here, and it has to call the
    # method that drops conversations rather than the one that drops everything --
    # calling `forget` would quietly unteach the reading level as a side effect of
    # tidying a sidebar.
    source = (PROJECT_ROOT / "src/ui/panels.py").read_text(encoding="utf-8")
    handler = source[source.index("def _delete_all_chats_button") :]
    handler = handler[: handler.index("\n\ndef ")]

    assert "store.forget_all_threads()" in handler
    assert "store.forget()" not in handler, "clearing the list erases the ratings too"
    armed = handler[handler.index('st.session_state.get("chat_delete_all_armed")') :]
    arms = 'st.session_state["chat_delete_all_armed"] = True'
    assert armed.index("store.forget_all_threads()") < armed.index(arms), "it erases before it asks"

    # Below the rows, not above them: a control that empties the list must not sit
    # where a reader aims for the newest conversation in it.
    listing = source[source.index("def past_chats(") : source.index("def _delete_chat_button")]
    assert listing.index("_delete_chat_button(") < listing.index("_delete_all_chats_button(")


def test_the_chat_page_hands_the_campaign_the_chain_it_is_showing() -> None:
    # Every physics dial the panel offers reaches `run_campaign`, and these five
    # reached it through nothing at all: they drew the circuit at the top of the page
    # and ran the Lab, so the page could show a ten-spin ring while the answer under
    # it read *TFIM L=6 (open)*. Checked on the call rather than through a campaign
    # because the connection is what was missing; what it then does is checked where
    # the dials are.
    source = (PROJECT_ROOT / "src/ui/pages/chat.py").read_text(encoding="utf-8")
    call = source[source.index("    return run_campaign(") :]
    call = call[: call.index("\n\n\ndef ")]

    assert "chain_defaults=chosen.physics.as_reading()" in call


# --------------------------------------------------------------------------
# The chat page's own settings, with the override ticked
# --------------------------------------------------------------------------
#
# `chat_settings` returns early while its checkbox is unticked, which it is by
# default -- so every other test in this file renders the page and stops at that
# guard. The whole override branch, some three hundred and seventy lines of it, had
# never been drawn by anything. It was not broken, but it was the largest unexercised
# block in the interface, and the last two defects found here both lived in a render
# path nothing entered.


CHAT_PAGE = "src/ui/pages/chat.py"


def _chat_with_override() -> AppTest:
    """Render the chat page with its settings panel taking control.

    Returns:
        The finished run, with the override checkbox ticked.
    """
    app = AppTest.from_file(str(PROJECT_ROOT / CHAT_PAGE), default_timeout=RENDER_TIMEOUT)
    app.session_state["chat_override"] = True
    return app.run()


def test_the_chat_settings_override_draws_without_raising() -> None:
    finished = _chat_with_override()
    assert not finished.exception, [str(error.value) for error in finished.exception]


def test_the_override_offers_the_dials_it_promises() -> None:
    # The docstring on `chat_settings` names the dials it puts on the page. A panel
    # that quietly stopped drawing one would still render, and the promise would be
    # wrong rather than the code.
    finished = _chat_with_override()
    labels = {widget.label for widget in finished.slider} | {
        widget.label for widget in finished.selectbox
    }
    for promised in (
        "Passages to use",
        "Searches per question",
        "Sites $L$",
        "Coupling $J$",
        "Field $h$",
        "Circuit layers",
        "Answering model",
    ):
        assert promised in labels, f"the override no longer offers {promised!r}"


def test_no_dial_on_the_chat_panel_collides_with_a_sidebar_dial() -> None:
    # Two widgets sharing a key share a value, so moving one would silently move the
    # other and the page would fight the sidebar it is meant to override.
    panel = (PROJECT_ROOT / "src/ui/panels.py").read_text(encoding="utf-8")
    sidebar = (PROJECT_ROOT / "src/ui/setting.py").read_text(encoding="utf-8")
    body = panel[panel.index("def chat_settings(") : panel.index("def past_chats(")]
    keys = set(re.findall(r'key="([^"]+)"', body))
    assert keys, "the override draws no keyed widget at all"
    assert not keys & set(re.findall(r'key="([^"]+)"', sidebar))
    elsewhere = panel.replace(body, "")
    assert not keys & set(re.findall(r'key="([^"]+)"', elsewhere))
