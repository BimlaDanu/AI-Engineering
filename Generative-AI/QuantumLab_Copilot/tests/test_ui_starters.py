"""Tests for the questions the chat page offers on buttons.

These are the six questions most visitors will actually ask, because they are the
ones that need no typing. The claim made about them is that they exercise the
agent's different paths rather than six variations of one search -- and a claim
about the openers is worth checking, because they are the part of the application
easiest to change without noticing what it costs.

What can be checked offline is the routing and the shape. Which *actions* the loop
then takes depends on what the grader makes of the passages that come back, so that
is checked live by ``make live-check``, which asks all six and prints each
trajectory.
"""

from __future__ import annotations

from src.agent.router import heuristic_route
from src.ui.starters import QUESTIONS, STARTERS


def test_every_starter_is_a_question_somebody_would_type() -> None:
    for key, question in STARTERS.items():
        assert question.strip() == question, key
        assert question.endswith(("?", ".")), key
        # Long enough to name what it wants, short enough to read on a button.
        assert 25 <= len(question) <= 110, (key, len(question))


def test_the_sequence_and_the_mapping_hold_the_same_questions() -> None:
    # `live_check` asks QUESTIONS and the page draws STARTERS; if they could drift,
    # the live check would be verifying questions nobody is offered.
    assert QUESTIONS == tuple(STARTERS.values())


def test_no_two_starters_are_the_same_question() -> None:
    assert len(set(QUESTIONS)) == len(QUESTIONS)


def test_the_starters_do_not_all_take_the_same_route() -> None:
    """The point of the set: six openers, not one opener six times.

    A visitor who presses every button should see the agent search the notes, solve
    and plot a chain, and reach past the corpus -- not six searches. This asserts
    the weakest form of that (more than one route) rather than pinning each question
    to a route, because the phrasings are meant to be editable and the router is
    meant to keep classifying them sensibly.
    """
    routes = {heuristic_route(question).route for question in QUESTIONS}
    assert len(routes) > 1, routes


def test_a_starter_asks_for_a_computation() -> None:
    # Without one, the openers would never demonstrate the half of this project that
    # computes and cross-checks anything.
    assert any(heuristic_route(question).route == "compute" for question in QUESTIONS)


def test_no_starter_opens_on_a_clarifying_question() -> None:
    # A vague opener would show a new visitor the clarification path first, which is
    # the least interesting thing the agent does.
    for question in QUESTIONS:
        assert heuristic_route(question).route != "clarify", question


def test_no_starter_asks_the_agent_about_itself() -> None:
    # "Who are you?" works when typed. Spending one of six openings on it would
    # advertise the machinery ahead of the physics.
    for question in QUESTIONS:
        assert heuristic_route(question).route != "about", question
