"""Offline tests for cross-page navigation wiring.

The 🎓 AI/ML Tutor's "Practise" button and the 🔬 AI/ML Lab's toggle row jump between
workspaces by *registry key* via :func:`src.ui.registry.switch_to_page`. That is convenient but
fragile: renaming a page's ``key`` would silently turn a "Practise"/"Learn" button into a
dead-end at runtime, with nothing to catch it offline. These tests pin the contract down —
every key the app navigates to resolves to a registered page, keys are unique, and an unknown
key degrades to a caller-handled ``False`` rather than crashing.

Importing :mod:`src.ui.pages` only needs Streamlit (a core dependency): the Lab's data
dependencies (pandas / scikit-learn / matplotlib) are imported lazily inside recipes, so this
stays in the offline suite.
"""

from __future__ import annotations

import pytest

import src.ui.pages  # noqa: F401  (import registers every workspace page as a side effect)
from src.lab.curriculum import TRACKS
from src.ui.registry import SECTION_ORDER, get_pages, get_sections, switch_to_page
from src.ui.theme import NAV_TITLE

# Every page key the app navigates to programmatically. Keep in step with the switch_to_page()
# calls in src/ui/pages/ml_tutor.py (→ ml_lab) and ml_lab.py (→ ml_tutor, chat).
_NAV_TARGETS = {"ml_tutor", "ml_lab", "chat"}


def _registered_keys() -> set[str]:
    return {p.key for p in get_pages()}


def test_registered_page_keys_are_unique():
    # switch_to_page() resolves by key; a duplicate key would make navigation ambiguous.
    keys = [p.key for p in get_pages()]
    assert len(keys) == len(set(keys)), f"duplicate page keys: {keys}"


def test_all_navigation_targets_resolve_to_registered_pages():
    missing = _NAV_TARGETS - _registered_keys()
    assert not missing, f"navigation targets not registered: {missing}"


def test_every_practice_lesson_can_reach_the_lab():
    # A lesson carrying a Practice renders a "🔬 Practise" button that switch_to_page("ml_lab")s;
    # if that key weren't registered the button would dead-end. Guard both the mapping's
    # existence and its destination.
    practices = [lesson for track in TRACKS for lesson in track.lessons if lesson.practice]
    assert practices, "curriculum has no runnable Practice lessons to navigate from"
    assert "ml_lab" in _registered_keys()


def test_switch_to_page_unknown_key_returns_false():
    # The offline-safe branch: an unknown key returns False *before* importing Streamlit,
    # letting callers fall back to an in-place message instead of raising.
    assert switch_to_page("no-such-page") is False


def test_quiz_page_sits_with_the_other_learning_workspaces():
    # The 🎯 Trivia page completes the learn→test loop, so it belongs beside the Tutor and the
    # Lab rather than off in its own group.
    quiz = next((p for p in get_pages() if p.key == "quiz"), None)
    assert quiz is not None, "quiz page not registered"
    assert quiz.section == "Learn"


def test_sidebar_groups_are_in_the_intended_order():
    # The sidebar and the Home overview are both built from get_sections(), so this pins the
    # shape of both: what you came to do first, the machinery under it, the knobs last.
    assert list(get_sections()) == [
        NAV_TITLE,
        "Learn",
        "Analyse",
        "Knowledge",
        "System",
    ]


def test_each_group_holds_exactly_the_pages_it_should():
    # Guards the grouping a page declares in its own decorator, which is otherwise only
    # visible by running the app. Order within a group is the registry's `order` field.
    grouped = {name: [p.key for p in pages] for name, pages in get_sections().items()}
    assert grouped == {
        NAV_TITLE: ["home", "chat"],
        "Learn": ["ml_tutor", "ml_lab", "quiz"],
        "Analyse": ["analytics", "evaluation", "ab_testing", "experiments"],
        "Knowledge": ["ai_news", "research", "knowledge_base"],
        "System": ["settings"],
    }


def test_home_is_the_first_page_of_the_first_group():
    # `st.navigation` takes the page marked `default` as the landing destination, and app.py
    # marks "home". It should also be the first row a reader sees, not buried mid-list.
    first_group = next(iter(get_sections().values()))
    assert first_group[0].key == "home"


def test_home_cards_cover_every_registered_page():
    # The sidebar (grouped by section) and the Home cards are both built from get_sections(),
    # so they can't drift. Guard that every non-home page has a Home blurb — a new page that
    # forgets one would ship a blank card.
    from src.ui.pages.home import _BLURBS

    keys = {p.key for p in get_pages() if p.key != "home"}
    missing = keys - set(_BLURBS)
    assert not missing, f"pages with no Home blurb: {missing}"


def test_every_registered_section_is_ordered():
    # get_sections() drives the grouped sidebar; every section a page declares should be a
    # known, ordered one (an unknown section would silently sort to the end).
    sections = set(get_sections())
    assert sections <= set(SECTION_ORDER), f"unordered sections: {sections - set(SECTION_ORDER)}"


# --- go_to_page: the jump and its fallback ----------------------------------------------------


def test_go_to_page_announces_nothing_when_the_jump_succeeds(monkeypatch):
    # A successful switch_page reruns the app, so anything drawn after it is discarded. Any
    # message here would be dead code that only ever appeared on the failure path anyway.
    import streamlit as st

    from src.ui import registry

    said: list[str] = []
    monkeypatch.setattr(registry, "switch_to_page", lambda _key: True)
    monkeypatch.setattr(st, "info", lambda msg: said.append(msg))
    monkeypatch.setattr(st, "warning", lambda msg: said.append(msg))

    registry.go_to_page("chat", "Open **💬 AI Chat** from the sidebar.")
    assert said == []


def test_go_to_page_names_the_sidebar_entry_when_the_destination_is_missing(monkeypatch):
    # The whole point of the fallback: a reader who cannot be moved there has to be told
    # where to go themselves, rather than pressing a button that does nothing.
    import streamlit as st

    from src.ui import registry

    said: list[str] = []
    monkeypatch.setattr(st, "info", lambda msg: said.append(msg))

    registry.go_to_page("no-such-page", "Open **💬 AI Chat** from the sidebar.")
    assert said == ["Open **💬 AI Chat** from the sidebar."]


def test_go_to_page_can_raise_its_voice(monkeypatch):
    import streamlit as st

    from src.ui import registry

    warned: list[str] = []
    monkeypatch.setattr(st, "info", lambda msg: pytest.fail("should have used warning"))
    monkeypatch.setattr(st, "warning", lambda msg: warned.append(msg))

    registry.go_to_page("no-such-page", "Practise it in the 🔬 Lab.", level="warning")
    assert warned == ["Practise it in the 🔬 Lab."]
