"""Offline tests for the 🏠 Home dashboard's cards and for where the hero banner is drawn.

Two properties that only exist once a page is actually rendered, and that nothing else in the
suite would notice going wrong:

* **Every workspace card navigates.** The cards were styled to look clickable long before they
  were, so a regression here would not look like a bug — it would look like the design. Each
  card carries a button keyed ``home_go_<page key>`` and resolves through the same registry
  the sidebar uses, so a page renamed in its own module cannot leave a dead card behind.
* **The hero is Home's alone.** It used to open every page, putting a gradient banner above
  the conversation on 💬 AI Chat. A test is the only thing that keeps it from drifting back:
  the entry point is where a "draw this everywhere" call naturally wants to live.

Both run against small harness apps rather than ``src/app.py``, so no Chroma index is opened
and no model is contacted.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

import src.ui.pages  # noqa: F401  (import registers every workspace page)
from src.ui import state
from src.ui.pages import chat, home
from src.ui.registry import get_pages
from src.ui.state import KBStatus

_HOME_APP = """
from src.ui.pages.home import render
from src.ui.state import init_state

init_state()
render()
"""

_CHAT_APP = """
import streamlit as st

from src.ui.pages.chat import render
from src.ui.state import init_state

init_state()
st.session_state.ctx = {"model": st.session_state.model, "level": "Beginner", "subjects": ()}
render()
"""


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render with no index behind it — no Chroma load, no embedding backend, no network."""
    monkeypatch.setattr(state, "load_kb_status", lambda: KBStatus(kb=None))
    monkeypatch.setattr(home, "load_kb", lambda: None)
    monkeypatch.setattr(chat, "load_kb", lambda: None)


def _run(script: str) -> AppTest:
    return AppTest.from_string(script, default_timeout=60).run()


# --- The cards are the navigation they look like ----------------------------------------------


def test_every_workspace_has_a_card_that_can_be_pressed(offline: None) -> None:
    at = _run(_HOME_APP)
    assert not at.exception
    pressable = {b.key for b in at.button if b.key.startswith("home_go_")}
    expected = {f"home_go_{p.key}" for p in get_pages() if p.key != "home"}
    assert pressable == expected


def test_a_card_carries_its_own_label_and_blurb(offline: None) -> None:
    # The label comes from the registry and the blurb from _BLURBS; a card showing one without
    # the other is a card a reader cannot tell apart from its neighbour.
    at = _run(_HOME_APP)
    chat_card = next(b for b in at.button if b.key == "home_go_chat")
    assert chat_card.label == "💬 AI Chat"
    assert home._BLURBS["chat"] in [c.value for c in at.caption]


def test_pressing_a_card_asks_the_registry_for_that_page(
    offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The jump itself is Streamlit's; what this pins is that the press reaches it at all, with
    # the right key. An inert card — the state these were in — would record nothing here.
    asked: list[str] = []
    monkeypatch.setattr(home, "go_to_page", lambda key, hint, **kw: asked.append(key))

    at = _run(_HOME_APP)
    at.button(key="home_go_ml_lab").click().run()
    assert asked == ["ml_lab"]


# --- The hero introduces the app once ---------------------------------------------------------


def _has_hero(at: AppTest) -> bool:
    return any('class="hero"' in element.value for element in at.markdown)


def test_home_opens_with_the_hero_banner(offline: None) -> None:
    assert _has_hero(_run(_HOME_APP))


def test_chat_does_not_repeat_it_above_the_conversation(offline: None) -> None:
    # The whole reason it moved: a workspace should start with its own content, and the
    # sidebar carries the identity on every page anyway.
    at = _run(_CHAT_APP)
    assert not at.exception
    assert not _has_hero(at)
