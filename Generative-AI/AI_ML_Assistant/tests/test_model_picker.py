"""Offline tests for the shared model picker (:mod:`src.ui.models`) and its two call sites.

The active model is one setting rendered in two places — ⚙️ Settings and the 💬 Chat
"Answer settings" popover — and both bind to ``st.session_state["model"]``. That sharing is
the whole design, and it is invisible in either page on its own, so these tests pin it:

* the list offered is what the deployment can actually reach (no Gemini without a key);
* a persisted choice that is no longer reachable is clamped rather than left to fail at the
  first question;
* picking a model in Chat is the model that answers the *next* question — the entry point
  builds ``ctx`` before the page runs, so the page has to take the value back afterwards.

The picker tests drive a two-line harness app through ``AppTest`` rather than the whole of
``src/app.py``: the widget is the unit under test, and the harness keeps the run under a
second. Nothing here touches a model or a Chroma index.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.config import OPENROUTER_MODELS
from src.core.service import AnswerBundle
from src.ui import models as ui_models
from src.ui import state
from src.ui.models import available_models, model_label, model_option
from src.ui.pages import chat
from src.ui.state import KBStatus

_PICKER_APP = """
from src.ui.models import model_cost_hint, model_picker

chosen = model_picker("Model")
model_cost_hint(chosen)
"""

# The Chat page as the entry point runs it: session state seeded, `ctx` handed in from
# outside, no knowledge base (patched out by the fixture below).
_CHAT_APP = """
import streamlit as st

from src.ui.pages.chat import render
from src.ui.state import init_state

init_state()
st.session_state.ctx = {
    "model": st.session_state.get("_stale_model", st.session_state.model),
    "level": "Beginner",
    "subjects": (),
}
render()
"""


@pytest.fixture
def no_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend no ``GOOGLE_API_KEY`` is set, so the list is the OpenRouter one exactly."""
    monkeypatch.setattr(ui_models, "google_api_key", lambda: "")


@pytest.fixture
def offline_chat(monkeypatch: pytest.MonkeyPatch, no_gemini: None) -> None:
    """Render the Chat page with no index behind it — no Chroma load, no embedding backend."""
    monkeypatch.setattr(state, "load_kb_status", lambda: KBStatus(kb=None))
    monkeypatch.setattr(chat, "load_kb", lambda: None)


# --- The list on offer ------------------------------------------------------------------------


def test_gemini_is_offered_only_when_its_key_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # gemini-native bypasses OpenRouter and goes straight to Google, so listing it without a
    # key would be a choice that looks available and fails at the first question.
    monkeypatch.setattr(ui_models, "google_api_key", lambda: "")
    assert available_models() == OPENROUTER_MODELS

    monkeypatch.setattr(ui_models, "google_api_key", lambda: "a-google-key")
    assert available_models() == [*OPENROUTER_MODELS, "gemini-native"]


def test_an_option_carries_its_label_and_price() -> None:
    option = model_option("openai/gpt-4o-mini")
    assert option.startswith("GPT-4o mini")
    assert "per 1M" in option


def test_an_unknown_id_is_shown_as_itself_rather_than_hidden() -> None:
    # A model retired from the registry while a session still points at it: show the raw id,
    # which is at least searchable, instead of a blank row.
    assert model_option("vendor/retired-model") == "vendor/retired-model"
    assert model_label("vendor/retired-model") == "vendor/retired-model"


# --- The widget -------------------------------------------------------------------------------


def test_the_picker_offers_every_reachable_model(no_gemini: None) -> None:
    at = AppTest.from_string(_PICKER_APP, default_timeout=30).run()
    assert not at.exception
    assert len(at.selectbox[0].options) == len(OPENROUTER_MODELS)
    assert at.session_state["model"] == OPENROUTER_MODELS[0]


def test_the_picker_explains_what_the_choice_costs(no_gemini: None) -> None:
    at = AppTest.from_string(_PICKER_APP, default_timeout=30).run()
    assert "per typical answer" in at.caption[0].value


def test_an_unreachable_persisted_choice_is_clamped(no_gemini: None) -> None:
    # The Gemini key disappearing between runs must not leave the session pointing at a
    # model it can no longer call.
    at = AppTest.from_string(_PICKER_APP, default_timeout=30)
    at.session_state["model"] = "gemini-native"
    at.run()
    assert at.session_state["model"] == OPENROUTER_MODELS[0]
    assert not at.exception


# --- The Chat popover -------------------------------------------------------------------------


def test_chat_answer_settings_carries_the_model_picker(offline_chat: None) -> None:
    at = AppTest.from_string(_CHAT_APP, default_timeout=60).run()
    assert not at.exception
    assert at.selectbox(key="model").label == "Model"


def test_chat_and_settings_are_one_setting_not_two(offline_chat: None) -> None:
    # Both pages bind the same session key, so a model picked beside the question is the
    # model ⚙️ Settings shows — and the one the entry point puts in `ctx`.
    at = AppTest.from_string(_CHAT_APP, default_timeout=60).run()
    at.selectbox(key="model").select(OPENROUTER_MODELS[2]).run()

    assert at.session_state["model"] == OPENROUTER_MODELS[2]
    assert at.session_state["ctx"]["model"] == OPENROUTER_MODELS[2]


def test_the_header_names_the_model_without_opening_the_popover(offline_chat: None) -> None:
    at = AppTest.from_string(_CHAT_APP, default_timeout=60).run()
    at.selectbox(key="model").select(OPENROUTER_MODELS[2]).run()

    wanted = model_label(OPENROUTER_MODELS[2])
    assert any(wanted in caption.value for caption in at.caption)


def test_the_picked_model_is_the_one_that_answers(
    offline_chat: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The property the popover exists for. `ctx` is built by the entry point *before* the
    # page runs, so a stale value there would quietly send the question to the old model
    # however loudly the picker disagreed.
    asked: list[str] = []

    class _RecordingService:
        def answer(self, request, on_node=None):  # a stub standing in for the pipeline
            asked.append(request.model)
            return AnswerBundle(text="…", meta={"model": request.model, "level": request.level})

    monkeypatch.setattr(chat, "get_service", _RecordingService)

    at = AppTest.from_string(_CHAT_APP, default_timeout=60)
    at.session_state["_stale_model"] = "vendor/retired-model"
    at.session_state["model"] = OPENROUTER_MODELS[1]
    at.session_state["pending"] = ("What is an embedding?", "Beginner")
    at.run()

    assert asked == [OPENROUTER_MODELS[1]]
