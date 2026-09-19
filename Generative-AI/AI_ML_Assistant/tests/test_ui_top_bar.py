"""Offline tests for the top bar and the 🕘 Past chats panel, driven through ``AppTest``.

These run the real ``src/app.py`` in Streamlit's headless test runtime, so they cover the
wiring that unit tests cannot: that the four controls actually reach the page, that the login
buttons are live or disabled for the right reason, and that two presses on a delete button
really remove a file. No network is touched — nothing here asks a model anything.

Secrets are injected into the test runtime rather than written to ``.streamlit/secrets.toml``;
no credential file is created by running the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.core.chats import ChatStore, namespace_for, new_thread_id

# What `src.ui.past_chats._identity` returns when no identity provider is configured — the
# local `make run` case, where there is exactly one person using the process.
_LOCAL = "local-operator"

_FAKE_OIDC = {
    "redirect_uri": "http://localhost:8501/oauth2callback",
    "cookie_secret": "x" * 32,
    "client_id": "fake-client-id",
    "client_secret": "fake-client-secret",
    "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
}


@pytest.fixture
def chat_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the conversation store at a temporary directory for the duration of a test."""
    monkeypatch.setenv("SYNAPSE_CHAT_DIR", str(tmp_path))
    return tmp_path


def _app() -> AppTest:
    return AppTest.from_file("src/app.py", default_timeout=120)


def _store(root: Path) -> ChatStore:
    return ChatStore(namespace_for(_LOCAL), root=root)


def _labels(at: AppTest) -> list[str]:
    return [button.label for button in at.button]


# --- The top bar ------------------------------------------------------------------------------


def test_the_app_renders_without_raising(chat_root: Path) -> None:
    at = _app().run()
    assert not at.exception


def test_all_four_top_bar_controls_reach_the_page(chat_root: Path) -> None:
    # The row the task asked for: New chat, Log in, Sign up for free, and the key form behind
    # the "Use a key" popover (whose submit button is what shows up here).
    at = _app().run()
    labels = _labels(at)
    assert "🆕 New chat" in labels
    assert "Log in" in labels
    assert "Sign up for free" in labels
    assert "Use this key" in labels


def test_login_buttons_are_disabled_with_a_reason_when_no_provider_is_configured(
    chat_root: Path,
) -> None:
    # Disabled rather than hidden or live. Hidden, and the deployment looks like one that
    # simply has no accounts — indistinguishable from broken to anybody who was told to sign
    # in. Live, and pressing one raises.
    at = _app().run()
    for label in ("Log in", "Sign up for free"):
        button = next(b for b in at.button if b.label == label)
        assert button.disabled is True
        assert "not set up for this deployment" in (button.help or "")


def test_login_buttons_go_live_once_a_provider_is_configured(chat_root: Path) -> None:
    at = _app()
    at.secrets["auth"] = _FAKE_OIDC
    at.run()
    for label in ("Log in", "Sign up for free"):
        button = next(b for b in at.button if b.label == label)
        assert button.disabled is False, f"{label} should be live with a provider configured"


def test_use_a_key_stays_available_even_with_no_provider(chat_root: Path) -> None:
    # The bring-your-own-credential path never depends on accounts being configured; it is the
    # one thing a visitor on an unconfigured deployment can still do.
    at = _app().run()
    assert "Use this key" in _labels(at)


# --- Past chats -------------------------------------------------------------------------------


def test_past_chats_panel_is_in_the_sidebar_on_load(chat_root: Path) -> None:
    at = _app().run()
    assert [e.label for e in at.sidebar.get("expander")] == ["🕘 Past chats (0)"]


def test_saved_conversations_are_listed_newest_first(chat_root: Path) -> None:
    store = _store(chat_root)
    store.save(new_thread_id(), [{"role": "user", "content": "older question"}])
    store.save(new_thread_id(), [{"role": "user", "content": "newer question"}])

    at = _app().run()
    rows = [b.label for b in at.sidebar.button if b.key.startswith("open_chat_")]
    assert len(rows) == 2
    assert rows[0].startswith("newer question")


def test_deleting_one_conversation_takes_two_presses(chat_root: Path) -> None:
    # The confirmation is worth its friction here: every other control in the app is
    # reversible by moving it back, and this one removes a file.
    store = _store(chat_root)
    doomed = new_thread_id()
    store.save(doomed, [{"role": "user", "content": "delete me"}])

    at = _app().run()
    at.sidebar.button(key=f"del_{doomed}").click().run()
    assert store.list(), "the first press must only arm the delete, not perform it"

    at.sidebar.button(key=f"confirm_del_{doomed}").click().run()
    assert store.list() == [], "the second press should have deleted it"
    assert not at.exception


def test_arming_one_delete_leaves_its_neighbours_alone(chat_root: Path) -> None:
    # A shared "pending" flag would, on the rerun, arm the row immediately below the one just
    # pressed — which is the row a reader is most likely to press next.
    store = _store(chat_root)
    armed, neighbour = new_thread_id(), new_thread_id()
    store.save(armed, [{"role": "user", "content": "arm me"}])
    store.save(neighbour, [{"role": "user", "content": "leave me"}])

    at = _app().run()
    at.sidebar.button(key=f"del_{armed}").click().run()

    keys = {b.key for b in at.sidebar.button}
    assert f"confirm_del_{armed}" in keys
    assert f"confirm_del_{neighbour}" not in keys


def test_delete_all_clears_every_conversation(chat_root: Path) -> None:
    store = _store(chat_root)
    for index in range(3):
        store.save(new_thread_id(), [{"role": "user", "content": f"question {index}"}])

    at = _app().run()
    at.sidebar.button(key="del_all").click().run()
    assert store.list(), "the first press must only arm it"

    at.sidebar.button(key="confirm_del_all").click().run()
    assert store.list() == []
    assert not at.exception


def test_delete_all_can_be_cancelled(chat_root: Path) -> None:
    store = _store(chat_root)
    store.save(new_thread_id(), [{"role": "user", "content": "keep me"}])

    at = _app().run()
    at.sidebar.button(key="del_all").click().run()
    at.sidebar.button(key="cancel_del_all").click().run()
    assert len(store.list()) == 1


def test_opening_a_stored_conversation_puts_it_back_on_screen(chat_root: Path) -> None:
    store = _store(chat_root)
    thread_id = new_thread_id()
    store.save(thread_id, [{"role": "user", "content": "What is backpropagation?"}])

    at = _app().run()
    at.sidebar.button(key=f"open_chat_{thread_id}").click().run()
    assert at.session_state["chat_thread_id"] == thread_id
    assert at.session_state["history"][0]["content"] == "What is backpropagation?"


def test_new_chat_keeps_the_conversation_it_clears(chat_root: Path) -> None:
    # Nothing is deleted. The thread is written out first and then let go of, so what just
    # left the screen is the top row of Past chats — where somebody who pressed this by
    # accident will go looking for it.
    at = _app().run()
    at.session_state["history"] = [{"role": "user", "content": "a question worth keeping"}]
    at.run()

    at.button(key="top_new_chat").click().run()
    assert at.session_state["history"] == [], "the screen should be clear"
    titles = [meta.title for meta in _store(chat_root).list()]
    assert titles == ["a question worth keeping"]
