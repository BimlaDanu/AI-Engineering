"""Tests for the identity layer.

The distinction being defended: only a login counts as an identity, and there is no
name box that could look like one. What an unsigned session gets instead is a key of
its own -- unique, so no two visitors share a history, and unwritten to disk, so
nothing outlives the tab.

The passphrase gate is tested for what it does when nothing is configured, since
that is the local case and it must stay frictionless.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.agent.credential_check import KeyStatus
from src.ui import identity as identity_module
from src.ui.identity import (
    OWN_KEY_STATE,
    PASSPHRASE_VARIABLE,
    Identity,
    configured_passphrase,
    matches,
)

UI = Path(__file__).resolve().parents[1] / "src" / "ui"
APP = str(UI / "app.py")


# --- what an identity is worth ----------------------------------------------


def test_an_unsigned_session_is_not_verified() -> None:
    # There is no way to construct a verified identity except from a provider, which
    # is what removing the name box bought: nothing a user can type reaches this.
    assert not Identity().verified
    assert "Not signed in" in Identity().explain()


def test_the_sidebar_offers_no_name_box() -> None:
    # The requirement this file exists to pin: a login when deployed, and no text
    # box asking who you are.
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception
    labels = [str(box.label).lower() for box in app.text_input]
    assert not any("name" in label for label in labels), labels


def test_a_platform_identity_is_verified() -> None:
    assert Identity("Ada", "platform").verified
    assert Identity("Ada", "platform").explain() == "Signed in as Ada."


def test_an_anonymous_session_still_has_something_to_call_you() -> None:
    assert Identity().display == "there"
    assert "this session only" in Identity().explain()


def test_a_thread_key_is_a_digest_rather_than_a_name() -> None:
    # This ends up in a log field and a checkpointer key. A name is personal data;
    # an identifier is all either needs.
    thread = Identity("Ada Lovelace", "platform").thread
    assert "Ada" not in thread
    assert thread.isalnum()


def test_the_same_name_always_gets_the_same_thread() -> None:
    assert Identity("Ada", "platform").thread == Identity("Ada", "platform").thread
    assert Identity("Ada", "platform").thread != Identity("Grace", "platform").thread


def test_an_anonymous_identity_has_no_digest_to_key() -> None:
    # And it is emphatically not what memory is keyed by: see `session_thread`,
    # which mints a per-session token so that two unnamed visitors never share one
    # history.
    assert Identity().thread == "anonymous"


def test_an_unsigned_session_gets_a_key_of_its_own() -> None:
    # Read off the real page, because the key is minted into session state while it
    # renders. The constant "anonymous" would be a shared history; a token is not.
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    thread = app.session_state.filtered_state.get("session_thread")
    assert isinstance(thread, str)
    assert thread and thread != "anonymous"


# --- the passphrase --------------------------------------------------------


def test_no_passphrase_is_configured_in_the_test_environment() -> None:
    # Asserted, because the gate test below depends on it: a developer who
    # exported the variable should see this fail rather than the page test.
    assert not os.environ.get(PASSPHRASE_VARIABLE)
    assert configured_passphrase() is None


def test_a_blank_passphrase_counts_as_unset(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Clearing a value in a deployment's configuration means "no gate", not "the
    # empty password".
    monkeypatch.setenv(PASSPHRASE_VARIABLE, "   ")
    assert configured_passphrase() is None


def test_a_configured_passphrase_is_read(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(PASSPHRASE_VARIABLE, "open sesame")
    assert configured_passphrase() == "open sesame"


def test_matching_is_exact() -> None:
    assert matches("open sesame", "open sesame")
    assert matches("  open sesame  ", "open sesame")  # a pasted value keeps working
    assert not matches("open sesam", "open sesame")
    assert not matches("", "open sesame")


# --- the gate, through the real page ---------------------------------------


def test_the_page_is_open_when_no_passphrase_is_configured() -> None:
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception
    assert any("QuantumLab" in str(title.value) for title in app.title)


def test_a_configured_passphrase_hides_the_application(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A gate that renders the application behind it has not gated anything.
    monkeypatch.setenv(PASSPHRASE_VARIABLE, "open sesame")
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception
    assert app.title == []
    assert any("private" in str(block.value) for block in app.markdown)


def test_the_right_passphrase_opens_it(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(PASSPHRASE_VARIABLE, "open sesame")
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    app.text_input[0].set_value("open sesame")
    app.button[0].click().run()
    assert not app.exception
    assert any("QuantumLab" in str(title.value) for title in app.title)


def test_the_wrong_passphrase_does_not(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(PASSPHRASE_VARIABLE, "open sesame")
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    app.text_input[0].set_value("guess")
    app.button[0].click().run()
    assert not app.exception
    assert app.title == []
    assert any("not the passphrase" in str(error.value) for error in app.error)


def test_the_passphrase_is_never_put_into_session_state(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Session state is readable by anything else drawing the page, and it does not
    # need the passphrase -- only the fact that one matched.
    monkeypatch.setenv(PASSPHRASE_VARIABLE, "open sesame")
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    app.text_input[0].set_value("open sesame")
    app.button[0].click().run()
    stored = app.session_state.filtered_state
    assert stored["access_granted"] is True
    assert "open sesame" not in [str(value) for value in stored.values()]


# --- the top bar: one row, one account surface -------------------------------


def top_bar_labels(app: AppTest) -> list[str]:
    """The labels of the buttons drawn in the page body, in the order drawn."""
    return [str(button.label) for button in app.button]


def test_the_top_bar_carries_the_account_controls_in_one_row() -> None:
    # The requirement: New chat, Log in and Sign up for free on one line at the top
    # of the window, with the key form behind "Use a key". Order matters -- the row
    # is read left to right and the control that acts on this page comes first.
    app = AppTest.from_file(APP, default_timeout=90)
    app.run()
    assert not app.exception
    labels = top_bar_labels(app)
    for expected in ("🆕 New chat", "Log in", "Sign up for free"):
        assert expected in labels, labels
    assert labels.index("🆕 New chat") < labels.index("Log in") < labels.index("Sign up for free")
    # The popover's own label is not a button; what is inside it is.
    assert "Use this key" in labels, labels


def test_the_account_surface_is_not_also_in_the_sidebar() -> None:
    # It was, and that was the defect: two places stating one fact, which is two
    # places for it to disagree. The sidebar reports the session -- past chats,
    # memory, cost -- and says nothing about the account.
    app = AppTest.from_file(APP, default_timeout=90)
    app.run()
    assert not app.exception
    sidebar = [str(button.label).lower() for button in app.sidebar.button]
    assert not any("sign in" in label or "log in" in label for label in sidebar), sidebar


def test_the_sign_in_buttons_are_disabled_rather_than_hidden_without_a_provider() -> None:
    # Hidden, and a deployment looks like one that simply has no accounts, which is
    # indistinguishable from broken to anybody who was told to sign in. The test
    # environment configures no provider, so this is that case.
    app = AppTest.from_file(APP, default_timeout=90)
    app.run()
    assert not app.exception
    for label in ("Log in", "Sign up for free"):
        button = next(entry for entry in app.button if str(entry.label) == label)
        assert button.disabled, label
    # And the reason is not left to be guessed at.
    assert "key" in str(button.help).lower()


def test_new_chat_is_offered_exactly_once() -> None:
    # It moved from a column on the chat page into the top bar, and the failure
    # mode of a move like that is an addition: the page keeps drawing its own and
    # the reader gets two buttons that do the same thing at two different heights.
    #
    # That it is offered on the chat page and *only* there is decided in the entry
    # point, which passes it as the row's lead when the page about to run is the
    # chat page. That branch is not reachable from here -- `AppTest.switch_page`
    # runs the target page without the entry script that draws the bar -- so what
    # is pinned here is the half this harness can see.
    app = AppTest.from_file(APP, default_timeout=90)
    app.run()
    assert not app.exception
    assert top_bar_labels(app).count("🆕 New chat") == 1


# --- a key of one's own ------------------------------------------------------


def test_no_key_pasted_means_the_host_configuration_is_used_unchanged() -> None:
    # `None` is what `graph.ask` reads as "resolve the process settings yourself",
    # so the ordinary case costs nothing and changes nothing.
    from src.ui import panels

    assert panels.session_settings() is None


def test_a_pasted_key_replaces_the_credential_and_nothing_else(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from src.ui import identity as identity_module
    from src.ui import panels

    monkeypatch.setattr(identity_module, "own_key", lambda: "pasted-key-value")
    settings = panels.session_settings()
    assert settings is not None
    assert settings.openrouter_api_key.get_secret_value() == "pasted-key-value"
    # The dials the deployment chose are still the deployment's.
    assert settings.chat_model
    assert settings.requests_per_second >= 0.0


def test_a_pasted_key_is_not_printable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # It ends up on a settings object that gets repr'd into logs and tracebacks.
    # `SecretStr` is what keeps it out of them, and this is the test that says so.
    from src.ui import identity as identity_module
    from src.ui import panels

    monkeypatch.setattr(identity_module, "own_key", lambda: "pasted-key-value")
    settings = panels.session_settings()
    assert settings is not None
    assert "pasted-key-value" not in repr(settings)


def test_a_pasted_key_collapses_the_row_into_one_account_menu() -> None:
    # Once somebody is running on their own key, their account is the least
    # interesting thing on the page: two sign-in buttons offering a flow they have
    # already been round is noise, so the row becomes a name and a popover.
    app = AppTest.from_file(APP, default_timeout=90)
    app.session_state[OWN_KEY_STATE] = "sk-or-v1-" + "a" * 40
    app.run()
    assert not app.exception
    labels = top_bar_labels(app)
    assert "Log in" not in labels, labels
    assert "Sign up for free" not in labels, labels
    # And taking the key back is a button rather than a promise about closing tabs.
    assert "Forget my key" in labels, labels


def paste_key(monkeypatch: pytest.MonkeyPatch, verdict: KeyStatus) -> AppTest:
    """Open the app, paste a key into the top bar and submit it.

    The gateway is never asked: ``check_key`` is replaced with the verdict the test
    is about, which is what makes these runnable with no network and no credential.
    """
    monkeypatch.setattr(identity_module, "check_key", lambda typed: verdict)
    app = AppTest.from_file(APP, default_timeout=90)
    app.run()
    box = next(entry for entry in app.text_input if "OpenRouter" in str(entry.label))
    box.set_value("sk-or-v1-" + "a" * 40)
    next(entry for entry in app.button if str(entry.label) == "Use this key").click().run()
    return app


def test_an_accepted_key_is_stored_and_the_visitor_is_told(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The verdict is drawn on the run *after* the one that produced it, because
    # storing a key reruns. Reported only from inside the form, it was drawn on a
    # run that was thrown away and the visitor saw nothing at all.
    app = paste_key(monkeypatch, KeyStatus(True, True, "Key accepted. Answers now run on it."))
    assert not app.exception
    assert app.session_state[OWN_KEY_STATE].startswith("sk-or-v1-")
    assert any("Key accepted" in str(note.value) for note in app.success), "no confirmation"


def test_a_key_that_could_not_be_checked_is_taken_but_said_to_be_unchecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The honest middle case, and the one that most needs saying out loud: an
    # offline laptop must not call a good key invalid, but a key nobody verified
    # must not be reported as verified either. Every number in an answer is
    # cross-checked before a model is consulted, so a key that silently fails
    # produces a correct answer written by the fallback -- nothing errors, and the
    # only warning a visitor will ever get is this one.
    app = paste_key(monkeypatch, KeyStatus(True, False, "Could not reach OpenRouter."))
    assert not app.exception
    assert OWN_KEY_STATE in app.session_state, "an unreachable gateway must not refuse the key"
    assert app.warning, "accepted-unchecked was reported as though it had been checked"
    assert not app.success


def test_a_refused_key_is_not_stored_and_the_error_is_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = paste_key(monkeypatch, KeyStatus(False, True, "OpenRouter rejected that key."))
    assert not app.exception
    assert OWN_KEY_STATE not in app.session_state
    assert any("rejected" in str(note.value) for note in app.error)


def test_the_verdict_on_a_key_is_shown_once_and_not_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # It is popped, not read. Left in session state it would surface on some later
    # run beside a control that has nothing to do with it -- "Key accepted" next to
    # an empty form being the worst of them, since by then it might not be.
    app = paste_key(monkeypatch, KeyStatus(True, True, "Key accepted. Answers now run on it."))
    assert app.success
    app.run()
    assert not app.success, "the verdict came back on a later run"


def test_a_key_can_be_taken_back(monkeypatch: pytest.MonkeyPatch) -> None:
    # A button rather than a promise about closing tabs: somebody on a shared
    # machine should be able to remove their credential without trusting the tab.
    app = paste_key(monkeypatch, KeyStatus(True, True, "Key accepted."))
    next(entry for entry in app.button if str(entry.label) == "Forget my key").click().run()
    assert not app.exception
    assert OWN_KEY_STATE not in app.session_state
    # And the row goes back to offering the two flows it offered before.
    assert "Log in" in top_bar_labels(app)
