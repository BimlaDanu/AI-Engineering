"""Offline tests for the optional login gate (:mod:`src.auth`).

The gate's decision logic — *is it configured?*, *does this config parse into a usable
AuthConfig?*, *is the display name escaped?* — is pure and injectable (``load_config`` takes a
``raw`` dict), so it is exercised here with no Streamlit runtime and without needing the
``streamlit_authenticator`` dependency (imported lazily only inside the live login path).

The guarantee that matters most: **with no config, the feature is off** — ``is_enabled()`` is
False and ``load_config`` returns None — so the gate can never accidentally lock out a local /
un-deployed app.
"""

from __future__ import annotations

import json

import pytest

from src.auth import (
    AuthConfig,
    KeyStatus,
    User,
    Visitor,
    _read_raw,
    _to_plain,
    check_openrouter_key,
    load_config,
    providers_in,
)

_VALID_RAW = {
    "cookie_name": "synapse_auth",
    "cookie_key": "a-very-long-random-signing-secret",
    "cookie_expiry_days": 14,
    "credentials": {
        "usernames": {
            "alice": {"name": "Alice Researcher", "password": "pw", "email": "a@example.com"},
        }
    },
}


def test_no_config_disables_the_gate() -> None:
    # The critical safety property: absent config → feature off, never a locked-out app.
    assert load_config(raw={}) is None
    assert load_config(raw=None if False else {}) is None  # explicit-empty is still "off"


def test_config_without_users_is_treated_as_off() -> None:
    assert load_config(raw={"cookie_key": "x", "credentials": {"usernames": {}}}) is None
    assert load_config(raw={"cookie_key": "x", "credentials": {}}) is None
    assert load_config(raw={"cookie_key": "x"}) is None


def test_valid_config_parses_into_authconfig() -> None:
    cfg = load_config(raw=_VALID_RAW)
    assert isinstance(cfg, AuthConfig)
    assert cfg.cookie_name == "synapse_auth"
    assert cfg.cookie_key == "a-very-long-random-signing-secret"
    assert cfg.cookie_expiry_days == 14
    assert "alice" in cfg.credentials["usernames"]


def test_config_defaults_fill_in_missing_cookie_fields() -> None:
    cfg = load_config(raw={"credentials": _VALID_RAW["credentials"]})
    assert cfg is not None
    assert cfg.cookie_name == "synapse_auth"  # default
    assert cfg.cookie_expiry_days == 30  # default
    # cookie_key intentionally NOT defaulted — an empty key must stay empty so the gate can
    # surface a visible "add a cookie_key" error rather than signing cookies with "".
    assert cfg.cookie_key == ""


def test_missing_cookie_key_still_parses_but_is_empty() -> None:
    raw = {k: v for k, v in _VALID_RAW.items() if k != "cookie_key"}
    cfg = load_config(raw=raw)
    assert cfg is not None and cfg.cookie_key == ""


def test_to_plain_deep_converts_nested_mappings() -> None:
    # Streamlit hands secrets back as attr-dicts; the library mutates them, so we must deep-copy
    # to plain dicts/lists. A mapping subclass should come out as a real dict, recursively.
    class AttrDict(dict):
        pass

    nested = AttrDict({"credentials": AttrDict({"usernames": AttrDict({"x": AttrDict()})})})
    plain = _to_plain(nested)
    assert type(plain) is dict
    assert type(plain["credentials"]["usernames"]["x"]) is dict


def test_read_raw_prefers_env_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNAPSE_AUTH_CONFIG", json.dumps(_VALID_RAW))
    cfg = load_config(raw=_read_raw())
    assert cfg is not None and "alice" in cfg.credentials["usernames"]


def test_read_raw_bad_env_json_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNAPSE_AUTH_CONFIG", "{not valid json")
    assert _read_raw() is None


def test_user_model_is_typed() -> None:
    user = User(username="alice", name="Alice Researcher")
    assert user.username == "alice" and user.name == "Alice Researcher"


# --- Native OpenID Connect ------------------------------------------------------------------
# ``providers_in`` is the pure half of ``oidc_providers``: the shape-matching, with no secrets
# file and no browser. Both layouts in Streamlit's own documentation are pinned here.


def test_single_provider_config_is_reached_with_a_bare_login() -> None:
    # Keys directly in [auth] mean one provider, and "" is the caller's cue to call
    # st.login() with no argument.
    assert providers_in({"client_id": "abc", "cookie_secret": "s"}) == ("",)


def test_multi_provider_config_lists_each_sub_table() -> None:
    section = {
        "redirect_uri": "http://localhost:8501/oauth2callback",
        "cookie_secret": "s",
        "google": {"client_id": "g"},
        "auth0": {"client_id": "a"},
    }
    assert providers_in(section) == ("google", "auth0")


def test_a_legacy_password_block_is_not_mistaken_for_a_provider() -> None:
    # The whole reason the password gate moved to [password_auth]: Streamlit reads an
    # unrecognised sub-table of [auth] as a provider name, so an old [auth.credentials] block
    # must report *no* providers rather than one called "credentials".
    assert providers_in(_VALID_RAW) == ()


def test_a_sub_table_without_a_client_id_is_not_a_provider() -> None:
    assert providers_in({"cookie_secret": "s", "notes": {"why": "because"}}) == ()


def test_missing_or_malformed_auth_section_means_not_configured() -> None:
    # Answered defensively: a checkout with no secrets is the normal local case, not an error.
    assert providers_in(None) == ()
    assert providers_in("nonsense") == ()
    assert providers_in({}) == ()


# --- The password gate now reads its own section ---------------------------------------------


def test_legacy_auth_section_still_parses_so_existing_installs_keep_working() -> None:
    # `load_config` takes the raw table either way; what changed is only where `_read_raw`
    # looks for it. A config that worked before must still parse.
    assert load_config(raw=_VALID_RAW) is not None


# --- Who is visiting --------------------------------------------------------------------------


def test_visitor_levels_report_whether_a_name_can_be_shown() -> None:
    # Drives the top bar: identified visitors get an account menu, everyone else gets the
    # Log in / Sign up pair.
    assert Visitor("signed_in", "Ada Lovelace").is_identified is True
    assert Visitor("password", "alice").is_identified is True
    assert Visitor("own_key").is_identified is False
    assert Visitor("open").is_identified is False


def test_the_greeting_is_the_most_specific_thing_known() -> None:
    assert Visitor("signed_in", "Ada Lovelace", "ada@example.com").greeting == "Ada"
    assert Visitor("signed_in", "", "ada@example.com").greeting == "ada@example.com"
    assert Visitor("own_key").greeting == "Your key"
    assert Visitor("open").greeting == "Guest"


# --- Checking a pasted credential --------------------------------------------------------------


def test_an_empty_or_misshapen_key_is_refused_without_a_network_call() -> None:
    assert check_openrouter_key("   ").usable is False
    verdict = check_openrouter_key("sk-proj-an-openai-key")
    assert verdict.usable is False
    assert "sk-or-" in verdict.reason


def test_a_key_the_provider_rejects_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 401

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response())
    verdict = check_openrouter_key("sk-or-v1-wrong")
    assert verdict.usable is False
    assert verdict.checked is True


def test_a_key_that_cannot_be_checked_is_accepted_and_said_to_be_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Refusing here would lock out a visitor whose key is fine whenever OpenRouter is briefly
    # unreachable. Accepted, but the UI is told not to claim it was verified.
    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("no network")

    monkeypatch.setattr("httpx.get", _boom)
    verdict = check_openrouter_key("sk-or-v1-probably-fine")
    assert verdict.usable is True
    assert verdict.checked is False


def test_a_good_key_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 200

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response())
    assert check_openrouter_key("sk-or-v1-good") == KeyStatus(
        True, True, "Key verified — your questions are billed to it."
    )


def test_a_rejected_key_never_appears_in_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    # The verdict is shown on screen; the credential must not travel with it.
    class _Response:
        status_code = 401

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response())
    secret = "sk-or-v1-0123456789abcdef"
    assert secret not in check_openrouter_key(secret).reason
