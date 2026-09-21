"""Tests for the check a pasted OpenRouter key goes through before it is stored.

The distinction being defended is the one the module exists for: *refused* and
*could not be checked* are different answers, and collapsing them either locks a
visitor out with a working key or accepts a broken one silently. Every test here
stubs the network -- the point is the decision, and a unit test that depends on a
gateway being up is a test that fails for reasons it is not about.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from src.agent import credential_check
from src.agent.credential_check import (
    DEFAULT_BASE_URL,
    MIN_KEY_CHARACTERS,
    check_key,
)
from src.settings import get_settings

GOOD = "sk-or-v1-" + "a" * 40


class Gateway:
    """A stand-in that always answers the same way and records what it was asked.

    A class rather than a lambda so that "was the network reached at all?" is a
    question the tests can put to it -- which is half of what is being checked
    here, since two of the three refusals are supposed to happen without it.
    """

    def __init__(self, status: int | None, label: str | None = None) -> None:
        self.status = status
        self.label = label
        self.asked: list[str] = []

    def __call__(self, key: str) -> tuple[int | None, str | None]:
        """Record the key it was handed and give the canned answer."""
        self.asked.append(key)
        return self.status, self.label


def gateway(
    monkeypatch: pytest.MonkeyPatch, status: int | None, label: str | None = None
) -> Gateway:
    """Put a stand-in gateway in place of the real probe and hand it back."""
    stub = Gateway(status, label)
    monkeypatch.setattr(credential_check, "_probe", stub)
    return stub


# --- what never reaches the network -----------------------------------------


def test_an_empty_paste_is_refused_without_asking_anybody(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = gateway(monkeypatch, 200)
    status = check_key("   ")
    assert not status.usable
    assert status.checked
    assert not asked.asked


def test_something_far_too_short_is_refused_without_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not a format rule: gateways change their prefixes, and a validator that knows
    # better than the gateway eventually rejects a working key. This catches the
    # paste that obviously went wrong and nothing else.
    asked = gateway(monkeypatch, 200)
    assert not check_key("a" * (MIN_KEY_CHARACTERS - 1)).usable
    assert not asked.asked


def test_a_key_with_a_space_in_it_is_a_bad_paste(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = gateway(monkeypatch, 200)
    assert not check_key(GOOD[:20] + " " + GOOD[20:]).usable
    assert not asked.asked


def test_surrounding_whitespace_is_an_artefact_and_not_part_of_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = gateway(monkeypatch, 200)
    assert check_key(f"  {GOOD}\n").usable
    assert asked.asked == [GOOD]


# --- what the gateway says ---------------------------------------------------


def test_a_key_the_gateway_recognises_is_accepted_and_known_to_be_good(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway(monkeypatch, 200, "my laptop")
    status = check_key(GOOD)
    assert status.usable
    assert status.checked
    assert "my laptop" in status.reason


def test_a_key_the_gateway_rejects_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway(monkeypatch, 401)
    status = check_key(GOOD)
    assert not status.usable
    assert status.checked


def test_unreachable_is_not_the_same_as_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole reason the module has three outcomes instead of two. A laptop
    # offline, a gateway having a bad minute and a proxy swallowing the request all
    # produce "no answer", and refusing there tells a visitor with a perfectly good
    # key that it is invalid, on the authority of a question nobody managed to ask.
    gateway(monkeypatch, None)
    status = check_key(GOOD)
    assert status.usable
    assert not status.checked


def test_an_answer_that_is_neither_yes_nor_no_is_accepted_unchecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway(monkeypatch, 500)
    status = check_key(GOOD)
    assert status.usable
    assert not status.checked


# --- what a verdict is allowed to contain ------------------------------------


def test_no_verdict_ever_quotes_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The sentence goes on screen and the branch goes into the log. Neither is a
    # place for somebody else's credential.
    for outcome in (200, 401, 500, None):
        gateway(monkeypatch, outcome)
        assert GOOD not in check_key(GOOD).reason


def test_nothing_is_logged_but_the_branch(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gateway(monkeypatch, 401)
    with caplog.at_level("INFO"):
        check_key(GOOD)
    assert GOOD not in caplog.text
    assert "key_check_refused" in caplog.text


# --- which gateway gets asked ------------------------------------------------


def test_the_question_goes_to_the_gateway_this_deployment_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not to openrouter.ai regardless. A deployment pointed at a compatible proxy
    # holds keys that proxy issued, so checking them against the default endpoint
    # would refuse every one -- and would send somebody else's credential to a host
    # this deployment never configured.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-configuration-only")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.example/v1/")
    get_settings.cache_clear()
    try:
        assert credential_check._gateway_url() == "https://proxy.example/v1"
    finally:
        get_settings.cache_clear()


def test_an_unreadable_configuration_falls_back_to_the_default_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `get_settings` raises when nothing is configured at all, which is exactly the
    # session most likely to be pasting a key. It must still be checkable.
    def unconfigured() -> object:
        raise RuntimeError("no credential configured")

    monkeypatch.setattr(credential_check, "get_settings", unconfigured)
    assert credential_check._gateway_url() == DEFAULT_BASE_URL


# --- the request itself ------------------------------------------------------
#
# `_probe` is stubbed out everywhere above, which is right for testing the
# decision and leaves untested the one function that actually puts a credential on
# the wire. These four replace `urlopen` instead, so the request is inspectable and
# nothing is sent.


class Response:
    """The shape of the thing `urlopen` returns, and nothing more of it."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> Response:
        """Enter the context `urlopen` is used as."""
        return self

    def __exit__(self, *unused: object) -> None:
        """Leave it, closing nothing."""

    def read(self) -> bytes:
        """The canned payload."""
        return self._body


def test_the_key_travels_in_the_header_and_the_url_is_the_key_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float = 0.0) -> Response:
        sent.append(request)
        return Response(200, b'{"data": {"label": "personal"}}')

    monkeypatch.setattr(credential_check.urllib.request, "urlopen", urlopen)
    assert credential_check._probe(GOOD) == (200, "personal")
    (request,) = sent
    assert request.full_url == f"{DEFAULT_BASE_URL}/key"
    assert request.get_method() == "GET"
    # In the header, never in the URL: a query string ends up in logs and proxies.
    assert request.get_header("Authorization") == f"Bearer {GOOD}"
    assert GOOD not in request.full_url


def test_a_refusal_comes_back_as_its_status(monkeypatch: pytest.MonkeyPatch) -> None:
    # `HTTPError` is raised rather than returned, so without this branch a 401
    # would read as "could not reach anybody" and a dead key would be accepted.
    def urlopen(request: urllib.request.Request, timeout: float = 0.0) -> Response:
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(credential_check.urllib.request, "urlopen", urlopen)
    assert credential_check._probe(GOOD) == (401, None)
    assert not check_key(GOOD).usable


def test_an_unreachable_network_is_no_answer_rather_than_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole point of the module: a laptop with no network must not be told its
    # perfectly good key is invalid.
    def urlopen(request: urllib.request.Request, timeout: float = 0.0) -> Response:
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(credential_check.urllib.request, "urlopen", urlopen)
    assert credential_check._probe(GOOD) == (None, None)
    verdict = check_key(GOOD)
    assert verdict.usable
    assert not verdict.checked


def test_a_reply_that_is_not_the_expected_shape_still_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A proxy in the way may answer 200 with something else entirely. The status is
    # what the decision turns on; the label is decoration and its absence is not an
    # error.
    def urlopen(request: urllib.request.Request, timeout: float = 0.0) -> Response:
        return Response(200, b"not json at all")

    monkeypatch.setattr(credential_check.urllib.request, "urlopen", urlopen)
    assert credential_check._probe(GOOD) == (None, None)
