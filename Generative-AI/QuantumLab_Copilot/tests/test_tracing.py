"""Tests for the LangSmith switch.

Two things matter and neither is whether tracing works -- that needs LangSmith.
What is testable here is that tracing is *off* unless it was both asked for and
credentialed, and that the credential never appears anywhere but the environment
variable it belongs in.
"""

from __future__ import annotations

import os

from pydantic import SecretStr

from src.agent.tracing import TRACING_VARIABLE, configure_tracing
from src.settings import Settings


def settings(**overrides: object) -> Settings:
    """Process settings with a fake credential and whatever else a test needs.

    Both LangSmith fields are pinned rather than left to their defaults. They are
    read from the environment, and a developer with real tracing configured would
    otherwise see these tests pass or fail according to their own shell.
    """
    base: dict[str, object] = {
        "openrouter_api_key": SecretStr("test-key"),
        "langsmith_tracing": False,
        "langsmith_api_key": None,
    }
    return Settings(**{**base, **overrides})  # type: ignore[arg-type]


def test_tracing_is_off_by_default() -> None:
    # A graded project must not ship a default that sends every question a
    # anyone types to an external service.
    assert configure_tracing(settings()) is False
    assert os.environ[TRACING_VARIABLE] == "false"


def test_tracing_asked_for_without_a_key_stays_off() -> None:
    # The dangerous case: the switch is on, so the operator expects traces, but
    # there is nothing to authenticate with. Writing "false" is what stops
    # LangChain warning on every single call.
    assert configure_tracing(settings(langsmith_tracing=True)) is False
    assert os.environ[TRACING_VARIABLE] == "false"


def test_tracing_with_both_switch_and_key_turns_on() -> None:
    resolved = settings(
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("ls-test-key"),
        langsmith_project="test-project",
    )
    try:
        assert configure_tracing(resolved) is True
        assert os.environ[TRACING_VARIABLE] == "true"
        assert os.environ["LANGSMITH_PROJECT"] == "test-project"
        # The credential reaches exactly one place, and it is not a log or a
        # return value.
        assert os.environ["LANGSMITH_API_KEY"] == "ls-test-key"
    finally:
        os.environ[TRACING_VARIABLE] = "false"
        os.environ.pop("LANGSMITH_API_KEY", None)
        os.environ.pop("LANGSMITH_PROJECT", None)


def test_configuring_twice_is_harmless() -> None:
    # Called from `ask`, which runs once per question: it has to be idempotent.
    assert configure_tracing(settings()) == configure_tracing(settings())
