"""Shared test fixtures: a fake OpenAI-style client (no network, no API key).

The fakes duck-type the small slice of the OpenAI SDK the app uses:
``client.chat.completions.create(**kwargs)`` returning an object with
``choices[0].message.content``, ``choices[0].finish_reason``, ``usage``,
``id``, and ``model``.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)
    id: str = "gen-test-123"
    model: str = "openai/gpt-5-mini"


def make_response(content: str | None, finish_reason: str = "stop") -> FakeResponse:
    """Build a fake chat-completion response with the given text."""
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(content=content), finish_reason=finish_reason)]
    )


class FakeCompletions:
    """Records every create() call; replays queued responses or raises."""

    def __init__(self, responses: list[FakeResponse] | None = None, error: Exception | None = None):
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError("FakeCompletions: no queued responses left")
        return self.responses.pop(0)


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeClient:
    """Duck-typed stand-in for the OpenAI client."""

    def __init__(self, responses: list[FakeResponse] | None = None, error: Exception | None = None):
        self.completions = FakeCompletions(responses=responses, error=error)
        self.chat = FakeChat(self.completions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


@pytest.fixture
def fake_client_factory():
    """Factory fixture: build a FakeClient with queued replies or an error."""

    def _factory(
        *contents: str | None, finish_reason: str = "stop", error: Exception | None = None
    ) -> FakeClient:
        responses = [make_response(c, finish_reason) for c in contents]
        return FakeClient(responses=responses, error=error)

    return _factory
