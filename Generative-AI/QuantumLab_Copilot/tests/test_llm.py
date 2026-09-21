"""Tests for the model factory and the middleware stack.

No test here reaches the network. Constructing a chat model does not call it,
which is what makes the wiring -- throttle, retry classification, call limit,
logging -- testable offline. The one thing that cannot be tested without a live
key is that the model slug exists at the provider; that belongs to the
evaluation run, not to a unit test.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelResponse,
    ModelRetryMiddleware,
)
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from src.agent.llm import (
    TRANSIENT_STATUS_CODES,
    LoggingMiddleware,
    ask_prose,
    ask_structured,
    build_chat_model,
    build_middleware,
    build_rate_limiter,
    is_transient,
    reasoning_effort_for,
)
from src.logging_setup import LOGGER_NAME, configure_logging
from src.settings import Settings


def build(**overrides: object) -> Settings:
    """Construct settings from explicit values, ignoring any ``.env``."""
    values: dict[str, object] = {"openrouter_api_key": "test-key-not-a-real-one"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


class ProviderError(Exception):
    """An error carrying an HTTP status, as the provider's exceptions do."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


# --------------------------------------------------------------------------
# Which failures are worth retrying
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", sorted(TRANSIENT_STATUS_CODES))
def test_a_transient_status_is_retried(status: int) -> None:
    assert is_transient(ProviderError(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_permanent_status_is_not_retried(status: int) -> None:
    # A wrong key is wrong on every attempt. Retrying it delays the one message
    # that would have told the user what to fix.
    assert not is_transient(ProviderError(status))


def test_a_rate_limit_is_retried() -> None:
    assert is_transient(ProviderError(429))


def test_a_connection_failure_without_a_status_is_retried_by_name() -> None:
    # Matched by name so that this module need not import openai or httpx,
    # neither of which pyproject.toml declares.
    error = type("APIConnectionError", (Exception,), {})()
    assert is_transient(error)


def test_a_subclass_of_a_transient_error_is_retried() -> None:
    base = type("APITimeoutError", (Exception,), {})
    assert is_transient(type("ProviderTimeout", (base,), {})())


def test_an_unrecognised_error_is_not_retried() -> None:
    # Retrying an unknown failure is just running a bug repeatedly.
    assert not is_transient(ValueError("something else entirely"))


# --------------------------------------------------------------------------
# Throttling
# --------------------------------------------------------------------------


def test_a_throttle_is_installed_by_default() -> None:
    # The value is read off the setting rather than repeated: what this defends is
    # that a throttle exists at all, and pinning the number here would make tuning
    # it for latency look like a broken test.
    limiter = build_rate_limiter(build())
    assert limiter is not None
    assert limiter.requests_per_second == build().requests_per_second


def test_throttling_can_be_switched_off() -> None:
    assert build_rate_limiter(build(requests_per_second=0.0)) is None


def test_every_model_shares_one_budget_at_a_given_rate() -> None:
    # The bug this defends against, which shipped: a client is built per model
    # *call* -- nine modules ask for one, about seven times per question -- and each
    # used to get a limiter of its own. A fresh limiter starts with a full bucket,
    # so every call took the first token of a private budget and nothing ever
    # waited. Identity is the assertion because a budget that is copied is not a
    # budget.
    assert build_rate_limiter(build()) is build_rate_limiter(build())


def test_a_different_rate_is_a_different_budget() -> None:
    fast = build_rate_limiter(build(requests_per_second=8.0))
    slow = build_rate_limiter(build(requests_per_second=2.0))
    assert fast is not slow
    assert fast is not None and slow is not None
    assert fast.requests_per_second == 8.0


def test_the_throttle_allows_a_short_burst() -> None:
    # An interactive question should not wait for a token after an idle period.
    limiter = build_rate_limiter(build(requests_per_second=4.0))
    assert limiter is not None
    assert limiter.max_bucket_size >= 1.0


# --------------------------------------------------------------------------
# The model itself
# --------------------------------------------------------------------------


def test_the_model_is_built_from_settings_without_calling_out() -> None:
    model = build_chat_model(build(chat_model="anthropic/claude-haiku-4.5"))
    assert model.model_name == "anthropic/claude-haiku-4.5"  # type: ignore[attr-defined]


def test_a_reasoning_model_is_asked_to_think_briefly() -> None:
    # Measured: without this, one routing decision cost 14.6 seconds and 669 output
    # tokens to fill four short fields. With it, 3.3 seconds and the same fields.
    assert reasoning_effort_for(build(chat_model="openai/gpt-5-mini")) == "minimal"


def test_a_model_without_the_parameter_is_never_sent_it() -> None:
    # Sending it to this model fails the call, and a failed structured call is
    # indistinguishable from having no model at all -- the agent would quietly fall
    # back to its heuristics and still look like it was working.
    assert reasoning_effort_for(build(chat_model="anthropic/claude-haiku-4.5")) is None


def test_the_reasoning_parameter_can_be_switched_off() -> None:
    assert reasoning_effort_for(build(chat_model="openai/gpt-5-mini", reasoning_effort="")) is None


def test_the_effort_reaches_the_model() -> None:
    model = build_chat_model(build(chat_model="openai/gpt-5-mini"))
    assert model.reasoning_effort == "minimal"  # type: ignore[attr-defined]


def test_client_side_retries_are_disabled_so_they_cannot_multiply() -> None:
    # Three client attempts inside three middleware attempts is nine calls
    # against a provider that is already refusing them.
    assert build_chat_model(build()).max_retries == 0  # type: ignore[attr-defined]


def test_the_credential_is_not_exposed_on_the_model() -> None:
    model = build_chat_model(build(openrouter_api_key="super-secret-value"))
    assert "super-secret-value" not in repr(model)


def test_the_timeout_and_temperature_come_from_settings() -> None:
    model = build_chat_model(build(temperature=0.3, request_timeout_s=12.0))
    assert model.temperature == 0.3  # type: ignore[attr-defined]
    assert model.request_timeout == 12.0  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# The middleware stack
# --------------------------------------------------------------------------


def test_the_stack_caps_the_budget_outside_the_retries() -> None:
    # Order is the contract: retries must count against the ceiling, or a retry
    # storm inside an unbounded loop escapes the budget entirely.
    stack = build_middleware(build())
    assert isinstance(stack[0], ModelCallLimitMiddleware)
    assert isinstance(stack[1], ModelRetryMiddleware)
    assert isinstance(stack[2], LoggingMiddleware)


def test_the_retry_policy_is_taken_from_settings() -> None:
    retry = build_middleware(build(max_retries=5))[1]
    assert isinstance(retry, ModelRetryMiddleware)
    assert retry.max_retries == 5
    assert retry.jitter is True
    assert retry.retry_on is is_transient


def test_retries_can_be_switched_off_entirely() -> None:
    retry = build_middleware(build(max_retries=0))[1]
    assert isinstance(retry, ModelRetryMiddleware)
    assert retry.max_retries == 0


# --------------------------------------------------------------------------
# Logging around each call
# --------------------------------------------------------------------------


class Model:
    """The minimum a middleware needs from a chat model."""

    model_name = "test/model"


class Request:
    """A stand-in for a ``ModelRequest``.

    Faked because the real one is a dataclass of nine fields, of which the
    middleware reads exactly one. The *response* is the real type -- that is the
    side the assertions are about.
    """

    model = Model()


def response() -> ModelResponse[Any]:
    """Build a model response carrying token usage and a finish reason."""
    message = AIMessage(
        content="an answer",
        usage_metadata={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
        response_metadata={"finish_reason": "stop"},
    )
    return ModelResponse(result=[message])


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    """Capture the project's log output for the duration of a test."""
    buffer = io.StringIO()
    configure_logging("DEBUG", stream=buffer)
    yield buffer
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def only_entry(stream: io.StringIO) -> dict[str, Any]:
    """Parse the single log line the test produced."""
    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 1
    return dict(json.loads(lines[0]))


def test_a_call_is_logged_with_its_model_tokens_and_finish_reason(stream: io.StringIO) -> None:
    middleware = LoggingMiddleware(purpose="answer")
    middleware.wrap_model_call(Request(), lambda request: response())  # type: ignore[arg-type]
    entry = only_entry(stream)
    assert entry["model"] == "test/model"
    assert entry["purpose"] == "answer"
    assert entry["total_tokens"] == 16
    assert entry["finish_reason"] == "stop"
    assert entry["outcome"] == "ok"


def test_the_response_is_passed_through_unchanged(stream: io.StringIO) -> None:
    # Observability must not edit the conversation.
    original = response()
    middleware = LoggingMiddleware()
    returned = middleware.wrap_model_call(Request(), lambda request: original)  # type: ignore[arg-type]
    assert returned is original


def test_a_failed_call_is_logged_and_still_raises(stream: io.StringIO) -> None:
    def fail(request: object) -> ModelResponse[Any]:
        raise ProviderError(429)

    with pytest.raises(ProviderError):
        LoggingMiddleware().wrap_model_call(Request(), fail)  # type: ignore[arg-type]
    entry = only_entry(stream)
    assert entry["outcome"] == "error"
    assert entry["error_type"] == "ProviderError"


def test_each_retry_attempt_is_logged_separately(stream: io.StringIO) -> None:
    # A retried call is two calls, both paid for. Logging it as one slow call
    # would hide the cost.
    middleware = LoggingMiddleware()
    for _ in range(3):
        middleware.wrap_model_call(Request(), lambda request: response())  # type: ignore[arg-type]
    assert len([line for line in stream.getvalue().splitlines() if line]) == 3


# --------------------------------------------------------------------------
# A call that produced nothing usable says so
# --------------------------------------------------------------------------


class Shape(BaseModel):
    """The smallest structured reply a test can ask for."""

    verdict: str


def entries(stream: io.StringIO) -> list[dict[str, Any]]:
    """Parse every log line the test produced."""
    return [dict(json.loads(line)) for line in stream.getvalue().splitlines() if line]


class Unstructured:
    """A model whose structured call comes back as the wrong type."""

    model_name = "test/model"

    def with_structured_output(self, schema: object, include_raw: bool = False) -> Unstructured:
        """Return itself, so the call below is the one under test."""
        return self

    def invoke(self, messages: object) -> dict[str, object]:
        """Answer successfully, with nothing that fits the schema."""
        return {"parsed": None, "raw": None}


class Broken:
    """A model whose call raises, the way a gateway outage arrives."""

    model_name = "test/model"

    def with_structured_output(self, schema: object, include_raw: bool = False) -> Broken:
        """Return itself, so the call below is the one under test."""
        return self

    def invoke(self, messages: object) -> dict[str, object]:
        """Fail the way a gateway outage does."""
        raise RuntimeError("gateway said no")


class Silent:
    """A model that answers, successfully, with nothing."""

    model_name = "test/model"

    def invoke(self, messages: object) -> AIMessage:
        """Answer with whitespace, which is a success as far as the client knows."""
        return AIMessage(content="   ")


def test_a_reply_that_does_not_fit_the_schema_is_reported(stream: io.StringIO) -> None:
    # The call succeeded and was paid for, and the caller still has to fall back.
    # Without this line that is a normal `llm_call` followed by an unexplained
    # deterministic answer, which is indistinguishable from having no key at all.
    assert ask_structured(Unstructured(), Shape, "sys", "text", "route") is None  # type: ignore[arg-type]
    unusable = [e for e in entries(stream) if e["event"] == "structured_call_unusable"]
    assert len(unusable) == 1
    assert unusable[0]["reason"] == "reply_did_not_fit_schema"
    assert unusable[0]["purpose"] == "route"
    assert unusable[0]["schema"] == "Shape"


def test_a_failed_structured_call_is_reported_as_well_as_logged(stream: io.StringIO) -> None:
    assert ask_structured(Broken(), Shape, "sys", "text", "decide") is None  # type: ignore[arg-type]
    found = {e["event"]: e for e in entries(stream)}
    assert found["llm_call"]["outcome"] == "error"
    assert found["structured_call_unusable"]["reason"] == "call_failed"
    assert found["structured_call_unusable"]["detail"] == "RuntimeError"


def test_an_empty_prose_reply_is_reported(stream: io.StringIO) -> None:
    # The quietest failure a model has: it answers, the usage is billed, and the
    # answer is whitespace.
    assert ask_prose(Silent(), "sys", "text", "compose") is None  # type: ignore[arg-type]
    unusable = [e for e in entries(stream) if e["event"] == "structured_call_unusable"]
    assert len(unusable) == 1
    assert unusable[0]["reason"] == "reply_was_empty"
    assert unusable[0]["schema"] == "prose"
