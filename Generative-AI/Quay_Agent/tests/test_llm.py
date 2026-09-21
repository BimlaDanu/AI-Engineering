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
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelResponse,
    ModelRetryMiddleware,
)
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool
from pydantic import BaseModel

from src.agent.llm import (
    MAX_TOOL_RESULT_CHARACTERS,
    TRANSIENT_STATUS_CODES,
    LoggingMiddleware,
    ask_prose,
    ask_structured,
    ask_with_tools,
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


# --------------------------------------------------------------------------
# The tool-calling loop
# --------------------------------------------------------------------------


class Recorder:
    """A chat model that asks for a scripted sequence of tool calls.

    Written here rather than mocked, because what is under test is the *loop*: how
    many times the model is called, what it is shown between calls, and what it is
    let get away with. A mock that returns a canned Consultation would assert
    nothing about any of that.

    Attributes:
        script: One entry per round, each either a list of tool calls to request or
            ``None`` to stop and reply.
        seen: The message list as it stood at each call, so a test can check that
            tool results were actually fed back.
    """

    model_name = "test/model"

    def __init__(self, script: list[list[dict[str, Any]] | None]) -> None:
        """Take the script of rounds this model will play out."""
        self.script = script
        self.seen: list[int] = []
        self.bound: list[Any] | None = None

    def bind_tools(self, tools: list[Any]) -> Recorder:
        """Record what was offered and return itself, so invoke is the call tested."""
        self.bound = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Play the next round of the script."""
        self.seen.append(len(messages))
        wanted = self.script.pop(0) if self.script else None
        if wanted is None:
            return AIMessage(content="I looked it up.")
        return AIMessage(content="", tool_calls=wanted)


@tool(parse_docstring=True)
def double(value: int) -> dict[str, int]:
    """Double a number.

    Args:
        value: The number to double.

    Returns:
        The doubled value.
    """
    return {"doubled": value * 2}


@tool(parse_docstring=True)
def explode(value: int) -> dict[str, int]:
    """Raise, so the loop's failure handling can be tested.

    Args:
        value: Ignored.

    Returns:
        Nothing; this always raises.
    """
    raise RuntimeError("this tool is broken")


def test_a_model_that_asks_for_no_tool_costs_one_round() -> None:
    model = Recorder([None])
    consultation = ask_with_tools(model, "system", "text", [double], "tool_consultation")  # type: ignore[arg-type]
    assert consultation is not None
    assert consultation.calls == ()
    assert consultation.rounds == 1
    assert consultation.reply == "I looked it up."
    assert not consultation.used_anything
    # Calling nothing is a legitimate outcome, and the tools were still offered.
    assert model.bound == [double]


def test_a_tool_result_is_recorded_and_fed_back_before_the_next_round() -> None:
    model = Recorder([[{"name": "double", "args": {"value": 21}, "id": "c1"}], None])
    consultation = ask_with_tools(model, "system", "text", [double], "tool_consultation")  # type: ignore[arg-type]
    assert consultation is not None
    assert consultation.rounds == 2
    assert len(consultation.calls) == 1
    call = consultation.calls[0]
    assert call.name == "double"
    assert call.arguments == {"value": 21}
    assert "42" in call.result
    assert not call.failed
    assert consultation.used_anything
    # The second call saw three more messages than the first: the model's own reply
    # and the tool's answer. Without that the model cannot read its own result and
    # the loop degenerates into asking the same thing repeatedly.
    assert model.seen[1] > model.seen[0]


def test_the_round_limit_stops_a_model_that_never_stops_asking() -> None:
    forever: list[list[dict[str, Any]] | None] = [
        [{"name": "double", "args": {"value": 1}, "id": f"c{index}"}] for index in range(20)
    ]
    consultation = ask_with_tools(
        Recorder(forever),  # type: ignore[arg-type]
        "system",
        "text",
        [double],
        "tool_consultation",
        max_rounds=3,
    )
    assert consultation is not None
    assert consultation.rounds == 3
    assert len(consultation.calls) == 3
    assert "3-round limit" in consultation.stopped_because


def test_a_tool_that_raises_is_reported_to_the_model_rather_than_killing_the_run() -> None:
    model = Recorder([[{"name": "explode", "args": {"value": 1}, "id": "c1"}], None])
    consultation = ask_with_tools(model, "system", "text", [explode], "tool_consultation")  # type: ignore[arg-type]
    assert consultation is not None
    assert consultation.calls[0].failed
    assert "RuntimeError" in consultation.calls[0].result
    # Nothing usable came back, which is what the writing call reads to decide
    # whether to mention the consultation at all.
    assert not consultation.used_anything


def test_a_tool_the_model_invented_is_answered_with_the_list_of_real_ones() -> None:
    model = Recorder([[{"name": "solve_it_exactly", "args": {}, "id": "c1"}], None])
    consultation = ask_with_tools(model, "system", "text", [double], "tool_consultation")  # type: ignore[arg-type]
    assert consultation is not None
    assert consultation.calls[0].failed
    # Naming the real tools is what turns a wasted round into a recoverable one.
    assert "double" in consultation.calls[0].result


def test_a_long_tool_result_is_clipped_before_it_reaches_the_model() -> None:
    @tool(parse_docstring=True)
    def verbose(value: int) -> str:
        """Return far more text than any answer needs.

        Args:
            value: Ignored.

        Returns:
            A very long string.
        """
        return "x" * (MAX_TOOL_RESULT_CHARACTERS * 2)

    model = Recorder([[{"name": "verbose", "args": {"value": 1}, "id": "c1"}], None])
    consultation = ask_with_tools(model, "system", "text", [verbose], "tool_consultation")  # type: ignore[arg-type]
    assert consultation is not None
    result = consultation.calls[0].result
    assert len(result) <= MAX_TOOL_RESULT_CHARACTERS + 40
    assert result.endswith("[result clipped]")


def test_a_model_that_cannot_call_tools_returns_nothing_rather_than_raising() -> None:
    class Plain:
        """A model with no bind_tools, which is what an older gateway looks like."""

        model_name = "test/model"

    assert ask_with_tools(Plain(), "system", "text", [double], "tool_consultation") is None  # type: ignore[arg-type]


def test_a_failure_after_some_tools_ran_keeps_what_they_returned() -> None:
    class Flaky(Recorder):
        """Answers one round, then fails."""

        def invoke(self, messages: list[Any]) -> AIMessage:
            """Ask for one tool, then raise on the next call."""
            if self.seen:
                raise RuntimeError("the provider dropped the connection")
            return super().invoke(messages)

    model = Flaky([[{"name": "double", "args": {"value": 4}, "id": "c1"}]])
    consultation = ask_with_tools(model, "system", "text", [double], "tool_consultation")  # type: ignore[arg-type]
    # Not None: the tool result is real, and discarding it because the summary call
    # failed would throw away work that was already paid for.
    assert consultation is not None
    assert len(consultation.calls) == 1
    assert "8" in consultation.calls[0].result
    assert consultation.reply == ""


# --------------------------------------------------------------------------
# Streaming the one call whose output is the answer
# --------------------------------------------------------------------------


class Piecewise:
    """A model that streams a reply in pieces, and reports usage on the last one.

    The usage placement is the point. A real provider reports token counts on the
    final chunk only, so a reply rebuilt by joining the text loses them -- and the
    longest, dearest call in the graph would be billed at nothing. This double
    reproduces that arrangement so the accounting can be held to it.

    Attributes:
        pieces: The text, in the order it is streamed.
        invoked: Whether anybody called ``invoke`` instead, which would mean the
            streaming path was silently skipped.
    """

    model_name = "test/model"

    def __init__(self, pieces: list[str]) -> None:
        """Build a model that will stream these pieces.

        Args:
            pieces: The text to stream.
        """
        self.pieces = pieces
        self.invoked = False

    def invoke(self, messages: object) -> AIMessage:
        """Answer in one piece, and record that this route was taken."""
        self.invoked = True
        return AIMessage(content="".join(self.pieces))

    def stream(self, messages: object) -> Iterator[AIMessageChunk]:
        """Stream the pieces, with the token counts on the last of them."""
        for index, piece in enumerate(self.pieces):
            last = index == len(self.pieces) - 1
            yield AIMessageChunk(
                content=piece,
                usage_metadata=(
                    {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18} if last else None
                ),
            )


class NothingAtAll:
    """A model whose stream yields no chunks whatsoever."""

    model_name = "test/model"

    def stream(self, messages: object) -> Iterator[AIMessageChunk]:
        """Yield nothing, which is not a reply."""
        return iter(())


def test_a_streamed_reply_arrives_in_pieces_and_whole() -> None:
    seen: list[str] = []
    model = Piecewise(["The gap ", "closes ", "linearly."])
    answer = ask_prose(model, "sys", "text", "explanation", sink=seen.append)  # type: ignore[arg-type]
    assert seen == ["The gap ", "closes ", "linearly."]
    assert answer == "The gap closes linearly."
    assert not model.invoked, "the sink was given and the call was still made in one piece"


def test_without_a_sink_nothing_is_streamed() -> None:
    # The default path, which every other caller and every test takes. A change that
    # streamed unconditionally would work and would quietly alter how every call in
    # the project is made.
    model = Piecewise(["one piece"])
    assert ask_prose(model, "sys", "text", "compose") == "one piece"  # type: ignore[arg-type]
    assert model.invoked


def test_a_streamed_call_is_still_billed(stream: io.StringIO) -> None:
    # The regression this is guarding. Usage rides on the final chunk, so a reply
    # accumulated as text rather than as messages loses it -- and the meter prices
    # the longest call in the graph at zero, which reads as a bill that goes down as
    # the work goes up.
    ask_prose(Piecewise(["a ", "b"]), "sys", "text", "explanation", sink=lambda _: None)  # type: ignore[arg-type]
    logged = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    calls = [record for record in logged if record.get("event") == "llm_call"]
    assert calls, "the streamed call was not logged at all"
    assert calls[-1]["prompt_tokens"] == 11
    assert calls[-1]["completion_tokens"] == 7


def test_a_stream_that_yields_nothing_is_a_failed_call() -> None:
    # Not an empty success. An empty reply that reported itself as fine would leave
    # the explanation branch falling back with nothing in the log to say why.
    assert ask_prose(NothingAtAll(), "sys", "text", "explanation", sink=lambda _: None) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The tools of one round run together, because the model asking for them
# together is what says they do not depend on each other
# --------------------------------------------------------------------------


def test_the_tools_of_one_round_run_at_the_same_time() -> None:
    # A barrier rather than a stopwatch. Asserting that three calls finished inside
    # some number of milliseconds passes on a fast laptop that has quietly broken
    # this and fails on a loaded one that has not; a barrier of three parties can
    # only be crossed if three calls are genuinely in flight, so the property is
    # proved rather than estimated. Serialised, the first call waits out the timeout
    # and every call comes back as a failure -- loudly, not as a hang.
    #
    # What was measured before this existed: a question that asks about three
    # algorithms is answered by three archive searches, and those are network round
    # trips that were being added up rather than overlapped.
    gate = threading.Barrier(3, timeout=10.0)

    @tool(parse_docstring=True)
    def rendezvous(name: str) -> str:
        """Wait until every tool this round asked for has arrived.

        Args:
            name: Which of the calls this is.

        Returns:
            The name, once all three are here.
        """
        gate.wait()
        return name

    asked = [{"name": "rendezvous", "args": {"name": name}, "id": name} for name in "abc"]
    consultation = ask_with_tools(
        Recorder([asked, None]),  # type: ignore[arg-type]
        "system",
        "text",
        [rendezvous],
        "tool_consultation",
    )

    assert consultation is not None
    assert [call.failed for call in consultation.calls] == [False, False, False]
    assert [call.result for call in consultation.calls] == ["a", "b", "c"]


def test_the_replies_keep_the_order_asked_for_rather_than_the_order_they_finished() -> None:
    # The ordering is not what makes the conversation valid -- a tool result is
    # matched to its request by identifier -- but a trace that reshuffles itself
    # between two identical runs is one nobody can diff, and the run view shows
    # these in the order they appear. So the slowest call is asked for first.
    @tool(parse_docstring=True)
    def linger(name: str, seconds: float) -> str:
        """Take a stated time, so finishing order can be made to differ from asking order.

        Args:
            name: Which of the calls this is.
            seconds: How long to take.

        Returns:
            The name.
        """
        time.sleep(seconds)
        return name

    asked = [
        {"name": "linger", "args": {"name": "first", "seconds": 0.20}, "id": "1"},
        {"name": "linger", "args": {"name": "second", "seconds": 0.10}, "id": "2"},
        {"name": "linger", "args": {"name": "third", "seconds": 0.0}, "id": "3"},
    ]
    model = Recorder([asked, None])
    consultation = ask_with_tools(
        model,  # type: ignore[arg-type]
        "system",
        "text",
        [linger],
        "tool_consultation",
    )

    assert consultation is not None
    assert [call.result for call in consultation.calls] == ["first", "second", "third"]


def test_one_tool_still_runs_and_is_reported() -> None:
    # The common case takes the inline path rather than the pool, so it needs its
    # own check: a round of one that stopped being run would break every
    # consultation in the project while both tests above still passed.
    consultation = ask_with_tools(
        Recorder([[{"name": "double", "args": {"value": 21}, "id": "1"}], None]),  # type: ignore[arg-type]
        "system",
        "text",
        [double],
        "tool_consultation",
    )

    assert consultation is not None
    assert len(consultation.calls) == 1
    assert "42" in consultation.calls[0].result


def test_a_tool_that_raises_beside_one_that_works_loses_only_itself() -> None:
    # Running them together must not let one failure take the round with it. The
    # loop's contract is that a failure comes back to the model as text it can act
    # on, and that has to survive the calls being concurrent.
    asked = [
        {"name": "explode", "args": {"value": 1}, "id": "1"},
        {"name": "double", "args": {"value": 4}, "id": "2"},
    ]
    consultation = ask_with_tools(
        Recorder([asked, None]),  # type: ignore[arg-type]
        "system",
        "text",
        [explode, double],
        "tool_consultation",
    )

    assert consultation is not None
    assert [call.failed for call in consultation.calls] == [True, False]
    assert "8" in consultation.calls[1].result
