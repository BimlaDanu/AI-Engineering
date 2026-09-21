"""Tests for structured logging.

Every case writes to an in-memory stream, so the suite never touches a file and
never depends on how the process running it configured its own logging. What is
asserted is the contract the rest of the project relies on: one JSON object per
line, structured fields survive, and a credential never reaches the output.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from src.logging_setup import (
    LOGGER_NAME,
    REDACTED,
    CallLog,
    JsonFormatter,
    configure_logging,
    get_logger,
    log_llm_call,
)


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    """Configure logging into a buffer, and restore the logger afterwards."""
    buffer = io.StringIO()
    configure_logging("DEBUG", stream=buffer)
    yield buffer
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def lines(stream: io.StringIO) -> list[dict[str, Any]]:
    """Parse the buffer as one JSON object per line."""
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


class Response:
    """A stand-in for a LangChain ``AIMessage``, without importing LangChain."""

    def __init__(self, usage: object = None, metadata: object = None) -> None:
        self.usage_metadata = usage
        self.response_metadata = metadata


# --------------------------------------------------------------------------
# Format
# --------------------------------------------------------------------------


def test_each_record_is_one_json_object_on_one_line(stream: io.StringIO) -> None:
    get_logger("test").info("first")
    get_logger("test").info("second")
    assert len(stream.getvalue().splitlines()) == 2
    assert [entry["event"] for entry in lines(stream)] == ["first", "second"]


def test_a_record_carries_level_logger_and_timestamp(stream: io.StringIO) -> None:
    get_logger("physics").warning("capped")
    entry = lines(stream)[0]
    assert entry["level"] == "WARNING"
    assert entry["logger"] == f"{LOGGER_NAME}.physics"
    assert entry["ts"].startswith("20")


def test_structured_fields_survive_as_fields_not_as_text(stream: io.StringIO) -> None:
    # The point of structured logging: `method` is queryable, not embedded in
    # an English sentence that a future format change would break.
    get_logger("agent").info("method_selected", extra={"method": "pfeuty_exact", "n_sites": 8})
    entry = lines(stream)[0]
    assert entry["method"] == "pfeuty_exact"
    assert entry["n_sites"] == 8


def test_an_unserialisable_value_degrades_instead_of_losing_the_record(
    stream: io.StringIO,
) -> None:
    get_logger("agent").info("odd", extra={"spec": object()})
    assert "object object at" in lines(stream)[0]["spec"]


def test_an_exception_is_recorded_with_its_traceback(stream: io.StringIO) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("agent").exception("failed")
    assert "ValueError: boom" in lines(stream)[0]["traceback"]


def test_the_level_is_honoured() -> None:
    buffer = io.StringIO()
    configure_logging("WARNING", stream=buffer)
    get_logger("test").debug("invisible")
    get_logger("test").warning("visible")
    assert [entry["event"] for entry in lines(buffer)] == ["visible"]


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------


def test_a_credential_in_a_message_is_masked() -> None:
    buffer = io.StringIO()
    configure_logging("INFO", stream=buffer, secrets=["sk-or-v1-abcdef123456"])
    get_logger("llm").info("calling with key sk-or-v1-abcdef123456")
    rendered = buffer.getvalue()
    assert "sk-or-v1-abcdef123456" not in rendered
    assert REDACTED in rendered


def test_a_credential_in_a_field_or_a_traceback_is_masked() -> None:
    # The realistic leak: an HTTP client echoes the request headers into the
    # exception text. Masking at the formatter catches that too.
    secret = "sk-or-v1-abcdef123456"
    buffer = io.StringIO()
    configure_logging("INFO", stream=buffer, secrets=[secret])
    get_logger("llm").info("call", extra={"url": f"https://x/?key={secret}"})
    try:
        raise RuntimeError(f"401 for {secret}")
    except RuntimeError:
        get_logger("llm").exception("rejected")
    assert secret not in buffer.getvalue()


def test_a_short_string_is_not_treated_as_a_secret() -> None:
    # Masking "ok" would turn every line into asterisks.
    buffer = io.StringIO()
    configure_logging("INFO", stream=buffer, secrets=["ok"])
    get_logger("llm").info("ok")
    assert lines(buffer)[0]["event"] == "ok"


def test_the_longest_matching_secret_is_masked_first() -> None:
    formatter = JsonFormatter(secrets=["abcdef123456", "abcdef123456-extended"])
    assert formatter.mask("abcdef123456-extended") == REDACTED


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_configuring_twice_does_not_double_print(stream: io.StringIO) -> None:
    # Streamlit re-executes the script on every interaction.
    configure_logging("DEBUG", stream=stream)
    configure_logging("DEBUG", stream=stream)
    get_logger("test").info("once")
    assert len(lines(stream)) == 1


def test_records_do_not_escape_to_the_root_logger(stream: io.StringIO) -> None:
    root = io.StringIO()
    handler = logging.StreamHandler(root)
    logging.getLogger().addHandler(handler)
    try:
        get_logger("test").info("contained")
    finally:
        logging.getLogger().removeHandler(handler)
    assert root.getvalue() == ""


# --------------------------------------------------------------------------
# Model calls
# --------------------------------------------------------------------------


def test_a_successful_call_logs_model_purpose_latency_and_outcome(stream: io.StringIO) -> None:
    with log_llm_call("anthropic/claude-haiku-4.5", purpose="answer") as call:
        call.finish_reason = "stop"
    entry = lines(stream)[0]
    assert entry["event"] == "llm_call"
    assert entry["model"] == "anthropic/claude-haiku-4.5"
    assert entry["purpose"] == "answer"
    assert entry["finish_reason"] == "stop"
    assert entry["outcome"] == "ok"
    assert entry["latency_ms"] >= 0.0


def test_extra_fields_reach_the_log_line(stream: io.StringIO) -> None:
    with log_llm_call("m", purpose="plan", run_id="abc"):
        pass
    assert lines(stream)[0]["run_id"] == "abc"


def test_a_failing_call_is_logged_and_the_error_still_propagates(stream: io.StringIO) -> None:
    # Observability must never swallow a failure.
    with pytest.raises(TimeoutError), log_llm_call("m", purpose="answer"):
        raise TimeoutError
    entry = lines(stream)[0]
    assert entry["level"] == "ERROR"
    assert entry["outcome"] == "error"
    assert entry["error_type"] == "TimeoutError"


def test_a_call_with_no_usage_omits_the_token_fields(stream: io.StringIO) -> None:
    with log_llm_call("m", purpose="answer"):
        pass
    assert "total_tokens" not in lines(stream)[0]


# --------------------------------------------------------------------------
# Token usage
# --------------------------------------------------------------------------


def test_usage_is_read_from_the_normalised_metadata() -> None:
    call = CallLog(model="m", purpose="answer")
    call.observe(Response(usage={"input_tokens": 120, "output_tokens": 30}))
    assert (call.prompt_tokens, call.completion_tokens, call.total_tokens) == (120, 30, 150)


def test_usage_falls_back_to_the_providers_raw_payload() -> None:
    # Some gateways populate only the OpenAI-shaped block.
    call = CallLog(model="m", purpose="answer")
    call.observe(Response(metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}}))
    assert call.total_tokens == 10


def test_the_finish_reason_is_read_from_the_response() -> None:
    # "length" is the usual explanation for an answer that looks truncated.
    call = CallLog(model="m", purpose="answer")
    call.observe(Response(metadata={"finish_reason": "length"}))
    assert call.finish_reason == "length"


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": None}, "unexpected"])
def test_a_missing_or_malformed_usage_report_is_not_an_error(usage: object) -> None:
    call = CallLog(model="m", purpose="answer")
    call.observe(Response(usage=usage))
    assert call.total_tokens is None


def test_a_response_without_the_attributes_at_all_is_ignored() -> None:
    call = CallLog(model="m", purpose="answer")
    call.observe(object())
    assert call.total_tokens is None
