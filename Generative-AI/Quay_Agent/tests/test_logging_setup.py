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
from pathlib import Path
from typing import Any

import pytest

from src import logging_setup
from src.logging_setup import (
    _STANDARD_FIELDS,
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


def test_a_streamed_replys_doubled_finish_reason_is_read_as_one() -> None:
    # Seen in a live log: `"finish_reason": "stopstop"`. Summing message chunks is
    # how a streamed reply is rebuilt -- it has to be, because the token counts ride
    # on the final chunk -- and merging two chunks that each say "stop" joins the
    # strings. Harmless for "stop" and not harmless for "length", which is the field's
    # whole purpose: it is what tells a reader a truncated answer was truncated rather
    # than broken, and "lengthlength" matches nothing anybody would compare against.
    call = CallLog(model="m", purpose="answer")
    call.observe(Response(metadata={"finish_reason": "stopstop"}))
    assert call.finish_reason == "stop"

    truncated = CallLog(model="m", purpose="answer")
    truncated.observe(Response(metadata={"finish_reason": "lengthlength"}))
    assert truncated.finish_reason == "length"


def test_an_unrecognised_finish_reason_is_printed_rather_than_guessed_at() -> None:
    # A reason this project has not seen is a fact about the provider. Collapsing it
    # to something familiar would be inventing data in the one place that exists to
    # report what actually happened.
    call = CallLog(model="m", purpose="answer")
    call.observe(Response(metadata={"finish_reason": "provider_specific"}))
    assert call.finish_reason == "provider_specific"


# --------------------------------------------------------------------------
# The terminal and the file carry different amounts
# --------------------------------------------------------------------------


def test_a_file_sink_takes_the_whole_stream_and_the_terminal_keeps_the_warnings(
    tmp_path: Path,
) -> None:
    # One question emits between forty and a hundred INFO records, and a person
    # waiting for an answer in the browser was reading all of them scroll past in
    # the terminal. They now go to a file. What must not happen is the obvious
    # over-correction: a quiet terminal that also drops the warnings, so a failed
    # model call or a missing credential becomes invisible.
    terminal = io.StringIO()
    path = tmp_path / "logs" / "quay.log"
    configure_logging("INFO", stream=terminal, log_file=path)

    logger = get_logger("probe")
    logger.info("routine_record", extra={"n": 1})
    logger.warning("worth_acting_on", extra={"n": 2})
    logger.error("definitely_worth_acting_on", extra={"n": 3})

    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()

    on_terminal = terminal.getvalue()
    assert "routine_record" not in on_terminal, "the flood is back on the terminal"
    assert "worth_acting_on" in on_terminal, "a warning must still reach the terminal"
    assert "definitely_worth_acting_on" in on_terminal

    # The parent directory did not exist before this call, so a first run on a
    # fresh checkout has to create it rather than fall back to the terminal.
    on_disk = path.read_text(encoding="utf-8")
    assert "routine_record" in on_disk, "the trace was not kept anywhere"
    assert "worth_acting_on" in on_disk
    assert "definitely_worth_acting_on" in on_disk
    # Still one JSON object per line, in the file as on the terminal.
    for line in on_disk.splitlines():
        assert json.loads(line)["logger"].startswith(LOGGER_NAME)


def test_the_default_stays_one_sink_at_full_volume(tmp_path: Path) -> None:
    # Every other caller -- the campaign runner, the eval harness, the MCP server,
    # an ingest -- runs to completion with somebody reading it, and for those the
    # terminal is the right place for everything. The quiet terminal is opt-in, and
    # asking for it without a file to move the records to would discard them rather
    # than relocate them, so `terminal_level` alone must not silence anything.
    terminal = io.StringIO()
    configure_logging("INFO", stream=terminal, terminal_level="WARNING")
    get_logger("probe").info("routine_record")
    assert "routine_record" in terminal.getvalue()


def test_reconfiguring_does_not_leak_a_file_for_every_streamlit_rerun(tmp_path: Path) -> None:
    # `configure_logging` is called on page draws, and Streamlit redraws on every
    # interaction. Removing a rotating file handler without closing it holds the
    # descriptor until the process runs out of them, which is a defect that only
    # appears after a few hundred clicks -- so it is worth a test rather than a
    # comment.
    path = tmp_path / "quay.log"
    for _ in range(50):
        configure_logging("INFO", stream=io.StringIO(), log_file=path)
    open_files = [
        handler
        for handler in logging.getLogger(LOGGER_NAME).handlers
        if isinstance(handler, logging.FileHandler)
    ]
    assert len(open_files) == 1, "a rerun added a handler instead of replacing it"


def test_an_unusable_log_path_keeps_the_terminal_rather_than_losing_the_trace(
    tmp_path: Path,
) -> None:
    # A read-only checkout, a full disk, or a directory where a file was expected.
    # The wrong failure here is a silent terminal with no file behind it, which
    # loses the whole trace to save some scrolling.
    blocked = tmp_path / "a-directory"
    blocked.mkdir()
    terminal = io.StringIO()
    configure_logging("INFO", stream=terminal, log_file=blocked)
    get_logger("probe").info("routine_record")
    assert "routine_record" in terminal.getvalue(), (
        "the file could not be opened and the records were dropped anyway"
    )


def test_the_start_up_notice_is_not_a_warning(tmp_path: Path) -> None:
    # A JSON object at WARNING is what a fault looks like, and on a clean start it
    # was the only line a person saw. The notice now goes to the terminal as plain
    # words and to the file as a record, so neither reader gets the wrong one.
    terminal = io.StringIO()
    path = tmp_path / "quay.log"
    configure_logging("INFO", stream=terminal, log_file=path)
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()

    on_terminal = terminal.getvalue().strip()
    assert '"level": "WARNING"' not in on_terminal, "the notice still reads as a fault"
    assert "logging_to_file" not in on_terminal, "the terminal got the machine record"
    assert not on_terminal.startswith("{"), "the notice is still JSON"
    assert path.name in on_terminal, "the notice does not say where the logs went"

    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["event"] == "logging_to_file"
    assert first["level"] == "INFO"


def test_the_start_up_notice_is_printed_once_however_often_streamlit_reruns(
    tmp_path: Path,
) -> None:
    # Streamlit re-executes the script on every interaction and this function with
    # it, so the notice was printed once per click and marched down the terminal --
    # which is the noise it had just been rewritten to stop being.
    terminal = io.StringIO()
    path = tmp_path / "quay.log"
    for _ in range(5):
        configure_logging("INFO", stream=terminal, log_file=path)

    assert terminal.getvalue().count("Logs:") == 1
    written = path.read_text(encoding="utf-8").splitlines()
    assert sum(json.loads(line)["event"] == "logging_to_file" for line in written) == 1

    # A different destination is a new fact and is said.
    elsewhere = tmp_path / "other.log"
    configure_logging("INFO", stream=terminal, log_file=elsewhere)
    assert terminal.getvalue().count("Logs:") == 2


def test_the_notice_also_survives_a_reload_of_this_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Streamlit re-imports a module whose source it sees change, which empties any
    # set at module level -- so a terminal that had already been told where the logs
    # go was told twice more over one editing session. Clearing the set is what the
    # re-import does to it.
    monkeypatch.setenv(logging_setup._ANNOUNCE_GUARD, "")
    terminal = io.StringIO()
    path = tmp_path / "quay.log"
    configure_logging("INFO", stream=terminal, log_file=path)
    logging_setup._ANNOUNCED.clear()
    configure_logging("INFO", stream=terminal, log_file=path)

    assert terminal.getvalue().count("Logs:") == 1


def test_no_structured_field_collides_with_a_field_logging_owns() -> None:
    # `extra={"thread": ...}` in the memory store crashed the whole interface on
    # the first click of a Delete button: LogRecord already has a `thread`
    # attribute holding the OS thread id, and logging raises KeyError rather than
    # let a caller overwrite one of its own. The failure is at the call site, so no
    # test of the formatter can catch it -- only reading the call sites can. And
    # the field was useless even without the crash: JsonFormatter filters standard
    # attributes out, so it would never have reached the log.
    # Parsed, not grepped: a regex over the source also matches the prose in a
    # docstring that names the problem, and the fix for that false positive is to
    # stop writing the docstring.
    import ast

    offenders: list[str] = []
    for root in (Path("src"), Path("scripts")):
        for module in sorted(root.rglob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg != "extra" or not isinstance(keyword.value, ast.Dict):
                        continue
                    for key in keyword.value.keys:
                        if isinstance(key, ast.Constant) and key.value in _STANDARD_FIELDS:
                            offenders.append(f"{module}:{node.lineno} passes {key.value!r}")

    assert not offenders, "these fields would raise KeyError when logged:\n" + "\n".join(offenders)


def test_a_caller_field_named_after_one_logging_owns_is_renamed_not_raised() -> None:
    # log_llm_call takes **extra, so the name of a field is the caller's to choose
    # and nothing checks it. Passing one logging reserves used to raise inside
    # makeRecord, which would lose a model call to a name clash.
    call = CallLog(model="some/model", purpose="answer", extra={"module": "solve", "run": "r1"})
    fields = call.fields()

    assert fields["call_module"] == "solve", "the value was dropped instead of renamed"
    assert "module" not in fields
    assert fields["run"] == "r1", "a field that clashes with nothing was renamed anyway"

    # The proof it is safe: logging accepts it.
    stream = io.StringIO()
    logger = get_logger("test-rename")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("llm_call", extra=fields)
    finally:
        logger.removeHandler(handler)
    assert json.loads(stream.getvalue())["call_module"] == "solve"
