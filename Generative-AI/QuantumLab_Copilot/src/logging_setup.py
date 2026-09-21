"""Structured logging, configured once at start-up.

Every model call, tool call and refusal in this project is recorded here as one
JSON object per line: machine-readable, greppable, and safe to ship to a log
collector. ``print()`` is banned project-wide because it cannot carry a level,
a timestamp or a field, and because it writes to stdout, which Streamlit and
the evaluation runner both use for their own output.

Two decisions worth stating.

*This module imports nothing from* :mod:`src.settings`. Configuration can fail
-- a missing credential raises at start-up, by design -- and a failure that
cannot be logged is a failure nobody can debug. Logging therefore has no
dependencies inside this project at all.

*Secrets are masked at the formatter*, not at the call site. Masking where the
text is rendered catches the key wherever it came from: an argument, an
``extra`` field, or an exception message from an HTTP client that helpfully
echoed the request headers. Relying on every caller to remember would not.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TextIO

LOGGER_NAME = "quantumlab"
"""Root of this project's logger hierarchy.

Configuration is attached here rather than to the root logger so that the
project's own records are formatted as JSON while Streamlit, httpx and urllib3
keep whatever logging their own operators configured.
"""

REDACTED = "***"
"""What a masked secret is replaced with."""

_MIN_SECRET_LENGTH = 8
"""Shortest string worth masking.

A one- or two-character "secret" would match everywhere and turn every log line
into asterisks. Real credentials are far longer than this floor.
"""

_STANDARD_FIELDS = frozenset(
    vars(logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None))
) | {"message", "asctime", "taskName"}
"""Attributes every :class:`logging.LogRecord` carries.

Anything on a record that is *not* in this set arrived through ``extra=`` and is
therefore one of our own structured fields. Deriving the set from a real record
keeps it correct across Python versions instead of hard-coding a list that
quietly rots.
"""


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object on one line.

    Attributes:
        secrets: Literal strings to mask wherever they appear in the output.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        """Build a formatter.

        Args:
            secrets: Values to mask. Short strings are ignored -- see
                :data:`_MIN_SECRET_LENGTH`.
        """
        super().__init__()
        self.secrets = tuple(
            sorted(
                {value for value in secrets if len(value) >= _MIN_SECRET_LENGTH},
                key=len,
                reverse=True,  # mask the longest first, so a prefix cannot pre-empt it
            )
        )

    def mask(self, text: str) -> str:
        """Replace every known secret in ``text`` with :data:`REDACTED`.

        Args:
            text: Rendered log output.

        Returns:
            The same text with credentials removed.
        """
        for secret in self.secrets:
            text = text.replace(secret, REDACTED)
        return text

    def format(self, record: logging.LogRecord) -> str:
        """Render one record.

        Args:
            record: The record to format.

        Returns:
            A JSON object: timestamp, level, logger and event first, then every
            field passed through ``extra=``, sorted so that two runs of the same
            code produce diffable logs.
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extras = {key: value for key, value in vars(record).items() if key not in _STANDARD_FIELDS}
        payload.update(dict(sorted(extras.items())))
        if record.exc_info is not None:
            payload["traceback"] = self.formatException(record.exc_info)
        # default=str so an un-serialisable value degrades to its repr instead
        # of raising inside the logging call and swallowing the record.
        return self.mask(json.dumps(payload, default=str))


def configure_logging(
    level: str = "INFO",
    *,
    stream: TextIO | None = None,
    secrets: Iterable[str] = (),
) -> logging.Logger:
    """Install the project's log handler. Safe to call more than once.

    Existing handlers are replaced rather than added to, so a Streamlit rerun --
    which re-executes the script top to bottom -- cannot end up printing every
    line four times.

    Args:
        level: Minimum level to emit, as a name such as ``"DEBUG"``.
        stream: Where to write. Defaults to ``sys.stderr``, resolved at call
            time so that a test's captured stream is honoured. Never stdout:
            that channel belongs to the application's own output.
        secrets: Literal credential values to mask in every line.

    Returns:
        The configured project logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(JsonFormatter(secrets=secrets))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    # Do not also hand records to the root logger, which would double-print them
    # under pytest and under Streamlit, both of which configure the root.
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger for one component.

    Args:
        name: A short component name such as ``"agent"`` or ``"tools"``.

    Returns:
        The logger ``quantumlab.<name>``, which inherits this module's handler
        and level.
    """
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


@dataclass(slots=True)
class CallLog:
    """The record of one model call, filled in as the call proceeds.

    Mutable on purpose: the caller receives it from :func:`log_llm_call` before
    the response exists and fills in what it learns.

    Attributes:
        model: Model identifier that was called.
        purpose: Why it was called -- ``"plan"``, ``"answer"``, ``"summarise"``.
            This is what makes the log answerable to "which step is burning the
            budget?" rather than only "how many tokens today?".
        prompt_tokens: Tokens sent, if the provider reported them.
        completion_tokens: Tokens generated, if reported.
        finish_reason: Why generation stopped. ``"length"`` here is the usual
            explanation for a truncated answer that otherwise looks like a bug.
        latency_ms: How long the call took, filled in by :func:`log_llm_call`.
        outcome: ``"ok"`` or ``"error"``, likewise.
        error_type: The exception class name, when the call raised.
        extra: Any further fields to log with the call.
    """

    model: str
    purpose: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    outcome: str | None = None
    error_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        """Tokens in and out, or ``None`` if the provider reported neither."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    def observe(self, response: object) -> None:
        """Read token usage and finish reason off a model response.

        Duck-typed rather than typed against ``AIMessage`` so that this module
        keeps its zero-dependency property and so that a test can pass a stub.
        Both shapes LangChain exposes are read: the normalised
        ``usage_metadata``, and the provider's raw ``response_metadata`` as a
        fallback for gateways that do not populate the former.

        Args:
            response: A model response, or anything carrying the same
                attributes. Objects carrying neither are ignored silently --
                missing usage is not an error, it is a provider that did not
                report any.
        """
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            self.prompt_tokens = _as_int(usage.get("input_tokens"))
            self.completion_tokens = _as_int(usage.get("output_tokens"))

        metadata = getattr(response, "response_metadata", None)
        if not isinstance(metadata, dict):
            return
        reason = metadata.get("finish_reason")
        if isinstance(reason, str):
            self.finish_reason = reason
        raw = metadata.get("token_usage")
        if isinstance(raw, dict):
            if self.prompt_tokens is None:
                self.prompt_tokens = _as_int(raw.get("prompt_tokens"))
            if self.completion_tokens is None:
                self.completion_tokens = _as_int(raw.get("completion_tokens"))

    def fields(self) -> dict[str, Any]:
        """Render the record as structured log fields.

        Returns:
            The fields to attach to a log line. Values the provider never
            reported are omitted rather than logged as ``null``, so a query for
            "calls with no usage data" is a query for a missing key.
        """
        fields: dict[str, Any] = {"model": self.model, "purpose": self.purpose}
        optional = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "outcome": self.outcome,
            "error_type": self.error_type,
        }
        fields.update({key: value for key, value in optional.items() if value is not None})
        fields.update(self.extra)
        return fields


def _as_int(value: object) -> int | None:
    """Coerce a reported token count to ``int``, or ``None`` if it is not one.

    Args:
        value: Whatever the provider put in the usage payload.

    Returns:
        The count, or ``None``. Providers have been known to send strings or
        nulls here, and a malformed usage report must not break the call it
        describes.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        return int(value)
    except ValueError:
        return None


CallObserver = Callable[[CallLog], None]
"""Something notified once per finished model call, successful or not."""

_observers: ContextVar[tuple[CallObserver, ...]] = ContextVar("_call_observers", default=())
"""Observers active on the current execution context.

A :class:`~contextvars.ContextVar` rather than a module-level list, because
Streamlit serves reruns from a worker pool: two people asking questions at the
same moment must not accumulate each other's tokens. A context variable is
inherited by whatever the current thread starts and invisible to its siblings,
which is exactly the scoping a per-answer meter needs.
"""


@contextmanager
def observing_calls(observer: CallObserver) -> Iterator[None]:
    """Notify ``observer`` of every model call made inside this block.

    Args:
        observer: Called once per finished call, with the completed record.

    Yields:
        Nothing. The block runs with the observer installed and it is removed
        again afterwards, including if the block raises.

    Examples:
        >>> seen: list[CallLog] = []
        >>> with observing_calls(seen.append):
        ...     with log_llm_call("some/model", purpose="answer") as call:
        ...         call.finish_reason = "stop"
        >>> [record.purpose for record in seen]
        ['answer']
    """
    token = _observers.set((*_observers.get(), observer))
    try:
        yield
    finally:
        _observers.reset(token)


def _notify(call: CallLog, sink: logging.Logger) -> None:
    """Hand a finished record to every active observer.

    Args:
        call: The completed record.
        sink: Where to report an observer that raised.

    An observer that fails is logged and skipped. Accounting is worth having but
    it is not worth an answer: a bookkeeping bug must not turn a computed result
    into a traceback.
    """
    for observer in _observers.get():
        try:
            observer(call)
        except Exception as error:  # pragma: no cover - defensive
            sink.warning("call_observer_failed", extra={"error_type": type(error).__name__})


@contextmanager
def log_llm_call(
    model: str,
    *,
    purpose: str,
    logger: logging.Logger | None = None,
    **extra: Any,
) -> Iterator[CallLog]:
    """Time a model call, log it, and report it, whether it succeeds or raises.

    Args:
        model: Model identifier being called.
        purpose: Why it is being called. See :attr:`CallLog.purpose`.
        logger: Where to log. Defaults to the ``quantumlab.llm`` logger.
        **extra: Further fields to record, such as a run identifier.

    Yields:
        The record to fill in, typically via :meth:`CallLog.observe`.

    Raises:
        Exception: Whatever the wrapped call raised, re-raised unchanged after
            the failure is logged. This context manager observes; it does not
            handle.

    Examples:
        >>> with log_llm_call("some/model", purpose="answer") as call:
        ...     call.finish_reason = "stop"
    """
    sink = get_logger("llm") if logger is None else logger
    call = CallLog(model=model, purpose=purpose, extra=dict(extra))
    # perf_counter, not time(): monotonic, so an NTP correction mid-call cannot
    # produce a negative latency.
    started = time.perf_counter()

    def finish(outcome: str, error: BaseException | None = None) -> None:
        call.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        call.outcome = outcome
        call.error_type = None if error is None else type(error).__name__

    try:
        yield call
    except Exception as error:
        finish("error", error)
        sink.error("llm_call", extra=call.fields())
        _notify(call, sink)
        raise
    finish("ok")
    sink.info("llm_call", extra=call.fields())
    # After the log line, not before: a record that was worth logging is worth
    # counting, and this order means the log is written even if an observer hangs.
    _notify(call, sink)
