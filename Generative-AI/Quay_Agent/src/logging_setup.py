"""Structured logging, configured once at start-up.

Every model call, tool call and refusal in this project is recorded here as one
JSON object per line: machine-readable, greppable, and safe to ship to a log
collector. Nothing inside the application prints: ``print()`` appears only in
the ``main()`` of a script or a runner, where stdout *is* the interface. It
cannot carry a level, a timestamp or a field, so it is no substitute for a log
record anywhere a caller might be the UI or the evaluation runner.

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
import logging.handlers
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

LOGGER_NAME = "quay"
"""Root of this project's logger hierarchy.

Configuration is attached here rather than to the root logger so that the
project's own records are formatted as JSON while Streamlit, httpx and urllib3
keep whatever logging their own operators configured.
"""

REDACTED = "***"
"""What a masked secret is replaced with."""

DEFAULT_LOG_PATH = Path("reports") / "quay.log"
"""Where the full structured log goes when a caller asks for a file.

Not a second copy of the terminal output -- the *only* complete copy. A long
interactive session emits a JSON object per model call and per graph node, which is
between forty and a hundred lines for one question, and a terminal that receives all
of them stops being readable: the answer a person is waiting on scrolls past under
its own diagnostics, and the one line that mattered is indistinguishable from the
ninety-nine that did not. Sending the stream to a file and leaving the terminal for
what a person must act on keeps both useful.

``.log`` is already ignored by version control, so this cannot be committed by
accident.
"""

LOG_PATH_VARIABLE = "QUAY_LOG_PATH"
"""Environment variable that overrides :data:`DEFAULT_LOG_PATH`.

Two callers want this and neither is exotic. An operator running the interface
somewhere the working directory is not writable, or who wants the stream where a log
collector is already watching, sets it to a path. **The test suite sets it to the
empty string, which means no file at all** -- and that is the reason it exists rather
than a happy side effect.

The suite has to be able to switch the file off from outside the process, because the
page tests import page modules, a Streamlit page module *runs* when imported, and so
the interface configures its logging during collection -- before the first fixture of
the first test. A monkeypatch cannot reach that; an environment variable read at call
time can. Without it the suite appends to a file inside the repository on every run,
which is the general form of a defect this project has hit before: a test that
writes outside its temporary directory changes the next run.

Empty rather than a sentinel word because "off" is the natural reading of an empty
path, and because a variable whose off-switch is a magic string is a variable people
get wrong.
"""

TERMINAL_LEVEL_WITH_A_FILE = "WARNING"
"""What the terminal keeps when the full stream is going to a file.

Warnings and errors, and never less than that. Silencing the terminal completely
would be the obvious way to make it quiet and the wrong one: a person watching a run
would lose the only channel that tells them a model call failed, a credential is
missing or a note was refused, and would learn it instead from an answer that came
back thinner than it should have. The rule is that the terminal carries what somebody
has to *act* on and the file carries everything, which is the same division a server
makes between its console and its log.
"""

MAX_LOG_BYTES = 5 * 1024 * 1024
"""Size at which the log file rolls over.

A log nobody rotates is a disk-space defect with a long fuse. Five megabytes is a few
hundred thousand records -- far more than any one session produces -- and small enough
to open in an editor.
"""

LOG_BACKUPS = 3
"""Rolled-over files to keep, so roughly the last twenty megabytes stays readable."""

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
    log_file: Path | str | None = None,
    terminal_level: str | None = None,
) -> logging.Logger:
    """Install the project's log handlers. Safe to call more than once.

    Existing handlers are replaced rather than added to, so a Streamlit rerun --
    which re-executes the script top to bottom -- cannot end up printing every
    line four times.

    There are two sinks on purpose and they carry different amounts. With no
    ``log_file`` this behaves exactly as it always has: one handler, everything at
    ``level``, on the terminal. Given one, the file takes the complete stream and
    the terminal is raised to ``terminal_level`` -- so a person watching a run sees
    what they have to act on, and the full trace is still on disk, complete and
    masked, for the question they ask afterwards. The alternative to a file is not
    a quieter terminal, it is a lost trace: this project logs a JSON object per
    model call and per graph node, which is the observability it is graded on, and
    the choice was never between noisy and quiet but between *scrolling past* and
    *keeping*.

    Args:
        level: Minimum level to emit, as a name such as ``"DEBUG"``. Applies to the
            file when there is one, and to the terminal otherwise.
        stream: Where to write. Defaults to ``sys.stderr``, resolved at call
            time so that a test's captured stream is honoured. Never stdout:
            that channel belongs to the application's own output.
        secrets: Literal credential values to mask in every line. Masking happens
            in the formatter, so both sinks are masked by the same code.
        log_file: A path to append the full stream to, rotated at
            :data:`MAX_LOG_BYTES`. Its parent is created if it does not exist. A
            path that cannot be opened is reported and then ignored -- see below.
        terminal_level: What the terminal keeps once a file is carrying everything.
            Defaults to :data:`TERMINAL_LEVEL_WITH_A_FILE`. Ignored without a file,
            because raising the terminal with nowhere else to write would discard
            the records rather than move them.

    Returns:
        The configured project logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        # Closed as well as removed. `StreamHandler.close` leaves its stream alone,
        # so stderr survives this, but a rotating file handler holds a descriptor
        # and Streamlit reruns this on every interaction -- dropping the reference
        # without closing leaks one file per rerun until the process runs out.
        existing.close()

    formatter = JsonFormatter(secrets=secrets)
    on_disk = _file_handler(log_file, formatter, level) if log_file is not None else None

    terminal = logging.StreamHandler(sys.stderr if stream is None else stream)
    terminal.setFormatter(formatter)
    if on_disk is not None:
        terminal.setLevel(
            (TERMINAL_LEVEL_WITH_A_FILE if terminal_level is None else terminal_level).upper()
        )
    logger.addHandler(terminal)
    if on_disk is not None:
        logger.addHandler(on_disk)

    # The logger itself stays at `level` so the file sees everything; the terminal
    # filters at its own handler. Setting the logger high instead would drop the
    # record before either sink saw it.
    logger.setLevel(level.upper())
    # Do not also hand records to the root logger, which would double-print them
    # under pytest and under Streamlit, both of which configure the root.
    logger.propagate = False

    if on_disk is not None:
        # The file gets a structured record; the terminal gets one plain sentence,
        # written straight to the stream. Through the logger there is no level that
        # is both honest and visible: the terminal keeps WARNING, and a JSON object
        # at WARNING on a clean start reads as a fault.
        #
        # Once per destination, not once per call. Streamlit re-executes the script
        # on every interaction and this function with it, so a line printed here
        # unconditionally repeats down the terminal for as long as somebody keeps
        # clicking -- which is how a start-up notice turns into the noise it was
        # written to replace.
        written_to = Path(on_disk.baseFilename)
        try:
            written_to = written_to.relative_to(Path.cwd())
        except ValueError:
            pass
        kept = logging.getLevelName(terminal.level)
        destination = (str(written_to), kept)
        stamp = f"{written_to} {kept}"
        spoken = os.environ.get(_ANNOUNCE_GUARD, "").splitlines()
        if destination not in _ANNOUNCED and stamp not in spoken:
            _ANNOUNCED.add(destination)
            os.environ[_ANNOUNCE_GUARD] = "\n".join([*spoken, stamp])
            logger.info(
                "logging_to_file",
                extra={"path": str(written_to), "terminal_keeps": kept},
            )
            _announce(terminal.stream, f"Logs: {written_to} (terminal shows {kept} and above)")
    return logger


_ANNOUNCED: set[tuple[str, str]] = set()
"""Destinations already announced, so a rerun repeats nothing.

Keyed on the path and the level rather than on a bare flag: reconfiguring to a
different file, or raising what the terminal keeps, is a new fact and is said. Not
cleared anywhere -- the set is per process and a process has one terminal.
"""

_ANNOUNCE_GUARD = "QUAY_LOGS_ANNOUNCED"
"""Environment name holding the same destinations, one to a line.

A module-level set is emptied by re-import, and Streamlit re-imports a module whose
source it sees change -- which put a second and a third ``Logs:`` line down a
terminal that was only ever meant to get one. The environment belongs to the process
rather than to the module, so it survives that.
"""


def _announce(stream: TextIO, sentence: str) -> None:
    """Write one plain line to the terminal, outside the logging levels.

    For a start-up fact that is not a fault. Never raises: a stream that will not
    take a write is not a reason to fail a start-up, and the record is in the file
    either way.

    Args:
        stream: Where the terminal handler writes.
        sentence: The line, without a trailing newline.
    """
    try:
        stream.write(f"{sentence}\n")
        stream.flush()
    except Exception:  # pragma: no cover - a closed or read-only stream
        pass


def _file_handler(
    log_file: Path | str,
    formatter: logging.Formatter,
    level: str,
) -> logging.handlers.RotatingFileHandler | None:
    """Open the rotating file sink, or report why it could not be opened.

    Args:
        log_file: Where to append.
        formatter: The masking formatter, shared with the terminal so a credential
            cannot be redacted on one sink and printed on the other.
        level: Minimum level the file keeps.

    Returns:
        The handler, or ``None`` when the path is unusable -- a read-only checkout,
        a directory where a file was expected, a full disk. ``None`` makes the
        caller leave the terminal at full volume, which is the right failure: a
        noisy terminal is an inconvenience and a silent one with no file behind it
        is the loss of the whole trace.
    """
    path = Path(log_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUPS,
            encoding="utf-8",
        )
    except OSError as error:
        # Written straight to the stream rather than logged, because this runs
        # while the logger is mid-reconfiguration and has no handler that would
        # carry it. `sys.stderr.write` rather than `print`, which this project
        # bans for the reasons in the module docstring -- and the ban is about
        # using it as a logging channel, which is exactly what a fallback for a
        # failed logging channel must not pretend to be.
        sys.stderr.write(
            f"quay: could not open {path} for logging ({type(error).__name__}), "
            "keeping the full stream on the terminal\n"
        )
        return None
    handler.setFormatter(formatter)
    handler.setLevel(level.upper())
    return handler


def log_path_from_environment(default: Path = DEFAULT_LOG_PATH) -> Path | None:
    """Where the full stream should be written, honouring the environment.

    Args:
        default: Used when :data:`LOG_PATH_VARIABLE` is not set at all.

    Returns:
        The path to write to, or ``None`` when the variable is set to the empty
        string -- which means "no file, keep everything on the terminal" and is how
        the test suite and any read-only deployment switch the file off.

    Examples:
        >>> import os
        >>> os.environ[LOG_PATH_VARIABLE] = ""
        >>> log_path_from_environment() is None
        True
        >>> os.environ[LOG_PATH_VARIABLE] = "/tmp/somewhere.log"
        >>> log_path_from_environment()
        PosixPath('/tmp/somewhere.log')
        >>> del os.environ[LOG_PATH_VARIABLE]
        >>> log_path_from_environment(Path("reports/quay.log"))
        PosixPath('reports/quay.log')
    """
    chosen = os.environ.get(LOG_PATH_VARIABLE)
    if chosen is None:
        return default
    stripped = chosen.strip()
    return Path(stripped) if stripped else None


def get_logger(name: str) -> logging.Logger:
    """Return a logger for one component.

    Args:
        name: A short component name such as ``"agent"`` or ``"tools"``.

    Returns:
        The logger ``quay.<name>``, which inherits this module's handler
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
            self.finish_reason = _one_finish_reason(reason)
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
            "calls with no usage data" is a query for a missing key. A caller
            field that shares a name with one :class:`logging.LogRecord` already
            owns is prefixed rather than passed through -- a caller field named
            ``module`` or ``thread`` raises inside ``logging`` itself, and losing
            a model call to a bookkeeping name clash is not a trade worth making.
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
        fields.update(
            {
                (f"call_{key}" if key in _STANDARD_FIELDS else key): value
                for key, value in self.extra.items()
            }
        )
        return fields


FINISH_REASONS: tuple[str, ...] = (
    "content_filter",
    "function_call",
    "tool_calls",
    "length",
    "stop",
)
"""The finish reasons a provider may report, longest first.

Longest first because :func:`_one_finish_reason` matches by prefix, and ``"stop"``
is a prefix of nothing here but ``"tool_calls"`` would be missed if a shorter token
matched ahead of it.
"""


def _one_finish_reason(reason: str) -> str:
    """Collapse a reason that arrived concatenated with itself.

    A streamed reply's metadata is merged, and strings are merged by joining them.
    LangChain sums message chunks to rebuild a streamed reply -- this
    project does that deliberately, because the token counts ride on the final
    chunk and a reply rebuilt by concatenating text would arrive unpriced -- and
    summing two chunks that each carry ``finish_reason="stop"`` produces
    ``"stopstop"``. Observed in a live log on the one call that streams.

    Cosmetic in the ``"stop"`` case and not cosmetic at all in the other one: the
    docstring on :attr:`CallLog.finish_reason` offers ``"length"`` as "the usual
    explanation for a truncated answer that otherwise looks like a bug", and
    ``"lengthlength"`` is a value no reader and no comparison will recognise. The
    field exists to be matched against, so it has to carry the value it means.

    Args:
        reason: What the provider reported, possibly repeated.

    Returns:
        The single reason, when the string is one known token repeated one or more
        times. Anything else is returned unchanged -- an unrecognised reason is a
        fact about the provider and guessing at it would be worse than printing it.

    Examples:
        >>> _one_finish_reason("stopstop")
        'stop'
        >>> _one_finish_reason("stop")
        'stop'
        >>> _one_finish_reason("lengthlengthlength")
        'length'
        >>> _one_finish_reason("something_new")
        'something_new'
    """
    for known in FINISH_REASONS:
        if reason == known:
            return known
        if len(reason) > len(known) and reason == known * (len(reason) // len(known)):
            return known
    return reason


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
        logger: Where to log. Defaults to the ``quay.llm`` logger.
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
