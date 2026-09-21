"""What the agent remembers between questions, and the rules on what it may keep.

A checkpointer saves graph state; this saves what was said. Resuming a graph
mid-run needs every intermediate value and is all-or-nothing; answering a
follow-up needs the previous question and answer and degrades gracefully, since
with no memory the agent is merely forgetful.

So this is the smaller mechanism: an append-only JSONL file, one record per line,
keyed by who asked and which conversation it belonged to. No migration, no
database, no daemon, and a person can read everything the application remembers
about them with ``cat``.

Two kinds of record, keyed differently on purpose. A *turn* is a question and the
verdict it reached, and belongs to one conversation -- recalling yesterday's
unrelated campaign is noise that invites the model to answer the wrong question. A
*rating* is the reading level someone asked for, and belongs to the person: a
preference that resets at a thread boundary is not really held.

Four rules on what may be stored:

- A blocked question is never written. Memory is replayed into a later prompt, so
  storing an injection turns one attempt into a persistent one.
- Everything written is defanged first, because the file is read back into a
  prompt and quoted in a terminal. Safe to display, not merely safe to parse.
- Every field is truncated. An unbounded memory is an unbounded prompt.
- Recall is four turns, not the whole history -- a cost decision rather than a
  storage one.

Failure is never fatal. A memory file that cannot be read or was left half-written
costs the user their history and nothing else; every operation here logs and
continues.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args

from src.logging_setup import get_logger
from src.security import neutralise, screen
from src.settings import Audience, Settings, get_settings

_log = get_logger("agent.memory")

Kind = Literal["turn", "rating"]
"""What one stored record is.

Two values rather than a hierarchy of classes because the file is meant to be read
by a person. A third kind would be a third column in something anybody can grep.
"""

ANONYMOUS = "anonymous"
"""Who a turn belongs to when no user has been identified.

A single-user desktop session has nobody to name and should not be forced to
invent one. The key still exists, so the day a second user appears the file does
not have to be re-keyed.
"""

DEFAULT_THREAD = "default"
"""The conversation a turn belongs to when the caller did not say."""

AUDIENCE_VALUES: tuple[Audience, ...] = get_args(Audience)
"""Every reading level, read off the type rather than restated.

A second hand-written list would be a second thing to update, and the one that
was missed would be the one validating a file a person can edit.
"""

RECALL_TURNS = 4
"""How many previous turns are offered to a prompt.

Four is roughly the depth at which follow-ups stop referring back -- "and at
larger $L$?" reaches one turn behind, almost never five. Raising it costs input
tokens on every question to serve a case that does not arise.
"""

MAX_STORED_CHARACTERS = 1200
"""Longest any single stored field may be.

A little over half the limit on a question, and comfortably more than a verdict
sentence. Truncation is visible in the file rather than silent.
"""

MAX_LINES_SCANNED = 2000
"""How far back a read looks before giving up on older history.

The file is read whole and the tail is kept, which is the right trade at the scale
of one person's log -- a few hundred kilobytes read in a millisecond. If a
deployment ever puts many users in one file, this is the line that has to become a
seek, and the constant is here so that it is findable.
"""

TRUNCATION_MARK = "..."


@dataclass(frozen=True, slots=True)
class Entry:
    """One line of the log.

    Flat rather than nested. Every field is a string and every record has the same
    keys, so a line can be read at a glance and the whole file survives
    ``grep``-ing for a user id. The cost is a few empty strings per line, which is
    a price worth paying for a format whose reader is a human being.

    Attributes:
        kind: Whether this records something said or something preferred.
        at: When it happened, ISO-8601 in UTC. UTC because a memory file that
            moves between machines should not reorder itself.
        user: Who it belongs to.
        thread: Which conversation it belonged to. Carried on ratings too, so the
            record says where a preference was expressed even though recall
            ignores it.
        question: What was asked. Empty on a rating.
        answer: What was answered, trimmed to the part worth recalling. Empty on
            a rating.
        verdict: The campaign's conclusion, when there was one.
        audience: The reading level asked for. Empty on a turn.
    """

    kind: Kind
    at: str
    user: str
    thread: str
    question: str = ""
    answer: str = ""
    verdict: str = ""
    audience: str = ""

    def as_line(self) -> str:
        """Render this entry as the JSON object that will occupy one line.

        Returns:
            A single-line JSON object with no trailing newline. Keys are written
            in declaration order rather than sorted, so the columns line up when
            several records are read together.
        """
        return json.dumps(
            {
                "kind": self.kind,
                "at": self.at,
                "user": self.user,
                "thread": self.thread,
                "question": self.question,
                "answer": self.answer,
                "verdict": self.verdict,
                "audience": self.audience,
            },
            ensure_ascii=False,
        )

    def summarise(self) -> str:
        """Render this entry as one line of prompt context.

        Returns:
            A plain-language line naming what was asked and what came back. Only
            meaningful for a turn; a rating summarises to the preference it set,
            which is what a reader of a trace would want to see.
        """
        if self.kind == "rating":
            return f"the reader asked for the {self.audience} reading level"
        tail = f" -- verdict: {self.verdict}" if self.verdict else ""
        return f"asked: {self.question}\nanswered: {self.answer}{tail}"


def _now() -> str:
    """Stamp the current moment in the one format this module writes.

    Returns:
        ISO-8601 to the second, in UTC. Seconds rather than microseconds because
        the extra digits are noise in a file meant to be read.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def _trim(text: str, limit: int = MAX_STORED_CHARACTERS) -> str:
    """Bound one field, leaving evidence that it was bounded.

    Args:
        text: The value to store.
        limit: Longest form to keep.

    Returns:
        The text, or its opening ``limit`` characters followed by a visible mark.
        The mark matters: a silently truncated answer read back into a prompt is
        a sentence that stops mid-clause for no reason the model can see.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + TRUNCATION_MARK


@dataclass(frozen=True, slots=True)
class Thread:
    """One conversation, summarised for a list of them.

    Attributes:
        name: The conversation's key, as :attr:`Memory.thread` holds it.
        turns: How many exchanges it contains.
        opened_with: The first question asked in it, which is what a person
            recognises a conversation by -- far better than a timestamp or an
            identifier, neither of which anybody remembers.
        last_at: When it was last added to, in ISO-8601 UTC.
    """

    name: str
    turns: int
    opened_with: str
    last_at: str

    def label(self, width: int = 48) -> str:
        """Render this conversation as one line for a list.

        Args:
            width: Characters of the opening question to keep.

        Returns:
            The trimmed question and the turn count.
        """
        opener = self.opened_with[:width].rstrip()
        if len(self.opened_with) > width:
            opener += TRUNCATION_MARK
        return f"{opener}  ({self.turns})"


@dataclass
class Memory:
    """One person's memory, in one conversation, backed by one file.

    The file is shared -- several users and several threads live in it -- but an
    instance is bound to a single pair, because that is how every call site uses
    it. Passing the keys to each method instead would put the same two arguments
    on every call and make it possible to write a turn under the wrong user by
    getting an argument order wrong.

    Attributes:
        user: Who this memory belongs to.
        thread: Which conversation it covers.
        path: Where the log lives. ``None`` asks ``settings``; a memory that is
            genuinely switched off is built with :meth:`disabled`, so "not
            configured yet" and "deliberately off" are distinguishable at the
            call site rather than only in a configuration file.
        settings: Configuration to read the path from. Defaults to the process
            settings.
        off: Set by :meth:`disabled`. Suppresses every read and write without
            needing a path.
    """

    user: str = ANONYMOUS
    thread: str = DEFAULT_THREAD
    path: Path | None = None
    settings: Settings | None = None
    off: bool = field(default=False)

    def __post_init__(self) -> None:
        """Resolve the log location once, so no later call has to."""
        if self.off or self.path is not None:
            return
        resolved = get_settings() if self.settings is None else self.settings
        if resolved.memory_path is not None:
            self.path = Path(resolved.memory_path)

    @classmethod
    def disabled(cls) -> Memory:
        """Build a memory that stores nothing and recalls nothing.

        A supported mode rather than a degraded one. Tests, one-shot scripts and
        the evaluation harness all want an agent that starts from nothing every
        time, and they should be able to say so in one call rather than by
        constructing settings around it.

        Returns:
            A memory whose reads return empty and whose writes are no-ops.
        """
        return cls(off=True, path=None)

    @property
    def enabled(self) -> bool:
        """Whether anything will actually be stored or recalled."""
        return not self.off and self.path is not None

    # -- writing ----------------------------------------------------------

    def remember_turn(self, question: str, answer: str, verdict: str = "") -> Entry | None:
        """Record one exchange, if the rules allow it.

        Args:
            question: What the user asked, as they typed it.
            answer: What came back. Pass the part worth recalling rather than a
                whole report -- this is prompt context for a follow-up, not an
                archive.
            verdict: The campaign's conclusion, when the exchange produced one.

        Returns:
            The entry written, or ``None`` if memory is off or the question was
            refused by the screen. A refusal is returned as ``None`` rather than
            raised because the caller has already answered the user; failing to
            record the exchange is not their problem to handle.
        """
        if not self.enabled:
            return None
        screening = screen(question)
        if screening.blocked:
            # Recording it would make one attempt permanent. The refusal is worth
            # a log line, though: repeated entries here are a pattern nobody
            # would otherwise see, because each individual attempt was blocked
            # upstream and looked like an isolated event.
            _log.warning(
                "memory_refused",
                extra={"user": self.user, "categories": list(screening.categories)},
            )
            return None
        return self._append(
            Entry(
                kind="turn",
                at=_now(),
                user=self.user,
                thread=self.thread,
                question=_trim(neutralise(question)),
                answer=_trim(neutralise(answer)),
                verdict=_trim(neutralise(verdict), limit=120),
            )
        )

    def remember_rating(self, audience: Audience) -> Entry | None:
        """Record the reading level this person asked for.

        Args:
            audience: The level they chose.

        Returns:
            The entry written, or ``None`` if memory is off. No screening: the
            value comes from a closed set rather than from typed text.
        """
        if not self.enabled:
            return None
        return self._append(
            Entry(kind="rating", at=_now(), user=self.user, thread=self.thread, audience=audience)
        )

    def _append(self, entry: Entry) -> Entry | None:
        """Add one line to the log.

        Args:
            entry: The record to write.

        Returns:
            The entry, or ``None`` if the file could not be written. Opened in
            append mode for every write rather than held open, so a crash cannot
            lose a buffer and two processes cannot interleave a partial line.

        Raises:
            Nothing. See the module docstring: a memory that cannot be written
            costs history and never an answer.
        """
        if self.path is None:
            return None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(entry.as_line() + "\n")
        except OSError as error:
            _log.warning(
                "memory_write_failed",
                extra={"path": str(self.path), "error_type": type(error).__name__},
            )
            return None
        return entry

    # -- reading ----------------------------------------------------------

    def recall(self, limit: int = RECALL_TURNS) -> tuple[Entry, ...]:
        """The most recent turns in this conversation, oldest first.

        Args:
            limit: How many to return. The default is the window the prompt is
                sized for; pass more only when something other than a prompt is
                doing the reading.

        Returns:
            Up to ``limit`` turns belonging to this user and thread, in the order
            they happened. Ratings are excluded -- they are a preference, not
            something the model should be shown as conversational history.
        """
        turns = [
            entry
            for entry in self._read()
            if entry.kind == "turn" and entry.user == self.user and entry.thread == self.thread
        ]
        return tuple(turns[-limit:])

    def preferred_audience(self, default: Audience | None = None) -> Audience:
        """The reading level this person last asked for.

        Deliberately not filtered by thread. A reading level is a fact about the
        reader and follows them into the next conversation; a preference that
        reset at a thread boundary would have to be re-stated so often that
        nobody would bother stating it.

        Args:
            default: What to return when nothing has been recorded. Defaults to
                the configured audience.

        Returns:
            The most recent recorded level, or the default.
        """
        fallback = default
        if fallback is None:
            resolved = get_settings() if self.settings is None else self.settings
            fallback = resolved.audience
        for entry in reversed(self._read()):
            if entry.kind != "rating" or entry.user != self.user:
                continue
            level = _as_audience(entry.audience)
            if level is not None:
                return level
        return fallback

    def as_context(self, limit: int = RECALL_TURNS) -> str:
        """Render recent history as the block a prompt can carry.

        The caller is responsible for labelling this as data in the prompt. It is
        neutralised on write and again unchanged on read, but a prompt that does
        not say "this is history, not instruction" is relying on the screen alone,
        and the screen is one layer of a defence rather than the whole of it.

        Args:
            limit: How many turns to include.

        Returns:
            One block per turn, blank-line separated, oldest first -- or the empty
            string when there is nothing to recall, which callers should treat as
            "omit the section" rather than as a section saying "no history".
        """
        return "\n\n".join(entry.summarise() for entry in self.recall(limit))

    def _read_all(self) -> list[Entry]:
        """Every parsable record in the file, with no window applied.

        Separate from :meth:`_read` because the two have opposite requirements and
        sharing one function silently destroyed data. Recall wants the *recent* tail
        and is bounded for latency; a rewrite wants the *whole* file, because
        whatever it does not read it does not write back.

        :meth:`forget` and :meth:`forget_thread` used :meth:`_read`, which returns
        only the last :data:`MAX_LINES_SCANNED` lines. So erasing one user's records
        from a long file deleted every record above the window as well -- other
        users' included -- while reporting only the count it had seen. Reproduced on
        a 2,110-line file: one user pressing *forget* took 110 of another user's
        turns with it and said it had removed 10.

        Returns:
            Every record, oldest first. Unparsable lines are dropped, which is the
            same policy as :meth:`_read` and is safe here for the same reason: a
            line this module cannot parse is a line it cannot honour a deletion
            request about either.
        """
        if not self.enabled or self.path is None or not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            _log.warning(
                "memory_read_failed",
                extra={"path": str(self.path), "error_type": type(error).__name__},
            )
            return []
        parsed = (_parse_line(line) for line in lines)
        return [entry for entry in parsed if entry is not None]

    def _read(self) -> list[Entry]:
        """Load the tail of the log.

        Returns:
            Every readable record in the last :data:`MAX_LINES_SCANNED` lines, in
            file order. A line that does not parse is skipped and logged rather
            than raised on: a file half-written by a crash has one bad line at the
            end, and losing the whole history to it would be a worse outcome than
            losing that turn.
        """
        if not self.enabled or self.path is None or not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            _log.warning(
                "memory_read_failed",
                extra={"path": str(self.path), "error_type": type(error).__name__},
            )
            return []
        entries: list[Entry] = []
        skipped = 0
        for line in lines[-MAX_LINES_SCANNED:]:
            parsed = _parse_line(line)
            if parsed is None:
                skipped += bool(line.strip())
                continue
            entries.append(parsed)
        if skipped:
            _log.warning("memory_lines_skipped", extra={"count": skipped})
        return entries

    # -- erasure ----------------------------------------------------------

    def forget(self) -> int:
        """Delete everything stored about this user, in every thread.

        The one operation that rewrites the file, and it is a method rather than a
        flag for that reason: append-only is the property that makes this store
        trustworthy, and the exception should be visible at the call site. Other
        users' records are preserved, which is why it is a rewrite rather than an
        unlink -- and why it reads the whole file rather than the recall window;
        see :meth:`_read_all`.

        Returns:
            How many records were removed. Zero when memory is off or nothing was
            stored, so a caller can report "nothing to erase" honestly.
        """
        if not self.enabled or self.path is None or not self.path.exists():
            return 0
        # `_read_all`, not `_read`: a rewrite must read every line it is about to
        # write back. See `_read_all` for what using the windowed read cost.
        existing = self._read_all()
        kept = [entry for entry in existing if entry.user != self.user]
        removed = len(existing) - len(kept)
        try:
            self.path.write_text(
                "".join(entry.as_line() + "\n" for entry in kept), encoding="utf-8"
            )
        except OSError as error:
            _log.warning(
                "memory_erase_failed",
                extra={"path": str(self.path), "error_type": type(error).__name__},
            )
            return 0
        _log.info("memory_erased", extra={"user": self.user, "removed": removed})
        return removed

    def forget_thread(self, thread: str) -> int:
        """Delete one conversation of this user's, leaving the others in place.

        The counterpart to :meth:`forget`, which clears everything. Both exist
        because "end this conversation" and "erase this conversation" are different
        requests and were being served by the same button: starting a new chat only
        stops the old one being recalled, so a listing of past chats could only ever
        grow, and a reader who wanted one gone had no way to say so short of erasing
        the lot.

        Scoped to this user by the same rewrite :meth:`forget` uses, so a thread name
        that happens to collide with another user's cannot reach across.

        Args:
            thread: Which conversation to remove. A name nothing is stored under is
                not an error -- it is the ordinary result of pressing delete twice.

        Returns:
            How many records went. Zero when memory is off, the file is absent, or
            that conversation held nothing.
        """
        if not self.enabled or self.path is None or not self.path.exists():
            return 0
        existing = self._read_all()
        kept = [
            entry for entry in existing if not (entry.user == self.user and entry.thread == thread)
        ]
        removed = len(existing) - len(kept)
        if removed == 0:
            return 0
        try:
            self.path.write_text(
                "".join(entry.as_line() + "\n" for entry in kept), encoding="utf-8"
            )
        except OSError as error:
            _log.warning(
                "memory_thread_erase_failed",
                extra={"path": str(self.path), "error_type": type(error).__name__},
            )
            return 0
        # "conversation", not "thread": every LogRecord already carries a "thread"
        # attribute holding the OS thread id, and logging raises rather than let a
        # caller overwrite one of its own fields.
        _log.info(
            "memory_thread_erased",
            extra={"user": self.user, "conversation": thread, "removed": removed},
        )
        return removed

    def forget_all_threads(self) -> int:
        """Delete every conversation of this user's, keeping what was learned.

        Between :meth:`forget_thread` and :meth:`forget`: clearing a listing of
        conversations is a housekeeping request, and erasing a reading level the
        reader taught the agent is not. Only turns go, so :meth:`threads` comes back
        empty while :meth:`preferred_audience` still answers.

        Returns:
            How many records went. Zero when memory is off or nothing was stored.
        """
        if not self.enabled or self.path is None or not self.path.exists():
            return 0
        existing = self._read_all()
        kept = [
            entry for entry in existing if not (entry.user == self.user and entry.kind == "turn")
        ]
        removed = len(existing) - len(kept)
        if removed == 0:
            return 0
        try:
            self.path.write_text(
                "".join(entry.as_line() + "\n" for entry in kept), encoding="utf-8"
            )
        except OSError as error:
            _log.warning(
                "memory_threads_erase_failed",
                extra={"path": str(self.path), "error_type": type(error).__name__},
            )
            return 0
        _log.info("memory_threads_erased", extra={"user": self.user, "removed": removed})
        return removed

    # -- observability -----------------------------------------------------

    def threads(self) -> tuple[Thread, ...]:
        """List this user's conversations, most recently used first.

        A conversation that has scrolled off the screen is not the same thing as a
        conversation that has been deleted, and an interface that offered no way back
        to yesterday's would be quietly discarding work the file still holds. This is
        what makes the difference visible.

        Only this user's threads, never everyone's: the file is shared between users
        and a listing that crossed that boundary would be a privacy failure dressed
        up as a feature.

        Returns:
            One record per conversation, newest first. Empty when memory is off.
        """
        if not self.enabled:
            return ()
        seen: dict[str, list[Entry]] = {}
        for entry in self._read():
            if entry.user == self.user and entry.kind == "turn":
                seen.setdefault(entry.thread, []).append(entry)
        threads = [
            Thread(
                name=name,
                turns=len(entries),
                opened_with=entries[0].question,
                last_at=entries[-1].at,
            )
            for name, entries in seen.items()
        ]
        return tuple(sorted(threads, key=lambda thread: thread.last_at, reverse=True))

    def describe(self) -> dict[str, Any]:
        """Report what this memory holds, in primitives.

        Returns:
            Enough for a status panel to say whether memory is on, where it lives
            and how much of it belongs to the current user. No stored text: a
            status panel is the wrong place to reprint somebody's questions.
        """
        mine = [entry for entry in self._read() if entry.user == self.user]
        return {
            "enabled": self.enabled,
            "path": str(self.path) if self.path is not None else "",
            "user": self.user,
            "thread": self.thread,
            "turns": sum(entry.kind == "turn" for entry in mine),
            "ratings": sum(entry.kind == "rating" for entry in mine),
        }


def _parse_line(line: str) -> Entry | None:
    """Read one line of the log back into a record.

    Args:
        line: One line of the file, without its newline.

    Returns:
        The entry, or ``None`` if the line is blank, is not a JSON object, or
        carries no recognised kind. Unknown keys are ignored rather than refused,
        so a file written by a later version stays readable by this one.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    kind = loaded.get("kind")
    if kind not in ("turn", "rating"):
        return None
    return Entry(
        kind=kind,
        at=str(loaded.get("at", "")),
        user=str(loaded.get("user", ANONYMOUS)),
        thread=str(loaded.get("thread", DEFAULT_THREAD)),
        question=str(loaded.get("question", "")),
        answer=str(loaded.get("answer", "")),
        verdict=str(loaded.get("verdict", "")),
        audience=str(loaded.get("audience", "")),
    )


def _as_audience(value: str) -> Audience | None:
    """Check a stored reading level against the set the application accepts.

    The file is plain text and invites hand-editing, so a value read back from it
    is untrusted in the ordinary sense: not hostile, just possibly a typo or a
    level this version no longer has. Comparing against the type's own members
    both validates and narrows, which is why this is a loop rather than a cast.

    Args:
        value: The stored string.

    Returns:
        The matching level, or ``None`` if it is not one.
    """
    for level in AUDIENCE_VALUES:
        if value == level:
            return level
    return None
