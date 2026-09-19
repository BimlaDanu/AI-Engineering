"""Past conversations: what was asked before, and how to get rid of it.

Synapse used to hold exactly one conversation, in memory, for as long as the browser tab
stayed open. That made two ordinary things impossible — coming back to yesterday's thread,
and *deleting* one — and the second is the one that matters, because a conversation a reader
cannot delete is a conversation they did not agree to keep.

**One JSON file per thread**, in a directory per visitor. The alternative was a single file
holding every thread, and it was rejected on two counts: every save would be a
read-modify-write of the whole file, so two open tabs would clobber each other; and deleting
one thread would mean rewriting the file that holds the rest, which is the worst possible
moment to be rewriting it. Here a delete is an ``unlink`` and cannot damage a neighbour.

This module imports no Streamlit, in line with the rest of ``src/core`` — it takes a root
path and a namespace, and the UI layer (``src.ui.past_chats``) decides what those are.

Scoping, and why it is not optional
-----------------------------------
A deployed Synapse serves everyone from one process and one filesystem. A store keyed only
by path would therefore show every visitor every other visitor's questions, which is a
privacy failure dressed up as a feature. Each visitor gets their own subdirectory, named by
a hash of their identity (see :func:`namespace_for`) so the directory listing on the server
does not itself become a list of who has used the app.

Durability is the host's to provide
-----------------------------------
On Streamlit Community Cloud the container filesystem is **ephemeral**: it survives page
refreshes and reconnections, and it is wiped on reboot or redeploy. That makes this genuinely
useful within a visit and across refreshes, and it means a deployment that wants threads to
outlive a redeploy must point ``SYNAPSE_CHAT_DIR`` at a mounted volume. The UI says so rather
than implying a permanence the host has not been asked for.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ``outputs/`` is already git-ignored, so saved conversations cannot be committed by accident
# and no change to ``.gitignore`` is needed to keep them out. A deployment that wants threads
# to survive a redeploy points SYNAPSE_CHAT_DIR at a mounted volume instead.
DEFAULT_CHAT_DIR = PROJECT_ROOT / "outputs" / "chat_history"

# Thread ids are generated here as uuid4 hex, but they arrive back from widget keys and URLs,
# so they are validated before they are ever joined onto a path. Anything else is refused
# rather than sanitised: a "cleaned" id would silently address a different file.
_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")

TITLE_MAX_CHARS = 60
"""How much of the opening question becomes the thread's label in the sidebar."""


def chat_dir() -> Path:
    """The root under which every visitor's thread directory lives."""
    configured = os.getenv("SYNAPSE_CHAT_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_CHAT_DIR


def namespace_for(identity: str) -> str:
    """Turn a visitor's identity into a directory name that does not reveal it.

    Args:
        identity: An email, username or session id — whatever most specifically names this
            visitor. Empty means anonymous, and the caller should pass a per-session id so
            an anonymous visitor still gets their own directory rather than a shared one.

    Returns:
        A short hex digest. Hashed rather than used directly so that a server-side directory
        listing is not a list of everyone who has used the deployment.
    """
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def title_from(messages: list[dict[str, str]]) -> str:
    """Label a thread by its opening question.

    The first thing the reader typed, rather than a model-written summary: it is what they
    will recognise in a list, it costs no tokens, and it cannot be wrong.
    """
    for message in messages:
        if message.get("role") == "user":
            text = " ".join(str(message.get("content", "")).split())
            if text:
                return text[: TITLE_MAX_CHARS - 1] + "…" if len(text) > TITLE_MAX_CHARS else text
    return "Untitled conversation"


def _serialisable(message: dict) -> dict:
    """Return ``message`` if the whole of it will survive a round trip through JSON.

    Falls back to the two fields every message is guaranteed to have. A page is free to hang
    whatever it likes off a message — an exception object, a dataframe — and the transcript
    should degrade to its text rather than refuse to save the conversation.
    """
    try:
        json.dumps(message)
    except (TypeError, ValueError):
        return {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
    return message


@dataclass(frozen=True, slots=True)
class ChatMeta:
    """One row in the **Past chats** list — enough to draw it without loading the thread.

    Attributes:
        id: The thread's identifier and the stem of its file.
        title: Its opening question, truncated.
        updated: When it was last written, UTC.
        turns: How many messages it holds.
    """

    id: str
    title: str
    updated: datetime
    turns: int

    def label(self) -> str:
        """The button text: the title, with the date it was last touched."""
        return f"{self.title}  ·  {self.updated:%d %b %H:%M}"


@dataclass(frozen=True, slots=True)
class ChatThread:
    """A stored conversation, as it goes to and comes from disk."""

    id: str
    title: str
    created: datetime
    updated: datetime
    messages: list[dict[str, str]]


def new_thread_id() -> str:
    """Mint an identifier for a conversation that is about to start."""
    return uuid.uuid4().hex


def _parse_time(raw: object) -> datetime:
    """Read an ISO timestamp back, falling back to the epoch rather than raising.

    A thread whose timestamp is unreadable should still be listed and still be deletable —
    losing the ability to delete it is a worse outcome than showing the wrong date on it.
    """
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return datetime.fromtimestamp(0, tz=UTC)


class ChatStore:
    """Every saved conversation belonging to one visitor.

    Args:
        root: Where all visitors' directories live; defaults to :func:`chat_dir`.
        namespace: This visitor's directory name, from :func:`namespace_for`.

    Every method degrades rather than raises. The store is a convenience wrapped around a
    directory that may be read-only, full, or absent on a given host, and a chatbot that
    refuses to answer because it could not write a history file would have its priorities
    backwards.
    """

    def __init__(self, namespace: str, root: Path | None = None) -> None:
        self._dir = (root or chat_dir()) / namespace

    @property
    def directory(self) -> Path:
        """This visitor's thread directory. May not exist until something is saved."""
        return self._dir

    @property
    def available(self) -> bool:
        """Whether the directory can be written to, so the UI can say when nothing is kept."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            return os.access(self._dir, os.W_OK)
        except OSError:
            return False

    def _path(self, thread_id: str) -> Path | None:
        """The file backing ``thread_id``, or ``None`` if the id is not one we could have made."""
        return self._dir / f"{thread_id}.json" if _ID_RE.match(thread_id) else None

    def list(self) -> list[ChatMeta]:
        """Every saved thread, most recently updated first.

        A file that cannot be read or parsed is skipped rather than fatal: one corrupt thread
        must not take the whole list — including the delete buttons — down with it.
        """
        try:
            paths = sorted(self._dir.glob("*.json"))
        except OSError:
            return []
        metas: list[ChatMeta] = []
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable chat file %s", path.name)
                continue
            messages = payload.get("messages") or []
            metas.append(
                ChatMeta(
                    id=path.stem,
                    title=str(payload.get("title") or title_from(messages)),
                    updated=_parse_time(payload.get("updated")),
                    turns=len(messages),
                )
            )
        return sorted(metas, key=lambda meta: meta.updated, reverse=True)

    def load(self, thread_id: str) -> ChatThread | None:
        """Read one thread back, or ``None`` if it is missing or unreadable."""
        path = self._path(thread_id)
        if path is None:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        messages = [m for m in payload.get("messages") or [] if isinstance(m, dict)]
        return ChatThread(
            id=thread_id,
            title=str(payload.get("title") or title_from(messages)),
            created=_parse_time(payload.get("created")),
            updated=_parse_time(payload.get("updated")),
            messages=messages,
        )

    def save(self, thread_id: str, messages: list[dict[str, str]]) -> ChatMeta | None:
        """Write a conversation, replacing any earlier version of it.

        Written to a temporary file and then moved into place, so a crash midway through
        leaves the previous version intact rather than a half-written one. An empty
        conversation is not saved at all — an empty thread in the list is a row that does
        nothing but invite a delete.

        Args:
            thread_id: Which thread this is; mint one with :func:`new_thread_id`.
            messages: The conversation, as the dicts the chat page keeps. The whole message
                is stored, not just ``role`` and ``content``, so a reopened thread still
                carries its citations, its tool calls and its per-answer cost — a transcript
                that dropped its sources would undercut the one promise the project makes
                about its answers. Anything in a message that will not serialise is dropped
                back to ``role`` and ``content`` rather than failing the save.

        Returns:
            The row this thread now occupies in the list, or ``None`` if nothing was written.
        """
        path = self._path(thread_id)
        if path is None or not messages:
            return None
        plain = [_serialisable(m) for m in messages if isinstance(m, dict)]
        now = datetime.now(UTC)
        existing = self.load(thread_id)
        title = title_from(plain)
        payload = {
            "title": title,
            "created": (existing.created if existing else now).isoformat(),
            "updated": now.isoformat(),
            "messages": plain,
        }
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(path)  # atomic within one filesystem
        except OSError:
            logger.warning("Could not save conversation %s", thread_id)
            return None
        return ChatMeta(id=thread_id, title=title, updated=now, turns=len(plain))

    def delete(self, thread_id: str) -> bool:
        """Remove one conversation for good.

        Returns:
            Whether a file was actually removed. ``False`` for an unknown id or a failed
            unlink, so the caller can tell the reader it is still there instead of claiming
            a deletion that did not happen.
        """
        path = self._path(thread_id)
        if path is None:
            return False
        try:
            path.unlink()
        except OSError:
            return False
        return True

    def delete_all(self) -> int:
        """Remove every conversation this visitor has stored.

        Only this visitor's directory is touched, never the root: the whole point of the
        namespace is that one person's **Delete all** cannot reach anybody else's threads.

        Returns:
            How many were actually removed, which is what the confirmation should report —
            not how many were listed.
        """
        removed = 0
        try:
            paths = list(self._dir.glob("*.json"))
        except OSError:
            return 0
        for path in paths:
            try:
                path.unlink()
                removed += 1
            except OSError:
                logger.warning("Could not delete chat file %s", path.name)
        return removed
