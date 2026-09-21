"""A LangGraph checkpointer, built and configured -- and currently used by nothing.

**Read this before using it.** A checkpointer saves a graph's state after every
step, keyed by a thread id, and this module opens one correctly. The agent does not
compile with it, and the reason is worth recording rather than rediscovering:

*The state is not serialisable.* :class:`src.agent.graph.State` carries
:class:`~src.physics.registry.MethodInfo` values, which hold the solver functions
themselves. A checkpointer would have to pickle a function to save a run.

*There is nothing to resume.* One question is one pass. The cost gate in
:mod:`src.agent.selection` suspends nothing: it returns ``approval_needed`` having
run only the cheap steps, and re-asking with ``approved=True`` runs them again for
nothing, which is the whole resume protocol.

*Conversation memory is a different thing.* What a follow-up question needs is the
previous *answer*, not the previous graph state, and that is
:mod:`src.agent.memory` -- an append-only log that can be read directly, keyed by user and
conversation, with rules about what may be stored. A checkpointer would have given
memory by accident and none of those rules.

So this is a working component of an unused shape. It is kept, documented and tested
because the decision above is a judgement about *this* graph -- make the state
serialisable and it becomes the right answer -- and because a reader who reaches for
a checkpointer should find the argument rather than an absence. If that reader is
you and the argument no longer holds, this module is ready.

SQLite is chosen over the in-memory saver because a restart should not erase a
conversation. The connection is opened with ``check_same_thread=False``: Streamlit
serves reruns from a worker pool, so the thread that resumes a conversation is
rarely the one that started it. That is safe here because SQLite serialises writes
internally and this application has one writer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.logging_setup import get_logger
from src.settings import Settings, get_settings

_log = get_logger("checkpoint")


def build_checkpointer(settings: Settings | None = None) -> BaseCheckpointSaver[str]:
    """Open the conversation store.

    Args:
        settings: Configuration to read. Defaults to the process settings.

    Returns:
        A SQLite-backed saver when a path is configured, otherwise an in-memory
        one. Falling back rather than raising is deliberate: an unwritable disk
        should cost the user their history, not their session.
    """
    resolved = get_settings() if settings is None else settings
    if resolved.checkpoint_path is None:
        return InMemorySaver()
    try:
        return open_sqlite_checkpointer(resolved.checkpoint_path)
    except (OSError, sqlite3.Error) as error:
        _log.warning(
            "checkpoint_fallback",
            extra={"path": resolved.checkpoint_path, "error_type": type(error).__name__},
        )
        return InMemorySaver()


def open_sqlite_checkpointer(path: str) -> SqliteSaver:
    """Open -- and if necessary create -- the SQLite checkpoint store.

    ``SqliteSaver.from_conn_string`` is a context manager that closes the
    connection on exit, which suits a script and not an application: the store
    has to outlive the function that opened it. The connection is therefore
    built directly, which is the documented way to own its lifetime.

    Args:
        path: File to store checkpoints in, relative to the working directory.
            The parent directory is created if it does not exist.

    Returns:
        A saver with its tables set up.

    Raises:
        OSError: If the directory cannot be created.
        sqlite3.Error: If the database cannot be opened.
    """
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(location, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()  # idempotent: creates the tables on first use
    return saver


def thread_config(thread_id: str) -> RunnableConfig:
    """Build the config that ties a request to a stored conversation.

    Args:
        thread_id: Identifier for one conversation. Stable across a session and
            distinct between users -- two people sharing a thread id would share
            a conversation.

    Returns:
        The config LangGraph reads the thread from. Typed as
        :class:`~langchain_core.runnables.RunnableConfig` rather than as a plain
        dict so that a misspelled key is caught by the type checker instead of
        by a conversation that silently starts from nothing.

    Examples:
        >>> thread_config("abc")
        {'configurable': {'thread_id': 'abc'}}
    """
    return {"configurable": {"thread_id": thread_id}}
