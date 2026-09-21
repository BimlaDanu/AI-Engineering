"""A LangGraph checkpointer: built, tested, and wired into nothing.

A checkpointer saves graph state after every step, keyed by thread id. This
module opens one correctly and the agent does not compile with it, because one
question is one pass and no node suspends -- there is nothing to resume.

The state would survive it. A finished campaign of three runs, two rejections
and a verdict serialises under LangGraph's ``JsonPlusSerializer`` to about 17 KB
of msgpack and round-trips; the solver callables live in the configuration (see
:data:`src.agent.graph.BENCH_KEY`) rather than in the state.

Conversation memory is :mod:`src.agent.memory`, not this: a follow-up needs the
previous answer, not the previous graph state, and needs rules about what may be
kept. Wiring this in would be for a node that suspends -- a human gate before the
depth ladder spends its shot budget.
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
