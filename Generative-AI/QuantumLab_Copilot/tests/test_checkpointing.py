"""Tests for conversation persistence.

Every case writes to a ``tmp_path``, so the suite never touches the real
checkpoint file and two runs cannot interfere with each other.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.checkpointing import build_checkpointer, open_sqlite_checkpointer, thread_config
from src.settings import Settings


def build(**overrides: object) -> Settings:
    """Construct settings from explicit values, ignoring any ``.env``."""
    values: dict[str, object] = {"openrouter_api_key": "test-key"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


def test_a_configured_path_gives_a_durable_store(tmp_path: Path) -> None:
    store = tmp_path / "state.sqlite"
    checkpointer = build_checkpointer(build(checkpoint_path=str(store)))
    assert isinstance(checkpointer, SqliteSaver)
    assert store.exists()


def test_no_path_keeps_state_in_memory() -> None:
    assert isinstance(build_checkpointer(build(checkpoint_path=None)), InMemorySaver)


def test_a_missing_directory_is_created(tmp_path: Path) -> None:
    # First run on a fresh checkout must not fail on a directory nobody made.
    store = tmp_path / "nested" / "deeper" / "state.sqlite"
    open_sqlite_checkpointer(str(store))
    assert store.exists()


def test_opening_an_existing_store_twice_is_safe(tmp_path: Path) -> None:
    # Streamlit re-executes the script on every interaction; table creation has
    # to be idempotent or the second message crashes.
    store = str(tmp_path / "state.sqlite")
    open_sqlite_checkpointer(store)
    assert isinstance(open_sqlite_checkpointer(store), SqliteSaver)


def test_an_unwritable_location_costs_history_not_the_session(tmp_path: Path) -> None:
    # A read-only disk should degrade the feature, not take down the app.
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    checkpointer = build_checkpointer(build(checkpoint_path=str(blocker / "state.sqlite")))
    assert isinstance(checkpointer, InMemorySaver)


def save(store: SqliteSaver, thread: str) -> None:
    """Write one checkpoint for a thread, as the graph would after a step.

    ``checkpoint_ns`` is added here because a direct write needs it; a running
    graph fills it in itself, which is why :func:`thread_config` -- the thing
    the application actually passes to ``invoke`` -- does not carry it.
    """
    config: RunnableConfig = {
        "configurable": {**thread_config(thread)["configurable"], "checkpoint_ns": ""}
    }
    store.put(config, empty_checkpoint(), {"source": "input", "step": 1}, {})


def test_a_conversation_survives_a_new_connection_to_the_same_file(tmp_path: Path) -> None:
    # The point of persisting: a restart must not erase the conversation.
    store = str(tmp_path / "state.sqlite")
    save(open_sqlite_checkpointer(store), "thread-1")
    reopened = open_sqlite_checkpointer(store)
    assert reopened.get(thread_config("thread-1")) is not None


def test_two_threads_do_not_share_a_conversation(tmp_path: Path) -> None:
    # Two users on the same deployment must not read each other's history.
    store = open_sqlite_checkpointer(str(tmp_path / "state.sqlite"))
    save(store, "alice")
    assert store.get(thread_config("alice")) is not None
    assert store.get(thread_config("bob")) is None


def test_the_thread_config_has_the_shape_langgraph_expects() -> None:
    assert thread_config("abc") == {"configurable": {"thread_id": "abc"}}
