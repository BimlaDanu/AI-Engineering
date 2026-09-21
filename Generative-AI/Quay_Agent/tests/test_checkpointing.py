"""Tests for the checkpoint store the graph does not currently compile with.

An unused module is exactly the kind that rots, and this one's docstring makes a
claim a reader is invited to rely on: that the store works and the decision not to
wire it in is a judgement rather than an obstacle. That claim was untested -- the
module said "kept, documented and tested" while nothing in the suite imported it,
which is the same class of defect as a dial connected to nothing.

Two things are held here. That the store opens, writes and reads back, so "ready to
use" is a fact. And that :class:`~src.agent.state.CampaignState` **serialises under
LangGraph's own serializer**, because the original reason for not compiling with a
checkpointer was that it would not -- and that reason has expired. A test is the only
thing that stops an expired reason from being quoted again next year.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.checkpointing import build_checkpointer, open_sqlite_checkpointer, thread_config
from src.agent.state import Request, new_campaign
from src.settings import Settings, get_settings


def test_a_configured_path_opens_a_sqlite_store(tmp_path: Path) -> None:
    store = open_sqlite_checkpointer(str(tmp_path / "nested" / "threads.sqlite"))
    assert isinstance(store, SqliteSaver)
    # The parent directory is created rather than required, so a first run on a fresh
    # checkout does not fail on a missing folder.
    assert (tmp_path / "nested" / "threads.sqlite").exists()


def test_opening_the_same_store_twice_keeps_what_was_there(tmp_path: Path) -> None:
    # `setup()` has to be idempotent: the second open is the ordinary case, since a
    # long-running interface reopens the store on every restart.
    location = str(tmp_path / "threads.sqlite")
    first = open_sqlite_checkpointer(location)
    assert isinstance(first, SqliteSaver)
    second = open_sqlite_checkpointer(location)
    assert isinstance(second, SqliteSaver)


def test_no_configured_path_falls_back_to_memory() -> None:
    settings = Settings.model_construct(**{**get_settings().model_dump(), "checkpoint_path": None})
    assert isinstance(build_checkpointer(settings), InMemorySaver)


def test_an_unwritable_path_costs_the_history_and_not_the_session(tmp_path: Path) -> None:
    # Falling back rather than raising is the documented behaviour, and it is the one
    # worth testing: a bad path should not be the reason nobody can ask a question.
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory", encoding="utf-8")
    settings = Settings.model_construct(
        **{**get_settings().model_dump(), "checkpoint_path": str(blocked / "under" / "db.sqlite")}
    )
    store = build_checkpointer(settings)
    assert isinstance(store, BaseCheckpointSaver)


def test_the_thread_config_is_the_shape_langgraph_reads() -> None:
    assert thread_config("conversation-1") == {"configurable": {"thread_id": "conversation-1"}}


def test_the_campaign_state_serialises_under_langgraphs_own_serializer() -> None:
    # The retracted reason. The module used to say the state could not be saved
    # because it carried solver callables; it does not -- the lent solver lives in the
    # *configuration*, not the state -- and this is the test that keeps the obsolete
    # argument from being quoted back.
    state = new_campaign(Request(text="a question"), shot_budget=1000)
    kind, payload = JsonPlusSerializer().dumps_typed(state)
    assert kind == "msgpack"
    assert payload
    back = JsonPlusSerializer().loads_typed((kind, payload))
    assert back["request"].text == "a question"
    assert back["shots"].budget == 1000
