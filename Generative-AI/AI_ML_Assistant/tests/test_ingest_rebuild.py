"""Offline tests for the rebuild contract in :mod:`src.rag.ingest`.

Re-indexing is the only routine operation that can destroy data. It wipes the collection and
rebuilds it from ``data/``, and the rebuild half is a network call over every chunk — so the
question these tests exist to answer is what is left on disk when that call fails.

``build_vector_store`` is exercised against a temporary ``CHROMA_DIR`` with the embedding
backend and Chroma both patched out: nothing here embeds anything, reaches a network, or
touches the project's real index.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.rag import ingest


@pytest.fixture
def index_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CHROMA_DIR and its staging sibling at a tmp dir holding a 'live' index."""
    live = tmp_path / "chroma_db"
    live.mkdir()
    (live / "chroma.sqlite3").write_text("the index that must survive a failed rebuild")
    monkeypatch.setattr(ingest, "CHROMA_DIR", live)
    monkeypatch.setattr(ingest, "STAGING_DIR", live.with_name("chroma_db.staging"))
    monkeypatch.setattr(ingest, "make_embeddings", lambda: object())
    return live


def _docs() -> list[Document]:
    return [Document(page_content="Gradient descent " * 200, metadata={"source": "ml.md"})]


def _fake_chroma(persist_directory: str, **_: object) -> object:
    """Stand in for Chroma.from_documents: write a marker where it was told to persist."""
    Path(persist_directory, "chroma.sqlite3").write_text("freshly built")
    return object()


def test_a_successful_rebuild_replaces_the_index(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ingest.Chroma, "from_documents", staticmethod(lambda *a, **k: _fake_chroma(**k))
    )
    n = ingest.build_vector_store(_docs())

    assert n > 0
    assert (index_dir / "chroma.sqlite3").read_text() == "freshly built"
    assert not ingest.STAGING_DIR.exists(), "staging must not outlive the rebuild"


def test_a_failed_embedding_leaves_the_live_index_intact(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point. This used to delete the index first and embed second, so an expired key
    # or a rate limit took the knowledge base with it and left nothing to fall back to.
    def _boom(*_a: object, **_k: object) -> None:
        raise ConnectionError("Connection error.")

    monkeypatch.setattr(ingest.Chroma, "from_documents", staticmethod(_boom))

    with pytest.raises(ConnectionError):
        ingest.build_vector_store(_docs())

    assert (index_dir / "chroma.sqlite3").read_text() == (
        "the index that must survive a failed rebuild"
    )
    assert not ingest.STAGING_DIR.exists(), "the half-built attempt must be cleaned up"


def test_an_interrupted_rebuild_also_leaves_it_intact(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # KeyboardInterrupt is a BaseException, so a bare `except Exception` would let a ctrl-C
    # during `make ingest` skip the cleanup and strand a staging directory.
    def _interrupt(*_a: object, **_k: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(ingest.Chroma, "from_documents", staticmethod(_interrupt))

    with pytest.raises(KeyboardInterrupt):
        ingest.build_vector_store(_docs())

    assert (index_dir / "chroma.sqlite3").exists()
    assert not ingest.STAGING_DIR.exists()


def test_indexing_nothing_is_refused_rather_than_emptying_the_index(index_dir: Path) -> None:
    # An empty data/ directory must not be a way to silently throw the index away.
    with pytest.raises(ValueError, match="No text to index"):
        ingest.build_vector_store([])

    assert (index_dir / "chroma.sqlite3").exists()


def test_a_stale_staging_directory_does_not_block_the_next_rebuild(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the process is killed outright, cleanup never runs; the next attempt must clear it
    # rather than build on top of someone else's leftovers.
    ingest.STAGING_DIR.mkdir()
    (ingest.STAGING_DIR / "leftover.txt").write_text("from a killed run")
    monkeypatch.setattr(
        ingest.Chroma, "from_documents", staticmethod(lambda *a, **k: _fake_chroma(**k))
    )

    ingest.build_vector_store(_docs())

    assert not (index_dir / "leftover.txt").exists()
    assert (index_dir / "chroma.sqlite3").read_text() == "freshly built"


# --- The button that calls it -----------------------------------------------------------------

_KB_APP = """
from src.ui.pages.knowledge_base import render
from src.ui.state import init_state

init_state()
render()
"""


def test_the_reindex_button_reports_a_failure_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Re-indexing is the one control on this page that always reaches the network and a
    # billing account, so it is the one most likely to fail — and it used to fail as a red
    # traceback in the middle of the page, against the app's own graded-degradation rule.
    from streamlit.testing.v1 import AppTest

    from src.ui import state
    from src.ui.pages import knowledge_base
    from src.ui.state import KBStatus

    monkeypatch.setattr(state, "load_kb_status", lambda: KBStatus(kb=None))
    monkeypatch.setattr(knowledge_base, "load_kb", lambda: None)
    monkeypatch.setattr(knowledge_base, "load_documents", list)

    def _boom(_docs: object) -> int:
        raise ConnectionError("Connection error.")

    monkeypatch.setattr(knowledge_base, "build_vector_store", _boom)

    at = AppTest.from_string(_KB_APP, default_timeout=60).run()
    reindex = next(b for b in at.button if "Re-index" in b.label)
    reindex.click().run()

    assert not at.exception, "a failed rebuild must not surface as a traceback"
    assert at.error, "a failed rebuild must say so"
    message = at.error[0].value
    assert "unchanged" in message and "ConnectionError" in message


# --- Surviving a rebuild that happened elsewhere -----------------------------------------------


def test_the_cache_key_follows_the_collection_not_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An external ``make ingest`` must invalidate the app's cached KnowledgeBase.

    The app caches the loaded index for the life of the process. A rebuild run from another
    terminal replaces the collection underneath it, and the cached handle then names a
    collection that no longer exists — every question failing with a raw Chroma UUID until
    the server is restarted. Keying on the collection id is what makes that self-correcting.
    """
    import sqlite3

    from src.ui import state

    db = tmp_path / "chroma.sqlite3"

    def _write_collection(uuid: str) -> None:
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS collections (id TEXT, name TEXT)")
            conn.execute("DELETE FROM collections")
            conn.execute("INSERT INTO collections VALUES (?, ?)", (uuid, "ai_ml_kb"))

    monkeypatch.setattr(state, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(state, "COLLECTION_NAME", "ai_ml_kb")

    _write_collection("11111111-1111-1111-1111-111111111111")
    before = state._live_collection_id()

    _write_collection("22222222-2222-2222-2222-222222222222")  # what `make ingest` does
    after = state._live_collection_id()

    assert before != after, "a rebuilt collection must produce a different cache key"


def test_a_missing_or_unreadable_store_yields_no_collection_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mid-rebuild the file can be absent, locked, or half-written; none of those may raise
    # out of the cache key, or the whole app dies on a transient condition.
    from src.ui import state

    monkeypatch.setattr(state, "CHROMA_DIR", tmp_path)
    assert state._live_collection_id() is None  # nothing there at all

    (tmp_path / "chroma.sqlite3").write_bytes(b"not a database")
    assert state._live_collection_id() is None  # unreadable, but no exception
