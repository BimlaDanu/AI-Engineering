"""Session-state initialisation and cached, UI-facing access to the knowledge base."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import streamlit as st

from src.config import CHROMA_DIR, COLLECTION_NAME, OPENROUTER_MODELS, RagSettings
from src.core.service import AssistantService
from src.rag.retriever import KnowledgeBase


@dataclass(frozen=True)
class KBStatus:
    """Why the knowledge base is (un)available — so the UI can advise the right fix.

    An *empty* index (``kb is None``, ``error is None``) is fixed by ``make ingest``; a *load
    failure* (``error`` set) is a different problem — a corrupt store, an unreadable
    ``chroma_db/``, or an embedding backend/dimension mismatch — and needs its own message
    rather than the misleading "run make ingest" that a bare ``None`` would trigger.
    """

    kb: KnowledgeBase | None
    error: str | None = None

    @property
    def failed(self) -> bool:
        """True when loading raised (as opposed to loading fine but being empty)."""
        return self.error is not None


def _live_collection_id() -> str | None:
    """The persisted collection's UUID, read straight from Chroma's SQLite catalogue.

    Cheap (one indexed row, read-only, no embedding backend) and it changes every time the
    collection is rebuilt, which is exactly what the cache below needs to notice.
    """
    db = CHROMA_DIR / "chroma.sqlite3"
    if not db.exists():
        return None
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT id FROM collections WHERE name = ?", (COLLECTION_NAME,)
            ).fetchone()
    except sqlite3.Error:
        # Mid-rebuild the file can be locked or half-written. Treat that as "no readable
        # collection" and let KnowledgeBase() report the real problem a moment later.
        return None
    return row[0] if row else None


@st.cache_resource(show_spinner="Loading knowledge base…")
def _kb_status_for(collection_id: str | None) -> KBStatus:
    """Load the index identified by ``collection_id``, keeping *why* it is unavailable.

    Returns a :class:`KBStatus` distinguishing the two failure modes the caller must advise
    on differently: an empty index (``make ingest``) versus a load error (surface the cause).

    ``collection_id`` is not used in the body — it is the cache key. See
    :func:`load_kb_status`.
    """
    try:
        kb = KnowledgeBase()
    except Exception as exc:  # corrupt store, unreadable dir, embedding/dim mismatch
        return KBStatus(kb=None, error=str(exc))
    return KBStatus(kb=kb if kb.size else None)


def load_kb_status() -> KBStatus:
    """The loaded index, cached per *collection*, not merely per process.

    Keying the cache on the collection's UUID is what makes ``make ingest`` safe to run
    against a live app. A rebuild writes a brand-new collection with a new UUID, so the key
    changes and the next run loads the new index. Cached on the process alone — as this was —
    the app kept a handle to the collection that had just been deleted underneath it, and
    every question failed with ``Collection [<uuid>] does not exist`` until someone restarted
    the server. That is a bad trap to leave lying around, because ``make doctor`` recommends
    ``make ingest`` as the fix for an empty index without mentioning that it breaks a running
    app; the recovery procedure was itself the thing that broke.
    """
    return _kb_status_for(_live_collection_id())


def clear_kb_cache() -> None:
    """Forget the loaded index, so the next :func:`load_kb_status` reads it afresh.

    Only needed for a rebuild done *in* this process, which reuses the id lookup before the
    swap lands; an external ``make ingest`` invalidates itself through the cache key.
    """
    _kb_status_for.clear()


def load_kb() -> KnowledgeBase | None:
    """The loaded knowledge base, or None if missing/empty/failed (see :func:`load_kb_status`)."""
    return load_kb_status().kb


def kb_status_notice(empty_message: str) -> None:
    """Render the right unavailable-KB notice: a load-error banner, or ``empty_message``.

    Shows nothing when the KB is present. On a load failure it surfaces the underlying error
    (so a dimension mismatch or unreadable store is not silently reported as "empty"); on a
    genuinely empty index it shows the caller's context-specific ``empty_message``.
    """
    status = load_kb_status()
    if status.failed:
        st.error(
            f"⚠️ The knowledge base failed to load: {status.error}. If you changed the "
            "embedding backend or model, re-run `make ingest` to rebuild the index with the "
            "current embeddings; otherwise check that `chroma_db/` exists and is readable."
        )
    elif status.kb is None:
        st.info(empty_message)


def get_service() -> AssistantService:
    """Build the assistant service over the currently loaded knowledge base."""
    return AssistantService(load_kb())


def init_state() -> None:
    """Create all st.session_state keys the app relies on (idempotent).

    These defaults double as the initial values of the sidebar/chat widgets that bind to
    them by ``key=`` — so no widget passes an explicit ``value``/``index`` (which would
    clash with the session-state binding).
    """
    ss = st.session_state
    ss.setdefault("user", None)  # signed-in src.auth.User when the optional login gate is active
    ss.setdefault("history", [])
    ss.setdefault("settings", RagSettings())
    ss.setdefault("trace", None)
    ss.setdefault("totals", {"input": 0, "output": 0, "cost": 0.0})
    ss.setdefault("pending", None)  # (question, level) queued by buttons
    ss.setdefault("lab_lesson", "")  # last AI/ML Tutor lesson, seeds the 🔬 AI/ML Lab
    ss.setdefault("lab_practice", None)  # Practice target queued by a Tutor "Practise in Lab" click
    ss.setdefault("lab_from", None)  # {track, title} breadcrumb of a Tutor→Lab "Practise" jump
    ss.setdefault("quiz", None)  # the active 📝 Knowledge Check quiz (src.quiz.Quiz)
    ss.setdefault("quiz_result", None)  # grading of the active quiz, once submitted
    ss.setdefault("arxiv_results", None)
    ss.setdefault("ctx", {})  # per-run selections shared with pages
    # Answer-shaping selections, now bound to widgets in the Chat popover / sidebar.
    ss.setdefault("model", OPENROUTER_MODELS[0])
    ss.setdefault("subject_label", "All subjects")
    ss.setdefault("level", "Beginner")
    ss.setdefault("technique", "Standard")
    ss.setdefault("response_length", "Balanced")
    ss.setdefault("extra_instructions", "")
