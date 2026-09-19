"""📚 Knowledge Base workspace: chunk statistics, document upload, and re-indexing."""

from __future__ import annotations

import streamlit as st

from src.config import DATA_DIR
from src.rag.ingest import VALID_SUBJECTS, build_vector_store, load_documents
from src.rag.promote import PROMOTED_SUBDIR, is_on_domain_passage, promote_chunks
from src.rag.retriever import RetrievedChunk
from src.ui.registry import register_page
from src.ui.state import clear_kb_cache, load_kb


def _reindex() -> None:
    """Rebuild the vector store from data/ and clear the cached KnowledgeBase.

    The confirmation is a toast rather than an ``st.success``: the rerun below is what makes
    the new chunk counts appear, and anything drawn before a rerun is discarded — so the
    success line was written and immediately thrown away, ending a minute-long rebuild with
    no visible acknowledgement at all. Toasts survive the rerun.

    The failure branch is the reason this is a ``try`` at all. Re-indexing embeds every chunk
    through the configured backend, so it is the one button here that reliably touches the
    network and a billing account; an expired key or a rate limit used to surface as a red
    traceback in the middle of the page, which is the opposite of the graded degradation the
    rest of the app promises. :func:`~src.rag.ingest.build_vector_store` now leaves the live
    index intact when it raises, so the message can honestly say the old one is still there.
    """
    try:
        with st.spinner("Re-indexing (chunking + embedding, may take a minute)…"):
            n_chunks = build_vector_store(load_documents())
    except Exception as exc:
        st.error(
            f"Re-indexing failed, so the existing index is unchanged and still searchable.\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Embedding runs through the configured backend, so the usual causes are a missing "
            "or expired API key, exhausted credit, or no connection. Check ⚙️ Settings, then "
            "try again."
        )
        return
    clear_kb_cache()  # this process rebuilt it, so drop the handle it is still holding
    st.toast(f"Re-indexed: {n_chunks} chunks.", icon="✅")
    st.rerun()


def _docs_on_disk() -> dict[str, list[str]]:
    """List .md/.txt/.pdf filenames per subject folder under data/ (freshness check)."""
    files: dict[str, list[str]] = {}
    for subject in sorted(VALID_SUBJECTS):
        folder = DATA_DIR / subject
        if folder.is_dir():
            files[subject] = sorted(
                p.name for p in folder.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"}
            )
    return files


# Lives in the Knowledge section, last, beside the pages whose material it holds: it is the
# maintenance surface for what 📰 AI News and 📚 Stacks read from — index stats, upload,
# re-index, promote — rather than a place to "do work". ``order`` puts it after both.
@register_page("📄 Knowledge Base", key="knowledge_base", section="Knowledge", order=80)
def render() -> None:
    """Knowledge Base tab: chunk statistics, document upload, and re-indexing."""
    st.subheader("📄 Knowledge Base")
    kb = load_kb()
    if kb is None:
        st.warning(
            "No index found. Add documents under `data/ml/`, `data/dl/`, `data/ai/`, or "
            "`data/overlap/` and re-index below."
        )
    else:
        stats = kb.stats()
        missing = [
            f"`{subject}/{name}`"
            for subject, names in _docs_on_disk().items()
            for name in names
            if name not in stats["by_source"]
        ]
        if missing:
            st.warning(
                f"⚠️ The index is out of date — {len(missing)} document(s) on disk are "
                f"not indexed yet: {', '.join(missing)}. Click "
                "**🔄 Re-index existing documents** below."
            )
        col_subject, col_source = st.columns(2)
        with col_subject:
            st.markdown("**Chunks by subject**")
            st.table(
                {
                    "subject": list(stats["by_subject"]),
                    "chunks": list(stats["by_subject"].values()),
                }
            )
        with col_source:
            st.markdown("**Chunks by document**")
            st.table(
                {
                    "document": list(stats["by_source"]),
                    "chunks": list(stats["by_source"].values()),
                }
            )
    st.divider()
    st.markdown("**Add a document** (PDF, Markdown, or text)")
    subject = st.selectbox(
        "Subject folder",
        sorted(VALID_SUBJECTS),
        help="'overlap' = shared foundations retrieved in every subject view.",
    )
    uploaded = st.file_uploader("Choose a file", type=["pdf", "md", "txt"])
    col_add, col_reindex = st.columns(2)
    if uploaded and col_add.button("Save & re-index", width="stretch"):
        target_dir = DATA_DIR / subject
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / uploaded.name).write_bytes(uploaded.getvalue())
        _reindex()
    if col_reindex.button("🔄 Re-index existing documents", width="stretch"):
        _reindex()

    st.divider()
    _promote_panel(kb)


def _candidate_checkbox(pid: str, cand: dict) -> bool:
    """Render one promotion candidate as a checkbox; return whether it's selected."""
    meta = cand["metadata"]
    title = meta.get("title") or meta.get("source") or pid
    label = f"{title} — {meta.get('source', '')}"
    return st.checkbox(label, key=f"promote_pick_{pid}")


def _promote_panel(kb) -> None:
    """Approve external (arXiv/web) passages gathered this session into the curated KB.

    Corrective-RAG cites external passages for one answer then forgets them; this panel keeps
    the good ones. Approved passages are written as curated notes under ``data/promoted/``
    (so they survive a rebuild and re-index) **and** added to the live index (so they're
    searchable immediately). Human-in-the-loop by design — nothing enters the KB unattended.
    """
    st.markdown("**➕ Promote external passages into the KB**")
    candidates: dict[str, dict] = st.session_state.get("promotable", {})
    if not candidates:
        st.caption(
            "No external passages gathered yet. Ask a question that triggers Corrective-RAG "
            "augmentation (arXiv / web search) and its citable passages will appear here to "
            "review and keep."
        )
        return

    # Split what Corrective-RAG gathered into on-domain (ML/AI) candidates and everything
    # else. arXiv's relevance ranking is noisy, so a physics-adjacent or author query can drag
    # in off-topic papers (e.g. particle physics); those clutter the curated-KB tray without
    # belonging in it. They're not dropped — just tucked into a collapsed "off-topic" section
    # so the default view is on-domain, while a human who disagrees can still promote them.
    on_domain = {
        pid: cand
        for pid, cand in candidates.items()
        if is_on_domain_passage(cand["text"], cand["metadata"])
    }
    off_domain = {pid: cand for pid, cand in candidates.items() if pid not in on_domain}

    st.caption(
        f"{len(candidates)} external passage(s) cited this session"
        + (f" ({len(off_domain)} look off-topic for AI/ML)" if off_domain else "")
        + ". Select the ones worth keeping — they'll be saved as curated notes **and** made "
        "searchable right away."
    )
    subject = st.selectbox(
        "File promoted notes under subject",
        sorted(VALID_SUBJECTS),
        index=sorted(VALID_SUBJECTS).index("overlap"),
        help="'overlap' = shared foundations, retrieved in every subject view (a safe default).",
        key="promote_subject",
    )
    chosen: list[str] = [pid for pid, cand in on_domain.items() if _candidate_checkbox(pid, cand)]
    if not on_domain:
        st.caption("None of this session's external passages look on-topic for AI/ML.")
    if off_domain:
        with st.expander(
            f"⚠️ {len(off_domain)} passage(s) look off-topic for AI/ML — review before keeping"
        ):
            st.caption(
                "These don't read as machine-learning / AI (arXiv relevance ranking is noisy, "
                "so an unrelated paper can slip in). Tick one only if it really belongs in the "
                "knowledge base."
            )
            chosen += [pid for pid, cand in off_domain.items() if _candidate_checkbox(pid, cand)]

    if st.button("➕ Add selected to Knowledge Base", type="primary", width="stretch"):
        if not chosen:
            st.info("Select at least one passage to promote.")
            return
        chunks = [
            RetrievedChunk(
                text=candidates[pid]["text"],
                metadata={**candidates[pid]["metadata"], "subject": subject},
                vector_score=0.0,
                bm25_score=0.0,
                score=0.0,
            )
            for pid in chosen
        ]
        result = promote_chunks(chunks, subject=subject)  # A: persist curated notes
        added = kb.add_chunks(chunks) if kb is not None else 0  # B: live index
        for pid in chosen:  # drop promoted candidates so they don't linger in the panel
            st.session_state.promotable.pop(pid, None)
        note = (
            "; will be searchable after re-indexing" if kb is None else f"; {added} searchable now"
        )
        st.success(
            f"Promoted {result.n_written} note(s) to `data/{PROMOTED_SUBDIR}/`"
            + (f" ({result.n_skipped} already promoted)" if result.n_skipped else "")
            + note
            + ". They fold into the index permanently on the next re-index."
        )
        st.rerun()
