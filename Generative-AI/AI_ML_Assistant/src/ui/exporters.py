"""Shared Streamlit export controls: JSON / CSV / PDF download buttons for a conversation.

Kept in one place so the ⚙️ Settings page and the 💬 Chat page render identical export
actions without duplicating logic. The heavy lifting lives in :mod:`src.export` (pure, no
Streamlit); this module only wires the results into ``st.download_button`` widgets and
handles the "PDF dependency not synced yet" case gracefully.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.export import conversation_to_csv, conversation_to_json, conversation_to_pdf


def export_controls(history: list[dict[str, Any]], *, key_prefix: str) -> None:
    """Render JSON, CSV, and PDF download buttons for ``history``.

    Args:
        history: The conversation messages (``st.session_state.history``).
        key_prefix: Unique per call site — Streamlit widget keys must not collide across
            the two pages that both render these controls.
    """
    if not history:
        st.caption("Ask something first — there's nothing to export yet.")
        return

    col_json, col_csv, col_pdf = st.columns(3)
    col_json.download_button(
        "⬇️ JSON",
        conversation_to_json(history),
        file_name="conversation.json",
        mime="application/json",
        width="stretch",
        key=f"{key_prefix}_json",
        help="Full-fidelity export of every field.",
    )
    col_csv.download_button(
        "⬇️ CSV",
        conversation_to_csv(history),
        file_name="conversation.csv",
        mime="text/csv",
        width="stretch",
        key=f"{key_prefix}_csv",
        help="One row per turn with model, tokens, cost, and sources.",
    )
    try:
        pdf_bytes = conversation_to_pdf(history)
    except RuntimeError as exc:  # fpdf2 not installed yet
        col_pdf.button(
            "⬇️ PDF",
            disabled=True,
            width="stretch",
            key=f"{key_prefix}_pdf_disabled",
            help=str(exc),
        )
    else:
        col_pdf.download_button(
            "⬇️ PDF",
            pdf_bytes,
            file_name="conversation.pdf",
            mime="application/pdf",
            width="stretch",
            key=f"{key_prefix}_pdf",
            help="A formatted transcript with per-answer metadata and sources.",
        )
