"""📈 Analytics: token usage and cost across the current session."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.ui.registry import register_page


def _answer_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract per-answer token/cost rows from chat history for charting."""
    rows: list[dict[str, Any]] = []
    for msg in history:
        meta = msg.get("meta") if msg.get("role") == "assistant" else None
        if not meta:
            continue
        rows.append(
            {
                "Answer": f"#{len(rows) + 1}",
                "Tokens in": meta["tokens_in"],
                "Tokens out": meta["tokens_out"],
                "Cost ($)": round(meta["cost"], 6),
                "Model": meta["model"],
            }
        )
    return rows


@register_page("📈 Analytics", key="analytics", section="Analyse", order=50)
def render() -> None:
    """Analytics tab: session-wide usage metrics and a per-answer token/cost breakdown."""
    st.subheader("📈 Analytics")
    st.caption(
        "Session-wide token usage and cost across every answer so far. "
        "A single question's per-model cost breakdown lives in 🧪 Experiments."
    )

    totals = st.session_state.totals
    col_in, col_out, col_cost = st.columns(3)
    col_in.metric("Total tokens in", f"{totals['input']:,}")
    col_out.metric("Total tokens out", f"{totals['output']:,}")
    col_cost.metric("Est. total cost", f"${totals['cost']:.4f}")

    rows = _answer_rows(st.session_state.history)
    if not rows:
        st.info("No answers yet — ask something in **💬 AI Chat** and the charts fill in here.")
        return

    st.markdown("### Tokens per answer")
    st.bar_chart(
        {
            "Tokens in": [row["Tokens in"] for row in rows],
            "Tokens out": [row["Tokens out"] for row in rows],
        }
    )

    st.markdown("### Cumulative cost ($)")
    running = 0.0
    cumulative: list[float] = []
    for row in rows:
        running += row["Cost ($)"]
        cumulative.append(round(running, 6))
    st.line_chart({"Cumulative cost ($)": cumulative})

    st.markdown("### Per-answer detail")
    st.dataframe(rows, width="stretch", hide_index=True)
