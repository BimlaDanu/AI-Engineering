"""🏠 Home workspace: the landing dashboard — status at a glance and jump-in links."""

from __future__ import annotations

import streamlit as st

from src.ui.registry import get_sections, go_to_page, register_page
from src.ui.state import kb_status_notice, load_kb
from src.ui.theme import NAV_TITLE, render_hero

# One-line pitch per workspace, keyed by its registry ``key`` (not its label, which carries an
# emoji and can be re-iconed). The card *labels and ordering* come straight from the page
# registry via :func:`get_sections`, so the Home page can never drift out of step with the
# sidebar again — only these blurbs are hand-authored. A key with no blurb still renders (with
# an empty body), which surfaces the omission rather than hiding a page.
_BLURBS: dict[str, str] = {
    "chat": "Ask grounded ML/AI questions with cited sources and tools.",
    "ml_tutor": "Guided, level-aware learning paths through AI/ML topics.",
    "ml_lab": "Hands-on ML/DL/NLP recipes, live AI & LLM exercises, and a code cell.",
    "quiz": "Test yourself with knowledge-base-grounded questions and instant, cited feedback.",
    "research": "Curated crash courses and the docs behind this app.",
    "ai_news": "Fresh, relevant papers straight from arXiv.",
    "knowledge_base": "Inspect, upload, and re-index the knowledge base.",
    "analytics": "Token usage and cost across your session.",
    "evaluation": "RAGAs-style quality metrics on a golden set.",
    "ab_testing": "Compare two RAG strategies head-to-head.",
    "experiments": "Trace the RAG pipeline and run the tool playground.",
    "settings": "Model, generation parameters, and RAG tuning.",
}


@register_page("🏠 Home", key="home", section=NAV_TITLE, order=10)
def render() -> None:
    """Landing dashboard: KB/session status metrics plus jump-in cards to each workspace."""
    # The gradient banner lives here rather than in the entry point, so it introduces the app
    # on the page a visitor lands on instead of repeating above every workspace they open next.
    render_hero()

    kb = load_kb()
    totals = st.session_state.totals
    col_docs, col_chunks, col_msgs, col_cost = st.columns(4)
    if kb is None:
        col_docs.metric("Documents", "—")
        col_chunks.metric("Chunks", "not indexed")
    else:
        col_docs.metric("Documents", len(kb.stats()["by_source"]))
        col_chunks.metric("Chunks", kb.size)
    col_msgs.metric("Messages this session", len(st.session_state.history))
    col_cost.metric("Est. cost", f"${totals['cost']:.4f}")

    kb_status_notice(
        "The knowledge base isn't indexed yet. Open **📄 Knowledge Base** to add "
        "documents and build the index, or run `make ingest` in a terminal."
    )

    st.divider()

    # Mirror the sidebar: one sub-heading per section, cards in the section's registry order.
    for section, pages in get_sections().items():
        cards = [p for p in pages if p.key != "home"]  # the landing page needn't link to itself
        if not cards:
            continue
        st.markdown(f"#### {section}")
        cols = st.columns(2)
        for i, page in enumerate(cards):
            with cols[i % 2], st.container(border=True):
                # A real button, not a card styled to look like one. These were eleven inert
                # divs with a hover-less border: they advertised a click and swallowed it,
                # which is worse than a plain list, because a list does not make the promise.
                # The registry already resolves a key to a page, so the jump is one call.
                if st.button(page.label, key=f"home_go_{page.key}", width="stretch"):
                    go_to_page(page.key, f"Open **{page.label}** from the sidebar.")
                st.caption(_BLURBS.get(page.key, ""))
