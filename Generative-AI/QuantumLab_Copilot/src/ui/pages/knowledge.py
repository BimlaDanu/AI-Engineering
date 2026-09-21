"""The Knowledge base page: what the agent has read, published.

A retrieval application misleads people most often by refusing a reasonable
question, which reads as "this cannot be answered" when it means "this is not in
my notes". Showing the corpus turns that into something a visitor can check.
"""

from __future__ import annotations

import streamlit as st

from src.ui import panels

panels.current_setting()

st.markdown("#### What the agent has read")
# No count in this sentence, deliberately. It said "Four notes" while the corpus
# held twelve -- a stale number on the one page whose claim is that nothing is
# hidden. The counts below are read from the corpus itself and cannot drift.
st.caption(
    "Every note carries a citation precise enough to check against the original. "
    "This is the whole knowledge base -- there is no hidden corpus, and anything "
    "outside it is either computed or refused."
)

panels.render_knowledge()
