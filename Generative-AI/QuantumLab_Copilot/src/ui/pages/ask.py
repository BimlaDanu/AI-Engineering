"""The chat page: one thread, and each answer in three layers.

Starter questions are offered at several levels, because the same application has
to serve someone who has never met the model and someone who works with it. The
beginner's question and the practitioner's question go through exactly the same
pipeline -- the difference is only what gets asked, so nothing here special-cases
an audience.

Two places carry settings and they are not redundant. The sidebar configures the
session; the expander here holds the handful of dials worth reaching for in the
middle of a conversation, and it says which of the two is currently winning.
"""

from __future__ import annotations

import streamlit as st

from src.ui import identity, panels
from src.ui.starters import STARTERS

# "New chat" was here, in a column beside this heading, and is now the first
# control in the top bar the entry point draws -- so the row a visitor looks at for
# "whose session is this?" is the same row that offers to end it. The column pair
# went with it: one heading does not need a layout.
st.markdown("#### Chat")

setting = panels.chat_settings(panels.current_setting())

who = st.session_state.get("identity")
greeting = who.name if isinstance(who, identity.Identity) else ""

panels.ask_panel(setting, greeting=greeting)

st.divider()
st.caption("Or start from one of these — the same pipeline answers all of them.")

# Three to a row rather than one long row. The labels are the questions themselves,
# 47 to 87 characters of them, and six columns across one row would leave each about
# 210px wide -- the longest would wrap to five lines and the row would come out as a
# ragged wall of text. Shortening the labels was the other option and it is worse:
# a button that reads "Magnetisation" and sends a different sentence is a small
# lie, and the point of showing the whole question is that clicking it asks
# exactly that.
STARTERS_PER_ROW = 3
questions = list(STARTERS.values())
for start in range(0, len(questions), STARTERS_PER_ROW):
    # Always the full width's worth of columns, so a short last row keeps the same
    # button width as the row above instead of stretching to fill.
    columns = st.columns(STARTERS_PER_ROW)
    for column, question in zip(columns, questions[start:], strict=False):
        column.button(
            question,
            width="stretch",
            key=f"starter-{question}",
            on_click=lambda text=question: st.session_state.__setitem__("starter", text),
        )
