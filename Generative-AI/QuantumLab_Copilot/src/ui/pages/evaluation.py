"""The Evaluation page: how the agent scores on the held-out suite.

The published result of ``make evals``, not a live run. Rendering it costs nothing
and says nothing new between commits, which is the correct behaviour for a
measurement: a page that re-scored the agent on every visit would report a different
number every time and would bill whoever opened the tab.

What makes the suite worth reading is that nothing in it is graded by a model.
Every case is a comparison -- a status, a path through the graph, a number against
arithmetic -- so the same code scores the same way on any machine, with or without a
credential. See :mod:`src.evals.cases` for what each family of cases defends and
:mod:`src.evals.tracking` for how a credentialed run is also filed in LangSmith.
"""

from __future__ import annotations

import streamlit as st

from src.ui import panels

panels.current_setting()

st.markdown("#### Evaluation")
st.caption(
    "A frozen suite of questions with known outcomes: which route each must take, "
    "which must be refused, which numbers must be corroborated, and two limits whose "
    "energy is known in closed form. Regenerate it with `make evals`."
)

panels.render_evaluation()
