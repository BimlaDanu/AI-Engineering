"""The Analytics page: what this session did, and what it cost.

A page rather than a sidebar expander, because two of the four things on it are
charts and a chart in a 300-pixel column is a decoration. It sits beside the
Knowledge base under *Explore* for a reason: both answer "what is actually going on
in here", one about the corpus and one about the run.

Everything here is derived from the answers already on screen -- token counts from
:class:`src.agent.usage.Usage`, routes from each answer's own routing record, nodes
from :func:`src.agent.graph.executed_nodes`. Nothing is accumulated in a separate
counter, so nothing here can drift out of step with the conversation it describes.
"""

from __future__ import annotations

import streamlit as st

from src.ui import panels

panels.current_setting()

st.markdown("#### Analytics")
st.caption(
    "Token usage and cost for this session, and the route each question took. "
    "The two right-hand bars are the agentic claim made measurable: the pipeline "
    "runs different nodes for different questions."
)

panels.render_analytics()
