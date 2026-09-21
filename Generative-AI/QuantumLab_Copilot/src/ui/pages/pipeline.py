"""The Under the hood page: how an answer is produced.

The diagram is drawn by LangGraph from the compiled graph, and the nodes the last
question travelled are lit up. That is worth a page rather than a paragraph: the
claim that this is an agent rather than a pipeline is exactly the claim that
different questions take different paths, and here that is visible instead of
asserted.
"""

from __future__ import annotations

import streamlit as st

from src.agent import graph
from src.ui import panels

panels.current_setting()

st.markdown("#### Pipeline trace")

answer: graph.Answer | None = panels.last_answer()
panels.render_pipeline(answer)

st.divider()
st.markdown("**What the model may ask for, and what it may not**")
st.markdown(
    "The graph decides *whether* work happens. Function calling decides *what else* "
    "would help, and it happens after the work rather than before it, so the model is "
    "choosing with the numbers already in front of it.\n\n"
    "| Tool | Reaches | Verified? |\n"
    "|---|---|---|\n"
    "| `CompareChain` | the same physics at another length | yes, cross-checked |\n"
    "| `SweepField` | the same physics across the field range | yes, every point |\n"
    "| `FindPapers` | arXiv, live | **no** -- further reading only |\n\n"
    "A tool call is a *request*: the name is looked up, the arguments are validated "
    "against a schema, and only then does tested Python run. An invented name and "
    "arguments of the wrong shape both come back as a recorded refusal. Neither can "
    "widen a cost gate -- a chain long enough to need your approval is declined inside "
    "a tool call, because there is nobody there to ask."
)

if answer is not None:
    st.divider()
    st.markdown("**The last run, quoted rather than composed**")
    st.code(answer.justification(), language="text")
