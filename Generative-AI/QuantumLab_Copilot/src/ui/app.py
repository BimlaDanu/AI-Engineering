"""Entry point for QuantumLab Copilot: page setup, the sidebar, the navigation.

Everything that draws anything lives in :mod:`src.ui.panels`; the pages under
:mod:`src.ui.pages` are a few lines each. This file only decides what the
application *is* -- one shared header, one shared settings knob, six pages.

**Six pages, because there are six questions a visitor arrives with.** *What is the
answer?* is the chat. *Is it right?* is the Quantum Ising Lab, which computes the
same physics with no model involved at all. *What does it know?* is the Knowledge
base, where the corpus is published so a refusal about something outside it stops
looking like a failure. *What did it cost?* is Analytics, which also counts which
routes and nodes the session actually used. *How did it decide?* is the Pipeline
trace, where LangGraph draws its own graph with the last question's path lit up. And
*how well does it do?* is Evaluation, which publishes the held-out scorecard.

**Rendering the page calls no model and reads no credential** -- a tested
invariant, so ``make run`` works on a checkout with no ``.env`` at all. A model is
reached only when somebody actually asks a question, and even then
:func:`src.agent.graph.ask` answers without one if there is none to be had.
"""

from __future__ import annotations

import streamlit as st

from src.physics.model import HAMILTONIAN_INLINE
from src.ui import identity, panels

panels.open_page("QuantumLab Copilot")

# Before anything else is drawn, and only when a deployment configured one. A gate
# that renders the application behind it has not gated anything.
if not identity.gate():
    st.stop()

st.title("QuantumLab Copilot")
st.caption(
    "Ground states of the 1D transverse-field Ising model, "
    f"{HAMILTONIAN_INLINE} — "
    "**every number is checked against an independent method before it is shown.**"
)

# Resolved before anything reads it, and it draws nothing: the thread it derives is
# what memory is keyed by, and the register learned from that thread's ratings is
# the settings knob's starting position, so the knob cannot be built before the
# identity that configures it. What the visitor *sees* of their account is the top
# bar below, which has to come after the title -- hence a function that answers the
# question and a separate one that draws it.
st.session_state["identity"] = identity.current_identity()

# The knob is drawn here, once, so the sidebar is identical on every page. Pages
# read the result through `panels.current_setting`.
remembered = panels.remembered_preference()
st.session_state["setting"] = panels.read_settings_knob(remembered.audience)

# Which conversations there have been, what the agent remembers of them, then what
# the asking has cost. All three are session-wide, which is why they belong to the
# sidebar rather than to any one page -- and the first two read the same store at the
# same scope, so they are neighbours.
with st.sidebar:
    panels.render_past_chats()
    panels.render_memory(remembered)
    panels.render_session_cost()

# Emoji rather than Material glyphs, for one reason: they carry colour. A column of
# monochrome outline icons reads as decoration, while six distinct symbols are
# scannable -- the reader learns "the atom one is where the curves are" and stops
# reading the labels. Each is a noun from the page it names, never a mascot.
#
# Grouped by what a visitor came for, not by how the code is organised. Analytics sits
# with the Knowledge base because both answer "what is going on in here"; Evaluation
# sits with the Pipeline trace because both are evidence about the machinery.
# Held in a name because the top bar below has to ask whether it is the page about
# to run, and identity is the only reliable way to ask. `navigation.title` was the
# obvious way and it is not safe: outside a script run -- importing this module, as
# the doctest sweep does -- Streamlit hands back a page whose title has not been
# resolved and reading it raises.
chat = st.Page("pages/ask.py", title="Chat", icon="💬", default=True)

navigation = st.navigation(
    {
        "Chat": [chat],
        "Explore": [
            st.Page("pages/chain.py", title="Quantum Ising Lab", icon="⚛️"),
            st.Page("pages/knowledge.py", title="Knowledge base", icon="📚"),
            st.Page("pages/analytics.py", title="Analytics", icon="📊"),
        ],
        "Under the hood": [
            st.Page("pages/pipeline.py", title="Pipeline trace", icon="🧭"),
            st.Page("pages/evaluation.py", title="Evaluation", icon="✅"),
        ],
    }
)
# One row, drawn once, for every page: New chat (on the page it acts on), Log in,
# Sign up for free, Use a key. `panels.TOP_BAR_CSS` lifts it onto Streamlit's header
# strip on a window wide enough to share one; on a narrower window it stays here, in
# the page, under the title -- which is why it is drawn after the title rather than
# before it. A page whose first line is somebody else's account controls has given
# its most valuable line away.
#
# `New chat` is passed as the row's lead only on the chat page, because that is the
# only page whose thread it clears. Drawn everywhere, it would be a control that
# quietly acts on a page the reader is not looking at. The page is known here and
# nowhere else: `st.navigation` has already decided which one runs, and it has not
# run it yet.
identity.account_bar(lead=panels.render_new_chat if navigation is chat else None)

navigation.run()
