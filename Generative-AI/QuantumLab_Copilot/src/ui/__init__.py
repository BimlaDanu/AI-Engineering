"""Streamlit presentation layer. Rendering only -- no physics, no agent logic.

This package may import from the rest of ``src``; nothing in ``src`` outside it may
import from here. If a computation, a decision or a validation rule is written in
this package, it is in the wrong place: it cannot be unit-tested without spinning up
Streamlit, and it will not survive a change of front end.

One structural consequence is worth naming, because Streamlit re-executes the whole
script on every interaction: **anything that must outlive a refresh cannot live in
``st.session_state``.** This application draws the line in one place. What is on
screen -- the thread being rendered, which answers have been rated, the chat number
-- is session state, and losing it on a refresh costs a reader their scrollback and
nothing else. What must survive is the agent's memory, and that is a store in
:mod:`src.agent.memory` keyed by user and conversation.

Which store, though, is this package's decision rather than the agent's, and it
follows the sign-in: a signed-in visitor gets the file, and everyone else gets
:class:`src.ui.panels.SessionMemory`, which is honest about lasting only as long as
the tab. See :func:`src.ui.panels.session_store`.
"""
