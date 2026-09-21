"""The presentation layer. Rendering only -- no physics, no agent logic.

This package may import from the rest of ``src``; nothing in ``src`` outside it may
import from here, and the dependency test enforces that. If a computation, a
decision or a validation rule ends up in this package it is in the wrong place: it
cannot be exercised without starting Streamlit, and it will not survive a change of
front end.

Where the work is
-----------------

:mod:`src.ui.panels` holds everything that draws, and the pages under
:mod:`src.ui.pages` are a few lines each that compose it. That inversion is
deliberate. A page in Streamlit runs on import, so anything written inside one is
unreachable to a test; anything written in the toolkit beside it can be called
directly. The rule that follows is short: **a page decides what appears and in what
order, and nothing else.**

:mod:`src.ui.status` is the other half -- the facts the pages report, read from the
filesystem and from installed package metadata, with no Streamlit anywhere in it. A
dashboard that computes its own numbers inline cannot be tested, and an untested
dashboard is the kind of thing that keeps reporting "healthy" long after the thing
it watches has stopped working.

Two invariants worth stating
----------------------------

Rendering a page calls no model and reads no credential. ``make run`` works on a
checkout with no ``.env`` at all. A model is reached only when somebody actually
asks a question, and even then the campaign answers without one if there is none to
be had -- offline is a supported mode rather than a degraded one, because every
decision the agent makes that matters is arithmetic.

Anything that must outlive a browser refresh cannot live in ``st.session_state``.
Streamlit re-executes the whole script on every interaction. What is on screen --
the thread being rendered, which tab is open, the last campaign -- is session state,
and losing it costs a reader their scrollback and nothing else. What must survive is
the agent's memory, and that is a file in :mod:`src.agent.memory` keyed by user and
conversation.
"""
