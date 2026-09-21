"""Entry point: page setup, the sidebar, the navigation.

Everything that draws lives in :mod:`src.ui.panels` and the pages under
:mod:`src.ui.pages` are a few lines each. This file decides only what the
application is: one shared window, one shared sidebar, and the page list.

The grouping is one distinction repeated. *Explore* is evidence about the
problem -- the Quay Lab, the Knowledge base that publishes the whole corpus so a
refusal does not look like a failure, and Physics and hardware, which recomputes
the physics three ways and holds three real wiring diagrams against the circuit
with no agent involved. Those last two are one page because pricing a circuit
that prepares the wrong Hamiltonian answers nobody's question.

*Under the hood* is evidence about the agent: the Pipeline trace, Evaluations,
and Analytics. Analytics sits here rather than in Explore because four of its
five tabs are language-model instrumentation. *Ask* is the default page.

:data:`src.ui.panels.PAGES` is the one place the list is written down.

The settings knob is drawn here and read everywhere, and its tabs are kept
visibly apart: Physics changes what is computed, Language model changes only how
it is described, Knowledge search changes what the answer stands on. That first
separation is the project's central claim.

Rendering a page calls no model and reads no credential -- ``make run`` works on
a checkout with no ``.env``. A model is reached only when somebody presses Ask,
and the campaign answers without one if there is none to be had.
"""

from __future__ import annotations

import streamlit as st

from src.ui import panels

panels.open_page()

# The sidebar's whole visible surface, and it is deliberately this short: the
# navigation Streamlit draws, then the settings knob and the memory panel, both
# collapsed. What was here and is not, in the order it went: a page index, which
# duplicated the navigation; a build-progress bar, which is a fact about the
# repository rather than about the answer, and now lives on the page about the
# repository; a table of terminal commands, which is documentation wearing a
# control's clothing; the elevator pitch, four lines repeated on nine pages to
# answer a question asked on one; the account block, which was the third place the
# same guest sentence appeared; and the identity caption, which the navigation's own
# "Quay" heading sat directly above and which the Ask page now carries under the
# name, once rather than nine times. A sidebar item that is merely true is not
# thereby useful -- and `panels.sidebar()` went with the last of them, because a
# function whose body had become one comment was a call that read as though
# something still happened there.
#
# Drawn once, here, so the sidebar is identical on every page and a page cannot
# answer with a position the visitor can no longer see. Pages read it back through
# `panels.current_setting()`; none of them writes it.
st.session_state[panels.SETTING_KEY] = panels.read_settings_knob()

# Drawn here rather than by the chat page, so that a reader who clicks away does
# not lose the conversation, the list of past ones, and -- the part that matters --
# the panel saying what the agent is holding about them and offering to clear it. A
# memory inspectable on one page out of seven is not meaningfully inspectable.
panels.conversation_controls(panels.conversation(), panels.open_conversation)

panels.navigation().run()
