"""The Quantum Ising Lab: everything about the chain in the knob, computed twice.

Deliberately a page of its own rather than a panel under the answer. This is the
part of the application no language model can affect -- the numbers, the curves
and the verdict are all produced by the physics layer -- so it is worth being able
to read it without an answer above it, and it is the page that still works when
there is no credential at all.
"""

from __future__ import annotations

from src.ui import panels

setting = panels.current_setting()

# No heading and no standing caption. Both said what the verdict line below says
# better -- that these numbers are computed twice and involve no model -- and the
# page's first screen is worth more as a curve than as a restatement of the sidebar.
panels.render_chain(setting)
