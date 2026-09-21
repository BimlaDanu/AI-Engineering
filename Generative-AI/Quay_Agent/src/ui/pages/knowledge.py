"""The Knowledge base page: everything the agent has read, published in full.

A retrieval application misleads people most often by refusing a reasonable
question. To the person asking, that reads as *this cannot be answered*, when what
it means is *this is not in my notes*. This page is the correction: the entire
corpus is on screen, shelf by shelf, with the citation for each note, so a refusal
about a subject that is not here stops being mysterious and becomes checkable.

Nothing on this page calls a model or reads a credential. It reads the committed
Markdown files and counts them.
"""

from __future__ import annotations

from src.ui import panels

panels.current_setting()

panels.header(
    "Knowledge base",
    "Every note the agent is allowed to quote from, and nothing it is not. "
    "**There is no hidden corpus** -- anything outside what is listed here is "
    "either computed from the model or refused.",
    "src/rag/ingest.py",
)

panels.render_knowledge()

panels.footer("src/rag/ingest.py")
