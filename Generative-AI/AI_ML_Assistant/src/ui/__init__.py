"""Streamlit presentation layer: page registry, theme, sidebar, and workspace pages.

Pages register themselves with :mod:`src.ui.registry`, so a new workspace is added by
dropping a module in ``src/ui/pages`` and decorating its ``render`` function — no edits
to the entry point required.
"""
