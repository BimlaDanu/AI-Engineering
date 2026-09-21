"""Secondary Streamlit pages, discovered automatically by the multipage router.

The main chat and lab view is the entry point; these pages hold the surfaces
that would crowd it -- the memory inspector, the evaluation scorecard, the run
history, and the developer panel where model, temperature and system prompts
are exposed. Keeping developer controls on their own page is deliberate: the
user of a physics assistant should not have to understand sampling temperature
to use it.
"""
