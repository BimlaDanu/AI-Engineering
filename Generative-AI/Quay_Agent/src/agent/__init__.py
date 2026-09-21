"""The agent: the campaign loop, its belief state, and the policy it applies.

Read `graph.py` first -- it is the map, and everything else here is something one
of its seventeen nodes calls.

======================  =====================================================
Module                  What it decides
======================  =====================================================
``graph.py``            the order things happen in, and what happens next
``state.py``            what a campaign knows and what it has spent
``router.py``           whether to search at all, and which shelf first
``grading.py``          whether a passage earned its place in the prompt
``prompts.py``          every instruction sent to a model, and its examples
``diagnosis.py``        what a finished optimisation actually did
``verdict.py``          go, no, or conditional -- by rule, not by persuasion
``report.py``           how the outcome is written up for a non-specialist
``tools.py``            what a language model is allowed to call
``model_selection.py``  which model serves which call, and what that costs
``middleware.py``       the layers every model call runs inside
``llm.py``              how a model is called, and what happens when it fails
``tracing.py``          whether runs are reported to an external service
``usage.py``            what the model calls cost
``memory.py``           what is recalled between questions, and what may be kept
``checkpointing.py``    a state store, built and tested, deliberately unwired
``mathmarkup.py``       making the mathematics in an answer render
======================  =====================================================

Facts live in :mod:`src.physics` -- which methods exist, when each applies, what
each costs. Policy lives here: which to prefer, when to stop, what the result
means.

Two import rules hold. Nothing here can reach an exact answer, which sits behind
a wall this package cannot cross, and nothing here imports :mod:`src.ui`.

A run leaves three records: the state's ``notes`` for a person, structured log
lines for whatever aggregates them, and :func:`src.agent.state.snapshot` for
anything that never imports this package.
"""
