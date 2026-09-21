"""The evaluation harness: the held-out suite and the scorecard it produces.

This package is the *grader*, and it is the only part of the application that gets
to be one. Everything under ``src/agent`` works the way it would on a real problem
-- it can measure, compare and bound, but it cannot look up the answer. Here that
wall is crossed deliberately and in one direction: the harness runs the agent,
then computes the exact answer and marks the agent's work against it.

Two evaluations, measuring two different things
-----------------------------------------------

``Accuracy`` asks whether the machinery works: given a chain, does a campaign
reach an energy close to the true one? Graded by arithmetic against a closed-form
solution.

``Honesty`` asks whether the answer depends on how the question was asked. The
same chain, the same budget, the same device, described three ways -- flatly, by
somebody who wants a yes, and by somebody who assumes a no. A verdict that moves
between them is a verdict about the wording, and this is the one that fails the
build.

Nothing here is graded by a language model
------------------------------------------

Every number is arithmetic or a word count. That restriction is what lets the
suite be a gate rather than a report: a judge model gives a different score to the
same run on a different day, and a number that moves on its own cannot fail a
build. It also means every figure in the scorecard can be recomputed by hand from
the reports, which is a stronger claim than a calibrated score nobody can check.

======================  =====================================================
Module                  What it holds
======================  =====================================================
``cases.py``            the held-out questions, and why each one is in the suite
``metrics.py``          what a finished campaign is scored on
``harness.py``          running the cases and grading what comes back
``scorecard.py``        turning outcomes into the document somebody reads
``run.py``              the entry point ``make evals`` calls
======================  =====================================================
"""
