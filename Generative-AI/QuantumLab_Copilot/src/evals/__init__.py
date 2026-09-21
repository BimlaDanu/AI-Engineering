"""Evaluation: the held-out suite and the scorecard it produces.

This package is why the project picked a domain where the answers are known. Most
agent evaluations reduce to an LLM judging helpfulness, which is hard to act on.
Here a question has an answer :mod:`src.physics.exact` can work out independently,
so accuracy is measured rather than asserted.

The suite is a frozen set of questions with known answers, held out from the
prompts and never edited to make a run look better. Each case records the
expected value, the tolerance, and which methods should legitimately be able to
answer it, so the scorecard can separate three different failures: a wrong
number, a right number from the wrong method, and a refusal that should have
been an answer.

Run it with ``make evals``.
"""
