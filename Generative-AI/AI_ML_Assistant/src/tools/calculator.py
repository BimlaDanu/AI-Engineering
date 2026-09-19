"""Symbolic and numeric maths tool (SymPy), guarded against abuse.

SymPy's ``sympify`` uses ``eval`` under the hood, so an untrusted expression is both a
code-execution and a resource-exhaustion risk. Defence in depth, in order:

1. A length cap and a leading-token blocklist reject obvious abuse *before* parsing.
2. The evaluation itself runs in a separate, **killable** process with a wall-clock timeout,
   so a pathological expression (e.g. ``factorial(10**8)`` or ``10**10**10``) can neither hang
   the app nor OOM the host — the child is terminated and an error is returned instead. That
   timeout covers the *maths*: starting the worker has its own, larger budget, because a
   ``spawn`` child re-imports the parent's ``__main__`` and charging that to the same clock
   made ordinary expressions report themselves as too expensive.
3. The rendered result is size-capped so a huge symbolic answer can't flood the transcript.

If the platform cannot spawn a worker process (rare), evaluation falls back to running inline
with the same input guards applied — losing only the hard timeout, never the input checks.
"""

from __future__ import annotations

import multiprocessing as mp
import re
from queue import Empty

from langchain_core.tools import tool

# Reject expressions longer than this before parsing (blocks giant literal payloads).
_MAX_EXPR_CHARS = 500
# Wall-clock budget for the evaluation *itself*, measured from the moment the worker reports
# that it is up and has SymPy loaded. The worker is killed if it overruns.
_TIMEOUT_SECONDS = 5.0
# Separate, much larger budget for getting that worker to the starting line. It is a different
# quantity and it used to be charged to the one above, which is a bug rather than a tuning
# choice: a "spawn" child is a fresh interpreter that re-imports the parent's ``__main__``, and
# under `streamlit run` or pytest that is a heavy module to pay for — several seconds of it,
# before a single symbol has been parsed. So ``integrate(2*x, (x, 0, 3))`` could come back as
# "too expensive to evaluate", which is both wrong and unactionable: the advice is to simplify
# an expression that was never the problem. This bound exists only so a worker that never
# starts at all cannot hang the caller; it is not a limit anyone should reach by computing.
_STARTUP_SECONDS = 30.0
# Cap the rendered answer so an enormous symbolic result can't flood the UI/transcript.
_MAX_RESULT_CHARS = 4000
# Raw substrings that have no place in a maths expression (dunder access, statement chaining).
_BANNED_SUBSTRINGS = ("__", ";")
# Dangerous Python builtins, matched as *whole identifiers* (``\b…\b``) so that legitimate
# SymPy methods which merely contain one as a substring are not rejected — most importantly
# ``.evalf()`` (numeric evaluation), which the naive ``"eval" in expr`` check used to block.
_BANNED_CALLS = re.compile(r"\b(?:import|open|eval|exec)\b")


def _has_banned_syntax(expression: str) -> bool:
    """True if the expression contains a Python-escape token: a dunder, a ``;``, or one of the
    dangerous builtins (``import``/``open``/``eval``/``exec``) used as a standalone identifier.

    ``.evalf()`` and similar method names are deliberately allowed — ``eval`` is only rejected
    when it appears as its own word (e.g. ``eval('...')``), never inside ``evalf``.
    """
    lowered = expression.lower()
    if any(token in lowered for token in _BANNED_SUBSTRINGS):
        return True
    return bool(_BANNED_CALLS.search(lowered))


def _compute(expression: str, operation: str) -> str:
    """Pure evaluation core (no process/queue machinery) — used inline and by the worker."""
    import sympy

    try:
        expr = sympy.sympify(expression)
        if operation == "solve":
            result = sympy.solve(expr)
        else:
            result = expr.doit() if hasattr(expr, "doit") else expr
            if operation == "simplify":
                result = sympy.simplify(result)
    except (sympy.SympifyError, TypeError, ValueError, NotImplementedError) as exc:
        return f"Could not compute that: {exc}"
    # Stringifying a pathological result can itself blow up — e.g. a huge integer trips
    # Python's int-to-str digit limit (ValueError) — so render defensively.
    try:
        result_str = str(result)
    except Exception:
        result_str = f"(a {type(result).__name__} result too large to display)"
    try:
        rendered = f"\n\nLaTeX: $ {sympy.latex(result)} $"
    except Exception:
        rendered = ""
    out = f"Result: {result_str}{rendered}"
    if len(out) > _MAX_RESULT_CHARS:
        out = out[:_MAX_RESULT_CHARS] + " … (result truncated)"
    return out


_READY = "\x00ready"
"""Sentinel the worker sends once it is importable and has SymPy loaded.

A private control value rather than a second queue, because everything else the child sends is
a user-facing string and one channel is easier to reason about. The null byte keeps it from
colliding with any result SymPy could render.
"""


def _compute_to_queue(expression: str, operation: str, queue) -> None:
    """Worker entry point: announce readiness, then evaluate and push the result string.

    The readiness ping is sent *after* SymPy is imported, so the parent can start the
    evaluation clock at the point evaluation actually begins rather than at ``start()``.
    """
    try:
        import sympy  # noqa: F401  (the expensive part of starting up; do it before the ping)

        queue.put(_READY)
    except Exception as exc:  # a child that cannot even import must still say so
        queue.put(f"Could not compute that: {exc}")
        return
    try:
        queue.put(_compute(expression, operation))
    except Exception as exc:  # never let the child exit without a message
        queue.put(f"Could not compute that: {exc}")


def _compute_with_timeout(expression: str, operation: str) -> str:
    """Evaluate in a killable worker, bounded by ``_TIMEOUT_SECONDS``; fall back inline.

    Two budgets, not one: ``_STARTUP_SECONDS`` covers getting the worker up, and only then
    does ``_TIMEOUT_SECONDS`` start running on the maths. See the note on those constants for
    why charging both to one clock made trivial integrals time out.
    """
    proc = None
    try:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_compute_to_queue, args=(expression, operation, queue))
        proc.start()
        try:
            first = queue.get(timeout=_STARTUP_SECONDS)
        except Empty:
            return (
                "The maths worker could not be started, so the expression was not evaluated. "
                "Please try again."
            )
        if first != _READY:
            return str(first)  # the child failed to import; its message is already user-facing
        try:
            return str(queue.get(timeout=_TIMEOUT_SECONDS))
        except Empty:
            return (
                f"Calculation timed out after {_TIMEOUT_SECONDS:g}s — the expression is too "
                "expensive to evaluate. Try a simpler or more bounded expression."
            )
    except Exception:
        # Platform can't spawn a worker — evaluate inline. Input guards already ran in the
        # tool body, so this is safe against the code-injection vectors; only the hard
        # timeout is lost.
        return _compute(expression, operation)
    finally:
        # Reached on every path, including the two timeouts: whatever the child is still doing,
        # it is no longer wanted, and leaving it running is how a bounded tool grows an
        # unbounded background process per abusive question.
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)


@tool
def math_calculator(expression: str, operation: str = "evaluate") -> str:
    """Symbolic and numeric maths: evaluate, simplify, or solve expressions.

    Handles calculus (integrate, diff, limit), linear algebra (Matrix(...).eigenvals(),
    .det(), .inv()), high-precision numeric evaluation (e.g. 'pi.evalf(50)'), and
    probability/statistics expressions in SymPy syntax.

    Args:
        expression: SymPy-parseable expression, e.g. 'integrate(exp(-x**2), (x, -oo, oo))',
            'Matrix([[2, 1], [1, 2]]).eigenvals()', 'pi.evalf(50)', or 'x**2 - 5*x + 6' with
            operation='solve'.
        operation: 'evaluate' (default), 'simplify', or 'solve' (treats expression as = 0).
    """
    if len(expression) > _MAX_EXPR_CHARS:
        return f"Expression is too long (max {_MAX_EXPR_CHARS} characters)."
    if _has_banned_syntax(expression):
        return "Expression contains disallowed syntax."
    return _compute_with_timeout(expression, operation)
