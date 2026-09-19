"""A restricted in-process executor for the 🔬 AI/ML Lab's opt-in code cell.

This runs short, learner-written Python (pandas / scikit-learn / matplotlib on a bundled
toy dataset) and captures its stdout and any matplotlib figure. It is **framework-agnostic**
(never imports Streamlit) and pure enough to unit-test offline.

Security posture — read this before trusting it. The executor is a *guardrail against
accidents*, not a true security sandbox:

* ``__builtins__`` is replaced with a small allow-list — no ``open``, ``exec``, ``eval``,
  ``compile``, ``input``, or ``__import__`` escape hatch.
* ``import`` is routed through :func:`_guarded_import`, which permits only a fixed allow-list
  of data-science modules (numpy, pandas, matplotlib, scikit-learn, and a few stdlib maths
  helpers) and refuses everything else — so ``os``, ``sys``, ``subprocess``, ``socket``,
  ``pathlib`` and friends are unreachable by name.

What it does **not** do is defend against a hostile user who deliberately walks the object
graph (``().__class__.__bases__`` …) to re-reach blocked builtins. Escaping a restricted
CPython namespace is a known, unsolved problem. This is acceptable here because the Lab runs
inside the user's *own* locally-run app on datasets they chose — the goal is to stop a typo
from deleting a file, not to safely run adversarial code. The UI states this plainly.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import warnings
from dataclasses import dataclass, field
from typing import Any

# Modules the sandboxed code may import. Everything is a pure data-science / maths library
# with no file, process, or network surface of its own that the lesson exercises need.
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "scipy",
        "math",
        "statistics",
        "random",
        "collections",
        "itertools",
        "functools",
        "json",
        "re",
        "decimal",
        "fractions",
    }
)

# Builtins that hand back a file, a process, or arbitrary code execution — never exposed.
_BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {"open", "exec", "eval", "compile", "input", "breakpoint", "help", "__import__"}
)

# Warning messages with no value to a learner: matplotlib complaining that ``plt.show()`` can't
# open a GUI window when we run headless (Agg) and capture the figure ourselves. Dropped so the
# terminal and the Lab stay quiet; substrings are matched case-insensitively.
_NOISE_WARNINGS: tuple[str, ...] = (
    "figurecanvasagg is non-interactive",
    "cannot be shown",
)


@dataclass
class SandboxResult:
    """The outcome of one sandboxed run: captured output plus an optional figure.

    ``ok`` is False only when the code raised; ``error`` then holds the exception's
    ``type: message`` (never a full host traceback, which could leak paths). ``figure`` is
    the active matplotlib ``Figure`` if the code drew one, else None. ``warnings`` holds any
    de-duplicated, learner-relevant warnings raised during the run (e.g. sklearn's
    ``ConvergenceWarning``); pure-noise warnings are filtered out and never leak to the terminal.
    """

    ok: bool
    stdout: str = ""
    error: str | None = None
    figure: Any | None = None
    namespace: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _guarded_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """A drop-in ``__import__`` that permits only :data:`ALLOWED_MODULES` (and submodules)."""
    root = name.split(".", 1)[0]
    if level != 0 or root not in ALLOWED_MODULES:
        raise ImportError(
            f"import of '{name}' is blocked in the AI/ML Lab sandbox "
            f"(allowed: {', '.join(sorted(ALLOWED_MODULES))})"
        )
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    """Return a copy of the builtins namespace with the dangerous names removed."""
    safe = {
        name: getattr(builtins, name)
        for name in dir(builtins)
        if not name.startswith("_") and name not in _BLOCKED_BUILTINS
    }
    safe["__import__"] = _guarded_import
    return safe


def _capture_figure() -> Any | None:
    """Return the current matplotlib figure if one has been drawn, else None.

    Imported lazily so the sandbox module stays importable even when matplotlib (an optional
    ML-Lab dependency) is absent — a run simply yields no figure in that case.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    nums = plt.get_fignums()
    if not nums:
        return None
    fig = plt.figure(nums[-1])
    return fig if fig.get_axes() else None


def _format_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """Turn captured warnings into de-duplicated ``Category: message`` lines, minus the noise."""
    seen: set[str] = set()
    lines: list[str] = []
    for record in caught:
        text = str(record.message)
        if any(noise in text.lower() for noise in _NOISE_WARNINGS):
            continue
        line = f"{record.category.__name__}: {text}"
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def run_sandboxed(code: str, namespace: dict[str, Any] | None = None) -> SandboxResult:
    """Execute ``code`` with a restricted builtins/import surface and capture its output.

    Args:
        code: The Python source to run (the learner's edited cell).
        namespace: Pre-bound names available to the code (e.g. ``df``, ``X``, ``y``, and the
            ``np`` / ``pd`` / ``plt`` handles the recipes preload). Copied, not mutated.

    Returns:
        A :class:`SandboxResult` with captured stdout, an optional matplotlib figure, and —
        on failure — a compact ``error`` string instead of raising.
    """
    if not code.strip():
        return SandboxResult(ok=True, stdout="")

    sandbox_globals: dict[str, Any] = dict(namespace or {})
    sandbox_globals["__builtins__"] = _safe_builtins()

    out = io.StringIO()
    # ``record=True`` intercepts every warning instead of letting it print to stderr, so a
    # non-converging model or a stray ``plt.show()`` no longer spams the terminal — we surface
    # the useful ones through ``SandboxResult.warnings`` and drop the noise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            compiled = compile(code, "<ml-lab>", "exec")
            with contextlib.redirect_stdout(out):
                # The exec IS the feature: this is a learner code cell. The guard is the
                # restricted globals built above (no __import__, no open, no builtins
                # beyond the allow-list), not the absence of exec. Flagged by S102 by
                # design, silenced here so the decision is visible rather than absent.
                exec(compiled, sandbox_globals)  # noqa: S102
        except Exception as exc:  # any learner error becomes a message, never a page crash
            return SandboxResult(
                ok=False,
                stdout=out.getvalue(),
                error=f"{type(exc).__name__}: {exc}",
                figure=_capture_figure(),
                namespace=sandbox_globals,
                warnings=_format_warnings(caught),
            )
        return SandboxResult(
            ok=True,
            stdout=out.getvalue(),
            figure=_capture_figure(),
            namespace=sandbox_globals,
            warnings=_format_warnings(caught),
        )
