"""One-off refactor: gather every physics module under ``src/physics/``.

Kept in the repository rather than run and forgotten, because a reader six
months from now will meet the new layout and want to know exactly what moved
and what the imports were rewritten from. Re-running it after the move is a
no-op: every step checks whether it has already happened.

The move
--------
::

    src/quantum/            -> src/physics/quantum/
    src/classical/          -> src/physics/classical/
    src/physics/exact.py    -> src/physics/reference/free_fermions.py
    src/physics/ed.py       -> src/physics/reference/exact_diagonalisation.py

Two things are being fixed at once. The first is the one that was asked for:
all of the physics lives together, and the agentic-AI half of the project --
``src/agent``, ``src/rag``, ``src/evals`` -- is a separate tree. The second is
that ``exact.py`` and ``ed.py`` sat side by side with names that are both
abbreviations of "exact", which is a genuinely bad pair of names to have to
tell apart under pressure. They are now ``free_fermions.py`` (a closed form,
no matrix anywhere) and ``exact_diagonalisation.py`` (a sparse matrix, no
closed form), which say what they are.

The seal
--------
Nesting the agent's two arms *inside* the package that holds the answers makes
the wall between them less obvious to the eye, so it is made more obvious in
the tree instead: everything the agent may not see now lives in one place,
``src/physics/reference/``, and ``tests/test_architecture.py`` bans that single
prefix rather than a list of individual module names.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MOVES: tuple[tuple[str, str], ...] = (
    ("src/quantum", "src/physics/quantum"),
    ("src/classical", "src/physics/classical"),
    ("src/physics/exact.py", "src/physics/reference/free_fermions.py"),
    ("src/physics/ed.py", "src/physics/reference/exact_diagonalisation.py"),
)

#: Applied in order. Longest patterns first, so that a rewritten prefix is not
#: rewritten a second time by a shorter rule.
REWRITES: tuple[tuple[str, str], ...] = (
    (r"\bsrc\.physics\.exact\b", "src.physics.reference.free_fermions"),
    (r"\bsrc\.physics\.ed\b", "src.physics.reference.exact_diagonalisation"),
    (r"\bsrc\.quantum\b", "src.physics.quantum"),
    (r"\bsrc\.classical\b", "src.physics.classical"),
    (
        r"^from src\.physics import ed, exact$",
        "from src.physics.reference import exact_diagonalisation, free_fermions",
    ),
    (
        r"^from src\.physics import exact, ed$",
        "from src.physics.reference import exact_diagonalisation, free_fermions",
    ),
    (r"^from src\.physics import ed$", "from src.physics.reference import exact_diagonalisation"),
    (r"^from src\.physics import exact$", "from src.physics.reference import free_fermions"),
    (r"\bsrc/quantum\b", "src/physics/quantum"),
    (r"\bsrc/classical\b", "src/physics/classical"),
    (r"\bsrc/physics/exact\.py\b", "src/physics/reference/free_fermions.py"),
    (r"\bsrc/physics/ed\.py\b", "src/physics/reference/exact_diagonalisation.py"),
    (r"\bphysics/exact\.py\b", "physics/reference/free_fermions.py"),
    (r"\bphysics/ed\.py\b", "physics/reference/exact_diagonalisation.py"),
)

#: Bare module aliases, rewritten only where they are used as a qualifier.
ALIASES: tuple[tuple[str, str], ...] = (
    (r"\bed\.(?=[a-z_]+)", "exact_diagonalisation."),
    (r"(?<![\w.])exact\.(?=[a-z_]+)", "free_fermions."),
)

REFERENCE_DOCSTRING = '''"""Ground truth. Nothing the agent runs may import from this package.

Two independent exact solutions of the transverse-field Ising chain, kept behind
one import prefix so that the rule guarding them is a single line rather than a
list that someone will forget to extend.

- :mod:`~src.physics.reference.free_fermions` -- the Pfeuty closed form, reached
  by Jordan-Wigner and a Bogoliubov rotation. No matrix is ever built, and the
  cost does not grow with chain length.
- :mod:`~src.physics.reference.exact_diagonalisation` -- a sparse matrix in the
  spin basis, diagonalised directly. Exponential in the chain length and
  therefore size-capped, and it shares no algebra at all with the route above.

Agreement between two methods with nothing in common is evidence; a passing
self-consistency check inside one method is not. That is the project's thesis,
and this package is where it is applied to the project's own answers.

**Why the seal matters.** The agent is graded by comparing its answers with
these. If any module the agent runs could read them, the comparison would prove
nothing: the agent might have looked the answer up, and no amount of good
behaviour elsewhere would say which had happened.
``tests/test_architecture.py`` walks the import graph of every agent-facing
package and fails if this prefix is reachable, directly or transitively.
:mod:`src.physics.model` is deliberately *not* here -- it is the problem
statement, and the agent is of course allowed to know what it was asked.
"""
'''


def move(source: str, destination: str) -> None:
    """Move a path, preferring ``git mv`` so that history follows the file.

    Falls back to a plain filesystem move when git declines -- which it does
    for a path it has never tracked, and most of this tree is not committed yet.
    """
    completed = subprocess.run(
        ("git", "mv", source, destination),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        shutil.move(str(ROOT / source), str(ROOT / destination))


def move_everything() -> None:
    """Perform the moves, discarding stale bytecode caches on the way."""
    for cache in ROOT.glob("src/**/__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    (ROOT / "src/physics/reference").mkdir(parents=True, exist_ok=True)
    for source, destination in MOVES:
        origin = ROOT / source
        if not origin.exists():
            print(f"  already moved: {source}")
            continue
        (ROOT / destination).parent.mkdir(parents=True, exist_ok=True)
        move(source, destination)
        print(f"  {source} -> {destination}")

    marker = ROOT / "src/physics/reference/__init__.py"
    if not marker.exists():
        marker.write_text(REFERENCE_DOCSTRING, encoding="utf-8")
        print("  wrote src/physics/reference/__init__.py")


def rewrite_references() -> None:
    """Rewrite every dotted import, path string and bare module alias."""
    targets = [
        path
        for pattern in ("src/**/*.py", "tests/**/*.py", "*.md", "Makefile", "pyproject.toml")
        for path in ROOT.glob(pattern)
        if "PROJECTS-IDEAS-CAPSTONE" not in str(path)
    ]
    for path in targets:
        original = path.read_text(encoding="utf-8")
        updated = original
        for pattern, replacement in REWRITES:
            updated = re.sub(pattern, replacement, updated, flags=re.MULTILINE)
        if path.suffix == ".py":
            for pattern, replacement in ALIASES:
                updated = re.sub(pattern, replacement, updated)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            print(f"  rewrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    print("moving:")
    move_everything()
    print("rewriting references:")
    rewrite_references()
    print("done -- now run `make check`")
