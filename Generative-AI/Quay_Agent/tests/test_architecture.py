"""The one rule that, if broken, voids the entire experiment.

The agent is graded by comparing its answers against exact solutions. If any module the
agent can reach is able to *read* those solutions, the comparison proves nothing — the
agent could look up the answer instead of computing it, and no amount of good behaviour
elsewhere would tell us which had happened.

So the exact solvers live in `src/physics/` and are reachable only by the grader. This
test walks the import graph of every module the agent runs and fails if any of them
reaches a forbidden name, directly or through another module it imports.

The guard was written before the packages it guards, and it names a *prefix* rather than a
list of modules, so a package added later is sealed the moment it appears rather than when
somebody remembers to extend a tuple. That is why `src/hardware` was already listed here
while it was still empty, and why nothing had to be changed here when it was filled in.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Packages the agent runs. Nothing here may see an exact answer. The two arms of the
# comparison are both here: the quantum circuit designer, and the classical baseline it
# has to beat. A baseline that could read the answer would report it as its own result.
AGENT_PACKAGES = (
    "src/agent",
    "src/hardware",
    "src/physics/quantum",
    "src/physics/classical",
    # The protocol server hands the agent's tools to a client outside this process.
    # A route from here to a reference solver would not merely let the agent see
    # the answer -- it would publish it, which is the same leak with a wider audience.
    "src/mcp_server",
)

# The reference answers, behind one prefix. It is a prefix rather than a list of module names so
# that a solver added to `src/physics/reference/` later is sealed by default instead of
# by someone remembering to extend a tuple here.
# `physics.model` is deliberately outside it — the problem statement, not its solution.
REFERENCE_ANSWERS = ("src.physics.reference",)

# The cross-check runs the exact solvers by design; it belongs to the grader, not the
# agent, and it is not reachable from AGENT_PACKAGES.
ALLOWED_TO_SEE_THE_ANSWER = ("src/verification", "src/evals")

# The agent's menu of methods. It is not inside an agent package -- physics facts are
# shared -- so the sweep above would not reach it until something imports it, which is
# too late: by then the leak is already written. Guarding it by name puts the check in
# place before the agent that reads it exists.
AGENT_READABLE_MODULES = ("src/physics/method_catalogue.py",)


def _imports_of(path: Path) -> set[str]:
    """Every `src.*` module named by an import statement in one file."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("src."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def _module_path(dotted: str) -> Path | None:
    """The file backing a dotted module name, if one exists."""
    parts = dotted.split(".")
    for candidate in (Path(*parts).with_suffix(".py"), Path(*parts) / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _reachable_from(start: Path) -> set[str]:
    """Transitive closure of `src.*` imports, so an indirect route is caught too."""
    seen: set[str] = set()
    queue = list(_imports_of(start))
    while queue:
        dotted = queue.pop()
        if dotted in seen:
            continue
        seen.add(dotted)
        path = _module_path(dotted)
        if path is not None:
            queue.extend(_imports_of(path))
    return seen


def test_the_agent_cannot_reach_an_exact_solution() -> None:
    leaks: list[str] = []
    for package in AGENT_PACKAGES:
        for source in Path(package).rglob("*.py") if Path(package).exists() else ():
            for forbidden in REFERENCE_ANSWERS:
                if any(reached.startswith(forbidden) for reached in _reachable_from(source)):
                    leaks.append(f"{source} reaches {forbidden}")
    assert not leaks, "the agent can see the answer:\n  " + "\n  ".join(leaks)


def test_the_grader_is_still_allowed_to_see_it() -> None:
    # The mirror image: if this ever fails, the exact solvers have been walled off from
    # the code that is supposed to use them, and nothing would be graded at all.
    graders = [
        p for d in ALLOWED_TO_SEE_THE_ANSWER if Path(d).exists() for p in Path(d).rglob("*.py")
    ]
    if not graders:  # pragma: no cover - only before the grader is written
        return
    assert any(
        any(reached.startswith(f) for f in REFERENCE_ANSWERS)
        for reached in (r for g in graders for r in _reachable_from(g))
    ), "no grader reaches an exact solution — nothing is being verified"


def test_the_method_catalogue_the_agent_reads_cannot_reach_an_exact_solution() -> None:
    # The catalogue describes the exact solvers -- names, cost, applicability -- so that
    # the agent can say honestly what it is being graded against. Describing them and
    # being able to call them are one import apart, and this is the test that keeps them
    # apart. The binding lives in `src/physics/registry.py`, which is grader-only.
    for module in AGENT_READABLE_MODULES:
        source = Path(module)
        assert source.exists(), f"{module} is guarded here but does not exist"
        reached = _reachable_from(source)
        leaks = [name for name in reached for f in REFERENCE_ANSWERS if name.startswith(f)]
        assert not leaks, f"{module} reaches a reference answer via {leaks}"
        assert "src.physics.registry" not in reached, (
            f"{module} imports the grader's bench, which binds the exact solvers"
        )


def test_the_graders_bench_is_the_only_thing_binding_the_exact_solvers() -> None:
    # If a second module ever imports the sealed package, the wall stops being one
    # reviewable file and becomes a property of the whole import graph. Listing the
    # exceptions here forces that to be a decision rather than an accident.
    permitted = {
        # The binding itself. This is the file to read if you want to know how the
        # grader reaches an exact answer.
        "src/physics/registry.py",
        # Diagnostics that need the ground state in hand -- entanglement, the weight
        # of the superposition, the disagreement against the closed form. Grader-side
        # by nature. It sits in `src/physics/` rather than behind the seal because it
        # is analysis rather than a reference answer, and the transitive sweep above still
        # catches an agent module that imports it.
        "src/physics/quantumness.py",
        # Two monitor pages that sit on the grader's side by design. The first draws
        # the live three-route agreement in its "Is the number right?" section, and
        # the lab draws the imaginary-time evolution the variational answer is
        # chasing. Both are a human view of the grader's own output; neither is
        # reachable from anything the agent runs, and the transitive sweep above is
        # what keeps that true.
        #
        # This entry was `src/ui/pages/crosscheck.py` until that page was merged with
        # the hardware page. Renaming it here is the decision this list exists to
        # force: the merged module imports the sealed package for exactly the reason
        # its predecessor did, and it gained no new claim on it by also carrying the
        # transpiler arithmetic, which reaches nothing behind the seal.
        "src/ui/pages/physics_and_hardware.py",
        "src/ui/pages/lab.py",
        # `panels.true_energy`, and only that function. It draws the true answer as a
        # flat line beneath curves the agent produced without it, which is the whole
        # point of the picture -- and it lives here rather than being copied into each
        # page so that two pages cannot grade against two different references. The
        # import is inside the function body, so nothing that merely draws a header
        # pays for it, and the transitive sweep above still fails the moment anything
        # under `src/agent/` imports this module.
        "src/ui/panels.py",
    }
    importers = {
        str(path)
        for path in Path("src").rglob("*.py")
        if not str(path).startswith("src/physics/reference")
        and any(name.startswith(f) for name in _imports_of(path) for f in REFERENCE_ANSWERS)
    }
    assert importers - permitted == set(), (
        "modules outside the grader's bench import the sealed package: "
        f"{sorted(importers - permitted)}"
    )


# Paths a docstring or comment names on purpose without them existing. Each is a
# deliberate reference to something absent, so each needs a reason recorded here.
PATHS_THAT_ARE_MEANT_TO_BE_ABSENT = {
    # A docstring example of what NOT to pass: `save_figure` takes a stem.
    "reports/figures/crossover.pdf",
    # A fixture: the status panel has to say something sensible about a component
    # nobody has written yet, so the test needs a path that will never exist.
    "src/agent/nothing_here.py",
    # The migration record in `scripts/restructure_physics_layout.py`, which names
    # the two modules by their pre-split names. Both sides of a rename cannot exist.
    "src/physics/ed.py",
    "src/physics/exact.py",
    # A comment recording that this page was merged into another one.
    "src/ui/pages/crosscheck.py",
}

REFERENCE = re.compile(
    r"\b(?:src|tests|scripts|reports|data)/[A-Za-z0-9_/]+\.(?:py|md|json|toml)\b"
)


def test_no_comment_or_docstring_cites_a_file_that_is_not_there() -> None:
    """A docstring that points at a missing file sends a reader nowhere.

    Two of these have been found by hand: a note in the corpus ingester citing a
    `test_corpus.py` that never existed, and one in the ansatz tests citing a
    `test_transpiled_depth.py`. Both were accurate when written and were left
    behind by a rename. A reader who follows one concludes the documentation is
    decorative, which is the expensive part.

    Scoped to the three directories the gate covers, so a scratch folder outside
    them cannot fail the build.
    """
    root = Path(__file__).resolve().parent.parent
    missing: list[str] = []
    for directory in ("src", "tests", "scripts"):
        for path in sorted((root / directory).glob("**/*.py")):
            if "__pycache__" in path.parts:
                continue
            for cited in REFERENCE.findall(path.read_text()):
                if cited in PATHS_THAT_ARE_MEANT_TO_BE_ABSENT:
                    continue
                if not (root / cited).exists():
                    missing.append(f"{path.relative_to(root)} cites {cited}, not there")
    assert not missing, "\n".join(missing)


CROSS_REFERENCE = re.compile(r":(?:mod|func|class|data|attr|meth|obj):`~?([^`<]+?)`")


def _names_defined_under(root: Path) -> set[str]:
    """Every dotted name `src/` defines, as a reST cross-reference would spell it.

    Args:
        root: The repository root.

    Returns:
        Module paths plus the top-level and nested classes, functions and assigned
        constants inside them. Directories count as modules: `src.ui.pages` has no
        `__init__.py` and is an implicit namespace package, which is a legitimate
        target.
    """
    found: set[str] = set()
    for path in sorted((root / "src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = str(path.relative_to(root).with_suffix("")).replace("/", ".")
        module = module.removesuffix(".__init__")
        found.add(module)
        for parent in path.relative_to(root).parents:
            if parent.parts:
                found.add(".".join(parent.parts))

        def walk(node: ast.AST, prefix: str) -> None:
            """Record every name this node defines, and recurse into it."""
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                    found.add(f"{prefix}.{child.name}")
                    walk(child, f"{prefix}.{child.name}")
                elif isinstance(child, ast.Assign):
                    found.update(
                        f"{prefix}.{target.id}"
                        for target in child.targets
                        if isinstance(target, ast.Name)
                    )
                elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    found.add(f"{prefix}.{child.target.id}")

        walk(ast.parse(path.read_text()), module)
    return found


def test_no_docstring_points_at_a_function_or_module_that_is_not_there() -> None:
    """A cross-reference to something absent is worse than no cross-reference.

    The same failure as the file-path gate above, one layer in, and it hid a
    substantive one: :mod:`src.physics.quantum.statevector` described a
    `parameter-shift` route in a `gradients` module and said the tests held the two
    against each other. There was no such module and no such test, so the docstring
    documented a cross-check the project did not have -- and the parameter-shift cost
    it quoted was wrong for this ansatz, where one layer angle drives many gates.

    Only ``src.``-qualified targets are checked. A bare name resolves against
    whatever module the reader is in, and a third-party one is not ours to verify.
    """
    root = Path(__file__).resolve().parent.parent
    defined = _names_defined_under(root)
    dangling: list[str] = []
    for path in sorted((root / "src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for cited in CROSS_REFERENCE.findall(line):
                target = cited.strip().split("(")[0]
                if not target.startswith("src.") or target in defined:
                    continue
                dangling.append(f"{path.relative_to(root)}:{number} cites {target}, not there")
    assert not dangling, "\n".join(dangling)
