"""What exists, what is installed, and what a run has left behind.

The monitor's facts live here rather than in the Streamlit pages, for one reason:
a dashboard that computes its own numbers inline cannot be tested, and an
untested dashboard is exactly the kind of thing that keeps reporting "healthy"
after the thing it monitors has stopped working. Every claim the web pages make
comes from a function in this module, and every function here is a pure read of
the filesystem or the installed distribution metadata.

Nothing here imports Qiskit, Streamlit, or any component it reports on.
Presence is decided by looking for the file, and a version by asking
``importlib.metadata`` -- never by importing, because importing a half-finished
module is how a status page acquires the power to crash the thing it is
watching.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""The repository root, resolved from this file rather than the process's cwd.

Streamlit is normally launched from the root, but ``make run`` is not the only
way anyone will ever start it, and a monitor that reports "nothing built yet"
because it was started from the wrong directory is worse than no monitor.
"""


@dataclass(frozen=True, slots=True)
class Component:
    """One planned module and how deep in the dependency order it sits.

    Attributes:
        path: Repository-relative path to the file that would implement it.
        tier: How much has to exist before this can. Tier one depends on nothing
            in the project, tier two on tier one, and so on. It is a statement
            about what blocks what, not about when anything is scheduled -- a
            component is ready to write when its tier below is complete, whatever
            order the work is actually taken in.
        purpose: One line on what it is for, shown next to the status.
    """

    path: str
    tier: int
    purpose: str

    @property
    def exists(self) -> bool:
        """Whether the file is present in the working tree."""
        return (PROJECT_ROOT / self.path).exists()

    @property
    def lines(self) -> int:
        """How many lines the file has, or zero if it does not exist yet.

        A crude size, and deliberately so: it is here to distinguish a written
        module from an empty placeholder, which is a distinction a presence
        check alone cannot make.
        """
        target = PROJECT_ROOT / self.path
        if not target.exists():
            return 0
        return len(target.read_text(encoding="utf-8").splitlines())


COMPONENTS: tuple[Component, ...] = (
    Component("src/settings.py", 1, "config with validators, secrets as SecretStr"),
    Component("src/logging_setup.py", 1, "JSON logs, credentials masked at the formatter"),
    Component("src/security.py", 1, "deterministic prompt-injection screening"),
    Component("src/physics/model.py", 1, "TFIMSpec — the problem specification"),
    Component(
        "src/physics/reference/free_fermions.py",
        1,
        "Pfeuty free-fermion solution — the reference answer",
    ),
    Component(
        "src/physics/reference/exact_diagonalisation.py",
        1,
        "sparse exact diagonalisation — the second reference answer",
    ),
    Component("src/verification/cross_check.py", 1, "two methods, no shared algebra"),
    Component("src/rag/retrieve.py", 1, "hybrid BM25 + vector retrieval"),
    Component("src/physics/quantum/hamiltonians.py", 2, "Pauli-term representation of the chain"),
    Component(
        "src/physics/quantum/ansatz.py", 2, "circuit shape and cost, priced before it is built"
    ),
    Component(
        "src/physics/quantum/circuit_algebra.py", 2, "the equations the circuit is a picture of"
    ),
    Component(
        "src/physics/classical/dual_lattice.py",
        2,
        "the quantum-to-classical mapping, and its lattice",
    ),
    Component("src/physics/classical/metropolis.py", 2, "checkerboard sampling of the dual model"),
    Component(
        "src/physics/classical/variational_imaginary_time.py",
        2,
        "VITA by stochastic reconfiguration — the classical baseline",
    ),
    Component("src/model_catalogue.py", 2, "the models this subscription can reach"),
    Component(
        "src/physics/method_catalogue.py",
        2,
        "what methods exist and which the agent may run — agent-readable",
    ),
    Component(
        "src/physics/registry.py", 2, "the grader's bench: catalogue bound to the exact solvers"
    ),
    Component("src/agent/tools.py", 2, "the functions the model may call, and nothing else"),
    Component("src/physics/quantum/statevector.py", 3, "exact layers and the adjoint gradient"),
    Component(
        "src/physics/quantum/variational_eigensolver.py", 3, "the VQE driver, with its stop reason"
    ),
    Component(
        "src/physics/quantum/quantum_approximate_optimisation.py",
        3,
        "ramp-time scan, INTERP warm start, depth sweep",
    ),
    Component(
        "src/physics/quantum/imaginary_time_evolution.py",
        3,
        "VarQITE: imaginary time on the angles, by stochastic reconfiguration",
    ),
    Component("src/hardware/devices.py", 5, "ideal · linear · a real heavy-hex lattice"),
    Component("src/hardware/transpile.py", 5, "decompose · place · route · schedule"),
    Component("src/hardware/fidelity.py", 5, "what survives the noise, and the depth ceiling"),
    Component("src/hardware/export.py", 6, "QASM 3 · run card · submission package"),
    Component("src/physics/classical/regime.py", 6, "where the classical baseline can be trusted"),
    Component("src/agent/state.py", 4, "the belief state a campaign carries"),
    Component("src/agent/diagnosis.py", 4, "what a finished optimisation actually did"),
    Component("src/agent/verdict.py", 4, "go, no or conditional, decided by rule"),
    Component("src/agent/report.py", 4, "the assessment, written for a non-specialist"),
    Component("src/agent/graph.py", 4, "the campaign loop"),
    Component("src/agent/router.py", 4, "whether to search, and which shelf first"),
    Component("src/agent/model_selection.py", 4, "which model serves which call"),
    Component("src/agent/middleware.py", 4, "the layers every model call runs inside"),
    Component("src/agent/prompts.py", 4, "every instruction sent to a model"),
    Component("src/agent/memory.py", 4, "what is recalled between questions"),
    Component("src/agent/followups.py", 5, "what to ask next, proposed and then screened"),
    Component("src/agent/intent.py", 5, "what kind of answer the question is asking for"),
    Component("src/agent/explaining.py", 5, "the prose branch: answered from the notes"),
    Component("src/agent/drafting.py", 5, "the code branch: written, and never run here"),
    Component("src/rag/external.py", 4, "fetching outside the corpus, and the gate on it"),
    Component("src/agent/campaign.py", 5, "the campaign, from a terminal"),
    Component("src/mcp_server/protocol.py", 5, "the tools, offered over the protocol"),
    Component("src/evals/cases.py", 6, "the held-out questions"),
    Component("src/evals/metrics.py", 6, "what a finished campaign is scored on"),
    Component("src/evals/harness.py", 6, "running the cases and grading them"),
    Component("src/evals/run.py", 6, "the scorecard"),
    Component("src/ui/panels.py", 6, "the shared pieces every page draws with"),
    Component("src/ui/progress.py", 6, "what a person is told while a campaign runs"),
    Component("src/ui/access.py", 6, "who is visiting, and which models that entitles them to"),
)
"""The intended shape of the project, as a checklist the monitor can read.

Hand-maintained rather than discovered, because the interesting question is what
is *missing*, and a directory walk can only ever show what is there.
"""


@dataclass(frozen=True, slots=True)
class Dependency:
    """A third-party package the project needs, and what it is needed for.

    Attributes:
        distribution: The name on PyPI, which is what ``importlib.metadata``
            answers to and is not always the import name.
        purpose: Why the project wants it.
        needed_from_tier: The dependency tier that first requires it, so a
            missing package can be read as "not yet" rather than "broken".
    """

    distribution: str
    purpose: str
    needed_from_tier: int

    @property
    def installed_version(self) -> str | None:
        """The installed version, or ``None`` if the distribution is absent."""
        try:
            return version(self.distribution)
        except PackageNotFoundError:
            return None


DEPENDENCIES: tuple[Dependency, ...] = (
    Dependency("numpy", "arrays; the whole physics layer", 1),
    Dependency("scipy", "sparse matrices and eigensolvers", 1),
    Dependency("pydantic-settings", "validated configuration", 1),
    Dependency("streamlit", "this monitor", 1),
    Dependency("langgraph", "the campaign loop", 8),
    Dependency("chromadb", "the vector store", 1),
    Dependency("qiskit", "circuits, transpiler, device models", 2),
    Dependency("qiskit-aer", "state-vector and shot-based simulation", 2),
    Dependency("qiskit-ibm-runtime", "the real FakeBackend snapshots", 5),
    Dependency("networkx", "graphs, for the classical baseline", 6),
    Dependency("cvxpy", "the Goemans-Williamson SDP", 6),
)
"""Everything the project depends on, with the tier at which it becomes load-bearing.

Ordered by that tier, so the monitor's dependency table doubles as an answer to
"what do I need to install before tomorrow's work?"
"""


@dataclass(frozen=True, slots=True)
class Artefact:
    """One file a run produced.

    Attributes:
        path: Repository-relative path.
        size_bytes: Size on disk.
        modified: Last modification time, in UTC.
    """

    path: str
    size_bytes: int
    modified: datetime


ARTEFACT_DIRECTORIES: tuple[str, ...] = (
    "reports",
    "reports/figures",
    "reports/submissions",
    "sealed",
    ".checkpoints",
)
"""Where a run leaves evidence behind.

Read-only, and listed rather than parsed: the monitor's job at this stage is to
say whether a campaign has run and what it wrote, not to interpret the contents.
"""

ARTEFACT_PURPOSE: dict[str, str] = {
    "reports": "the scorecard, the figures, and the transcripts a campaign leaves",
    "reports/figures": "every figure a run draws, as PDF",
    "reports/submissions": ("one submission package per case: QASM 3, the run card, the drawing"),
    "sealed": (
        "the expert plan, committed *before* any agent run and never revised. It is "
        "worthless if written after the results are in, and it is the arm that makes "
        "the comparison mean something"
    ),
    ".checkpoints": (
        "where LangGraph state *would* be saved. The store is built and tested, and "
        "the graph does not compile with it: one question is one pass and no step "
        "suspends, so there is nothing yet to resume. `src/agent/checkpointing.py` "
        "carries the argument in full"
    ),
}
"""What each watched directory is for, in the words the About page prints.

Here rather than in the page for the reason the module docstring gives: every
claim a page makes comes from a function in this module. The page had drifted --
it named ``reports/circuits/`` and ``data/cases/``, neither of which is watched,
and omitted ``reports/submissions``, which is. A reader was being told the
software watches two directories it does not look at.
"""


def build_progress() -> tuple[int, int]:
    """Count the components that exist against the total planned.

    Returns:
        ``(built, total)``. Deliberately blunt -- a file that exists is counted
        even if it is a stub -- because the number is a progress indicator and
        not a claim of correctness. Correctness is what the test suite is for.
    """
    return sum(1 for component in COMPONENTS if component.exists), len(COMPONENTS)


def components_in_tier(tier: int) -> tuple[Component, ...]:
    """Every planned component sitting at one level of the dependency order.

    Args:
        tier: The level, counting from one.

    Returns:
        The components, in declaration order.
    """
    return tuple(component for component in COMPONENTS if component.tier == tier)


def missing_dependencies(up_to_tier: int) -> tuple[Dependency, ...]:
    """Dependencies that are needed by ``up_to_tier`` and are not installed.

    This is the question the monitor exists to answer at a glance: not "what is
    installed" but "what is missing that I need *now*". A package that only a
    later tier needs is not a problem yet, and reporting it as one trains the reader
    to ignore the page.

    Args:
        up_to_tier: How deep into the dependency order the project has reached.

    Returns:
        The absent dependencies whose tier has arrived, in declaration order.
    """
    return tuple(
        dependency
        for dependency in DEPENDENCIES
        if dependency.needed_from_tier <= up_to_tier and dependency.installed_version is None
    )


def artefacts(limit: int = 50) -> tuple[Artefact, ...]:
    """List what previous runs have written, newest first.

    Args:
        limit: Most entries to return.

    Returns:
        The files found under :data:`ARTEFACT_DIRECTORIES`, newest first. An
        empty tuple is the honest answer before anything has run, and the pages
        say so in words rather than drawing an empty table.
    """
    found: list[Artefact] = []
    for directory in ARTEFACT_DIRECTORIES:
        root = PROJECT_ROOT / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            found.append(
                Artefact(
                    path=str(path.relative_to(PROJECT_ROOT)),
                    size_bytes=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                )
            )
    found.sort(key=lambda artefact: artefact.modified, reverse=True)
    return tuple(found[:limit])


def corpus_size() -> int:
    """How many Markdown notes the retrieval corpus holds, across every shelf.

    The corpus is one directory per shelf -- ``physics-notes``,
    ``quantum-computing``, ``applications`` -- and no note sits at the root, so
    the count has to descend. Counting only the root is how this reported "0
    notes" while nineteen were sitting one level down, which is exactly the
    failure mode this module exists to avoid.

    Returns:
        The file count, or zero if the corpus directory is absent.
    """
    corpus = PROJECT_ROOT / "data" / "corpus"
    if not corpus.is_dir():
        return 0
    return sum(1 for _ in corpus.rglob("*.md"))
