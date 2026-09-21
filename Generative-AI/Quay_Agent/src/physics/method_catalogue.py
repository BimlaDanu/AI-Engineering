r"""What methods exist, when each applies, and which of them the agent may run.

This module is the seam the agent's method-selection step reads. It answers one
question -- *given this problem, what could solve it, and why can the rest not?*
-- in pure Python, with no language model, no API key and no network. That is
deliberate: applicability is a physics fact, and a fact the agent is allowed to
invent is a fact that will eventually be wrong.

This is a separate module from :mod:`src.physics.registry`. The obvious design
puts the facts and the solver functions in one place. That design cannot
work here. The agent must be able to read the menu -- it has to know that a
closed-form solution exists for a ring of even length, and it has to be able to
say so honestly in a report -- but it must never be able to *call* one, because
the entire experiment is the comparison of its answer against that solution. A
module holding both would give it the second the moment it was handed the first.

So the split runs along that line and no other:

- Here: names, descriptions, cost and accuracy classes, applicability
  checks, and the callables of the methods the agent is *allowed to run* --
  today, the classical variational baseline. Nothing in this module imports
  :mod:`src.physics.reference`, and the architecture test enforces it.
- In :mod:`src.physics.registry`: the binding from a catalogue entry to the
  exact solver behind it. That module imports the sealed package and belongs to
  the grader alone.

An exact method therefore appears here with :data:`Availability` of
``grader_only`` and ``ground_state_energy`` of ``None``. The agent can see that
it exists, can quote its cost and can explain to a reader why the number it is
being graded against is trustworthy -- and holds no way to evaluate it.

Facts, not policy. The catalogue does not choose. It reports that two
methods apply, that one is ``linear`` and the other ``exponential``, and that a
third was refused with a specific reason. *Preferring* the cheap one, asking
before an expensive run, or giving up when nothing applies is policy and lives
in :mod:`src.agent`. Keeping the line here means a notebook can do method
selection with no agent at all, and means the selection logic can be
unit-tested against a fake catalogue.

Registration is an explicit tuple, not auto-discovery. A method that nobody
listed does not silently appear in the agent's menu.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src.physics.model import MAX_SITES_SPARSE, MAX_SITES_STATEVECTOR, TFIMSpec

CostClass = Literal["constant", "linear", "polynomial", "exponential"]
"""How a method's cost scales with chain length ``L``.

``constant`` closes the problem in closed form regardless of size; ``linear``
does ``O(L)`` arithmetic; ``polynomial`` is the Monte Carlo baseline, whose cost
is set by the sample count and grows only with the lattice it sweeps; and
``exponential`` builds an object of dimension ``2**L`` and is therefore always
size-capped.
"""

AccuracyClass = Literal["exact", "variational_bound", "uncontrolled"]
"""What kind of number a method returns.

``exact`` is right to machine precision. ``variational_bound`` is guaranteed to
lie above the true ground-state energy, which makes it checkable at any size
with no reference -- and it is what every method the agent can actually run
returns. ``uncontrolled`` carries an error with no rigorous bound -- mean-field
theory near criticality -- and must never be reported without that caveat.
"""

Availability = Literal["agent", "grader_only"]
"""Who is permitted to run a method.

``agent`` methods carry a callable and may be run by anything. ``grader_only``
methods carry ``None``: they are described here so the agent can reason about
what exists, and bound to their implementations in :mod:`src.physics.registry`,
which the agent cannot import.
"""

COST_ORDER: dict[CostClass, int] = {
    "constant": 0,
    "linear": 1,
    "polynomial": 2,
    "exponential": 3,
}
"""Ranking of :data:`CostClass` from cheapest to most expensive.

Exposed as a fact so that policy code can sort by it. That cheap beats
expensive *when accuracy is equal* is a judgement, and it is not made here.
"""


def free_fermion_unsupported_reason(spec: TFIMSpec) -> str | None:
    r"""Explain why the closed-form solution cannot handle ``spec``.

    This predicate lives in the catalogue rather than beside the solver for the
    reason the module docstring gives: *whether* a method applies is a statement
    about the problem and carries no information about the answer, so the agent
    is entitled to it. :mod:`src.physics.reference.free_fermions` imports this
    function and re-exports it, so there is one definition and no drift.

    Args:
        spec: The problem to check.

    Returns:
        A human-readable reason the method is inapplicable, or ``None`` if it
        applies. A sentence rather than a boolean, so a rejection can be shown
        to the user and cited in the run's justification.
    """
    if not spec.is_one_dimensional:
        return (
            "The Jordan-Wigner transformation linearises a chain and nothing else, "
            "so there is no closed form for a "
            f"{spec.rows}x{spec.cols} {spec.geometry} lattice. Its exact energy has "
            "to be diagonalised, which costs 2**L."
        )
    if spec.boundary != "periodic":
        return (
            "The free-fermion solution implemented here assumes a periodic ring; "
            f"this spec uses {spec.boundary} boundaries."
        )
    if spec.n_sites % 2 != 0:
        return (
            "The even-parity momentum set k = (2n+1)pi/L is only a complete, "
            f"symmetric half-Brillouin-zone for even L; this spec has L={spec.n_sites}."
        )
    return None


def exact_diagonalisation_unsupported_reason(spec: TFIMSpec) -> str | None:
    """Explain why exact diagonalisation cannot handle ``spec``.

    Refused on size alone. The cap is a deliberate refusal rather than an
    attempt that might exhaust memory: an agent that declines a run it cannot
    afford is more useful than one that hangs.

    Its cap is :data:`~src.physics.model.MAX_SITES_SPARSE`, its own constant rather
    than the state-vector one. The two currently hold the same number, sixteen, and
    the separate constant still earns its place: this method never applies a gate --
    it hands a sparse matrix to a Lanczos iteration -- so it is the cheaper of the
    two and the one that could be raised first. It is also the grader's method, and
    reading the circuit simulator's cap here would set the ceiling on what can be
    *graded* by the cost of what is being graded.

    Args:
        spec: The problem to check.

    Returns:
        The reason, or ``None`` if the chain is within the project's cap.
    """
    if spec.n_sites > MAX_SITES_SPARSE:
        return (
            f"the Hilbert space has dimension 2**{spec.n_sites} = {2**spec.n_sites}, "
            f"above the project cap of 2**{MAX_SITES_SPARSE}; use a method "
            f"whose cost does not grow exponentially with L"
        )
    return None


def exact_diagonalisation_memory_bytes(spec: TFIMSpec) -> int:
    """Predict the peak memory an exact-diagonalisation run would need.

    This is what the agent shows the user *before* committing to a run, so it
    deliberately errs high: it counts the CSR matrix, the spin table used to
    build the diagonal, and the Krylov vectors the iterative solver allocates.
    It is arithmetic on ``L`` and reveals nothing about the eigenvalue.

    Args:
        spec: The problem to cost. Need not be supported; the estimate is what
            tells the agent it is not.

    Returns:
        An approximate byte count, accurate to a factor of about two, which is
        all a go/no-go decision requires.
    """
    dimension = 2**spec.n_sites
    nonzeros = dimension * (spec.n_sites + 1)
    csr_bytes = nonzeros * (8 + 4) + (dimension + 1) * 4  # data + indices + indptr
    spin_table_bytes = dimension * spec.n_sites  # int8, one entry per site
    krylov_bytes = 20 * dimension * 8  # ARPACK's default subspace, float64
    return int(csr_bytes + spin_table_bytes + krylov_bytes)


def variational_imaginary_time_unsupported_reason(spec: TFIMSpec) -> str | None:
    r"""Explain why the sampled variational baseline cannot handle ``spec``.

    The one refusal is an odd ring. The Metropolis sweep updates a whole
    sublattice at once, which is only correct if no two sites of the same parity
    share a bond; an odd ring closes parity onto itself and would silently
    correlate the update rather than fail. Refusing here turns a wrong answer
    into a stated one.

    Args:
        spec: The problem to check.

    Returns:
        The reason, or ``None``. Note that there is no size cap: this is the
        method that is supposed to keep working where the others stop.
    """
    if not spec.is_one_dimensional:
        return (
            "the dual classical lattice this sampler walks is sites by imaginary-time "
            f"slices, which is two-dimensional for a line; a {spec.rows}x{spec.cols} "
            f"{spec.geometry} lattice would need a three-dimensional one, and sampling "
            "it as a line of the same size would report the wrong problem's answer"
        )
    if spec.boundary == "periodic" and spec.n_sites > 2 and spec.n_sites % 2 != 0:
        return (
            "the checkerboard sweep needs a bipartite lattice, and an odd ring "
            f"closes parity onto itself; this spec has L={spec.n_sites}"
        )
    return None


CATALOGUE_CIRCUIT_DEPTH = 2
"""Circuit depth the catalogue's one-number quantum entry point runs at.

The catalogue's uniform ``(spec) -> float`` signature has nowhere to put a depth,
so one has to be chosen here. Two layers rather than one because depth one is
where the ramp scan happens and is not yet a result, and rather than ten because
this entry exists to answer "what does the quantum route give?" cheaply. Anything
that cares about the depth should sweep it explicitly instead.
"""


def variational_quantum_eigensolver_unsupported_reason(spec: TFIMSpec) -> str | None:
    """Explain why the simulated quantum solver cannot handle ``spec``.

    Refused on size alone, and for the same reason exact diagonalisation is: the
    circuit is evaluated by carrying a full ``2**L`` state vector, so its memory
    grows exponentially even though no matrix is ever assembled. A real device
    would not have this limit -- which is the whole point of the feasibility
    question -- and stating the refusal as a property of the *simulation* keeps
    that distinction visible.

    Args:
        spec: The problem to check.

    Returns:
        The reason, or ``None`` if the chain is within the simulation cap.
    """
    if spec.n_sites > MAX_SITES_STATEVECTOR:
        return (
            f"simulating the circuit means holding 2**{spec.n_sites} amplitudes, "
            f"above the project cap of 2**{MAX_SITES_STATEVECTOR}; the algorithm is "
            "unaffected, but nothing here can run it at that length"
        )
    return None


def _variational_quantum_eigensolver_energy(spec: TFIMSpec) -> float:
    """Run the simulated quantum solver and return its energy per spin.

    The adapter that gives the quantum route the catalogue's uniform signature,
    exactly as the classical baseline has one. It runs the depth sweep rather
    than a single optimisation, because a lone run at a fixed depth inherits
    whatever starting schedule it was handed, and a bad schedule reports a
    barren plateau as a physical result. The sweep chooses its schedule by
    scanning, then carries it upward.

    The import is deferred to call time so that reading the menu stays free.

    Args:
        spec: The chain to solve.

    Returns:
        The best variational energy per spin -- an upper bound, not an estimate.
    """
    from src.physics.quantum.quantum_approximate_optimisation import depth_sweep

    sweep = depth_sweep(
        n_sites=spec.n_sites,
        max_depth=CATALOGUE_CIRCUIT_DEPTH,
        coupling=spec.coupling,
        transverse_field=spec.field,
        boundary=spec.boundary,
        lattice=spec.shape_to_solve,
    )
    return sweep.best.energy / spec.n_sites


def _variational_imaginary_time_energy(spec: TFIMSpec) -> float:
    """Run the classical baseline and return its energy per spin.

    A thin adapter so the baseline satisfies the catalogue's uniform
    ``(spec) -> float`` signature. Its depth, sampling budget and seed are left
    at the module defaults; anything wanting control over those should call
    :func:`src.physics.classical.variational_imaginary_time.ground_state_energy`
    directly and read the error bar, which this signature has nowhere to put.

    The import is deferred to call time rather than done at module scope so that
    reading the menu stays free -- method selection happens on every campaign
    step, and importing a Monte Carlo sampler to answer "what exists?" is a cost
    with no matching benefit.

    Args:
        spec: The chain to solve.

    Returns:
        The variational energy per spin -- an upper bound, not an estimate.
    """
    from src.physics.classical import variational_imaginary_time

    return variational_imaginary_time.ground_state_energy(spec, geometry=spec.shape_to_solve).energy


@dataclass(frozen=True, slots=True)
class MethodFacts:
    """Everything that is known about one solver without running it.

    Frozen so it can be hashed and cached, and so a node cannot accidentally
    rewrite the catalogue it was handed.

    Attributes:
        name: The catalogue key, matching the module's ``METHOD_NAME``.
        summary: One sentence on what the method actually does.
        when_to_use: One sentence on the situation in which it is the right
            choice. This is the text the agent paraphrases when it justifies a
            selection to the user.
        cost: How the cost scales with ``L``.
        accuracy: What kind of number comes back.
        availability: Whether the agent may run it, or only read about it.
        unsupported_reason: The applicability check.
        ground_state_energy: The solver entry point, or ``None`` for a
            ``grader_only`` method. The ``None`` is the wall, expressed as a
            value rather than as a convention.
        estimate_memory_bytes: Peak-memory predictor, or ``None`` for methods
            whose memory use is trivially small.
    """

    name: str
    summary: str
    when_to_use: str
    cost: CostClass
    accuracy: AccuracyClass
    availability: Availability
    unsupported_reason: Callable[[TFIMSpec], str | None]
    ground_state_energy: Callable[[TFIMSpec], float] | None = None
    estimate_memory_bytes: Callable[[TFIMSpec], int] | None = None

    @property
    def runnable_by_agent(self) -> bool:
        """Whether this entry carries a callable the agent is allowed to use."""
        return self.availability == "agent" and self.ground_state_energy is not None

    def applies_to(self, spec: TFIMSpec) -> bool:
        """Report whether this method can solve ``spec``.

        Args:
            spec: The problem to check.

        Returns:
            ``True`` if the method's applicability check raises no objection.
            Applicability is about the *problem*, not about permission: an exact
            solver applies to a small ring whether or not the caller may run it.
        """
        return self.unsupported_reason(spec) is None

    def memory_bytes(self, spec: TFIMSpec) -> int | None:
        """Predict peak memory for a run, if the method can predict it.

        Args:
            spec: The problem to cost.

        Returns:
            An approximate byte count, or ``None`` if the method declares no
            estimator because its footprint is negligible at any size.
        """
        if self.estimate_memory_bytes is None:
            return None
        return self.estimate_memory_bytes(spec)

    def solve(self, spec: TFIMSpec) -> float:
        """Run the method, if this caller is permitted to.

        Args:
            spec: The problem to solve.

        Returns:
            The ground-state energy the method reports.

        Raises:
            PermissionError: If the method is ``grader_only``. Not ``KeyError``
                or ``TypeError``: reaching this line means something tried to
                evaluate the answer it is supposed to be graded against, and the
                exception should name that rather than look like a lookup slip.
        """
        if self.ground_state_energy is None:
            raise PermissionError(
                f"{self.name} is grader-only and cannot be run from here; "
                "the agent is graded against it, so it must not evaluate it. "
                "Use src.physics.registry, which the agent cannot import."
            )
        return self.ground_state_energy(spec)


@dataclass(frozen=True, slots=True)
class Rejection:
    """A method that cannot solve the problem, together with its reason.

    Rejections are carried around rather than discarded because they are the
    most useful thing the agent can say when nothing applies: "exact
    diagonalisation was refused because the Hilbert space would be 2**30" is an
    answer, whereas "no method available" is a dead end.

    Attributes:
        method: The method that declined.
        reason: The sentence its ``unsupported_reason`` returned.
    """

    method: MethodFacts
    reason: str


@dataclass(frozen=True, slots=True)
class MethodSurvey:
    """The full picture of what can and cannot solve one specification.

    Attributes:
        spec: The problem that was surveyed.
        applicable: Methods that accepted it, in registration order. The order
            carries no preference; ranking is policy.
        rejected: Methods that declined it, each with its reason.
    """

    spec: TFIMSpec
    applicable: tuple[MethodFacts, ...]
    rejected: tuple[Rejection, ...]

    @property
    def has_applicable_method(self) -> bool:
        """Whether at least one catalogued method can solve the problem."""
        return len(self.applicable) > 0

    def names(self) -> tuple[str, ...]:
        """Return the names of the applicable methods, in registration order."""
        return tuple(facts.name for facts in self.applicable)

    def runnable(self) -> tuple[MethodFacts, ...]:
        """Applicable methods the agent is actually permitted to run.

        This, not :attr:`applicable`, is what a planning step should iterate
        over. The difference between the two is the experiment: a chain of six
        sites has an exact answer *and* a baseline, and the agent may only reach
        the second.

        Returns:
            The applicable entries carrying a callable, in registration order.
        """
        return tuple(facts for facts in self.applicable if facts.runnable_by_agent)

    def describe(self) -> str:
        """Render the survey as plain text for a prompt or a log line.

        The output is deterministic and contains no floating-point noise, so it
        is safe to include in a cached prompt and to assert on in a test.
        Grader-only entries are marked as such in words, because a menu that
        silently omitted them would leave the agent unable to explain what it is
        being measured against.

        Returns:
            A multi-line description naming every method, its cost and accuracy
            class, and -- for those that declined -- why.
        """
        lines = [self.spec.label(), "  available:"]
        if not self.applicable:
            lines.append("    (none)")
        for facts in self.applicable:
            memory = facts.memory_bytes(self.spec)
            footprint = f", ~{_format_bytes(memory)}" if memory is not None else ""
            seal = "" if facts.runnable_by_agent else ", grader only"
            lines.append(f"    {facts.name} [{facts.cost} cost, {facts.accuracy}{footprint}{seal}]")
            lines.append(f"      {facts.when_to_use}")
        if self.rejected:
            lines.append("  unavailable:")
            lines.extend(f"    {item.method.name}: {item.reason}" for item in self.rejected)
        return "\n".join(lines)


def _format_bytes(count: int) -> str:
    """Render a byte count in the largest unit that keeps it above one.

    Args:
        count: A non-negative number of bytes.

    Returns:
        A short string such as ``"1.4 MB"``. Rounded to one decimal place
        because these are order-of-magnitude estimates, not measurements.
    """
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


PFEUTY_EXACT = "pfeuty_exact"
"""Catalogue key of the closed-form free-fermion solution."""

EXACT_DIAGONALISATION = "exact_diagonalisation"
"""Catalogue key of the sparse exact diagonalisation."""

VARIATIONAL_IMAGINARY_TIME = "variational_imaginary_time"
"""Catalogue key of the sampled classical baseline."""

VARIATIONAL_QUANTUM_EIGENSOLVER = "variational_quantum_eigensolver"
"""Catalogue key of the simulated quantum circuit solver."""

_CATALOGUE: tuple[MethodFacts, ...] = (
    MethodFacts(
        name=PFEUTY_EXACT,
        summary=(
            "Closed-form free-fermion solution: a Jordan-Wigner transformation maps the "
            "chain to non-interacting fermions and a Bogoliubov rotation diagonalises them."
        ),
        when_to_use=(
            "The reference every other method is measured against, for any uniform ring "
            "of even length. It costs O(L) at every size and never approximates, which is "
            "exactly why the agent may read that it exists and may not evaluate it."
        ),
        cost="linear",
        accuracy="exact",
        availability="grader_only",
        unsupported_reason=free_fermion_unsupported_reason,
    ),
    MethodFacts(
        name=EXACT_DIAGONALISATION,
        summary=(
            "Sparse exact diagonalisation: the full Hamiltonian is assembled in CSR form "
            "and its lowest eigenpair found by Lanczos iteration."
        ),
        when_to_use=(
            "The second, independent reference -- small chains the closed form does not "
            "cover, open boundaries and odd L -- and a check on the closed form itself, "
            "since the two share no algebra. The Hilbert space grows as 2**L."
        ),
        cost="exponential",
        accuracy="exact",
        availability="grader_only",
        unsupported_reason=exact_diagonalisation_unsupported_reason,
        estimate_memory_bytes=exact_diagonalisation_memory_bytes,
    ),
    MethodFacts(
        name=VARIATIONAL_IMAGINARY_TIME,
        summary=(
            "Variational imaginary-time ansatz sampled by Metropolis on the dual classical "
            "Ising lattice, optimised by stochastic reconfiguration. Alternating "
            "exp(-a H_diag) and exp(-b H_field) pulses -- QAOA's layers at imaginary angle."
        ),
        when_to_use=(
            "The baseline the quantum arm has to beat, and the only method here that runs "
            "at any length. It returns a variational upper bound with a statistical error "
            "bar, so it can be checked without a reference -- which is what lets the agent "
            "run it without ever seeing the exact answer."
        ),
        cost="polynomial",
        accuracy="variational_bound",
        availability="agent",
        unsupported_reason=variational_imaginary_time_unsupported_reason,
        ground_state_energy=_variational_imaginary_time_energy,
    ),
    MethodFacts(
        name=VARIATIONAL_QUANTUM_EIGENSOLVER,
        summary=(
            "A parameterised quantum circuit -- alternating exp(-i a H_diag) and "
            "exp(-i b H_field) layers -- whose angles are tuned by a classical optimiser "
            "until the measured energy stops falling. Here the circuit is simulated "
            "exactly rather than measured on hardware."
        ),
        when_to_use=(
            "The method under test. It is the same alternating layer structure as the "
            "classical baseline, at real angles instead of imaginary ones, which is what "
            "makes the two comparable rather than a straw man. It returns a variational "
            "upper bound, so it can be checked without a reference; what it cannot do is "
            "beat a closed form that costs O(L)."
        ),
        cost="polynomial",
        accuracy="variational_bound",
        availability="agent",
        unsupported_reason=variational_quantum_eigensolver_unsupported_reason,
        ground_state_energy=_variational_quantum_eigensolver_energy,
    ),
)


def all_methods() -> tuple[MethodFacts, ...]:
    """Return every catalogued method, in registration order.

    Returns:
        The full catalogue. The tuple is immutable, so callers cannot extend the
        agent's menu by accident.
    """
    return _CATALOGUE


def method_names() -> tuple[str, ...]:
    """Return the names of every catalogued method, in registration order."""
    return tuple(facts.name for facts in _CATALOGUE)


def agent_methods() -> tuple[MethodFacts, ...]:
    """Return only the methods the agent is permitted to run.

    Returns:
        The entries whose :attr:`MethodFacts.availability` is ``"agent"``, in
        registration order.
    """
    return tuple(facts for facts in _CATALOGUE if facts.runnable_by_agent)


def get_method(name: str) -> MethodFacts:
    """Look up one method by its catalogue key.

    Args:
        name: A catalogue key, as returned by :func:`method_names`.

    Returns:
        The matching method's facts.

    Raises:
        KeyError: If no method is catalogued under that name. The message lists
            the valid names, because this is the failure mode when a language
            model hallucinates a solver.
    """
    for facts in _CATALOGUE:
        if facts.name == name:
            return facts
    raise KeyError(f"unknown method {name!r}; catalogued methods are {method_names()}")


def survey(spec: TFIMSpec) -> MethodSurvey:
    """Ask every catalogued method whether it can solve ``spec``.

    This is the single entry point the agent's selection step uses. One call
    yields both the menu and the reasons behind everything absent from it.

    Args:
        spec: The problem to survey.

    Returns:
        The applicable methods and the rejections, both in registration order.

    Examples:
        A periodic ring of even length is solvable every way there is:

        >>> from src.physics.model import TFIMSpec
        >>> survey(TFIMSpec(n_sites=4)).names()  # doctest: +NORMALIZE_WHITESPACE
        ('pfeuty_exact', 'exact_diagonalisation', 'variational_imaginary_time',
         'variational_quantum_eigensolver')

        But only two of those are the agent's to run. The two exact routes are the
        grader's, and reach the agent with no way to evaluate them -- which is what
        lets it *state* that an exact answer exists, as an honest feasibility report
        must, while holding no way to look at one:

        >>> [facts.name for facts in survey(TFIMSpec(n_sites=4)).runnable()]
        ['variational_imaginary_time', 'variational_quantum_eigensolver']

        An open chain falls outside the closed form, and the survey says so:

        >>> result = survey(TFIMSpec(n_sites=4, boundary="open"))
        >>> result.names()
        ('exact_diagonalisation', 'variational_imaginary_time', 'variational_quantum_eigensolver')
        >>> "periodic ring" in result.rejected[0].reason
        True
    """
    applicable: list[MethodFacts] = []
    rejected: list[Rejection] = []
    for facts in _CATALOGUE:
        reason = facts.unsupported_reason(spec)
        if reason is None:
            applicable.append(facts)
        else:
            rejected.append(Rejection(method=facts, reason=reason))
    return MethodSurvey(spec=spec, applicable=tuple(applicable), rejected=tuple(rejected))
