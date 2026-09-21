r"""Reading a finished optimisation and saying what actually went wrong.

A variational run that stops returns one number, and it looks the same whether
the optimiser found the ground state, ran out of iterations while descending, or
never moved off a plateau. The diagnosis rests on the evidence instead: the stop
reason, the shape of the energy history, the gradient norms, and for a depth
sweep whether a deeper ansatz came back worse.

Nothing here calls a language model -- these judgements have right answers, so
they belong in code a test can pin. Deciding what to *do* about one lives in
:mod:`src.agent.graph`.

One falsifier is available without touching the sealed exact answer. For
:math:`\hat H = \sum_\alpha c_\alpha \hat P_\alpha` with each Pauli string of
eigenvalues :math:`\pm 1`, every state satisfies

.. math::

    \langle \hat H \rangle \;\ge\; -\sum_\alpha \lvert c_\alpha \rvert .

An energy below that is arithmetically impossible, so it proves a bug rather
than a good run. Shot-noise floors and fidelity limits are not diagnosed:
nothing in the project measures the evidence they need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from src.physics.lattice import Lattice
from src.physics.quantum.hamiltonians import ising_chain
from src.physics.quantum.quantum_approximate_optimisation import DepthSweep
from src.physics.quantum.variational_eigensolver import VqeResult

Signal = Literal[
    "healthy",
    "still_descending",
    "barren_plateau",
    "local_minimum",
    "never_started",
    "below_variational_bound",
]
"""What the run looks like, named for the cause rather than for the symptom.

Ordered here roughly from "nothing to do" to "stop everything". The names are the
ones a practitioner would use out loud, which matters because they are quoted
verbatim into the feasibility report a non-physicist reads.
"""

PLATEAU_GRADIENT_NORM = 1e-6
"""Gradient norm at the first step below which the landscape counts as flat.

Well above the optimiser's own ``gradient_tolerance`` of :math:`10^{-8}`, because
these are two different questions. The optimiser asks "am I standing still?"; this
asks "was there ever anywhere to go?", and a run that begins with a gradient this
small was never going to move regardless of how long it was given.
"""

DESCENT_PER_STEP = 1e-8
"""Energy drop at the final step above which a run counts as still descending.

Compared against the *last* step rather than the average, because a run that fell
quickly and then flattened has converged in every sense that matters, and averaging
over its whole history would hide that behind the early progress.
"""

BOUND_SLACK = 1e-9
"""How far below the coefficient bound an energy may sit before it is called a bug.

Room for floating-point summation over the Pauli terms and nothing more. The bound
is exact mathematics, so the only legitimate reason to be under it is rounding.
"""


class DiagnosisRecord(TypedDict):
    """The flat form of a :class:`Diagnosis`, as it appears in logs and tool results.

    Spelled out as a ``TypedDict`` rather than left as ``dict[str, object]`` because
    this mapping crosses three boundaries where a mistyped key is silent: into a JSON
    log line, into a language model's context, and into the run table the report is
    built from. Naming the keys means the type checker catches a rename that a
    string-keyed dictionary would carry all the way to a missing column in the report.

    Attributes:
        signal: The recognised failure mode, matching :data:`Signal`.
        evidence: One sentence carrying the number the call rests on.
        repair: What to change before running again.
        is_actionable: Whether another attempt could plausibly do better.
    """

    signal: Signal
    evidence: str
    repair: str
    is_actionable: bool


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """What the run did, what shows it, and what to do about it.

    The three fields are separated on purpose. ``signal`` is what the graph
    branches on and what the evaluation harness counts; ``evidence`` is the number
    that justifies it, so a reader can disagree with the conclusion while checking
    the arithmetic; ``repair`` is the instruction handed back to the planner.

    Attributes:
        signal: Which of the recognised failure modes this is.
        evidence: The measurement the call rests on, written for a reader who has
            never optimised a circuit. One sentence, always carrying a number.
        repair: What to change before running again, or why running again will not
            help. Phrased as an action, since it is read by whatever picks the next
            configuration.
        is_actionable: Whether trying again with a different configuration could
            plausibly do better. ``False`` covers both "this is already as good as
            this family gets" and "this is a bug, and another run will reproduce it".
    """

    signal: Signal
    evidence: str
    repair: str
    is_actionable: bool

    def describe(self) -> DiagnosisRecord:
        """Render the diagnosis as plain data for a log line or a tool result.

        Returns:
            A flat mapping of primitives, so it survives being serialised into a
            JSON log or handed to a language model without a custom encoder.
        """
        return {
            "signal": self.signal,
            "evidence": self.evidence,
            "repair": self.repair,
            "is_actionable": self.is_actionable,
        }


def coefficient_bound(
    n_sites: int,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: Literal["periodic", "open"] = "open",
    lattice: Lattice | None = None,
) -> float:
    r"""The lowest energy the Hamiltonian could conceivably have.

    Sums the absolute values of the Pauli coefficients and negates the total. For
    this chain that is :math:`-(J n_\text{bonds} + hL + gL)`, and it is reached only
    in the fictional case where every term is simultaneously minimised -- which the
    Ising terms and the transverse field cannot be, since they do not commute. So
    the true ground state always sits strictly above it.

    Built by asking :func:`~src.physics.quantum.hamiltonians.ising_chain` for the
    operator and summing what it returns, rather than by writing the formula out
    again here. The formula would be a second source of truth for the same fact,
    and the two would disagree the first time a term was added.

    Args:
        n_sites: Number of spins in the chain.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        lattice: The shape the spins sit on, or ``None`` for a chain. It changes the
            bond count and therefore the bound, and a bound computed for the wrong
            shape is worse than none: a lattice has more bonds than a chain of the
            same width, so the chain's bound sits *above* the lattice's true energy
            and :func:`diagnose_run` would report every correct run as a bug.

    Returns:
        A strict lower bound on the ground-state energy.

    Examples:
        Four sites, open, all couplings unity: three bonds and four field terms.

        >>> coefficient_bound(n_sites=4, boundary="open")
        -7.0
    """
    hamiltonian = ising_chain(
        n_sites=n_sites,
        coupling=coupling,
        transverse_field=transverse_field,
        longitudinal_field=longitudinal_field,
        boundary=boundary,
        lattice=lattice,
    )
    return -hamiltonian.coefficient_l1()


def diagnose_run(result: VqeResult, lower_bound: float) -> Diagnosis:
    """Judge one finished optimisation from its own record.

    The checks run in order of severity and the first match wins, because the
    orderings matter: an energy below the bound is a bug whatever the stop reason
    says, and a plateau is worth reporting even though a plateaued run also happens
    to satisfy the optimiser's stationarity test.

    Args:
        result: The completed run, carrying its stop reason and both histories.
        lower_bound: The value from :func:`coefficient_bound` for this problem. Passed
            in rather than recomputed so that a caller sweeping twenty depths on one
            Hamiltonian pays for it once.

    Returns:
        The diagnosis, always populated. There is no "unknown" outcome: a run that
        matches none of the failure modes is healthy, and saying so is a result.
    """
    if result.energy < lower_bound - BOUND_SLACK:
        return Diagnosis(
            signal="below_variational_bound",
            evidence=(
                f"the run reported {result.energy:.6f}, which is below the "
                f"arithmetic floor of {lower_bound:.6f} that no state can go under"
            ),
            repair=(
                "do not run this configuration again -- an impossible energy means "
                "the estimator, the angle convention or the Hamiltonian is wrong, "
                "and a second run will reproduce the same bug"
            ),
            is_actionable=False,
        )

    if result.stop_reason == "depth_zero":
        return Diagnosis(
            signal="never_started",
            evidence="the circuit had no layers, so nothing was optimised",
            repair="give the ansatz at least one layer",
            is_actionable=True,
        )

    if result.stop_reason == "iteration_limit" and _final_drop(result) > DESCENT_PER_STEP:
        return Diagnosis(
            signal="still_descending",
            evidence=(
                f"the optimiser used all {result.n_iterations} of its steps and the "
                f"energy was still falling by {_final_drop(result):.2e} at the last one"
            ),
            repair="raise the iteration cap -- this run was cut off, not finished",
            is_actionable=True,
        )

    if _started_flat(result):
        return Diagnosis(
            signal="barren_plateau",
            evidence=(
                f"the gradient norm was {result.gradient_norm_history[0]:.2e} at the "
                f"first step and the energy moved by {result.improvement:.2e} in total"
            ),
            repair=(
                "the landscape is flat where this run started, so more iterations "
                "cannot help: use fewer layers, or start from the adiabatic ramp "
                "rather than from random angles"
            ),
            is_actionable=True,
        )

    if result.stop_reason == "optimiser_failed":
        return Diagnosis(
            signal="local_minimum",
            evidence="the optimiser stopped without reaching a stationary point",
            repair="retry from a different starting schedule",
            is_actionable=True,
        )

    return Diagnosis(
        signal="healthy",
        evidence=(
            f"the run stopped at {result.stop_reason} after {result.n_iterations} steps, "
            f"having improved the energy by {result.improvement:.3e}"
        ),
        repair="nothing to repair -- add layers if a lower energy is wanted",
        is_actionable=True,
    )


def diagnose_sweep(sweep: DepthSweep, lower_bound: float) -> Diagnosis:
    """Judge a whole depth sweep, which can see one thing a single run cannot.

    A deeper ansatz strictly contains every shallower one: set the extra angles to
    zero and the deeper circuit *is* the shallower circuit. So a deeper depth that
    comes back with a worse energy has not found a limit of the family -- it has
    found a local minimum, and that is a proof rather than a suspicion. It is the
    only failure mode in this module diagnosed with certainty rather than by
    threshold, and it is invisible to any one run.

    Where the sweep is well behaved, the best run is judged on its own terms by
    :func:`diagnose_run`.

    Args:
        sweep: The completed sweep, carrying the depths it tried and any regressions.
        lower_bound: The value from :func:`coefficient_bound` for this problem.

    Returns:
        The diagnosis for the sweep as a whole.
    """
    if sweep.regressions:
        worse = ", ".join(str(depth) for depth in sweep.regressions)
        return Diagnosis(
            signal="local_minimum",
            evidence=(
                f"depth {worse} returned a worse energy than the depth below it, which "
                "is impossible for the family itself -- a deeper circuit contains every "
                "shallower one -- so the optimiser was trapped, not the ansatz"
            ),
            repair=(
                "restart the trapped depths from a different ramp time, or warm-start "
                "them from the best schedule found so far"
            ),
            is_actionable=True,
        )
    return diagnose_run(sweep.best, lower_bound)


def _final_drop(result: VqeResult) -> float:
    """How much the energy fell over the last accepted step.

    Args:
        result: The completed run.

    Returns:
        The drop, positive when the energy was still improving. Zero when the history
        is too short to have a last step to measure.
    """
    history = result.energy_history
    if len(history) < 2:
        return 0.0
    return history[-2] - history[-1]


def _started_flat(result: VqeResult) -> bool:
    """Whether the run began on a plateau and never left it.

    Both halves are required. A tiny opening gradient on its own is what a warm start
    looks like when it is already nearly right, and calling that a barren plateau
    would condemn the best runs in the project. It is a plateau only if the run also
    had nowhere to go: the gradient was negligible *and* the energy stayed put.

    Args:
        result: The completed run.

    Returns:
        ``True`` if the opening gradient was negligible and the energy barely moved.
    """
    if not result.gradient_norm_history:
        return False
    return (
        result.gradient_norm_history[0] < PLATEAU_GRADIENT_NORM
        and abs(result.improvement) < DESCENT_PER_STEP
    )
