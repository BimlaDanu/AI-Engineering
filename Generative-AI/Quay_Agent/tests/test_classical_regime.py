r"""The honesty gate: can the classical baseline be believed on this shape?

The phase of the two-dimensional work that had to come first.
The risk this file exists to remove is a **false "yes"** -- a quantum advantage
reported because the number it was measured against was degraded rather than
because the circuit did anything. That failure is silent by construction: a
degraded classical baseline is looser and noisier, so it is *easier to beat*, and
every symptom points the wrong way.

Three groups, and each checks a different kind of claim:

1. **The refusal.** A shape the sampler cannot represent must raise rather than
   return the number it would have produced for a different problem.
2. **The verdict.** A lead over a baseline that cannot carry one must not be
   reported as a lead -- and, asymmetrically, a *loss* to such a baseline still
   must be, since the degradation only widens that gap.
3. **The physics claim, checked rather than repeated.** The plan originally said
   frustration gives this model a sign problem. It does not, and
   :func:`test_this_model_has_no_sign_problem_on_any_shape` proves it from an
   assembled matrix. Getting that wrong would not be a wording defect: the whole
   argument for extending to 2D rests on what frustration actually costs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agent.diagnosis import Diagnosis
from src.agent.state import (
    BaselineResult,
    CampaignState,
    FormalModel,
    Request,
    RunRecord,
    new_campaign,
)
from src.agent.verdict import judge
from src.physics.classical.regime import (
    BaselineUnavailableError,
    Regime,
    assess,
)
from src.physics.lattice import Boundary, Geometry, Lattice
from src.physics.model import TFIMSpec
from src.physics.quantum.hamiltonians import PauliSum, PauliTerm

# --------------------------------------------------------------------------
# What the sampler will and will not answer
# --------------------------------------------------------------------------


def test_the_calibration_case_is_sound() -> None:
    """An unfrustrated chain is the regime the sampler was built and checked on."""
    standing = assess(Lattice("chain", 1, 12), coupling=1.0)

    assert standing.standing == "sound"
    assert standing.is_trustworthy
    assert standing.can_run
    assert standing.confidence == "high"


@pytest.mark.parametrize("geometry", ["square", "triangular"])
def test_a_two_dimensional_lattice_is_refused_rather_than_flattened(geometry: str) -> None:
    """The concrete landmine Phase 1 exists to remove.

    ``optimise`` builds its dual lattice from ``spec.n_sites`` alone, so a nine-site
    lattice would otherwise be sampled as a *line* of nine and the answer reported
    as the classical result for the lattice. Nothing would look wrong: the energy
    would be in the right range and the error bar small.
    """
    standing = assess(Lattice(geometry, 3, 3), coupling=1.0)  # type: ignore[arg-type]

    assert standing.standing == "unavailable"
    assert not standing.can_run
    # It must say what it cannot do, and name the shape it was asked for -- a
    # refusal a reader cannot act on is barely better than a wrong number.
    assert "3x3" in standing.reason
    assert "imaginary time" in standing.detail


def test_the_refusal_is_an_exception_and_not_a_number_with_a_warning() -> None:
    """A caveat beside a plausible energy is read as a formality. There is no energy."""
    from src.physics.classical.variational_imaginary_time import ground_state_energy

    spec = TFIMSpec(n_sites=4, boundary="open")
    with pytest.raises(BaselineUnavailableError) as raised:
        ground_state_energy(spec, geometry=Lattice("square", 2, 2))

    assert raised.value.regime.standing == "unavailable"


def test_a_chain_still_runs_and_reports_its_own_standing() -> None:
    """The gate must not cost the case it was calibrated on."""
    from src.physics.classical.variational_imaginary_time import (
        OptimiserConfig,
        SamplingConfig,
        ground_state_energy,
    )

    result = ground_state_energy(
        TFIMSpec(n_sites=4, boundary="open"),
        sampling=SamplingConfig(warmup=20, sweeps=4, bins=4),
        optimiser=OptimiserConfig(iterations=4),
    )

    assert result.regime.standing == "sound"
    assert result.energy < 0.0


# --------------------------------------------------------------------------
# Frustration: runnable, and not to be believed to its error bar
# --------------------------------------------------------------------------


def test_an_odd_ring_is_refused_on_the_same_ground_the_catalogue_refuses_it() -> None:
    """Two statements of one rule, pinned together so they cannot drift.

    The checkerboard sweep updates every other site at once, which is correct only
    if no two sites in a group are connected -- and an odd ring closes parity onto
    itself. :mod:`src.physics.method_catalogue` already refused this and
    :func:`assess` has to agree, because a shape declared runnable in one place and
    unsupported in another is a shape that will be run.
    """
    from src.physics.method_catalogue import variational_imaginary_time_unsupported_reason

    for n_sites in (5, 7, 9):
        ring = Lattice("chain", 1, n_sites, "periodic")
        assert not ring.is_bipartite
        assert assess(ring, coupling=1.0).standing == "unavailable"
        assert (
            variational_imaginary_time_unsupported_reason(
                TFIMSpec(n_sites=n_sites, boundary="periodic")
            )
            is not None
        )

    even = Lattice("chain", 1, 8, "periodic")
    assert assess(even, coupling=-1.0).standing == "sound"
    assert (
        variational_imaginary_time_unsupported_reason(TFIMSpec(n_sites=8, boundary="periodic"))
        is None
    )


def test_no_shape_the_sampler_accepts_can_reach_the_degraded_branch() -> None:
    """The honest statement about which branch is live, asserted rather than assumed.

    Frustration needs a graph that does not two-colour, and every such graph is
    refused *earlier* by a rule that is not about frustration at all -- a 2D lattice
    has no third axis, an odd ring breaks the checkerboard. So the frustration rule
    is currently reached by nothing, and that is pinned here so a reader does not
    mistake a guard installed ahead of the hazard for live behaviour. When a 2D
    sampler exists this test is what will fail, which is the right place to be told.
    """
    shapes: tuple[tuple[Geometry, int, int], ...] = (
        ("chain", 1, 4),
        ("chain", 1, 5),
        ("chain", 1, 8),
        ("square", 2, 2),
        ("square", 3, 3),
        ("triangular", 2, 2),
        ("triangular", 3, 3),
    )
    boundaries: tuple[Boundary, ...] = ("open", "periodic")
    reachable = {
        assess(Lattice(geometry, rows, cols, boundary), coupling).standing
        for geometry, rows, cols in shapes
        for boundary in boundaries
        for coupling in (1.0, -1.0)
    }

    assert reachable == {"sound", "unavailable"}


def test_a_refused_frustrated_shape_is_told_both_reasons() -> None:
    """A refusal that gave only the first reason would mislead about what it needs.

    "No third axis in the dual model" reads as an implementation gap somebody could
    close. It is one -- and closing it would not make the comparison believable on
    this shape, because frustration degrades the error bar independently. A reader
    is owed both facts in the one place they are told nothing ran.
    """
    frustrated = assess(Lattice("triangular", 3, 3), coupling=-1.0)
    unfrustrated = assess(Lattice("triangular", 3, 3), coupling=1.0)

    assert frustrated.standing == unfrustrated.standing == "unavailable"
    assert "third axis" in frustrated.detail
    assert "even with that axis built" in frustrated.detail
    # And the caveat is attached to frustration, not to being two-dimensional.
    assert "even with that axis built" not in unfrustrated.detail


def test_the_reason_names_slow_mixing_and_denies_the_sign_problem() -> None:
    """The wording is load-bearing, because the plain-language reason is wrong.

    "Frustration causes a sign problem" is true of frustrated Heisenberg models
    and false of this one -- see
    :func:`test_this_model_has_no_sign_problem_on_any_shape`. A report that gave
    the familiar reason would be teaching a reader with no physics something
    incorrect, in the one document whose job is to be trustworthy.
    """
    detail = assess(Lattice("triangular", 3, 3), coupling=-1.0).detail

    assert "one spin at a time" in detail
    assert "under-reported" in detail
    assert "no sign problem" in detail


def test_a_bipartite_lattice_is_never_frustrated_by_either_sign() -> None:
    """The property that makes the FM/AFM cross-check in Phase 2 meaningful."""
    for shape in (Lattice("chain", 1, 8), Lattice("chain", 1, 8, "periodic")):
        assert not shape.frustrated_by(-1.0)
        assert assess(shape, coupling=-1.0).standing == "sound"


def test_zero_field_is_refused_because_nothing_is_being_sampled() -> None:
    """At ``h = 0`` the estimator is multiplied by zero: the model is not quantum."""
    standing = assess(Lattice("chain", 1, 8), coupling=1.0, transverse_field=0.0)

    assert standing.standing == "unavailable"
    assert "not a quantum one" in standing.reason


# --------------------------------------------------------------------------
# The verdict: a lead over a degraded baseline is not a lead
# --------------------------------------------------------------------------

HEALTHY = Diagnosis(signal="healthy", evidence="converged", repair="nothing", is_actionable=True)


def campaign_with(
    *,
    confidence: str,
    quantum_energy: float,
    classical_energy: float = -1.20,
    classical_error: float = 0.01,
) -> CampaignState:
    """A campaign that passes every other screen, so only this one is under test."""
    state = new_campaign(Request(text="is it worth it?"), shot_budget=100_000)
    # A non-zero longitudinal field, so the closed-form screen does not fire first
    # and steal the verdict -- it outranks this one, correctly, because a problem
    # with an exact answer needs no comparison at all.
    state["model"] = FormalModel(n_sites=6, longitudinal_field=0.3)
    state["runs"] = (
        RunRecord(
            label="hva-p2",
            family="hva",
            depth=2,
            two_qubit_depth=8,
            energy=quantum_energy * 6,
            energy_per_site=quantum_energy,
            shots_spent=1000,
            diagnosis=HEALTHY,
        ),
    )
    state["classical"] = BaselineResult(
        method="variational_imaginary_time",
        energy_per_site=classical_energy,
        energy_error=classical_error,
        n_measurements=400,
        confidence=confidence,  # type: ignore[arg-type]
        caveat="the sampler moves one spin at a time and cannot cross the barriers",
    )
    return state


def test_a_lead_over_a_degraded_baseline_is_not_reported_as_a_lead() -> None:
    """Phase 1's headline requirement.

    The same campaign that earns a "conditional" on a sound baseline must not earn
    one on a degraded baseline -- because the lead is measured in units of the
    error bar, and that is the quantity known to be wrong.
    """
    trusted = judge(campaign_with(confidence="high", quantum_energy=-1.30))
    assert trusted[0].call == "conditional"
    assert trusted[1] == "comparison"

    verdict, decided_by = judge(campaign_with(confidence="low", quantum_energy=-1.30))

    assert decided_by == "baseline_not_trustworthy"
    assert verdict.confidence == "low"
    assert "could not be made reliably" in verdict.summary
    # It must not read as a promising result. The call is the existing vocabulary's
    # "unsettled", so the prose has to carry the distinction.
    assert "not a verdict about whether a quantum computer helps" in verdict.summary


def test_the_refusal_says_what_would_make_the_comparison_believable() -> None:
    """A verdict with no crossover condition expires silently and teaches nothing."""
    verdict, _ = judge(campaign_with(confidence="low", quantum_energy=-1.30))

    assert "decorrelate" in verdict.crossover_condition


def test_a_loss_to_a_degraded_baseline_is_still_reported_as_a_loss() -> None:
    """The asymmetry, and the substance of the screen.

    A degraded baseline fails in a *known* direction: looser and noisier, so
    easier to beat. One that beats the circuit anyway has settled the question --
    the true classical answer is lower still, which only widens the gap. Calling
    that "cannot tell" would be false modesty, and it would let the screen suppress
    the honest "no" that this project's verdicts are mostly made of.
    """
    verdict, decided_by = judge(campaign_with(confidence="low", quantum_energy=-0.50))

    assert decided_by == "comparison"
    assert verdict.call == "no"


def test_a_tie_against_a_degraded_baseline_is_refused_not_called_a_tie() -> None:
    """A tie is a claim about the error bar too, so it needs the same bar to be real."""
    _, decided_by = judge(campaign_with(confidence="low", quantum_energy=-1.20))

    assert decided_by == "baseline_not_trustworthy"


def test_a_missing_baseline_still_outranks_an_untrustworthy_one() -> None:
    """No number is a cleaner statement than an unreliable one, and it fires first."""
    state = campaign_with(confidence="low", quantum_energy=-1.30)
    state["classical"] = None

    _, decided_by = judge(state)

    assert decided_by == "no_classical_comparison"


def test_an_error_bar_that_is_not_a_number_cannot_carry_a_comparison() -> None:
    """A live bug this phase found, and the route that actually reaches the screen.

    ``_bin_error`` returns ``nan`` for a single bin **on purpose** -- one bin carries
    no information about its own spread, and returning ``0.0`` would let a caller
    divide by it and call the result significant. The verdict did not honour that
    choice. Every comparison against ``nan`` is false, so a signed margin checked
    against ``> nan`` and then ``< -nan`` fell through both branches into the tie
    case, and the reader was told *"the gap of 0.100000 is inside the classical error
    bar"* beside a printed ``± nan``. Both halves of that sentence were false.
    """
    verdict, decided_by = judge(
        campaign_with(confidence="high", quantum_energy=-1.30, classical_error=float("nan"))
    )

    assert decided_by == "baseline_not_trustworthy"
    assert "no usable uncertainty at all" in verdict.summary
    # And it must not print the non-number as though it were a quantity.
    assert "nan" not in verdict.summary


def test_a_zero_error_bar_is_refused_for_the_same_reason() -> None:
    """A lead measured in units of zero is infinite, so every result would win."""
    _, decided_by = judge(
        campaign_with(confidence="high", quantum_energy=-1.30, classical_error=0.0)
    )

    assert decided_by == "baseline_not_trustworthy"


def test_an_unusable_bar_is_refused_even_when_the_classical_arm_appears_to_win() -> None:
    """The one place the asymmetry does not apply.

    A *degraded but usable* bar still shows the direction of a decisive classical
    win, and believing the bar only widens that gap -- so that case is let through
    to the comparison. An unusable bar shows no direction at all, so there is
    nothing to let through, however the two numbers happen to fall.
    """
    _, decided_by = judge(
        campaign_with(confidence="high", quantum_energy=-0.50, classical_error=float("nan"))
    )

    assert decided_by == "baseline_not_trustworthy"


def test_a_refused_baseline_gets_a_crossover_condition_that_can_be_acted_on() -> None:
    """The wrong sentence and the right one used to arrive here looking identical.

    A campaign that *skipped* the baseline should be told to run it. A campaign whose
    problem the baseline cannot represent must not be, because following that advice
    is impossible -- and the reader would reasonably conclude the application simply
    forgot a step. Same defect as :func:`src.agent.verdict._named_a_chain_too_long`,
    one layer down.
    """
    from src.agent.state import Belief
    from src.agent.verdict import BASELINE_REFUSAL_MARK

    standing = assess(Lattice("triangular", 3, 3), coupling=-1.0)
    assert BASELINE_REFUSAL_MARK in standing.reason

    state = campaign_with(confidence="high", quantum_energy=-1.30)
    state["classical"] = None
    state["beliefs"] = (Belief(claim=standing.reason, support=(standing.detail,)),)

    verdict, decided_by = judge(state)

    assert decided_by == "no_classical_comparison"
    assert "3x3 triangular" in verdict.summary
    assert "not the same thing as a quantum computer losing" in verdict.summary
    assert "run the classical method" not in verdict.crossover_condition
    assert "nobody has measured what it would have to beat" in verdict.crossover_condition


def test_a_skipped_baseline_still_gets_told_to_run_one() -> None:
    """The guard against the fix swallowing the case it was carved out of."""
    state = campaign_with(confidence="high", quantum_energy=-1.30)
    state["classical"] = None

    verdict, decided_by = judge(state)

    assert decided_by == "no_classical_comparison"
    assert "run the classical method on the same problem" in verdict.crossover_condition


def test_the_report_says_what_it_could_not_check() -> None:
    """The caveat gets a heading, not a footnote beside a six-decimal number."""
    from src.agent.report import compose

    state = campaign_with(confidence="low", quantum_energy=-1.30)
    verdict, decided_by = judge(state)
    state["verdict"] = verdict

    written = compose(state, decided_by)

    assert "### What this comparison could not check" in written
    assert "smaller than the real one" in written
    assert "one spin at a time" in written


def test_a_sound_baseline_adds_no_caveat_section() -> None:
    """The guard against a caveat that is always there and therefore never read."""
    from src.agent.report import compose

    state = campaign_with(confidence="high", quantum_energy=-1.30)
    verdict, decided_by = judge(state)
    state["verdict"] = verdict

    written = compose(state, decided_by)

    assert "could not check" not in written


# --------------------------------------------------------------------------
# The physics claim, checked rather than repeated
# --------------------------------------------------------------------------


def assembled(lattice: Lattice, coupling: float, field: float = 1.0) -> np.ndarray:
    """The dense Hamiltonian, built here from the bond list and no project solver.

    Args:
        lattice: The shape, which supplies only the bond list.
        coupling: ``J``. Negative is antiferromagnetic.
        field: ``h``.

    Returns:
        The dense matrix in the ``sigma^z`` basis.
    """
    terms = [PauliTerm.from_mapping(-coupling, {i: "Z", j: "Z"}) for i, j in lattice.bonds()]
    terms += [PauliTerm.from_mapping(-field, {i: "X"}) for i in range(lattice.n_sites)]
    return np.asarray(
        PauliSum(n_qubits=lattice.n_sites, terms=tuple(terms)).to_matrix().toarray().real
    )


@pytest.mark.parametrize(
    ("geometry", "rows", "cols"),
    [("chain", 1, 6), ("square", 2, 2), ("triangular", 3, 3)],
)
@pytest.mark.parametrize("coupling", [1.0, -1.0])
def test_this_model_has_no_sign_problem_on_any_shape(
    geometry: str, rows: int, cols: int, coupling: float
) -> None:
    r"""The claim the plan got wrong, disproved from the matrix itself.

    "Frustration gives quantum Monte Carlo a sign problem" is a true and famous
    statement about frustrated *Heisenberg* antiferromagnets, and false about this
    model. The transverse-field Ising Hamiltonian is diagonal in the
    :math:`\sigma^z` basis apart from the field, so every off-diagonal element is
    :math:`-h` -- negative for any :math:`h > 0`, on **any** graph and for
    **either** sign of :math:`J`. A Hamiltonian with no positive off-diagonal
    element is *stoquastic*: :math:`e^{-\beta \hat H}` has non-negative entries,
    the path-integral weights are non-negative, and there is no sign to cancel.
    Frustration moves the *diagonal*, which no weight's sign depends on.

    This is asserted rather than commented because it is what licenses the wording
    in :mod:`src.physics.classical.regime`, and because a reader with no physics
    has no way to check it other than seeing it fail if it were untrue.
    """
    lattice = Lattice(geometry, rows, cols)  # type: ignore[arg-type]
    matrix = assembled(lattice, coupling)
    off_diagonal = matrix - np.diag(np.diag(matrix))

    assert off_diagonal.max() <= 0.0
    # And the elements are not merely non-positive by accident of being zero.
    assert off_diagonal.min() == pytest.approx(-1.0)


def test_frustration_costs_energy_which_is_what_it_actually_does() -> None:
    """The true consequence of frustration, as a number rather than a word.

    On a triangular lattice the antiferromagnet cannot satisfy every bond at once,
    so its ground state sits strictly above the ferromagnet's. That is what
    frustration *is*, and it is checkable without any reference to sampling.
    """
    lattice = Lattice("triangular", 3, 3)
    ferromagnetic = float(np.linalg.eigvalsh(assembled(lattice, 1.0))[0])
    antiferromagnetic = float(np.linalg.eigvalsh(assembled(lattice, -1.0))[0])

    assert lattice.frustrated_by(-1.0)
    assert antiferromagnetic > ferromagnetic


def test_a_bipartite_lattice_gives_the_two_signs_the_same_spectrum() -> None:
    """The free cross-check bipartiteness buys, and Phase 2's first property test.

    Flipping every spin on one sublattice maps the ferromagnet onto the
    antiferromagnet exactly, so the two must agree to machine precision. They do
    not agree on a triangular lattice, which is the same fact from the other side.
    """
    for shape in (Lattice("chain", 1, 6), Lattice("square", 2, 2)):
        assert shape.is_bipartite
        assert float(np.linalg.eigvalsh(assembled(shape, 1.0))[0]) == pytest.approx(
            float(np.linalg.eigvalsh(assembled(shape, -1.0))[0])
        )


def test_the_standing_of_every_shape_is_stated_rather_than_defaulted() -> None:
    """No shape may fall through to a standing nobody chose.

    A vacuity guard: the parametrisation above would still pass if ``assess``
    returned ``sound`` for everything, so this asserts the three standings are all
    actually reachable and that every geometry in the project resolves to one.
    """
    reached = {
        assess(Lattice("chain", 1, 8), 1.0).standing,
        assess(Lattice("chain", 1, 9, "periodic"), -1.0).standing,
        assess(Lattice("triangular", 3, 3), 1.0).standing,
    }

    assert reached == {"sound", "unavailable"}
    for geometry in ("chain", "square", "triangular"):
        rows = 1 if geometry == "chain" else 3
        standing = assess(Lattice(geometry, rows, 3), 1.0)
        assert isinstance(standing, Regime)
        assert standing.reason and standing.detail
