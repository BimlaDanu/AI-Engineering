"""The verdict screens: does the call hold when it would be easier not to?

These are the tests that make the project's central claim checkable. Each one sets
up a campaign that a helpful assessor would want to call a success, and asserts
that the screens refuse to. If any of them can be made to pass by an agent that
simply tries harder to be encouraging, the screen it guards is not doing its job.

The last group is the important one. It runs the same campaign under all three
framings and asserts the verdict is identical, which is a property the code has by
construction -- none of the screens can see the framing -- and which is worth
pinning anyway, because the obvious "improvement" to any of these functions is to
pass them the request text.
"""

from __future__ import annotations

import pytest

from src.agent.diagnosis import Diagnosis
from src.agent.state import (
    BaselineResult,
    CampaignState,
    FormalModel,
    Framing,
    Request,
    RuledOut,
    RunRecord,
    new_campaign,
)
from src.agent.verdict import SCREENS, judge
from src.physics.quantum.method_race import MethodRace, MethodRun

HEALTHY = Diagnosis(signal="healthy", evidence="converged", repair="nothing", is_actionable=True)
IMPOSSIBLE = Diagnosis(
    signal="below_variational_bound",
    evidence="below the arithmetic floor",
    repair="find the bug",
    is_actionable=False,
)


def make_run(label: str, energy_per_site: float, diagnosis: Diagnosis = HEALTHY) -> RunRecord:
    return RunRecord(
        label=label,
        family="hva",
        depth=2,
        two_qubit_depth=8,
        energy=energy_per_site * 6,
        energy_per_site=energy_per_site,
        shots_spent=1000,
        diagnosis=diagnosis,
    )


def make_campaign(
    *,
    framing: Framing = "neutral",
    longitudinal_field: float = 0.3,
    quantum_energy: float | None = -1.30,
    classical_energy: float | None = -1.20,
    classical_error: float = 0.01,
    diagnosis: Diagnosis = HEALTHY,
) -> CampaignState:
    """A campaign with every screen passing, so one thing at a time can be broken."""
    state = new_campaign(Request(text="is it worth it?", framing=framing), shot_budget=100_000)
    state["model"] = FormalModel(n_sites=6, longitudinal_field=longitudinal_field)
    if quantum_energy is not None:
        state["runs"] = (make_run("hva-p2", quantum_energy, diagnosis),)
    if classical_energy is not None:
        state["classical"] = BaselineResult(
            method="variational_imaginary_time",
            energy_per_site=classical_energy,
            energy_error=classical_error,
            n_measurements=400,
        )
    return state


# ── the screens, one at a time ─────────────────────────────────────────────────


def test_a_closed_form_problem_gets_a_no_however_well_the_circuit_ran() -> None:
    """The screen the whole project turns on.

    The quantum run here is excellent and beats the classical baseline outright.
    It still has to lose, because the problem has an exact answer that a laptop
    produces instantly, and nothing a circuit does can improve on that.
    """
    state = make_campaign(longitudinal_field=0.0, quantum_energy=-5.0, classical_energy=-1.0)

    verdict, decided_by = judge(state)

    assert verdict.call == "no"
    assert decided_by == "closed_form_available"
    assert "closed form" in verdict.summary


def test_a_missing_classical_baseline_cannot_be_argued_past() -> None:
    """Even a perfect quantum result gets a no with nothing to compare it against."""
    state = make_campaign(quantum_energy=-99.0, classical_energy=None)

    verdict, decided_by = judge(state)

    assert verdict.call == "no"
    assert decided_by == "no_classical_comparison"


def test_an_impossible_energy_outranks_a_winning_comparison() -> None:
    """A broken run must not be judged on how good its number looked."""
    state = make_campaign(quantum_energy=-500.0, diagnosis=IMPOSSIBLE)

    verdict, decided_by = judge(state)

    assert verdict.call == "no"
    assert decided_by == "impossible_energy"
    assert "fault in the calculation" in verdict.summary


def test_a_campaign_that_never_formalised_anything_gets_a_no() -> None:
    state = new_campaign(Request(text="something vague"), shot_budget=10)

    verdict, decided_by = judge(state)

    assert verdict.call == "no"
    assert decided_by == "no_model"


def test_a_campaign_with_no_completed_run_gets_a_no() -> None:
    state = make_campaign(quantum_energy=None)
    state["ruled_out"] = (RuledOut(label="hva-p8", reason="too deep"),)

    verdict, decided_by = judge(state)

    assert verdict.call == "no"
    assert decided_by == "no_usable_quantum_run"
    assert "rejected before running" in verdict.summary


# ── the comparison itself ──────────────────────────────────────────────────────


def test_beating_the_baseline_by_more_than_its_error_bar_is_conditional_not_go() -> None:
    """A simulated lead is a reason to keep going, not a reason to buy device time."""
    state = make_campaign(quantum_energy=-1.30, classical_energy=-1.20, classical_error=0.01)

    verdict, decided_by = judge(state)

    assert decided_by == "comparison"
    assert verdict.call == "conditional"
    assert "hardware" in verdict.crossover_condition


def test_a_lead_inside_the_error_bar_is_not_a_lead() -> None:
    """The easiest way to fake an advantage is to ignore the uncertainty."""
    state = make_campaign(quantum_energy=-1.205, classical_energy=-1.20, classical_error=0.05)

    verdict, decided_by = judge(state)

    assert decided_by == "comparison"
    assert verdict.call == "no"


def test_losing_to_the_baseline_is_a_no() -> None:
    state = make_campaign(quantum_energy=-1.10, classical_energy=-1.20)

    verdict, _ = judge(state)

    assert verdict.call == "no"


# ── the property the project exists to measure ─────────────────────────────────


@pytest.mark.parametrize("framing", ["neutral", "vendor", "skeptical"])
def test_the_verdict_does_not_move_with_the_framing(framing: Framing) -> None:
    """Same numbers, three voices, one answer.

    True by construction -- no screen is given the request text -- and pinned here
    because the natural way to "improve" these functions is to hand them the
    question, at which point the property quietly disappears.
    """
    state = make_campaign(framing=framing)

    verdict, decided_by = judge(state)

    assert verdict.call == "conditional"
    assert decided_by == "comparison"


@pytest.mark.parametrize("framing", ["neutral", "vendor", "skeptical"])
def test_a_closed_form_problem_stays_a_no_under_pressure(framing: Framing) -> None:
    """The case an enthusiastic assessor would most want to talk itself out of."""
    state = make_campaign(framing=framing, longitudinal_field=0.0, quantum_energy=-9.0)

    verdict, decided_by = judge(state)

    assert verdict.call == "no"
    assert decided_by == "closed_form_available"


# ── properties every verdict must have ─────────────────────────────────────────


def test_every_verdict_states_what_would_change_it() -> None:
    """A verdict with no stated condition expires silently as hardware improves."""
    campaigns = [
        new_campaign(Request(text="q"), shot_budget=10),
        make_campaign(longitudinal_field=0.0),
        make_campaign(classical_energy=None),
        make_campaign(quantum_energy=-500.0, diagnosis=IMPOSSIBLE),
        make_campaign(quantum_energy=None),
        make_campaign(),
        make_campaign(quantum_energy=-1.10),
    ]

    for state in campaigns:
        verdict, _ = judge(state)
        assert verdict.crossover_condition.strip()
        assert verdict.summary.strip()


def test_no_campaign_ever_earns_an_unqualified_go() -> None:
    """Nothing in this project has run on hardware, so nothing has earned a "go".

    Not a limitation being hidden -- it is the honest ceiling on what a noiseless
    simulation can support, and it should fail loudly if a future change starts
    handing out "go" without a device in the loop.
    """
    campaigns = [
        make_campaign(quantum_energy=-99.0),
        make_campaign(quantum_energy=-1.30, classical_error=0.0001),
        make_campaign(longitudinal_field=0.9, quantum_energy=-50.0),
    ]

    for state in campaigns:
        verdict, _ = judge(state)
        assert verdict.call != "go"


def test_the_screens_are_ordered_most_disqualifying_first() -> None:
    """The order is load-bearing, so it is pinned rather than left to convention."""
    assert [screen.name for screen in SCREENS] == [
        # Ahead of `no_model`, which it is a special case of: both fire when nothing
        # was formalised, and only this one can say why.
        "chain_too_long",
        "no_model",
        "impossible_energy",
        "no_classical_comparison",
        "closed_form_available",
        "no_usable_quantum_run",
        # Last, because it fires exactly where a comparison was about to be made and
        # says this one cannot be. Every screen above decides on something other than
        # the two energies, so none is affected by how far the classical number can be
        # trusted -- and `no_classical_comparison` must keep outranking it, since no
        # number at all is a cleaner statement than an unreliable one.
        "baseline_not_trustworthy",
    ]


# --------------------------------------------------------------------------
# Answering the question that was asked, when three methods were named
# --------------------------------------------------------------------------


def _raced(energies: dict[str, float], steps: dict[str, int]) -> MethodRace:
    """Build a finished race without running one.

    Args:
        energies: Energy per spin each method reached.
        steps: Steps each method took.

    Returns:
        The race, on a chain whose numbers nothing here depends on.
    """
    return MethodRace(
        n_sites=8,
        depth=3,
        coupling=1.0,
        transverse_field=1.0,
        longitudinal_field=0.0,
        boundary="open",
        runs=tuple(
            MethodRun(
                method=name,  # type: ignore[arg-type]
                energy=energies[name] * 8,
                energy_per_site=energies[name],
                energy_history=(0.0, energies[name] * 8),
                n_steps=steps[name],
                n_energy_evaluations=steps[name] * 2,
                stop_reason="converged",
            )
            for name in energies
        ),
    )


def _asked_about_three_methods(race: MethodRace) -> CampaignState:
    """A finished campaign that raced three methods on an exactly solvable chain.

    Args:
        race: The race to attach.

    Returns:
        The campaign.
    """
    state = make_campaign(longitudinal_field=0.0)
    state["race"] = race
    return state


def test_a_question_naming_three_methods_is_answered_about_the_three_methods() -> None:
    # The failure this fixes. "VQE, QAOA or imaginary time?" used to come back as the
    # closed-form paragraph and nothing else -- a true statement about the chain, and
    # an answer in which none of the three words the reader used appears. Every screen
    # asks about the *problem*, so none of them could ever have mentioned a method.
    verdict, _ = judge(
        _asked_about_three_methods(
            _raced(
                {"VQE": -1.20, "QAOA": -1.24, "VarQITE": -1.22},
                {"VQE": 40, "QAOA": 25, "VarQITE": 80},
            )
        )
    )
    for method in ("VQE", "QAOA", "VarQITE"):
        assert method in verdict.summary
    assert verdict.summary.startswith("**Of the 3 methods")


def test_the_method_that_got_closest_is_named_before_the_standing_reason() -> None:
    verdict, decided_by = judge(
        _asked_about_three_methods(
            _raced(
                {"VQE": -1.20, "QAOA": -1.24, "VarQITE": -1.22},
                {"VQE": 40, "QAOA": 25, "VarQITE": 80},
            )
        )
    )
    # The screen that decided is unchanged: this adds a sentence, it does not move a
    # call. A race that could change the verdict would be a race that decides
    # feasibility, which is exactly what the screens exist to stop.
    assert decided_by == "closed_form_available"
    assert verdict.call == "no"
    assert "QAOA got closest" in verdict.summary
    assert "has an exact solution in closed form" in verdict.summary


def test_when_a_slower_method_gets_closer_both_facts_are_stated() -> None:
    # Two rankings that disagree is the interesting case and the one a reader is
    # deciding between: fewest steps is the right answer under a shot budget and lowest
    # energy is the right answer without one.
    verdict, _ = judge(
        _asked_about_three_methods(
            _raced(
                {"VQE": -1.20, "QAOA": -1.21, "VarQITE": -1.24},
                {"VQE": 40, "QAOA": 12, "VarQITE": 90},
            )
        )
    )
    assert "VarQITE got closest" in verdict.summary
    assert "QAOA settled fastest" in verdict.summary


def test_three_methods_that_reach_the_same_energy_are_reported_as_having_tied() -> None:
    # The usual outcome, because all three drive the same circuit. Resolving it into a
    # winner on the twelfth decimal place would report the optimisers' noise floor as a
    # result and hide the actual finding, which is that the circuit sets the accuracy.
    verdict, _ = judge(
        _asked_about_three_methods(
            _raced(
                {"VQE": -1.229553, "QAOA": -1.229553, "VarQITE": -1.229553},
                {"VQE": 48, "QAOA": 23, "VarQITE": 82},
            )
        )
    )
    assert "reached the same energy" in verdict.summary
    assert "got closest" not in verdict.summary
    assert "QAOA in 23" in verdict.summary


def test_a_campaign_that_raced_nothing_says_nothing_about_a_race() -> None:
    verdict, _ = judge(make_campaign(longitudinal_field=0.0))
    assert "methods raced" not in verdict.summary


def test_the_ranking_says_out_loud_that_it_was_measured_without_noise() -> None:
    # A comparison quietly dropping its conditions is how a feasibility study reaches a
    # confident wrong verdict, and "which method is best" read off a noiseless
    # simulation is precisely the sentence a reader will carry away.
    verdict, _ = judge(
        _asked_about_three_methods(
            _raced(
                {"VQE": -1.20, "QAOA": -1.24, "VarQITE": -1.22},
                {"VQE": 40, "QAOA": 25, "VarQITE": 80},
            )
        )
    )
    assert "no noise in it" in verdict.summary


@pytest.mark.parametrize(
    ("runs", "refused", "expected"),
    [
        (1, 0, "1 circuit was run here"),
        (1, 1, "1 circuit was run here and 1 was refused"),
        (3, 0, "3 circuits were run here"),
        (2, 2, "2 circuits were run here and 2 were refused"),
    ],
)
def test_the_run_count_sentence_agrees_with_its_own_number(
    runs: int, refused: int, expected: str
) -> None:
    """One run is the common case on a small chain, and it read "1 circuits were run".

    The first line under a verdict, and a sentence that cannot count to one is not
    read as a slip in the prose -- it is read as doubt about the arithmetic printed
    beside it. Found by running a starter question live and reading the report.
    """
    from src.agent.verdict import _what_this_run_found

    state = make_campaign()
    state["runs"] = tuple(
        make_run(f"hva-p{index + 1}", -1.30 - index / 100) for index in range(runs)
    )
    state["ruled_out"] = tuple(
        RuledOut(label=f"hva-p{8 + index}", reason="too deep") for index in range(refused)
    )

    assert expected in _what_this_run_found(state)
