"""The belief state: do the budgets actually refuse, and does nothing leak?

The budgets are the part worth testing hardest. A shot ledger that quietly allows
an overdraft, or a coherence budget that waves through a circuit longer than the
hardware can hold, would turn every affordability claim in a report into fiction
while leaving every test that checks energies perfectly green.
"""

from __future__ import annotations

import pytest

from src.agent.diagnosis import Diagnosis
from src.agent.state import (
    NOMINAL_COHERENCE_NS,
    NOMINAL_TWO_QUBIT_GATE_NS,
    USABLE_COHERENCE_FRACTION,
    BaselineResult,
    Belief,
    Citation,
    CoherenceBudget,
    FormalModel,
    Request,
    RuledOut,
    RunRecord,
    ShotLedger,
    Verdict,
    best_run,
    extend,
    new_campaign,
    snapshot,
)
from src.physics.lattice import Geometry

HEALTHY = Diagnosis(signal="healthy", evidence="converged", repair="nothing", is_actionable=True)
IMPOSSIBLE = Diagnosis(
    signal="below_variational_bound",
    evidence="under the floor",
    repair="find the bug",
    is_actionable=False,
)


def make_run(label: str, energy: float, diagnosis: Diagnosis = HEALTHY) -> RunRecord:
    return RunRecord(
        label=label,
        family="hva",
        depth=2,
        two_qubit_depth=8,
        energy=energy,
        energy_per_site=energy / 6,
        shots_spent=1000,
        diagnosis=diagnosis,
    )


# ── the shot ledger ────────────────────────────────────────────────────────────


def test_a_fresh_ledger_has_everything_left() -> None:
    ledger = ShotLedger(budget=1000)

    assert ledger.remaining == 1000
    assert ledger.refusal(1000) is None


def test_spending_within_budget_leaves_the_original_untouched() -> None:
    """A rejected branch of the graph must not be able to reduce the budget."""
    ledger = ShotLedger(budget=1000)

    after = ledger.debit(400)

    assert after.spent == 400
    assert ledger.spent == 0


def test_an_unaffordable_run_is_refused_with_the_shortfall_named() -> None:
    ledger = ShotLedger(budget=1000, spent=900)

    reason = ledger.refusal(500)

    assert reason is not None
    assert "500" in reason
    assert "100" in reason


def test_overdrawing_raises_rather_than_silently_clamping() -> None:
    """Reaching this exception means a run started without being priced."""
    ledger = ShotLedger(budget=1000, spent=900)

    with pytest.raises(ValueError, match="shot budget exceeded"):
        ledger.debit(500)


def test_spending_exactly_the_remainder_is_allowed() -> None:
    ledger = ShotLedger(budget=1000, spent=900)

    assert ledger.debit(100).remaining == 0


def test_a_negative_cost_is_refused() -> None:
    """Otherwise a negative debit would top the budget back up."""
    assert ShotLedger(budget=1000).refusal(-50) is not None


def test_an_impossible_ledger_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="spent"):
        ShotLedger(budget=100, spent=200)


# ── the coherence budget ───────────────────────────────────────────────────────


def test_the_depth_limit_matches_the_arithmetic_it_claims() -> None:
    """Checked against the multiplication written out by hand, not against itself."""
    budget = CoherenceBudget()

    expected = int(USABLE_COHERENCE_FRACTION * NOMINAL_COHERENCE_NS / NOMINAL_TWO_QUBIT_GATE_NS)

    assert budget.max_two_qubit_depth == expected
    assert budget.max_two_qubit_depth == 33


def test_a_circuit_at_the_limit_fits_and_one_layer_deeper_does_not() -> None:
    budget = CoherenceBudget()
    limit = budget.max_two_qubit_depth

    assert budget.refusal(limit) is None
    assert budget.refusal(limit + 1) is not None


def test_the_refusal_says_how_long_the_circuit_would_run() -> None:
    """The planner needs the size of the overshoot, not just a rejection."""
    budget = CoherenceBudget()

    reason = budget.refusal(1000)

    assert reason is not None
    assert "microseconds" in reason
    assert str(budget.max_two_qubit_depth) in reason


def test_a_faster_gate_buys_a_deeper_circuit() -> None:
    """The trade the hardware layer will eventually make real.

    Five times the gate speed buys close to five times the depth, but not exactly:
    the limit is a whole number of layers, so the division is floored and the two
    remainders do not match. Asserting the ratio rather than the product is what
    keeps this a statement about the physics instead of about the rounding.
    """
    slow = CoherenceBudget(two_qubit_gate_ns=300.0)
    fast = CoherenceBudget(two_qubit_gate_ns=60.0)

    assert fast.max_two_qubit_depth == pytest.approx(5 * slow.max_two_qubit_depth, rel=0.01)
    assert fast.max_two_qubit_depth > slow.max_two_qubit_depth


def test_nonsense_device_numbers_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        CoherenceBudget(coherence_ns=0.0)
    with pytest.raises(ValueError, match="fraction"):
        CoherenceBudget(usable_fraction=1.5)


# ── the model ──────────────────────────────────────────────────────────────────


def test_a_chain_with_no_longitudinal_field_is_exactly_solvable() -> None:
    """The fact the whole verdict turns on."""
    assert FormalModel(n_sites=8).is_exactly_solvable


def test_a_longitudinal_field_removes_the_closed_form() -> None:
    assert not FormalModel(n_sites=8, longitudinal_field=0.3).is_exactly_solvable


def test_the_label_names_every_number_that_changes_the_answer() -> None:
    label = FormalModel(
        n_sites=8, coupling=2.0, transverse_field=0.5, longitudinal_field=0.1
    ).label()

    assert "L=8" in label
    assert "J=2" in label
    assert "h=0.5" in label
    assert "g=0.1" in label


# ── beliefs and accumulation ───────────────────────────────────────────────────


def test_a_belief_without_support_is_refused() -> None:
    """An unsupported claim is an assertion, and the campaign must not collect them."""
    with pytest.raises(ValueError, match="support"):
        Belief(claim="quantum wins", support=())


def test_the_reducer_appends_rather_than_replacing() -> None:
    """Two branches writing the same field must not erase each other."""
    assert extend(("a", "b"), ("c",)) == ("a", "b", "c")
    assert extend((), ("c",)) == ("c",)


# ── choosing the best run ──────────────────────────────────────────────────────


def test_the_best_run_is_the_lowest_energy() -> None:
    state = new_campaign(Request(text="q"), shot_budget=10)
    state["runs"] = (make_run("a", -5.0), make_run("b", -9.0), make_run("c", -7.0))

    best = best_run(state)

    assert best is not None
    assert best.label == "b"


def test_an_impossible_energy_can_never_win() -> None:
    """The failure this guard exists for: a bug produces the lowest number of all.

    Without the exclusion, the more badly broken a run is, the more likely it is to
    be selected and quoted as the campaign's headline result.
    """
    state = new_campaign(Request(text="q"), shot_budget=10)
    state["runs"] = (make_run("good", -5.0), make_run("broken", -500.0, IMPOSSIBLE))

    best = best_run(state)

    assert best is not None
    assert best.label == "good"


def test_a_campaign_of_only_broken_runs_has_no_best() -> None:
    state = new_campaign(Request(text="q"), shot_budget=10)
    state["runs"] = (make_run("broken", -500.0, IMPOSSIBLE),)

    assert best_run(state) is None


# ── the snapshot ───────────────────────────────────────────────────────────────


def test_a_fresh_campaign_distinguishes_unknown_from_empty() -> None:
    """Not-established and established-to-be-nothing must not collapse together."""
    state = new_campaign(Request(text="q"), shot_budget=100)

    assert state["model"] is None
    assert state["classical"] is None
    assert state["verdict"] is None
    assert state["runs"] == ()


def test_the_snapshot_holds_only_primitives() -> None:
    """It has to survive being written to a JSON log and read back elsewhere."""
    state = new_campaign(Request(text="how many?", framing="vendor"), shot_budget=1000)
    state["model"] = FormalModel(n_sites=6, longitudinal_field=0.2)
    state["runs"] = (make_run("hva-p2", -6.0),)
    state["ruled_out"] = (RuledOut(label="hva-p8", reason="too deep"),)
    state["classical"] = BaselineResult(
        method="variational_imaginary_time",
        energy_per_site=-1.0,
        energy_error=0.01,
        n_measurements=400,
    )
    state["beliefs"] = (Belief(claim="it converged", support=("run hva-p2",)),)
    state["citations"] = (Citation(identifier="1234.5678", title="A paper", snippet="text"),)
    state["verdict"] = Verdict(
        call="conditional", summary="maybe", crossover_condition="better hardware"
    )
    state["shots"] = ShotLedger(budget=1000, spent=250)

    facts = snapshot(state)

    assert facts["request"] == "how many?"
    assert facts["framing"] == "vendor"
    assert facts["exactly_solvable"] is False
    assert facts["shots_spent"] == 250
    assert facts["verdict"] == "conditional"
    assert facts["runs"][0]["label"] == "hva-p2"
    assert facts["citations"] == ["1234.5678"]

    import json

    assert json.loads(json.dumps(facts))["verdict"] == "conditional"


def test_the_snapshot_of_an_empty_campaign_still_renders() -> None:
    """The run view has to show a campaign that stopped early, not crash on it."""
    facts = snapshot(new_campaign(Request(text="q"), shot_budget=10))

    assert facts["model"] is None
    assert facts["verdict"] is None
    assert facts["runs"] == []


# --------------------------------------------------------------------------
# The closed-form screen needs both conditions, and geometry is the forgotten one
# --------------------------------------------------------------------------


def test_a_chain_with_no_tilt_is_exactly_solvable() -> None:
    assert FormalModel(n_sites=16).is_exactly_solvable


def test_a_tilted_chain_is_not() -> None:
    assert not FormalModel(n_sites=16, longitudinal_field=0.4).is_exactly_solvable


@pytest.mark.parametrize("geometry", ["square", "triangular"])
def test_a_two_dimensional_lattice_is_never_exactly_solvable(geometry: Geometry) -> None:
    # The landmine this defuses, and it points the opposite way to the tilt. The
    # Jordan-Wigner mapping needs each site to have one forward neighbour with an
    # empty string between them, which is true on a chain and false on a lattice --
    # so a 4x4 square at g = 0 has no closed form at all. Reading only the tilt here
    # made the verdict layer's *first* screen answer "no" with high confidence and
    # fluent prose about independent particles, for the flagship two-dimensional
    # size. A confident wrong answer, ahead of any comparison of numbers.
    model = FormalModel(n_sites=16, geometry=geometry, rows=4)
    assert not model.is_exactly_solvable


def test_a_lattice_says_its_shape_in_the_label() -> None:
    # The label reaches figure legends and the composed answer, so a report about a
    # square lattice must not be readable as one about a chain of sixteen.
    assert FormalModel(n_sites=16, geometry="square", rows=4).label().startswith("TFIM 4x4")
    assert FormalModel(n_sites=16).label().startswith("TFIM L=16")


def test_the_lattice_is_derived_from_the_site_count_and_cannot_disagree_with_it() -> None:
    model = FormalModel(n_sites=12, geometry="square", rows=3)
    assert model.lattice.cols == 4
    assert model.lattice.n_sites == model.n_sites
    with pytest.raises(ValueError, match="rectangle"):
        _ = FormalModel(n_sites=10, geometry="square", rows=4).lattice
