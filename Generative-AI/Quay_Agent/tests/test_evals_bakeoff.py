"""The model bake-off: what it measures, and the order it insists on.

Nothing here calls a model. Every test builds a finished campaign by hand or a
:class:`~src.evals.bakeoff.Trial` directly, which is the point of keeping the
measurement a pure function of a state: running four campaigns is expensive and the
arithmetic is the part that has to be right.

The tests that matter most are about :class:`~src.evals.bakeoff.Agreement`. It
encodes a claim about *reading order* -- that a disagreement about what was being
asked makes every other column meaningless -- and a headline that reported the
columns anyway would be worse than no headline at all.
"""

from __future__ import annotations

from src.agent.diagnosis import Diagnosis
from src.agent.state import (
    Answer,
    BaselineResult,
    CampaignState,
    FormalModel,
    Request,
    RunRecord,
    Verdict,
    new_campaign,
)
from src.agent.usage import Call, Usage
from src.evals.bakeoff import (
    Trial,
    agreement,
    cheapest,
    cost_label,
    fastest,
    measure,
    unpriced,
)

HEALTHY = Diagnosis(
    signal="healthy",
    evidence="the energy fell steadily and flattened out",
    repair="nothing to repair",
    is_actionable=False,
)


def campaign(
    *,
    n_sites: int = 6,
    classical_per_site: float | None = -1.24,
    quantum_per_site: float | None = -1.19,
    depth: int = 2,
    verdict: str = "no",
    report: str = "",
    cited: int = 0,
    written_by: str = "model",
) -> CampaignState:
    """A finished campaign, assembled rather than run.

    Args:
        n_sites: Magnets in the chain the campaign decided it had been given.
        classical_per_site: What the baseline reached, or ``None``.
        quantum_per_site: What the best circuit reached, or ``None``.
        depth: Layers behind that circuit.
        verdict: The call reached.
        report: The written document.
        cited: Passages the answer leans on.
        written_by: Who wrote the prose.

    Returns:
        The state.
    """
    state = new_campaign(request=Request(text="a question"), shot_budget=1_000_000)
    state["model"] = FormalModel(n_sites=n_sites, coupling=1.0, transverse_field=1.0)
    if classical_per_site is not None:
        state["classical"] = BaselineResult(
            method="variational_imaginary_time",
            energy_per_site=classical_per_site,
            energy_error=0.01,
            n_measurements=100_000,
        )
    if quantum_per_site is not None:
        state["runs"] = (
            RunRecord(
                label=f"hva-p{depth}",
                family="hva",
                depth=depth,
                two_qubit_depth=depth * 4,
                energy=quantum_per_site * n_sites,
                energy_per_site=quantum_per_site,
                shots_spent=400_000,
                diagnosis=HEALTHY,
            ),
        )
    state["verdict"] = Verdict(
        call=verdict,  # type: ignore[arg-type]
        summary="a summary",
        crossover_condition="a condition",
    )
    state["report"] = report
    state["answer"] = Answer(text=report, written_by=written_by, cited=cited)
    return state


def spent(*, calls: int = 3, tokens: int = 900, cost: float = 0.004) -> Usage:
    """A metered run, assembled rather than measured.

    Args:
        calls: How many calls to fabricate.
        tokens: Prompt tokens on each.
        cost: Cost recorded on each.

    Returns:
        The usage.
    """
    return Usage(
        calls=tuple(
            Call(
                model="vendor/model",
                purpose="explanation",
                prompt_tokens=tokens,
                completion_tokens=0,
                reported=True,
                cost_usd=cost,
                latency_ms=500.0,
            )
            for _ in range(calls)
        )
    )


def trial(**overrides: object) -> Trial:
    """One comparison row, with sensible defaults.

    Args:
        **overrides: Fields to replace.

    Returns:
        The row.
    """
    base: dict[str, object] = {
        "slug": "vendor/model",
        "read_as": "TFIM L=6 J=1 h=1 g=0 (open)",
        "seconds": 4.0,
        "calls": 3,
        "tokens": 900,
        "cost_usd": 0.01,
        "verdict": "no",
        "baseline_energy_per_site": -1.24,
    }
    base.update(overrides)
    return Trial(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# What one row records
# --------------------------------------------------------------------------


def test_the_row_records_what_the_campaign_decided_it_was_asked() -> None:
    # The first column, and the one that decides whether the rest can be compared.
    row = measure(campaign(n_sites=8), "vendor/model", 3.2, spent())
    assert "L=8" in row.read_as


def test_the_row_carries_the_untouched_classical_number() -> None:
    # No model reaches this. It is the thing the comparison is testing for drift.
    row = measure(campaign(classical_per_site=-1.2731), "vendor/model", 3.2, spent())
    assert row.baseline_energy_per_site == -1.2731


def test_the_row_carries_the_depth_behind_the_quantum_number() -> None:
    # The quantum energy may legitimately differ between models because the depth
    # is proposed by a model call, so the depth has to be visible beside it.
    row = measure(campaign(depth=5), "vendor/model", 3.2, spent())
    assert row.depth == 5


def test_a_campaign_that_ran_no_circuit_has_no_quantum_number() -> None:
    row = measure(campaign(quantum_per_site=None), "vendor/model", 3.2, spent())
    assert row.best_energy_per_site is None
    assert row.depth is None


def test_the_writing_columns_are_arithmetic_and_not_a_judgement() -> None:
    # Every quality column here is a count. A judge model would score the same run
    # differently on a different day, and a number that moves on its own cannot
    # support a decision.
    document = "It may be that the classical baseline wins. Perhaps not. " * 2
    row = measure(campaign(report=document), "vendor/model", 3.2, spent())
    assert row.words == len(document.split())
    assert row.hedges > 0
    assert row.states_baseline
    assert row.baseline_at is not None


def test_a_document_that_never_names_the_baseline_says_so() -> None:
    # Burying the comparison is the failure this looks for, and never making it is
    # the worse version of that failure rather than a large position.
    row = measure(campaign(report="Quantum hardware wins outright."), "m", 1.0, spent())
    assert not row.states_baseline
    assert row.baseline_at is None


def test_the_hedge_rate_is_per_hundred_words_rather_than_a_count() -> None:
    # A model that hedges four times in two paragraphs and one that hedges four
    # times in two pages are not doing the same thing.
    short = Trial(slug="a", words=50, hedges=5)
    long = Trial(slug="b", words=500, hedges=5)
    assert short.hedge_rate == 10.0
    assert long.hedge_rate == 1.0


def test_an_empty_document_has_no_hedge_rate_rather_than_a_division_error() -> None:
    assert Trial(slug="a", words=0, hedges=0).hedge_rate == 0.0


def test_seconds_per_call_is_zero_when_no_model_was_called() -> None:
    assert Trial(slug="a", seconds=3.0, calls=0).seconds_per_call == 0.0


def test_a_failed_run_is_not_ok_and_a_finished_one_is() -> None:
    assert trial().ok
    assert not trial(failure="no such model").ok


# --------------------------------------------------------------------------
# What survived the swap
# --------------------------------------------------------------------------


def test_identical_runs_agree_on_everything() -> None:
    found = agreement([trial(slug="a"), trial(slug="b")])
    assert found.compared == 2
    assert found.read_alike
    assert found.numbers_held
    assert found.verdict_held
    assert "same chain" in found.headline()


def test_a_different_reading_is_reported_before_anything_else() -> None:
    # This is the reading-order claim. Two models handed different chains are
    # supposed to produce different numbers, and leading with the number would
    # report the wrong finding.
    found = agreement(
        [
            trial(slug="a", read_as="TFIM L=6", baseline_energy_per_site=-1.24),
            trial(slug="b", read_as="TFIM L=10", baseline_energy_per_site=-1.27),
        ]
    )
    assert not found.read_alike
    assert "did not agree on what was being asked" in found.headline()


def test_a_moved_baseline_on_one_reading_is_called_an_architecture_bug() -> None:
    found = agreement(
        [
            trial(slug="a", baseline_energy_per_site=-1.24),
            trial(slug="b", baseline_energy_per_site=-1.31),
        ]
    )
    assert found.read_alike
    assert not found.numbers_held
    assert "architecture bug" in found.headline()


def test_floating_point_noise_in_the_baseline_is_not_a_disagreement() -> None:
    # Two models can produce 1.0 and 1.0000000000000002 for the same stated
    # coupling, and reporting that as drift would cry wolf on every run.
    found = agreement(
        [
            trial(slug="a", baseline_energy_per_site=-1.24),
            trial(slug="b", baseline_energy_per_site=-1.24 + 1e-13),
        ]
    )
    assert found.numbers_held


def test_a_moved_verdict_on_held_numbers_names_the_rule() -> None:
    found = agreement([trial(slug="a", verdict="no"), trial(slug="b", verdict="go")])
    assert found.numbers_held
    assert not found.verdict_held
    assert "The rule is reading something a model wrote" in found.headline()


def test_a_model_that_could_not_run_keeps_its_row_and_is_named() -> None:
    # "This model is unreachable from here" is a result about that model.
    found = agreement([trial(slug="a"), trial(slug="broken", failure="404")])
    assert found.failures == ("broken",)
    assert found.compared == 1


def test_one_finished_run_is_not_a_comparison() -> None:
    found = agreement([trial(slug="a"), trial(slug="broken", failure="404")])
    assert "nothing to compare" in found.headline()


def test_the_distinct_values_keep_the_order_they_were_seen_in() -> None:
    # So a reader can tell which model introduced the disagreement by matching
    # against the table, which is printed in the order the caller asked for.
    found = agreement(
        [trial(slug="a", verdict="no"), trial(slug="b", verdict="go"), trial(slug="c")]
    )
    assert found.verdicts == ("no", "go")


# --------------------------------------------------------------------------
# Picking a winner
# --------------------------------------------------------------------------


def test_the_cheapest_and_fastest_ignore_runs_that_did_not_finish() -> None:
    rows = [
        trial(slug="broken", cost_usd=0.0, seconds=0.1, failure="404"),
        trial(slug="cheap", cost_usd=0.001, seconds=9.0),
        trial(slug="quick", cost_usd=0.02, seconds=1.0),
    ]
    picked_price = cheapest(rows)
    picked_speed = fastest(rows)
    assert picked_price is not None
    assert picked_speed is not None
    assert picked_price.slug == "cheap"
    assert picked_speed.slug == "quick"


def test_nothing_finished_means_no_winner_rather_than_an_exception() -> None:
    assert cheapest([trial(slug="a", failure="404")]) is None
    assert fastest([]) is None


def test_a_model_with_no_price_on_record_is_never_called_the_cheapest() -> None:
    # The single most expensive mistake this table could invite: a catalogue gap
    # reads as $0.000000 and would recommend the one row nobody has costed.
    rows = [
        trial(slug="unpriced-model", cost_usd=0.0, fully_priced=False),
        trial(slug="known", cost_usd=0.004),
    ]
    picked = cheapest(rows)
    assert picked is not None
    assert picked.slug == "known"


def test_the_unpriced_models_are_named_rather_than_counted() -> None:
    # The fix is to add that slug to the price table, and a count does not say which.
    rows = [trial(slug="a"), trial(slug="mystery", fully_priced=False)]
    assert unpriced(rows) == ("mystery",)


def test_an_unpriced_model_run_three_times_is_named_once() -> None:
    # A stability check runs one slug repeatedly, and telling the reader to add the
    # same entry to the price table three times is noise, not emphasis.
    rows = [trial(slug="mystery", fully_priced=False) for _ in range(3)]
    assert unpriced(rows) == ("mystery",)


def test_nothing_priced_means_no_cheapest_rather_than_a_free_one() -> None:
    assert cheapest([trial(slug="a", cost_usd=0.0, fully_priced=False)]) is None


def test_an_estimated_cost_is_labelled_and_still_loses_the_cheapest_line() -> None:
    # The two halves of the design in one test. The row gets a number, so the reader
    # is not staring at the word "unpriced" for fifteen of the twenty-three models
    # this project offers -- and the number is still barred from naming a winner,
    # because reasoning from a neighbouring model is not a rate card.
    estimated = trial(
        slug="z-ai/glm-5.2", cost_usd=0.0, fully_priced=False, estimated_cost_usd=0.001
    )
    published = trial(slug="openai/gpt-4o", cost_usd=0.004)
    assert cost_label(estimated) == "0.001000 est."
    picked = cheapest([estimated, published])
    assert picked is not None
    assert picked.slug == "openai/gpt-4o"


def test_a_row_with_no_estimate_at_all_still_says_unpriced() -> None:
    # A blank a reader notices beats a zero they do not.
    assert cost_label(trial(slug="mystery", fully_priced=False)) == "unpriced"


def test_a_published_row_carries_no_hedge() -> None:
    assert cost_label(trial(slug="openai/gpt-4o", cost_usd=0.0123)) == "0.012300"
