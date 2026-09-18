"""The sweep: configuration generation and aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest

import src.experiments as experiments_module
from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import build_splits
from src.experiments import (
    FACTORS,
    QUICK_FACTORS,
    best_levels,
    configuration_for,
    configuration_from,
    run_sweep,
    summarise,
    summary_path_for,
    sweep_configurations,
    sweep_paths,
    use_quick_sweep,  # noqa: F401
    write_summary,
)
from tests.conftest import make_frame

SMALL_BASELINE = ExperimentConfig(
    name="test",
    data=DataConfig(),
    model=ModelConfig(hidden_sizes=(8,)),
    train=TrainConfig(max_epochs=1, device="cpu"),
)


def test_every_required_factor_is_swept() -> None:
    """These seven are the core factors; the sweep must cover every one."""
    required = {
        "optimizer",
        "batch_size",
        "activation",
        "learning_rate",
        "dropout",
        "hidden_sizes",
        "loss",
    }
    assert required <= set(FACTORS)
    assert required <= set(QUICK_FACTORS)


def test_a_model_factor_changes_only_the_model_block() -> None:
    config = configuration_for("dropout", 0.5, SMALL_BASELINE)
    assert config.model.dropout == 0.5
    assert config.model.hidden_sizes == SMALL_BASELINE.model.hidden_sizes
    assert config.train == SMALL_BASELINE.train


def test_a_training_factor_changes_only_the_training_block() -> None:
    config = configuration_for("optimizer", "sgd", SMALL_BASELINE)
    assert config.train.optimizer == "sgd"
    assert config.model == SMALL_BASELINE.model


def test_the_run_name_records_the_change() -> None:
    assert configuration_for("hidden_sizes", (64, 32), SMALL_BASELINE).name == "hidden_sizes=64-32"


def test_an_unknown_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown factor"):
        configuration_for("temperature", 1.0, SMALL_BASELINE)


def test_the_plan_covers_every_level_and_seed() -> None:
    factors: dict[str, list[Any]] = {"dropout": [0.0, 0.5], "optimizer": ["adam", "sgd"]}
    planned = list(sweep_configurations(factors, SMALL_BASELINE, [0, 1]))
    assert len(planned) == 2 * 2 * 2
    assert {seed for _, _, seed, _ in planned} == {0, 1}


def test_each_run_uses_its_own_seed() -> None:
    planned = list(sweep_configurations({"dropout": [0.1]}, SMALL_BASELINE, [3]))
    assert planned[0][3].train.seed == 3


def test_run_sweep_writes_one_row_per_run(tmp_path: Path) -> None:
    splits = build_splits(make_frame(200), DataConfig())
    destination = tmp_path / "experiments.csv"

    results = run_sweep(
        splits,
        {"dropout": [0.0, 0.3]},
        baseline=SMALL_BASELINE,
        seeds=[0],
        output_path=destination,
        verbose=False,
    )

    assert len(results) == 2
    assert destination.exists()
    assert {"factor", "level", "seed", "val_accuracy"} <= set(results.columns)


def test_summarise_averages_the_seeds() -> None:
    results = pd.DataFrame(
        {
            "factor": ["dropout"] * 4,
            "level": ["0.0", "0.0", "0.5", "0.5"],
            "seed": [0, 1, 0, 1],
            "val_accuracy": [0.90, 0.92, 0.80, 0.84],
            "epochs_run": [1, 1, 1, 1],
            "seconds": [1.0, 1.0, 1.0, 1.0],
            "n_parameters": [10, 10, 10, 10],
        }
    )
    summary = summarise(results)

    assert len(summary) == 2
    best = summary.loc[summary["best_of_factor"]].iloc[0]
    assert best["level"] == "0.0"
    assert best["val_accuracy_mean"] == pytest.approx(0.91)
    assert summary["n_runs"].tolist() == [2, 2]


def test_summarise_reports_a_spread_to_compare_against() -> None:
    results = pd.DataFrame(
        {
            "factor": ["optimizer"] * 2,
            "level": ["adam", "adam"],
            "seed": [0, 1],
            "val_accuracy": [0.90, 0.94],
            "epochs_run": [1, 1],
            "seconds": [1.0, 1.0],
            "n_parameters": [10, 10],
        }
    )
    assert summarise(results)["val_accuracy_std"].iloc[0] > 0


def test_write_summary_produces_a_markdown_table(tmp_path: Path) -> None:
    results = pd.DataFrame(
        {
            "factor": ["dropout"],
            "level": ["0.0"],
            "seed": [0],
            "val_accuracy": [0.9],
            "epochs_run": [1],
            "seconds": [1.0],
            "n_parameters": [10],
        }
    )
    path = write_summary(summarise(results), tmp_path / "summary.md")
    text = path.read_text()
    assert "## dropout" in text
    assert "0.9000" in text


def test_the_improvement_factors_are_swept_as_well() -> None:
    """The schedule, the weight decay and the augmentation were listed as
    improvements; they are measured here rather than assumed."""
    added = {"scheduler", "weight_decay", "augment"}
    assert added <= set(FACTORS)
    assert added <= set(QUICK_FACTORS)


def test_a_schedule_changes_only_the_training_block() -> None:
    config = configuration_for("scheduler", "cosine", SMALL_BASELINE)
    assert config.train.scheduler == "cosine"
    assert config.model == SMALL_BASELINE.model


def test_a_boolean_level_reads_cleanly_in_the_run_name() -> None:
    assert configuration_for("augment", True, SMALL_BASELINE).name == "augment=True"


def test_every_level_of_every_factor_builds_a_valid_configuration() -> None:
    """A typo in the FACTORS table would otherwise surface hours into a sweep."""
    for factor, levels in FACTORS.items():
        for level in levels:
            configuration_for(factor, level, SMALL_BASELINE)


def test_a_chunked_sweep_keeps_its_summary_beside_its_own_rows(tmp_path: Path) -> None:
    """A long sweep is run in chunks, each with its own --out. Without this the
    last chunk overwrites the shared summary and the report covers only the
    factors that happened to run last."""
    first = summary_path_for(tmp_path / "chunk1.csv")
    second = summary_path_for(tmp_path / "chunk2.csv")

    assert first is not None and second is not None
    assert first != second
    assert first.parent == tmp_path
    assert first.suffix == ".md"


def test_a_plain_sweep_uses_the_default_summary_location() -> None:
    assert summary_path_for(None) is None


def test_one_failed_run_does_not_end_the_sweep(tmp_path: Path) -> None:
    """The full study is 132 runs over several hours.

    Before this, a single run that raised -- an out-of-memory error, a level the
    hardware will not take -- threw away every run before it. The sweep now
    records the failure and carries on, and the CSV says which run failed and
    why rather than quietly being one row short.
    """
    splits = build_splits(make_frame(200), DataConfig())
    destination = tmp_path / "experiments.csv"
    calls = {"n": 0}
    real = experiments_module.train_from_splits

    def fail_the_second_run(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("out of memory")
        return real(*args, **kwargs)

    with mock.patch.object(experiments_module, "train_from_splits", fail_the_second_run):
        results = run_sweep(
            splits,
            {"dropout": [0.0, 0.3, 0.5]},
            baseline=SMALL_BASELINE,
            seeds=[0],
            output_path=destination,
            verbose=False,
        )

    assert len(results) == 3, "every planned run is still represented"
    failed = results[results["error"] != ""]
    assert len(failed) == 1
    assert "out of memory" in failed.iloc[0]["error"]
    assert results["val_accuracy"].notna().sum() == 2
    assert len(pd.read_csv(destination)) == 3


def test_a_failed_run_is_not_averaged_in_as_a_zero() -> None:
    """A run that never produced a number must not drag the level's mean down."""
    results = pd.DataFrame(
        {
            "factor": ["dropout"] * 3,
            "level": ["0.0"] * 3,
            "seed": [0, 1, 2],
            "val_accuracy": [0.90, 0.92, float("nan")],
            "epochs_run": [1, 1, 0],
            "seconds": [1.0, 1.0, 0.0],
            "n_parameters": [10, 10, 0],
        }
    )
    summary = summarise(results)

    assert summary["val_accuracy_mean"].iloc[0] == pytest.approx(0.91)
    assert summary["n_runs"].iloc[0] == 2, "n_runs shows the shortfall"


@pytest.mark.parametrize("argv", [["--factors"], ["--seeds"]])
def test_a_list_flag_given_no_values_is_an_error_not_the_default(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--factors` alone used to fall through to all ten, which is hours of runs."""
    monkeypatch.setattr("sys.argv", ["src.experiments", *argv])
    with pytest.raises(SystemExit, match="no values"):
        experiments_module.main()


def test_the_sweep_size_follows_the_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """A laptop must not start an overnight job because nobody passed a flag."""
    monkeypatch.setattr("src.experiments.full_study_is_affordable", lambda device: False)
    assert use_quick_sweep(False, False, "auto") is True
    monkeypatch.setattr("src.experiments.full_study_is_affordable", lambda device: True)
    assert use_quick_sweep(False, False, "auto") is False


@pytest.mark.parametrize(
    ("quick", "full", "expected"),
    [(True, False, True), (False, True, False)],
)
def test_either_flag_overrides_the_hardware(
    quick: bool, full: bool, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detection is a default, not a rule: a GPU session can still ask for
    the short sweep, and a laptop can still be told to run the whole thing."""
    monkeypatch.setattr("src.experiments.full_study_is_affordable", lambda device: not expected)
    assert use_quick_sweep(quick, full, "auto") is expected


def test_asking_for_both_sweeps_is_refused() -> None:
    """Silently picking one would run the wrong thing for hours."""
    with pytest.raises(SystemExit, match="contradict"):
        use_quick_sweep(True, True, "auto")


def test_the_short_sweep_writes_beside_the_full_one_not_over_it() -> None:
    """The bug this replaces: `make experiments --quick` wrote experiments.csv
    and experiment_summary.md, the same names the full study uses, so a
    pipeline check silently replaced hours of results with twenty short runs."""
    quick_csv, quick_summary = sweep_paths(quick=True)
    full_csv, full_summary = sweep_paths(quick=False)

    assert quick_csv != full_csv
    assert quick_summary != full_summary
    assert full_csv.name == "experiments.csv"
    assert full_summary.name == "experiment_summary.md"
    # The notebook builds the same names from its own QUICK switch.
    assert quick_csv.name == "experiments-quick.csv"
    assert quick_summary.name == "experiment_summary-quick.md"


def test_an_explicit_out_path_overrides_both_names(tmp_path: Path) -> None:
    csv_path, summary_path = sweep_paths(quick=True, output_path=tmp_path / "part1.csv")

    assert csv_path == tmp_path / "part1.csv"
    assert summary_path == tmp_path / "part1_summary.md"


def test_the_winning_level_of_each_factor_comes_back_as_a_usable_value() -> None:
    """The summary holds labels, so hidden_sizes reads back as "1024-512-256".

    Feeding that string to ModelConfig builds a network with one hidden layer
    per character. The mapping back to the tuple is what this checks.
    """
    summary = pd.DataFrame(
        {
            "factor": ["hidden_sizes", "hidden_sizes", "augment", "augment"],
            "level": ["512-256", "1024-512-256", "False", "True"],
            "val_accuracy_mean": [0.97, 0.98, 0.97, 0.99],
        }
    )
    winners = best_levels(summary)

    assert winners == {"hidden_sizes": (1024, 512, 256), "augment": True}

    config = configuration_from(winners, SMALL_BASELINE)
    assert config.model.hidden_sizes == (1024, 512, 256)
    assert config.train.augment is True
    assert config.name == SMALL_BASELINE.name


def test_the_winners_of_a_short_sweep_are_read_against_its_own_levels() -> None:
    """The short sweep has levels the full study does not, such as a 32-unit
    layer. Looking them up in FACTORS would raise on a local notebook run."""
    summary = pd.DataFrame(
        {
            "factor": ["hidden_sizes"],
            "level": ["64-32"],
            "val_accuracy_mean": [0.9],
        }
    )
    assert best_levels(summary, QUICK_FACTORS) == {"hidden_sizes": (64, 32)}
    with pytest.raises(ValueError, match="swept levels"):
        best_levels(summary)


def test_a_factor_whose_runs_all_failed_is_left_out_of_the_winners() -> None:
    """Every level of a factor can fail together -- an optimizer the hardware
    refuses, say. Picking a winner from all-NaN would return a level nothing
    scored."""
    summary = pd.DataFrame(
        {
            "factor": ["augment", "dropout"],
            "level": ["True", "0.0"],
            "val_accuracy_mean": [float("nan"), 0.97],
        }
    )
    assert best_levels(summary) == {"dropout": 0.0}
