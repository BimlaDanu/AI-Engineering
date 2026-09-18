"""The single-run CLI: argument wiring, checkpoints and the evaluation round trip."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import build_splits
from src.evaluate import evaluate_run
from src.train import (
    build_parser,
    check_run_name,
    config_from_args,
    load_run,
    save_run,
    train_from_splits,
)
from tests.conftest import make_frame

SMALL = ExperimentConfig(
    name="round_trip",
    data=DataConfig(max_rows=200),
    model=ModelConfig(hidden_sizes=(8,), activation="tanh", dropout=0.0, batch_norm=False),
    train=TrainConfig(max_epochs=1, batch_size=32, device="cpu"),
)


@pytest.fixture
def artifact_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the checkpoint and report directories into a temporary folder."""
    for module in ("src.train", "src.evaluate"):
        monkeypatch.setattr(f"{module}.CHECKPOINTS", tmp_path, raising=False)
        monkeypatch.setattr(f"{module}.REPORTS", tmp_path, raising=False)
        monkeypatch.setattr(f"{module}.FIGURES", tmp_path, raising=False)
    return tmp_path


def test_the_parser_fills_every_configuration_block() -> None:
    args = build_parser().parse_args(
        ["--hidden-sizes", "16", "8", "--optimizer", "sgd", "--lr", "0.05", "--max-rows", "500"]
    )
    config = config_from_args(args)
    assert config.model.hidden_sizes == (16, 8)
    assert config.train.optimizer == "sgd"
    assert config.train.learning_rate == 0.05
    assert config.data.max_rows == 500


def test_no_batch_norm_flag_switches_batch_norm_off() -> None:
    config = config_from_args(build_parser().parse_args(["--no-batch-norm"]))
    assert config.model.batch_norm is False


def test_a_saved_run_loads_back_with_the_same_architecture(artifact_dirs: Path) -> None:
    splits = build_splits(make_frame(300), SMALL.data)
    result = train_from_splits(splits, SMALL, verbose=False)

    save_run(result, SMALL)
    model, checkpoint = load_run(SMALL.name)

    assert model.config == SMALL.model
    assert checkpoint["best_epoch"] == result.best_epoch
    for saved, loaded in zip(
        result.model.state_dict().values(), model.state_dict().values(), strict=True
    ):
        assert saved.cpu().equal(loaded)


def test_data_config_survives_the_checkpoint() -> None:
    """Evaluation rebuilds the split from the row, so every field must survive."""
    config = ExperimentConfig(
        data=DataConfig(train_fraction=0.5, val_fraction=0.25, seed=7, standardize=False)
    )
    assert DataConfig.from_row(config.to_row()) == config.data


def test_evaluate_scores_the_split_the_model_was_trained_away_from(
    artifact_dirs: Path, csv_path: Path
) -> None:
    config = ExperimentConfig(
        name="evaluated",
        data=DataConfig(train_fraction=0.5, val_fraction=0.25, seed=7, max_rows=200),
        model=SMALL.model,
        train=SMALL.train,
    )
    splits = build_splits(make_frame(400), config.data)
    save_run(train_from_splits(splits, config, verbose=False), config)

    payload = evaluate_run(config.name, csv_path=csv_path, device_preference="cpu")

    assert payload["n_test"] == splits.sizes["test"]
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert payload["confusion"].to_numpy().sum() == payload["n_test"]
    assert (artifact_dirs / "test_report.md").exists()


def test_the_schedule_and_augmentation_flags_reach_the_configuration() -> None:
    args = build_parser().parse_args(
        ["--scheduler", "cosine", "--augment", "--max-shift", "3", "--max-rotation", "15"]
    )
    config = config_from_args(args)
    assert config.train.scheduler == "cosine"
    assert config.train.augment is True
    assert config.train.max_shift == 3
    assert config.train.max_rotation == 15.0


def test_augmentation_is_off_unless_it_is_asked_for() -> None:
    """A default run must stay the plain baseline the sweep compares against."""
    config = config_from_args(build_parser().parse_args([]))
    assert config.train.augment is False
    assert config.train.scheduler == "none"


def test_every_training_field_the_sweep_varies_has_a_flag() -> None:
    """A factor that cannot be set from the command line cannot be reproduced
    from one either."""
    from src.experiments import FACTORS

    parser_flags = {action.dest for action in build_parser()._actions}
    model_fields = {"hidden_sizes", "activation", "dropout"}
    for factor in FACTORS:
        assert factor in parser_flags or factor in model_fields, factor


def test_the_training_seed_does_not_move_the_split() -> None:
    """The sweep repeats a configuration over training seeds while holding the
    partition fixed. If --seed moved the split too, a sweep row could not be
    reproduced from a command line, which is the claim this CLI makes."""
    default = config_from_args(build_parser().parse_args([]))
    reseeded = config_from_args(build_parser().parse_args(["--seed", "7"]))

    assert reseeded.train.seed == 7
    assert reseeded.data == default.data


def test_the_split_can_still_be_moved_deliberately() -> None:
    config = config_from_args(build_parser().parse_args(["--split-seed", "7"]))
    assert config.data.seed == 7
    assert config.train.seed == TrainConfig().seed


def test_a_sweep_row_can_be_reproduced_from_a_command_line() -> None:
    """Take a configuration the sweep generated and rebuild it from flags."""
    from src.experiments import BASELINE, configuration_for

    swept = configuration_for("dropout", 0.5, BASELINE)
    swept = replace(swept, train=replace(swept.train, seed=2))

    rebuilt = config_from_args(
        build_parser().parse_args(
            ["--hidden-sizes", "512", "256", "--dropout", "0.5", "--seed", "2"]
        )
    )
    assert rebuilt.model == swept.model
    assert rebuilt.data == swept.data
    assert rebuilt.train.seed == swept.train.seed
    assert rebuilt.train.optimizer == swept.train.optimizer
    assert rebuilt.train.batch_size == swept.train.batch_size


@pytest.mark.parametrize("name", ["../../escaped", "a/b", "", "..", "."])
def test_a_run_name_that_is_not_a_file_name_is_refused(name: str) -> None:
    """Artifacts are named after the run, so a separator would write elsewhere."""
    with pytest.raises(ValueError, match="plain file name"):
        check_run_name(name)


def test_saving_a_run_whose_name_escapes_writes_nothing(artifact_dirs: Path) -> None:
    from src.train import save_run

    result = train_from_splits(build_splits(make_frame(200), SMALL.data), SMALL, verbose=False)
    with pytest.raises(ValueError, match="plain file name"):
        save_run(result, replace(SMALL, name="../escaped"))
    assert not list(artifact_dirs.parent.glob("escaped*"))
