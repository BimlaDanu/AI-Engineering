"""Cleaning and splitting: the parts where a silent mistake inflates the score."""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import (
    LABEL_COLUMN,
    N_CLASSES,
    DataSplits,
    build_splits,
    clean,
    group_identical_rows,
    load_raw_train,
    make_loaders,
    prepare_data,
    resolve_train_csv,
    subsample,
)
from src.train import train_from_splits
from src.utils import TEST_CSV_GZ, TRAIN_CSV, TRAIN_CSV_GZ
from tests.conftest import PIXEL_COLUMNS, make_frame


def replace_train_split(splits: DataSplits, n_rows: int) -> DataSplits:
    """Return the same splits with the training split cut to `n_rows` rows."""
    return dataclasses.replace(
        splits, x_train=splits.x_train[:n_rows], y_train=splits.y_train[:n_rows]
    )


def test_load_raw_train_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        load_raw_train(tmp_path / "absent.csv")


def test_load_raw_train_rejects_a_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="digit-recognizer"):
        load_raw_train(path)


def test_clean_keeps_clean_data_untouched(frame: pd.DataFrame) -> None:
    cleaned, report = clean(frame)
    assert len(cleaned) == len(frame)
    assert report.n_removed == 0


def test_clean_drops_repeated_images_but_keeps_one(frame: pd.DataFrame) -> None:
    duplicated = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
    cleaned, report = clean(duplicated)
    assert report.n_duplicate_removed == 2
    assert len(cleaned) == len(frame)


def test_clean_drops_every_copy_when_labels_disagree(frame: pd.DataFrame) -> None:
    contradiction = frame.iloc[[0]].copy()
    contradiction[LABEL_COLUMN] = (frame.iloc[0][LABEL_COLUMN] + 1) % N_CLASSES
    with_conflict = pd.concat([frame, contradiction], ignore_index=True)

    cleaned, report = clean(with_conflict)

    assert report.n_conflicting_removed == 2
    assert len(cleaned) == len(frame) - 1


def test_clean_drops_constant_and_out_of_range_rows(frame: pd.DataFrame) -> None:
    blank = frame.iloc[[0]].copy()
    blank[PIXEL_COLUMNS] = 0
    corrupt = frame.iloc[[1]].copy()
    corrupt["pixel0"] = 999

    cleaned, report = clean(pd.concat([frame, blank, corrupt], ignore_index=True))

    assert report.n_constant_removed == 1
    assert report.n_out_of_range_removed == 1
    assert len(cleaned) == len(frame)


def test_group_identical_rows_matches_identical_pixels() -> None:
    pixels = np.array([[1, 2], [1, 2], [3, 4]])
    groups = group_identical_rows(pixels)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_splits_use_the_requested_proportions(frame: pd.DataFrame) -> None:
    splits = build_splits(frame, DataConfig(train_fraction=0.6, val_fraction=0.2))
    sizes = splits.sizes
    total = sum(sizes.values())

    assert total == len(frame)
    assert sizes["train"] / total == pytest.approx(0.6, abs=0.02)
    assert sizes["val"] / total == pytest.approx(0.2, abs=0.02)
    assert sizes["test"] / total == pytest.approx(0.2, abs=0.02)


def test_splits_are_stratified(frame: pd.DataFrame) -> None:
    counts = build_splits(frame).class_counts()
    shares = counts / counts.sum()
    # Every class holds roughly the same share in each split.
    assert (shares.max(axis=1) - shares.min(axis=1)).max() < 0.02


def test_no_image_appears_in_two_splits(frame: pd.DataFrame) -> None:
    splits = build_splits(frame, DataConfig(standardize=False))
    as_bytes = {
        name: {row.tobytes() for row in tensor.numpy()}
        for name, tensor in (
            ("train", splits.x_train),
            ("val", splits.x_val),
            ("test", splits.x_test),
        )
    }
    assert not as_bytes["train"] & as_bytes["val"]
    assert not as_bytes["train"] & as_bytes["test"]
    assert not as_bytes["val"] & as_bytes["test"]


def test_standardisation_uses_training_statistics_only(frame: pd.DataFrame) -> None:
    splits = build_splits(frame)
    assert float(splits.x_train.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(splits.x_train.std()) == pytest.approx(1.0, abs=1e-3)
    # Validation is scaled with the training statistics, so it need not be centred.
    assert float(splits.x_val.mean()) != pytest.approx(0.0, abs=1e-9)


def test_subsample_keeps_the_class_balance() -> None:
    frame = make_frame(n_rows=500)
    sampled = subsample(frame, 100, seed=0)
    assert 80 <= len(sampled) <= 120
    assert sampled[LABEL_COLUMN].nunique() == N_CLASSES


def test_max_rows_caps_the_dataset(frame: pd.DataFrame) -> None:
    splits = build_splits(frame, DataConfig(max_rows=100))
    assert sum(splits.sizes.values()) <= 120


def test_loaders_produce_batches_of_the_right_shape(frame: pd.DataFrame) -> None:
    splits = build_splits(frame)
    train_loader, val_loader, _ = make_loaders(splits, batch_size=16)
    inputs, targets = next(iter(train_loader))

    assert inputs.shape == (16, 784)
    assert targets.shape == (16,)
    assert inputs.dtype.is_floating_point
    # Evaluation stores no gradients, so it uses a larger batch.
    assert (val_loader.batch_size or 0) >= 512


def test_prepare_data_reads_cleans_and_splits(csv_path: Path) -> None:
    splits, report = prepare_data(DataConfig(), csv_path)
    assert report.n_input_rows == sum(splits.sizes.values())


def test_a_final_batch_of_one_row_is_dropped() -> None:
    """BatchNorm1d raises on a batch of one, so a split whose last batch holds a
    single row would kill a sweep part way through. The loader drops that batch;
    nothing else is lost."""
    splits = build_splits(make_frame(215), DataConfig())
    assert splits.sizes["train"] % 32 == 1, "fixture no longer produces the awkward remainder"

    train_loader, _, _ = make_loaders(splits, batch_size=32)
    sizes = [targets.shape[0] for _, targets in train_loader]

    assert 1 not in sizes
    assert sum(sizes) == splits.sizes["train"] - 1


def test_every_other_split_keeps_all_of_its_training_rows() -> None:
    splits = build_splits(make_frame(400), DataConfig())
    train_loader, _, _ = make_loaders(splits, batch_size=32)
    assert sum(targets.shape[0] for _, targets in train_loader) == splits.sizes["train"]


def test_a_batch_norm_model_trains_on_the_awkward_split() -> None:
    """The regression test for the guard above: this configuration used to end
    with "Expected more than 1 value per channel when training"."""
    splits = build_splits(make_frame(215), DataConfig())
    config = ExperimentConfig(
        model=ModelConfig(hidden_sizes=(8,), batch_norm=True),
        train=TrainConfig(max_epochs=1, batch_size=32, device="cpu"),
    )
    assert train_from_splits(splits, config, verbose=False).history


def test_a_split_of_one_batch_is_not_dropped_into_nothing() -> None:
    """`drop_last` exists to avoid a BatchNorm batch of one, not to empty the loader.

    When the whole training split is a single short batch, dropping it leaves a
    loader with no batches at all, and the epoch then divides by a zero row
    count instead of reporting what is actually wrong.
    """
    splits = build_splits(make_frame(200), DataConfig())
    one_row = replace_train_split(splits, n_rows=1)

    train_loader, _, _ = make_loaders(one_row, batch_size=128)
    assert len(train_loader) == 1, "the only batch must survive"


def test_the_last_batch_of_one_row_is_still_dropped() -> None:
    """The original reason for `drop_last` has to keep working."""
    splits = build_splits(make_frame(400), DataConfig())
    # 33 training rows with batch_size 32 leaves a final batch of exactly one.
    awkward = replace_train_split(splits, n_rows=33)

    train_loader, _, _ = make_loaders(awkward, batch_size=32)
    assert len(train_loader) == 1
    assert all(len(targets) == 32 for _, targets in train_loader)


def test_a_config_read_back_through_a_csv_keeps_its_row_cap_off() -> None:
    """`max_rows=None` survives a CSV round trip as None, not as NaN.

    A NaN compares false against everything, so it happens to behave like None
    here -- but by accident, and a reader cannot tell which was meant.
    """
    row = ExperimentConfig().to_row()
    round_tripped = pd.read_csv(io.StringIO(pd.DataFrame([row]).to_csv(index=False)))

    assert DataConfig.from_row(round_tripped.iloc[0].to_dict()).max_rows is None


def test_splitting_nothing_points_at_the_cleaning_step() -> None:
    """Cleaning can empty a frame; sklearn's own error does not say that."""
    empty = make_frame(10).iloc[:0]
    with pytest.raises(ValueError, match="No rows left to split"):
        build_splits(empty, DataConfig())


def test_a_row_cap_below_the_class_count_says_so(frame: pd.DataFrame) -> None:
    """--max-rows 5 cannot keep one row of each of the ten digits."""
    with pytest.raises(ValueError, match="below the 10 classes"):
        build_splits(frame, DataConfig(max_rows=5))


@pytest.mark.parametrize("path", [TRAIN_CSV_GZ, TEST_CSV_GZ])
def test_the_compressed_competition_files_are_committed(path: Path) -> None:
    """A clone must be able to train without downloading anything.

    The uncompressed CSVs are gitignored at 73 MB and 49 MB; the gzips are
    8.9 MB and 5.6 MB and ship with the repository, so a Colab or Kaggle
    session that clones the project has the data already. pandas reads them
    compressed -- nothing is unpacked.
    """
    assert path.exists(), f"{path} is missing; regenerate it with `make data-gz`"


def test_the_plain_csv_wins_when_both_are_present(tmp_path: Path) -> None:
    """Reading 73 MB of text beats decompressing it, so prefer it when it is there."""
    assert resolve_train_csv() == (TRAIN_CSV if TRAIN_CSV.exists() else TRAIN_CSV_GZ)
    explicit = tmp_path / "somewhere.csv"
    assert resolve_train_csv(explicit) == explicit, "an explicit path must be used as given"


def test_a_gzipped_csv_loads_the_same_frame_as_a_plain_one(csv_path: Path) -> None:
    """The fallback is only safe if the two routes produce identical data."""
    import gzip

    packed = csv_path.parent / "train.csv.gz"
    packed.write_bytes(gzip.compress(csv_path.read_bytes()))
    assert load_raw_train(packed).equals(load_raw_train(csv_path))
