"""Loading, cleaning and splitting the digit-recognizer data.

Cleaning runs before the split, so no repeated image can straddle it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import DataConfig
from src.utils import TRAIN_CSV, TRAIN_CSV_GZ

LABEL_COLUMN = "label"
N_PIXELS = 784
N_CLASSES = 10
PIXEL_MAX = 255.0


@dataclass
class CleaningReport:
    """What cleaning removed, and why.

    Attributes:
        n_input_rows: Rows read from the CSV.
        n_missing_removed: Rows holding at least one missing value.
        n_out_of_range_removed: Rows with a pixel outside [0, 255] or a label
            outside [0, 9].
        n_duplicate_removed: Redundant copies of an image that carried the same
            label. One representative of each group is kept.
        n_conflicting_removed: Images that appear more than once with different
            labels. Every copy is dropped, because keeping one would teach the
            model a coin flip.
        n_constant_removed: Images with a single pixel value throughout, such as
            an all-black frame. These carry no signal.
        n_output_rows: Rows surviving all of the above.
    """

    n_input_rows: int = 0
    n_missing_removed: int = 0
    n_out_of_range_removed: int = 0
    n_duplicate_removed: int = 0
    n_conflicting_removed: int = 0
    n_constant_removed: int = 0
    n_output_rows: int = 0

    @property
    def n_removed(self) -> int:
        """Total number of rows removed."""
        return self.n_input_rows - self.n_output_rows

    def summary(self) -> str:
        """Return a short multi-line summary suitable for printing or a report."""
        return "\n".join(
            [
                f"rows in                  {self.n_input_rows:>7,}",
                f"  missing values         {self.n_missing_removed:>7,}",
                f"  out of range           {self.n_out_of_range_removed:>7,}",
                f"  duplicate images       {self.n_duplicate_removed:>7,}",
                f"  label conflicts        {self.n_conflicting_removed:>7,}",
                f"  constant images        {self.n_constant_removed:>7,}",
                f"rows out                 {self.n_output_rows:>7,}  ({self.n_removed:,} removed)",
            ]
        )


def resolve_train_csv(path: Path | None = None) -> Path:
    """Return the training file to read, preferring an uncompressed one.

    The repository ships `train.csv.gz`, which pandas reads without unpacking,
    so a fresh clone can train with nothing downloaded. An uncompressed
    `train.csv` wins when it is present: it is what `make data` writes, and
    reading it is faster.

    Args:
        path: An explicit file, which is always used as given.

    Returns:
        The path `load_raw_train` will open.
    """
    if path is not None:
        return path
    return TRAIN_CSV if TRAIN_CSV.exists() else TRAIN_CSV_GZ


def load_raw_train(path: Path | None = None) -> pd.DataFrame:
    """Read the labelled competition CSV.

    Args:
        path: Location of the CSV. Defaults to data/raw/train.csv, falling
            back to the committed data/raw/train.csv.gz.

    Returns:
        The raw dataframe, unmodified.

    Raises:
        FileNotFoundError: if the file is missing, with instructions on how to
            obtain it.
        ValueError: if the file does not have the expected columns.
    """
    csv_path = resolve_train_csv(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\n"
            "Get it with `make data`, or download train.csv from "
            "https://www.kaggle.com/competitions/digit-recognizer/data "
            "and place it in data/raw/."
        )

    frame = pd.read_csv(csv_path)
    expected = [LABEL_COLUMN] + [f"pixel{i}" for i in range(N_PIXELS)]
    if list(frame.columns) != expected:
        raise ValueError(
            f"{csv_path} does not look like the digit-recognizer training file: "
            f"expected {len(expected)} columns starting with '{LABEL_COLUMN}', "
            f"found {len(frame.columns)}."
        )
    return frame


def group_identical_rows(pixels: np.ndarray) -> np.ndarray:
    """Give every distinct image a group id.

    Rows with identical pixel values share an id, which is what makes duplicate
    detection and the leakage assertions in the tests cheap.

    Args:
        pixels: Integer array of shape (n_rows, n_pixels).

    Returns:
        Array of shape (n_rows,) holding the group id of each row.
    """
    _, group_ids = np.unique(pixels, axis=0, return_inverse=True)
    return np.asarray(group_ids).reshape(-1)


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove unusable rows and report what was removed.

    Args:
        frame: Raw dataframe as read from train.csv.

    Returns:
        The cleaned dataframe with a fresh index, and the report.
    """
    report = CleaningReport(n_input_rows=len(frame))

    complete = frame.dropna()
    report.n_missing_removed = len(frame) - len(complete)

    labels = complete[LABEL_COLUMN].to_numpy()
    pixels = complete.drop(columns=LABEL_COLUMN).to_numpy()
    in_range = (
        (pixels >= 0).all(axis=1)
        & (pixels <= PIXEL_MAX).all(axis=1)
        & (labels >= 0)
        & (labels < N_CLASSES)
    )
    report.n_out_of_range_removed = int((~in_range).sum())
    kept = complete.loc[in_range].reset_index(drop=True)

    kept, duplicates_removed, conflicts_removed = _drop_duplicate_images(kept)
    report.n_duplicate_removed = duplicates_removed
    report.n_conflicting_removed = conflicts_removed

    pixel_values = kept.drop(columns=LABEL_COLUMN).to_numpy()
    is_constant = pixel_values.min(axis=1) == pixel_values.max(axis=1)
    report.n_constant_removed = int(is_constant.sum())
    kept = kept.loc[~is_constant].reset_index(drop=True)

    report.n_output_rows = len(kept)
    return kept, report


def _drop_duplicate_images(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop repeated images, separating redundant copies from label conflicts.

    Returns:
        The surviving rows, the number of redundant copies dropped, and the
        number of rows dropped because the same image carried different labels.
    """
    pixels = frame.drop(columns=LABEL_COLUMN).to_numpy()
    group_ids = group_identical_rows(pixels)
    grouped = pd.DataFrame({"group": group_ids, "label": frame[LABEL_COLUMN].to_numpy()})

    labels_per_group = grouped.groupby("group")["label"].nunique()
    conflicting = set(labels_per_group[labels_per_group > 1].index)
    is_conflicting = grouped["group"].isin(conflicting).to_numpy()
    n_conflicting = int(is_conflicting.sum())

    consistent = frame.loc[~is_conflicting].reset_index(drop=True)
    consistent_groups = group_ids[~is_conflicting]
    is_first_copy = ~pd.Series(consistent_groups).duplicated().to_numpy()
    n_duplicates = int((~is_first_copy).sum())

    return consistent.loc[is_first_copy].reset_index(drop=True), n_duplicates, n_conflicting


def subsample(frame: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Take a stratified sample of the rows.

    For local smoke runs. The class proportions are kept so a small sample still
    covers all ten digits.

    Args:
        frame: Cleaned dataframe.
        n_rows: Approximate number of rows to keep.
        seed: Sampling seed.

    Returns:
        The sampled rows, with a fresh index.
    """
    sampled, _ = train_test_split(
        frame,
        train_size=n_rows,
        random_state=seed,
        stratify=frame[LABEL_COLUMN],
    )
    return sampled.reset_index(drop=True)


@dataclass
class DataSplits:
    """Model-ready tensors for the three splits, plus the scaling used.

    Pixel tensors are float32 of shape (n_rows, 784); label tensors are int64.
    The mean and standard deviation come from the training split alone, so that
    no information about validation or test reaches the model through scaling.
    """

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    mean: float
    std: float

    @property
    def sizes(self) -> dict[str, int]:
        """Number of rows in each split."""
        return {
            "train": int(self.x_train.shape[0]),
            "val": int(self.x_val.shape[0]),
            "test": int(self.x_test.shape[0]),
        }

    def class_counts(self) -> pd.DataFrame:
        """Return the per-class row count of each split, for a balance check."""
        counts = {
            name: np.bincount(tensor.numpy(), minlength=N_CLASSES)
            for name, tensor in (
                ("train", self.y_train),
                ("val", self.y_val),
                ("test", self.y_test),
            )
        }
        return pd.DataFrame(counts, index=pd.RangeIndex(N_CLASSES, name="digit"))


def build_splits(frame: pd.DataFrame, config: DataConfig | None = None) -> DataSplits:
    """Split cleaned data into train, validation and test tensors.

    The split is stratified on the label, so each part holds the class
    proportions of the whole, and seeded, so that two runs compare models rather
    than partitions.

    Args:
        frame: Cleaned dataframe.
        config: Split fractions, seed and whether to standardise.

    Returns:
        Scaled tensors for the three splits.
    """
    cfg = config or DataConfig()
    if frame.empty:
        # sklearn's own message here talks about n_samples and train_size, which
        # does not point at the thing that actually went wrong upstream.
        raise ValueError(
            "No rows left to split. Either the CSV was empty or cleaning removed "
            "every row; run `make data-check` to see what the cleaning report says."
        )
    if cfg.max_rows is not None and cfg.max_rows < len(frame):
        if cfg.max_rows < N_CLASSES:
            raise ValueError(
                f"max_rows={cfg.max_rows} is below the {N_CLASSES} classes, so a sample "
                "cannot keep one row of each. Raise it, or leave it at None."
            )
        frame = subsample(frame, cfg.max_rows, cfg.seed)
    labels = frame[LABEL_COLUMN].to_numpy().astype(np.int64)
    pixels = frame.drop(columns=LABEL_COLUMN).to_numpy().astype(np.float32)

    x_train, x_rest, y_train, y_rest = train_test_split(
        pixels,
        labels,
        train_size=cfg.train_fraction,
        random_state=cfg.seed,
        stratify=labels,
    )
    # Of what is left, the validation share is val / (val + test).
    val_share = cfg.val_fraction / (cfg.val_fraction + cfg.test_fraction)
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest,
        y_rest,
        train_size=val_share,
        random_state=cfg.seed,
        stratify=y_rest,
    )

    x_train, x_val, x_test = (x / PIXEL_MAX for x in (x_train, x_val, x_test))
    mean, std = (float(x_train.mean()), float(x_train.std())) if cfg.standardize else (0.0, 1.0)
    if std == 0.0:
        std = 1.0

    return DataSplits(
        x_train=torch.from_numpy((x_train - mean) / std),
        y_train=torch.from_numpy(y_train),
        x_val=torch.from_numpy((x_val - mean) / std),
        y_val=torch.from_numpy(y_val),
        x_test=torch.from_numpy((x_test - mean) / std),
        y_test=torch.from_numpy(y_test),
        mean=mean,
        std=std,
    )


def make_loaders(
    splits: DataSplits,
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap the splits in DataLoaders.

    Only the training loader shuffles. Validation and test use a larger batch
    because no gradients are stored there.

    Args:
        splits: Tensors produced by build_splits.
        batch_size: Training batch size.
        num_workers: Worker processes. 0 is fastest here, since the tensors are
            already in memory and workers would only add copies.

    Returns:
        Loaders for train, validation and test, in that order.
    """
    eval_batch_size = max(batch_size, 512)
    # BatchNorm1d raises on a batch of one, so drop a final batch of that size.
    # Not when it is the only batch: an empty loader divides by zero instead of
    # reporting the real problem, which is a training split this small.
    n_train = len(splits.y_train)
    drop_last = n_train % batch_size == 1 and n_train > batch_size
    train_loader = DataLoader(
        TensorDataset(splits.x_train, splits.y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        TensorDataset(splits.x_val, splits.y_val),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(splits.x_test, splits.y_test),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def prepare_data(
    config: DataConfig | None = None,
    path: Path | None = None,
) -> tuple[DataSplits, CleaningReport]:
    """Load, clean and split in one call.

    Args:
        config: Split configuration.
        path: Location of train.csv.

    Returns:
        The splits and the cleaning report.
    """
    frame = load_raw_train(path)
    cleaned, report = clean(frame)
    return build_splits(cleaned, config), report


def main() -> None:
    """Report what is in data/raw and whether it can be used (`make data-check`)."""
    splits, report = prepare_data()
    print(report.summary())
    print()
    print("split sizes:", splits.sizes)
    print(f"train pixel mean {splits.mean:.4f}, std {splits.std:.4f}")
    print()
    print("rows per class:")
    print(splits.class_counts().to_string())


if __name__ == "__main__":
    main()
