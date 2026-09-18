"""Shared fixtures: a small synthetic stand-in for train.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import LABEL_COLUMN, N_CLASSES, N_PIXELS

PIXEL_COLUMNS = [f"pixel{i}" for i in range(N_PIXELS)]


def make_frame(n_rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Build a dataframe with the same schema as the competition file.

    The pixels are noise, so no test here asserts anything about accuracy. What
    they check is the plumbing: shapes, splits, cleaning and wiring.
    """
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(n_rows, N_PIXELS), dtype=np.int64)
    labels = np.tile(np.arange(N_CLASSES), n_rows // N_CLASSES + 1)[:n_rows]
    frame = pd.DataFrame(pixels, columns=PIXEL_COLUMNS)
    frame.insert(0, LABEL_COLUMN, labels)
    return frame


@pytest.fixture
def frame() -> pd.DataFrame:
    """A clean synthetic dataset."""
    return make_frame()


@pytest.fixture
def csv_path(tmp_path: Path, frame: pd.DataFrame) -> Path:
    """The synthetic dataset written to disk as train.csv."""
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return path
