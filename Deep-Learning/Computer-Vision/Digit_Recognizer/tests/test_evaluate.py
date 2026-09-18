"""Evaluation helpers: metrics, confusions and calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import DataConfig
from src.data import build_splits
from src.evaluate import (
    calibration_table,
    confident_mistakes,
    confusion_frame,
    expected_calibration_error,
    per_class_report,
    plot_confident_mistakes,
    top_confusions,
)
from src.utils import PROJECT_ROOT
from tests.conftest import make_frame


def test_confusion_matrix_is_ten_by_ten() -> None:
    y_true = np.arange(10)
    matrix = confusion_frame(y_true, y_true)
    assert matrix.shape == (10, 10)
    assert np.array_equal(np.diag(matrix.to_numpy()), np.ones(10))


def test_top_confusions_ranks_the_most_frequent_mistake_first() -> None:
    y_true = np.array([4] * 10 + [3] * 10)
    y_pred = np.array([9] * 6 + [4] * 4 + [5] * 2 + [3] * 8)
    ranked = top_confusions(confusion_frame(y_true, y_pred))
    assert ranked[0] == (4, 9, 6)


def test_per_class_report_covers_every_digit() -> None:
    y_true = np.arange(10)
    report = per_class_report(y_true, y_true)
    assert report.loc["macro avg", "f1-score"] == pytest.approx(1.0)
    assert all(str(digit) in report.index for digit in range(10))


def test_confident_mistakes_are_ordered_by_confidence() -> None:
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 0])
    probabilities = np.array([[0.4, 0.6, 0.0], [0.01, 0.99, 0.0], [0.9, 0.1, 0.0]])
    assert confident_mistakes(y_true, y_pred, probabilities, k=2).tolist() == [1, 0]


def test_confident_mistakes_is_empty_for_a_perfect_model() -> None:
    y_true = np.array([0, 1])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert len(confident_mistakes(y_true, y_true, probabilities)) == 0


def test_calibration_of_a_perfectly_calibrated_model_has_no_gap() -> None:
    rng = np.random.default_rng(0)
    # Ten thousand predictions made at 90% confidence, 90% of which are correct.
    n = 10_000
    correct = rng.random(n) < 0.9
    probabilities = np.zeros((n, 2))
    probabilities[:, 0] = 0.9
    probabilities[:, 1] = 0.1
    y_true = np.where(correct, 0, 1)

    table = calibration_table(y_true, probabilities)
    assert expected_calibration_error(table) < 0.02


def test_calibration_detects_an_overconfident_model() -> None:
    n = 1_000
    probabilities = np.tile([0.99, 0.01], (n, 1))
    y_true = np.array([0] * (n // 2) + [1] * (n // 2))  # only half are right
    table = calibration_table(y_true, probabilities)
    assert expected_calibration_error(table) > 0.4


def test_mistake_grid_handles_a_model_with_no_mistakes(tmp_path: Path) -> None:
    """A perfect test split must not break the evaluation run."""
    splits = build_splits(make_frame(120), DataConfig())
    path = plot_confident_mistakes(
        splits,
        np.array([], dtype=int),
        np.zeros(3, dtype=int),
        np.zeros(3, dtype=int),
        np.ones((3, 10)) / 10,
        tmp_path / "mistakes.png",
    )
    assert path.exists()


def test_mistake_grid_renders_the_examples(tmp_path: Path) -> None:
    splits = build_splits(make_frame(120), DataConfig())
    y_true = np.zeros(len(splits.y_test), dtype=int)
    y_pred = np.ones(len(splits.y_test), dtype=int)
    probabilities = np.tile([0.3, 0.7], (len(y_true), 1))
    path = plot_confident_mistakes(
        splits, np.array([0, 1, 2]), y_true, y_pred, probabilities, tmp_path / "grid.png"
    )
    assert path.exists()


@pytest.mark.parametrize(
    ("fake_notebook", "expected"),
    [(False, "agg"), (True, "template")],
)
def test_importing_evaluate_only_forces_agg_outside_a_notebook(
    fake_notebook: bool, expected: str
) -> None:
    """The bug this replaces: `matplotlib.use("Agg")` ran at import time, so the
    notebook's `from src.evaluate import evaluate_run` switched the whole session
    to the file-only backend and every figure in the report rendered into a
    buffer nobody sees -- on Colab and locally alike.

    MPLBACKEND names a backend the module must leave alone; forcing Agg over it
    is exactly the failure, and it reads the same on every platform.
    """
    import os
    import subprocess
    import sys

    script = "import matplotlib, src.evaluate; print(matplotlib.get_backend())"
    if fake_notebook:
        script = f"import sys, types; sys.modules['ipykernel'] = types.ModuleType('x'); {script}"

    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "MPLBACKEND": "template"},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().lower() == expected
