"""Final evaluation on the held-out test split.

Nothing is selected on the test split: every choice -- hyperparameters, epoch,
checkpoint -- is made on validation, and the test rows are read only to score
what validation already picked. `src/ensemble.py` and `src/reference_cnn.py`
also score on it, for the same reason and under the same rule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

# A command-line run writes figures to files and must not need a display. A
# notebook already has a working backend, and switching it here would render
# every figure in the report into a buffer nobody ever sees.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src.config import DataConfig, TrainConfig
from src.data import N_CLASSES, DataSplits, make_loaders, prepare_data
from src.models import count_parameters
from src.train import load_run
from src.training import predict
from src.utils import FIGURES, REPORTS, ensure_dirs, select_device


def test_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over the test split.

    Returns:
        Predicted classes, true classes and the full probability matrix.
    """
    logits, targets = predict(model, test_loader, device)
    probabilities = F.softmax(logits, dim=-1).numpy()
    return probabilities.argmax(axis=1), targets.numpy(), probabilities


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return precision, recall and F1 for every digit, plus the averages."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose().round(4)


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return the confusion matrix with labelled rows and columns."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    index = pd.Index(range(N_CLASSES), name="true")
    return pd.DataFrame(matrix, index=index, columns=pd.Index(range(N_CLASSES), name="predicted"))


def top_confusions(matrix: pd.DataFrame, k: int = 5) -> list[tuple[int, int, int]]:
    """Return the k most frequent (true, predicted, count) mistakes."""
    off_diagonal = [
        (int(true), int(pred), int(matrix.iat[true, pred]))
        for true in range(N_CLASSES)
        for pred in range(N_CLASSES)
        if true != pred and matrix.iat[true, pred] > 0
    ]
    return sorted(off_diagonal, key=lambda item: item[2], reverse=True)[:k]


def confident_mistakes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    k: int = 12,
) -> np.ndarray:
    """Return the indices of the wrong predictions the model was surest about.

    Confident mistakes are the ones worth looking at: a model that is wrong and
    certain fails differently from one that is wrong and hesitant.
    """
    wrong = np.flatnonzero(y_true != y_pred)
    confidence = probabilities[wrong].max(axis=1)
    return wrong[np.argsort(-confidence)[:k]]


def calibration_table(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Compare predicted confidence with observed accuracy, bin by bin.

    A calibrated model matches: among predictions made with 90% confidence,
    about 90% are correct.
    """
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        in_bin = bin_index == b
        if not in_bin.any():
            continue
        rows.append(
            {
                "bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})",
                "n": int(in_bin.sum()),
                "mean_confidence": float(confidence[in_bin].mean()),
                "accuracy": float(correct[in_bin].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = (table["mean_confidence"] - table["accuracy"]).round(4)
    return table.round(4)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Weighted average of the per-bin confidence-accuracy gap."""
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


def plot_confusion_matrix(matrix: pd.DataFrame, path: Path) -> Path:
    """Save the confusion matrix as a heatmap, off-diagonal entries annotated."""
    figure, axes = plt.subplots(figsize=(6.5, 5.5))
    image = axes.imshow(matrix.to_numpy(), cmap="Blues")
    tick_labels = [str(digit) for digit in range(N_CLASSES)]
    axes.set_xticks(range(N_CLASSES), labels=tick_labels)
    axes.set_yticks(range(N_CLASSES), labels=tick_labels)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title("Confusion matrix, test split")

    for true in range(N_CLASSES):
        for pred in range(N_CLASSES):
            count = int(matrix.iat[true, pred])
            if true != pred and count > 0:
                axes.text(pred, true, str(count), ha="center", va="center", fontsize=7, color="0.2")

    figure.colorbar(image, ax=axes, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_confident_mistakes(
    splits: DataSplits,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
) -> Path:
    """Save a grid of the misclassified digits the model was surest about.

    A model with no mistakes on the test split still produces a figure, saying
    so, rather than failing the evaluation run.
    """
    n_shown = len(indices)
    if n_shown == 0:
        figure, axis = plt.subplots(figsize=(4, 2))
        axis.text(0.5, 0.5, "no misclassified test examples", ha="center", va="center")
        axis.axis("off")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path

    columns = min(6, n_shown)
    rows = int(np.ceil(n_shown / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(1.6 * columns, 1.9 * rows))

    for axis, index in zip(np.atleast_1d(axes).ravel(), indices, strict=False):
        # Undo the standardisation so the digit is readable.
        image = splits.x_test[index].numpy() * splits.std + splits.mean
        axis.imshow(image.reshape(28, 28), cmap="gray")
        axis.set_title(
            f"true {y_true[index]}, said {y_pred[index]}\np={probabilities[index].max():.2f}",
            fontsize=8,
        )
    for axis in np.atleast_1d(axes).ravel():
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_training_curves(history: pd.DataFrame, path: Path) -> Path:
    """Save the loss and accuracy curves of a run side by side."""
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 4))

    loss_axis.plot(history["epoch"], history["train_loss"], label="train")
    loss_axis.plot(history["epoch"], history["val_loss"], label="validation")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.set_title("Loss")
    loss_axis.legend()

    accuracy_axis.plot(history["epoch"], history["train_accuracy"], label="train")
    accuracy_axis.plot(history["epoch"], history["val_accuracy"], label="validation")
    accuracy_axis.set_xlabel("epoch")
    accuracy_axis.set_ylabel("accuracy")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def write_report(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Write the test-set results as markdown.

    Returns:
        Path of the written report.
    """
    ensure_dirs()
    destination = path or REPORTS / "test_report.md"
    confusions = "\n".join(
        f"- {true} read as {pred}: {count} times" for true, pred, count in payload["top_confusions"]
    )
    lines = [
        "# Test-set evaluation",
        "",
        f"Model: `{payload['name']}`  ",
        f"Test rows: {payload['n_test']:,}  ",
        f"Parameters: {payload['n_parameters']:,}",
        "",
        f"**Accuracy: {payload['accuracy']:.4f}**  ",
        f"Macro F1: {payload['macro_f1']:.4f}  ",
        f"Expected calibration error: {payload['ece']:.4f}",
        "",
        "## Per class",
        "",
        payload["per_class"].to_markdown(),
        "",
        "## Confusion matrix",
        "",
        payload["confusion"].to_markdown(),
        "",
        "### Most frequent mistakes",
        "",
        confusions,
        "",
        "## Calibration",
        "",
        payload["calibration"].to_markdown(index=False),
        "",
    ]
    destination.write_text("\n".join(lines))
    return destination


def evaluate_run(
    name: str = "baseline",
    csv_path: Path | None = None,
    device_preference: str = "auto",
) -> dict[str, Any]:
    """Score a saved model on the test split and write the report and figures.

    Args:
        name: Run name of the checkpoint to load.
        csv_path: Location of train.csv.
        device_preference: "auto", "cuda", "mps" or "cpu".

    Returns:
        The metrics and tables that went into the report.
    """
    ensure_dirs()
    model, checkpoint = load_run(name)
    device = select_device(device_preference)
    model = model.to(device)

    # Rebuild the exact split the model was trained on, fractions and seed
    # included, so that the rows scored here are the ones it never saw.
    splits, _ = prepare_data(DataConfig.from_row(checkpoint["config"]), csv_path)
    _, _, test_loader = make_loaders(splits, batch_size=TrainConfig().batch_size)

    y_pred, y_true, probabilities = test_predictions(model, test_loader, device)
    per_class = per_class_report(y_true, y_pred)
    confusion = confusion_frame(y_true, y_pred)
    calibration = calibration_table(y_true, probabilities)

    payload: dict[str, Any] = {
        "name": name,
        "n_test": int(len(y_true)),
        "n_parameters": count_parameters(model),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(per_class.loc["macro avg", "f1-score"]),
        "ece": expected_calibration_error(calibration),
        "per_class": per_class,
        "confusion": confusion,
        "top_confusions": top_confusions(confusion),
        "calibration": calibration,
    }

    plot_confusion_matrix(confusion, FIGURES / f"confusion_{name}.png")
    plot_confident_mistakes(
        splits,
        confident_mistakes(y_true, y_pred, probabilities),
        y_true,
        y_pred,
        probabilities,
        FIGURES / f"confident_mistakes_{name}.png",
    )
    history_path = REPORTS / f"history_{name}.csv"
    if history_path.exists():
        plot_training_curves(pd.read_csv(history_path), FIGURES / f"curves_{name}.png")

    payload["report_path"] = write_report(payload)
    return payload


def main() -> None:
    """Entry point for `python -m src.evaluate`."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--name", default="baseline", help="run name of the checkpoint to load")
    parser.add_argument("--csv", type=Path, default=None, help="path to train.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    payload = evaluate_run(args.name, args.csv, args.device)
    print(f"test accuracy {payload['accuracy']:.4f} on {payload['n_test']:,} rows")
    print(f"macro F1 {payload['macro_f1']:.4f}, calibration error {payload['ece']:.4f}")
    print("\nmost frequent mistakes:")
    for true, pred, count in payload["top_confusions"]:
        print(f"  {true} read as {pred}: {count}")
    print(f"\nwrote {payload['report_path']}")


if __name__ == "__main__":
    main()
