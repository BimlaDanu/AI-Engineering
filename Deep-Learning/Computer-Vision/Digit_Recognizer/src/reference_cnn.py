r"""A small convolutional network, for comparison only.

The model this project builds is the MLP. This exists so that "use a CNN", the
first item on the improvement list, can be measured rather than asserted. It
trains on the same data, the same split, the same loop and the same epoch
budget.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.config import ConvConfig, DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import DataSplits, make_loaders, prepare_data
from src.train import train_from_splits
from src.training import TrainingResult, build_loss_fn, evaluate, train_model
from src.utils import REPORTS, ensure_dirs, select_device, set_seed

IMAGE_SIZE = 28


class SmallCNN(nn.Module):
    """Conv-BN-ReLU-MaxPool blocks followed by a small classifier head.

    Takes the same flat (batch, 784) tensors as the MLP and reshapes them
    internally, so the data pipeline, training loop and evaluation are shared.
    """

    def __init__(self, config: ConvConfig | None = None) -> None:
        """Build the network described by the configuration."""
        super().__init__()
        self.config = config or ConvConfig()
        self.features = self._build_features(self.config)
        side = IMAGE_SIZE // 2 ** len(self.config.channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.channels[-1] * side * side, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.num_classes),
        )

    @staticmethod
    def _build_features(config: ConvConfig) -> nn.Sequential:
        """Assemble the convolutional blocks."""
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in config.channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        layers.append(nn.Dropout(config.dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of images to class logits.

        Args:
            x: Float tensor of shape (batch, 784) or (batch, 1, 28, 28).

        Returns:
            Logits of shape (batch, num_classes).
        """
        square = x.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        logits: torch.Tensor = self.classifier(self.features(square))
        return logits


def build_cnn(config: ConvConfig | None = None) -> SmallCNN:
    """Build the reference CNN from its configuration."""
    return SmallCNN(config)


def train_cnn(
    splits: DataSplits,
    config: ExperimentConfig,
    conv: ConvConfig | None = None,
    verbose: bool = False,
) -> TrainingResult:
    """Train the reference CNN with the same loop and data as the MLP.

    Args:
        splits: Data prepared once and shared with the MLP run.
        config: Supplies the training block; its model block is ignored.
        conv: Shape of the convolutional network.
        verbose: Whether to print per-epoch progress.

    Returns:
        The finished training run.
    """
    set_seed(config.train.seed)
    train_loader, val_loader, _ = make_loaders(
        splits, batch_size=config.train.batch_size, num_workers=config.train.num_workers
    )
    return train_model(build_cnn(conv), train_loader, val_loader, config.train, verbose=verbose)


def test_accuracy(model: nn.Module, splits: DataSplits, config: TrainConfig) -> float:
    """Score a trained model on the held-out test split."""
    device = select_device(config.device)
    _, _, test_loader = make_loaders(splits, batch_size=config.batch_size)
    _, accuracy = evaluate(
        model, test_loader, build_loss_fn(config.loss, config.label_smoothing), device
    )
    return accuracy


def write_report(rows: list[dict[str, float | str]], path: Path | None = None) -> Path:
    """Write the MLP-versus-CNN comparison as a markdown table.

    Returns:
        Path of the written file.
    """
    ensure_dirs()
    destination = path or REPORTS / "mlp_vs_cnn.md"
    lines = [
        "# Multi-layer perceptron versus a convolutional network",
        "",
        "Same data, same split, same training loop, same epoch budget.",
        "The only difference is how the first layers read the image.",
        "",
        "| model | parameters | val accuracy | test accuracy | seconds |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {int(row['parameters']):,} | "
            f"{float(row['val_accuracy']):.4f} | {float(row['test_accuracy']):.4f} | "
            f"{float(row['seconds']):.1f} |"
        )
    lines.append("")
    destination.write_text("\n".join(lines))
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the comparison."""
    parser = argparse.ArgumentParser(description="Compare the MLP with a small CNN.")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None, help="cap the rows, for smoke runs")
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 256])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--csv", type=Path, default=None, help="path to train.csv")
    return parser


def main() -> None:
    """Entry point for `python -m src.reference_cnn`."""
    args = build_parser().parse_args()
    config = ExperimentConfig(
        name="mlp_vs_cnn",
        data=DataConfig(max_rows=args.max_rows),
        model=ModelConfig(hidden_sizes=tuple(args.hidden_sizes)),
        train=TrainConfig(
            max_epochs=args.max_epochs,
            device=args.device,
            scheduler="cosine",
            # Both models train the schedule out, so the comparison is not
            # between one annealed run and one cut off part way.
            patience=args.max_epochs,
        ),
    )

    splits, report = prepare_data(config.data, args.csv)
    print(report.summary())
    print(f"\nsplit sizes: {splits.sizes}\n")

    mlp = train_from_splits(splits, config, verbose=False)
    cnn = train_cnn(splits, config)
    rows: list[dict[str, float | str]] = [
        {
            "model": f"MLP {list(config.model.hidden_sizes)}",
            # The run already counted its own model; rebuilding one here would
            # report a second network's size beside the first one's accuracy.
            "parameters": mlp.n_parameters,
            "val_accuracy": mlp.best_val_accuracy,
            "test_accuracy": test_accuracy(mlp.model, splits, config.train),
            "seconds": mlp.total_seconds,
        },
        {
            "model": "CNN 32-64",
            "parameters": cnn.n_parameters,
            "val_accuracy": cnn.best_val_accuracy,
            "test_accuracy": test_accuracy(cnn.model, splits, config.train),
            "seconds": cnn.total_seconds,
        },
    ]

    for row in rows:
        print(
            f"{row['model']:<18} {int(row['parameters']):>9,} params  "
            f"val {float(row['val_accuracy']):.4f}  test {float(row['test_accuracy']):.4f}"
        )
    print(f"\nwrote {write_report(rows)}")


if __name__ == "__main__":
    main()
