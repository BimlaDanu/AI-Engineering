r"""Seed ensembling and test-time augmentation.

Seed ensembling trains the same architecture from several initialisations and
averages the class probabilities. Test-time augmentation averages one model's
predictions over several shifted and rotated copies of the same image.

Both are read on validation first; the test split is scored once, at the end.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.augment import augment_batch
from src.config import ExperimentConfig
from src.data import DataSplits, make_loaders, prepare_data
from src.train import train_from_splits
from src.training import TrainingResult, predict
from src.utils import REPORTS, ensure_dirs, select_device, set_seed


@torch.no_grad()
def model_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Return the softmax probabilities of one model over a loader."""
    logits, _ = predict(model, loader, device)
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def tta_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_views: int = 4,
    max_shift: int = 2,
    max_rotation: float = 10.0,
    seed: int = 0,
) -> torch.Tensor:
    """Average one model's predictions over augmented copies of each image.

    The first view is always the unmodified image, so the average is anchored on
    the input the model was asked about; the rest are random warps of it.

    Args:
        model: A trained network.
        loader: Batches to predict. Must not shuffle, or the views would be
            averaged across different images.
        device: Where to run the forward passes.
        n_views: Total number of views, the original included.
        max_shift: Largest translation in pixels for the extra views.
        max_rotation: Largest rotation in degrees for the extra views.
        seed: Seed for the warps, so the report can be reproduced.

    Returns:
        Probabilities of shape (n_rows, n_classes), on the CPU.
    """
    model.eval().to(device)
    generator = torch.Generator().manual_seed(seed)
    batches: list[torch.Tensor] = []

    for inputs, _ in loader:
        inputs = inputs.to(device)
        total = F.softmax(model(inputs), dim=-1)
        for _ in range(max(n_views - 1, 0)):
            warped = augment_batch(inputs, max_shift, max_rotation, generator)
            total = total + F.softmax(model(warped), dim=-1)
        batches.append((total / max(n_views, 1)).cpu())

    return torch.cat(batches)


def average_probabilities(
    models: Sequence[nn.Module], loader: DataLoader, device: torch.device
) -> torch.Tensor:
    """Average the class probabilities of several models over a loader.

    Probabilities, not logits: averaging unbounded logits lets one
    over-confident member dominate the vote.

    Raises:
        ValueError: if no models are given.
    """
    if not models:
        raise ValueError("An ensemble needs at least one model.")
    total = model_probabilities(models[0], loader, device)
    for model in models[1:]:
        total = total + model_probabilities(model, loader, device)
    return total / len(models)


def accuracy(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the accuracy of a probability matrix against integer targets."""
    return float((probabilities.argmax(dim=-1) == targets).float().mean())


def train_ensemble(
    splits: DataSplits,
    config: ExperimentConfig,
    seeds: list[int],
    verbose: bool = True,
) -> list[TrainingResult]:
    """Train one model per seed, on the same data and configuration.

    Only the seed changes: the members differ in initialisation and batch order
    and in nothing else.

    Args:
        splits: Data prepared once and shared by every member.
        config: The configuration every member is trained with.
        seeds: One seed per member.
        verbose: Whether to print one line per finished member.

    Returns:
        The finished runs, in seed order.
    """
    results = []
    for seed in seeds:
        member = replace(
            config, name=f"{config.name}-seed{seed}", train=replace(config.train, seed=seed)
        )
        result = train_from_splits(splits, member, verbose=False)
        results.append(result)
        if verbose:
            print(f"seed {seed}: val accuracy {result.best_val_accuracy:.4f}")
    return results


def write_report(rows: list[tuple[str, float, float]], path: Path | None = None) -> Path:
    """Write the member and ensemble accuracies as a markdown table.

    Args:
        rows: (name, validation accuracy, test accuracy) per entry.
        path: Where to write; defaults to reports/ensemble.md.

    Returns:
        Path of the written file.
    """
    ensure_dirs()
    destination = path or REPORTS / "ensemble.md"
    lines = [
        "# Ensembling and test-time augmentation",
        "",
        "Members differ only in their seed. The ensemble averages class",
        "probabilities; TTA averages one model over augmented copies of the input.",
        "",
        "| model | val accuracy | test accuracy |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {val:.4f} | {test:.4f} |" for name, val, test in rows]
    lines.append("")
    destination.write_text("\n".join(lines))
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the ensemble study."""
    parser = argparse.ArgumentParser(description="Train a seed ensemble and compare it to TTA.")
    parser.add_argument("--name", default="ensemble", help="name used in the report")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="one per member")
    parser.add_argument("--tta-views", type=int, default=4, help="views averaged per image")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--max-rows", type=int, default=None, help="cap the rows, for smoke runs")
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 256])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--csv", type=Path, default=None, help="path to train.csv")
    return parser


def main() -> None:
    """Entry point for `python -m src.ensemble`."""
    from src.config import DataConfig, ModelConfig, TrainConfig

    args = build_parser().parse_args()
    # `--seeds` takes a list, so typing it with nothing after it parses as the
    # empty list and would train no members at all.
    if not args.seeds:
        raise SystemExit("--seeds was given no values. Name at least one, or leave the flag off.")

    config = ExperimentConfig(
        name=args.name,
        data=DataConfig(max_rows=args.max_rows),
        model=ModelConfig(hidden_sizes=tuple(args.hidden_sizes)),
        train=TrainConfig(
            max_epochs=args.max_epochs,
            device=args.device,
            scheduler="cosine",
            # Cosine anneals over max_epochs, so a run early stopping ends at a
            # rate it was never meant to finish at. These few runs are reported,
            # so they train the schedule out; the kept weights are the best
            # validation epoch either way.
            patience=args.max_epochs,
        ),
    )

    splits, report = prepare_data(config.data, args.csv)
    print(report.summary())
    print(f"\nsplit sizes: {splits.sizes}\n")

    set_seed(config.train.seed)
    device = select_device(config.train.device)
    _, val_loader, test_loader = make_loaders(splits, batch_size=config.train.batch_size)
    results = train_ensemble(splits, config, args.seeds)
    models = [result.model for result in results]

    rows = [
        (
            f"member seed={seed}",
            accuracy(model_probabilities(model, val_loader, device), splits.y_val),
            accuracy(model_probabilities(model, test_loader, device), splits.y_test),
        )
        for seed, model in zip(args.seeds, models, strict=True)
    ]
    rows.append(
        (
            f"ensemble of {len(models)}",
            accuracy(average_probabilities(models, val_loader, device), splits.y_val),
            accuracy(average_probabilities(models, test_loader, device), splits.y_test),
        )
    )
    rows.append(
        (
            f"TTA x{args.tta_views} (first member)",
            accuracy(
                tta_probabilities(models[0], val_loader, device, args.tta_views), splits.y_val
            ),
            accuracy(
                tta_probabilities(models[0], test_loader, device, args.tta_views), splits.y_test
            ),
        )
    )

    for name, val, test in rows:
        print(f"{name:<28} val {val:.4f}  test {test:.4f}")
    print(f"\nwrote {write_report(rows)}")


if __name__ == "__main__":
    main()
