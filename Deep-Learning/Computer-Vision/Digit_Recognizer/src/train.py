"""Train one model from the command line.

    python -m src.train --hidden-sizes 512 256 --optimizer adamw --lr 1e-3

Every field the experiments vary has a flag, so a run is reproducible from its
command line alone.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import DataSplits, make_loaders, prepare_data
from src.models import MLP, build_model
from src.training import TrainingResult, train_model
from src.utils import CHECKPOINTS, REPORTS, ensure_dirs, set_seed


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for a single training run."""
    parser = argparse.ArgumentParser(description="Train an MLP on the digit-recognizer data.")
    parser.add_argument("--name", default="baseline", help="name used for the saved artifacts")
    parser.add_argument("--csv", type=Path, default=None, help="path to train.csv")

    model = parser.add_argument_group("model")
    model.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 256])
    model.add_argument(
        "--activation",
        default="relu",
        choices=["relu", "leaky_relu", "gelu", "tanh", "sigmoid"],
    )
    model.add_argument("--dropout", type=float, default=0.2)
    model.add_argument("--no-batch-norm", action="store_true", help="disable batch normalisation")

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument(
        "--optimizer",
        default="adamw",
        choices=["sgd", "sgd_momentum", "adam", "adamw", "rmsprop"],
    )
    optimisation.add_argument("--lr", type=float, default=1e-3, dest="learning_rate")
    optimisation.add_argument("--weight-decay", type=float, default=1e-4)
    optimisation.add_argument("--batch-size", type=int, default=128)
    optimisation.add_argument("--max-epochs", type=int, default=50)
    optimisation.add_argument("--patience", type=int, default=10)
    optimisation.add_argument(
        "--loss",
        default="cross_entropy",
        choices=["cross_entropy", "label_smoothing", "nll", "mse"],
    )
    optimisation.add_argument("--label-smoothing", type=float, default=0.1)
    optimisation.add_argument(
        "--scheduler",
        default="none",
        choices=["none", "step", "cosine", "onecycle"],
        help="learning-rate schedule; the rate used by each epoch is written to the history",
    )

    augmentation = parser.add_argument_group("augmentation")
    augmentation.add_argument(
        "--augment",
        action="store_true",
        help="randomly shift and rotate the training batches",
    )
    augmentation.add_argument("--max-shift", type=int, default=2, help="largest shift in pixels")
    augmentation.add_argument(
        "--max-rotation", type=float, default=10.0, help="largest rotation in degrees"
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="use only this many rows, sampled per class; for local smoke runs",
    )
    run.add_argument(
        "--seed", type=int, default=TrainConfig().seed, help="seed for the weights and batch order"
    )
    run.add_argument(
        "--split-seed",
        type=int,
        default=DataConfig().seed,
        help="seed for the train/validation/test partition; left fixed so that changing "
        "--seed compares models rather than partitions",
    )
    run.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    run.add_argument("--quiet", action="store_true", help="do not print per-epoch progress")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Turn parsed arguments into an experiment configuration."""
    return ExperimentConfig(
        name=args.name,
        # The split seed is separate from the training seed on purpose: the sweep
        # repeats a configuration over training seeds while holding the
        # partition fixed, and a command line has to be able to reproduce that.
        data=DataConfig(seed=args.split_seed, max_rows=args.max_rows),
        model=ModelConfig(
            hidden_sizes=tuple(args.hidden_sizes),
            activation=args.activation,
            dropout=args.dropout,
            batch_norm=not args.no_batch_norm,
        ),
        train=TrainConfig(
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            loss=args.loss,
            label_smoothing=args.label_smoothing,
            scheduler=args.scheduler,
            augment=args.augment,
            max_shift=args.max_shift,
            max_rotation=args.max_rotation,
            seed=args.seed,
            device=args.device,
        ),
    )


def check_run_name(name: str) -> str:
    """Return the name if it is usable as a file name, else raise.

    Every artifact of a run is named after it, so `--name ../../thing` would
    write above the project root.

    Raises:
        ValueError: if the name is empty or is not a plain file name.
    """
    if not name or name in {".", ".."} or name != Path(name).name:
        raise ValueError(
            f"--name must be a plain file name, got {name!r}. "
            "It names files under checkpoints/ and reports/, so it cannot hold '/' or '..'."
        )
    return name


def save_run(result: TrainingResult, config: ExperimentConfig) -> Path:
    """Write the weights, the configuration and the epoch history to disk.

    The configuration is stored as plain dicts, not dataclasses, so the
    checkpoint loads with `weights_only=True` and never executes code.

    Returns:
        Path of the checkpoint file.
    """
    ensure_dirs()
    name = check_run_name(config.name)
    checkpoint_path = CHECKPOINTS / f"{name}.pt"
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "config": config.to_row(),
            "model_config": asdict(config.model),
            "best_epoch": result.best_epoch,
            "best_val_accuracy": result.best_val_accuracy,
        },
        checkpoint_path,
    )
    result.history_frame().to_csv(REPORTS / f"history_{name}.csv", index=False)
    return checkpoint_path


def load_run(name: str) -> tuple[MLP, dict[str, Any]]:
    """Load a saved model and its checkpoint metadata.

    Args:
        name: The run name used when it was saved.

    Returns:
        The model with its trained weights, and the rest of the checkpoint.

    Raises:
        FileNotFoundError: if no checkpoint with that name exists.
    """
    checkpoint_path = CHECKPOINTS / f"{check_run_name(name)}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found. Train the model first.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    fields = dict(checkpoint["model_config"])
    fields["hidden_sizes"] = tuple(fields["hidden_sizes"])
    model = build_model(ModelConfig(**fields))
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def train_from_splits(
    splits: DataSplits, config: ExperimentConfig, verbose: bool = True
) -> TrainingResult:
    """Train one model on splits that have already been prepared.

    Separate from `run` so a sweep reuses one cleaned, split copy of the data
    instead of re-reading the CSV for all 132 runs.

    Args:
        splits: Tensors for the three splits.
        config: The full run configuration.
        verbose: Whether to print per-epoch progress.

    Returns:
        The finished training run.
    """
    # Seed here, not inside train_model: weight initialisation happens in
    # build_model, so a seed set afterwards would leave every run starting from
    # different weights and make the sweep irreproducible.
    set_seed(config.train.seed)
    train_loader, val_loader, _ = make_loaders(
        splits, batch_size=config.train.batch_size, num_workers=config.train.num_workers
    )
    model = build_model(config.model)
    return train_model(model, train_loader, val_loader, config.train, verbose=verbose)


def run(
    config: ExperimentConfig, csv_path: Path | None = None, verbose: bool = True
) -> TrainingResult:
    """Prepare the data, train one model and report the outcome.

    Args:
        config: The full run configuration.
        csv_path: Location of train.csv, if not the default.
        verbose: Whether to print the cleaning report and per-epoch progress.

    Returns:
        The finished training run.
    """
    splits, report = prepare_data(config.data, csv_path)
    if verbose:
        print(report.summary())
        print(f"\nsplit sizes: {splits.sizes}\n")

    result = train_from_splits(splits, config, verbose=verbose)

    if verbose:
        print(
            f"\nbest epoch {result.best_epoch}: "
            f"val accuracy {result.best_val_accuracy:.4f}, "
            f"{result.n_parameters:,} parameters, "
            f"{result.total_seconds:.1f}s"
        )
    return result


def main() -> None:
    """Entry point for `python -m src.train`."""
    args = build_parser().parse_args()
    config = config_from_args(args)
    result = run(config, csv_path=args.csv, verbose=not args.quiet)
    checkpoint_path = save_run(result, config)
    print(f"saved {checkpoint_path}")
    print(json.dumps({"val_accuracy": round(result.best_val_accuracy, 4)}))


if __name__ == "__main__":
    main()
