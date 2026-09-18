"""The hyperparameter study.

Ten factors: optimizer, batch size, activation, learning rate, dropout,
architecture and loss, plus schedule, weight decay and augmentation, which the
same machinery covers.

A full grid over ten factors is not affordable, so this is a screening design:
one factor varied at a time from a fixed baseline, every configuration repeated
over several seeds. Results are read as mean +/- spread across seeds, and a gap
smaller than that spread is not a result.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import DataSplits, prepare_data
from src.train import train_from_splits
from src.training import TrainingResult
from src.utils import REPORTS, ensure_dirs, full_study_is_affordable

# The reference configuration every sweep varies one field of.
BASELINE = ExperimentConfig(
    name="baseline",
    data=DataConfig(),
    model=ModelConfig(hidden_sizes=(512, 256), activation="relu", dropout=0.2, batch_norm=True),
    train=TrainConfig(optimizer="adamw", learning_rate=1e-3, batch_size=128, max_epochs=50),
)

# Factor -> levels to try. The names match the dataclass fields.
FACTORS: dict[str, list[Any]] = {
    "optimizer": ["sgd", "sgd_momentum", "adam", "adamw", "rmsprop"],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "batch_size": [32, 64, 128, 256, 512],
    "activation": ["relu", "leaky_relu", "gelu", "tanh", "sigmoid"],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
    "hidden_sizes": [(128,), (512,), (512, 256), (1024, 512, 256), (512, 512, 512, 512)],
    "loss": ["cross_entropy", "label_smoothing", "nll", "mse"],
    "scheduler": ["none", "step", "cosine", "onecycle"],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "augment": [False, True],
}

# Fields that belong to ModelConfig; everything else belongs to TrainConfig.
MODEL_FACTORS = {"hidden_sizes", "activation", "dropout", "batch_norm"}

# A deliberately small profile: the whole sweep runs locally in a few minutes on
# CPU. It is for checking that the pipeline works, never for reporting numbers.
QUICK_BASELINE = ExperimentConfig(
    name="quick",
    data=DataConfig(max_rows=3_000),
    model=ModelConfig(hidden_sizes=(64,), dropout=0.1),
    train=TrainConfig(max_epochs=3, patience=2, batch_size=256),
)
QUICK_FACTORS: dict[str, list[Any]] = {
    "optimizer": ["sgd_momentum", "adamw"],
    "learning_rate": [1e-3, 1e-2],
    "batch_size": [128, 256],
    "activation": ["relu", "tanh"],
    "dropout": [0.0, 0.3],
    "hidden_sizes": [(32,), (64, 32)],
    "loss": ["cross_entropy", "mse"],
    "scheduler": ["none", "cosine"],
    "weight_decay": [0.0, 1e-3],
    "augment": [False, True],
}


def configuration_for(factor: str, value: Any, baseline: ExperimentConfig) -> ExperimentConfig:
    """Return the baseline with a single field changed.

    Args:
        factor: Name of a ModelConfig or TrainConfig field.
        value: The level to set it to.
        baseline: Configuration to start from.

    Returns:
        A new configuration, named after the change.

    Raises:
        ValueError: if the factor is not a known field.
    """
    name = f"{factor}={_label(value)}"
    if factor in MODEL_FACTORS:
        return replace(baseline, name=name, model=replace(baseline.model, **{factor: value}))
    if factor in TrainConfig.__dataclass_fields__:
        return replace(baseline, name=name, train=replace(baseline.train, **{factor: value}))
    raise ValueError(f"Unknown factor {factor!r}.")


def _label(value: Any) -> str:
    """Render a factor level compactly enough for a run name and a table."""
    if isinstance(value, tuple):
        return "-".join(str(v) for v in value)
    return str(value)


def sweep_configurations(
    factors: dict[str, list[Any]],
    baseline: ExperimentConfig,
    seeds: list[int],
) -> Iterator[tuple[str, Any, int, ExperimentConfig]]:
    """Yield every (factor, level, seed, configuration) the sweep will run."""
    for factor, levels in factors.items():
        for value in levels:
            for seed in seeds:
                config = configuration_for(factor, value, baseline)
                yield factor, value, seed, replace(config, train=replace(config.train, seed=seed))


def _result_row(factor: str, value: Any, seed: int, result: TrainingResult) -> dict[str, Any]:
    """Turn one finished run into a row of the results table."""
    best = result.history[result.best_epoch - 1] if result.history else None
    return {
        "factor": factor,
        "level": _label(value),
        "seed": seed,
        "val_accuracy": result.best_val_accuracy,
        "train_accuracy": best.train_accuracy if best else float("nan"),
        "val_loss": best.val_loss if best else float("nan"),
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "n_parameters": result.n_parameters,
        "seconds": round(result.total_seconds, 1),
        "stopped_early": result.stopped_early,
        "error": "",
    }


def _failed_row(factor: str, value: Any, seed: int, error: BaseException) -> dict[str, Any]:
    """Turn one run that raised into a row, so the sweep can carry on past it."""
    return {
        "factor": factor,
        "level": _label(value),
        "seed": seed,
        "val_accuracy": float("nan"),
        "train_accuracy": float("nan"),
        "val_loss": float("nan"),
        "best_epoch": 0,
        "epochs_run": 0,
        "n_parameters": 0,
        "seconds": 0.0,
        "stopped_early": False,
        "error": f"{type(error).__name__}: {error}",
    }


def run_sweep(
    splits: DataSplits,
    factors: dict[str, list[Any]],
    baseline: ExperimentConfig = BASELINE,
    seeds: list[int] | None = None,
    output_path: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train every configuration of the sweep and collect the results.

    Rows are written to the CSV as they finish, so an interrupted sweep still
    leaves usable results.

    A run that raises is recorded with its error and the sweep continues, rather
    than losing 131 good runs to the 132nd. `summarise` counts only the runs
    that produced a number, so a level that failed on one seed is averaged over
    the seeds that worked and its `n_runs` shows the shortfall.

    Args:
        splits: Data prepared once and shared by every run.
        factors: Factor to levels mapping.
        baseline: Configuration each run varies one field of.
        seeds: Seeds to repeat every configuration over.
        output_path: Where to write the per-run table.
        verbose: Whether to print one line per finished run.

    Returns:
        One row per (factor, level, seed).
    """
    seed_list = seeds or [0, 1, 2]
    destination = output_path or REPORTS / "experiments.csv"
    ensure_dirs()

    rows: list[dict[str, Any]] = []
    planned = list(sweep_configurations(factors, baseline, seed_list))

    for index, (factor, value, seed, config) in enumerate(planned, start=1):
        label = f"[{index:>3}/{len(planned)}] {factor}={_label(value)} seed={seed}"
        try:
            result = train_from_splits(splits, config, verbose=False)
        except Exception as error:  # noqa: BLE001 - one bad run must not end the sweep
            rows.append(_failed_row(factor, value, seed, error))
            # Printed whatever `verbose` says: a silently skipped run is how a
            # sweep comes back looking complete while missing a third of itself.
            print(f"{label}  FAILED  {type(error).__name__}: {error}")
        else:
            rows.append(_result_row(factor, value, seed, result))
            if verbose:
                print(
                    f"{label}  val acc {result.best_val_accuracy:.4f}  "
                    f"({result.total_seconds:.1f}s)"
                )
        pd.DataFrame(rows).to_csv(destination, index=False)

    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Average the repeated runs of each level.

    Returns:
        One row per (factor, level) with the mean and standard deviation of the
        validation accuracy. The standard deviation is the yardstick a gap
        between two levels has to clear.
    """
    summary = (
        results.groupby(["factor", "level"], sort=False)
        .agg(
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            # count, not size: a run that failed is not a run that scored 0.
            n_runs=("val_accuracy", "count"),
            mean_epochs=("epochs_run", "mean"),
            mean_seconds=("seconds", "mean"),
            n_parameters=("n_parameters", "max"),
        )
        .reset_index()
    )
    summary["val_accuracy_std"] = summary["val_accuracy_std"].fillna(0.0)
    best_per_factor = summary.groupby("factor")["val_accuracy_mean"].transform("max")
    summary["best_of_factor"] = summary["val_accuracy_mean"] == best_per_factor
    return summary


def write_summary(summary: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the aggregated sweep as a markdown table.

    Returns:
        Path of the written file.
    """
    ensure_dirs()
    destination = path or REPORTS / "experiment_summary.md"
    lines = [
        "# Hyperparameter study",
        "",
        "One factor varied at a time from the baseline, repeated over seeds.",
        "A difference smaller than the standard deviation is not a result.",
        "",
    ]
    for factor, block in summary.groupby("factor", sort=False):
        lines += [
            f"## {factor}",
            "",
            "| level | val accuracy | std | epochs | params |",
            "|---|---|---|---|---|",
        ]
        for row in block.itertuples():
            marker = " **<-**" if row.best_of_factor else ""
            lines.append(
                f"| {row.level} | {row.val_accuracy_mean:.4f}{marker} | "
                f"{row.val_accuracy_std:.4f} | {row.mean_epochs:.0f} | {row.n_parameters:,} |"
            )
        lines.append("")

    destination.write_text("\n".join(lines))
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the sweep."""
    parser = argparse.ArgumentParser(description="Run the MLP hyperparameter study.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="force the short sweep: small model, few epochs, subsampled data, 20 runs",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="force the full 132-run study even without a CUDA GPU",
    )
    parser.add_argument(
        "--factors",
        nargs="*",
        default=None,
        help=f"subset of factors to sweep (default: all of {', '.join(FACTORS)})",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="seeds per level")
    parser.add_argument("--max-epochs", type=int, default=None, help="override the epoch budget")
    parser.add_argument("--max-rows", type=int, default=None, help="cap the number of rows used")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--csv", type=Path, default=None, help="path to train.csv")
    parser.add_argument("--out", type=Path, default=None, help="where to write experiments.csv")
    return parser


def summary_path_for(output_path: Path | None) -> Path | None:
    """Return where the markdown summary of a sweep belongs.

    A sweep split into chunks gives each chunk its own `--out`. The summary has
    to follow, or the last chunk overwrites the earlier ones and the report
    covers only the factors that happened to run last.

    Args:
        output_path: The `--out` value, or None for the default sweep.

    Returns:
        A path beside the chunk's CSV, or None to use the default location.
    """
    if output_path is None:
        return None
    return output_path.with_name(f"{output_path.stem}_summary.md")


def sweep_paths(quick: bool, output_path: Path | None = None) -> tuple[Path, Path]:
    """Return where a sweep writes its per-run table and its summary.

    The short sweep writes to its own names, so trying the pipeline out does not
    overwrite a full study's results. An explicit `--out` wins, and its summary
    sits beside it.

    Args:
        quick: Whether this is the short sweep.
        output_path: The `--out` value, if one was given.

    Returns:
        The CSV path and the markdown summary path.
    """
    if output_path is not None:
        summary = summary_path_for(output_path)
        assert summary is not None
        return output_path, summary
    suffix = "-quick" if quick else ""
    return REPORTS / f"experiments{suffix}.csv", REPORTS / f"experiment_summary{suffix}.md"


def best_levels(
    summary: pd.DataFrame, factors: dict[str, list[Any]] | None = None
) -> dict[str, Any]:
    """Return the highest-scoring level of each factor, typed as the field wants.

    The summary stores levels as labels, so `(1024, 512, 256)` comes back as the
    string `"1024-512-256"`. This maps each label to the value it was rendered
    from, which is what a configuration needs.

    Args:
        summary: The table `summarise` returns.
        factors: The factor to levels mapping the sweep ran, needed to recover
            the values behind the labels. Defaults to the full study's.

    Returns:
        Factor name to level, ready to pass to `configuration_from`. Factors
        whose runs all failed are left out.

    Raises:
        ValueError: if a winning label matches no level of that factor.
    """
    available = factors if factors is not None else FACTORS
    winners: dict[str, Any] = {}
    for name, block in summary.groupby("factor", sort=False):
        factor = str(name)
        if factor not in available or block["val_accuracy_mean"].isna().all():
            continue
        label = block.loc[block["val_accuracy_mean"].idxmax(), "level"]
        levels = {_label(value): value for value in available[factor]}
        if label not in levels:
            raise ValueError(f"{factor}={label!r} is not one of its swept levels.")
        winners[factor] = levels[label]
    return winners


def configuration_from(winners: dict[str, Any], baseline: ExperimentConfig) -> ExperimentConfig:
    """Apply several factor levels at once to the baseline.

    Args:
        winners: Factor name to level, as `best_levels` returns.
        baseline: Configuration to start from.

    Returns:
        The baseline with every named field changed.
    """
    config = baseline
    for factor, value in winners.items():
        config = configuration_for(factor, value, config)
    return replace(config, name=baseline.name)


def use_quick_sweep(force_quick: bool, force_full: bool, device: str = "auto") -> bool:
    """Decide between the short sweep and the full study.

    Without a flag the hardware decides. The full study is 132 training runs:
    about an hour on a T4, two to three on an Apple-silicon laptop. A local
    `make experiments` should not start that by accident.

    Args:
        force_quick: `--quick` was passed.
        force_full: `--full` was passed.
        device: The requested device, which can rule the full study out.

    Returns:
        True for the short sweep.

    Raises:
        SystemExit: if both flags are given, which cannot be honoured.
    """
    if force_quick and force_full:
        raise SystemExit("--quick and --full contradict each other. Pass one, or neither.")
    if force_quick:
        return True
    if force_full:
        return False
    return not full_study_is_affordable(device)


def main() -> None:
    """Entry point for `python -m src.experiments`."""
    args = build_parser().parse_args()

    quick = use_quick_sweep(args.quick, args.full, args.device)
    baseline = QUICK_BASELINE if quick else BASELINE
    factors = QUICK_FACTORS if quick else FACTORS

    # `--factors` and `--seeds` take a list, so typing either one with nothing
    # after it parses as the empty list. Falling back to the default there
    # would run the whole sweep -- hours of it -- for someone who asked for
    # less, so say so instead.
    if args.seeds is not None and not args.seeds:
        raise SystemExit("--seeds was given no values. Name at least one, or leave the flag off.")
    if args.factors is not None and not args.factors:
        raise SystemExit(
            f"--factors was given no values. Name at least one of {sorted(factors)}, "
            "or leave the flag off to sweep all of them."
        )
    seeds = args.seeds or ([0] if quick else [0, 1, 2])

    if args.factors:
        unknown = set(args.factors) - set(factors)
        if unknown:
            raise SystemExit(f"Unknown factors: {sorted(unknown)}. Available: {sorted(factors)}.")
        factors = {name: factors[name] for name in args.factors}
    if args.max_epochs is not None:
        baseline = replace(baseline, train=replace(baseline.train, max_epochs=args.max_epochs))
    if args.max_rows is not None:
        baseline = replace(baseline, data=replace(baseline.data, max_rows=args.max_rows))
    baseline = replace(baseline, train=replace(baseline.train, device=args.device))

    splits, report = prepare_data(baseline.data, args.csv)
    print(report.summary())
    print(f"\nsplit sizes: {splits.sizes}")
    runs = sum(len(levels) for levels in factors.values()) * len(seeds)
    print(
        f"{'short' if quick else 'full'} sweep: {len(factors)} factors, "
        f"{runs} runs over seeds {seeds}\n"
    )

    csv_path, summary_path = sweep_paths(quick, args.out)
    results = run_sweep(splits, factors, baseline, seeds, output_path=csv_path)
    summary = summarise(results)
    print()
    print(summary.to_string(index=False))
    print(f"\nwrote {csv_path}")
    print(f"wrote {write_summary(summary, summary_path)}")


if __name__ == "__main__":
    main()
