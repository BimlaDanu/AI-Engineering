"""Paths, seeding and device selection shared by the rest of the package."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

# All paths are relative to the project root so that nothing is written outside it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"

TRAIN_CSV = DATA_RAW / "train.csv"
# The committed copies. The uncompressed CSVs are 73 MB and 49 MB and are
# gitignored; gzipped they are 8.9 MB and 5.6 MB, so they travel with the
# repository and a cloud session needs no download.
TRAIN_CSV_GZ = DATA_RAW / "train.csv.gz"
# The competition's unlabelled set. Nothing here reads it -- accuracy cannot be
# computed without labels -- but it is what a leaderboard submission predicts on.
TEST_CSV_GZ = DATA_RAW / "test.csv.gz"


def full_study_is_affordable(preference: str = "auto") -> bool:
    """Whether this machine can run the 132-run study in a sensible time.

    CUDA only. MPS trains one model well enough and is roughly twice as slow per
    epoch, which over 132 runs is the difference between an hour and three, so
    an Apple laptop counts as a small machine here.

    Args:
        preference: The requested device. An explicit "cpu" or "mps" means the
            full study is not affordable whatever else is attached.

    Returns:
        True when the full sweep is worth starting.
    """
    if preference in {"cpu", "mps"}:
        return False
    return torch.cuda.is_available()


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so a run can be repeated.

    Comparable, not bit-identical: cuDNN and MPS kernels may still pick
    different reduction orders.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preference: str = "auto") -> torch.device:
    """Return the device to train on.

    Args:
        preference: "auto", "cuda", "mps" or "cpu". "auto" picks the fastest
            available, preferring cuda over mps over cpu.

    Raises:
        RuntimeError: if a specific device is requested but is not available.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine.")
    return torch.device(preference)


def describe_device(device: torch.device) -> str:
    """Return a one-line human-readable description of a device."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"


def ensure_dirs() -> None:
    """Create the output directories used for artifacts, if they are missing."""
    for directory in (DATA_PROCESSED, REPORTS, FIGURES, CHECKPOINTS):
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Print the interpreter, the accelerator and the key package versions."""
    import platform
    import sys

    device = select_device()
    print(f"python   {sys.version.split()[0]} ({platform.machine()})")
    print(f"torch    {torch.__version__}")
    print(f"numpy    {np.__version__}")
    print(f"device   {describe_device(device)}")


if __name__ == "__main__":
    main()
