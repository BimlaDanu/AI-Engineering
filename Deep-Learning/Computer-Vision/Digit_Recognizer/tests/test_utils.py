"""Paths, seeding and device selection."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.utils import (
    CHECKPOINTS,
    DATA_PROCESSED,
    DATA_RAW,
    FIGURES,
    PROJECT_ROOT,
    REPORTS,
    TRAIN_CSV,
    describe_device,
    ensure_dirs,
    full_study_is_affordable,
    select_device,
    set_seed,
)


@pytest.mark.parametrize(
    "path", [DATA_RAW, DATA_PROCESSED, REPORTS, FIGURES, CHECKPOINTS, TRAIN_CSV]
)
def test_every_path_stays_inside_the_project(path) -> None:  # type: ignore[no-untyped-def]
    """Nothing this package writes may land outside the repository."""
    assert PROJECT_ROOT in path.parents


def test_the_directories_are_created_where_they_are_expected() -> None:
    ensure_dirs()
    for directory in (DATA_PROCESSED, REPORTS, FIGURES, CHECKPOINTS):
        assert directory.is_dir()


def test_seeding_repeats_every_generator_the_pipeline_uses() -> None:
    """Weight initialisation, batch order and the augmentation draws all
    depend on these three, so one of them left unseeded is enough to make a
    run irreproducible."""
    set_seed(3)
    drawn = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(3)
    assert drawn == (random.random(), np.random.rand(), torch.rand(1).item())


def test_different_seeds_draw_differently() -> None:
    set_seed(3)
    first = torch.rand(4)
    set_seed(4)
    assert not torch.equal(first, torch.rand(4))


def test_auto_never_asks_for_a_device_that_is_missing() -> None:
    device = select_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}


def test_cpu_is_always_available() -> None:
    assert select_device("cpu").type == "cpu"


@pytest.mark.parametrize("preference", ["cuda", "mps"])
def test_an_unavailable_accelerator_is_refused_rather_than_silently_downgraded(
    preference: str,
) -> None:
    """A run asked for a GPU must fail loudly. Falling back to the CPU would
    turn a two-hour sweep into a two-day one without saying so."""
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if available[preference]:
        assert select_device(preference).type == preference
        return
    with pytest.raises(RuntimeError, match="not available"):
        select_device(preference)


def test_the_device_description_names_the_kind_of_hardware() -> None:
    assert describe_device(torch.device("cpu")) == "cpu"


@pytest.mark.parametrize("preference", ["cpu", "mps"])
def test_an_explicit_small_device_rules_the_full_study_out(preference: str) -> None:
    """Asking for the CPU says what the machine is, whatever else is attached."""
    assert full_study_is_affordable(preference) is False


def test_the_full_study_follows_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA is the only thing fast enough for 132 runs; MPS is not."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert full_study_is_affordable() is True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert full_study_is_affordable() is False
