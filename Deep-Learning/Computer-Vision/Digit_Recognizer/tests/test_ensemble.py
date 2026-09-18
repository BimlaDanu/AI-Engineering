"""Seed ensembling and test-time augmentation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import build_splits, make_loaders
from src.ensemble import (
    accuracy,
    average_probabilities,
    build_parser,
    model_probabilities,
    train_ensemble,
    tta_probabilities,
    write_report,
)
from src.models import build_model
from tests.conftest import make_frame

SMALL = ExperimentConfig(
    name="ens",
    data=DataConfig(max_rows=200),
    model=ModelConfig(hidden_sizes=(8,), batch_norm=False),
    train=TrainConfig(max_epochs=1, batch_size=32, device="cpu"),
)
CPU = torch.device("cpu")

# Train, validation and test loaders, as make_loaders returns them.
Loaders = tuple[DataLoader, DataLoader, DataLoader]


class ConstantModel(nn.Module):
    """A stand-in that always predicts the same class, for exact arithmetic."""

    def __init__(self, predicted: int, confidence: float = 10.0) -> None:
        super().__init__()
        self.predicted, self.confidence = predicted, confidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], 10)
        logits[:, self.predicted] = self.confidence
        return logits


@pytest.fixture
def loaders() -> Loaders:
    """Loaders over a small synthetic dataset."""
    splits = build_splits(make_frame(200), DataConfig())
    return make_loaders(splits, batch_size=64)


def test_probabilities_form_a_distribution(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    probabilities = model_probabilities(
        build_model(ModelConfig(hidden_sizes=(8,))), val_loader, CPU
    )
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(probabilities.shape[0]), atol=1e-5)


def test_averaging_identical_models_changes_nothing(loaders: Loaders) -> None:
    """The gain comes from disagreement; identical members average to themselves."""
    _, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    model.eval()
    single = model_probabilities(model, val_loader, CPU)
    averaged = average_probabilities([model, model], val_loader, CPU)
    assert torch.allclose(single, averaged, atol=1e-6)


def test_the_majority_of_members_carries_the_vote(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    members = [ConstantModel(3), ConstantModel(3), ConstantModel(7)]
    averaged = average_probabilities(members, val_loader, CPU)
    assert (averaged.argmax(dim=-1) == 3).all()


def test_the_average_is_still_a_distribution(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    averaged = average_probabilities([ConstantModel(1), ConstantModel(2)], val_loader, CPU)
    assert torch.allclose(averaged.sum(dim=-1), torch.ones(averaged.shape[0]), atol=1e-5)


def test_an_empty_ensemble_is_rejected(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    with pytest.raises(ValueError, match="at least one model"):
        average_probabilities([], val_loader, CPU)


def test_one_view_of_tta_is_just_the_model(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    model.eval()
    plain = model_probabilities(model, val_loader, CPU)
    augmented = tta_probabilities(model, val_loader, CPU, n_views=1)
    assert torch.allclose(plain, augmented, atol=1e-6)


def test_tta_keeps_one_row_per_image(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    probabilities = tta_probabilities(model, val_loader, CPU, n_views=3)
    assert probabilities.shape[1] == 10
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(probabilities.shape[0]), atol=1e-5)


def test_tta_moves_the_prediction_away_from_the_plain_one(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    model.eval()
    plain = model_probabilities(model, val_loader, CPU)
    augmented = tta_probabilities(model, val_loader, CPU, n_views=4)
    assert not torch.allclose(plain, augmented, atol=1e-4)


def test_tta_is_reproducible_for_a_fixed_seed(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    model.eval()
    first = tta_probabilities(model, val_loader, CPU, n_views=3, seed=5)
    second = tta_probabilities(model, val_loader, CPU, n_views=3, seed=5)
    assert torch.allclose(first, second)


def test_accuracy_counts_the_argmax() -> None:
    probabilities = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]])
    assert accuracy(probabilities, torch.tensor([1, 0, 1])) == pytest.approx(2 / 3)


@pytest.mark.slow
def test_the_ensemble_trains_one_model_per_seed() -> None:
    splits = build_splits(make_frame(200), DataConfig())
    results = train_ensemble(splits, SMALL, seeds=[0, 1], verbose=False)
    assert len(results) == 2
    assert all(result.history for result in results)


@pytest.mark.slow
def test_members_trained_with_different_seeds_differ() -> None:
    """If the seed did not reach the weights, averaging would buy nothing."""
    splits = build_splits(make_frame(200), DataConfig())
    first, second = train_ensemble(splits, SMALL, seeds=[0, 1], verbose=False)
    weights = [dict(model.model.named_parameters()) for model in (first, second)]
    assert not torch.allclose(weights[0]["network.1.weight"], weights[1]["network.1.weight"])


def test_the_report_names_every_row(tmp_path: Path) -> None:
    path = write_report(
        [("member seed=0", 0.91, 0.90), ("ensemble", 0.93, 0.92)], tmp_path / "e.md"
    )
    text = path.read_text()
    assert "member seed=0" in text
    assert "0.9300" in text


def test_the_parser_reads_a_seed_list() -> None:
    args = build_parser().parse_args(["--seeds", "0", "1", "2", "3", "--tta-views", "6"])
    assert args.seeds == [0, 1, 2, 3]
    assert args.tta_views == 6
