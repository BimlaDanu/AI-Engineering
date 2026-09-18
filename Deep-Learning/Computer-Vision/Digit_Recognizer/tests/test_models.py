"""Model construction: shapes, options and initialisation."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import nn

from src.config import Activation, ModelConfig, TrainConfig
from src.models import ACTIVATIONS, MLP, build_activation, build_model, count_parameters


def test_forward_returns_one_logit_per_class() -> None:
    model = build_model(ModelConfig(hidden_sizes=(32, 16)))
    logits = model(torch.randn(8, 784))
    assert logits.shape == (8, 10)


def test_forward_accepts_image_shaped_input() -> None:
    model = build_model(ModelConfig(hidden_sizes=(16,)))
    assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)


def test_output_is_logits_not_probabilities() -> None:
    model = build_model(ModelConfig(hidden_sizes=(16,)))
    model.eval()
    sums = model(torch.randn(4, 784)).sum(dim=-1)
    # Probabilities would sum to one; logits have no such constraint.
    assert not torch.allclose(sums, torch.ones_like(sums))


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_every_activation_builds_and_runs(activation: Activation) -> None:
    model = build_model(ModelConfig(hidden_sizes=(16,), activation=activation))
    assert model(torch.randn(4, 784)).shape == (4, 10)


def test_build_activation_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        build_activation("swish")  # type: ignore[arg-type]  # deliberately invalid


def test_empty_hidden_stack_is_a_linear_classifier() -> None:
    model = build_model(ModelConfig(hidden_sizes=()))
    # Flatten plus one Linear, and 784 * 10 weights plus 10 biases.
    assert count_parameters(model) == 784 * 10 + 10


def test_parameter_count_matches_the_architecture() -> None:
    model = build_model(ModelConfig(hidden_sizes=(100,), batch_norm=False))
    expected = 784 * 100 + 100 + 100 * 10 + 10
    assert count_parameters(model) == expected


def test_dropout_and_batch_norm_can_be_switched_off() -> None:
    model = build_model(ModelConfig(hidden_sizes=(16,), dropout=0.0, batch_norm=False))
    kinds = {type(layer) for layer in model.network}
    assert nn.Dropout not in kinds
    assert nn.BatchNorm1d not in kinds


def test_dropout_changes_the_output_only_in_training_mode() -> None:
    model = build_model(ModelConfig(hidden_sizes=(64,), dropout=0.5))
    batch = torch.randn(16, 784)

    model.eval()
    assert torch.allclose(model(batch), model(batch))

    model.train()
    assert not torch.allclose(model(batch), model(batch))


def test_configuration_rejects_an_impossible_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.0)


def test_relu_and_tanh_get_different_initialisations() -> None:
    torch.manual_seed(0)
    relu_model = MLP(ModelConfig(hidden_sizes=(256,), activation="relu"))
    torch.manual_seed(0)
    tanh_model = MLP(ModelConfig(hidden_sizes=(256,), activation="tanh"))

    # Index 1 is the first Linear layer; index 0 is the Flatten.
    relu_std = cast(nn.Linear, relu_model.network[1]).weight.std().item()
    tanh_std = cast(nn.Linear, tanh_model.network[1]).weight.std().item()
    # Kaiming carries a gain of sqrt(2) that Xavier does not.
    assert relu_std > tanh_std


@pytest.mark.parametrize("hidden_sizes", [(0,), (-4,), (16, 0)])
def test_a_layer_of_no_units_is_rejected_by_name(hidden_sizes: tuple[int, ...]) -> None:
    """Torch would report this as a tensor dimension, several frames away."""
    with pytest.raises(ValueError, match="at least one unit"):
        ModelConfig(hidden_sizes=hidden_sizes)


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda: TrainConfig(batch_size=0), "batch_size"),
        (lambda: TrainConfig(max_epochs=0), "max_epochs"),
        (lambda: TrainConfig(patience=0), "patience"),
    ],
)
def test_a_run_budget_of_nothing_is_rejected(build: Callable[[], TrainConfig], field: str) -> None:
    """A budget of zero reaches the loop as an epoch range that trains nothing."""
    with pytest.raises(ValueError, match=field):
        build()
