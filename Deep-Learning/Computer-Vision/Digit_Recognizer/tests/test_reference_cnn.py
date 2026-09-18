"""The reference convolutional network used for the MLP comparison."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import ConvConfig, DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import build_splits
from src.models import build_model, count_parameters
from src.reference_cnn import IMAGE_SIZE, SmallCNN, build_cnn, train_cnn, write_report
from tests.conftest import make_frame

SMALL = ExperimentConfig(
    name="cnn",
    data=DataConfig(max_rows=200),
    model=ModelConfig(hidden_sizes=(8,)),
    train=TrainConfig(max_epochs=1, batch_size=32, device="cpu"),
)
TINY = ConvConfig(channels=(4, 8), hidden_size=16)


def test_it_accepts_the_flat_batches_the_mlp_is_trained_on() -> None:
    """Sharing the data pipeline is the point: no separate loader for the CNN."""
    logits = build_cnn(TINY)(torch.randn(5, IMAGE_SIZE * IMAGE_SIZE))
    assert logits.shape == (5, 10)


def test_it_accepts_square_batches_too() -> None:
    logits = build_cnn(TINY)(torch.randn(5, 1, IMAGE_SIZE, IMAGE_SIZE))
    assert logits.shape == (5, 10)


def test_the_output_is_logits_not_probabilities() -> None:
    """Softmax belongs to the loss; applying it here would flatten the gradients."""
    logits = build_cnn(TINY)(torch.randn(8, IMAGE_SIZE * IMAGE_SIZE))
    assert not torch.allclose(logits.sum(dim=-1), torch.ones(8))
    assert (logits < 0).any()


def test_each_block_halves_the_feature_map() -> None:
    model = SmallCNN(ConvConfig(channels=(4, 8)))
    features = model.features(torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE))
    assert features.shape == (2, 8, IMAGE_SIZE // 4, IMAGE_SIZE // 4)


def test_a_network_deep_enough_to_pool_the_image_away_is_rejected() -> None:
    with pytest.raises(ValueError, match="pool down to nothing"):
        ConvConfig(channels=(8,) * 6)


def test_a_network_with_no_blocks_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ConvConfig(channels=())


def test_weight_sharing_buys_the_comparison_its_point() -> None:
    """The CNN reaches its accuracy with fewer parameters than the MLP, because
    one kernel is reused at every position instead of one weight per pixel."""
    cnn = count_parameters(build_cnn())
    mlp = count_parameters(build_model(ModelConfig(hidden_sizes=(512, 256))))
    assert cnn < mlp


def test_the_translation_response_moves_with_the_digit() -> None:
    """Shifting the input shifts the feature map, which is exactly the property
    a flattened MLP does not have."""
    model = SmallCNN(ConvConfig(channels=(4,))).eval()
    image = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    image[0, 0, 8:12, 8:12] = 1.0

    with torch.no_grad():
        original = model.features(image)
        shifted = model.features(torch.roll(image, shifts=2, dims=3))
    assert torch.allclose(original[:, :, :, :-1], shifted[:, :, :, 1:], atol=1e-5)


@pytest.mark.slow
def test_it_trains_through_the_same_loop_as_the_mlp() -> None:
    splits = build_splits(make_frame(200), DataConfig())
    result = train_cnn(splits, SMALL, TINY)
    assert len(result.history) == 1
    assert result.n_parameters == count_parameters(build_cnn(TINY))


def test_the_comparison_report_lists_both_models(tmp_path: Path) -> None:
    rows: list[dict[str, float | str]] = [
        {
            "model": "MLP [512, 256]",
            "parameters": 535_818,
            "val_accuracy": 0.98,
            "test_accuracy": 0.979,
            "seconds": 42.0,
        },
        {
            "model": "CNN 32-64",
            "parameters": 421_642,
            "val_accuracy": 0.992,
            "test_accuracy": 0.991,
            "seconds": 61.0,
        },
    ]
    text = write_report(rows, tmp_path / "cnn.md").read_text()
    assert "MLP [512, 256]" in text
    assert "535,818" in text
    assert "0.9910" in text
