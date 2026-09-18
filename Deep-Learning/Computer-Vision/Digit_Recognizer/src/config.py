"""Configuration objects for the training pipeline.

Every setting the experiments vary lives in one of these dataclasses, so a run
is fully described by its configuration and writes to a results table as one
row.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Activation = Literal["relu", "leaky_relu", "gelu", "tanh", "sigmoid"]
Optimizer = Literal["sgd", "sgd_momentum", "adam", "adamw", "rmsprop"]
LossName = Literal["cross_entropy", "label_smoothing", "nll", "mse"]
Scheduler = Literal["none", "step", "cosine", "onecycle"]


@dataclass(frozen=True)
class DataConfig:
    """How the raw CSV is turned into three splits.

    Attributes:
        train_fraction: Share of rows used for training.
        val_fraction: Share of rows used for validation.
        seed: Seed for the split, fixed so that models are compared on the
            same partition rather than on different ones.
        standardize: Whether to standardise pixels with the training mean and
            standard deviation after scaling them to [0, 1].
        max_rows: Optional cap on the number of rows used, sampled while keeping
            the class proportions. Meant for fast local smoke runs; leave it at
            None for any result that gets reported.
    """

    train_fraction: float = 0.6
    val_fraction: float = 0.2
    seed: int = 42
    standardize: bool = True
    max_rows: int | None = None

    @property
    def test_fraction(self) -> float:
        """Share of rows left for the held-out test split."""
        return 1.0 - self.train_fraction - self.val_fraction

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> DataConfig:
        """Rebuild a split configuration from a row written by `ExperimentConfig.to_row`.

        Evaluation reads this out of the checkpoint so it scores on the split
        the model was trained away from. Every field has to come back, not just
        the seed, or the test split moves under the model.

        `max_rows=None` goes through a CSV as an empty field and returns as
        float NaN, which happens to behave like None but does not read like it.
        Restore the None.

        Args:
            row: Flattened configuration; keys are prefixed with "data.".

        Returns:
            The split configuration, with defaults for any missing key.
        """
        defaults = asdict(cls())
        values = {key: row.get(f"data.{key}", value) for key, value in defaults.items()}
        if isinstance(values["max_rows"], float) and math.isnan(values["max_rows"]):
            values["max_rows"] = None
        return cls(**values)

    def __post_init__(self) -> None:
        """Reject fractions that do not leave a usable test split."""
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between 0 and 1.")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie strictly between 0 and 1.")
        if self.test_fraction <= 0.0:
            raise ValueError(
                "train_fraction + val_fraction must be below 1 so that a test split remains."
            )


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the multi-layer perceptron.

    Attributes:
        hidden_sizes: Width of each hidden layer, in order. Empty means a plain
            linear classifier, which is a useful floor to compare against.
        activation: Nonlinearity used after every hidden layer.
        dropout: Dropout probability applied after each activation; 0 disables it.
        batch_norm: Whether to insert batch normalisation before each activation.
        input_dim: Number of input features (784 pixels for MNIST).
        num_classes: Number of output classes.
    """

    hidden_sizes: tuple[int, ...] = (512, 256)
    activation: Activation = "relu"
    dropout: float = 0.2
    batch_norm: bool = True
    input_dim: int = 784
    num_classes: int = 10

    def __post_init__(self) -> None:
        """Reject a shape the network cannot be built from.

        A width of zero reaches torch as a tensor dimension, and the error it
        raises there names the dimension rather than the setting behind it.
        """
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if any(width < 1 for width in self.hidden_sizes):
            raise ValueError(
                f"Every hidden layer needs at least one unit, got {self.hidden_sizes}."
            )
        if self.input_dim < 1 or self.num_classes < 2:
            raise ValueError("input_dim must be positive and num_classes at least 2.")


@dataclass(frozen=True)
class ConvConfig:
    """Shape of the reference convolutional network.

    Used only for the MLP-versus-CNN comparison, not for the reported model.

    Attributes:
        channels: Output channels of each convolutional block. Every block
            halves the spatial size, so two blocks take 28x28 to 7x7.
        hidden_size: Width of the fully connected layer before the classifier.
        dropout: Dropout probability after the pooling and the hidden layer.
        num_classes: Number of output classes.
    """

    channels: tuple[int, ...] = (32, 64)
    hidden_size: int = 128
    dropout: float = 0.25
    num_classes: int = 10

    def __post_init__(self) -> None:
        """Reject a network that would pool the image away entirely."""
        if not self.channels:
            raise ValueError("channels must name at least one convolutional block.")
        if any(width < 1 for width in self.channels) or self.hidden_size < 1:
            raise ValueError("Every channel count and the hidden size must be positive.")
        if 28 // 2 ** len(self.channels) < 1:
            raise ValueError("Too many blocks: the feature map would pool down to nothing.")


@dataclass(frozen=True)
class TrainConfig:
    """How a single model is trained.

    Attributes:
        optimizer: Which optimizer to build.
        learning_rate: Step size. Note that it does not transport between
            optimizers; SGD needs a much larger value than Adam.
        weight_decay: L2 penalty, applied by the optimizer.
        momentum: Used by sgd_momentum and rmsprop only.
        batch_size: Samples per gradient step.
        max_epochs: Upper bound on epochs; early stopping usually ends sooner.
        patience: Epochs without validation-loss improvement before stopping.
        loss: Which objective to minimise.
        label_smoothing: Smoothing factor, used by the label_smoothing loss.
        scheduler: Learning-rate schedule. Every schedule spreads itself over
            max_epochs, so a run that stops early ends part way through it.
        augment: Whether to apply random shifts and rotations to the training
            batches.
        max_shift: Largest translation in pixels, when augment is on.
        max_rotation: Largest rotation in degrees, when augment is on.
        seed: Seed for initialisation and batch shuffling.
        device: "auto", "cuda", "mps" or "cpu".
        num_workers: DataLoader worker processes. 0 is fastest here because the
            whole dataset already sits in memory as tensors.
    """

    optimizer: Optimizer = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 10
    loss: LossName = "cross_entropy"
    label_smoothing: float = 0.1
    scheduler: Scheduler = "none"
    augment: bool = False
    max_shift: int = 2
    max_rotation: float = 10.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        """Reject settings a run cannot be made from."""
        if self.max_shift < 0:
            raise ValueError("max_shift must be zero or positive.")
        if self.max_rotation < 0.0:
            raise ValueError("max_rotation must be zero or positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1: a run of no epochs trains nothing.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")


@dataclass(frozen=True)
class ExperimentConfig:
    """One complete run: a name plus the three configuration blocks."""

    name: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_row(self) -> dict[str, Any]:
        """Flatten the configuration into one row for the results table."""
        row: dict[str, Any] = {"name": self.name}
        for block, values in (
            ("data", asdict(self.data)),
            ("model", asdict(self.model)),
            ("train", asdict(self.train)),
        ):
            for key, value in values.items():
                row[f"{block}.{key}"] = list(value) if isinstance(value, tuple) else value
        return row
