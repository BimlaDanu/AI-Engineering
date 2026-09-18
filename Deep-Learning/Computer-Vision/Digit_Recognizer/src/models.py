"""The multi-layer perceptron and its building blocks."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from src.config import Activation, ModelConfig

# Activations the experiments compare. Each entry builds a fresh module.
ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

# Activations whose gain matches Kaiming initialisation. The saturating ones
# below are paired with Xavier instead.
_RELU_FAMILY = {"relu", "leaky_relu", "gelu"}


def build_activation(name: Activation) -> nn.Module:
    """Return a fresh activation module.

    Args:
        name: Key into ACTIVATIONS.

    Raises:
        ValueError: if the name is unknown.
    """
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}. Available: {sorted(ACTIVATIONS)}.")
    return ACTIVATIONS[name]()


class MLP(nn.Module):
    """A fully connected classifier with a configurable hidden stack.

    Each hidden block is Linear -> (BatchNorm) -> Activation -> (Dropout), and a
    final linear layer maps to one logit per class.

    Returns raw logits, never probabilities: CrossEntropyLoss applies
    log-softmax itself, and a second softmax flattens the gradients silently.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Build the network described by the configuration."""
        super().__init__()
        self.config = config or ModelConfig()
        self.network = self._build_layers(self.config)
        self.apply(self._initialise_weights)

    @staticmethod
    def _build_layers(config: ModelConfig) -> nn.Sequential:
        """Assemble the hidden blocks and the output layer."""
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = config.input_dim

        for width in config.hidden_sizes:
            layers.append(nn.Linear(in_features, width))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(build_activation(config.activation))
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_features = width

        layers.append(nn.Linear(in_features, config.num_classes))
        return nn.Sequential(*layers)

    def _initialise_weights(self, module: nn.Module) -> None:
        """Match the initialisation to the activation.

        Kaiming holds the activation variance roughly constant across depth for
        the ReLU family, Xavier for the saturating ones.
        """
        if not isinstance(module, nn.Linear):
            return
        if self.config.activation in _RELU_FAMILY:
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of images to class logits.

        Args:
            x: Float tensor of shape (batch, 784) or (batch, 1, 28, 28).

        Returns:
            Logits of shape (batch, num_classes).
        """
        logits: torch.Tensor = self.network(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: ModelConfig | None = None) -> MLP:
    """Build an MLP from its configuration."""
    return MLP(config)
