"""The training loop, in plain PyTorch.

The forward pass, the loss, the backward pass and the optimizer step are all
written out here. `src/lightning_module.py` wraps these same pieces to run the
identical model through Lightning.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.augment import augment_batch
from src.config import LossName, TrainConfig
from src.models import count_parameters
from src.utils import select_device, set_seed

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def accumulator_dtype(device: torch.device) -> torch.dtype:
    """Return the dtype an epoch's running totals are summed in.

    float64 everywhere except MPS, which has no float64 kernels.
    """
    return torch.float32 if device.type == "mps" else torch.float64


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Build the optimizer named in the configuration.

    The learning rate does not transport between them: SGD needs one to two
    orders of magnitude more than Adam.

    Raises:
        ValueError: if the optimizer name is unknown.
    """
    params = model.parameters()
    lr, decay = config.learning_rate, config.weight_decay

    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=decay)
    if config.optimizer == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=config.momentum, weight_decay=decay)
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=decay)
    if config.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=config.momentum, weight_decay=decay)
    raise ValueError(f"Unknown optimizer {config.optimizer!r}.")


def build_loss_fn(name: LossName, label_smoothing: float = 0.1) -> LossFn:
    """Return a loss that takes raw logits and integer targets.

    - cross_entropy: the standard choice for single-label classification.
    - label_smoothing: the same, with target mass spread over the wrong
      classes. Caps how confident the logits are driven to become.
    - nll: log-softmax then negative log-likelihood, which is cross-entropy
      written out.
    - mse: squared error against a one-hot target. Its gradient vanishes where
      the model is most wrong, so it converges slowly here.

    The class count for the one-hot target is read off the logits rather than
    passed in, because a one-hot wider than the logits broadcasts silently.

    Raises:
        ValueError: if the loss name is unknown.
    """
    if name == "cross_entropy":
        return F.cross_entropy
    if name == "label_smoothing":
        return lambda logits, targets: F.cross_entropy(
            logits, targets, label_smoothing=label_smoothing
        )
    if name == "nll":
        return lambda logits, targets: F.nll_loss(F.log_softmax(logits, dim=-1), targets)
    if name == "mse":
        return lambda logits, targets: F.mse_loss(
            F.softmax(logits, dim=-1),
            F.one_hot(targets, num_classes=logits.shape[-1]).float(),
        )
    raise ValueError(f"Unknown loss {name!r}.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> tuple[LRScheduler | None, bool]:
    """Build the learning-rate schedule named in the configuration.

    - none: constant rate.
    - step: multiply by 0.1 every third of the budget.
    - cosine: anneal smoothly to the minimum rate over the budget.
    - onecycle: warm up to the peak rate, then anneal below the starting value.
      Steps per batch, not per epoch.

    `step`, `cosine` and `onecycle` all spread themselves over
    `config.max_epochs`, so a run that stops early ends part way through its
    schedule. Set `patience` to the budget for any run whose numbers are
    reported.

    Args:
        optimizer: The optimizer whose rate is scheduled.
        config: Holds the schedule name and the epoch budget.
        steps_per_epoch: Number of batches per epoch, needed by onecycle.

    Returns:
        The scheduler, or None for a constant rate, and whether it is stepped
        after every batch instead of after every epoch.

    Raises:
        ValueError: if the schedule name is unknown.
    """
    if config.scheduler == "none":
        return None, False
    if config.scheduler == "step":
        step_size = max(config.max_epochs // 3, 1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1), False
    if config.scheduler == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.max_epochs, eta_min=config.learning_rate * 0.01
            ),
            False,
        )
    if config.scheduler == "onecycle":
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.learning_rate,
                epochs=config.max_epochs,
                steps_per_epoch=max(steps_per_epoch, 1),
            ),
            True,
        )
    raise ValueError(f"Unknown scheduler {config.scheduler!r}.")


@dataclass
class EpochRecord:
    """Metrics from one epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    seconds: float
    learning_rate: float = 0.0
    grad_norms: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, float]:
        """Flatten the record, gradient norms included, into one table row."""
        row: dict[str, float] = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "seconds": self.seconds,
            "learning_rate": self.learning_rate,
        }
        row.update({f"grad_norm/{name}": value for name, value in self.grad_norms.items()})
        return row


@dataclass
class TrainingResult:
    """A finished training run: the fitted model and everything it recorded.

    `best_epoch`, `best_val_accuracy` and `best_val_loss` all describe the one
    epoch whose weights were kept, so they always belong to the same model. The
    run's lowest validation loss, which early stopping watches, may sit at a
    different epoch; read it off `history`.
    """

    model: nn.Module
    history: list[EpochRecord]
    best_epoch: int
    best_val_accuracy: float
    best_val_loss: float
    n_parameters: int
    total_seconds: float
    stopped_early: bool

    def history_frame(self) -> pd.DataFrame:
        """Return the per-epoch history as a dataframe."""
        return pd.DataFrame([record.to_row() for record in self.history])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: LossFn,
    device: torch.device,
) -> tuple[float, float]:
    """Return the average loss and the accuracy of a model on one loader."""
    # The batches are moved below, so the model has to be on the same device.
    # A checkpoint loads onto the CPU, and scoring it on the GPU without this
    # fails on the first matrix multiply.
    model.eval().to(device)
    total_loss = torch.zeros((), device=device, dtype=accumulator_dtype(device))
    n_correct = torch.zeros((), device=device, dtype=torch.long)
    n_seen = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += loss_fn(logits, targets) * targets.size(0)
        n_correct += (logits.argmax(dim=-1) == targets).sum()
        n_seen += targets.size(0)

    if n_seen == 0:
        raise ValueError("Cannot evaluate on an empty loader: it yielded no rows.")
    return float(total_loss.item()) / n_seen, int(n_correct.item()) / n_seen


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader.

    Returns:
        Logits of shape (n_rows, n_classes) and the matching targets, both on
        the CPU.
    """
    model.eval().to(device)
    all_logits, all_targets = [], []

    for inputs, targets in loader:
        all_logits.append(model(inputs.to(device)).cpu())
        all_targets.append(targets)

    return torch.cat(all_logits), torch.cat(all_targets)


def _layer_gradient_norms(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return the L2 norm of the gradient of every linear weight matrix.

    Read down the layers, these show a vanishing or exploding gradient directly.
    Normalisation weights are skipped: they are per-feature scales, not matrices
    the signal passes through.

    The norms stay on the device as tensors. A `.item()` here would sync the GPU
    on every batch, which for a model this small costs more than the step.
    """
    return {
        name: module.weight.grad.detach().norm()
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.weight.grad is not None
    }


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
    scheduler: LRScheduler | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Run one pass over the training data.

    Args:
        model: Network to update, in place.
        loader: Training batches.
        loss_fn: Objective to minimise.
        optimizer: Optimizer holding the parameters.
        device: Where the batches are moved to.
        config: Supplies the augmentation settings.
        scheduler: Stepped after every batch when given. Only the one-cycle
            schedule is passed here; the per-epoch ones are stepped by the
            caller.

    Returns:
        Average loss, accuracy, and the per-layer gradient norms averaged over
        the batches of the epoch.
    """
    model.train()
    # Running totals stay on the device and are read back once, after the epoch.
    total_loss = torch.zeros((), device=device, dtype=accumulator_dtype(device))
    n_correct = torch.zeros((), device=device, dtype=torch.long)
    n_seen = 0
    norm_totals: dict[str, torch.Tensor] = {}

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if config.augment:
            # Augment after the move, so the warp runs on the training device.
            inputs = augment_batch(inputs, config.max_shift, config.max_rotation)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for name, norm in _layer_gradient_norms(model).items():
            norm_totals[name] = norm_totals.get(name, norm.new_zeros(())) + norm

        total_loss += loss.detach() * targets.size(0)
        n_correct += (logits.argmax(dim=-1) == targets).sum()
        n_seen += targets.size(0)

    if n_seen == 0:
        raise ValueError(
            "The training loader yielded no batches. The training split is smaller "
            "than one batch; lower batch_size or raise max_rows."
        )
    n_batches = max(len(loader), 1)
    grad_norms = {name: float(total.item()) / n_batches for name, total in norm_totals.items()}
    return float(total_loss.item()) / n_seen, int(n_correct.item()) / n_seen, grad_norms


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train a model and return it with the weights of its best epoch.

    Two different metrics are in play. Early stopping watches the validation
    loss, which turns up as soon as the model starts to overfit. The weights
    kept are those of the best validation *accuracy*, the headline metric.
    They usually pick the same epoch; the history shows it when they do not.

    The learning rate each epoch was trained with is recorded alongside its
    metrics, so a schedule can be read off the run.

    Args:
        model: The network to train. Modified in place.
        train_loader: Batches used for gradient steps.
        val_loader: Batches used for model selection only.
        config: Optimizer, loss, epoch budget and patience.
        verbose: Whether to print one line per epoch.

    Returns:
        The fitted model, its history and the selected epoch.
    """
    cfg = config or TrainConfig()
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    model = model.to(device)

    optimizer = build_optimizer(model, cfg)
    loss_fn = build_loss_fn(cfg.loss, cfg.label_smoothing)
    scheduler, steps_per_batch = build_scheduler(optimizer, cfg, len(train_loader))

    history: list[EpochRecord] = []
    best_weights = copy.deepcopy(model.state_dict())
    best_val_accuracy, best_val_loss, best_epoch = 0.0, float("inf"), 0
    # Early stopping watches the lowest loss of the run; model selection keeps
    # the best accuracy. They are usually the same epoch, but not always, so
    # they are counted separately rather than through one variable.
    lowest_val_loss = float("inf")
    epochs_without_improvement, stopped_early = 0, False
    started = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        epoch_started = time.perf_counter()
        # Read the rate before the schedule advances it: this is the one the
        # epoch was actually trained with.
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss, train_accuracy, grad_norms = _train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            cfg,
            scheduler if steps_per_batch else None,
        )
        if scheduler is not None and not steps_per_batch:
            scheduler.step()
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                seconds=time.perf_counter() - epoch_started,
                learning_rate=learning_rate,
                grad_norms=grad_norms,
            )
        )
        if verbose:
            print(
                f"epoch {epoch:>3}  "
                f"train loss {train_loss:.4f}  acc {train_accuracy:.4f}  |  "
                f"val loss {val_loss:.4f}  acc {val_accuracy:.4f}"
            )

        # The first epoch is the incumbent whatever it scored, so that a run
        # which never beats chance still reports the epoch it kept.
        if epoch == 1 or val_accuracy > best_val_accuracy:
            best_val_accuracy, best_val_loss, best_epoch = val_accuracy, val_loss, epoch
            best_weights = copy.deepcopy(model.state_dict())

        if val_loss < lowest_val_loss:
            lowest_val_loss, epochs_without_improvement = val_loss, 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                stopped_early = True
                if verbose:
                    # Name the metric. Stopping watches the loss while selection
                    # keeps the best accuracy, so "no improvement" on its own
                    # reads as a failure when the accuracy was still climbing.
                    print(
                        f"early stop at epoch {epoch}: val loss has not improved for "
                        f"{cfg.patience} epochs (lowest {lowest_val_loss:.4f}) -- "
                        f"keeping epoch {best_epoch}, val accuracy {best_val_accuracy:.4f}"
                    )
                break

    model.load_state_dict(best_weights)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        best_val_loss=best_val_loss,
        n_parameters=count_parameters(model),
        total_seconds=time.perf_counter() - started,
        stopped_early=stopped_early,
    )
