"""PyTorch Lightning wrappers around the same model and data.

The same run, expressed through Lightning. It reuses `MLP`, `build_optimizer`,
`build_loss_fn`, `build_scheduler` and `augment_batch` unchanged, so there is
no second implementation to keep in step with the plain loop.

One difference to know about: the plain loop restores the weights of its best
epoch, while `trainer.validate` below scores whatever the last epoch left.
Checkpointing is off here; the reported model comes from `src/train.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import DataLoader

from src.augment import augment_batch
from src.config import ExperimentConfig
from src.data import DataSplits, make_loaders, prepare_data
from src.models import build_model
from src.training import build_loss_fn, build_optimizer, build_scheduler
from src.utils import set_seed


class LitMLP(L.LightningModule):
    """Lightning wrapper: the MLP, the loss and the optimizer, nothing more."""

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        """Build the model described by the configuration."""
        super().__init__()
        self.config = config or ExperimentConfig()
        self.model: nn.Module = build_model(self.config.model)
        self.loss_fn = build_loss_fn(self.config.train.loss, self.config.train.label_smoothing)
        self.save_hyperparameters(self.config.to_row())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images."""
        logits: torch.Tensor = self.model(x)
        return logits

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared body of the training, validation and test steps."""
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One gradient step, on warped inputs when the configuration asks for them.

        Augmentation belongs here rather than in `_step`: it is a property of
        how the model is trained, and applying it to the validation batches
        would score the model on images it was never asked about.
        """
        inputs, targets = batch
        if self.config.train.augment:
            inputs = augment_batch(
                inputs, self.config.train.max_shift, self.config.train.max_rotation
            )
        return self._step((inputs, targets), "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One validation batch."""
        return self._step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One test batch."""
        return self._step(batch, "test")

    def _steps_per_epoch(self) -> int:
        """Optimizer steps in one epoch, which only the one-cycle schedule needs.

        Lightning can only answer this once the trainer holds the dataloaders,
        and the module can also be built without a trainer at all, so fall back
        to 1. Rounded up, because a one-cycle schedule raises if it is stepped
        past the total it was built for.
        """
        if self._trainer is None:
            return 1
        epochs = max(self.config.train.max_epochs, 1)
        return max(-(-int(self.trainer.estimated_stepping_batches) // epochs), 1)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the optimizer, and the schedule when the configuration names one.

        Without this the wrapper ignores `scheduler` silently, and a Lightning
        run asked for cosine annealing trains at a constant rate instead.
        """
        optimizer = build_optimizer(self.model, self.config.train)
        scheduler, per_batch = build_scheduler(
            optimizer, self.config.train, self._steps_per_epoch()
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" if per_batch else "epoch",
            },
        }


class DigitsDataModule(L.LightningDataModule):
    """Lightning view of the same cleaning, splitting and loaders."""

    def __init__(
        self, config: ExperimentConfig | None = None, csv_path: Path | None = None
    ) -> None:
        """Store the configuration; the data is read in `setup`."""
        super().__init__()
        self.config = config or ExperimentConfig()
        self.csv_path = csv_path
        self.splits: DataSplits | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load, clean and split the data once."""
        if self.splits is None:
            self.splits, _ = prepare_data(self.config.data, self.csv_path)

    def _loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build the three loaders, calling setup first if needed."""
        self.setup()
        assert self.splits is not None
        return make_loaders(
            self.splits,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training batches."""
        return self._loaders()[0]

    def val_dataloader(self) -> DataLoader:
        """Validation batches."""
        return self._loaders()[1]

    def test_dataloader(self) -> DataLoader:
        """Test batches. Used once, at the end."""
        return self._loaders()[2]


def train_with_lightning(
    config: ExperimentConfig | None = None,
    csv_path: Path | None = None,
    accelerator: str = "auto",
) -> dict[str, Any]:
    """Train through Lightning and return its validation metrics.

    Args:
        config: The run configuration.
        csv_path: Location of train.csv.
        accelerator: Passed to the Trainer; "auto" picks the available device.

    Returns:
        The metrics Lightning collected on the validation split.
    """
    cfg = config or ExperimentConfig()
    set_seed(cfg.train.seed)

    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
        # The progress bar redraws once per batch, and a notebook stores every
        # redraw as its own output rather than collapsing the carriage returns.
        # At batch size 32 that is 788 per epoch: one executed run wrote a 52 MB
        # cell. The bar tells a watching human nothing the epoch lines do not.
        enable_progress_bar=False,
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.train.patience)
        ],
    )
    module = LitMLP(cfg)
    datamodule = DigitsDataModule(cfg, csv_path)
    trainer.fit(module, datamodule=datamodule)
    return {
        key: float(value)
        for key, value in trainer.validate(module, datamodule=datamodule)[0].items()
    }
