"""The Lightning wrappers must train the same network as the plain loop."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.config import DataConfig, ExperimentConfig, ModelConfig, Scheduler, TrainConfig
from src.lightning_module import DigitsDataModule, LitMLP, train_with_lightning

SMALL = ExperimentConfig(
    name="lightning-test",
    data=DataConfig(max_rows=200),
    model=ModelConfig(hidden_sizes=(16,)),
    train=TrainConfig(max_epochs=1, batch_size=32, device="cpu"),
)


def test_forward_matches_the_plain_model_interface() -> None:
    module = LitMLP(SMALL)
    assert module(torch.randn(4, 784)).shape == (4, 10)


def test_it_builds_the_configured_optimizer() -> None:
    module = LitMLP(ExperimentConfig(train=TrainConfig(optimizer="sgd_momentum")))
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.SGD)


def test_the_datamodule_serves_all_three_splits(csv_path: Path) -> None:
    datamodule = DigitsDataModule(SMALL, csv_path)
    datamodule.setup()
    assert datamodule.splits is not None
    assert set(datamodule.splits.sizes) == {"train", "val", "test"}


def test_the_datamodule_reads_the_csv_once(csv_path: Path) -> None:
    """Re-reading per loader would triple the work and could reshuffle the split."""
    datamodule = DigitsDataModule(SMALL, csv_path)
    datamodule.train_dataloader()
    first = datamodule.splits
    datamodule.val_dataloader()
    assert datamodule.splits is first


@pytest.mark.slow
def test_a_full_lightning_run_reports_validation_metrics(csv_path: Path) -> None:
    metrics = train_with_lightning(SMALL, csv_path, accelerator="cpu")
    assert "val_accuracy" in metrics
    assert 0.0 <= metrics["val_accuracy"] <= 1.0


@pytest.mark.slow
def test_lightning_and_the_plain_loop_reach_a_similar_place(csv_path: Path) -> None:
    """Not bit-identical -- the two loops differ in order of operations -- but
    they share the model, loss and optimizer, so neither should be far off."""
    from src.data import prepare_data
    from src.train import train_from_splits

    config = ExperimentConfig(
        data=DataConfig(max_rows=400),
        model=ModelConfig(hidden_sizes=(32,)),
        train=TrainConfig(max_epochs=3, device="cpu", seed=0),
    )
    splits, _ = prepare_data(config.data, csv_path)
    plain = train_from_splits(splits, config, verbose=False).best_val_accuracy
    lightning = train_with_lightning(config, csv_path, accelerator="cpu")["val_accuracy"]

    assert abs(plain - lightning) < 0.35


def learning_rates_per_epoch(config: ExperimentConfig, csv_path: Path) -> list[float]:
    """Run a short Lightning fit and return the rate each epoch started with."""
    import lightning as L

    rates: list[float] = []

    class RecordRate(L.Callback):
        def on_train_epoch_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
            rates.append(round(float(trainer.optimizers[0].param_groups[0]["lr"]), 8))

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[RecordRate()],
    )
    trainer.fit(LitMLP(config), datamodule=DigitsDataModule(config, csv_path))
    return rates


@pytest.mark.slow
@pytest.mark.parametrize("scheduler", ["cosine", "onecycle"])
def test_the_wrapper_honours_the_configured_schedule(scheduler: Scheduler, csv_path: Path) -> None:
    """It used to ignore it: a run asked for cosine trained at a flat rate, silently."""
    config = replace(SMALL, train=replace(SMALL.train, max_epochs=3, scheduler=scheduler))
    rates = learning_rates_per_epoch(config, csv_path)
    assert len(set(rates)) > 1, f"{scheduler} left the rate at {rates}"


@pytest.mark.slow
def test_no_schedule_means_a_constant_rate(csv_path: Path) -> None:
    config = replace(SMALL, train=replace(SMALL.train, max_epochs=3, scheduler="none"))
    assert len(set(learning_rates_per_epoch(config, csv_path))) == 1


def test_the_wrapper_augments_only_when_the_configuration_asks(
    csv_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And only the training batches: warping validation would score the wrong images."""
    import src.lightning_module as module

    calls: list[int] = []
    original = module.augment_batch

    def counting_augment(images: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(images.shape[0])
        return original(images, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "augment_batch", counting_augment)
    # The steps are called directly, without a Trainer, because what is under
    # test is which batches get warped. self.log needs a trainer, so stub it.
    monkeypatch.setattr(LitMLP, "log", lambda *args, **kwargs: None)

    off = replace(SMALL, train=replace(SMALL.train, augment=False))
    LitMLP(off).training_step((torch.randn(8, 784), torch.zeros(8, dtype=torch.long)), 0)
    assert calls == []

    on = replace(SMALL, train=replace(SMALL.train, augment=True))
    LitMLP(on).training_step((torch.randn(8, 784), torch.zeros(8, dtype=torch.long)), 0)
    assert calls == [8]

    calls.clear()
    LitMLP(on).validation_step((torch.randn(8, 784), torch.zeros(8, dtype=torch.long)), 0)
    assert calls == [], "validation batches must not be warped"
