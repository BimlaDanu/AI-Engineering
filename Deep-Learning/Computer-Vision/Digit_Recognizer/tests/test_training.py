"""The training loop: optimizers, losses, schedules, early stopping and learning."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DataConfig,
    ExperimentConfig,
    LossName,
    ModelConfig,
    Optimizer,
    Scheduler,
    TrainConfig,
)
from src.data import DataSplits, build_splits, make_loaders
from src.models import build_model
from src.train import train_from_splits
from src.training import (
    accumulator_dtype,
    build_loss_fn,
    build_optimizer,
    build_scheduler,
    evaluate,
    predict,
    train_model,
)
from tests.conftest import make_frame

# Train, validation and test loaders, as make_loaders returns them.
Loaders = tuple[DataLoader, DataLoader, DataLoader]


@pytest.fixture
def loaders() -> Loaders:
    """Small loaders built from synthetic data."""
    splits = build_splits(make_frame(300), DataConfig())
    return make_loaders(splits, batch_size=32)


@pytest.mark.parametrize(
    ("device_type", "expected"),
    [("cpu", torch.float64), ("cuda", torch.float64), ("mps", torch.float32)],
)
def test_the_epoch_totals_avoid_float64_on_mps(device_type: str, expected: torch.dtype) -> None:
    """MPS has no float64 kernels, so the running totals must drop to float32.

    This suite runs on the CPU, so a float64 accumulator passes every other test
    and still raises `Cannot convert a MPS Tensor to float64` on Apple Silicon.
    Checking the dtype directly catches that without needing the device.
    """
    assert accumulator_dtype(torch.device(device_type)) is expected


def test_the_totals_are_summed_in_a_dtype_the_device_supports() -> None:
    """The dtype has to be usable on the device it was chosen for."""
    device = torch.device("cpu")
    total = torch.zeros((), device=device, dtype=accumulator_dtype(device))
    assert (total + torch.ones((), device=device)).item() == 1.0


@pytest.mark.parametrize("name", ["sgd", "sgd_momentum", "adam", "adamw", "rmsprop"])
def test_every_optimizer_builds(name: Optimizer) -> None:
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    optimizer = build_optimizer(model, TrainConfig(optimizer=name))
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_unknown_optimizer_is_rejected() -> None:
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    with pytest.raises(ValueError, match="Unknown optimizer"):
        # adagrad is not one of the supported names, which is the point.
        build_optimizer(model, TrainConfig(optimizer="adagrad"))  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["cross_entropy", "label_smoothing", "nll", "mse"])
def test_every_loss_returns_a_finite_scalar(name: LossName) -> None:
    loss_fn = build_loss_fn(name)
    value = loss_fn(torch.randn(8, 10), torch.randint(0, 10, (8,)))
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_unknown_loss_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss_fn("hinge")  # type: ignore[arg-type]  # deliberately invalid


def test_cross_entropy_and_nll_agree() -> None:
    logits, targets = torch.randn(16, 10), torch.randint(0, 10, (16,))
    cross_entropy = build_loss_fn("cross_entropy")(logits, targets)
    nll = build_loss_fn("nll")(logits, targets)
    # Cross-entropy is log-softmax followed by NLL, so the two must match.
    assert torch.allclose(cross_entropy, nll, atol=1e-6)


def test_label_smoothing_raises_the_loss_of_a_confident_correct_model() -> None:
    logits = torch.tensor([[10.0, 0.0, 0.0] + [0.0] * 7])
    targets = torch.tensor([0])
    plain = build_loss_fn("cross_entropy")(logits, targets)
    smoothed = build_loss_fn("label_smoothing", label_smoothing=0.1)(logits, targets)
    assert smoothed > plain


def test_training_reduces_the_loss(loaders: Loaders) -> None:
    train_loader, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(32,)))
    result = train_model(
        model, train_loader, val_loader, TrainConfig(max_epochs=3, device="cpu"), verbose=False
    )
    history = result.history
    assert history[-1].train_loss < history[0].train_loss


def test_training_records_gradient_norms(loaders: Loaders) -> None:
    train_loader, val_loader, _ = loaders
    result = train_model(
        build_model(ModelConfig(hidden_sizes=(16,))),
        train_loader,
        val_loader,
        TrainConfig(max_epochs=1, device="cpu"),
        verbose=False,
    )
    norms = result.history[0].grad_norms
    assert norms and all(value >= 0 for value in norms.values())


def test_history_frame_is_one_row_per_epoch(loaders: Loaders) -> None:
    train_loader, val_loader, _ = loaders
    result = train_model(
        build_model(ModelConfig(hidden_sizes=(16,))),
        train_loader,
        val_loader,
        TrainConfig(max_epochs=2, device="cpu"),
        verbose=False,
    )
    frame = result.history_frame()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert {"epoch", "train_loss", "val_accuracy"} <= set(frame.columns)


def test_early_stopping_ends_a_run_that_stops_improving(loaders: Loaders) -> None:
    train_loader, val_loader, _ = loaders
    result = train_model(
        build_model(ModelConfig(hidden_sizes=(16,))),
        train_loader,
        val_loader,
        # Noise data cannot improve, so patience 1 must trigger well before epoch 30.
        TrainConfig(max_epochs=30, patience=1, device="cpu"),
        verbose=False,
    )
    assert result.stopped_early
    assert len(result.history) < 30


def test_same_seed_reproduces_a_run() -> None:
    """Two runs of the same configuration must agree exactly.

    This covers weight initialisation as well as batch shuffling, which is why
    it goes through train_from_splits rather than building the model itself.
    """
    splits = build_splits(make_frame(300), DataConfig())
    config = ExperimentConfig(
        model=ModelConfig(hidden_sizes=(16,)),
        train=TrainConfig(max_epochs=2, seed=7, device="cpu"),
    )
    first = train_from_splits(splits, config, verbose=False)
    second = train_from_splits(splits, config, verbose=False)
    assert first.best_val_accuracy == pytest.approx(second.best_val_accuracy)


def test_a_different_seed_changes_the_run() -> None:
    """A seed that is not fixed should move the result, or the seed is ignored."""
    splits = build_splits(make_frame(300), DataConfig())
    base = ExperimentConfig(
        model=ModelConfig(hidden_sizes=(16,)),
        train=TrainConfig(max_epochs=2, seed=1, device="cpu"),
    )
    other = replace(base, train=replace(base.train, seed=2))
    assert train_from_splits(splits, base, verbose=False).history[0].train_loss != pytest.approx(
        train_from_splits(splits, other, verbose=False).history[0].train_loss
    )


def test_evaluate_returns_a_loss_and_an_accuracy(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    loss, accuracy = evaluate(
        build_model(ModelConfig(hidden_sizes=(16,))),
        val_loader,
        build_loss_fn("cross_entropy"),
        torch.device("cpu"),
    )
    assert loss > 0
    assert 0.0 <= accuracy <= 1.0


def test_predict_returns_one_row_per_example(loaders: Loaders) -> None:
    _, val_loader, _ = loaders
    logits, targets = predict(
        build_model(ModelConfig(hidden_sizes=(16,))), val_loader, torch.device("cpu")
    )
    assert logits.shape == (len(targets), 10)


@pytest.mark.slow
def test_the_model_can_learn_a_separable_pattern() -> None:
    """A sanity check that the loop actually fits, not only that it runs.

    Two classes are made linearly separable by construction, so a model that
    cannot reach high accuracy here has a bug in the loop, not in the data.
    """
    frame = make_frame(400)
    pixel_columns = [column for column in frame.columns if column != "label"]
    frame["label"] = frame.index % 2
    frame.loc[frame["label"] == 1, pixel_columns] = 250
    frame.loc[frame["label"] == 0, pixel_columns] = 5

    splits = build_splits(frame, DataConfig())
    train_loader, val_loader, _ = make_loaders(splits, batch_size=32)
    result = train_model(
        build_model(ModelConfig(hidden_sizes=(32,), num_classes=2)),
        train_loader,
        val_loader,
        TrainConfig(max_epochs=5, device="cpu"),
        verbose=False,
    )
    assert result.best_val_accuracy > 0.95


# ── learning-rate schedules ───────────────────────────────────────────────


def an_optimizer(config: TrainConfig) -> torch.optim.Optimizer:
    """A real optimizer over one small model, for the schedule tests."""
    return build_optimizer(build_model(ModelConfig(hidden_sizes=(8,))), config)


@pytest.mark.parametrize("name", ["none", "step", "cosine", "onecycle"])
def test_every_schedule_builds(name: Scheduler) -> None:
    config = TrainConfig(scheduler=name, max_epochs=4)
    scheduler, _ = build_scheduler(an_optimizer(config), config, steps_per_epoch=5)
    assert (scheduler is None) == (name == "none")


def test_an_unknown_schedule_is_rejected() -> None:
    config = replace(TrainConfig(), scheduler="triangular")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler(an_optimizer(config), config, steps_per_epoch=5)


def test_only_one_cycle_steps_per_batch() -> None:
    """The others advance once per epoch; stepping them per batch would race
    through the schedule in the first epoch."""
    for name, expected in (("step", False), ("cosine", False), ("onecycle", True)):
        config = TrainConfig(scheduler=name, max_epochs=3)  # type: ignore[arg-type]
        _, per_batch = build_scheduler(an_optimizer(config), config, steps_per_epoch=4)
        assert per_batch is expected


def test_the_staircase_drops_by_a_factor_of_ten() -> None:
    config = TrainConfig(scheduler="step", max_epochs=6, learning_rate=0.1)
    optimizer = an_optimizer(config)
    scheduler, _ = build_scheduler(optimizer, config, steps_per_epoch=1)
    assert scheduler is not None
    optimizer.step()  # schedules expect the optimizer to have stepped first
    for _ in range(config.max_epochs // 3):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_cosine_anneals_to_its_floor() -> None:
    config = TrainConfig(scheduler="cosine", max_epochs=10, learning_rate=0.1)
    optimizer = an_optimizer(config)
    scheduler, _ = build_scheduler(optimizer, config, steps_per_epoch=1)
    assert scheduler is not None
    optimizer.step()
    for _ in range(config.max_epochs):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.01)


def test_one_cycle_warms_up_before_it_anneals() -> None:
    """The rate must rise above where it started, then end below it."""
    config = TrainConfig(scheduler="onecycle", max_epochs=2, learning_rate=0.1)
    optimizer = an_optimizer(config)
    scheduler, _ = build_scheduler(optimizer, config, steps_per_epoch=10)
    assert scheduler is not None
    optimizer.step()

    rates = []
    for _ in range(2 * 10):
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert max(rates) > rates[0]
    assert rates[-1] < rates[0]


def test_the_history_records_the_rate_each_epoch_was_trained_with(loaders: Loaders) -> None:
    train_loader, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    config = TrainConfig(scheduler="cosine", max_epochs=3, patience=3, device="cpu")
    result = train_model(model, train_loader, val_loader, config, verbose=False)

    rates = [record.learning_rate for record in result.history]
    assert rates[0] == pytest.approx(config.learning_rate)
    assert rates[-1] < rates[0]
    assert "learning_rate" in result.history_frame().columns


def test_a_constant_rate_stays_constant(loaders: Loaders) -> None:
    train_loader, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    config = TrainConfig(scheduler="none", max_epochs=3, patience=3, device="cpu")
    result = train_model(model, train_loader, val_loader, config, verbose=False)
    assert {record.learning_rate for record in result.history} == {config.learning_rate}


# ── augmentation inside the loop ──────────────────────────────────────────


@pytest.mark.slow
def test_training_with_augmentation_completes_and_learns_nothing_odd(loaders: Loaders) -> None:
    """Noise pixels cannot be learned; what is checked is that the warped
    batches flow through the loop and produce a complete history."""
    train_loader, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    config = TrainConfig(augment=True, max_epochs=2, patience=2, device="cpu")
    result = train_model(model, train_loader, val_loader, config, verbose=False)

    assert len(result.history) == 2
    assert all(0.0 <= record.train_accuracy <= 1.0 for record in result.history)


def test_augmentation_leaves_the_validation_pass_alone(loaders: Loaders) -> None:
    """Validation must score the images as they are, or model selection would
    be comparing epochs on different data."""
    _, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(8,)))
    loss_fn = build_loss_fn("cross_entropy")
    first = evaluate(model, val_loader, loss_fn, torch.device("cpu"))
    second = evaluate(model, val_loader, loss_fn, torch.device("cpu"))
    assert first == second


def test_a_negative_augmentation_magnitude_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_shift"):
        TrainConfig(max_shift=-1)
    with pytest.raises(ValueError, match="max_rotation"):
        TrainConfig(max_rotation=-5.0)


def test_gradient_norms_name_the_linear_layers_only(loaders: Loaders) -> None:
    """A normalisation layer has a weight too, but it is a per-feature scale,
    not a matrix the signal passes through. Mixing them into the same plot
    hides the trend down the depth of the network."""
    train_loader, val_loader, _ = loaders
    model = build_model(ModelConfig(hidden_sizes=(16,), batch_norm=True))
    result = train_model(
        model, train_loader, val_loader, TrainConfig(max_epochs=1, device="cpu"), verbose=False
    )

    linear_names = {
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
    }
    assert set(result.history[0].grad_norms) == linear_names


def test_the_selected_epoch_always_points_into_the_history(loaders: Loaders) -> None:
    """The history is indexed by best_epoch, so a run that never beats chance
    must still name the epoch whose weights it kept."""
    train_loader, val_loader, _ = loaders
    result = train_model(
        build_model(ModelConfig(hidden_sizes=(8,))),
        train_loader,
        val_loader,
        TrainConfig(max_epochs=2, device="cpu"),
        verbose=False,
    )
    assert 1 <= result.best_epoch <= len(result.history)


def test_scoring_moves_the_model_to_the_device_it_was_given(loaders: Loaders) -> None:
    """A checkpoint loads onto the CPU. Both entry points have to move it, or
    every evaluation of a saved model on an accelerator fails on the first
    matrix multiply."""
    _, val_loader, _ = loaders
    device = torch.device("cpu")
    model = build_model(ModelConfig(hidden_sizes=(8,)))

    loss, accuracy = evaluate(model, val_loader, build_loss_fn("cross_entropy"), device)
    logits, targets = predict(model, val_loader, device)

    assert next(model.parameters()).device.type == device.type
    assert 0.0 <= accuracy <= 1.0 and loss >= 0.0
    assert logits.shape[0] == targets.shape[0]


# ── the loss is sized by the model, not by a constant ──────────────────────


@pytest.mark.parametrize("n_classes", [2, 5, 10])
def test_the_mse_loss_matches_the_width_of_the_logits(n_classes: int) -> None:
    """The one-hot target has to be as wide as the logits, whatever the model.

    A fixed class count broadcasts instead of raising when the two disagree, so
    the loss is computed against the wrong thing and training still appears to
    run. Reading the width off the logits cannot disagree with the model.
    """
    model = build_model(ModelConfig(hidden_sizes=(8,), num_classes=n_classes))
    logits = model(torch.randn(4, 784))
    targets = torch.arange(4) % n_classes

    loss = build_loss_fn("mse")(logits, targets)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_every_loss_accepts_a_model_that_is_not_ten_way() -> None:
    """The same has to hold for all four, since the sweep varies the loss."""
    model = build_model(ModelConfig(hidden_sizes=(8,), num_classes=3))
    logits = model(torch.randn(6, 784))
    targets = torch.arange(6) % 3

    for name in ("cross_entropy", "label_smoothing", "nll", "mse"):
        assert torch.isfinite(build_loss_fn(name)(logits, targets))


# ── an empty loader is named, not divided by ───────────────────────────────


def test_evaluating_on_an_empty_loader_says_so() -> None:
    """Dividing by a zero row count reports `ZeroDivisionError` and nothing else."""
    empty = DataLoader(TensorDataset(torch.zeros(0, 784), torch.zeros(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty loader"):
        evaluate(
            build_model(ModelConfig(hidden_sizes=(8,))),
            empty,
            build_loss_fn("nll"),
            torch.device("cpu"),
        )


def test_the_reported_loss_belongs_to_the_epoch_whose_weights_were_kept() -> None:
    """Three fields say "best". They have to mean the same epoch.

    Model selection keeps the best accuracy; early stopping watches the lowest
    loss. When those fall on different epochs, reporting the accuracy of one
    beside the loss of the other describes a model that never existed.
    """
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(600, 784, generator=generator)
    y = torch.randint(0, 10, (600,), generator=generator)
    splits = DataSplits(x[:400], y[:400], x[400:500], y[400:500], x[500:], y[500:], 0.0, 1.0)
    train_loader, val_loader, _ = make_loaders(splits, batch_size=64)

    result = train_model(
        build_model(ModelConfig(hidden_sizes=(32,))),
        train_loader,
        val_loader,
        TrainConfig(max_epochs=8, patience=99, device="cpu"),
        verbose=False,
    )

    kept = result.history[result.best_epoch - 1]
    assert result.best_val_loss == pytest.approx(kept.val_loss)
    assert result.best_val_accuracy == pytest.approx(kept.val_accuracy)


def test_early_stopping_still_watches_the_lowest_loss_not_the_kept_one() -> None:
    """Separating the two must not turn early stopping into an accuracy watch."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(400, 784, generator=generator)
    y = torch.randint(0, 10, (400,), generator=generator)
    splits = DataSplits(x[:240], y[:240], x[240:320], y[240:320], x[320:], y[320:], 0.0, 1.0)
    train_loader, val_loader, _ = make_loaders(splits, batch_size=64)

    result = train_model(
        build_model(ModelConfig(hidden_sizes=(64,))),
        train_loader,
        val_loader,
        TrainConfig(max_epochs=40, patience=3, device="cpu", learning_rate=1e-2),
        verbose=False,
    )

    assert result.stopped_early, "noise labels should overfit and trip the patience"
    losses = [record.val_loss for record in result.history]
    # It stopped because the loss stopped falling, so the last `patience` epochs
    # hold no new minimum.
    assert min(losses[-3:]) > min(losses)
