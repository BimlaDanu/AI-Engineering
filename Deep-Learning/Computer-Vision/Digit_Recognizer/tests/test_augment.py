"""Random shifts and rotations applied to the training batches."""

from __future__ import annotations

import pytest
import torch

from src.augment import IMAGE_SIZE, augment_batch, random_affine_matrices


def a_batch(n_rows: int = 8, flat: bool = True) -> torch.Tensor:
    """A batch of digit-like images: a bright square on a dark background."""
    images = torch.zeros(n_rows, 1, IMAGE_SIZE, IMAGE_SIZE)
    images[:, :, 10:18, 10:18] = 1.0
    return images.view(n_rows, -1) if flat else images


def test_the_shape_and_device_survive_the_warp() -> None:
    batch = a_batch()
    warped = augment_batch(batch, max_shift=2, max_rotation=10.0)
    assert warped.shape == batch.shape
    assert warped.device == batch.device
    assert warped.dtype == batch.dtype


def test_a_square_batch_is_accepted_too() -> None:
    batch = a_batch(flat=False)
    assert augment_batch(batch).shape == batch.shape


def test_zero_magnitude_is_the_identity() -> None:
    """With nothing to shift or rotate the batch must come back untouched."""
    batch = a_batch()
    assert torch.equal(augment_batch(batch, max_shift=0, max_rotation=0.0), batch)


def test_the_warp_actually_moves_the_image() -> None:
    batch = a_batch()
    warped = augment_batch(batch, max_shift=3, max_rotation=15.0)
    assert not torch.allclose(warped, batch)


def test_the_total_brightness_is_roughly_preserved() -> None:
    """A shift moves ink around; it must not create or destroy much of it."""
    batch = a_batch()
    warped = augment_batch(batch, max_shift=2, max_rotation=8.0)
    assert warped.sum() == pytest.approx(float(batch.sum()), rel=0.15)


def test_the_background_is_kept_not_replaced_by_zero() -> None:
    """Standardised pixels have a negative background; padding must match it."""
    batch = a_batch() - 5.0  # background now -5, ink -4
    warped = augment_batch(batch, max_shift=4, max_rotation=0.0)
    assert warped.min() == pytest.approx(-5.0, abs=1e-4)


def test_a_seeded_generator_repeats_the_same_warp() -> None:
    batch = a_batch()
    first = augment_batch(batch, generator=torch.Generator().manual_seed(7))
    second = augment_batch(batch, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_different_seeds_give_different_warps() -> None:
    batch = a_batch()
    first = augment_batch(batch, generator=torch.Generator().manual_seed(1))
    second = augment_batch(batch, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(first, second)


def test_every_image_in_the_batch_gets_its_own_warp() -> None:
    """One shared warp per batch would teach the model much less."""
    batch = a_batch(n_rows=16)
    warped = augment_batch(batch, max_shift=3, max_rotation=20.0)
    assert not torch.allclose(warped[0], warped[1])


def test_the_affine_matrices_have_the_shape_affine_grid_expects() -> None:
    matrices = random_affine_matrices(5, max_shift=2, max_rotation=10.0)
    assert matrices.shape == (5, 2, 3)


def test_the_rotation_block_stays_a_rotation() -> None:
    """Its determinant is 1, so the warp rotates without scaling or shearing."""
    matrices = random_affine_matrices(6, max_shift=2, max_rotation=30.0)
    determinants = torch.linalg.det(matrices[:, :, :2])
    assert torch.allclose(determinants, torch.ones(6), atol=1e-5)


def test_the_translation_stays_within_the_requested_bound() -> None:
    matrices = random_affine_matrices(64, max_shift=2, max_rotation=0.0)
    assert matrices[:, :, 2].abs().max() <= 2 / (IMAGE_SIZE / 2) + 1e-6


def test_a_batch_that_is_not_made_of_images_is_rejected() -> None:
    with pytest.raises(ValueError, match="images"):
        augment_batch(torch.zeros(4, 100))


def test_an_empty_batch_comes_back_unchanged() -> None:
    """The background level is the batch minimum, which an empty batch has not got."""
    empty = torch.zeros(0, 784)
    assert augment_batch(empty, max_shift=2, max_rotation=10.0).shape == empty.shape
