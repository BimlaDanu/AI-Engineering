r"""Random shifts and rotations for the training batches.

Each image is warped by one affine map, a rotation followed by a translation,
with the angle and the offsets drawn per image and per batch. The warp runs on
the training device as one batched operation, not per image on the host.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IMAGE_SIZE = 28


def random_affine_matrices(
    batch_size: int,
    max_shift: int,
    max_rotation: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one rotation-plus-translation matrix per image.

    Args:
        batch_size: How many matrices to draw.
        max_shift: Largest translation in pixels, in either direction.
        max_rotation: Largest rotation in degrees, in either direction.
        generator: Optional seeded generator, so a test can repeat a draw.

    Returns:
        Float tensor of shape (batch_size, 2, 3), the form `affine_grid` expects.
        Its translation column is in normalised units, where the full width of
        the image spans 2, which is why the pixel shift is divided by half the
        image size.
    """
    angles = (torch.rand(batch_size, generator=generator) * 2 - 1) * math.radians(max_rotation)
    shifts = (torch.rand(batch_size, 2, generator=generator) * 2 - 1) * max_shift
    shifts = shifts / (IMAGE_SIZE / 2)

    cos, sin = torch.cos(angles), torch.sin(angles)
    matrices = torch.zeros(batch_size, 2, 3)
    matrices[:, 0, 0], matrices[:, 0, 1] = cos, -sin
    matrices[:, 1, 0], matrices[:, 1, 1] = sin, cos
    matrices[:, :, 2] = shifts
    return matrices


def augment_batch(
    images: torch.Tensor,
    max_shift: int = 2,
    max_rotation: float = 10.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply a random shift and rotation to every image in a batch.

    The corners that rotate into view are filled with the smallest value in the
    batch. The pixels have been standardised by this point, so the background is
    no longer zero but (0 - mean) / std, and padding with a literal zero would
    paint a grey border around every digit. Every MNIST batch contains
    background pixels, so the minimum is exactly that value.

    Args:
        images: Float tensor of shape (batch, 784) or (batch, 1, 28, 28).
        max_shift: Largest translation in pixels.
        max_rotation: Largest rotation in degrees.
        generator: Optional seeded generator.

    Returns:
        The warped batch, in the same shape and on the same device as the input.

    Raises:
        ValueError: if the batch does not hold 28x28 images.
    """
    if images.numel() != images.shape[0] * IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(
            f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} images, got shape {tuple(images.shape)}."
        )
    if max_shift == 0 and max_rotation == 0.0 or images.shape[0] == 0:
        # Nothing to warp. An empty batch is returned rather than reduced: the
        # background level is read off the batch minimum, and an empty tensor
        # has no minimum to read.
        return images

    square = images.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    background = square.amin()

    matrices = random_affine_matrices(square.shape[0], max_shift, max_rotation, generator)
    grid = F.affine_grid(
        matrices.to(square.device, square.dtype), list(square.shape), align_corners=False
    )
    warped = F.grid_sample(square - background, grid, align_corners=False, padding_mode="zeros")
    return (warped + background).view(images.shape)
