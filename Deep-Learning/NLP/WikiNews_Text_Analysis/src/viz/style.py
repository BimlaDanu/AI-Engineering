"""Shared figure styling.

One palette and one set of axis defaults for every figure. Hues are used in a
fixed order and never cycled, so a category keeps its colour across figures.
Every bar also carries its printed value, so colour never carries a number.
"""

from __future__ import annotations

from typing import Final

import matplotlib

# No display on this machine, and the figures are written to disk.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  -- must follow the backend choice

# Fixed categorical order: slot 1 goes to the largest category.
PALETTE: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# Extra slots for figures tracing more series than there are categories.
# Six is the limit; `colour_for` clamps rather than inventing a seventh.
EXTENDED_PALETTE: Final[tuple[str, ...]] = PALETTE + ("#e87ba4", "#4a3aa7")

# One hue, light to dark, for magnitude. Never a rainbow.
SEQUENTIAL_CMAP: Final[str] = "Blues"

TEXT_PRIMARY: Final[str] = "#1a1a19"
TEXT_MUTED: Final[str] = "#6b6b66"
GRID: Final[str] = "#e4e4e0"


def apply_style() -> None:
    """Set the matplotlib defaults used by every figure in the report."""
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def colour_for(index: int) -> str:
    """Return the categorical colour for a slot, without cycling past the end.

    Args:
        index: Zero-based slot number.

    Returns:
        A hex colour. Slots beyond the palette reuse the last hue.
    """
    return EXTENDED_PALETTE[min(index, len(EXTENDED_PALETTE) - 1)]


def strip_chrome(axes: plt.Axes) -> None:
    """Remove the top and right spines so the data reads before the frame.

    Args:
        axes: The axes to clean up.
    """
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
