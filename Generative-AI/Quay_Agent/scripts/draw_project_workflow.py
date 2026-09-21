r"""One picture of the whole system: every technology, and what it is wired to.

The master slide. Five bands, read top to bottom, because that is the order a
question actually travels, with the cross-cutting concerns on a strip beneath.

The layout is arguing two things, and they are the two claims the project rests on:

* **The boundary between "the model" and "the numbers" is real.** Everything above
  the green band is language; the green band is arithmetic. No arrow carries a
  number upward that was not computed there.
* **The grader is behind a wall.** Drawn below a barrier rather than wired in,
  because an agent that can read the answer key proves nothing.

Every library named is in ``pyproject.toml``. Nothing aspirational is drawn.

Sized to be read on a projector, which is why each band carries at most four chips,
one line of detail each, and one number in the label column. The first version put
six chips and three lines of small type in every band: a reference diagram nobody
could read from the back of a room.

Run with ``make figures``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.figure_export import save_figure, use_house_style

BAND_COLOUR = {
    "interface": "#2B3A67",
    "agent": "#5B6CFF",
    "retrieval": "#E0803C",
    "deterministic": "#4C956C",
    "sealed": "#B23A48",
    "crosscut": "#7C5BA6",
}
"""One colour per band. The eye should be able to group without reading."""

BANDS: tuple[tuple[str, str, str, float], ...] = (
    # key, heading, the one number that band is worth, bottom edge
    ("interface", "Interface", "7 pages, no key needed", 6.95),
    ("agent", "Agent", "17 nodes, one loop", 5.00),
    ("retrieval", "Retrieval", "126 notes · 431 chunks", 3.05),
    ("deterministic", "Deterministic core", "no model, ever", 1.10),
    ("sealed", "The grader", "two routes, no shared algebra", -1.05),
)
"""Band key, heading, headline number, and lower edge.

The number sits in the label column rather than inside a chip because it is the one
thing a reader should take from the band if they read nothing else in it.
"""

CHIPS: dict[str, tuple[tuple[str, str], ...]] = {
    # band: ((name, what it does here -- one line, always), ...)
    "interface": (
        ("Streamlit", "the app · runs with no API key"),
        ("Matplotlib", "every figure a PDF"),
        ("AppTest", "every page rendered in CI"),
    ),
    "agent": (
        ("LangGraph", "StateGraph · conditional edges"),
        ("LangChain", "tool calling · structured output"),
        ("OpenRouter", "a model chosen per node"),
        ("Memory", "JSONL · inspectable · erasable"),
    ),
    "retrieval": (
        ("Chroma", "vector index"),
        ("BM25", "lexical, over the same chunks"),
        ("RRF + MMR", "fuse k=10 · diversity λ=0.5"),
        ("arXiv API", "live, screened, labelled"),
    ),
    "deterministic": (
        ("Physics", "NumPy · SciPy · VQE · QAOA"),
        ("Hardware", "place · route · fidelity"),
        ("Shot ledger", "quantum measurements, priced first"),
        ("Classical", "the answer to beat"),
    ),
    "sealed": (
        ("Pfeuty closed form", "O(L) over momenta — no matrix"),
        ("Sparse Lanczos", "2^L matrix, other bit order"),
    ),
    "crosscut": (
        ("Observability", "LangSmith · JSON logs · model spend"),
        ("Evaluation", "3 suites → scorecard.json"),
        ("Quality gate", "pytest · ruff · mypy · CI"),
        ("Security", "injection screen · scope gate"),
    ),
}
"""What sits in each band. One line of detail, and it states a fact."""

BAND_LEFT = 0.0
BAND_RIGHT = 11.10
LABEL_RIGHT = 2.30
CHIP_LEFT = 2.55
BAND_HEIGHT = 1.15
CHIP_HEIGHT = 0.78
"""Geometry.

The label column on the left carries the heading and the number, so no chip has to
make room for either. That separation is what bought the space back.
"""


def draw_band(axes: plt.Axes, key: str, title: str, number: str, bottom: float) -> None:
    """Draw one band: its panel, its heading, and its headline number.

    Args:
        axes: The panel to draw into.
        key: Band key, used for the colour.
        title: Heading, set in the label column.
        number: The one figure worth taking from this band.
        bottom: Lower edge, in data units.
    """
    colour = BAND_COLOUR[key]
    axes.add_patch(
        FancyBboxPatch(
            (BAND_LEFT, bottom),
            BAND_RIGHT - BAND_LEFT,
            BAND_HEIGHT,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=1.0,
            edgecolor=colour,
            facecolor=colour,
            alpha=0.06,
            zorder=1,
        )
    )
    axes.plot(
        [LABEL_RIGHT, LABEL_RIGHT],
        [bottom + 0.14, bottom + BAND_HEIGHT - 0.14],
        linewidth=1.0,
        color=colour,
        alpha=0.35,
        zorder=2,
    )
    middle = bottom + BAND_HEIGHT / 2
    axes.text(
        0.18, middle + 0.17, title, fontsize=10.5, color=colour, va="center", fontweight="bold"
    )
    axes.text(0.18, middle - 0.20, number, fontsize=7.2, color=colour, va="center", alpha=0.85)


def draw_chip(
    axes: plt.Axes, x: float, y: float, width: float, name: str, does: str, colour: str
) -> None:
    """Draw one technology chip.

    Args:
        axes: The panel to draw into.
        x: Left edge.
        y: Bottom edge.
        width: How wide.
        name: The technology's name.
        does: What it does here, in one line.
        colour: The band's colour.
    """
    axes.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            CHIP_HEIGHT,
            boxstyle="round,pad=0.015,rounding_size=0.10",
            linewidth=1.2,
            edgecolor=colour,
            facecolor="white",
            zorder=3,
        )
    )
    axes.text(x + width / 2, y + 0.50, name, fontsize=8.6, color=colour, ha="center", va="center")
    axes.text(x + width / 2, y + 0.22, does, fontsize=6.1, color="0.36", ha="center", va="center")


def lay_out_chips(axes: plt.Axes, key: str, bottom: float, colour: str | None = None) -> None:
    """Place a band's chips in one row, left-aligned from the label column.

    Left-aligned rather than stretched to the right edge: a band of two chips
    stretched across the full width reads as two chips with something missing
    between them.

    Args:
        axes: The panel to draw into.
        key: Which band.
        bottom: The band's lower edge.
        colour: Override the band colour, for the cross-cutting strip.
    """
    chips = CHIPS[key]
    span = BAND_RIGHT - CHIP_LEFT - 0.20
    width = min(2.12, (span - 0.14 * (len(chips) - 1)) / len(chips))
    y = bottom + (BAND_HEIGHT - CHIP_HEIGHT) / 2
    for index, (name, does) in enumerate(chips):
        draw_chip(
            axes,
            CHIP_LEFT + index * (width + 0.14),
            y,
            width,
            name,
            does,
            colour or BAND_COLOUR[key],
        )


def draw_flow(axes: plt.Axes, top: float, bottom: float, x: float, label: str) -> None:
    """Draw one arrow between bands, with what travels along it written beside.

    Args:
        axes: The panel to draw into.
        top: Where it starts.
        bottom: Where it ends.
        x: Horizontal position.
        label: What moves along this edge. Empty for the two long ones, whose
            labels need more room than an arrow's midpoint offers.
    """
    axes.add_patch(
        FancyArrowPatch(
            (x, top),
            (x, bottom),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.6,
            color="0.42",
            zorder=5,
        )
    )
    if label:
        axes.text(
            x + 0.16, (top + bottom) / 2, label, fontsize=6.9, color="0.36", va="center", ha="left"
        )


def draw() -> plt.Figure:
    """Build the figure.

    Returns:
        The finished figure. The caller closes it.
    """
    use_house_style()
    figure, axes = plt.subplots(figsize=(12.8, 8.2))

    for key, title, number, bottom in BANDS:
        draw_band(axes, key, title, number, bottom)
        lay_out_chips(axes, key, bottom)

    draw_flow(axes, 8.92, 8.20, 3.4, "a question in ordinary language")
    draw_flow(axes, 6.95, 6.23, 3.4, "typed request")
    draw_flow(axes, 5.00, 4.28, 3.4, "expanded queries")
    draw_flow(axes, 4.20, 4.92, 7.7, "passages, graded, with citations")

    # The two long ones run outside the bands on the right, where nothing else is.
    draw_flow(axes, 5.00, 2.33, 11.45, "")
    axes.text(
        11.60,
        3.9,
        "a specification:\nL, J, h,\nboundary, depth",
        fontsize=6.9,
        color="0.36",
        va="center",
    )
    draw_flow(axes, 2.25, 4.92, 13.05, "")
    axes.text(
        13.20, 3.9, "numbers —\nnever from\nthe model", fontsize=6.9, color="0.36", va="center"
    )

    # The wall, as a barrier rather than an edge: what matters is that nothing
    # crosses it. The only route to the grader is the evaluation harness.
    axes.plot(
        [BAND_LEFT, BAND_RIGHT],
        [0.32, 0.32],
        linestyle=(0, (7, 5)),
        linewidth=2.2,
        color=BAND_COLOUR["sealed"],
        zorder=6,
    )
    axes.text(
        BAND_RIGHT,
        0.44,
        "import wall — tests/test_architecture.py walks the import graph; "
        "only the eval harness reads below it",
        fontsize=7.0,
        color=BAND_COLOUR["sealed"],
        ha="right",
        va="bottom",
    )

    # Cross-cutting strip along the bottom rather than down the side: as a column it
    # cost a fifth of the width to say four things that apply everywhere equally.
    colour = BAND_COLOUR["crosscut"]
    axes.add_patch(
        FancyBboxPatch(
            (BAND_LEFT, -2.95),
            BAND_RIGHT - BAND_LEFT,
            BAND_HEIGHT,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=1.0,
            edgecolor=colour,
            facecolor=colour,
            alpha=0.06,
            zorder=1,
        )
    )
    axes.plot(
        [LABEL_RIGHT, LABEL_RIGHT],
        [-2.81, -1.94],
        linewidth=1.0,
        color=colour,
        alpha=0.35,
        zorder=2,
    )
    axes.text(
        0.18, -2.16, "Across all of it", fontsize=10.5, color=colour, va="center", fontweight="bold"
    )
    lay_out_chips(axes, "crosscut", -2.95, colour)

    axes.text(
        BAND_LEFT,
        9.85,
        "Quay — how a question becomes a checkable answer",
        fontsize=15.5,
        color="#1F2937",
        va="center",
    )
    axes.text(
        BAND_LEFT,
        9.35,
        "Read top to bottom: that is the order a question travels. Two budgets are tracked "
        "and they are not the same: quantum measurements (green) and money spent on model "
        "calls (purple).",
        fontsize=8.2,
        color="0.44",
        va="center",
    )

    axes.set_xlim(-0.2, 14.4)
    axes.set_ylim(-3.15, 10.15)
    axes.set_axis_off()
    figure.tight_layout()
    return figure


def main() -> None:
    """Draw the workflow and write it, PDF plus a PNG for slides."""
    figure = draw()
    written = save_figure(figure, "project-workflow", also_png=True)
    plt.close(figure)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
