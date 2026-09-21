"""Draw the application icon, as a PDF to look at and a PNG for the browser tab.

The icon was originally an SVG. SVG is the right format for a favicon and the
wrong one for a file anyone has to open: it renders inside a browser and
refuses to open on a lot of desktops. So it is drawn here instead, and written
twice -- a PDF that opens anywhere, and the PNG that Streamlit actually serves.

The glyph is two qubit wires carrying one parameterised rotation and one CNOT.
That is the smallest picture that is unambiguously a *quantum circuit* rather
than a generic atom or wave: the control dot joined to a crossed target is the
one symbol nothing else in scientific graphics uses.

Run with ``make icon``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from src.figure_export import save_figure

ICON_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "ui" / "assets"
"""Where Streamlit looks for the page icon."""

WIRE_COLOUR = "#2B3A67"
ACCENT_COLOUR = "#5B6CFF"


def draw() -> plt.Figure:
    """Draw the icon on a square canvas with no axes and no margin."""
    figure, axes = plt.subplots(figsize=(2.0, 2.0))
    axes.set_xlim(0, 64)
    axes.set_ylim(0, 64)
    axes.set_aspect("equal")
    axes.axis("off")

    top, bottom = 42.0, 22.0
    for wire in (top, bottom):
        axes.plot(
            [4, 60],
            [wire, wire],
            color=WIRE_COLOUR,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=1,
        )

    # A parameterised rotation on the upper wire: the variational part.
    axes.add_patch(
        FancyBboxPatch(
            (14, top - 9),
            18,
            18,
            boxstyle="round,pad=0,rounding_size=3",
            facecolor="white",
            edgecolor=WIRE_COLOUR,
            linewidth=2.4,
            zorder=2,
        )
    )
    axes.text(
        23,
        top,
        "R",
        ha="center",
        va="center",
        fontsize=15,
        style="italic",
        color=WIRE_COLOUR,
        zorder=3,
    )

    # A CNOT: the entangling part, and the gate the whole depth budget is spent on.
    control_x = 46.0
    axes.plot(
        [control_x, control_x],
        [bottom, top],
        color=ACCENT_COLOUR,
        linewidth=2.4,
        solid_capstyle="round",
        zorder=2,
    )
    axes.add_patch(
        Circle((control_x, top), 3.6, facecolor=ACCENT_COLOUR, edgecolor="none", zorder=4)
    )
    # The target ring is opaque so that the wire and the vertical link stop at
    # its edge, which is how a CNOT is drawn and how it reads at 16 pixels.
    axes.add_patch(
        Circle(
            (control_x, bottom),
            7.0,
            facecolor="white",
            edgecolor=ACCENT_COLOUR,
            linewidth=2.4,
            zorder=3,
        )
    )
    axes.plot(
        [control_x - 4.6, control_x + 4.6],
        [bottom, bottom],
        color=ACCENT_COLOUR,
        linewidth=2.4,
        zorder=4,
    )
    axes.plot(
        [control_x, control_x],
        [bottom - 4.6, bottom + 4.6],
        color=ACCENT_COLOUR,
        linewidth=2.4,
        zorder=4,
    )

    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return figure


if __name__ == "__main__":
    figure = draw()
    pdf = save_figure(figure, "app-icon", also_png=True)
    ICON_DIRECTORY.mkdir(parents=True, exist_ok=True)
    icon = ICON_DIRECTORY / "quantum-circuit.png"
    figure.savefig(icon, format="png", dpi=128, transparent=True)
    plt.close(figure)
    print(f"wrote {pdf}")
    print(f"wrote {pdf.with_suffix('.png')}")
    print(f"wrote {icon}  (served by Streamlit)")
