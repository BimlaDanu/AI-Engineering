"""Where figures go, and in what format.

One rule, applied everywhere: **every figure this project writes is a PDF.**
Not a policy for its own sake -- three practical reasons, in order of how often
they bite.

1. A PDF opens on any machine without a browser or an image viewer that
   understands vector formats. SVG does not: it renders perfectly inside a web
   page and refuses to open as a file on a lot of desktops, which makes it the
   worst possible choice for a figure someone has to *look at* while working.
2. It is vector, so a circuit diagram or a crossover curve stays sharp at any
   zoom and in a printed report. A raster figure of a circuit is unreadable the
   moment anyone enlarges it to count the gates.
3. It embeds directly into a written report, which is where these end up.

PNG is written **only** where something needs a raster, and there are exactly two
places. The Streamlit page icon, because a browser tab cannot show a PDF. And the
ansatz circuit, because it is embedded in Markdown that renders on sites showing no
PDF inline, and that circuit is the one figure a reader needs before deciding whether
to run anything -- every number this project reports is a statement about it. Both are
written beside the PDF rather than instead of it, so the vector original is still
what a report embeds.

The style is set here rather than in each plotting call, because a report whose
figures disagree about font size reads as three reports stapled together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""Repository root, resolved from this file rather than the working directory."""

FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures"
"""Where every figure a run produces is written."""

FIGURE_FORMAT = "pdf"
"""The one format. See the module docstring for why it is not SVG."""

FIGURE_STYLE: dict[str, Any] = {
    "figure.figsize": (5.5, 3.6),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.transparent": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Keep text as text in the PDF, so a reader can select and search it, and a
    # mislabelled axis is distinguishable from a rendering artefact.
    "pdf.fonttype": 42,
}
"""House plotting style, applied by :func:`use_house_style`.

Small and deliberate. Grids are faint rather than absent because most of these
figures are read for a *value* -- where a curve crosses, what a gap is at
criticality -- and a figure read for values needs gridlines.
"""


def use_house_style() -> None:
    """Apply the house style to the current matplotlib session.

    Imported lazily so that importing this module costs nothing in a process
    that never draws anything -- which is most of them, including every test
    that only needs :data:`FIGURE_DIRECTORY`.
    """
    import matplotlib.pyplot as plt

    # rcParams is typed with a literal key union, so a plain dict of settings is
    # rejected even when every key in it is valid. The cast is the narrowest way
    # through; the keys are checked at runtime by matplotlib itself, which raises
    # on an unknown one.
    plt.rcParams.update(cast(Any, FIGURE_STYLE))


def figure_path(name: str) -> Path:
    """Where a figure of the given name belongs.

    Args:
        name: A short slug, with no extension and no directory. ``"crossover"``,
            not ``"reports/figures/crossover.pdf"``.

    Returns:
        The absolute path the figure will be written to.

    Raises:
        ValueError: If ``name`` carries a directory or an extension. Both are
            ways of quietly writing outside the figure directory or in a format
            the project has decided against, and both are easier to reject than
            to notice later.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"figure name must be a bare slug, got {name!r}")
    if Path(name).suffix:
        raise ValueError(f"figure name must carry no extension, got {name!r}")
    return FIGURE_DIRECTORY / f"{name}.{FIGURE_FORMAT}"


def save_figure(figure: Any, name: str, also_png: bool = False) -> Path:
    """Write a figure to the project's figure directory, as a PDF.

    Args:
        figure: A matplotlib ``Figure``.
        name: A bare slug; see :func:`figure_path`.
        also_png: Additionally write a raster copy beside it. Only for figures
            that have to be displayed somewhere a PDF cannot go, such as inside
            the Streamlit app.

    Returns:
        The path of the PDF, which is always the canonical copy.
    """
    destination = figure_path(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format=FIGURE_FORMAT)
    if also_png:
        figure.savefig(destination.with_suffix(".png"), format="png", dpi=200)
    return destination
