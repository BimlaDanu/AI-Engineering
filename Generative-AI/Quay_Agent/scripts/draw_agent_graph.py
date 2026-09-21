r"""Draw the campaign graph: every node, and the edges that choose between them.

The README carries this as a mermaid diagram, which renders on GitHub and nowhere
else -- not in a slide, not in a printed report, not in a PDF. This draws the same
graph as a figure so it can be shown where mermaid cannot go.

The node list is read from :func:`~src.agent.graph.pipeline_nodes` rather than
written out here, so a node added to the graph and not to this layout fails the
build instead of quietly going missing. That check is the reason this is a script
rather than a drawing: the mermaid block in the README lost ``converge`` for exactly
as long as nobody counted the boxes.

Three things the layout is trying to make visible:

* ``baseline`` hangs off the fan-out and nothing routes around it, because a quantum
  number with no classical number beside it is the characteristic failure here.
* ``plan -> solve -> analyse -> plan`` is a loop, drawn as one, because the back edge
  is the agentic part of the agent.
* ``converge`` is a sibling of that loop rather than a stage in it, and rejoins
  wherever the reading of the question sends it.

Run with ``make figures``.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.agent.graph import pipeline_nodes
from src.figure_export import save_figure, use_house_style

ROLE_COLOUR = {
    "spine": "#2B3A67",
    "loop": "#5B6CFF",
    "prose": "#E0803C",
    "close": "#4C956C",
}
"""One colour per role. Held constant so the eye groups by job, not by position."""

TERMINAL_COLOUR = "#6B7280"
REFUSAL_COLOUR = "#B23A48"

BOX_WIDTH = 2.05
BOX_HEIGHT = 0.80
"""Node size in data units. Arrows are clipped to this rectangle, so it is geometry
rather than decoration -- a box resized here moves every arrowhead with it."""

LAYOUT: dict[str, tuple[float, float, str, str]] = {
    # node: (x, y, role, one-line caption)
    "recall": (0.0, 8.0, "spine", "what this conversation\nalready established"),
    "screen": (0.0, 6.6, "spine", "injection and scope"),
    "interpret": (0.0, 5.2, "spine", "feasibility · explain\n· implement"),
    "formalise": (0.0, 3.8, "spine", "read the lattice,\nJ, h, g, boundary"),
    "retrieve": (0.0, 2.4, "spine", "hybrid search, graded,\nre-queried once"),
    "baseline": (-4.2, 0.4, "loop", "the classical\nanswer to beat"),
    "plan": (-1.6, 0.4, "loop", "propose a depth,\nprice its shots"),
    "solve": (-1.6, -1.3, "loop", "run it, or refuse\non arithmetic"),
    "analyse": (-1.6, -3.0, "loop", "did it improve?\nis it a plateau?"),
    "skeptic": (-1.6, -4.7, "loop", "argue against\nthe verdict"),
    "converge": (1.2, 0.4, "prose", "race the three methods,\nkeep the curves"),
    "consult": (4.0, 0.4, "prose", "the model\nchooses tools"),
    "explain": (2.9, -1.6, "prose", "prose from\nthe corpus"),
    "implement": (5.3, -1.6, "prose", "the algebra,\nthen the code"),
    "scribe": (0.6, -6.4, "close", "write the report"),
    "suggest": (0.6, -7.8, "close", "what to ask next"),
    "remember": (0.6, -9.2, "close", "one line, inspectable\nand erasable"),
}
"""Where each node sits, and what it does in one line.

Hand-placed rather than solved for. A spring layout puts the loop somewhere
different on every run, and a figure whose shape moves between builds cannot be
referred to in prose that says "on the left".
"""

EDGES: tuple[tuple[str, str], ...] = (
    ("recall", "screen"),
    ("screen", "interpret"),
    ("interpret", "formalise"),
    ("formalise", "retrieve"),
    ("retrieve", "baseline"),
    ("retrieve", "plan"),
    ("retrieve", "converge"),
    ("retrieve", "consult"),
    ("plan", "solve"),
    ("solve", "analyse"),
    ("analyse", "skeptic"),
    ("baseline", "skeptic"),
    ("consult", "explain"),
    ("consult", "implement"),
    ("converge", "consult"),
    ("skeptic", "scribe"),
    ("explain", "scribe"),
    ("implement", "scribe"),
    ("scribe", "suggest"),
    ("suggest", "remember"),
)
"""The edges drawn plain. The three that are decisions are drawn separately."""


def clip_to_box(
    start: tuple[float, float], end: tuple[float, float], pad: float = 0.06
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Trim a centre-to-centre segment back to the two box edges it runs between.

    Matplotlib's ``shrinkA``/``shrinkB`` are in points, so a single value lands
    inside the box on a short vertical edge and short of it on a long diagonal.
    Solving for the rectangle boundary instead puts every arrowhead on the border
    whatever the angle.

    Args:
        start: Centre of the source box.
        end: Centre of the target box.
        pad: Extra clearance beyond the border, in data units.

    Returns:
        The trimmed start and end points.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    if dx == 0.0 and dy == 0.0:
        return start, end
    half_width, half_height = BOX_WIDTH / 2.0 + pad, BOX_HEIGHT / 2.0 + pad
    scale = min(
        half_width / abs(dx) if dx else math.inf,
        half_height / abs(dy) if dy else math.inf,
    )
    offset = (scale * dx, scale * dy)
    return (
        (start[0] + offset[0], start[1] + offset[1]),
        (end[0] - offset[0], end[1] - offset[1]),
    )


def draw_node(axes: plt.Axes, name: str) -> None:
    """Draw one node: its box, its name, and the line saying what it does.

    Args:
        axes: The panel to draw into.
        name: The node, which must appear in :data:`LAYOUT`.
    """
    x, y, role, caption = LAYOUT[name]
    colour = ROLE_COLOUR[role]
    axes.add_patch(
        FancyBboxPatch(
            (x - BOX_WIDTH / 2, y - BOX_HEIGHT / 2),
            BOX_WIDTH,
            BOX_HEIGHT,
            boxstyle="round,pad=0.01,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=colour,
            facecolor=colour,
            alpha=0.14,
            zorder=3,
        )
    )
    axes.text(x, y + 0.17, name, ha="center", va="center", fontsize=9.5, color=colour, zorder=4)
    axes.text(x, y - 0.19, caption, ha="center", va="center", fontsize=6.2, color="0.30", zorder=4)


def draw_edge(
    axes: plt.Axes,
    source: str,
    target: str,
    curve: float = 0.0,
    colour: str = "0.45",
    dashed: bool = False,
    width: float = 0.9,
) -> None:
    """Draw one arrow between two node boxes, clipped so it touches neither.

    Args:
        axes: The panel to draw into.
        source: Node the arrow leaves.
        target: Node the arrow enters.
        curve: Bow of the arc. Zero is straight.
        colour: Line colour.
        dashed: Whether to draw it dashed, for the edges that are decisions.
        width: Line width.
    """
    start = (LAYOUT[source][0], LAYOUT[source][1])
    end = (LAYOUT[target][0], LAYOUT[target][1])
    trimmed_start, trimmed_end = clip_to_box(start, end)
    axes.add_patch(
        FancyArrowPatch(
            trimmed_start,
            trimmed_end,
            connectionstyle=f"arc3,rad={curve}",
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=colour,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )


def check_every_node_is_drawn() -> None:
    """Fail loudly if the graph and this layout have drifted apart.

    Raises:
        SystemExit: If a node exists in one and not the other. A figure that
            silently omits a node is worse than no figure: it is read as a
            complete picture.
    """
    graph, drawn = set(pipeline_nodes()), set(LAYOUT)
    if graph != drawn:
        raise SystemExit(
            "the layout and the graph disagree -- in the graph but not drawn: "
            f"{', '.join(sorted(graph - drawn)) or 'none'}; drawn but not in the graph: "
            f"{', '.join(sorted(drawn - graph)) or 'none'}"
        )


def draw_terminal(axes: plt.Axes, point: tuple[float, float], label: str, into: str) -> None:
    """Draw the question mark going in, or the answer coming out.

    Args:
        axes: The panel to draw into.
        point: Where the marker sits.
        label: What to call it.
        into: The node it connects to.
    """
    axes.scatter(*point, s=46, marker="o", color=TERMINAL_COLOUR, zorder=4)
    axes.text(point[0] + 0.32, point[1], label, fontsize=8.5, color=TERMINAL_COLOUR, va="center")
    node = (LAYOUT[into][0], LAYOUT[into][1])
    downwards = point[1] > node[1]
    edge = (node[0], node[1] + (BOX_HEIGHT / 2 + 0.06) * (1 if downwards else -1))
    axes.add_patch(
        FancyArrowPatch(
            point if downwards else edge,
            edge if downwards else point,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="0.45",
            shrinkA=6 if downwards else 0,
            shrinkB=0 if downwards else 6,
            zorder=2,
        )
    )


def draw() -> plt.Figure:
    """Build the figure.

    Separate from :func:`main` so the presentation exporter can import the drawing
    rather than copy it -- one graph, drawn once, wherever it is shown.

    Returns:
        The finished figure. The caller closes it.
    """
    check_every_node_is_drawn()
    use_house_style()
    figure, axes = plt.subplots(figsize=(10.0, 9.4))

    for source, target in EDGES:
        draw_edge(axes, source, target)

    # The three edges that are decisions rather than sequence, drawn apart from the
    # rest so the eye finds them: the loop that makes this an agent, the refusal that
    # costs one hop, and the race rejoining whichever branch was asked for.
    draw_edge(axes, "analyse", "plan", curve=0.62, colour=ROLE_COLOUR["loop"], width=1.6)
    axes.text(
        -0.05,
        -1.35,
        "climb the\ndepth ladder",
        fontsize=6.4,
        color=ROLE_COLOUR["loop"],
        ha="center",
        va="center",
    )
    # Bowed hard left, right around the outside of the feasibility branch. Routed
    # through the middle it crossed plan, solve and analyse, which made the one edge
    # that skips the whole graph look like part of it.
    draw_edge(axes, "screen", "scribe", curve=0.49, colour=REFUSAL_COLOUR, dashed=True)
    axes.text(
        -5.55,
        -1.1,
        "out of scope —\nrefused\nin one hop",
        fontsize=6.4,
        color=REFUSAL_COLOUR,
        ha="left",
    )
    draw_edge(axes, "converge", "plan", curve=0.42, colour=ROLE_COLOUR["prose"], dashed=True)
    axes.text(
        1.2,
        -0.62,
        "the race rejoins whichever\nbranch was asked for",
        fontsize=6.4,
        color=ROLE_COLOUR["prose"],
        ha="center",
    )

    for name in LAYOUT:
        draw_node(axes, name)

    draw_terminal(axes, (0.0, 9.3), "question", "recall")
    draw_terminal(axes, (0.6, -10.4), "answer", "remember")

    axes.text(
        -6.05,
        9.3,
        f"{len(LAYOUT)} nodes. Which one runs next is a conditional edge\n"
        "on typed state — never a language model's decision.",
        fontsize=8.2,
        color="0.30",
        va="center",
    )
    # Both ends run on every question, so both said "every question" and the legend
    # distinguished nothing. What separates them is *when*, which is also what the
    # reader is trying to work out from the picture.
    for index, (role, label) in enumerate(
        (
            ("spine", "every question — before the fork"),
            ("loop", "feasibility branch"),
            ("prose", "prose branch"),
            ("close", "every question — after it"),
        )
    ):
        y = -8.6 - 0.5 * index
        axes.add_patch(
            FancyBboxPatch(
                (-6.05, y - 0.14),
                0.32,
                0.28,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                linewidth=1.0,
                edgecolor=ROLE_COLOUR[role],
                facecolor=ROLE_COLOUR[role],
                alpha=0.35,
            )
        )
        axes.text(-5.60, y, label, fontsize=6.8, color="0.30", va="center")

    axes.set_xlim(-6.25, 6.7)
    axes.set_ylim(-10.9, 9.9)
    axes.set_axis_off()
    figure.tight_layout()
    return figure


def main() -> None:
    """Draw the graph and write it as a PDF, with a PNG beside it for slides."""
    figure = draw()
    written = save_figure(figure, "agent-graph", also_png=True)
    plt.close(figure)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
