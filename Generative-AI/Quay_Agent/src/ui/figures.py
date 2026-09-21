r"""Every figure the interface draws, built with no Streamlit anywhere in the file.

Kept apart from :mod:`src.ui.panels` deliberately. A figure is a pure function of
numbers -- arrays in, a ``Figure`` out -- so it can be called from a test, from a
script that writes a PDF into ``reports/figures/``, and from a page, and the three
cannot drift apart. A panel is the opposite: it needs a running session before it
means anything, and it can only be exercised through ``AppTest``. Putting both in
one module makes the cheap half as expensive to test as the dear half.

Why draw at all, when Streamlit has charts? ``st.line_chart`` is quick and it is
mute: it cannot mark where the current setting sits, cannot annotate the one number
a reader came for, and cannot say which of two curves is the answer and which is the
attempt. Every figure here does at least one of those, which is the whole reason it
is a figure and not a chart -- a curve nobody can read a value off is decoration.

The house style comes from :mod:`src.figure_export`, the same one that governs the
PDFs written to ``reports/figures/``, so a figure looked at on screen and the same
figure pasted into a report are the same figure.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import wraps
from itertools import pairwise
from typing import Any, ParamSpec

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from src.figure_export import use_house_style
from src.physics.model import MAX_SITES_STATEVECTOR, TFIMSpec
from src.physics.quantum.ansatz import AnsatzFamily, Initialisation
from src.physics.quantumness import Configuration, Superposition

FloatArray = NDArray[np.float64]

DRAWING = threading.RLock()
"""Held while any figure here is built or rasterised.

Matplotlib parses ``$...$`` with one parser object shared by the whole process, and
that object keeps parse state on itself. Streamlit runs each script in its own
thread, so two pages drawing at once step on that state and one of them dies with an
empty ``ParseException`` from inside the maths parser -- a crash with nothing in it
pointing at this project. Serialising the drawing is the whole fix; figures are
milliseconds and are not the thing worth parallelising.
"""

_P = ParamSpec("_P")


def serialised(draw: Callable[_P, Figure]) -> Callable[_P, Figure]:
    """Make one figure builder safe to call from more than one thread.

    Args:
        draw: A builder returning a finished figure.

    Returns:
        The same builder, holding :data:`DRAWING` while it runs.
    """

    @wraps(draw)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> Figure:
        """Build the figure with the drawing lock held."""
        with DRAWING:
            return draw(*args, **kwargs)

    return guarded


ACCENT = "#4c6ef5"
"""Blue. The measured thing: the curve that was computed, a spin pointing up."""

WARM = "#e8590c"
"""Orange. The thing to look at: a marker, a critical line, a spin pointing down."""

MUTED = "#888888"
"""Grey. Everything structural -- axes, ticks, the reference a curve is chasing."""

ORDERED_FILL = "#4c6ef5"
DISORDERED_FILL = "#2f9e44"
"""One colour per phase of the chain, used only in the phase diagram.

Blue for the ordered side and green for the disordered one. Deliberately not the
red-amber-green of a verdict: which phase a chain is in is not good news or bad
news, and a reader who has just left the Chat page should not read the diagram as
a judgement.
"""

MAX_DRAWN_MAGNETS = 14
"""How many magnets the opening cartoon draws before it truncates the row.

Past this the arrows crowd into each other and the drawing stops being readable at
the size it is shown, so the row is cut and the true count is written beside it.
The knob goes to a good deal more than fourteen, and a cartoon that silently drew
the wrong number of magnets would be worse than one that admits to a truncation.
"""

ARROW_LENGTH = 0.62
"""How tall a spin arrow is drawn, in row units. Under one, so rows do not touch."""

_STYLED = False


def _styled_figure(width: float, height: float) -> Figure:
    """Make a transparent figure with the project's plotting style applied.

    Transparent because the page behind it may be light or dark, and a figure with
    a baked-in white panel is a white rectangle on a dark page -- the single most
    common way a chart in a Streamlit application looks broken.

    Args:
        width: Width in inches.
        height: Height in inches.

    Returns:
        An empty figure, styled and ready to draw on.
    """
    global _STYLED
    if not _STYLED:
        use_house_style()
        _STYLED = True
    figure = Figure(figsize=(width, height), dpi=200)
    figure.patch.set_alpha(0.0)
    return figure


def style(axes: Any) -> None:
    """Apply the shared axis styling to one set of axes.

    Args:
        axes: The axes to style.
    """
    axes.set_facecolor("none")
    axes.tick_params(labelsize=8, colors=MUTED)
    axes.grid(color=MUTED, alpha=0.15, linewidth=0.6)
    for spine in axes.spines.values():
        spine.set_color(MUTED)
        spine.set_alpha(0.3)


def _legend(axes: Any, location: str = "best") -> None:
    """Draw a legend in the muted colour the rest of the figure uses.

    Args:
        axes: The axes to draw it on.
        location: Where to put it.
    """
    drawn = axes.legend(fontsize=7.5, frameon=False, loc=location)
    for text in drawn.get_texts():
        text.set_color(MUTED)


# --------------------------------------------------------------------------
# The ground state, drawn as spins
# --------------------------------------------------------------------------


def spin_rows(
    axes: Any,
    rows: tuple[Configuration, ...],
    *,
    periodic: bool,
    label_rows: bool = True,
) -> None:
    r"""Draw spin configurations as rows of up and down arrows.

    The one drawing primitive behind every cartoon here, so the large figure and
    the thumbnails cannot disagree about which way up is or which colour down is.
    A reader who learns the convention once has learnt it for the page.

    Args:
        axes: Where to draw.
        rows: The arrangements, heaviest first, drawn top to bottom.
        periodic: Whether the chain wraps. Only the domain-wall labels depend on
            it, but they would be wrong by one bond if it were guessed.
        label_rows: Whether to label each row with how many bonds it breaks.
    """
    sites = len(rows[0].spins)
    horizontal: list[int] = []
    vertical: list[int] = []
    lengths: list[float] = []
    colours: list[str] = []
    for row, configuration in enumerate(rows):
        for site, spin in enumerate(configuration.spins):
            horizontal.append(site)
            vertical.append(-row)
            lengths.append(ARROW_LENGTH * spin)
            colours.append(ACCENT if spin > 0 else WARM)

    axes.quiver(
        horizontal,
        vertical,
        [0.0] * len(horizontal),
        lengths,
        color=colours,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.6,
        pivot="mid",
    )
    axes.set_xlim(-0.8, sites - 0.2)
    axes.set_ylim(-len(rows) + 0.45, 0.55)
    axes.set_xticks([])
    if not label_rows:
        axes.set_yticks([])
        return
    walls = [configuration.domain_walls(periodic=periodic) for configuration in rows]
    axes.set_yticks([-row for row in range(len(rows))])
    axes.set_yticklabels(
        [f"{count} broken bond" + ("" if count == 1 else "s") for count in walls],
        fontsize=7,
        color=MUTED,
    )


@serialised
def superposition_figure(picture: Superposition) -> Figure:
    r"""Draw the ground state as the arrangements it is made of.

    This is the figure that answers "what does *superposition* actually mean here"
    without using the word twice. Each row is one arrangement of the magnets, and
    the bar beside it is how much of the ground state that arrangement is. A
    classical chain would be one row at 100%; anything else is the quantum part,
    drawn rather than asserted.

    Args:
        picture: The measured superposition.

    Returns:
        The figure: arrows on the left, shares on the right.
    """
    rows = picture.configurations
    figure = _styled_figure(7.4, 0.42 * len(rows) + 1.0)
    spins, shares = figure.subplots(1, 2, gridspec_kw={"width_ratios": [3.0, 1.0]})

    style(spins)
    spins.grid(False)
    spin_rows(spins, rows, periodic=picture.spec.boundary == "periodic")
    spins.set_title(
        f"the {len(rows)} heaviest of {picture.full_count} arrangements",
        fontsize=8,
        color=MUTED,
    )

    style(shares)
    positions = [-row for row in range(len(rows))]
    probabilities = [configuration.probability for configuration in rows]
    shares.barh(positions, probabilities, height=0.5, color=ACCENT, alpha=0.75)
    for position, probability in zip(positions, probabilities, strict=True):
        shares.annotate(
            f"{probability:.1%}",
            xy=(probability, position),
            xytext=(4, -2),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
    # Room on the right for the labels, which sit outside their bars so a short bar
    # is still readable. Without the headroom the widest label runs off the axes.
    shares.set_xlim(0.0, max(probabilities) * 1.45)
    shares.set_ylim(-len(rows) + 0.45, 0.55)
    shares.set_xticks([])
    shares.set_yticks([])
    shares.set_title("share of the state", fontsize=8, color=MUTED)

    figure.tight_layout()
    return figure


@serialised
def spreading_figure(
    pictures: tuple[Superposition, ...],
    ratios: tuple[float, ...],
    rows_each: int = 4,
) -> Figure:
    r"""Draw the same ground state at several fields, side by side.

    Read left to right this is the phase transition itself: the weight starts on
    the two aligned arrangements, moves through the ones with a broken bond in
    them, and ends shared out evenly among all :math:`2^L`. It is the one figure
    on the page that shows *change* rather than a state, which is why it is worth
    four panels.

    Args:
        pictures: One measured superposition per ratio, in order.
        ratios: The values of :math:`h/J` they were measured at.
        rows_each: Arrangements drawn in each panel.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 2.5)
    panels = figure.subplots(1, len(pictures))
    for axes, picture, ratio in zip(np.atleast_1d(panels), pictures, ratios, strict=True):
        style(axes)
        axes.grid(False)
        rows = picture.configurations[:rows_each]
        spin_rows(axes, rows, periodic=picture.spec.boundary == "periodic", label_rows=False)
        # A bar under each row rather than fading the arrows. Opacity would fight
        # the up/down colours, and it stops being readable in the rightmost panel
        # -- which is precisely the panel whose point is that the weights are equal.
        heaviest = max(rows[0].probability, 1e-12)
        for row, configuration in enumerate(rows):
            axes.barh(
                -row - 0.36,
                configuration.probability / heaviest * (len(rows) - 1),
                height=0.1,
                left=-0.5,
                color=MUTED,
                alpha=0.55,
            )
        axes.set_title(
            f"$h/J = {ratio:g}$\n{picture.effective_count:.1f} of {picture.full_count}",
            fontsize=8,
            color=MUTED,
        )
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# Where this chain sits
# --------------------------------------------------------------------------


@serialised
def phase_diagram_figure(
    spec: TFIMSpec,
    ratios: FloatArray,
    infinite_gap: FloatArray,
    finite_gap: FloatArray | None = None,
) -> Figure:
    r"""Draw the two phases beside the energy gap that separates them.

    Two panels, because the plane alone answers "which phase" and never "why
    there". On the left each phase is one flat colour, so the boundary between the
    fills *is* the critical line and can be seen before anything is read. On the
    right the gap along this chain's coupling, which is what makes that boundary a
    boundary: it reaches zero at :math:`h = J` and nowhere else.

    Shading the plane continuously by the gap was the first attempt. It was
    prettier and much harder to read -- a smooth gradient has no boundary in it, so
    both the phases and the critical field had to be found by reading labels.

    Args:
        spec: The chain to mark.
        ratios: The values of :math:`h/J` the gap curves are sampled at.
        infinite_gap: The infinite chain's gap at each of those ratios.
        finite_gap: This chain's own gap, where a closed form covers it. ``None``
            leaves the panel showing the infinite chain alone rather than showing
            a second curve computed a different way and captioned as if it were
            the same quantity.

    Returns:
        The figure.
    """
    # Wide enough to contain the marker. The knob allows a coupling and a field
    # well above one, and a star drawn outside its own axes is a bug the reader
    # cannot see -- the figure simply looks as though this chain is not on it.
    span = max(2.0, 1.18 * max(spec.coupling, spec.field))
    figure = _styled_figure(7.4, 3.5)
    plane, curve = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.15]})

    style(plane)
    plane.grid(False)
    plane.fill_between([0.0, span], [0.0, span], color=ORDERED_FILL, alpha=0.30)
    plane.fill_between([0.0, span], [0.0, span], [span, span], color=DISORDERED_FILL, alpha=0.30)
    plane.plot([0.0, span], [0.0, span], color=WARM, linewidth=2.2, zorder=3)
    plane.annotate(
        "ORDERED\nthe magnets line up\n↑↑↑↑↑↑  or  ↓↓↓↓↓↓",
        xy=(0.70 * span, 0.20 * span),
        fontsize=8,
        color=ORDERED_FILL,
        ha="center",
        va="center",
        fontweight="bold",
    )
    plane.annotate(
        "DISORDERED\nthe push wins\n→→→→→→",
        xy=(0.26 * span, 0.80 * span),
        fontsize=8,
        color=DISORDERED_FILL,
        ha="center",
        va="center",
        fontweight="bold",
    )
    plane.annotate(
        "critical line  $h = J$",
        xy=(0.52 * span, 0.52 * span),
        fontsize=7.5,
        color=WARM,
        rotation=45,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
    )
    plane.axvline(spec.coupling, color=MUTED, linestyle=":", linewidth=1.0, zorder=2)
    plane.scatter(
        [spec.coupling],
        [spec.field],
        s=170,
        marker="*",
        color=WARM,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    plane.annotate(
        f"this chain\n$h/J = {spec.ratio:.2f}$",
        xy=(spec.coupling, spec.field),
        xytext=(8, -16),
        textcoords="offset points",
        fontsize=7.5,
        color=WARM,
    )
    plane.set_xlabel("coupling $J$", fontsize=9, color=MUTED)
    plane.set_ylabel("sideways push $h$", fontsize=9, color=MUTED)
    plane.set_xlim(0.0, span)
    plane.set_ylim(0.0, span)
    plane.set_aspect("equal")

    style(curve)
    widest = float(ratios[-1]) if len(ratios) else 2.0
    curve.fill_between(
        [0.0, 1.0],
        0.0,
        1.0,
        transform=curve.get_xaxis_transform(),
        color=ORDERED_FILL,
        alpha=0.10,
    )
    curve.fill_between(
        [1.0, widest],
        0.0,
        1.0,
        transform=curve.get_xaxis_transform(),
        color=DISORDERED_FILL,
        alpha=0.10,
    )
    curve.plot(ratios, infinite_gap, color=MUTED, linewidth=1.6, label="infinitely long chain")
    if finite_gap is not None:
        curve.plot(
            ratios,
            finite_gap,
            color=ACCENT,
            linewidth=1.8,
            label=f"this chain, $L = {spec.n_sites}$",
        )
    curve.axvline(1.0, color=WARM, linewidth=2.2, zorder=3)
    curve.annotate(
        f"critical push\n$h_c = J = {spec.coupling:g}$",
        xy=(1.0, 0.94),
        xycoords=curve.get_xaxis_transform(),
        xytext=(7, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=WARM,
        va="top",
    )
    curve.axvline(spec.ratio, color=WARM, alpha=0.25, linewidth=6.0, zorder=0)
    # Labelled "h / J" and never "g". The sidebar spends a slider on the
    # longitudinal field g, and the Hamiltonian printed on the same page carries
    # a -g sum sigma^z term, so borrowing the letter here -- as much of the TFIM
    # literature does, where there is no longitudinal field to collide with --
    # would put two different quantities under one symbol on one screen.
    curve.set_xlabel("$h / J$", fontsize=9, color=MUTED)
    # One quasiparticle, and the label says so. The physical excitations of a ring
    # come in pairs and cost twice this -- `free_fermions.parity_even_gap`. This
    # curve is the right one to draw against the infinite chain, because that is the
    # same single-quasiparticle quantity, but a label reading "the gap" would make
    # the pair of curves a claim about a number they are both half of.
    curve.set_ylabel(r"one quasiparticle  $\epsilon$", fontsize=9, color=MUTED)
    curve.set_xlim(0.0, widest)
    curve.set_ylim(bottom=0.0)
    _legend(curve, "upper left")

    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# What the circuit did
# --------------------------------------------------------------------------


def _mark_reference(axes: Any, exact: float | None, label: str) -> None:
    """Draw the exact answer as a flat line, if there is one to draw.

    Args:
        axes: Where to draw.
        exact: The reference value, or ``None`` when no method could supply one.
        label: What to call it in the legend.
    """
    if exact is None:
        return
    axes.axhline(exact, color=MUTED, linewidth=1.4, linestyle="--", label=label)


@serialised
def convergence_figure(history: tuple[float, ...], exact: float | None) -> Figure:
    """Draw the optimiser's energy against the answer it is trying to reach.

    Args:
        history: The energy after each optimiser step, in order.
        exact: The true ground-state energy, or ``None`` if the chain is beyond
            every method that could supply one.

    Returns:
        The figure. The gap between the two lines at the right-hand edge is the
        whole result, so it is annotated rather than left to be eyeballed.
    """
    figure = _styled_figure(7.4, 3.0)
    axes = figure.subplots()
    style(axes)
    steps = list(range(len(history)))
    axes.plot(steps, history, color=ACCENT, linewidth=1.8, label="what the circuit reached")
    _mark_reference(axes, exact, "the true answer")
    if exact is not None and history:
        reached = history[-1]
        axes.annotate(
            f"short by {reached - exact:.2e}",
            xy=(steps[-1], reached),
            xytext=(-6, 14),
            textcoords="offset points",
            fontsize=8,
            color=WARM,
            ha="right",
        )
    axes.set_xlabel("optimiser step", fontsize=9, color=MUTED)
    # Same label as the method race, for the same reason: this axis carries the
    # quantity the optimiser is driving down, which a reader from machine learning
    # calls the cost and a reader from physics calls the energy.
    axes.set_ylabel("cost function (energy)", fontsize=9, color=MUTED)
    _legend(axes, "upper right")
    figure.tight_layout()
    return figure


@serialised
def ladder_figure(
    depths: tuple[int, ...],
    energies: tuple[float, ...],
    exact: float | None,
) -> Figure:
    """Draw what each added circuit layer bought, on a log scale.

    The energy itself barely moves after the first layer or two, so plotting it
    hides the finding. What a depth decision turns on is the distance still left
    to the true answer, and that falls by orders of magnitude -- which is a log
    axis or it is a flat line along the bottom.

    Args:
        depths: How many layers each rung has.
        energies: The energy that rung reached.
        exact: The true ground-state energy. ``None`` gives the raw energies
            instead, because a distance to an unknown answer is not a distance.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 3.0)
    axes = figure.subplots()
    style(axes)
    if exact is None:
        axes.plot(depths, energies, marker="o", color=ACCENT, linewidth=1.8)
        axes.set_ylabel("energy reached", fontsize=9, color=MUTED)
    else:
        # Clipped rather than dropped. A rung that landed on the answer to machine
        # precision is a success, and a log axis would silently omit the point --
        # leaving a gap in the line that reads as a failed run.
        distance = [max(energy - exact, 1e-16) for energy in energies]
        axes.plot(depths, distance, marker="o", color=ACCENT, linewidth=1.8)
        axes.set_yscale("log")
        axes.set_ylabel("how far above the true answer", fontsize=9, color=MUTED)
    axes.set_xlabel("circuit layers", fontsize=9, color=MUTED)
    axes.set_xticks(list(depths))
    figure.tight_layout()
    return figure


@serialised
def imaginary_time_figure(
    times: FloatArray,
    energies: tuple[float, ...],
    exact: float | None,
) -> Figure:
    r"""Draw the energy falling under :math:`e^{-\tau\hat H}`.

    Args:
        times: The values of :math:`\tau` sampled.
        energies: The energy at each of them.
        exact: The true ground-state energy, or ``None``.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 3.0)
    axes = figure.subplots()
    style(axes)
    axes.plot(times, energies, color=ACCENT, linewidth=1.8, label="energy as time runs")
    _mark_reference(axes, exact, "the true answer")
    axes.set_xlabel(r"imaginary time  $\tau$", fontsize=9, color=MUTED)
    axes.set_ylabel("energy", fontsize=9, color=MUTED)
    _legend(axes, "upper right")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# The circuit, drawn as a circuit
# --------------------------------------------------------------------------

MAX_DRAWN_QUBITS = MAX_SITES_STATEVECTOR
"""Wires drawn before the picture stops being a picture.

Set to the model's own ceiling, so **no problem this project can pose is drawn
short**. It was eight, on the reasoning that a repeating pattern is legible at
eight wires and illegible at twenty -- true of the horizontal direction, which is
set by the layer count, and not of this one: the figure's height grows with the
wire count while its width does not, so a sixteen-wire drawing is a taller picture
rather than a denser one. A ten-magnet question answered under an eight-magnet
diagram is the wrong trade, and it was the first thing readers noticed.
"""

MAX_DRAWN_LAYERS = 6
"""Layers drawn before the picture stops being a picture.

Unlike the wire count, this one is a real limit: layers run left to right and the
figure has to get wider to hold them, so past a point the gates are thinner than
the gaps between them. Six covers every rung the depth ladder actually climbs on a
chain this size, which is what matters -- a reader who asked about depth 4 and was
shown three layers is being told about a different circuit. Deeper than six the
drawing does abbreviate, and says so in its own title.
"""

COLUMN_INCHES = 0.82
"""Horizontal room one column of gates needs to stay legible.

The figure's width is this times the column count rather than a constant, so adding
a layer adds space instead of compressing every layer already drawn.
"""


@serialised
def circuit_figure(
    n_qubits: int,
    depth: int,
    bonds: tuple[tuple[int, int], ...],
    rounds: tuple[tuple[tuple[int, int], ...], ...],
) -> Figure:
    r"""Draw the ansatz as an actual circuit diagram.

    The tab that this belongs to used to open on four numbers -- how many gates,
    how deep, how many dials. Those answer a question somebody has after they know
    what the thing *is*, and a reader who has never seen a circuit diagram cannot
    get from ``two_qubit_gates = 30`` to a mental picture of anything. So the
    picture comes first and the counts are folded away underneath it.

    What the drawing has to make obvious is the one structural fact the whole cost
    argument rests on: within a layer the two-qubit gates come in **rounds of pairs
    that share no wire**, and a chain needs two such rounds no matter how long it
    is. Drawn, that is two columns of vertical connectors, offset from each other,
    and it is visible in half a second.

    Args:
        n_qubits: Width of the register.
        depth: How many layers the real circuit has. Only the first few are drawn;
            the rest are reported in the title rather than silently dropped.
        bonds: Every coupled pair, used only to decide the drawn width.
        rounds: The pairs grouped into rounds of mutually disjoint bonds, which is
            the grouping the depth arithmetic assumes. Passed in rather than
            recomputed, so the picture cannot claim a schedule the costing did not
            use.

    Returns:
        The figure.
    """
    wires = min(n_qubits, MAX_DRAWN_QUBITS)
    layers = min(max(depth, 1), MAX_DRAWN_LAYERS)
    columns_per_layer = len(rounds) + 1
    total_columns = layers * columns_per_layer

    figure = _styled_figure(max(7.4, COLUMN_INCHES * total_columns), 0.42 * wires + 1.75)
    axes = figure.subplots()
    style(axes)
    axes.grid(False)

    for wire in range(wires):
        axes.plot(
            [-0.7, total_columns - 0.3],
            [-wire, -wire],
            color=MUTED,
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )

    # Where the operator labels sit: one line below the lowest wire. Collected while
    # the columns are drawn rather than recomputed afterwards, so a label can never
    # end up under a column that was skipped.
    label_y = -wires + 0.15
    column = 0
    for layer in range(layers):
        # One label for the whole layer's two-qubit block, centred across it, rather
        # than one per round. There is a single angle per layer: the rounds exist
        # because gates sharing a wire cannot run at once, which is a fact about the
        # schedule and not about the operator. A label on each column would draw a
        # reader to the opposite conclusion -- that each round has its own dial.
        coupling_start = column
        for group in rounds:
            for left, right in group:
                # A bond wrapping round the ring joins the top wire to the bottom
                # one, which would draw as a connector straight through every gate
                # between them. Left out of the drawing and named in the caption.
                if max(left, right) >= wires or abs(left - right) != 1:
                    continue
                axes.plot(
                    [column, column],
                    [-left, -right],
                    color=ACCENT,
                    linewidth=1.6,
                    zorder=2,
                )
                for end in (left, right):
                    axes.scatter([column], [-end], s=26, color=ACCENT, zorder=3)
            column += 1
        axes.annotate(
            rf"$e^{{-i\gamma_{layer + 1}\sum\hat\sigma^z\hat\sigma^z}}$",
            xy=((coupling_start + column - 1) / 2, label_y),
            fontsize=7,
            color=ACCENT,
            ha="center",
            va="top",
        )
        axes.annotate(
            rf"$e^{{-i\beta_{layer + 1}\sum\hat\sigma^x}}$",
            xy=(column, label_y),
            fontsize=6.5,
            color=WARM,
            ha="center",
            va="top",
        )
        for wire in range(wires):
            axes.add_patch(
                Rectangle(
                    (column - 0.26, -wire - 0.22),
                    0.52,
                    0.44,
                    facecolor=WARM,
                    edgecolor="none",
                    alpha=0.85,
                    zorder=3,
                )
            )
        axes.annotate(
            f"layer {layer + 1}",
            xy=(column - (columns_per_layer - 1) / 2, 0.62),
            fontsize=7.5,
            color=MUTED,
            ha="center",
        )
        column += 1

    axes.set_xlim(-0.9, total_columns - 0.1)
    axes.set_ylim(-wires - 0.45, 1.05)
    axes.set_xticks([])
    axes.set_yticks([-wire for wire in range(wires)])
    axes.set_yticklabels([f"magnet {wire + 1}" for wire in range(wires)], fontsize=7, color=MUTED)
    for spine in axes.spines.values():
        spine.set_visible(False)

    hidden = []
    if n_qubits > wires:
        spare = n_qubits - wires
        hidden.append(f"{spare} more magnet{'' if spare == 1 else 's'}")
    if depth > layers:
        spare = depth - layers
        hidden.append(f"{spare} more identical layer{'' if spare == 1 else 's'}")
    if any(abs(left - right) != 1 for left, right in bonds):
        # A ring's closing bond joins the top wire to the bottom one, which would
        # draw as a connector straight through every gate in between. Left out and
        # said so, rather than left out and hoped nobody counts the connectors.
        hidden.append("the bond that closes the ring")
    axes.set_title(
        "blue = a gate joining two neighbouring magnets   ·   "
        "orange = a nudge to one magnet\n"
        r"every box is a unitary $e^{-i\theta \hat P}$ -- a reversible rotation whose "
        r"angle $\gamma$ or $\beta$ is learned, one per layer"
        + (f"\nnot drawn: {' and '.join(hidden)}" if hidden else ""),
        fontsize=7.5,
        color=MUTED,
    )
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# The machine, and the chain laid onto it
# --------------------------------------------------------------------------


def _positions(n_nodes: int, edges: tuple[tuple[int, int], ...]) -> FloatArray:
    """Place the qubits of a machine on the page.

    Three cases rather than one general algorithm, because the general one draws
    the two easy shapes badly. A row of qubits laid out by a force simulation comes
    out as a wiggle, and a machine where everything talks to everything comes out
    as an indistinct blob -- and in both cases the shape *is* the information.

    Args:
        n_nodes: How many qubits.
        edges: The undirected coupling pairs.

    Returns:
        An ``(n_nodes, 2)`` array of positions.
    """
    degree = [0] * n_nodes
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1

    if n_nodes > 1 and len(edges) == n_nodes * (n_nodes - 1) // 2:
        angles = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)
    if all(count <= 2 for count in degree):
        return np.stack([np.arange(n_nodes, dtype=float), np.zeros(n_nodes)], axis=1)
    return _spring_positions(n_nodes, edges)


def _spring_positions(
    n_nodes: int,
    edges: tuple[tuple[int, int], ...],
    iterations: int = 300,
    seed: int = 0,
) -> FloatArray:
    """Lay out a graph by force simulation, deterministically.

    Written here in a dozen lines rather than taken from a graph library, because
    the only graph this project draws has twenty-seven nodes and adding a
    dependency to place them would be the more expensive choice. Seeded, so the
    same machine is drawn the same way every time -- a diagram that rearranges
    itself between reruns is one a reader cannot get familiar with.

    Args:
        n_nodes: How many nodes.
        edges: The undirected pairs.
        iterations: Simulation steps.
        seed: Seed for the starting positions.

    Returns:
        An ``(n_nodes, 2)`` array of positions.
    """
    generator = np.random.default_rng(seed)
    positions: FloatArray = generator.uniform(-1.0, 1.0, size=(n_nodes, 2))
    ideal = 1.0 / np.sqrt(max(n_nodes, 1))
    for step in range(iterations):
        offsets = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(offsets, axis=-1)
        np.fill_diagonal(distances, np.inf)
        force = (offsets / distances[..., None] ** 2 * ideal**2).sum(axis=1)
        for left, right in edges:
            along = positions[left] - positions[right]
            length = max(float(np.linalg.norm(along)), 1e-9)
            pull = along / length * (length**2 / ideal)
            force[left] -= pull
            force[right] += pull
        # Cooling, so early steps untangle the graph and late ones only settle it.
        temperature = 0.1 * (1.0 - step / iterations)
        lengths = np.linalg.norm(force, axis=1, keepdims=True)
        positions = positions + force / np.maximum(lengths, 1e-9) * np.minimum(lengths, temperature)
    return positions


@serialised
def wiring_figure(
    n_qubits: int,
    coupling: tuple[tuple[int, int], ...],
    sites: tuple[int, ...],
    stretched: tuple[tuple[int, ...], ...],
    device_name: str,
) -> Figure:
    r"""Draw the machine's wiring with the chain laid onto it.

    This is the picture behind the phrase *wiring costs*, and without it that
    phrase is a number with no meaning. The chain wants each magnet next to the
    next one. A real machine is wired in a fixed pattern that was not chosen with
    this chain in mind, so some neighbouring pair ends up on two qubits that are
    not wired together, and the compiler has to shuffle states along until they
    are. Drawn, the argument needs no explanation: the orange path is a pair of
    neighbours that had to travel to meet.

    Args:
        n_qubits: How many qubits the machine has.
        coupling: The machine's undirected pairs.
        sites: Physical qubit for each chain magnet, in order along the chain.
        stretched: The routed path for every bond that needed more than one edge,
            as physical qubit indices from one end to the other.
        device_name: What to call the machine in the title.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 4.2)
    axes = figure.subplots()
    style(axes)
    axes.grid(False)
    places = _positions(n_qubits, coupling)

    for left, right in coupling:
        axes.plot(
            [places[left, 0], places[right, 0]],
            [places[left, 1], places[right, 1]],
            color=MUTED,
            linewidth=0.9,
            alpha=0.35,
            zorder=1,
        )
    axes.scatter(
        places[:, 0], places[:, 1], s=90, color=MUTED, alpha=0.30, zorder=2, edgecolor="none"
    )

    for first, second in pairwise(sites):
        axes.plot(
            [places[first, 0], places[second, 0]],
            [places[first, 1], places[second, 1]],
            color=ACCENT,
            linewidth=2.4,
            zorder=3,
        )
    for path in stretched:
        axes.plot(
            [places[qubit, 0] for qubit in path],
            [places[qubit, 1] for qubit in path],
            color=WARM,
            linewidth=2.6,
            linestyle="--",
            zorder=4,
        )
    used = np.asarray(sites)
    axes.scatter(
        places[used, 0],
        places[used, 1],
        s=180,
        color=ACCENT,
        zorder=5,
        edgecolor="white",
        linewidth=0.8,
    )
    for order, qubit in enumerate(sites):
        axes.annotate(
            str(order + 1),
            xy=(places[qubit, 0], places[qubit, 1]),
            fontsize=7,
            color="white",
            ha="center",
            va="center",
            zorder=6,
        )

    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_aspect("equal")
    axes.set_title(
        f"{device_name} — grey is the machine's own wiring, blue is where the chain went"
        + ("   ·   orange = neighbours that had to travel to meet" if stretched else ""),
        fontsize=7.5,
        color=MUTED,
    )
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# The mapping, and the loop
# --------------------------------------------------------------------------


@serialised
def chain_terms_figure(n_sites: int, coupling: float, field: float, *, periodic: bool) -> Figure:
    r"""Draw the chain as its two competing terms, before either is written down.

    The page this opens used to begin with five stacked paragraphs and no picture,
    which is a poor trade: a reader arriving from the Chat page has been handed a
    verdict about this chain and has never been shown what a chain *is*. The
    Hamiltonian below the figure then says the same thing in symbols, and a symbol
    is easier to read second.

    What the drawing has to carry is the one fact the whole page rests on, and it
    is a fact about *disagreement*: the blue links want every magnet pointing the
    same way along :math:`\hat\sigma^z`, the orange arrows want each of them along
    :math:`\hat\sigma^x`, and no arrangement of magnets satisfies both. Everything
    the circuits do later is an attempt to settle that argument.

    Drawn small and wide deliberately. It sits above the tabs, so it is on screen
    the whole time a reader is on any of them, and an opener that takes a third of
    the window is an opener that has to be scrolled past eight times.

    Args:
        n_sites: How many magnets. Drawn up to :data:`MAX_DRAWN_MAGNETS`, past
            which the row is truncated and the count is stated instead.
        coupling: The Ising coupling :math:`J`, shown on the links.
        field: The transverse field :math:`h`, shown on the push arrows.
        periodic: Whether the chain closes into a ring, which adds the wrapping
            bond as a stub off each end rather than an arc over the row.

    Returns:
        The figure.
    """
    drawn = min(n_sites, MAX_DRAWN_MAGNETS)
    figure = _styled_figure(7.4, 1.45)
    axes = figure.subplots()
    style(axes)
    axes.grid(False)

    for site in range(drawn - 1):
        axes.plot([site, site + 1], [0.0, 0.0], color=ACCENT, linewidth=2.2, zorder=1)
    if periodic and drawn > 2:
        # A stub rather than an arc, for the same reason the sheet uses one: an arc
        # over the row crosses every magnet it passes and the reader ends up
        # decoding the drawing instead of the physics.
        for stub in (-0.5, drawn - 1 + 0.5):
            axes.plot(
                [stub, round(stub)],
                [0.0, 0.0],
                color=ACCENT,
                linewidth=2.2,
                linestyle=":",
                zorder=1,
            )

    # Up-arrows for the aligned state the couplings are asking for, and sideways
    # arrows underneath for the push that will not let them have it. Two colours,
    # two directions, one glance.
    axes.quiver(
        list(range(drawn)),
        [0.0] * drawn,
        [0.0] * drawn,
        [ARROW_LENGTH] * drawn,
        color=ACCENT,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        headwidth=3.4,
        headlength=4.4,
        headaxislength=3.8,
        pivot="mid",
        zorder=3,
    )
    axes.quiver(
        [site - 0.30 for site in range(drawn)],
        [-0.72] * drawn,
        [0.58] * drawn,
        [0.0] * drawn,
        color=WARM,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0035,
        headwidth=3.6,
        headlength=4.6,
        headaxislength=4.0,
        zorder=3,
    )

    axes.annotate(
        rf"$-J\,\hat\sigma^z_i\hat\sigma^z_{{i+1}}$   keeps neighbours agreeing"
        rf"   ($J = {coupling:g}$)",
        xy=(-0.55, 0.80),
        fontsize=7.5,
        color=ACCENT,
        ha="left",
        va="center",
    )
    axes.annotate(
        rf"$-h\,\hat\sigma^x_i$   pushes each one sideways   ($h = {field:g}$)",
        xy=(-0.55, -1.18),
        fontsize=7.5,
        color=WARM,
        ha="left",
        va="center",
    )
    if drawn < n_sites:
        # Above the row, not beside it: a ring draws a wrapping stub off the right
        # end and the two landed on top of each other.
        axes.annotate(
            f"… {n_sites} magnets in all",
            xy=(drawn - 1.15, 0.52),
            fontsize=7.5,
            color=MUTED,
            ha="right",
            va="center",
        )
    axes.set_xlim(-0.75, drawn - 0.25)
    axes.set_ylim(-1.40, 1.05)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    return figure


@serialised
def dual_lattice_figure(n_sites: int, n_slices: int, *, periodic: bool) -> Figure:
    r"""Draw the classical lattice one quantum chain turns into.

    The mapping is the least intuitive idea in the project and the easiest one to
    draw. A row of magnets with a quantum push on it becomes a *sheet* of ordinary
    magnets with no quantum anything in it: one copy of the chain per slice of
    imaginary time, the copies stacked and joined to their neighbours in the stack.
    The extra direction is time, never a second row of magnets, and the picture is
    the only way to say that in one gesture.

    Args:
        n_sites: Length of the quantum chain, drawn left to right.
        n_slices: Imaginary-time slices, :math:`2P + 1`, drawn bottom to top.
        periodic: Whether the chain is a ring, which adds the wrapping bond.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 3.6)
    chain, sheet = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.9]})

    style(chain)
    chain.grid(False)
    for site in range(n_sites - 1):
        chain.plot([site, site + 1], [0, 0], color=ACCENT, linewidth=2.0, zorder=1)
    chain.quiver(
        list(range(n_sites)),
        [0] * n_sites,
        [0.0] * n_sites,
        [ARROW_LENGTH] * n_sites,
        color=ACCENT,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.012,
        pivot="mid",
        zorder=2,
    )
    chain.quiver(
        [site - 0.30 for site in range(n_sites)],
        [-0.85] * n_sites,
        [0.58] * n_sites,
        [0.0] * n_sites,
        color=WARM,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.010,
        headwidth=3.6,
        headlength=4.6,
        headaxislength=4.0,
        zorder=2,
    )
    chain.set_xlim(-0.8, n_sites - 0.2)
    chain.set_ylim(-1.6, 1.6)
    chain.set_xticks([])
    chain.set_yticks([])
    chain.set_title(
        f"one quantum chain\n{n_sites} magnets, coupled in blue, pushed in orange",
        fontsize=8,
        color=MUTED,
    )
    for spine in chain.spines.values():
        spine.set_visible(False)

    style(sheet)
    sheet.grid(False)
    for slice_index in range(n_slices):
        for site in range(n_sites - 1):
            sheet.plot(
                [site, site + 1],
                [slice_index, slice_index],
                color=ACCENT,
                linewidth=1.5,
                zorder=1,
            )
        if periodic and n_sites > 2:
            # The wrapping bond drawn as a stub off each end rather than as an arc
            # over the row. Arcs from five rows at once cross each other and every
            # row they pass, and the reader ends up decoding the drawing instead of
            # the physics -- which is the one thing a cartoon may not cost.
            for stub in (-0.45, n_sites - 1 + 0.45):
                sheet.plot(
                    [stub, round(stub)],
                    [slice_index, slice_index],
                    color=ACCENT,
                    linewidth=1.5,
                    linestyle=":",
                    zorder=1,
                )
    for site in range(n_sites):
        for slice_index in range(n_slices - 1):
            sheet.plot(
                [site, site],
                [slice_index, slice_index + 1],
                color=WARM,
                linewidth=1.5,
                linestyle="--",
                zorder=1,
            )
    middle = (n_slices - 1) // 2
    for slice_index in range(n_slices):
        colour = WARM if slice_index == middle else MUTED
        sheet.scatter(
            list(range(n_sites)),
            [slice_index] * n_sites,
            s=44 if slice_index == middle else 30,
            color=colour,
            zorder=3,
            edgecolor="none",
        )
    sheet.annotate(
        "the slice the answer is read off",
        xy=(n_sites - 1 + 0.6, middle),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=7,
        color=WARM,
        va="center",
    )
    sheet.set_xlim(-0.9, n_sites + 2.6)
    sheet.set_ylim(-0.8, n_slices - 0.2)
    sheet.set_xticks([])
    sheet.set_yticks([])
    sheet.set_xlabel("along the chain", fontsize=8, color=MUTED)
    sheet.set_ylabel("imaginary time", fontsize=8, color=MUTED)
    sheet.set_title(
        f"one ordinary sheet of magnets\n{n_sites} x {n_slices}, nothing quantum left in it",
        fontsize=8,
        color=MUTED,
    )
    for spine in sheet.spines.values():
        spine.set_visible(False)

    figure.tight_layout()
    return figure


HYBRID_STEPS: tuple[tuple[str, str, str, str], ...] = (
    (
        "quantum",
        "1. run",
        "run the circuit\nwith the dials\nwhere they are",
        r"$|\psi(\theta)\rangle = \hat{U}(\theta)|{+}\rangle^{\otimes L}$",
    ),
    (
        "quantum",
        "2. measure",
        "read the magnets,\nmany thousands\nof times over",
        r"$m = \pm 1$  at  $\langle\psi|\hat{\Pi}_\pm|\psi\rangle$",
    ),
    (
        "classical",
        "3. average",
        "turn those\nreadings into\none energy",
        r"$\hat{E} = \sum_\alpha c_\alpha \bar{P}_\alpha$",
    ),
    (
        "classical",
        "4. decide",
        "move the dials,\nor stop if they\nhave stopped\nhelping",
        r"$\theta \leftarrow \theta - \eta\,\nabla_\theta E$",
    ),
)
"""The four steps of the loop: the machine each runs on, the plain words, the algebra.

Written as data rather than drawn inline so that the diagram and any prose about
it cannot disagree about how many steps there are or where the boundary falls.

**The fourth field is a headline, not the equation.** The full forms live in the Lab
page's *What the machine does* tab, written in LaTeX where they can be read; what
sits in the box is the shortest symbol group that identifies which of them the step
is. Two reasons it is abbreviated rather than complete. A box 1.4 inches wide fits
the step-one product only at a size nobody can read, and it would have to displace
the plain English -- which is the half of this diagram that works for a reader who
does not do physics. And this is matplotlib's *mathtext* rather than LaTeX: a
complete equation here would be a second spelling of one already written elsewhere,
in a dialect that supports less, with nothing checking the two against each other.
"""


@serialised
def hybrid_loop_figure() -> Figure:
    """Draw the quantum-classical loop as a loop.

    The single most important structural fact about every algorithm in this
    project is that the quantum computer is not the algorithm -- it is a
    subroutine inside an ordinary one, called thousands of times, and it returns a
    single number each time. Somebody who has only heard *quantum algorithm*
    imagines the whole computation happening on the device. The diagram corrects
    that in less time than a paragraph does.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 3.7)
    axes = figure.subplots()
    style(axes)
    axes.grid(False)

    # Taller than the text needs, because the bottom strip of each box holds the
    # step's algebra and the English above it must not be squeezed to make room.
    width, height, gap = 1.55, 1.36, 0.42
    for index, (side, title, body, algebra) in enumerate(HYBRID_STEPS):
        left = index * (width + gap)
        colour = ACCENT if side == "quantum" else WARM
        axes.add_patch(
            Rectangle(
                (left, 0.0),
                width,
                height,
                facecolor=colour,
                edgecolor="none",
                alpha=0.16,
                zorder=1,
            )
        )
        axes.annotate(
            title,
            xy=(left + width / 2, height - 0.18),
            fontsize=8.5,
            color=colour,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=2,
        )
        axes.annotate(
            body,
            xy=(left + width / 2, height / 2 + 0.10),
            fontsize=7.5,
            color=MUTED,
            ha="center",
            va="center",
            zorder=2,
        )
        # A rule rather than a gap, so the algebra reads as a restatement of the box
        # above it and not as a fifth thing the box happens to contain.
        axes.plot(
            [left + 0.14, left + width - 0.14],
            [0.30, 0.30],
            color=colour,
            linewidth=0.6,
            alpha=0.45,
            zorder=2,
        )
        axes.annotate(
            algebra,
            xy=(left + width / 2, 0.15),
            fontsize=7.0,
            color=colour,
            ha="center",
            va="center",
            zorder=2,
        )
        if index + 1 < len(HYBRID_STEPS):
            axes.annotate(
                "",
                xy=(left + width + gap, height / 2),
                xytext=(left + width, height / 2),
                arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.2},
            )

    span = len(HYBRID_STEPS) * (width + gap) - gap
    # The return arrow is the whole point of calling it a loop, so it is drawn
    # under the boxes at full width rather than as a short hop between the last
    # two -- a reader should see that step four goes back to step one.
    axes.annotate(
        "",
        xy=(width / 2, -0.30),
        xytext=(span - width / 2, -0.30),
        arrowprops={
            "arrowstyle": "->",
            "color": MUTED,
            "linewidth": 1.2,
            "connectionstyle": "arc3,rad=-0.28",
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    # The one equation the whole loop exists to evaluate, drawn *inside* the return
    # arc rather than beneath it. The arc encloses empty space, and what belongs in
    # the middle of a loop is the quantity the loop is minimising -- a reader who
    # takes nothing else from the diagram should take this.
    axes.annotate(
        r"$E_0 \;\leq\; \min_\theta\, \langle\psi(\theta)|\hat{H}|\psi(\theta)\rangle$",
        xy=(span / 2, -0.66),
        fontsize=10.5,
        color=MUTED,
        ha="center",
        va="center",
    )
    # One block below the arrow rather than two, and below its lowest point rather
    # than on it: the arc dips to about -1.3 at mid-span and the caption used to sit
    # at -1.18, with the line drawn straight through the words. Both sentences are
    # kept -- what the loop does, and what the inequality above means -- because the
    # second is the only reason the first is worth running.
    axes.annotate(
        "and round again, until the dials stop improving\n"
        "— whatever it reports is at or above the true lowest energy, "
        "so lower is always better",
        xy=(span / 2, -1.62),
        fontsize=7.5,
        color=MUTED,
        ha="center",
        va="center",
        linespacing=1.6,
    )
    axes.annotate(
        "on the quantum machine",
        xy=(width + gap / 2, height + 0.22),
        fontsize=8,
        color=ACCENT,
        ha="center",
        fontweight="bold",
    )
    axes.annotate(
        "on an ordinary computer",
        xy=(span - width - gap / 2, height + 0.22),
        fontsize=8,
        color=WARM,
        ha="center",
        fontweight="bold",
    )
    axes.set_xlim(-0.3, span + 0.3)
    axes.set_ylim(-1.95, height + 0.55)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# The family of methods, and the race between them
# --------------------------------------------------------------------------

METHODS: tuple[tuple[str, str, str, str], ...] = (
    (
        "VQE",
        "lowest energy of\na quantum system",
        "a search turns\nthe dials",
        "molecules,\nmaterials",
    ),
    (
        "imaginary time",
        "the same target,\nwithout a search",
        "the physics picks\nthe next step",
        "when the search\nitself stalls",
    ),
    (
        "QAOA",
        "best of exponentially\nmany choices",
        "a search sets\np pairs of angles",
        "routing,\nscheduling",
    ),
    (
        "annealing",
        "the same target,\nin one slow sweep",
        "no dials at all:\njust go slowly",
        "large discrete\nproblems",
    ),
)
"""The four near-term methods: name, what it targets, how it moves, what it is for.

Four short columns rather than a paragraph each. A reader arriving at this page has
been told that a quantum computer is a subroutine inside an ordinary loop and has no
way yet to tell these four apart; the useful first fact about each is what it is
*aiming at*, because that is the axis on which they genuinely differ.

**Ordered in two pairs, and the order is the argument.** The first pair asks for an
energy and the second asks for a choice -- that is the split worth carrying away, so
it is contiguous rather than interleaved, and the header above each pair can then
mean what it says. Within a pair the left method searches for its own settings and
the right one lets the physics fix them, which is the second-most-useful fact and
comes free from the same arrangement.

The prose version lives on the ``method-comparison`` shelf of the corpus, and the
Knowledge base page publishes it.
"""


@serialised
def method_map_figure() -> Figure:
    """Draw the four methods side by side, one column each.

    The cartoon exists because the alternative is four paragraphs, and a reader
    deciding whether this page is for them will not read four paragraphs. Ordered
    left to right by how much freedom the method has: VQE chooses everything,
    annealing chooses nothing but the clock.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 3.2)
    axes = figure.subplots()
    style(axes)
    axes.grid(False)

    width, height, gap = 1.62, 2.05, 0.30
    for index, (name, target, motion, uses) in enumerate(METHODS):
        left = index * (width + gap)
        # Blue for the first pair, which asks for an energy; orange for the second,
        # which asks for a choice. The pairs are contiguous, so the colour blocks
        # agree with the two headers drawn above them.
        colour = ACCENT if index < 2 else WARM
        axes.add_patch(
            Rectangle(
                (left, 0.0),
                width,
                height,
                facecolor=colour,
                edgecolor="none",
                alpha=0.14,
                zorder=1,
            )
        )
        centre = left + width / 2
        axes.annotate(
            name,
            xy=(centre, height - 0.22),
            fontsize=9,
            color=colour,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=2,
        )
        for offset, text, size in (
            (1.42, target, 7.4),
            (0.82, motion, 7.0),
            (0.26, uses, 7.0),
        ):
            axes.annotate(
                text,
                xy=(centre, offset),
                fontsize=size,
                color=MUTED,
                ha="center",
                va="center",
                zorder=2,
            )

    span = len(METHODS) * (width + gap) - gap
    # Each header is centred over its own pair of columns, computed from the column
    # geometry rather than nudged into place, so a change to `width` or `gap` cannot
    # leave a label sitting over the wrong pair.
    for label, first_column, colour in (
        ("asks for an energy", 0, ACCENT),
        ("asks for a choice", 2, WARM),
    ):
        position = first_column * (width + gap) + width + gap / 2
        axes.annotate(
            label,
            xy=(position, height + 0.30),
            fontsize=8,
            color=colour,
            ha="center",
            fontweight="bold",
        )
    axes.annotate(
        "in each pair, the left method searches for its own settings "
        "and the right one lets the physics fix them",
        xy=(span / 2, -0.42),
        fontsize=7.5,
        color=MUTED,
        ha="center",
    )
    axes.set_xlim(-0.3, span + 0.3)
    axes.set_ylim(-0.75, height + 0.70)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    return figure


RACE_STARTS: tuple[tuple[str, AnsatzFamily, Initialisation], ...] = (
    ("QAOA start: follow the slow sweep", "qaoa", "adiabatic_ramp"),
    ("VQE start: nudge off the identity", "hva", "small_angle"),
    ("cold start: every dial at zero", "hva", "zeros"),
)
"""The three opening positions raced against each other on the Lab's method tab.

The same circuit and the same optimiser in all three, so the only variable is where
the angles began -- which is what makes the comparison mean anything. The third is
not a straw man: zero is the obvious thing to try, it is what a reader would guess,
and it is stationary. Watching the optimiser stop on its first step is the cheapest
honest demonstration of a barren plateau this project can produce, and it costs a
second to run.

Lives here, beside the figure that draws it, because the page and the script that
writes the PDF must race the same three starts or the figure on screen and the
figure in the report are different figures.
"""


@serialised
def method_race_figure(
    runs: tuple[tuple[str, tuple[float, ...]], ...],
    exact: float | None,
    xlabel: str = "optimiser step",
    xlimit: tuple[int, int] | None = None,
) -> Figure:
    """Draw several strategies descending towards the same answer on one axis.

    One panel rather than three, because the comparison is the point: three curves
    on separate axes let a reader believe whichever they looked at last. What
    differs between the runs is only where the angles started and which tradition
    named them, so a gap between the curves is attributable to that and to nothing
    else -- same chain, same circuit shape, same optimiser.

    Args:
        runs: One ``(label, energy history)`` pair per strategy, in the order they
            should appear in the legend.
        exact: The true ground-state energy, or ``None`` when the chain is beyond
            every method that could supply one.
        xlabel: What one position along the axis means. A parameter rather than a
            constant because the same drawing serves two comparisons whose steps are
            not the same thing: three *starts* of one optimiser share its steps,
            while three *methods* do not -- an L-BFGS step and an imaginary-time step
            are different purchases, and an axis that called both "optimiser step"
            would invite the reader to compare the lengths of the curves directly.
        xlimit: The stretch of the axis to show, when a reader asked for one. The
            curves are drawn in full and the *view* is narrowed, rather than the
            histories being truncated: a cropped picture drawn from cropped data
            would show a run ending where it was cut off, which is a different claim
            entirely and a very easy one to make by accident.

    Returns:
        The figure. Each curve is annotated at its own end with how far short it
        stopped, because the endpoints are the result and they are usually too
        close together to read off the axis.
    """
    figure = _styled_figure(7.4, 3.2)
    axes = figure.subplots()
    style(axes)

    palette = (ACCENT, WARM, "#2f9e44")
    dashes = ("-", "--", "-.")
    for index, (label, history) in enumerate(runs):
        if not history:
            continue
        axes.plot(
            range(len(history)),
            history,
            color=palette[index % len(palette)],
            linestyle=dashes[index % len(dashes)],
            linewidth=1.7,
            # A run that stopped at its first step is a single point, and a line
            # through one point draws nothing at all -- which would read as a
            # missing curve rather than as the result it is.
            marker="o" if len(history) == 1 else None,
            markersize=5,
            label=label,
        )
    _mark_reference(axes, exact, "the true answer")

    if exact is not None:
        for index, (_, history) in enumerate(runs):
            if not history:
                continue
            # A curve that ran puts its shortfall to the right of its own last
            # point; a single dot sits at step zero, where "to the right" is on top
            # of every other curve's beginning, so that one goes above instead.
            stalled = len(history) == 1
            axes.annotate(
                f"{history[-1] - exact:.1e}",
                xy=(len(history) - 1, history[-1]),
                xytext=(0, 11) if stalled else (5, 0),
                textcoords="offset points",
                fontsize=7.5,
                color=palette[index % len(palette)],
                ha="center" if stalled else "left",
                va="center",
            )

    # Room on the right for the shortfall labels, which sit outside the last data
    # point and are clipped by a tight axis.
    longest = max((len(history) for _, history in runs), default=1)
    if xlimit is None:
        axes.set_xlim(-longest * 0.04, longest * 1.09)
    else:
        first, last = xlimit
        span = max(last - first, 1)
        axes.set_xlim(first - span * 0.04, last + span * 0.09)
    axes.set_xlabel(xlabel, fontsize=9, color=MUTED)
    # "cost function" first and "energy" in brackets, because the two audiences for
    # this figure read the same axis under two names and only one of them is exact.
    # What is plotted is the expectation value the methods drive downhill; a reader
    # from machine learning calls that the cost -- or the loss -- and a reader from
    # physics calls it the energy, and they are the same number.
    #
    # Not "loss", deliberately. Loss implies an optimiser minimising a training
    # objective, which is true of VQE and QAOA and false of VarQITE: imaginary-time
    # evolution flows downhill in energy with no optimiser and no loss to speak of.
    # "Cost function" covers a gradient-free flow as well as a search, and the
    # bracket makes the physics name unmissable either way.
    axes.set_ylabel("cost function (energy)", fontsize=9, color=MUTED)
    _legend(axes, "upper right")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# The exact solution, traced across the field
# --------------------------------------------------------------------------

CRITICAL_RATIO = 1.0
r"""Where the infinite chain's transition sits, in units of :math:`J`.

Drawn on every sweep figure as a vertical marker, because the whole point of these
curves is *what happens near it*, and a reader who has to find it by eye on the axis
has been given a chart rather than a figure.
"""


def _mark_critical(axes: Any, label: bool = True) -> None:
    r"""Draw the infinite chain's critical field on a sweep.

    Args:
        axes: The axes to mark.
        label: Whether to name it in the legend. Off for the second and later
            panels of a stack, where one entry per panel is repetition.
    """
    axes.axvline(
        CRITICAL_RATIO,
        color=WARM,
        linestyle=":",
        linewidth=1.2,
        label="$h/J = 1$: the infinite chain's transition" if label else None,
    )


@serialised
def spectrum_sweep_figure(
    ratios: FloatArray,
    excitations: tuple[tuple[float, ...], ...],
    thermodynamic_gap: FloatArray | None = None,
) -> Figure:
    r"""Draw the low-lying levels against the field, measured from the ground state.

    The figure for *plot the low-lying spectrum as a function of the external
    field*. Levels are drawn as :math:`E_n - E_0` rather than as absolute energies,
    and that is the difference between a figure and a chart: the absolute levels all
    slide down the page together as the field rises, so the structure the question is
    about -- which levels approach each other, and where -- is squeezed into the
    thickness of a line.

    Two features are worth naming before a reader looks for them. Below
    :math:`h/J = 1` the lowest excitation sits on the axis: it is the symmetry-broken
    partner of the ground state, split from it by an amount that falls exponentially
    with chain length. Above it that pair separates, and the cheapest excitation
    becomes a genuine quasiparticle pair whose energy grows with the field.

    Args:
        ratios: The swept :math:`h/J` values.
        excitations: One tuple per swept point, each holding :math:`E_n - E_0` for
            the computed levels and opening with the ground state's own ``0.0``.
        thermodynamic_gap: :math:`2\lvert J - h \rvert` at each point, the infinite
            chain's gap, drawn as the reference the finite one is failing to reach.
            ``None`` leaves it off.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 3.4)
    axes = figure.subplots()
    style(axes)
    drawn = min((len(point) for point in excitations), default=0)
    for level in range(1, drawn):
        axes.plot(
            ratios,
            [point[level] for point in excitations],
            color=ACCENT,
            alpha=max(0.25, 1.0 - 0.13 * level),
            linewidth=1.5,
            label="$E_n - E_0$, exact levels" if level == 1 else None,
        )
    if thermodynamic_gap is not None:
        axes.plot(
            ratios,
            thermodynamic_gap,
            color=MUTED,
            linestyle="--",
            linewidth=1.3,
            label="$2|J-h|$: the infinite chain's gap",
        )
    _mark_critical(axes)
    axes.set_xlabel("field over coupling, $h/J$", fontsize=9, color=MUTED)
    axes.set_ylabel("energy above the ground state", fontsize=9, color=MUTED)
    # Headroom for the legend, which sits over the top-left corner where the levels
    # at zero field start. Without it the first entry overlaps the highest curve.
    highest = max((max(point[1:], default=0.0) for point in excitations), default=1.0)
    axes.set_ylim(0.0, highest * 1.3 if highest > 0.0 else 1.0)
    _legend(axes, "upper left")
    figure.tight_layout()
    return figure


@serialised
def observable_sweep_figure(
    ratios: FloatArray,
    values: FloatArray,
    label: str,
    reference: FloatArray | None = None,
    reference_label: str = "the infinite chain",
) -> Figure:
    r"""Draw one exact ground-state quantity against the field.

    The figure for *plot the ground-state energy against* :math:`h/J` and for *plot
    the magnetisation against* :math:`h/J`. One curve, the critical field marked,
    and -- where there is one -- the same quantity in the thermodynamic limit
    underneath it, because "how close is a chain of ten magnets to an infinite one?"
    is the question a finite-size figure exists to answer and it cannot be answered
    by a curve on its own.

    Args:
        ratios: The swept :math:`h/J` values.
        values: The quantity at each point.
        label: What the quantity is, for the vertical axis and the legend. Written
            in the notation the rest of the project uses -- ``$E_0/L$``,
            ``$\langle \sigma^x \rangle$`` -- so that the figure and the prose above
            it name the same thing.
        reference: The same quantity in the thermodynamic limit, or ``None``.
        reference_label: What to call the reference in the legend.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 3.0)
    axes = figure.subplots()
    style(axes)
    axes.plot(ratios, values, color=ACCENT, linewidth=1.8, label=label)
    if reference is not None:
        axes.plot(
            ratios,
            reference,
            color=MUTED,
            linestyle="--",
            linewidth=1.3,
            label=f"{label}, {reference_label}",
        )
    _mark_critical(axes)
    axes.set_xlabel("field over coupling, $h/J$", fontsize=9, color=MUTED)
    axes.set_ylabel(label, fontsize=9, color=MUTED)
    _legend(axes)
    figure.tight_layout()
    return figure


@serialised
def derivative_sweep_figure(
    ratios: FloatArray,
    quantity: FloatArray,
    first: FloatArray,
    second: FloatArray,
    labels: tuple[str, str, str],
) -> Figure:
    r"""Draw a quantity and its first two field derivatives, stacked.

    The figure for *plot it and its first and second derivatives*, and the stacking
    is the argument. The quantity itself is nearly featureless through the
    transition, its first derivative bends, and only the **second** has a sharp
    feature there -- three panels on a shared axis say that in one glance, while
    three separate figures make a reader hold two of them in their head.

    Every derivative drawn here is analytic. Nothing is finite-differenced, so
    nothing on the figure depends on how many points were sampled -- which is worth
    stating because a numerically differentiated second derivative is the classic
    way a plot acquires structure that is not in the physics.

    Args:
        ratios: The swept :math:`h/J` values.
        quantity: The quantity itself.
        first: Its first derivative with respect to the field.
        second: Its second derivative with respect to the field.
        labels: What to call the three panels, in order, in project notation.

    Returns:
        The figure.
    """
    figure = _styled_figure(7.4, 6.0)
    panels = figure.subplots(3, 1, sharex=True)
    for index, (axes, series, label) in enumerate(
        zip(panels, (quantity, first, second), labels, strict=True)
    ):
        style(axes)
        axes.plot(ratios, series, color=ACCENT if index == 0 else WARM, linewidth=1.7)
        axes.set_ylabel(label, fontsize=9, color=MUTED)
        _mark_critical(axes, label=index == 0)
        axes.axhline(0.0, color=MUTED, linewidth=0.7, alpha=0.4)
        if index == 0:
            _legend(axes)
    panels[-1].set_xlabel("field over coupling, $h/J$", fontsize=9, color=MUTED)
    figure.tight_layout()
    return figure
