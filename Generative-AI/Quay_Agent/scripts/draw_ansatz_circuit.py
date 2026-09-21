r"""Draw the alternating ansatz that both arms of this project share.

The figure exists to make one claim visible, because it is the claim the whole
comparison rests on: **the classical baseline and the quantum circuit are the
same variational family.** One layer applies the diagonal generator, the next
applies the transverse one, and the pair repeats :math:`P` times:

.. math::

    \prod_{p=P}^{1} e^{-\theta^{(2)}_p \hat H_{\rm field}}
                    e^{-\theta^{(1)}_p \hat H_{\rm diag}} |+\rangle^{\otimes L} .

Real angles give QAOA, run on a device. Imaginary ones give VITA, evaluated by
Monte Carlo on the dual classical lattice in
:mod:`src.physics.classical.variational_imaginary_time`. Nothing else about the
structure changes -- which is why "is the quantum version worth it?" is a
question about the machine rather than about the algorithm.

The drawing also carries the depth arithmetic that decides feasibility. The
:math:`\hat\sigma^z\hat\sigma^z` rotations on disjoint bonds commute and can run
at the same time, so the even bonds go first and the odd bonds second: two
:math:`\sigma^z\sigma^z` blocks per layer regardless of chain length, rather
than :math:`L - 1` of them in sequence. That is the difference between a circuit
whose depth grows with the chain and one whose depth does not, and it comes from
noticing a commutation structure rather than from buying hardware.

Drawn by hand rather than through a circuit library: this has to render before
Qiskit is installed, and the even/odd colouring is the point of the picture and
is not something a generic drawer would produce.

Run with ``make figures``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

from src.figure_export import save_figure, use_house_style

N_QUBITS = 5
"""Chain length shown. Odd, so that the even and odd bond sets differ in size."""

DEPTH = 2
"""Layers drawn. Two is the smallest number that shows the pattern repeating."""

EVEN_COLOUR = "#5B6CFF"
ODD_COLOUR = "#E0803C"
FIELD_COLOUR = "#2B3A67"


def draw_two_qubit_block(axes: plt.Axes, x: float, upper: int, lower: int, colour: str) -> None:
    r"""Draw one :math:`e^{-\theta \hat\sigma^z_i \hat\sigma^z_j}` rotation as a CNOT pair."""
    axes.plot([x, x], [upper, lower], color=colour, linewidth=1.6, zorder=2)
    axes.add_patch(Circle((x, upper), 0.09, facecolor=colour, edgecolor="none", zorder=4))
    axes.add_patch(Circle((x, lower), 0.09, facecolor=colour, edgecolor="none", zorder=4))


def draw_single_qubit_gate(axes: plt.Axes, x: float, wire: int, label: str, colour: str) -> None:
    """Draw a boxed single-qubit gate centred on a wire."""
    axes.add_patch(
        FancyBboxPatch(
            (x - 0.26, wire - 0.2),
            0.52,
            0.4,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor="white",
            edgecolor=colour,
            linewidth=1.4,
            zorder=3,
        )
    )
    axes.text(x, wire, label, ha="center", va="center", fontsize=7, color=colour, zorder=4)


def draw() -> plt.Figure:
    """Draw the full depth-``DEPTH`` ansatz on ``N_QUBITS`` wires."""
    use_house_style()
    figure, axes = plt.subplots(figsize=(7.2, 2.9))

    even_bonds = [(i, i + 1) for i in range(0, N_QUBITS - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, N_QUBITS - 1, 2)]

    column = 1.2
    columns: list[tuple[float, str]] = []
    for layer in range(DEPTH):
        for bonds, colour in ((even_bonds, EVEN_COLOUR), (odd_bonds, ODD_COLOUR)):
            for upper, lower in bonds:
                draw_two_qubit_block(axes, column, upper, lower, colour)
            columns.append((column, "zz"))
            column += 0.75
        for wire in range(N_QUBITS):
            draw_single_qubit_gate(axes, column, wire, r"$R_x$", FIELD_COLOUR)
        columns.append((column, "field"))
        # Above the wires, because the space below belongs to the legend.
        axes.text(
            column - 0.75,
            -0.45,
            f"layer {layer + 1}",
            ha="center",
            fontsize=8,
            color="#666666",
        )
        column += 1.1

    # Wires, and the |+> preparation that every layer builds on.
    for wire in range(N_QUBITS):
        axes.plot([0.0, column - 0.55], [wire, wire], color="#333333", linewidth=1.0, zorder=1)
        axes.text(-0.12, wire, r"$|0\rangle$", ha="right", va="center", fontsize=8)
        draw_single_qubit_gate(axes, 0.55, wire, "H", "#333333")

    axes.set_xlim(-0.75, column - 0.4)
    axes.set_ylim(-0.75, N_QUBITS - 0.05)
    axes.invert_yaxis()
    axes.axis("off")
    axes.set_title(
        f"One ansatz, two machines: depth $P={DEPTH}$ on $L={N_QUBITS}$ qubits", fontsize=9
    )

    axes.legend(
        handles=[
            Line2D([], [], color=EVEN_COLOUR, lw=2, label=r"even bonds  $\sigma^z\sigma^z$"),
            Line2D([], [], color=ODD_COLOUR, lw=2, label=r"odd bonds  $\sigma^z\sigma^z$"),
            Line2D([], [], color=FIELD_COLOUR, lw=2, label=r"transverse field  $\sigma^x$"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.06),
    )
    return figure


if __name__ == "__main__":
    figure = draw()
    print(f"wrote {save_figure(figure, 'ansatz-structure', also_png=True)}")
    plt.close(figure)
