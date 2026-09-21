r"""Draw where three machines stop being able to run this problem.

The figure answers the question a feasibility study exists to answer, for a reader
who does not work on quantum computers: **how big a problem, and how good an
answer, can each machine actually manage?** Three panels, left to right, each one
narrowing the claim the panel before it allowed.

**Left: what the wiring costs.** A chain of magnets is a row, and the interactions
it needs are between neighbours in that row. A machine whose qubits are also wired
in a row carries that for free. A machine wired in a sparser pattern does not, and
the difference is paid in operations that move information around instead of
computing with it. The panel shows the depth penalty as a ratio: one means free.
A ring is drawn as well as a segment, because a ring needs its two ends to
interact and no machine here wires them together -- it is the cheapest way to see
what connectivity is worth.

**Middle: what the noise leaves.** Every operation has a chance of going wrong, and
the chances compound. This panel is the fraction of the result that is still the
intended one, against how many layers the circuit has. The horizontal line is the
point below which more than half of what comes back is noise.

**Right: the bill for that.** Recovering a fixed accuracy from a damped signal costs
the inverse square of what survived. This panel turns the middle one into the
number that decides feasibility: how many times more measurements than a perfect
machine would need. It is drawn on a logarithmic axis because it is not a small
correction -- by depth eight it is two orders of magnitude.

The honest summary the figure is meant to produce: the ideal machine is flat in
every panel, and the two real ones run out of *signal* long before they run out of
*time*. That distinction is the actionable one, because a faster gate fixes a
coherence limit and does nothing at all for a fidelity limit.

Run with ``make figures``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.figure_export import save_figure, use_house_style
from src.hardware.devices import DEVICES, Device
from src.hardware.fidelity import USABLE_FIDELITY_FLOOR, estimate
from src.hardware.transpile import transpile
from src.physics.quantum.ansatz import AnsatzSpec

N_SITES = 12
"""Chain length every panel is drawn for.

Twelve because it is large enough that connectivity and noise both bite, and small
enough to sit on every machine here -- including the twenty-seven-qubit lattice,
whose longest unbroken run of qubits is twenty-one rather than twenty-seven.
"""

DEPTHS = tuple(range(1, 13))
"""Circuit depths swept. One to twelve, which is past the useful range on purpose."""

COLOURS = {
    "ideal": "#2B3A67",
    "linear": "#5B6CFF",
    "heavy-hex-27": "#E0803C",
}
"""One colour per machine, held constant across all three panels."""


def routing_overhead(device: Device, boundary: str) -> list[float]:
    """Depth penalty the wiring imposes, at each depth in the sweep.

    Args:
        device: The machine.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.

    Returns:
        One ratio per depth. One means the wiring cost nothing.
    """
    return [
        transpile(
            AnsatzSpec(n_qubits=N_SITES, depth=depth, boundary=boundary),  # type: ignore[arg-type]
            device,
        ).routing_overhead
        for depth in DEPTHS
    ]


def surviving(device: Device) -> list[float]:
    """Fraction of the signal left, at each depth in the sweep.

    Args:
        device: The machine.

    Returns:
        One fidelity per depth.
    """
    return [
        estimate(transpile(AnsatzSpec(n_qubits=N_SITES, depth=depth), device)).total
        for depth in DEPTHS
    ]


def draw_routing(axes: plt.Axes) -> None:
    """Draw what the wiring costs, for a segment and for a ring.

    Args:
        axes: The panel to draw into.
    """
    # The two real machines route a ring identically -- both lay it out folded along
    # a line of qubits -- so their dashed curves coincide exactly. Drawn at different
    # widths so the one underneath stays visible, since a hidden curve reads as a
    # machine that was not measured.
    for width, device in enumerate(DEVICES):
        colour = COLOURS[device.name]
        axes.plot(DEPTHS, routing_overhead(device, "open"), color=colour, label=device.name)
        axes.plot(
            DEPTHS,
            routing_overhead(device, "periodic"),
            color=colour,
            linestyle="--",
            linewidth=3.2 - 0.9 * width,
            alpha=0.85,
        )
    axes.axhline(1.0, color="0.6", linewidth=0.8, zorder=0)
    axes.set_xlabel("circuit layers")
    axes.set_ylabel("depth on the machine / depth in principle")
    axes.set_title("what the wiring costs")
    axes.text(
        0.97,
        0.5,
        "solid: a segment\ndashed: a ring",
        transform=axes.transAxes,
        ha="right",
        va="center",
        fontsize=8,
        color="0.35",
    )


def draw_fidelity(axes: plt.Axes) -> None:
    """Draw what survives the noise.

    Args:
        axes: The panel to draw into.
    """
    for device in DEVICES:
        axes.plot(DEPTHS, surviving(device), color=COLOURS[device.name], label=device.name)
    axes.axhline(USABLE_FIDELITY_FLOOR, color="0.6", linewidth=0.8, linestyle=":")
    axes.text(
        DEPTHS[-1],
        USABLE_FIDELITY_FLOOR + 0.02,
        "half the result is noise",
        ha="right",
        fontsize=8,
        color="0.35",
    )
    axes.set_ylim(0.0, 1.05)
    axes.set_xlabel("circuit layers")
    axes.set_ylabel("fraction of the result that is signal")
    axes.set_title("what the noise leaves")
    axes.legend(frameon=False, fontsize=8, loc="lower left")


def draw_shot_cost(axes: plt.Axes) -> None:
    """Draw the measurement bill the noise produces.

    Args:
        axes: The panel to draw into.
    """
    for device in DEVICES:
        axes.plot(
            DEPTHS,
            [1.0 / value**2 for value in surviving(device)],
            color=COLOURS[device.name],
            label=device.name,
        )
    axes.set_yscale("log")
    axes.set_xlabel("circuit layers")
    axes.set_ylabel("measurements, relative to a perfect machine")
    axes.set_title("what that costs in shots")


def main() -> None:
    """Draw all three panels and write the figure."""
    use_house_style()
    figure, panels = plt.subplots(1, 3, figsize=(12.0, 4.0))
    draw_routing(panels[0])
    draw_fidelity(panels[1])
    draw_shot_cost(panels[2])
    figure.suptitle(
        f"A {N_SITES}-magnet chain on three machines: connectivity, noise, and the bill",
        fontsize=11,
    )
    figure.tight_layout()
    # A PNG as well, because README.md embeds this one and GitHub renders no PDF
    # inline. The PDF stays the copy to read: three panels of curves do not
    # survive being enlarged.
    written = save_figure(figure, "device-limits", also_png=True)
    plt.close(figure)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
