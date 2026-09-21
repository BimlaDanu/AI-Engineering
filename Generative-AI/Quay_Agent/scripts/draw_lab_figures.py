r"""Draw every cartoon the Quay Lab page shows, as a PDF in ``reports/figures/``.

The Lab page explains this project to somebody who does not do physics, and it does
it with pictures rather than with tables of numbers. Those pictures are the part of
the project most likely to end up in a written report or a slide, and a picture that
can only be screenshotted out of a running Streamlit session is one that arrives at
the wrong resolution with a scrollbar in it.

So the same functions the page calls are called here and written as vector PDFs. It
is not a duplicate of the page -- :mod:`src.ui.figures` holds the drawing and neither
this script nor the page does any of its own. What this adds is a second caller, and
a second caller is what stops a figure from quietly depending on Streamlit.

The chain drawn is the project's default, not the one in anybody's sidebar: these are
illustrations of *what the pictures are*, and a reader comparing a report against the
repository should get the same file both times.

Run with ``make figures``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.figure_export import save_figure
from src.physics import quantumness
from src.physics.model import DEFAULT_SITES, TFIMSpec
from src.physics.quantum.ansatz import AnsatzSpec
from src.physics.quantum.variational_eigensolver import solve
from src.physics.reference import exact_diagonalisation, free_fermions
from src.ui import figures

SITES = DEFAULT_SITES
"""Chain length for every figure. The project default, so the files are stable."""

DEPTH = 2
"""Circuit layers. Two, because the point of the drawing is that a layer repeats."""

SPREAD_RATIOS: tuple[float, ...] = (0.2, 0.7, 1.0, 2.0)
"""The fields the spreading strip is drawn at. Matches the Lab page."""

GAP_POINTS = 121
"""Samples along the gap curve. Odd, so ``g = 1`` is hit exactly."""

MAX_GAP_RATIO = 2.5
"""How far past the critical point the gap curve runs."""

MAPPING_SLICES = 5
"""Imaginary-time slices in the classical-sheet drawing, ``2P + 1`` for ``P = 2``."""

RASTER_TOO: frozenset[str] = frozenset(
    {
        "lab-chain-terms",
        "lab-method-map",
        "lab-phase-diagram",
        "lab-ansatz-circuit",
        "lab-hybrid-loop",
    }
)
"""Figures that also get a PNG beside the PDF.

The list is closed by a rule rather than by taste: a figure is rastered here if
and only if ``README.md`` embeds it, because GitHub renders no PDF inline and a
project whose figures cannot be seen without cloning it has no figures. See
:mod:`src.figure_export` for why PDF is otherwise the only format, and what the
raster costs -- a circuit diagram enlarged to count its gates is the case a raster
serves worst, which is why the PDF stays the copy to read.

Listed in the order the README shows them, which is also the order the argument is
made in: where this model sits among the methods, what the chain does as its field
is turned up, what it looks like as a program, and what is done with that program.
"""

RACE_DEPTH = 3
"""Layers for the three-way start comparison.

Deeper than :data:`DEPTH`, which exists to show that a layer repeats. Three layers
is where the two informed starts separate on step count while still landing on the
same energy, which is the whole content of the figure.
"""


def main() -> None:
    """Draw every Lab figure and write it as a PDF.

    Prints each path as it is written, so a run that silently produced nothing is
    distinguishable from one that produced everything.
    """
    spec = TFIMSpec(n_sites=SITES, coupling=1.0, field=1.0, boundary="periodic")
    ansatz = AnsatzSpec(n_qubits=SITES, depth=DEPTH, boundary=spec.boundary)

    ratios = [MAX_GAP_RATIO * index / (GAP_POINTS - 1) for index in range(GAP_POINTS)]
    infinite = [free_fermions.gap_thermodynamic(spec.coupling, spec.coupling * r) for r in ratios]
    finite = [
        free_fermions.gap(TFIMSpec(SITES, spec.coupling, spec.coupling * r, spec.boundary))
        for r in ratios
    ]

    drawings = {
        # First, because it is the picture the README and the Lab page both open on:
        # two competing terms and no arrangement that satisfies both. Everything
        # below it is a way of settling that argument.
        "lab-chain-terms": figures.chain_terms_figure(
            SITES, spec.coupling, spec.field, periodic=spec.boundary == "periodic"
        ),
        "lab-phase-diagram": figures.phase_diagram_figure(
            spec, np.asarray(ratios), np.asarray(infinite), np.asarray(finite)
        ),
        "lab-ground-state-superposition": figures.superposition_figure(
            quantumness.superposition(spec)
        ),
        "lab-superposition-across-field": figures.spreading_figure(
            tuple(
                quantumness.superposition(
                    TFIMSpec(SITES, spec.coupling, spec.coupling * ratio, spec.boundary)
                )
                for ratio in SPREAD_RATIOS
            ),
            SPREAD_RATIOS,
        ),
        "lab-hybrid-loop": figures.hybrid_loop_figure(),
        "lab-ansatz-circuit": figures.circuit_figure(SITES, DEPTH, ansatz.bonds, ansatz.rounds),
        "lab-quantum-to-classical-lattice": figures.dual_lattice_figure(
            SITES, MAPPING_SLICES, periodic=spec.boundary == "periodic"
        ),
        "lab-method-map": figures.method_map_figure(),
        "lab-method-race": figures.method_race_figure(
            tuple(
                (
                    label,
                    tuple(
                        solve(
                            n_sites=SITES,
                            depth=RACE_DEPTH,
                            coupling=spec.coupling,
                            transverse_field=spec.field,
                            boundary=spec.boundary,
                            family=family,
                            initialisation=initialisation,
                        ).energy_history
                    ),
                )
                for label, family, initialisation in figures.RACE_STARTS
            ),
            exact_diagonalisation.ground_state_energy(spec),
        ),
    }
    for name, drawn in drawings.items():
        # The two figures `README.md` embeds get a raster copy as well, and only
        # those two -- see RASTER_TOO. Every other figure stays vector-only and is
        # reached by opening the file, which is the right trade for a figure nobody
        # has to see before deciding whether to run anything.
        print(save_figure(drawn, name, also_png=name in RASTER_TOO))


if __name__ == "__main__":
    main()
