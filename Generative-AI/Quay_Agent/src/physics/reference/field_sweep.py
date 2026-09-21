r"""Sweeping the field: the observable a single solve cannot show.

Most of what a reader wants to know about this model is not at a point but in a
curve. How the magnetisation turns on, where the gap closes, what the second
derivative of the energy does at the transition -- a solver called once answers
none of them.

At each field this evaluates the free-fermion closed form: energy density,
transverse magnetisation, their first and second field derivatives, and the
low-lying many-body levels. On a chain short enough to afford it, the same two
observables are evaluated again by exact diagonalisation and the difference
recorded, so a curve is verified point by point rather than being a cheaper class
of result that happens to be plotted.

Sealed, and reached anyway. This imports
:mod:`src.physics.reference.free_fermions`, so nothing under ``src/agent/`` may
import it and the architecture test proves nothing does. The agent uses
it through a callable the grader\'s bench lends to one tool at run time -- see
:func:`src.physics.registry.field_sweep_bench` and
:func:`src.agent.tools.field_sweep_tool`. Two things make that safe:

* the tool is offered only at the ``consult`` node, which fronts the two prose
  branches and is not on the feasibility branch, so the campaign that is graded
  against an exact answer structurally cannot call it;
* every call is recorded on the campaign state and shown to the reader.

Every derivative here is analytic. The first derivative of the energy density is
minus the magnetisation exactly, by Hellmann-Feynman, and the second and third
come from differentiating :math:`\epsilon(k)` in closed form. Nothing is
finite-differenced: a numerically differentiated curve depends on the point
spacing, so a coarser plot would change the physics it reported, and a numerical
second derivative is the classic place a figure acquires structure the model does
not have.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Literal, get_args

from src.logging_setup import get_logger
from src.physics.model import TFIMSpec
from src.physics.reference import exact_diagonalisation, free_fermions

_logger = get_logger("physics.field_sweep")

Curve = Literal[
    "spectrum",
    "energy",
    "magnetisation",
    "energy_derivatives",
    "magnetisation_derivatives",
]
r"""Which curve a caller asked to see.

Five, because these are the five different questions a reader actually asks about
this model as the field is turned up, and they want different pictures:

* ``spectrum`` -- the low-lying many-body levels :math:`E_n` against
  :math:`h/J`. The gap-closing question.
* ``energy`` -- the ground-state energy per site, :math:`E_0/L`.
* ``magnetisation`` -- :math:`\langle \sigma^x \rangle`, the order parameter of the
  polarised phase.
* ``energy_derivatives`` -- :math:`\partial (E_0/L)/\partial h` and
  :math:`\partial^2 (E_0/L)/\partial h^2`.
* ``magnetisation_derivatives`` -- :math:`\partial \langle \sigma^x
  \rangle/\partial h` and its second derivative.

The last two are kept apart from ``energy`` and ``magnetisation`` deliberately, and
apart from *each other* for a reason found by getting it wrong. The energy density
is featureless through the transition, its first derivative merely bends, and only
its **curvature** has a sharp feature there -- so a question about derivatives
answered with an energy curve has been answered with the one panel that shows
nothing. And by Hellmann-Feynman the energy's first derivative *is* minus the
magnetisation, so a question about *the magnetisation's* derivatives answered with
the energy's has been shown the right numbers under the wrong name.

Every quantity is computed whichever curve is asked for -- the whole sweep is
:math:`O(L)` per point -- so this selects what is **shown and described**, not what
is calculated.
"""

DEFAULT_POINTS = 41
"""Field values sampled unless the caller asks for more or fewer.

Enough that the susceptibility peak and the gap minimum land within a couple of per
cent of their true positions, and cheap enough that a question does not feel like a
job submission: a 41-point sweep of a 12-site ring is a few milliseconds.
"""

MAX_POINTS = 201
"""Most points any one sweep will take. A finer curve tells a reader nothing new."""

DEFAULT_RATIO_MAX = 2.0
r"""Top of the swept range, in units of :math:`J`.

The transition sits at :math:`h/J = 1`, so the default range runs from the fully
ordered chain to twice the critical field: both phases and the crossover between
them, with the interesting part in the middle rather than at an edge.
"""

MAX_RATIO_MAX = 8.0
r"""Highest :math:`h/J` a sweep will run to.

Past this the chain is a set of independent spins in a field and every curve is
flat, so a wider range buys resolution loss in the part that matters.
"""

MAX_SWEEP_SITES = 4096
"""Longest chain this module will sweep.

The closed form is ``O(L)`` per point, so the limit is not the arithmetic -- it is
that a curve at four thousand sites and a curve in the thermodynamic limit are the
same picture, and the second is one function call away in
:func:`src.physics.reference.free_fermions.energy_density_thermodynamic`.
"""

MAX_CROSSCHECK_SITES = 10
r"""Longest chain each point is *also* solved by exact diagonalisation.

The cross-check is what makes a curve evidence rather than output, so it is on by
default -- but it costs a :math:`2^L` eigenproblem at every point, and 41 points at
:math:`L = 12` is a wait a reader did not ask for. At :math:`L = 10` the whole
sweep is well under a second.

Above this the sweep still runs, from the closed form alone, and
:attr:`FieldSweep.detail` says so. A curve that quietly stopped being checked would
be the worse failure.
"""

DEFAULT_LEVELS = free_fermions.DEFAULT_LEVELS
"""Levels drawn on a spectrum sweep. Deferred to the solver's own default."""

MAX_LATTICE_SWEEP_SITES = 16
"""Largest two-dimensional lattice this module will sweep.

A very different limit from :data:`MAX_SWEEP_SITES`, and for a different reason. A
line is swept by a closed form costing ``O(L)`` per point, so four thousand sites is
free; a lattice has no closed form and every point is a ``2**L`` eigenproblem.
Measured on this machine, sparse diagonalisation of a ``4 x 4`` square costs about
0.2 s, so a 21-point sweep is a few seconds -- a wait, but a wait for something.
Twenty sites would be a minute per curve, which is a job submission rather than an
answer to a question.

Sixteen is also the size worth having: a ``4 x 4`` square is the smallest
two-dimensional cluster with an interior site, so it is the smallest lattice whose
curve is about a lattice rather than about its own boundary.
"""

MAX_LATTICE_CROSSCHECK_SITES = 12
"""Largest lattice each point is *also* assembled a second way.

Lower than :data:`MAX_LATTICE_SWEEP_SITES` because the second route builds the
Hamiltonian as a Kronecker product rather than by acting on basis integers, which
is the more expensive of the two constructions. Twelve sites covers ``2 x 6``,
``3 x 4`` and the triangular clusters; a ``4 x 4`` square is swept once per point
and :attr:`FieldSweep.detail` says so.
"""

LATTICE_CURVES: tuple[Curve, ...] = ("spectrum", "energy", "magnetisation")
r"""The curves a two-dimensional sweep can supply, and it is not all of them.

Three of the five, and the two that are missing are missing for a stated reason
rather than because nobody wrote them.

**What a lattice can still give exactly.** The energy and the magnetisation come
straight out of a diagonalisation, the spectrum is the lowest few eigenvalues, and
the energy's *first* derivative is minus the magnetisation by Hellmann-Feynman --
which is a statement about any Hamiltonian depending linearly on a parameter, so it
holds on a triangular lattice exactly as it does on a line. Those are all reported.

**What it cannot.** The second and third derivatives on a line come from
differentiating the dispersion :math:`\epsilon(k)` in closed form, and a lattice has
no dispersion to differentiate. The exact alternative is a Kubo sum over *every*
excited state, which is a harder problem than the one being plotted. The remaining
option is a finite difference, and this module declines it: a numerically
differentiated curve depends on the point spacing, so asking for a coarser plot
would change the physics it reported. Requesting a derivatives curve on a lattice
therefore gets the curves that can be computed plus a sentence in
:attr:`FieldSweep.detail` saying which one could not, rather than a plausible plot
of a quantity nobody computed.
"""


@dataclass(frozen=True, slots=True)
class SweepPoint:
    r"""One field value, and everything both methods said about it.

    Attributes:
        ratio: The control parameter :math:`h/J`.
        field: The field itself, :math:`h`.
        energy_density: :math:`E_0/L`, the quantity comparable across lengths.
        magnetisation: :math:`\langle \sigma^x \rangle`, in :math:`[0, 1]`.
        energy_slope: :math:`\partial (E_0/L)/\partial h`. Analytic: minus
            :attr:`magnetisation` exactly, by Hellmann-Feynman, and carried as its
            own field so that a derivatives plot is drawing the quantity it names.
        energy_curvature: :math:`\partial^2 (E_0/L)/\partial h^2`, closed form.
        magnetisation_slope: :math:`\partial \langle \sigma^x \rangle/\partial h`,
            the transverse susceptibility. Minus :attr:`energy_curvature`.
        magnetisation_curvature: :math:`\partial^2 \langle \sigma^x
            \rangle/\partial h^2` -- the energy density's third derivative, and the
            one quantity here with no other name.
        levels: The low-lying many-body energies, ascending, absolute rather than
            measured from the ground state. Empty when no spectrum was asked for.
        thermodynamic_energy_density: :math:`E_0/L` of the *infinite* chain at the
            same field. Carried at every point because "how far is this finite
            chain from the thermodynamic limit?" is the question a finite-size
            figure exists to answer.
        thermodynamic_gap: :math:`2\lvert J - h \rvert`, the infinite chain's gap.
            Zero exactly at :math:`h = J`; the finite ring's own gap is not.
        disagreement: Largest absolute difference between the closed form and exact
            diagonalisation at this point, over the energy density and the
            magnetisation. ``None`` when only one method ran here.
    """

    ratio: float
    field: float
    energy_density: float
    magnetisation: float
    energy_slope: float
    energy_curvature: float
    magnetisation_slope: float
    magnetisation_curvature: float
    thermodynamic_energy_density: float
    thermodynamic_gap: float
    levels: tuple[float, ...] = ()
    disagreement: float | None = None

    @property
    def excitations(self) -> tuple[float, ...]:
        r"""The levels measured from the ground state, :math:`E_n - E_0`.

        Returns:
            One entry per computed level, opening with the ``0.0`` of the ground
            state itself. Empty when no spectrum was computed. This is what a
            spectrum figure plots: absolute energies slide down the page as the
            field rises and hide the structure the question is about.
        """
        if not self.levels:
            return ()
        return tuple(level - self.levels[0] for level in self.levels)

    @property
    def gap(self) -> float | None:
        r"""The spacing to the first excited state, :math:`E_1 - E_0`.

        Returns:
            The gap, or ``None`` when the spectrum was not computed here.

            Named rather than left as an index because it is *the* quantity of the
            transition -- and because below :math:`h = J` on a ring it is not the
            number most readers expect: the first excited state there is the
            symmetry-broken partner of the ground state, split from it by an amount
            that falls exponentially with :math:`L`. That near-zero is physics, not
            a numerical artefact, and it is why :attr:`pair_gap` exists beside it.
        """
        return self.levels[1] - self.levels[0] if len(self.levels) > 1 else None


@dataclass(frozen=True, slots=True)
class FieldSweep:
    """A curve, what it was asked for, and its verification.

    Attributes:
        spec: The chain that was swept. Its own ``field`` is where the question was
            asked; the sweep varies it and leaves everything else fixed.
        points: The curve, in increasing field.
        curves: What the caller asked to see, deduplicated and in canonical order.
        methods: Names of the solvers that ran at *every* point.
        detail: One sentence on what happened, populated whether or not a curve
            came back.
        levels_requested: How many many-body levels were asked for.
    """

    spec: TFIMSpec | None
    points: tuple[SweepPoint, ...]
    curves: tuple[Curve, ...]
    methods: tuple[str, ...]
    detail: str
    levels_requested: int = 0

    @property
    def ok(self) -> bool:
        """Whether a curve was produced at all."""
        return bool(self.points)

    @property
    def shows_spectrum(self) -> bool:
        """Whether a spectrum was asked for and every point carries one."""
        return "spectrum" in self.curves and self.levels_drawn > 1

    @property
    def levels_drawn(self) -> int:
        """How many levels every point has in common.

        Returns:
            The smallest level count across the curve, so a line drawn through the
            spectrum has a value at every field. Zero when no spectrum was
            computed. All-or-nothing on purpose: a spectrum plot with holes in it
            is not a spectrum plot.
        """
        if not self.points:
            return 0
        return min(len(point.levels) for point in self.points)

    @property
    def max_disagreement(self) -> float | None:
        """Worst disagreement between the two methods anywhere on the curve.

        Returns:
            The largest gap, or ``None`` when no point had two methods to compare.
            One number for the whole curve, because a reader deciding whether to
            trust the plot is asking about its worst point rather than its average.
        """
        gaps = [point.disagreement for point in self.points if point.disagreement is not None]
        return max(gaps) if gaps else None

    @property
    def is_corroborated(self) -> bool:
        """Whether two independent methods agreed at every single point.

        Returns:
            ``True`` only when every point was computed twice and every comparison
            came out at floating-point noise. A curve with one unchecked point is
            not a checked curve.
        """
        if not self.points or any(point.disagreement is None for point in self.points):
            return False
        worst = self.max_disagreement
        return worst is not None and worst < 1e-9

    @property
    def critical_gap(self) -> tuple[float, float] | None:
        r"""The gap at the sampled field nearest :math:`h = J`.

        Returns:
            ``(ratio, gap)``, or ``None`` without a spectrum.

            This rather than the *smallest* gap on the curve, which would be a
            misleading number on a ferromagnetic ring: the two lowest levels are
            all but degenerate throughout the ordered phase, so the minimum always
            sits at the left-hand edge and says nothing about criticality. What
            carries the physics is the gap at :math:`h/J = 1` -- small, non-zero,
            and shrinking as :math:`L` grows, which is how a finite calculation
            sees a phase transition at all.
        """
        found = [(point.ratio, point.gap) for point in self.points if point.gap is not None]
        if not found:
            return None
        where, size = min(found, key=lambda pair: abs(pair[0] - 1.0))
        return where, size

    @property
    def peak_susceptibility(self) -> tuple[float, float] | None:
        r"""Where :math:`\partial \langle \sigma^x \rangle/\partial h` is largest.

        Returns:
            ``(ratio, value)`` at the steepest rise of the magnetisation, or
            ``None`` on an empty sweep. This is the finite chain's *own* answer to
            "where is the transition?" -- computed rather than assumed, which is
            why it is worth showing next to the exact :math:`h/J = 1` of the
            infinite chain rather than instead of it.

            A finite ring peaks **below** the critical field and climbs towards it
            from there: on a fine grid the peak sits at :math:`h/J = 0.835` for
            :math:`L = 6`, ``0.949`` at :math:`L = 12`, ``0.994`` at :math:`L = 40`
            and ``0.999`` at :math:`L = 100`, while its height grows without
            bound -- ``0.83``, ``0.99``, ``1.34``, ``1.63``. Both halves of that
            are the transition announcing itself: the feature sharpens *and*
            migrates to :math:`h/J = 1`, which is the only honest way a chain of
            eight magnets can point at a singularity it does not have.
        """
        if not self.points:
            return None
        best = max(self.points, key=lambda point: point.magnetisation_slope)
        return best.ratio, best.magnetisation_slope

    @property
    def sharpest_curvature(self) -> tuple[float, float] | None:
        r"""Where the energy density bends most sharply, and by how much.

        Returns:
            ``(ratio, value)`` at the most negative
            :math:`\partial^2 (E_0/L)/\partial h^2`, or ``None`` on an empty sweep.

            The derivatives question's answer in one number, and the same feature
            as :attr:`peak_susceptibility` seen from the other side -- the two are
            one quantity up to a sign, which is why a figure showing both is
            showing a reader a check rather than two results.
        """
        if not self.points:
            return None
        deepest = min(self.points, key=lambda point: point.energy_curvature)
        return deepest.ratio, deepest.energy_curvature

    def unavailable(self) -> str | None:
        """Say why a requested curve is missing, when one is.

        Returns:
            One sentence naming what is absent and why, or ``None`` when the sweep
            produced everything it was asked for. A sweep that succeeded at a
            different question than the one it was asked has to say so out loud:
            the alternative is a reader getting the header for one quantity and the
            numbers for another.
        """
        if not self.points:
            return None
        if "spectrum" in self.curves and not self.shows_spectrum:
            return (
                "the low-lying spectrum: fewer than two levels were computed, so "
                "there is nothing to plot against the ground state"
            )
        return None

    def table(self) -> str:
        r"""Render the curve as compact text for a language model to write from.

        Thinned to a readable number of rows, with one column per quantity that was
        asked for and none of the others -- a narrator writes about what it is
        shown, so a table carrying every column answers a question about the
        magnetisation with a paragraph about the gap.

        The summary lines below the rows are not decoration. A model may only use
        numbers it was given, so every quantity an answer is expected to *discuss*
        -- where the susceptibility peaks, what the gap is at the critical field,
        how far the finite chain sits from the thermodynamic limit -- has to appear
        here explicitly rather than be inferable from the rows.

        Returns:
            The table, or an empty string when nothing was computed.
        """
        if not self.points or self.spec is None:
            return ""
        wanted = set(self.curves)
        show_energy = bool(wanted & {"energy", "energy_derivatives"})
        show_magnetisation = bool(wanted & {"magnetisation", "magnetisation_derivatives"})
        show_energy_derivatives = "energy_derivatives" in wanted
        show_magnetisation_derivatives = "magnetisation_derivatives" in wanted
        show_gap = self.shows_spectrum
        step = max(1, len(self.points) // 13)
        rows: list[str] = []
        for point in self.points[::step]:
            cells = [f"  h/J={point.ratio:.3f}"]
            if show_energy:
                cells.append(f"E0/L={_shown(point.energy_density):.6f}")
            if show_magnetisation:
                # ASCII rather than a Greek letter: RUF001 rejects the character,
                # and this string is what a narrator copies its notation from.
                cells.append(f"<sigma^x>={_shown(point.magnetisation):.6f}")
            if show_energy_derivatives:
                cells.append(f"d(E0/L)/dh={_shown(point.energy_slope):.6f}")
                cells.append(f"d2(E0/L)/dh2={_shown(point.energy_curvature):.6f}")
            if show_magnetisation_derivatives:
                cells.append(f"d<sigma^x>/dh={_shown(point.magnetisation_slope):.6f}")
                cells.append(f"d2<sigma^x>/dh2={_shown(point.magnetisation_curvature):.6f}")
            if show_gap and point.gap is not None:
                cells.append(f"E1-E0={point.gap:.6f}")
                cells.append(
                    "levels-above-ground="
                    + ",".join(f"{value:.4f}" for value in point.excitations[1:])
                )
            rows.append("  ".join(cells))
        lines = [
            f"EXACT FIELD SWEEP of {self.spec.label()}, computed by {' and '.join(self.methods)}.",
            f"{len(self.points)} field values, h/J from {self.points[0].ratio:.2f} to "
            f"{self.points[-1].ratio:.2f}. Asked for: {', '.join(self.curves)}.",
        ]
        # The named features come *before* the rows, and the order is load-bearing.
        # A tool result is clipped to fit a context window, so whatever sits last is
        # what gets cut -- and these lines are the ones an answer has to be able to
        # quote. The rows are the evidence behind them and lose the least by being
        # trimmed from the end.
        lines.extend(
            self._summary_lines(
                show_energy_derivatives=show_energy_derivatives,
                show_magnetisation_derivatives=show_magnetisation_derivatives,
                show_magnetisation=show_magnetisation,
                show_gap=show_gap,
            )
        )
        missing = self.unavailable()
        if missing is not None:
            lines.append(f"  NOT COMPUTED: {missing}")
        lines.append("The curve itself, thinned to a readable number of rows:")
        lines.extend(rows)
        return "\n".join(lines)

    def _summary_lines(
        self,
        show_energy_derivatives: bool,
        show_magnetisation_derivatives: bool,
        show_magnetisation: bool,
        show_gap: bool,
    ) -> list[str]:
        """Compose the lines under the rows that name the features by number.

        Args:
            show_energy_derivatives: Whether the energy's derivatives were asked for.
            show_magnetisation_derivatives: Whether the magnetisation's were.
            show_magnetisation: Whether the magnetisation itself was asked for.
            show_gap: Whether a spectrum was computed.

        Returns:
            Zero or more summary lines, each a complete sentence.
        """
        lines: list[str] = []
        if show_magnetisation:
            peak = self.peak_susceptibility
            if peak is not None:
                lines.append(
                    f"  the magnetisation rises fastest at h/J = {peak[0]:.3f}, where "
                    f"d<sigma^x>/dh = {peak[1]:.6f}. The infinite chain's transition is "
                    f"at h/J = 1 exactly; a finite ring peaks just below it and both "
                    f"sharpens and migrates towards it as L grows"
                )
        if show_energy_derivatives:
            bend = self.sharpest_curvature
            if bend is not None:
                lines.append(
                    f"  d2(E0/L)/dh2 is most negative at h/J = {bend[0]:.3f}, reaching "
                    f"{bend[1]:.6f}: the energy density itself is featureless through "
                    f"the transition, its first derivative only bends, and the sharp "
                    f"feature is in the curvature alone"
                )
        if show_energy_derivatives or show_magnetisation_derivatives:
            # Said for *either* set of derivatives, and that is a correction. It was
            # tied to the energy's alone, so a question about the magnetisation's
            # derivatives -- which is half the questions asked here -- got the numbers
            # with nothing saying they were exact or how the two curves are related.
            lines.append(
                "  every derivative here is analytic, not finite-differenced: "
                "d(E0/L)/dh is minus <sigma^x> exactly by Hellmann-Feynman, and "
                "d<sigma^x>/dh is minus d2(E0/L)/dh2 for the same reason, so the two "
                "sets are one physics with a sign between them. The rest come from "
                "differentiating the dispersion in closed form"
            )
        if show_gap:
            critical = self.critical_gap
            if critical is not None:
                lines.append(
                    f"  at h/J = {critical[0]:.3f} the gap E1-E0 is {critical[1]:.6f}, "
                    f"against 2|J-h| = 0 for the infinite chain. A finite ring has no "
                    f"gapless point: the momentum that would close the gap, k = 0, is "
                    f"not one the ring allows"
                )
            lines.append(
                f"  {self.levels_drawn} many-body levels were computed at every point "
                f"from the free-fermion solution, both fermion-parity sectors included. "
                f"Below h/J = 1 the two lowest are the symmetry-broken pair and their "
                f"splitting falls exponentially with L, which is why E1-E0 is near zero "
                f"there rather than large"
            )
        last = self.points[-1]
        first_ratio = self.points[0].ratio
        lines.append(
            f"  finite-size check: at h/J = {last.ratio:.3f} this chain's E0/L is "
            f"{last.energy_density:.6f} against {last.thermodynamic_energy_density:.6f} "
            f"for the infinite chain (swept from h/J = {first_ratio:.2f})"
        )
        worst = self.max_disagreement
        if worst is not None:
            lines.append(
                f"  cross-check: the closed form and exact diagonalisation were run "
                f"independently at every point and agree to {worst:.1e}"
            )
        else:
            lines.append(
                f"  cross-check: not run -- diagonalising every point is limited to "
                f"L = {MAX_CROSSCHECK_SITES}, so this curve is the closed form alone"
            )
        return lines

    def payload(self) -> dict[str, Any]:
        """Render the sweep as plain JSON for a tool result.

        Primitives only, and no project dataclasses: this is about to be serialised
        into a model's context window, where a type rendering as
        ``<SweepPoint object at 0x...>`` is a silent failure costing a whole turn.

        Returns:
            The table a narrator writes from, the curve as parallel arrays a caller
            can plot, and the provenance of both. Rounded to six decimals -- more
            digits than any answer quotes, and fewer than doubles print.
        """
        if not self.ok or self.spec is None:
            return {"error": self.detail}
        return {
            # "problem", not "chain". The key reaches a language model, and calling
            # a 4x4 square lattice a chain in the one field that names what was
            # solved is how an answer comes to describe the wrong problem in
            # confident prose.
            "problem": self.spec.label(),
            "shape": self.spec.lattice.in_words(),
            "geometry": self.spec.geometry,
            "n_sites": self.spec.n_sites,
            "coupling": self.spec.coupling,
            "boundary": self.spec.boundary,
            "curves_requested": list(self.curves),
            "methods": list(self.methods),
            "cross_checked": self.is_corroborated,
            "worst_disagreement": self.max_disagreement,
            "table": self.table(),
            "ratio": [round(point.ratio, 6) for point in self.points],
            "energy_density": [round(_shown(point.energy_density), 6) for point in self.points],
            # The infinite chain at the same fields. Carried beside the finite curve
            # because "how close is a chain of ten magnets to an infinite one?" is the
            # question a finite-size figure exists to answer, and a curve on its own
            # cannot answer it.
            "thermodynamic_energy_density": [
                round(point.thermodynamic_energy_density, 6) for point in self.points
            ],
            "thermodynamic_gap": [round(point.thermodynamic_gap, 6) for point in self.points],
            "magnetisation": [round(_shown(point.magnetisation), 6) for point in self.points],
            "energy_slope": [round(_shown(point.energy_slope), 6) for point in self.points],
            "energy_curvature": [round(_shown(point.energy_curvature), 6) for point in self.points],
            "magnetisation_slope": [
                round(_shown(point.magnetisation_slope), 6) for point in self.points
            ],
            "magnetisation_curvature": [
                round(_shown(point.magnetisation_curvature), 6) for point in self.points
            ],
            "excitations": [
                [round(value, 6) for value in point.excitations] for point in self.points
            ],
            "detail": self.detail,
        }

    def explain(self) -> str:
        """Say what the sweep did, in one line for the campaign trail.

        Returns:
            The point count, the methods, and the worst disagreement between them --
            or the reason nothing ran.
        """
        if not self.points:
            return f"no field sweep: {self.detail}"
        worst = self.max_disagreement
        agreement = "one method only" if worst is None else f"methods agree to {worst:.1e}"
        asked = ", ".join(self.curves)
        return (
            f"swept {len(self.points)} field values for {asked} "
            f"with {' and '.join(self.methods)} ({agreement})"
        )


def _shown(value: float) -> float:
    """Round a value that is zero to the printed precision onto an actual zero.

    The magnetisation at ``h = 0`` is a cancelling sum, so it lands on ``-1e-17``
    and prints as ``-0.000000``. A minus sign on a quantity a reader has just been
    told is zero reads as a bug in the physics, and a narrator writing from the
    table repeats it.

    Args:
        value: The number about to be printed at six decimals.

    Returns:
        ``0.0`` when the value cannot be distinguished from it at that precision,
        and the value itself otherwise.
    """
    return 0.0 if abs(value) < 5e-7 else value


def _canonical_curves(requested: object) -> tuple[Curve, ...]:
    """Normalise what was asked for into a deduplicated, ordered tuple.

    Args:
        requested: What the caller asked for: a single name, a sequence of names,
            or ``None`` for the default. Typed loosely because the caller on the
            other side of this is a language model's tool argument, which arrives
            as whatever JSON it produced.

    Returns:
        The recognised curve names, in :data:`Curve`'s own order so that two
        callers asking for the same set in different orders get the same figure.
        Falls back to ``("energy", "magnetisation")`` when nothing recognisable was
        asked for -- the pair that answers "what does the ground state do as the
        field rises", which is the question behind most of the others.
    """
    order = get_args(Curve)
    # A string is iterable, so it has to be caught before the iterable branch --
    # otherwise "spectrum" arrives as eight one-letter curve names and matches none.
    if isinstance(requested, str):
        names: list[str] = [requested]
    elif isinstance(requested, Iterable):
        names = [str(item) for item in requested]
    else:
        names = []
    kept = tuple(name for name in order if name in set(names))
    return kept or ("energy", "magnetisation")


def _point(
    spec: TFIMSpec,
    ratio: float,
    levels: int,
    cross_check: bool,
) -> SweepPoint:
    """Evaluate every observable at one field.

    Args:
        spec: The chain, at this field.
        ratio: The control parameter :math:`h/J`, carried through for the curve.
        levels: How many many-body levels to compute; zero for none.
        cross_check: Whether to solve this point a second time by exact
            diagonalisation and record the disagreement.

    Returns:
        The point.
    """
    magnetisation = free_fermions.transverse_magnetisation(spec)
    curvature = free_fermions.energy_density_curvature(spec)
    density = free_fermions.energy_density(spec)
    computed = (
        tuple(float(value) for value in free_fermions.low_lying_levels(spec, count=levels))
        if levels
        else ()
    )
    disagreement: float | None = None
    if cross_check:
        result = exact_diagonalisation.solve(spec)
        disagreement = max(
            abs(density - result.energy_density),
            abs(magnetisation - result.transverse_magnetisation),
        )
    return SweepPoint(
        ratio=ratio,
        field=spec.field,
        energy_density=density,
        magnetisation=magnetisation,
        # Hellmann-Feynman, exactly rather than approximately: the derivative of the
        # energy density with respect to the field *is* minus the magnetisation.
        energy_slope=-magnetisation,
        energy_curvature=curvature,
        magnetisation_slope=-curvature,
        magnetisation_curvature=free_fermions.magnetisation_curvature(spec),
        thermodynamic_energy_density=free_fermions.energy_density_thermodynamic(
            spec.coupling, spec.field
        ),
        thermodynamic_gap=free_fermions.gap_thermodynamic(spec.coupling, spec.field),
        levels=computed,
        disagreement=disagreement,
    )


def _lattice_point(
    spec: TFIMSpec,
    ratio: float,
    levels: int,
    cross_check: bool,
) -> SweepPoint:
    r"""Evaluate one field on a two-dimensional lattice, by diagonalisation.

    The counterpart of :func:`_point` for a shape with no closed form. Everything
    it reports is exact for the finite lattice; everything it cannot compute is
    :data:`math.nan` rather than a finite difference, and the curves that would have
    plotted those numbers are not offered -- see :data:`LATTICE_CURVES`.

    Args:
        spec: The problem, at this field. Its geometry supplies the bonds.
        ratio: The control parameter :math:`h/J`, carried through for the curve.
        levels: How many many-body levels to compute; zero for none.
        cross_check: Whether to assemble the same Hamiltonian a second way and
            record the disagreement.

    Returns:
        The point. Its ``energy_curvature``, ``magnetisation_slope``,
        ``magnetisation_curvature`` and both thermodynamic fields are ``nan``: the
        first three have no closed form here, and the last two describe an infinite
        *chain*, which is not the limit this lattice approaches.
    """
    result = exact_diagonalisation.solve(spec)
    computed: tuple[float, ...] = ()
    if levels:
        computed = tuple(float(value) for value in exact_diagonalisation.low_levels(spec, levels))
    disagreement: float | None = None
    if cross_check:
        disagreement = _second_opinion(spec, result.energy_density)
    return SweepPoint(
        ratio=ratio,
        field=spec.field,
        energy_density=result.energy_density,
        magnetisation=result.transverse_magnetisation,
        # Hellmann-Feynman holds on any graph: the Hamiltonian depends linearly on
        # h, so the energy density's field derivative *is* minus the magnetisation.
        # This is the one derivative a lattice gets exactly.
        energy_slope=-result.transverse_magnetisation,
        energy_curvature=math.nan,
        magnetisation_slope=math.nan,
        magnetisation_curvature=math.nan,
        thermodynamic_energy_density=math.nan,
        thermodynamic_gap=math.nan,
        levels=computed,
        disagreement=disagreement,
    )


def _second_opinion(spec: TFIMSpec, energy_density: float) -> float | None:
    """Rebuild one lattice point's Hamiltonian by a different algebra and compare.

    A line's curve is corroborated by having been computed two ways that share no
    code -- a closed form and a matrix. A lattice has no closed form, so the second
    opinion has to come from somewhere else, and the honest place is the *other*
    assembly of the same operator:
    :meth:`src.physics.quantum.hamiltonians.PauliSum.to_matrix` builds it as a
    Kronecker product of two-by-two matrices, where
    :func:`src.physics.reference.exact_diagonalisation.hamiltonian` builds it by
    acting on basis integers with an XOR. Those share no bit convention and no code,
    so agreement between them is evidence and not a tautology.

    The import is local rather than at module scope. It reaches across the seal in
    the harmless direction -- a reference module reading the agent's code, not the
    reverse -- but keeping it inside the function means the sealed module's import
    graph does not grow for the benefit of a path most sweeps never take.

    Args:
        spec: The problem, at one field.
        energy_density: What the first route reported, to compare against.

    Returns:
        The absolute difference in energy density, or ``None`` if the second route
        could not run -- in which case the point is honestly recorded as unchecked
        rather than as agreeing with itself.
    """
    from scipy.sparse.linalg import eigsh

    from src.physics.quantum.hamiltonians import ising_chain

    try:
        operator = ising_chain(
            spec.n_sites,
            coupling=spec.coupling,
            transverse_field=spec.field,
            lattice=spec.lattice,
        )
        matrix = operator.to_matrix()
        lowest = float(eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0])
    except Exception as error:  # an unchecked point is better than a false check
        _logger.warning(
            "lattice_second_opinion_failed",
            extra={"error_type": type(error).__name__, "field": round(spec.field, 4)},
        )
        return None
    return abs(energy_density - lowest / spec.n_sites)


def sweep_field(
    base: TFIMSpec,
    curves: object = None,
    points: int = DEFAULT_POINTS,
    ratio_max: float = DEFAULT_RATIO_MAX,
    levels: int = DEFAULT_LEVELS,
    cross_check_sites: int = MAX_CROSSCHECK_SITES,
) -> FieldSweep:
    r"""Trace the field range and record what the caller asked to see.

    Args:
        base: The chain to sweep. Its coupling, length and boundary are held fixed;
            only the field varies, so the sweep is a curve in :math:`h/J`.
        curves: Which curves the question was about. See :data:`Curve`. ``None``
            takes the ground-state pair.
        points: How many field values to sample, held between 3 and
            :data:`MAX_POINTS`. Three is the fewest that has an interior point, so
            a smaller request is raised to it rather than refused.
        ratio_max: Top of the range in units of :math:`J`, held between 0.5 and
            :data:`MAX_RATIO_MAX`.
        levels: How many many-body levels to compute per point. Ignored unless a
            spectrum was asked for, since it is the only curve that draws them.
        cross_check_sites: Longest chain each point is *also* diagonalised on.
            Lowering it does not coarsen the curve -- it drops the independent
            check, and the result says so rather than going quiet.

    Returns:
        The sweep. Never raises: this runs behind a tool call, where an exception
        would discard an answer that was otherwise complete, so a chain no method
        can handle comes back as a :class:`FieldSweep` carrying the reason in
        :attr:`~FieldSweep.detail` and no points.

    Examples:
        The ordered chain has no transverse magnetisation and the polarised one is
        nearly saturated:

        >>> from src.physics.model import TFIMSpec
        >>> sweep = sweep_field(TFIMSpec(n_sites=8), points=21)
        >>> sweep.ok and sweep.is_corroborated
        True
        >>> abs(sweep.points[0].magnetisation) < 1e-12
        True
        >>> sweep.points[-1].magnetisation > 0.9
        True

        The susceptibility peaks just *below* the critical field of the infinite
        chain and climbs towards it as the ring grows -- measured on this machine at
        ``h/J`` = 0.85, 0.90, 0.95, 0.95 for ``L`` = 6, 8, 10, 12. An earlier version
        of this docstring said "just above", which is the wrong direction and was
        never caught because no doctest in this project was being run:

        >>> peak, _ = sweep.peak_susceptibility
        >>> 0.8 <= peak <= 1.0
        True

        And a spectrum sweep carries the levels at every point:

        >>> spectrum = sweep_field(TFIMSpec(n_sites=8), curves="spectrum", points=11)
        >>> spectrum.shows_spectrum
        True
    """
    wanted = _canonical_curves(curves)
    if not base.is_one_dimensional:
        return _sweep_lattice(base, wanted, points, ratio_max, levels)
    reason = free_fermions.unsupported_reason(base)
    if reason is not None:
        return FieldSweep(
            spec=None,
            points=(),
            curves=wanted,
            methods=(),
            detail=(
                f"the closed-form solution does not apply to {base.label()}: {reason}. "
                "A field sweep needs one exact solve per point, and this is the only "
                "method cheap enough to supply them"
            ),
        )
    if base.n_sites > MAX_SWEEP_SITES:
        return FieldSweep(
            spec=None,
            points=(),
            curves=wanted,
            methods=(),
            detail=(
                f"L = {base.n_sites} is above the sweep limit of {MAX_SWEEP_SITES}; at "
                "that length the finite chain and the infinite one draw the same curve"
            ),
        )
    wanted_points = max(3, min(int(points), MAX_POINTS))
    top = min(max(float(ratio_max), 0.5), MAX_RATIO_MAX)
    cross_check = base.n_sites <= cross_check_sites
    methods = (free_fermions.METHOD_NAME,) + (
        (exact_diagonalisation.METHOD_NAME,) if cross_check else ()
    )
    per_point = min(max(int(levels), 2), free_fermions.MAX_LEVELS) if "spectrum" in wanted else 0
    curve: list[SweepPoint] = []
    for index in range(wanted_points):
        ratio = top * index / (wanted_points - 1)
        try:
            at_field = replace(base, field=ratio * base.coupling)
            curve.append(_point(at_field, ratio, per_point, cross_check))
        except Exception as error:  # one bad point must not lose the curve
            _logger.warning(
                "sweep_point_failed",
                extra={"error_type": type(error).__name__, "ratio": round(ratio, 4)},
            )
    if not curve:
        return FieldSweep(
            spec=None,
            points=(),
            curves=wanted,
            methods=(),
            detail="every point of the sweep failed",
        )
    _logger.info(
        "field_swept",
        extra={"points": len(curve), "curves": list(wanted), "cross_checked": cross_check},
    )
    return FieldSweep(
        spec=base,
        points=tuple(curve),
        curves=wanted,
        methods=methods,
        detail=(
            f"{len(curve)} field values from h/J = 0 to {top:g}, each solved by "
            f"{' and '.join(methods)}"
        ),
        levels_requested=per_point,
    )


def _sweep_lattice(
    base: TFIMSpec,
    wanted: tuple[Curve, ...],
    points: int,
    ratio_max: float,
    levels: int,
) -> FieldSweep:
    r"""Trace the field range on a two-dimensional lattice.

    A separate function rather than a branch inside :func:`sweep_field` because
    almost nothing is shared: the solver is different, the size limits are three
    orders of magnitude apart, two of the five curves are unavailable, and there is
    no thermodynamic-limit comparison to draw. Interleaving the two would produce a
    function in which every second line was a conditional.

    Without this, a question about a lattice's curve is answered by whichever route
    does not refuse, and the route that does not refuse sweeps a line of the same
    number of sites: the per-point spec is rebuilt from four fields and the shape is
    not one of them. A reader who
    asked for the ``4 x 4`` square's energy against the field was shown the
    sixteen-site chain's, and the closed form was quoted as its exact answer. Those
    two numbers are :math:`-1.28` and :math:`-2.13` per spin: not a small error, a
    different problem. It also made every variational energy for the lattice look
    like a broken variational bound, since a real 2D answer sits far below a line's.

    Args:
        base: The problem to sweep. Only its field varies.
        wanted: Curves the caller asked for, already canonicalised.
        points: How many field values to sample.
        ratio_max: Top of the range in units of :math:`J`.
        levels: How many many-body levels per point, if a spectrum was asked for.

    Returns:
        The sweep. Curves that a lattice cannot supply are dropped from
        :attr:`FieldSweep.curves` and named in :attr:`FieldSweep.detail`; a lattice
        too large to diagonalise comes back with no points and the size in the
        detail, rather than raising into a tool call.
    """
    if base.n_sites > MAX_LATTICE_SWEEP_SITES:
        return FieldSweep(
            spec=None,
            points=(),
            curves=wanted,
            methods=(),
            detail=(
                f"{base.rows}x{base.cols} is {base.n_sites} sites, above the "
                f"{MAX_LATTICE_SWEEP_SITES} this module will sweep on a lattice. There "
                "is no closed form off a line, so every point of the curve is a 2**L "
                "eigenproblem; ask for a smaller lattice, or for one field value "
                "rather than a curve"
            ),
        )

    supplied = tuple(curve for curve in wanted if curve in LATTICE_CURVES)
    declined = tuple(curve for curve in wanted if curve not in LATTICE_CURVES)
    if not supplied:
        supplied = ("energy", "magnetisation")

    wanted_points = max(3, min(int(points), MAX_POINTS))
    top = min(max(float(ratio_max), 0.5), MAX_RATIO_MAX)
    cross_check = base.n_sites <= MAX_LATTICE_CROSSCHECK_SITES
    methods = ("exact_diagonalisation",) + (("pauli_kronecker",) if cross_check else ())
    per_point = min(max(int(levels), 2), free_fermions.MAX_LEVELS) if "spectrum" in supplied else 0

    curve: list[SweepPoint] = []
    for index in range(wanted_points):
        ratio = top * index / (wanted_points - 1)
        try:
            at_field = replace(base, field=ratio * base.coupling)
            curve.append(_lattice_point(at_field, ratio, per_point, cross_check))
        except Exception as error:  # one bad point must not lose the curve
            _logger.warning(
                "lattice_sweep_point_failed",
                extra={"error_type": type(error).__name__, "ratio": round(ratio, 4)},
            )
    if not curve:
        return FieldSweep(
            spec=None,
            points=(),
            curves=supplied,
            methods=(),
            detail=f"every point of the {base.rows}x{base.cols} sweep failed",
        )

    note = (
        f"{len(curve)} field values from h/J = 0 to {top:g} on the "
        f"{base.lattice.in_words()}, each diagonalised"
    )
    note += (
        " and assembled a second way as a Kronecker product for comparison"
        if cross_check
        else f", once per point -- above {MAX_LATTICE_CROSSCHECK_SITES} sites the second "
        "assembly costs more than the curve"
    )
    if declined:
        note += (
            f". {' and '.join(declined)} not drawn: a lattice has no dispersion to "
            "differentiate in closed form, and a finite difference would make the "
            "plotted curve depend on how many points were asked for"
        )
    _logger.info(
        "lattice_field_swept",
        extra={
            "points": len(curve),
            "curves": list(supplied),
            "shape": base.lattice.describe(),
            "cross_checked": cross_check,
        },
    )
    return FieldSweep(
        spec=base,
        points=tuple(curve),
        curves=supplied,
        methods=methods,
        detail=note,
        levels_requested=per_point,
    )
