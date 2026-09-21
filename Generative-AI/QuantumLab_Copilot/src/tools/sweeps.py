r"""Sweeping the field: the observable a single run cannot show.

One question, one chain, one field value -- and most of the interesting physics
here is not at a point, it is in a *curve*. "How does the magnetisation turn on?"
and "where does this chain look critical?" are questions about a trajectory, and a
solver called once cannot answer either.

This module runs the trajectory. For each field value it computes the energy
density and the transverse magnetisation
:math:`\langle \sigma^x \rangle = L^{-1}\sum_i \langle \sigma^x_i \rangle`, and where both
solvers apply it computes each point **twice, independently** -- the free-fermion
closed form and exact diagonalisation on the state vector -- and records how far
apart they were. So the curve is verified the same way a single number is, point
by point, rather than being a cheaper class of result that happens to be plotted.

Two derived quantities come out of the curve and are worth more than either raw
column:

* :attr:`FieldSweep.turning_point` -- where the magnetisation is rising fastest.
  In an infinite chain that is exactly the critical point; in a finite one it is
  nearby and sharpens as the chain grows, which is what a phase transition
  actually looks like before the thermodynamic limit is taken.
* :attr:`FieldSweep.max_disagreement` -- the largest gap between the two methods
  anywhere on the curve. One number that stands behind every point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.logging_setup import get_logger
from src.physics import ed, exact
from src.physics.model import TFIMSpec

LOG = get_logger("tools.sweeps")

Observable = Literal["ground_state", "spectrum", "derivatives", "both"]
"""Which curve a sweep was asked for.

The agent chooses this, not the code: "plot the magnetisation", "plot the
low-lying spectrum" and "plot the energy and its derivatives" are different requests
about different physics, and a tool that always draws the same figure is a tool the
caller cannot aim. See :class:`~src.tools.calling.SweepField`, where it is a field
the model fills in.

``derivatives`` is separate from ``ground_state`` although both are properties of
the ground state, and the reason is the transition: :math:`E_0/L` is featureless
through it, its first derivative rises and its second dips sharply, so a question
about *derivatives* is answered by three stacked panels and a table of slopes --
not by a magnetisation curve that happens to be the first derivative in disguise.
Answering it with the magnetisation was the bug this member fixes.

Every column is computed whichever one is asked for, so this setting selects what
is *shown* and what the narrator is told about, not what is calculated.

That costs something, and the cost is worth knowing before anyone changes it.
Timed per sweep point on a warm process: at the default ``L = 6`` the levels take
0.24 ms against 0.43 ms for the ground state (they skip the eigenvectors, so they
are cheaper than the run that happens anyway); at ``L = 8``, the longest chain a
sweep will diagonalise, they take 4.9 ms against 0.74 ms and dominate the point.

Computing everything is still the default, because a sweep is cached: a follow-up
about the other curve is answered from the sweep already in hand rather than by
running a second one.
"""

DEFAULT_POINTS = 21
"""Points sampled by default.

Enough to see the shape and to locate the turning point to a few per cent of the
field range, cheap enough that a question does not feel like a job submission.
Matches :data:`~src.agent.setting.DEFAULT_SWEEP_POINTS`: an agent answering a
question and a visitor dragging a slider should get the same curve.
"""

MAX_POINTS = 121
"""Most points any sweep will take. A finer curve tells a reader nothing new."""

MAX_RATIO = 2.0
r"""Top of the swept range, in units of ``J``.

The transition sits at :math:`g = 1`, so the range runs from the fully ordered
chain to twice the critical field -- both phases and the crossover between them,
with the interesting part in the middle rather than at an edge.
"""

MAX_COSTLY_SITES = 8
"""Longest chain swept when each point needs a diagonalisation.

At ``L = 8`` a point is a 256-dimensional sparse eigenproblem and the whole sweep
is well under a second. The limit is deliberately below the length at which a run
would stop to ask the user about cost: a sweep multiplies that cost by the number
of points, so it stops earlier, not later.
"""


@dataclass(frozen=True, slots=True)
class SweepPoint:
    r"""One field value, and what both methods said about it.

    Attributes:
        ratio: The control parameter ``g = h / J``.
        field: The field itself, ``h``.
        energy_density: ``E_0 / L``, the quantity that is comparable across chain
            lengths.
        magnetisation: The transverse magnetisation per site, in ``[0, 1]``.
        disagreement: Largest absolute difference between the two methods at this
            point, or ``None`` when only one method applies here.
        excitations: Energies of the low-lying states measured from the ground
            state, ``E_n - E_0``, starting with the ``0.0`` of the ground state
            itself. Empty when no method here can reach the excited states.
        slope: :math:`\partial (E_0/L)/\partial h`, or ``None`` where the closed
            form does not apply. Analytic, not a finite difference.
        curvature: :math:`\partial^2 (E_0/L)/\partial h^2`, on the same terms.
        magnetisation_slope: :math:`\partial \langle \sigma^x \rangle/\partial h`,
            on the same terms. Carried separately from :attr:`curvature` even
            though it is minus it: *plot the magnetisation and its derivatives* is
            a question this could not answer while the only derivatives on the
            point were the energy's, and telling a reader their curve is another
            curve with a sign on it is not answering them.
        magnetisation_curvature: :math:`\partial^2 \langle \sigma^x
            \rangle/\partial h^2`, on the same terms. The third derivative of the
            energy density, and the one quantity here that has no other name.
    """

    ratio: float
    field: float
    energy_density: float
    magnetisation: float
    disagreement: float | None
    excitations: tuple[float, ...] = ()
    slope: float | None = None
    curvature: float | None = None
    magnetisation_slope: float | None = None
    magnetisation_curvature: float | None = None

    @property
    def gap(self) -> float | None:
        """Energy of the first excitation, ``E_1 - E_0``.

        Returns:
            The gap, or ``None`` when the spectrum was not computed here. This is
            the quantity that closes at the transition, so it is named rather than
            left as an index into :attr:`excitations`.
        """
        return self.excitations[1] if len(self.excitations) > 1 else None


@dataclass(frozen=True, slots=True)
class FieldSweep:
    """A curve, its provenance, and its verification.

    Attributes:
        spec: The chain that was swept. Its ``field`` is the value the question
            was asked at; the sweep varies it.
        points: The curve, in increasing field.
        methods: Names of the solvers that ran at every point.
        detail: One sentence on what happened, populated whether or not the sweep
            ran.
        observable: What the caller asked to see. Carried on the result rather than
            left at the call site so the interface renders what the agent planned,
            not what the renderer happens to draw.
        max_costly_sites: The ceiling this sweep actually ran under. Carried for the
            same reason as ``observable``: the messages below quote the limit, and
            once a user can move it the module default is no longer that number.
    """

    spec: TFIMSpec | None
    points: tuple[SweepPoint, ...]
    methods: tuple[str, ...]
    detail: str
    observable: Observable = "ground_state"
    max_costly_sites: int = MAX_COSTLY_SITES

    @property
    def wants_spectrum(self) -> bool:
        """Whether the spectrum was asked for and is available to show."""
        return self.observable in {"spectrum", "both"} and self.levels_drawn > 1

    @property
    def wants_ground_state(self) -> bool:
        """Whether the ground-state curves were asked for.

        Returns:
            ``True`` unless the question was about the spectrum or about the
            derivatives. A sweep asked for the levels still computed the ground
            state -- it had to, since the levels are measured from it -- but showing
            curves nobody asked about buries the answer.

            ``derivatives`` is excluded even though it is ground-state physics:
            asking for the first derivative of the energy and being shown the
            magnetisation is being shown the right number under the wrong name, and
            the point of a derivatives question is the *curvature*, which the
            magnetisation panel does not carry at all.
        """
        return self.observable in {"ground_state", "both"}

    @property
    def wants_derivatives(self) -> bool:
        """Whether the energy's field derivatives were asked for and are available."""
        return self.observable == "derivatives" and self.has_derivatives

    @property
    def has_derivatives(self) -> bool:
        """Whether every point carries both derivatives.

        Returns:
            ``True`` only when the closed form applied at every field, since that is
            where the derivatives come from. All-or-nothing for the same reason as
            :attr:`has_spectrum`: a curve with holes is not a curve.
        """
        return bool(self.points) and all(point.curvature is not None for point in self.points)

    @property
    def ok(self) -> bool:
        """Whether a curve was produced."""
        return bool(self.points)

    @property
    def max_disagreement(self) -> float | None:
        """Worst disagreement between the two methods anywhere on the curve.

        Returns:
            The largest gap, or ``None`` when no point had two methods to compare.
            One number for the whole curve, because a reader who wants to know
            whether to trust the plot is asking about its worst point, not its
            average.
        """
        gaps = [point.disagreement for point in self.points if point.disagreement is not None]
        return max(gaps) if gaps else None

    @property
    def is_corroborated(self) -> bool:
        """Whether two independent methods agreed at every point.

        Returns:
            ``True`` only when every point was computed twice and every
            comparison came out at floating-point noise. A curve with one
            unchecked point is not a checked curve.
        """
        if not self.points:
            return False
        if any(point.disagreement is None for point in self.points):
            return False
        worst = self.max_disagreement
        return worst is not None and worst < 1e-9

    @property
    def has_spectrum(self) -> bool:
        """Whether the low-lying levels were computed at every point.

        Returns:
            ``True`` only when every point carries a gap. A spectrum plot with
            holes in it is not a spectrum plot, so this is all-or-nothing rather
            than per-point.
        """
        return bool(self.points) and all(point.gap is not None for point in self.points)

    @property
    def levels_drawn(self) -> int:
        """How many low-lying levels every point has in common.

        Returns:
            The smallest number of levels found across the curve, so a line drawn
            through the spectrum has a value at every field. Zero when the
            spectrum was not computed.
        """
        if not self.points:
            return 0
        return min(len(point.excitations) for point in self.points)

    @property
    def critical_gap(self) -> tuple[float, float] | None:
        r"""The gap at the swept point nearest the critical field.

        Returns:
            ``(ratio, gap)`` at the sampled ``g`` closest to 1, or ``None`` without
            a spectrum.

            This rather than the *smallest* gap on the curve, which would be a
            useless number: in a ferromagnetic chain the two lowest levels are
            exactly degenerate at :math:`h = 0` and separate monotonically from
            there, so the minimum always sits at the left edge and says nothing
            about criticality. What carries the physics is that the gap at
            :math:`g = 1` is small but **not zero** -- it vanishes there only in the
            infinite chain, and watching that number shrink as ``L`` grows is how a
            finite calculation sees a phase transition at all.
        """
        gaps = [(point.ratio, point.gap) for point in self.points if point.gap is not None]
        if not gaps:
            return None
        where, gap = min(gaps, key=lambda pair: abs(pair[0] - 1.0))
        return where, gap

    @property
    def sharpest_curvature(self) -> tuple[float, float] | None:
        r"""Where the energy density bends most sharply, and by how much.

        Returns:
            ``(ratio, curvature)`` at the most negative
            :math:`\partial^2 (E_0/L)/\partial h^2` on the curve, or ``None`` when
            the derivatives were not computed.

            This is the derivatives question's answer in one number. The energy
            itself shows nothing at the transition and its first derivative only
            bends; the second derivative dips, and the dip deepens and moves towards
            :math:`g = 1` as the chain grows -- which is a finite chain's way of
            saying that the infinite one has a discontinuity there.
        """
        bends = [(point.ratio, point.curvature) for point in self.points]
        found = [(ratio, value) for ratio, value in bends if value is not None]
        return min(found, key=lambda pair: pair[1]) if found else None

    @property
    def turning_point(self) -> float | None:
        r"""Where the magnetisation rises fastest, in units of ``J``.

        Returns:
            The ratio ``g`` at the steepest rise of
            :math:`\langle \sigma^x \rangle`, or
            ``None`` for a curve too short to have a slope. This is the finite
            chain's own answer to "where is the transition?" -- computed, not
            assumed, which is why it is worth showing next to the exact ``g = 1``
            of the infinite chain rather than instead of it.
        """
        if len(self.points) < 3:
            return None
        best, where = -1.0, None
        for left, right in zip(self.points, self.points[1:], strict=False):
            span = right.ratio - left.ratio
            if span <= 0.0:
                continue
            slope = (right.magnetisation - left.magnetisation) / span
            if slope > best:
                best, where = slope, 0.5 * (left.ratio + right.ratio)
        return where

    def unavailable_observable(self) -> str | None:
        """Say why the curve the caller asked for is not in this sweep.

        A sweep runs whatever solvers apply to the chain, and the two do not cover
        the same ground: the excitation spectrum comes only from diagonalisation,
        which a long chain declines, and the energy derivatives come only from the
        closed form, which an open or odd chain declines. Either way the sweep
        still succeeds -- it just succeeds at a different question than the one
        that was asked, and that has to be said out loud.

        Returns:
            One sentence naming the missing quantity and the reason, or ``None``
            when the sweep produced what it was asked for.

        Examples:
            An open chain has no closed form, so it has no analytic derivatives:

            >>> sweep = sweep_field(TFIMSpec(n_sites=6, boundary="open"),
            ...                     points=5, observable="derivatives")
            >>> "derivatives" in (sweep.unavailable_observable() or "")
            True

            And a sweep that got what it asked for says nothing:

            >>> sweep_field(TFIMSpec(n_sites=6), points=5).unavailable_observable() is None
            True
        """
        if not self.points or self.spec is None:
            return None
        if self.observable in {"spectrum", "both"} and not self.has_spectrum:
            return (
                f"the excitation spectrum, which needs {ed.METHOD_NAME} at every point; "
                f"this sweep ran {' and '.join(self.methods)} only, and diagonalisation "
                f"is limited to L = {self.max_costly_sites} across a whole sweep"
            )
        if self.observable == "derivatives" and not self.has_derivatives:
            reason = exact.unsupported_reason(self.spec) or "the closed form did not apply"
            return f"the energy derivatives, which come from {exact.METHOD_NAME}: {reason}"
        return None

    def table(self) -> str:
        """Render the curve as compact text for a prompt.

        Returns:
            A header line and one line per point, thinned so a narrator sees the
            shape rather than a wall of digits, followed by whichever summary
            lines the requested observable calls for -- including, when it could
            not be computed at all, a line saying so and why. Empty when nothing
            was computed.
        """
        if not self.points:
            return ""
        step = max(1, len(self.points) // 9)
        # One column per quantity that was asked for, and none of the others. The
        # narrator writes about what it is shown, so a table that always carried
        # every quantity would answer a question about the magnetisation with a
        # paragraph about the gap -- and a table that carried too few would make it
        # say the numbers were not determined, which is how this began.
        show_gap = self.wants_spectrum
        show_derivatives = self.wants_derivatives
        # The magnetisation stays on a derivatives sweep, and that is a correction.
        # Withholding it is what the figure does -- three stacked panels, not a
        # magnetisation curve -- and copying that rule into the text made the
        # narrator say the quantity had not been computed. It always is: every point
        # carries it, and by Hellmann-Feynman it *is* the first derivative, up to a
        # sign. So *plot the magnetisation and its derivatives* was answered with
        # "those were not computed in the supplied data", about numbers sitting one
        # column away. What must not happen is a derivatives question answered by a
        # magnetisation panel; showing the narrator both columns is not that.
        show_magnetisation = True
        rows = [
            f"  g={point.ratio:.3f}  E0/L={point.energy_density:.6f}"
            # The explicit operator name, matching the docstrings: "<X>" reads as a
            # gate to anyone coming from the quantum-computing side, and this string
            # is what the narrator writes its paragraph from. ASCII rather than a
            # literal Greek letter, which RUF001 rejects.
            + (f"  <sigma_x>={point.magnetisation:.6f}" if show_magnetisation else "")
            + (f"  gap={point.gap:.6f}" if show_gap and point.gap is not None else "")
            + (
                f"  d(E0/L)/dh={point.slope:.6f}  d2(E0/L)/dh2={point.curvature:.6f}"
                if show_derivatives and point.slope is not None and point.curvature is not None
                else ""
            )
            # The magnetisation's own two, so that "plot the magnetisation and its
            # derivatives" is answered with the magnetisation's derivatives rather
            # than with the energy's under a note about Hellmann-Feynman.
            + (
                f"  d<sigma_x>/dh={point.magnetisation_slope:.6f}"
                f"  d2<sigma_x>/dh2={point.magnetisation_curvature:.6f}"
                if show_derivatives
                and point.magnetisation_slope is not None
                and point.magnetisation_curvature is not None
                else ""
            )
            for point in self.points[::step]
        ]
        asked = {
            "ground_state": "the ground-state properties (magnetisation and energy density)",
            "spectrum": "the low-lying excitation spectrum, E_n - E_0",
            "derivatives": (
                "how the ground state responds to the field: the energy density and "
                "the magnetisation, each with its first and second derivative, all "
                "in closed form. The two sets are one physics -- by Hellmann-Feynman "
                "d(E0/L)/dh = -<sigma_x> -- so a question about either is answered "
                "from these columns, and neither quantity is uncomputed"
            ),
            "both": "the ground-state properties and the excitation spectrum",
        }[self.observable]
        lines = [
            f"FIELD SWEEP of {self.spec.label() if self.spec else 'the chain'}, "
            f"{len(self.points)} points, g = h/J from 0 to {MAX_RATIO:g}. "
            f"The question asked about {asked}:",
            *rows,
        ]
        if show_magnetisation:
            turning = self.turning_point
            where = f"{turning:.3f}" if turning is not None else "not determined"
            lines.append(f"  magnetisation rises fastest at g = {where}")
        sharpest = self.sharpest_curvature if show_derivatives else None
        if sharpest is not None:
            # Spelled out for the same reason as the gap below: the narrator may
            # only use numbers it was given, and "the derivatives were not
            # determined" is what it writes when this line is missing.
            lines.append(
                f"  both derivatives are analytic, from the {exact.METHOD_NAME} closed form: "
                f"the first is minus the transverse magnetisation by Hellmann-Feynman, "
                f"the second is differentiated in closed form. No finite differences."
            )
            lines.append(
                f"  d2(E0/L)/dh2 is most negative at g = {sharpest[0]:.3f}, where it "
                f"reaches {sharpest[1]:.6f}: the energy is featureless through the "
                f"transition, its first derivative bends, and only its curvature has a "
                f"sharp feature there"
            )
        missing = self.unavailable_observable()
        if missing is not None:
            # Without this line the narrator gets the header for one quantity and the
            # numbers for another, and writes about the numbers. Found in review: a
            # derivatives request on an open chain returned the magnetisation curve.
            lines.append(f"  NOT COMPUTED: {missing}")
        critical = self.critical_gap if show_gap else None
        if critical is not None:
            # Spelled out because of the narrator's first rule: it may only use
            # numbers it was given, so a gap it is expected to discuss has to
            # appear here rather than be inferable from the rows.
            lines.append(
                f"  gap (E1 - E0) at g = {critical[0]:.3f} is {critical[1]:.6f}, "
                f"small but not zero because the chain is finite"
            )
            lines.append(
                f"  the {self.levels_drawn} lowest levels were computed at every point "
                f"by {ed.METHOD_NAME}; the plot shows them measured from the ground state"
            )
        return "\n".join(lines)

    def explain(self) -> str:
        """Say what the sweep did, in one line.

        Returns:
            The point count, the methods and the worst disagreement -- or the
            reason nothing ran.
        """
        if not self.points:
            return f"no sweep: {self.detail}"
        worst = self.max_disagreement
        agreement = "one method only" if worst is None else f"methods agree to {worst:.1e}"
        return (
            f"swept {len(self.points)} field values with {' and '.join(self.methods)} ({agreement})"
        )


def _point(spec: TFIMSpec, ratio: float, methods: tuple[str, ...]) -> SweepPoint:
    """Compute one field value with the methods this sweep committed to.

    Args:
        spec: The chain at this field.
        ratio: The control parameter ``g``, carried through for the curve.
        methods: The names :func:`methods_for` returned for the sweep.

    Returns:
        The point.

    The method list is passed in rather than re-derived here, and that is a fix
    rather than a tidy-up: :func:`methods_for` drops diagonalisation past
    :data:`MAX_COSTLY_SITES`, while ``ed.unsupported_reason`` allows it up to
    ``L = 12``. Asking each function separately meant a ten-site sweep quietly ran
    the expensive method at every point while reporting that only the closed form
    had run -- both a cost the caller had declined and a verification claim that
    understated itself.
    """
    density: list[float] = []
    magnetisation: list[float] = []
    excitations: tuple[float, ...] = ()
    slope: float | None = None
    curvature: float | None = None
    mag_slope: float | None = None
    mag_curvature: float | None = None
    if exact.METHOD_NAME in methods:
        density.append(exact.energy_density(spec))
        magnetisation.append(exact.transverse_magnetisation(spec))
        # Both derivatives analytic, and both from the closed form only. The first
        # is minus the magnetisation -- Hellmann-Feynman, not an approximation of
        # it -- and the second is differentiated by hand in `exact`. A finite
        # difference taken across the sweep would depend on the point spacing, so
        # asking for a coarse curve would change the physics it reported.
        slope = -magnetisation[0]
        curvature = exact.energy_density_curvature(spec)
        # And the magnetisation's own two, which cost one more closed-form sum
        # between them: the first is this curvature negated, the second is the
        # energy density's third derivative. Asked to plot the magnetisation and
        # its derivatives, the sweep used to return the energy's instead and the
        # answer said the quantity had not been computed.
        mag_slope = exact.magnetisation_slope(spec)
        mag_curvature = exact.magnetisation_curvature(spec)
    if ed.METHOD_NAME in methods:
        result = ed.solve(spec)
        density.append(result.energy_density)
        magnetisation.append(result.transverse_magnetisation)
        # The spectrum comes from diagonalisation only. The free-fermion solver
        # has the ingredients -- the dispersion is exact and the ground-state
        # energy it returns agrees with this one to machine precision -- but
        # assembling the many-body levels from it means tracking which parity
        # sector each state lives in, and that bookkeeping changes character at
        # `h = J`. A curve that is right in one phase and wrong in the other is
        # worse than a curve computed by one method and labelled as such.
        levels = ed.low_levels(spec)
        excitations = tuple(float(level - levels[0]) for level in levels)
    gap = (
        max(abs(density[0] - density[1]), abs(magnetisation[0] - magnetisation[1]))
        if len(density) > 1
        else None
    )
    return SweepPoint(
        ratio=ratio,
        field=spec.field,
        energy_density=density[0],
        magnetisation=magnetisation[0],
        disagreement=gap,
        excitations=excitations,
        slope=slope,
        curvature=curvature,
        magnetisation_slope=mag_slope,
        magnetisation_curvature=mag_curvature,
    )


def methods_for(spec: TFIMSpec, max_costly_sites: int = MAX_COSTLY_SITES) -> tuple[str, ...]:
    """Name the solvers that can cover a whole sweep of this chain.

    Args:
        spec: The chain, at any field -- applicability does not depend on ``h``.
        max_costly_sites: Longest chain diagonalisation is allowed to sweep.
            Defaults to :data:`MAX_COSTLY_SITES`; the interface passes the user's
            own ceiling, which is the only thing that makes offering one honest.

    Returns:
        The method names, in the order they will be called. Empty when nothing
        applies, which is the caller's signal to decline rather than to sweep.
    """
    names: list[str] = []
    if exact.unsupported_reason(spec) is None:
        names.append(exact.METHOD_NAME)
    if ed.unsupported_reason(spec) is None and spec.n_sites <= max_costly_sites:
        names.append(ed.METHOD_NAME)
    return tuple(names)


def sweep_field(
    base: TFIMSpec,
    points: int = DEFAULT_POINTS,
    observable: Observable = "ground_state",
    max_costly_sites: int = MAX_COSTLY_SITES,
) -> FieldSweep:
    """Trace the field range and record what the caller asked to see.

    Args:
        base: The chain to sweep. Its coupling, length and boundary are held
            fixed; only the field varies.
        points: How many field values to sample. Held between three and
            :data:`MAX_POINTS`: three is the fewest that has a turning point, so
            a smaller request is raised rather than refused.
        observable: Which curve the question was about. See :data:`Observable`.
        max_costly_sites: Longest chain diagonalisation may sweep, defaulting to
            :data:`MAX_COSTLY_SITES`. Lowering it does not make the curve coarser --
            it drops the expensive solver from the sweep altogether, so what comes
            back is the closed form alone, without its independent check and
            without the spectrum. The sweep says so rather than going quiet.

    Returns:
        raises: this runs inside a tool call, where an exception would discard an
        answer that was otherwise complete.

    Examples:
        >>> sweep = sweep_field(TFIMSpec(n_sites=6), points=21)
        >>> sweep.ok and sweep.is_corroborated
        True
        >>> abs(sweep.points[0].magnetisation) < 1e-12  # ordered chain at h = 0
        True
        >>> sweep.points[-1].magnetisation > 0.9  # field dominates
        True

        The spectrum is computed whichever curve was asked for, and the gap at the
        critical field is small on a short chain without being zero:

        >>> sweep = sweep_field(TFIMSpec(n_sites=6), points=21, observable="spectrum")
        >>> sweep.wants_spectrum and not sweep.wants_ground_state
        True
        >>> ratio, gap = sweep.critical_gap
        >>> ratio == 1.0 and 0.0 < gap < 0.5
        True

        A question about the energy's derivatives gets them, and does not get the
        magnetisation curve that would otherwise stand in for the first one:

        >>> sweep = sweep_field(TFIMSpec(n_sites=6), points=21, observable="derivatives")
        >>> sweep.wants_derivatives and not sweep.wants_ground_state
        True
        >>> ratio, bend = sweep.sharpest_curvature
        >>> 0.5 < ratio < 1.5 and bend < 0.0
        True
    """
    names = methods_for(base, max_costly_sites)
    if not names:
        return FieldSweep(
            spec=None,
            points=(),
            methods=(),
            detail=(
                f"no method can sweep {base.label()}; a sweep needs one solve per point, "
                f"so it is limited to L = {max_costly_sites} when diagonalisation is involved"
            ),
            max_costly_sites=max_costly_sites,
        )
    wanted = max(3, min(points, MAX_POINTS))
    step = MAX_RATIO / (wanted - 1)
    curve: list[SweepPoint] = []
    for index in range(wanted):
        ratio = index * step
        try:
            spec = TFIMSpec(
                n_sites=base.n_sites,
                coupling=base.coupling,
                field=ratio * base.coupling,
                boundary=base.boundary,
            )
            curve.append(_point(spec, ratio, names))
        except Exception as error:  # a single bad point must not lose the curve
            LOG.warning(
                "sweep_point_failed",
                extra={"error_type": type(error).__name__, "ratio": round(ratio, 4)},
            )
    if not curve:
        return FieldSweep(
            spec=None,
            points=(),
            methods=(),
            detail="every sweep point failed",
            max_costly_sites=max_costly_sites,
        )
    LOG.info(
        "swept",
        extra={"points": len(curve), "methods": list(names), "observable": observable},
    )
    return FieldSweep(
        spec=base,
        points=tuple(curve),
        methods=names,
        detail=f"{len(curve)} field values, each solved by {' and '.join(names)}",
        observable=observable,
        max_costly_sites=max_costly_sites,
    )
