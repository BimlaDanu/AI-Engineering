r"""Run the near-term methods against each other on one chain and keep the curves.

Three methods, one chain, one axis: the energy each has reached against the step
it has taken, so *which is best* is read off the picture rather than asserted.

Every difference between the curves is a difference of method, because nothing
else varies -- same chain, same :math:`J`, :math:`h` and :math:`g`, same
boundary, circuit, depth and starting angles wherever the method permits. Only
the rule that moves the angles differs:

``VQE``
    L-BFGS on the energy, started just off the identity
    (:mod:`src.physics.quantum.variational_eigensolver`).

``QAOA``
    The identical circuit from a Trotterised adiabatic ramp, which is what makes
    it a different method in practice despite being the same gates. Same module.

``VarQITE``
    No optimiser; the angles follow a differential equation imitating
    imaginary-time decay
    (:mod:`src.physics.quantum.imaginary_time_evolution`).

A step does not cost the same in all three, and the axis cannot say so. An L-BFGS
step includes a line search and so costs several energy evaluations; a VarQITE
step costs one energy, one gradient and a Fubini-Study metric, which on hardware
is :math:`O(P^2)` extra circuits. :attr:`MethodRun.n_energy_evaluations` is
carried beside the curve and reported under it: steps are the shape of the
descent, evaluations are the bill.

Nothing here is noisy. Every energy is an exact expectation value from a
simulated state vector, so the curves describe a perfect device and say nothing
about shot noise or gate error -- which must be stated wherever the figure is
shown. The arithmetic that prices the noise is in :mod:`src.hardware`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.physics.lattice import Geometry, Lattice
from src.physics.model import BoundaryCondition
from src.physics.quantum.imaginary_time_evolution import evolve_in_imaginary_time
from src.physics.quantum.variational_eigensolver import solve

MethodName = Literal["VQE", "QAOA", "VarQITE"]
"""The three methods this module can race, written as a reader would write them."""

METHOD_NAMES: tuple[MethodName, ...] = ("VQE", "QAOA", "VarQITE")
"""All three, in the order they should appear in a legend and a table.

VQE first because it is the one every other is explained against, QAOA second because it
is the same circuit under another name, and VarQITE last because it is the one that does
not search at all.
"""

METHOD_BLURBS: dict[MethodName, str] = {
    "VQE": "an ordinary optimiser turning the circuit's dials downhill",
    "QAOA": "the same circuit, started from a slow sweep instead of from nothing",
    "VarQITE": "no optimiser: the dials follow an equation that imitates cooling",
}
"""One plain sentence per method, for a caption or a legend on a page.

Kept beside the runner rather than in the interface so that the words under the figure
and the method the figure drew cannot drift apart.
"""

MAX_RACED_SITES = 16
r"""Chain length past which this refuses to race rather than trying and hanging.

Every method here evolves a :math:`2^L` state vector and VarQITE assembles a metric from
:math:`P` of them, so sixteen sites is already a quarter of a million amplitudes carried
:math:`P` times over. Past that the honest answer is that this comparison is a
single-machine demonstration and the chain has outgrown it -- which is the project's
own thesis and not an embarrassment, so it is returned as a reason rather than raised
as an error.
"""


def unsupported_reason(n_sites: int, depth: int) -> str | None:
    """Say why a race cannot be run, in words a reader can act on.

    A string rather than an exception because the caller is usually an interface
    drawing a page: a refusal it can print beats a traceback it has to catch, and the
    reason names the knob to turn.

    Args:
        n_sites: Chain length asked for.
        depth: Circuit layers asked for.

    Returns:
        The reason, or ``None`` when the race can go ahead.
    """
    if n_sites < 2:
        return f"a chain needs at least two spins to have a bond, and this has {n_sites}"
    if n_sites > MAX_RACED_SITES:
        return (
            f"racing three methods on {n_sites} spins means carrying several "
            f"{2**n_sites:,}-amplitude state vectors at once, which is past what this "
            f"demonstration runs in a browser -- {MAX_RACED_SITES} is the most it will attempt"
        )
    if depth < 1:
        return f"a circuit with {depth} layers has no angles to move, so nothing would converge"
    return None


@dataclass(frozen=True, slots=True)
class MethodRun:
    """One method's descent, flattened to what a figure and a table both need.

    A deliberately narrow record. The eigensolver and the imaginary-time evolver return
    rich, differently shaped results, and something has to reduce them to a common shape
    before they can be drawn on one axis -- doing that here, once, is what stops the
    interface from growing an ``isinstance`` ladder.

    Attributes:
        method: Which of the three this was.
        energy: The lowest energy it reached. An upper bound on the true ground state.
        energy_per_site: That energy per spin, which is what makes two chain lengths
            comparable.
        energy_history: The energy after each accepted step, oldest first.
        n_steps: Steps accepted, which is the length of the curve less its starting point.
        n_energy_evaluations: Energies computed, including those a line search or a
            rejected step spent. The honest cost, and always larger than ``n_steps``.
        stop_reason: The method's own word for why it stopped. Not normalised across the
            three: "converged" means the same thing in all of them, and the others say
            something specific that would be lost by flattening.
    """

    method: MethodName
    energy: float
    energy_per_site: float
    energy_history: tuple[float, ...]
    n_steps: int
    n_energy_evaluations: int
    stop_reason: str

    @property
    def improvement(self) -> float:
        """How far the energy fell from where the method started."""
        if not self.energy_history:
            return 0.0
        return self.energy_history[0] - self.energy


@dataclass(frozen=True, slots=True)
class MethodRace:
    """Three descents on one chain, with the chain recorded beside them.

    The chain is carried in the record rather than left with the caller because a curve
    without its :math:`J`, :math:`h` and :math:`g` is not evidence of anything: two races
    at different fields look alike and mean entirely different things.

    Attributes:
        n_sites: Chain length raced on.
        depth: Circuit layers every method was given.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        geometry: The shape raced on -- ``"chain"``, ``"square"`` or
            ``"triangular"``. Recorded rather than left to be inferred, because
            whoever grades these curves has to solve the same problem the methods
            saw. Without it the interface graded a square lattice against the
            *chain's* exact energy, which is far higher: sixteen spins on a ring
            reach -20.40 and the same sixteen on a 4x4 square reach -34.01, so
            honest variational curves appeared to fall straight through the exact
            answer and keep going.
        rows: Rows of the lattice, ``1`` for a chain. With ``n_sites`` this fixes
            the shape, since the columns follow from it.
        runs: One entry per method, in :data:`METHOD_NAMES` order.
    """

    n_sites: int
    depth: int
    coupling: float
    transverse_field: float
    longitudinal_field: float
    boundary: BoundaryCondition
    runs: tuple[MethodRun, ...]
    geometry: Geometry = "chain"
    rows: int = 1

    @property
    def at_criticality(self) -> bool:
        r"""Whether this chain sits at :math:`h = J`, where every method has its worst time.

        Worth a property rather than a comparison at each call site: the gap between the
        ground state and the next level closes here, which is precisely what makes a
        variational method slow and a comparison between methods interesting.
        """
        return abs(self.transverse_field - self.coupling) < 1e-9

    def curves(self) -> tuple[tuple[str, tuple[float, ...]], ...]:
        """Render the race in the shape a figure wants.

        Returns:
            One ``(label, history)`` pair per method, in legend order.
        """
        return tuple((run.method, run.energy_history) for run in self.runs)

    def best(self) -> MethodRun | None:
        """The method that reached the lowest energy.

        Returns:
            The winning run, or ``None`` when nothing was raced. Lowest wins because
            every energy here is a variational upper bound, so lower is strictly closer.
        """
        return min(self.runs, key=lambda run: run.energy, default=None)

    def label(self) -> str:
        """A short identifier for a log line or a figure caption.

        The longitudinal field is named only when it is set, matching
        :meth:`src.agent.state.FormalModel.label` and the pages: a caption reading
        ``g=0`` under a two-term Hamiltonian asks the reader what :math:`g` is and
        then does not tell them.
        """
        parts = [
            f"L={self.n_sites}",
            f"J={self.coupling:g}",
            f"h={self.transverse_field:g}",
        ]
        if self.longitudinal_field != 0.0:
            parts.append(f"g={self.longitudinal_field:g}")
        parts.append(f"depth={self.depth}")
        return f"{' '.join(parts)} ({self.boundary[:4]})"

    def describe(self) -> dict[str, Any]:
        """Render the race as plain data for a tool result, a log line or a report.

        The histories are included in full here, unlike in the individual solvers'
        ``describe``. This is the one place where the *curve* is the deliverable rather
        than a diagnostic, so truncating it to its endpoints would discard the answer.

        Returns:
            A flat mapping.
        """
        return {
            "chain": self.label(),
            "n_sites": self.n_sites,
            "depth": self.depth,
            "coupling": self.coupling,
            "transverse_field": self.transverse_field,
            "longitudinal_field": self.longitudinal_field,
            "boundary": self.boundary,
            "at_criticality": self.at_criticality,
            "methods": {
                run.method: {
                    "energy": round(run.energy, 12),
                    "energy_per_site": round(run.energy_per_site, 12),
                    "n_steps": run.n_steps,
                    "n_energy_evaluations": run.n_energy_evaluations,
                    "stop_reason": run.stop_reason,
                    "energy_history": [round(value, 12) for value in run.energy_history],
                }
                for run in self.runs
            },
        }


def race_methods(
    n_sites: int,
    depth: int,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: BoundaryCondition = "open",
    methods: tuple[MethodName, ...] = METHOD_NAMES,
    lattice: Lattice | None = None,
) -> MethodRace:
    r"""Run each named method on the same chain and return their loss curves.

    Works for **any** :math:`J`, :math:`h` and :math:`g`, which is the point: the chain
    comes from whatever the question described, so the picture underneath an answer is
    of the problem that was actually asked about rather than of a stock example. The
    critical point :math:`h = J` is only the most interesting case, not a special one.

    Args:
        n_sites: Number of spins, equal to the number of qubits.
        depth: Circuit layers. Every method gets the same number, or the comparison
            measures the depth rather than the method.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis. Non-zero breaks
            the free-fermion mapping, so there is then no closed-form answer to grade
            against -- which is a fact about the chain, not a failure of the race.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        methods: Which to run, in the order they should be drawn.
        lattice: The shape the spins sit on, or ``None`` for a chain. Passed straight
            through to each solver, so the three methods race on the same geometry as
            well as the same couplings -- a comparison in which one method saw a
            different problem measures nothing.

    Returns:
        The race, with one :class:`MethodRun` per named method.

    Raises:
        ValueError: If the chain or depth is one this cannot race. Ask
            :func:`unsupported_reason` first -- reaching this exception means a race
            was started without checking, which the interface is expected not to do.

    Examples:
        Every method returns an energy above the true ground state, and the curve is
        what the caller came for:

        >>> race = race_methods(n_sites=4, depth=2)
        >>> [run.method for run in race.runs]
        ['VQE', 'QAOA', 'VarQITE']
        >>> all(len(run.energy_history) > 1 for run in race.runs)
        True
    """
    reason = unsupported_reason(n_sites, depth)
    if reason is not None:
        raise ValueError(f"cannot race methods on this chain: {reason}")

    runs: list[MethodRun] = []
    for method in methods:
        if method == "VarQITE":
            evolved = evolve_in_imaginary_time(
                n_sites=n_sites,
                depth=depth,
                coupling=coupling,
                transverse_field=transverse_field,
                longitudinal_field=longitudinal_field,
                boundary=boundary,
                lattice=lattice,
            )
            runs.append(
                MethodRun(
                    method=method,
                    energy=evolved.energy,
                    energy_per_site=evolved.energy_per_site,
                    energy_history=evolved.energy_history,
                    n_steps=evolved.n_steps,
                    n_energy_evaluations=evolved.n_energy_evaluations,
                    stop_reason=evolved.stop_reason,
                )
            )
            continue
        # VQE and QAOA are the same solver on the same circuit. What separates them is
        # the tradition's starting point, and that difference is the whole finding: at
        # shallow depth the ramp start and the near-identity start land in different
        # basins and disagree in the first significant figure.
        result = solve(
            n_sites=n_sites,
            depth=depth,
            coupling=coupling,
            transverse_field=transverse_field,
            longitudinal_field=longitudinal_field,
            boundary=boundary,
            family="hva" if method == "VQE" else "qaoa",
            initialisation="small_angle" if method == "VQE" else "adiabatic_ramp",
            lattice=lattice,
        )
        runs.append(
            MethodRun(
                method=method,
                energy=result.energy,
                energy_per_site=result.energy_per_site,
                energy_history=result.energy_history,
                n_steps=max(len(result.energy_history) - 1, 0),
                n_energy_evaluations=result.n_energy_evaluations,
                stop_reason=result.stop_reason,
            )
        )

    return MethodRace(
        n_sites=n_sites,
        depth=depth,
        coupling=coupling,
        transverse_field=transverse_field,
        longitudinal_field=longitudinal_field,
        boundary=boundary,
        runs=tuple(runs),
        # Read back off the lattice the solvers were actually given, rather than
        # taken as a second argument, so the record cannot disagree with the run.
        geometry="chain" if lattice is None else lattice.geometry,
        rows=1 if lattice is None else lattice.rows,
    )
