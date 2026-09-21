r"""The grader's bench: catalogue entries bound to the exact solvers behind them.

:mod:`src.physics.method_catalogue` says *what exists*. This module says *how to
run it*, and it is the only place where the two are joined. It imports
:mod:`src.physics.reference`, so it belongs to the grader alone and nothing the
agent runs may import it -- the architecture test walks the import
graph and fails if anything does.

The split matters more than it looks. The obvious design has one registry
holding both the facts and the callables, and the agent's method-selection step
reads it; that step then holds, one attribute away, a function returning the
exact ground-state energy it is about to be graded against. No amount of good
behaviour elsewhere would let anyone tell afterwards whether it had computed its
answer or looked it up. So the catalogue is handed to the agent with
``ground_state_energy=None`` on every exact method, and the binding lives here.

What this module adds on top of the catalogue is therefore small on purpose:
:func:`bind`, which fills in the callables, and :func:`exact_survey`, which is
what :mod:`src.verification.cross_check` iterates over.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol, cast

from src.physics.lattice import Geometry
from src.physics.method_catalogue import (
    EXACT_DIAGONALISATION,
    PFEUTY_EXACT,
    MethodFacts,
    MethodSurvey,
    Rejection,
    get_method,
)
from src.physics.model import BoundaryCondition, TFIMSpec
from src.physics.reference import exact_diagonalisation, field_sweep, free_fermions


class Method(Protocol):
    """The contract every exact solver in the sealed package satisfies.

    Modules -- not classes -- implement this:
    :mod:`src.physics.reference.free_fermions` and
    :mod:`src.physics.reference.exact_diagonalisation` are both valid
    :class:`Method` values. Stating it as a protocol lets the type checker prove
    that a newly added solver exposes the right names before any test runs.

    Attributes:
        METHOD_NAME: The catalogue key, defined beside the implementation so the
            two cannot drift apart.
    """

    METHOD_NAME: str

    def unsupported_reason(self, spec: TFIMSpec) -> str | None:
        """Explain why the method cannot handle ``spec``, or return ``None``."""
        ...

    def ground_state_energy(self, spec: TFIMSpec) -> float:
        """Return the ground-state energy of ``spec``."""
        ...


_CONFORMANCE: tuple[Method, ...] = (free_fermions, exact_diagonalisation)
"""Static proof that each bound module satisfies :class:`Method`.

Never read at runtime. It exists so that mypy fails the build if a solver drops
or renames one of the contract's names, which a test could only catch by
importing every module by hand.
"""

_SOLVERS: dict[str, Method] = {
    PFEUTY_EXACT: free_fermions,
    EXACT_DIAGONALISATION: exact_diagonalisation,
}
"""Which sealed module implements which catalogue key.

Keyed by the catalogue's own constants rather than by string literals, so a
renamed method is a failed import here and not a silently empty binding.
"""


def bind(facts: MethodFacts) -> MethodFacts:
    """Attach the real solver to a catalogue entry.

    Args:
        facts: A catalogue entry, ``grader_only`` or not.

    Returns:
        A copy carrying a working ``ground_state_energy``. An entry the agent
        may already run is returned unchanged rather than rejected: the point of
        this function is to make every entry runnable *here*, and the classical
        baseline is a legitimate route for the grader to compare against too.

    Raises:
        KeyError: If a ``grader_only`` entry has no implementation bound to it,
            which means a method was catalogued and then never wired up.
    """
    if facts.ground_state_energy is not None:
        return facts
    solver = _SOLVERS[facts.name]
    return dataclasses.replace(facts, ground_state_energy=solver.ground_state_energy)


def solver_for(name: str) -> MethodFacts:
    """Look up one method by catalogue key, with its solver attached.

    Args:
        name: A catalogue key.

    Returns:
        The bound entry, ready to run.

    Raises:
        KeyError: If the name is not catalogued, or is catalogued but unbound.
    """
    return bind(get_method(name))


def exact_methods() -> tuple[MethodFacts, ...]:
    """Every exact solver, bound and ready to run.

    Restricted to :data:`~src.physics.method_catalogue.AccuracyClass` ``exact``
    because these are the routes that must *agree* with one another. The
    variational baseline returns an upper bound with an error bar; comparing it
    against a closed form at a relative tolerance of ``1e-9`` would report a
    contradiction where there is only a variational gap, and that is a verdict
    about the tolerance rather than about the physics.

    Returns:
        The bound exact entries, in catalogue order.
    """
    return tuple(bind(get_method(name)) for name in (PFEUTY_EXACT, EXACT_DIAGONALISATION))


def exact_survey(spec: TFIMSpec) -> MethodSurvey:
    """Ask every exact solver whether it can solve ``spec``, and bind those that can.

    The grader's counterpart to
    :func:`src.physics.method_catalogue.survey`. Same partition, but restricted
    to the exact routes and with the callables filled in.

    Args:
        spec: The problem to survey.

    Returns:
        The applicable exact methods and the rejections, both in catalogue order.

    Examples:
        >>> from src.physics.model import TFIMSpec
        >>> exact_survey(TFIMSpec(n_sites=4)).names()
        ('pfeuty_exact', 'exact_diagonalisation')

        And unlike the agent's view of the catalogue, these can be run:

        >>> facts = exact_survey(TFIMSpec(n_sites=4)).applicable[0]
        >>> round(facts.solve(TFIMSpec(n_sites=4)), 6)
        -5.226252
    """
    applicable: list[MethodFacts] = []
    rejected: list[Rejection] = []
    for facts in exact_methods():
        reason = facts.unsupported_reason(spec)
        if reason is None:
            applicable.append(facts)
        else:
            rejected.append(Rejection(method=facts, reason=reason))
    return MethodSurvey(spec=spec, applicable=tuple(applicable), rejected=tuple(rejected))


def field_sweep_bench(
    cross_check_sites: int = field_sweep.MAX_CROSSCHECK_SITES,
) -> Callable[..., dict[str, Any]]:
    """Lend the exact field sweep to one agent tool, as plain data.

    The seam that lets the agent *plot the exact solution* without ever being able
    to *read* it. :mod:`src.physics.reference.field_sweep` is sealed from everything
    under ``src/agent/``; this returns a closure over it that speaks only
    primitives, and the host -- the interface, a script, an eval -- hands that
    closure to :func:`src.agent.graph.run_campaign`, which offers it to the
    consultation node as a tool. Nothing in the agent's own import graph names
    anything in :mod:`src.physics.reference`, which is what
    the architecture test proves.

    This is not a hole in the wall. The seal exists so that the graded comparison
    means something: an agent that could look up the ground-state energy
    it is being marked against would make the whole experiment worthless. The tool
    this lends is reachable only from the ``consult`` node, which sits in front of
    the two prose branches and is not on the feasibility branch -- the branch
    that runs circuits, spends the shot budget and is graded. So the campaign that
    is marked cannot call it, by the shape of the graph rather than by a promise.
    What it does let the agent do is answer a question about the physics with the
    physics: *plot the low-lying spectrum against the field* is a question with a
    right answer, and an agent that could only gesture at the literature for it was
    failing a reader for no gain in honesty.

    Args:
        cross_check_sites: Longest chain each swept point is *also* solved on by
            exact diagonalisation. The default is the sweep module's own; a caller
            with time to spend can raise it, and the payload says what was checked.

    Returns:
        A callable taking the chain, which curves to trace, how many points and how
        far in :math:`h/J`, and returning the JSON payload of
        :meth:`src.physics.reference.field_sweep.FieldSweep.payload` -- or a
        one-key ``{"error": ...}`` dictionary, which is what a tool hands back to a
        model that can still recover from it.

    Examples:
        >>> bench = field_sweep_bench()
        >>> answered = bench(n_sites=8, curves=["magnetisation"], points=5)
        >>> answered["cross_checked"], len(answered["ratio"])
        (True, 5)

        A chain the closed form cannot solve comes back as an error rather than an
        exception, because the caller is a language model mid-turn:

        >>> "error" in bench(n_sites=7, curves=["energy"], points=5)
        True

        A lattice is swept on the lattice. No closed form exists off a line, so
        every point is diagonalised and the payload names what ran:

        >>> lattice = bench(
        ...     n_sites=9, curves=["energy"], points=5, geometry="square", rows=3
        ... )
        >>> lattice["methods"][0]
        'exact_diagonalisation'
    """

    def sweep(
        n_sites: int,
        coupling: float = 1.0,
        boundary: str = "periodic",
        curves: object = None,
        points: int = field_sweep.DEFAULT_POINTS,
        ratio_max: float = field_sweep.DEFAULT_RATIO_MAX,
        levels: int = field_sweep.DEFAULT_LEVELS,
        geometry: str = "chain",
        rows: int = 1,
    ) -> dict[str, Any]:
        """Trace the field range for one problem and return it as plain data.

        Args:
            n_sites: Total number of sites ``L``.
            coupling: The Ising coupling ``J``.
            boundary: ``"periodic"`` or ``"open"``.
            curves: Which curves to trace; see
                :data:`src.physics.reference.field_sweep.Curve`.
            points: How many field values to sample.
            ratio_max: Top of the swept range in units of ``J``.
            levels: How many many-body levels a spectrum request draws.
            geometry: ``"chain"``, ``"square"`` or ``"triangular"``. Carried
                through because a curve is a curve *of a problem*, and without it
                a question about a lattice's energy against the field was answered
                by sweeping a line of the same size -- then labelled with the
                lattice's name and corroborated by a closed form that only a line
                has. The two energies differ by a factor of nearly two per site,
                so it was not a small error.
            rows: Rows of sites for a two-dimensional shape; one for a chain.

        Returns:
            The payload, or ``{"error": ...}``.
        """
        try:
            spec = TFIMSpec(
                n_sites=n_sites,
                coupling=coupling,
                field=coupling,
                boundary=cast(BoundaryCondition, boundary),
                geometry=cast(Geometry, geometry),
                rows=rows,
            )
        except (TypeError, ValueError) as error:
            return {"error": str(error)}
        return field_sweep.sweep_field(
            spec,
            curves=curves,
            points=points,
            ratio_max=ratio_max,
            levels=levels,
            cross_check_sites=cross_check_sites,
        ).payload()

    return sweep
