"""What methods exist, when each applies, and what each costs.

This module is the seam the agent's method-selection step reads. It answers one
question -- *given this problem, what could solve it, and why can the rest not?*
-- and it answers it in pure Python, with no language model, no API key and no
network. That is deliberate: applicability is a physics fact, and a fact the
agent is allowed to invent is a fact that will eventually be wrong.

**Facts, not policy.** The registry deliberately does not choose. It reports
that two methods apply, that one is ``linear`` and the other ``exponential``,
and that a third was refused with a specific reason. *Preferring* the cheap one,
asking the user before an expensive run, or giving up when nothing applies is
policy and lives in :mod:`src.agent`. Keeping the line here means a notebook can
do method selection with no agent at all, and means the selection logic can be
unit-tested against a fake registry.

Registration is an explicit tuple, not auto-discovery. A method that nobody
listed does not silently appear in the agent's menu.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from src.physics import ed, exact
from src.physics.model import TFIMSpec

CostClass = Literal["constant", "linear", "exponential"]
"""How a method's cost scales with chain length ``L``.

``constant`` closes the problem in closed form regardless of size; ``linear``
does ``O(L)`` arithmetic; ``exponential`` builds an object of dimension
``2**L`` and is therefore always size-capped.
"""

AccuracyClass = Literal["exact", "variational_bound", "uncontrolled"]
"""What kind of number a method returns.

``exact`` is right to machine precision. ``variational_bound`` is guaranteed to
lie above the true ground-state energy, which makes it checkable at any size
with no reference. ``uncontrolled`` carries an error with no rigorous bound --
mean-field theory near criticality -- and must never be reported without that
caveat attached.
"""

COST_ORDER: dict[CostClass, int] = {"constant": 0, "linear": 1, "exponential": 2}
"""Ranking of :data:`CostClass` from cheapest to most expensive.

Exposed as a fact so that policy code can sort by it. That cheap beats
expensive *when accuracy is equal* is a judgement, and it is not made here.
"""


class Method(Protocol):
    """The contract every physics module in this package satisfies.

    Modules -- not classes -- implement this: :mod:`src.physics.exact` and
    :mod:`src.physics.ed` are both valid :class:`Method` values. Stating it as a
    protocol lets the type checker prove that a newly added solver exposes the
    right names before any test runs.

    Attributes:
        METHOD_NAME: The registry key, defined beside the implementation so the
            two cannot drift apart.
    """

    METHOD_NAME: str

    def unsupported_reason(self, spec: TFIMSpec) -> str | None:
        """Explain why the method cannot handle ``spec``, or return ``None``."""
        ...

    def ground_state_energy(self, spec: TFIMSpec) -> float:
        """Return the ground-state energy of ``spec``."""
        ...


_CONFORMANCE: tuple[Method, ...] = (exact, ed)
"""Static proof that each registered module satisfies :class:`Method`.

Never read at runtime. It exists so that mypy fails the build if a solver drops
or renames one of the contract's names, which a test could only catch by
importing every module by hand.
"""


@dataclass(frozen=True, slots=True)
class MethodInfo:
    """Everything the agent knows about one solver without running it.

    Frozen so it can be hashed and cached, and so a node cannot accidentally
    rewrite the registry it was handed.

    Attributes:
        name: The registry key, matching the module's ``METHOD_NAME``.
        summary: One sentence on what the method actually does.
        when_to_use: One sentence on the situation in which it is the right
            choice. This is the text the agent paraphrases when it justifies a
            selection to the user.
        cost: How the cost scales with ``L``.
        accuracy: What kind of number comes back.
        unsupported_reason: The applicability check, taken from the module.
        ground_state_energy: The solver entry point, taken from the module.
        estimate_memory_bytes: Peak-memory predictor, or ``None`` for methods
            whose memory use is trivially small.
    """

    name: str
    summary: str
    when_to_use: str
    cost: CostClass
    accuracy: AccuracyClass
    unsupported_reason: Callable[[TFIMSpec], str | None]
    ground_state_energy: Callable[[TFIMSpec], float]
    estimate_memory_bytes: Callable[[TFIMSpec], int] | None = None

    def applies_to(self, spec: TFIMSpec) -> bool:
        """Report whether this method can solve ``spec``.

        Args:
            spec: The problem to check.

        Returns:
            ``True`` if the method's applicability check raises no objection.
        """
        return self.unsupported_reason(spec) is None

    def memory_bytes(self, spec: TFIMSpec) -> int | None:
        """Predict peak memory for a run, if the method can predict it.

        Args:
            spec: The problem to cost.

        Returns:
            An approximate byte count, or ``None`` if the method declares no
            estimator because its footprint is negligible at any size.
        """
        if self.estimate_memory_bytes is None:
            return None
        return self.estimate_memory_bytes(spec)


@dataclass(frozen=True, slots=True)
class Rejection:
    """A method that cannot solve the problem, together with its reason.

    Rejections are carried around rather than discarded because they are the
    most useful thing the agent can say when nothing applies: "exact
    diagonalisation was refused because the Hilbert space would be 2**30" is an
    answer, whereas "no method available" is a dead end.

    Attributes:
        method: The method that declined.
        reason: The sentence its ``unsupported_reason`` returned.
    """

    method: MethodInfo
    reason: str


@dataclass(frozen=True, slots=True)
class MethodSurvey:
    """The full picture of what can and cannot solve one specification.

    Attributes:
        spec: The problem that was surveyed.
        applicable: Methods that accepted it, in registration order. The order
            carries no preference; ranking is policy.
        rejected: Methods that declined it, each with its reason.
    """

    spec: TFIMSpec
    applicable: tuple[MethodInfo, ...]
    rejected: tuple[Rejection, ...]

    @property
    def has_applicable_method(self) -> bool:
        """Whether at least one registered method can solve the problem."""
        return len(self.applicable) > 0

    def names(self) -> tuple[str, ...]:
        """Return the names of the applicable methods, in registration order."""
        return tuple(info.name for info in self.applicable)

    def describe(self) -> str:
        """Render the survey as plain text for a prompt or a log line.

        The output is deterministic and contains no floating-point noise, so it
        is safe to include in a cached prompt and to assert on in a test.

        Returns:
            A multi-line description naming every method, its cost and accuracy
            class, and -- for those that declined -- why.
        """
        lines = [self.spec.label()]
        lines.append("  available:")
        if not self.applicable:
            lines.append("    (none)")
        for info in self.applicable:
            memory = info.memory_bytes(self.spec)
            footprint = f", ~{_format_bytes(memory)}" if memory is not None else ""
            lines.append(f"    {info.name} [{info.cost} cost, {info.accuracy}{footprint}]")
            lines.append(f"      {info.when_to_use}")
        if self.rejected:
            lines.append("  unavailable:")
            lines.extend(f"    {item.method.name}: {item.reason}" for item in self.rejected)
        return "\n".join(lines)


def _format_bytes(count: int) -> str:
    """Render a byte count in the largest unit that keeps it above one.

    Args:
        count: A non-negative number of bytes.

    Returns:
        A short string such as ``"1.4 MB"``. Rounded to one decimal place
        because these are order-of-magnitude estimates, not measurements.
    """
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


_REGISTRY: tuple[MethodInfo, ...] = (
    MethodInfo(
        name=exact.METHOD_NAME,
        summary=(
            "Closed-form free-fermion solution: a Jordan-Wigner transformation maps the "
            "chain to non-interacting fermions and a Bogoliubov rotation diagonalises them."
        ),
        when_to_use=(
            "The first choice for any uniform TFIM ring of even length. It costs O(L) at "
            "every size and never approximates, so when it applies nothing else can beat "
            "it -- and it is the reference every other method is measured against."
        ),
        cost="linear",
        accuracy="exact",
        unsupported_reason=exact.unsupported_reason,
        ground_state_energy=exact.ground_state_energy,
    ),
    MethodInfo(
        name=ed.METHOD_NAME,
        summary=(
            "Sparse exact diagonalisation: the full Hamiltonian is assembled in CSR form "
            "and its lowest eigenpair found by Lanczos iteration."
        ),
        when_to_use=(
            "Small chains the closed form does not cover -- open boundaries, odd L -- and "
            "as an independent check on it, since the two share no algebra. The Hilbert "
            "space grows as 2**L, so it is refused above the project's size cap."
        ),
        cost="exponential",
        accuracy="exact",
        unsupported_reason=ed.unsupported_reason,
        ground_state_energy=ed.ground_state_energy,
        estimate_memory_bytes=ed.estimate_memory_bytes,
    ),
)


def all_methods() -> tuple[MethodInfo, ...]:
    """Return every registered method, in registration order.

    Returns:
        The full registry. The tuple is immutable, so callers cannot extend the
        agent's menu by accident.
    """
    return _REGISTRY


def method_names() -> tuple[str, ...]:
    """Return the names of every registered method, in registration order."""
    return tuple(info.name for info in _REGISTRY)


def get_method(name: str) -> MethodInfo:
    """Look up one method by its registry key.

    Args:
        name: A registry key, as returned by :func:`method_names`.

    Returns:
        The matching method's facts.

    Raises:
        KeyError: If no method is registered under that name. The message lists
            the valid names, because this is the failure mode when a language
            model hallucinates a solver.
    """
    for info in _REGISTRY:
        if info.name == name:
            return info
    raise KeyError(f"unknown method {name!r}; registered methods are {method_names()}")


def survey(spec: TFIMSpec) -> MethodSurvey:
    """Ask every registered method whether it can solve ``spec``.

    This is the single entry point the agent's selection step uses. One call
    yields both the menu and the reasons behind everything absent from it.

    Args:
        spec: The problem to survey.

    Returns:
        The applicable methods and the rejections, both in registration order.

    Examples:
        A periodic ring of even length is solvable both ways:

        >>> from src.physics.model import TFIMSpec
        >>> survey(TFIMSpec(n_sites=4)).names()
        ('pfeuty_exact', 'exact_diagonalisation')

        An open chain falls outside the closed form, and the survey says so:

        >>> result = survey(TFIMSpec(n_sites=4, boundary="open"))
        >>> result.names()
        ('exact_diagonalisation',)
        >>> "periodic ring" in result.rejected[0].reason
        True
    """
    applicable: list[MethodInfo] = []
    rejected: list[Rejection] = []
    for info in _REGISTRY:
        reason = info.unsupported_reason(spec)
        if reason is None:
            applicable.append(info)
        else:
            rejected.append(Rejection(method=info, reason=reason))
    return MethodSurvey(spec=spec, applicable=tuple(applicable), rejected=tuple(rejected))
