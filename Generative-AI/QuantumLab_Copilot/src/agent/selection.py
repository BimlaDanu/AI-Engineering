"""Choosing which method to run. This is policy; the registry holds the facts.

:mod:`src.physics.registry` answers *what could solve this problem, and why can
the rest not?* This module answers the questions that follow, all of which are
judgements rather than physics:

* which applicable method to prefer,
* which of the rest to run as an independent check,
* when a run is expensive enough to ask the human first,
* what to say when nothing applies.

**No language model decides any of this.** A model that picks the solver will
eventually pick a method that does not apply, or invent one that does not exist,
and it will do so in a fluent sentence that reads exactly like a correct answer.
The choice is therefore made by the ranking below, which is deterministic and
testable; the model's job is to explain the choice to the user, not to make it.

The ordering is: **accuracy first, then cost.** A method that is exact beats one
that is merely bounded no matter how cheap the latter is, because this project's
claim is verification and a corroboration between two approximations
corroborates nothing. Cost breaks ties among methods of equal standing.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.physics.model import MAX_SITES_STATEVECTOR, TFIMSpec
from src.physics.registry import (
    COST_ORDER,
    AccuracyClass,
    MethodInfo,
    MethodSurvey,
    Rejection,
    survey,
)

ACCURACY_ORDER: dict[AccuracyClass, int] = {
    "exact": 0,
    "variational_bound": 1,
    "uncontrolled": 2,
}
"""Ranking of accuracy classes, best first.

A variational bound outranks an uncontrolled estimate because it is *checkable*:
it is guaranteed to lie above the true ground-state energy, so it can be
falsified by any exact result at the same size. An uncontrolled approximation
carries no such guarantee and is therefore ranked last however close it happens
to be.
"""

APPROVAL_SITES = 10
"""Chain length at or above which an exponential-cost run needs a human's word.

Not a safety limit -- :data:`~src.physics.model.MAX_SITES_STATEVECTOR` is the
limit, and methods refuse outright beyond it. This is the band just underneath
it, where cost quadruples with every added site and the next request but one
would be refused. Asking here means the user meets the wall in a sentence
rather than in a stalled process.
"""


@dataclass(frozen=True, slots=True)
class Approval:
    """A run the agent will not start without the user's word.

    Attributes:
        method: The method whose cost prompted the question.
        reason: What to show the user, in plain language.
        estimated_memory_bytes: Predicted peak memory, where the method can
            predict it.
    """

    method: MethodInfo
    reason: str
    estimated_memory_bytes: int | None


@dataclass(frozen=True, slots=True)
class Plan:
    """What the agent intends to run for one specification, and why.

    Produced before anything is computed, so it can be shown to the user,
    logged, or refused. Frozen: a node downstream cannot quietly retarget a plan
    that has already been justified to the user.

    Attributes:
        spec: The problem the plan is for.
        chosen: The method whose answer will be reported, or ``None`` if nothing
            applies.
        corroborators: Applicable methods that will also run, purely to check
            the first. Best-ranked first.
        rejected: Methods that declined, each with its reason. Kept because
            "nothing corroborated this" is only useful next to *why*.
        approval: The question to put to the user before running, or ``None``
            if the run is cheap enough to start unattended.
        caveat: A warning that must accompany the answer, or ``None``. Set when
            the chosen method is not exact.
    """

    spec: TFIMSpec
    chosen: MethodInfo | None
    corroborators: tuple[MethodInfo, ...]
    rejected: tuple[Rejection, ...]
    approval: Approval | None
    caveat: str | None

    @property
    def is_runnable(self) -> bool:
        """Whether any method accepted the problem."""
        return self.chosen is not None

    @property
    def methods(self) -> tuple[MethodInfo, ...]:
        """Every method the plan would run, chosen first."""
        if self.chosen is None:
            return ()
        return (self.chosen, *self.corroborators)

    @property
    def will_be_corroborated(self) -> bool:
        """Whether a second, independent method will check the answer.

        Independence is structural here: the registry holds one method per
        module, and two entries never share an implementation. Corroboration by
        a method that reused the first one's algebra would prove only that the
        shared code is deterministic.
        """
        return len(self.corroborators) > 0

    @property
    def needs_approval(self) -> bool:
        """Whether the plan is waiting on a human before it may run."""
        return self.approval is not None

    def justify(self) -> str:
        """Explain the plan in plain text.

        This is the string the agent quotes rather than composes: the reasoning
        is already fixed by the time a model sees it, so the model cannot
        rationalise a different choice than the one that will actually run.

        Returns:
            A multi-line explanation naming the chosen method, the check, the
            unavailable methods and their reasons, and any question or caveat.
        """
        lines = [self.spec.label()]
        if self.chosen is None:
            lines.append("  no method applies; this problem cannot be answered as specified")
            lines.extend(
                f"  {item.method.name} unavailable: {item.reason}" for item in self.rejected
            )
            return "\n".join(lines)

        lines.append(f"  run {self.chosen.name} [{self.chosen.cost} cost, {self.chosen.accuracy}]")
        lines.append(f"    {self.chosen.when_to_use}")
        if self.corroborators:
            names = ", ".join(info.name for info in self.corroborators)
            lines.append(f"  check against {names}, which shares no algebra with it")
        else:
            lines.append("  no independent check available; the answer will be reported unverified")
        lines.extend(f"  {item.method.name} unavailable: {item.reason}" for item in self.rejected)
        if self.caveat is not None:
            lines.append(f"  caveat: {self.caveat}")
        if self.approval is not None:
            lines.append(f"  approval needed: {self.approval.reason}")
        return "\n".join(lines)


def rank(methods: tuple[MethodInfo, ...]) -> tuple[MethodInfo, ...]:
    """Order methods best-first: accuracy, then cost, then registration order.

    Args:
        methods: Applicable methods, in any order.

    Returns:
        The same methods, ranked. Ties fall back to registration order, so the
        result is a total order and two runs of the same query plan identically.
    """
    return tuple(
        sorted(
            methods,
            key=lambda info: (ACCURACY_ORDER[info.accuracy], COST_ORDER[info.cost]),
        )
    )


def approval_for(spec: TFIMSpec, methods: tuple[MethodInfo, ...]) -> Approval | None:
    """Decide whether a planned set of runs should be put to the user first.

    Args:
        spec: The problem to be solved.
        methods: Every method the plan would run.

    Returns:
        The question to ask, or ``None`` if every planned run is cheap. Only the
        most expensive method is reported: two questions about one request is
        one question too many.
    """
    if spec.n_sites < APPROVAL_SITES:
        return None
    costly = [info for info in methods if info.cost == "exponential"]
    if not costly:
        return None
    method = max(costly, key=lambda info: info.memory_bytes(spec) or 0)
    memory = method.memory_bytes(spec)
    footprint = f" and about {memory / 1024**2:.1f} MB" if memory is not None else ""
    return Approval(
        method=method,
        reason=(
            f"{method.name} on L={spec.n_sites} builds a {2**spec.n_sites}-dimensional "
            f"Hilbert space{footprint}. Cost doubles with every added site, and above "
            f"L={MAX_SITES_STATEVECTOR} the method refuses outright."
        ),
        estimated_memory_bytes=memory,
    )


def caveat_for(method: MethodInfo) -> str | None:
    """State what must be said alongside a method's answer, if anything.

    Args:
        method: The method whose answer will be reported.

    Returns:
        A warning, or ``None`` for an exact method. An approximate number
        presented without its status is the failure this project exists to
        prevent, so the caveat travels with the plan rather than being left to
        whatever the model remembers to mention.
    """
    if method.accuracy == "exact":
        return None
    if method.accuracy == "variational_bound":
        return (
            f"{method.name} returns a variational bound: the true ground-state energy is "
            "at or below the number reported, never above it."
        )
    return (
        f"{method.name} carries no rigorous error bound. Treat the number as an "
        "indication, not a result, and do not quote it without this sentence."
    )


def select_from(available: MethodSurvey) -> Plan:
    """Turn a survey into a plan.

    Separate from :func:`select` so that the policy can be tested against a
    constructed survey -- including method classes no solver in this project
    implements yet -- without going anywhere near a real diagonalisation.

    Args:
        available: What the registry said about the problem.

    Returns:
        The plan. Nothing has been computed; this only decides what would be.
    """
    ranked = rank(available.applicable)
    if not ranked:
        return Plan(
            spec=available.spec,
            chosen=None,
            corroborators=(),
            rejected=available.rejected,
            approval=None,
            caveat=None,
        )
    chosen, corroborators = ranked[0], ranked[1:]
    methods = (chosen, *corroborators)
    return Plan(
        spec=available.spec,
        chosen=chosen,
        corroborators=corroborators,
        rejected=available.rejected,
        approval=approval_for(available.spec, methods),
        caveat=caveat_for(chosen),
    )


def select(spec: TFIMSpec) -> Plan:
    """Decide how to solve ``spec``.

    Args:
        spec: The problem to plan for.

    Returns:
        The plan: what to run, what will check it, what to warn about, and
        whether to ask first.

    Examples:
        An even ring admits both methods, and the cheap exact one leads:

        >>> from src.physics.model import TFIMSpec
        >>> plan = select(TFIMSpec(n_sites=4))
        >>> plan.chosen is not None and plan.chosen.name
        'pfeuty_exact'
        >>> plan.will_be_corroborated
        True

        An odd chain leaves one method, so the answer will be unchecked:

        >>> select(TFIMSpec(n_sites=5)).will_be_corroborated
        False
    """
    return select_from(survey(spec))
