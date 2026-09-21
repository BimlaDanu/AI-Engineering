"""Solving a second chain, so a question about *size* can be answered.

One run computes one chain, because the chain comes from the settings knob rather
than from the question -- a deliberate choice, since a model that mis-reads
"L = 12" as 10 produces a confidently wrong number. That leaves one honest
question unanswerable: *how does this change as the chain gets longer?* Finite-size
behaviour is most of what a small exact calculation is good for, and answering it
needs a second and a third point.

This module provides them. The Hamiltonian is not up for negotiation -- ``J``,
``h`` and the boundary are inherited from the chain the user set on screen -- and
only the length varies. So the knob still owns the physics, and the tool answers
the one thing the knob cannot express in a single run.

**A number from here is cross-checked like any other.** It goes through the same
:func:`~src.verification.cross_check.cross_check`, so a comparison point carries
its own corroboration and is not a cheaper class of result.

**A tool cannot spend what a human has not approved.** Beyond
:data:`MAX_COMPARISON_SITES` this refuses, rather than starting a run whose cost
the interface would normally stop and ask about. There is nobody to ask inside a
tool call, and treating silence as consent is how a cost gate becomes decorative.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.physics.model import TFIMSpec
from src.physics.registry import survey
from src.verification.cross_check import CrossCheck, cross_check

MAX_COMPARISON_SITES = 9
"""Longest chain this tool will solve unattended.

One site below :data:`src.agent.selection.APPROVAL_SITES`, the length at which an
exponential-cost run stops to ask the user first. The relationship is the point,
not the number: this tool stops just short of the gate instead of walking through
it. A test pins the two together so raising one cannot silently bypass the other.
"""

MIN_COMPARISON_SITES = 2
"""Shortest chain worth comparing. Below two sites there is no bond to couple."""


@dataclass(frozen=True, slots=True)
class ChainRun:
    """A comparison point: one extra chain, solved and checked, or refused.

    Attributes:
        spec: The chain that was asked for.
        check: The verified result, or ``None`` when the run was declined.
        detail: One sentence on what happened. Populated either way, and it is
            the refusal message when :attr:`check` is ``None``.
    """

    spec: TFIMSpec | None
    check: CrossCheck | None
    detail: str

    @property
    def ok(self) -> bool:
        """Whether a number was actually produced."""
        return self.check is not None

    def explain(self) -> str:
        """Say what this comparison point is, in one line.

        Returns:
            The chain, its energy and whether two methods agreed -- or the reason
            nothing ran.
        """
        if self.check is None:
            return f"no comparison run: {self.detail}"
        energy = self.check.energy
        value = "no energy" if energy is None else f"E0 = {energy:.9f}"
        agreed = "corroborated" if self.check.is_corroborated else "unverified"
        return f"{self.check.spec.label()}: {value} ({agreed})"


def compare_chain(base: TFIMSpec, n_sites: int) -> ChainRun:
    """Solve the same Hamiltonian at a different chain length.

    Args:
        base: The chain the run is about. Its coupling, field and boundary are
            carried over unchanged; only the length is replaced.
        n_sites: The length to solve instead.

    Returns:
        The comparison point, or a refusal carrying its reason. Never raises: a
        length outside the allowed band, a chain no method accepts and an invalid
        specification all come back as a declined :class:`ChainRun`, because this
        runs inside a tool call where an exception would take down an answer that
        was otherwise fine.

    Examples:
        >>> run = compare_chain(TFIMSpec(n_sites=8), 4)
        >>> run.ok and run.check is not None and run.check.spec.n_sites
        4
        >>> compare_chain(TFIMSpec(n_sites=8), 8).detail
        'that is the chain already solved, so there is nothing to compare with'
        >>> compare_chain(TFIMSpec(n_sites=8), 12).ok
        False
    """
    if n_sites == base.n_sites:
        return ChainRun(
            spec=None,
            check=None,
            detail="that is the chain already solved, so there is nothing to compare with",
        )
    if not MIN_COMPARISON_SITES <= n_sites <= MAX_COMPARISON_SITES:
        return ChainRun(
            spec=None,
            check=None,
            detail=(
                f"a comparison chain must have {MIN_COMPARISON_SITES} to "
                f"{MAX_COMPARISON_SITES} sites; longer runs need the user's approval, "
                "which is granted through the settings knob and not inside a tool call"
            ),
        )
    try:
        # Constructed rather than copied with an update: ``model_copy`` skips
        # validation, and the length here came from a model.
        spec = TFIMSpec(
            n_sites=n_sites,
            coupling=base.coupling,
            field=base.field,
            boundary=base.boundary,
        )
    except Exception as error:
        return ChainRun(
            spec=None,
            check=None,
            detail=f"that chain is not a valid specification ({type(error).__name__})",
        )
    if not survey(spec).applicable:
        return ChainRun(
            spec=spec,
            check=None,
            detail=f"no method applies to {spec.label()}",
        )
    return ChainRun(
        spec=spec,
        check=cross_check(spec),
        detail=f"solved {spec.label()} for comparison",
    )
