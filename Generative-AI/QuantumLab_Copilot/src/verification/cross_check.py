"""Run every applicable method and compare the answers.

This is the project's central claim, expressed as one function: a number is
reported only when at least two methods that *share no algebra* produce it. The
closed-form free-fermion solution and sparse exact diagonalisation have no code
in common -- one does ``O(L)`` arithmetic over momenta, the other builds a
``2**L`` matrix and iterates -- so their agreement is evidence, not a tautology.

There is no language model here, and there is no policy here either. This
module runs what the registry says is applicable and reports what came back;
deciding whether an uncorroborated answer is good enough to show is the agent's
call.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.physics.model import TFIMSpec
from src.physics.registry import Rejection, survey

AGREEMENT_TOLERANCE = 1e-9
"""Relative tolerance at which two methods count as agreeing.

Generous by five orders of magnitude: the two implemented methods agree to
about ``2.5e-14``, near the floor set by accumulated floating-point error. The
gap is headroom for a future method with a looser but still exact-in-principle
convergence criterion, and it is still far tighter than any physically
meaningful difference.
"""


@dataclass(frozen=True, slots=True)
class MethodResult:
    """One method's answer for one specification.

    Attributes:
        method: The registry name of the method that produced the number.
        energy: The ground-state energy it returned.
    """

    method: str
    energy: float


@dataclass(frozen=True, slots=True)
class CrossCheck:
    """What every applicable method said, and whether they agreed.

    Attributes:
        spec: The problem that was solved.
        results: One entry per applicable method, in registration order.
        rejected: Methods that declined the problem, each with its reason.
            Carried through because "nothing corroborated this" is only
            actionable alongside *why* the other methods were unavailable.
    """

    spec: TFIMSpec
    results: tuple[MethodResult, ...]
    rejected: tuple[Rejection, ...]

    @property
    def energy(self) -> float | None:
        """The ground-state energy, or ``None`` if no method could solve it.

        Where several methods ran, the first is returned; they agree to within
        :data:`AGREEMENT_TOLERANCE` whenever :attr:`is_corroborated` holds, so
        the choice is immaterial.
        """
        return self.results[0].energy if self.results else None

    @property
    def max_disagreement(self) -> float | None:
        """The largest absolute gap between any two answers.

        Returns:
            The spread, or ``None`` if fewer than two methods ran and there is
            therefore nothing to compare.
        """
        if len(self.results) < 2:
            return None
        energies = [result.energy for result in self.results]
        return max(energies) - min(energies)

    @property
    def is_corroborated(self) -> bool:
        """Whether at least two independent methods agree on the answer.

        A single unopposed answer is *not* corroborated, however exact the
        method claims to be. That is the point of the check.
        """
        spread = self.max_disagreement
        if spread is None:
            return False
        scale = max(1.0, *(abs(result.energy) for result in self.results))
        return spread <= AGREEMENT_TOLERANCE * scale

    def summary(self) -> str:
        """Render the outcome as plain text for a log line or a prompt.

        Deterministic, so it is safe to assert on and safe to cache.

        Returns:
            A multi-line report naming each method, its answer, and the verdict.
        """
        lines = [self.spec.label()]
        if not self.results:
            lines.append("  no method could solve this specification")
        for result in self.results:
            lines.append(f"  {result.method}: E0 = {result.energy:.12f}")
        for item in self.rejected:
            lines.append(f"  {item.method.name} unavailable: {item.reason}")
        spread = self.max_disagreement
        if spread is None:
            lines.append("  UNCORROBORATED: only one method applies, nothing to check against")
        elif self.is_corroborated:
            lines.append(
                f"  VERIFIED: {len(self.results)} independent methods agree to {spread:.1e}"
            )
        else:
            lines.append(f"  CONTRADICTION: methods disagree by {spread:.1e}")
        return "\n".join(lines)


def cross_check(spec: TFIMSpec) -> CrossCheck:
    """Solve ``spec`` with every method that accepts it and compare.

    Args:
        spec: The problem to solve.

    Returns:
        Every answer obtained, every refusal and its reason, and the verdict.

    Examples:
        An even ring is solvable two independent ways, and they agree:

        >>> from src.physics.model import TFIMSpec
        >>> check = cross_check(TFIMSpec(n_sites=4, field=0.7))
        >>> check.is_corroborated
        True

        An odd chain leaves only one method, so nothing corroborates it:

        >>> cross_check(TFIMSpec(n_sites=5)).is_corroborated
        False
    """
    available = survey(spec)
    results = tuple(
        MethodResult(method=info.name, energy=info.ground_state_energy(spec))
        for info in available.applicable
    )
    return CrossCheck(spec=spec, results=results, rejected=available.rejected)
