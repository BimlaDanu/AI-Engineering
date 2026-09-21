r"""Where the classical baseline can be trusted, and where it cannot.

The classical arm is the number a quantum feasibility claim has to beat, so the
worst thing it can do is return a plausible number from outside the regime it was
built for. A missing baseline is caught -- :func:`src.agent.verdict.judge` refuses
a verdict without one -- but a silently degraded one is caught by nothing, and it
fails in the flattering direction: looser and noisier means easier to beat.

:func:`assess` returns a :class:`Regime`, a standing plus the reason, which
travels on the result and is read by the verdict and the report. It holds no
solver and no energy, so every layer may import it.

The concrete hazard:
:func:`src.physics.classical.variational_imaginary_time.optimise` builds its dual
lattice from ``spec.n_sites`` alone, so a ``4 x 4`` square would be sampled as a
line of sixteen and reported as the square\'s classical result, with the energy in
range and the error bar small.

There is no sign problem here. Frustration gives quantum Monte Carlo a sign
problem on Heisenberg antiferromagnets, not on this model: the transverse-field
Ising Hamiltonian is diagonal in the :math:`\sigma^z` basis apart from the field,
so every off-diagonal element is :math:`-h`, negative on any graph and for either
sign of :math:`J`. Such a Hamiltonian is stoquastic -- :math:`e^{-\beta \hat H}`
has non-negative entries, the path-integral weights have no sign to cancel, and
frustration moves only the diagonal. The regime test asserts
this from an assembled matrix. What frustration costs is slow mixing, see
:data:`FRUSTRATION_CAVEAT`, which is enough: an understated error bar is an
understated bar for claiming a lead.

:func:`assess` cannot currently return ``degraded``. Frustration needs a graph
that does not two-colour, and every such graph is refused earlier by a rule that
is not about frustration; a test pins that rather than letting the branch look
live. It is kept because it is the rule a two-dimensional sampler must obey, and
because its text is appended to the refusal of a frustrated shape, so a reader
learns both why nothing ran and why running it would not have settled anything.

The live route to a low-confidence baseline is an error bar that is not a number:
see :attr:`src.agent.state.BaselineResult.uncertainty_is_usable`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.physics.lattice import Lattice

Standing = Literal["sound", "degraded", "unavailable"]
"""How far the classical baseline's number may be trusted.

``sound``
    The regime the sampler was built and calibrated for. Use the number as it stands.

``degraded``
    It runs and returns a real variational upper bound, but a quantity the
    comparison depends on is unreliable -- in practice the error bar. A verdict may
    be reported and may not be reported *confidently*.

``unavailable``
    It cannot represent this problem at all, and must refuse rather than return the
    number it would have produced for a different one.
"""

CONFIDENCE_BY_STANDING: dict[Standing, Literal["low", "medium", "high"]] = {
    "sound": "high",
    "degraded": "low",
    "unavailable": "low",
}
"""The confidence each standing licenses, mapped in one place.

``degraded`` maps to ``low`` rather than ``medium`` deliberately. The failure mode is
an under-reported error bar, and :func:`src.agent.verdict._compare` divides by that
bar -- so the quantity that is wrong is the one the verdict is most sensitive to.
"medium" would invite a reader to split a difference that does not exist.
"""

FRUSTRATION_CAVEAT = (
    " And even with that axis built, the number would not be trustworthy on this "
    "shape: frustration means no arrangement of the spins satisfies every connection "
    "at once, so there are many nearly-equal-cost arrangements separated by barriers. "
    "The sampler moves one spin at a time and cannot cross them, so successive "
    "measurements stay correlated -- and the error bar is computed on the assumption "
    "that they do not. It is therefore under-reported, which makes the classical "
    "number look more precise than it is and a quantum lead over it easier to claim. "
    "This is slow mixing, not the sign problem: this model has no sign problem on any "
    "shape."
)
"""What frustration costs a sampled baseline, in one place and two uses.

It is the whole ``detail`` of a ``degraded`` standing, and it is *appended* to the
refusal of a two-dimensional shape -- where the reader deserves both facts, since a
refusal giving only "no third axis" leaves somebody thinking that is all this needs.
Written to read correctly in both positions, hence the leading space and
conjunction; the ``degraded`` branch strips them.
"""


@dataclass(frozen=True, slots=True)
class Regime:
    """Whether the classical baseline may be believed here, and why.

    Carried alongside the energy rather than checked at the call site, because a
    caller that has to remember to ask is a caller that will forget once.

    Attributes:
        standing: How far the number may be trusted. See :data:`Standing`.
        reason: Why, in words a reader with no physics can act on. Printed into the
            report verbatim, so it is a sentence rather than a label.
        detail: The mechanism, for a reader who wants it. Separate from
            :attr:`reason` so a summary can show one and a report both.
    """

    standing: Standing
    reason: str
    detail: str = ""

    @property
    def confidence(self) -> Literal["low", "medium", "high"]:
        """The confidence this standing licenses. See :data:`CONFIDENCE_BY_STANDING`."""
        return CONFIDENCE_BY_STANDING[self.standing]

    @property
    def is_trustworthy(self) -> bool:
        """Whether a verdict may lean on this baseline without qualification.

        Examples:
            >>> assess(Lattice("chain", 1, 8), coupling=1.0).is_trustworthy
            True
            >>> assess(Lattice("square", 2, 2), coupling=1.0).is_trustworthy
            False
        """
        return self.standing == "sound"

    @property
    def can_run(self) -> bool:
        """Whether the sampler can produce a number for this problem at all.

        Examples:
            >>> assess(Lattice("chain", 1, 8, "periodic"), coupling=-1.0).can_run
            True
            >>> assess(Lattice("triangular", 3, 3), coupling=-1.0).can_run
            False
        """
        return self.standing != "unavailable"


class BaselineUnavailableError(RuntimeError):
    """Raised when the baseline is asked for a problem it cannot represent.

    An exception rather than a number with a warning attached. A caveat beside a
    plausible figure is read as a formality on a result, when what happened is that
    there is no result. :func:`src.agent.graph.run_classical_baseline` catches it and leaves
    ``state["classical"]`` as ``None``, so the existing no-comparison screen fires.

    Attributes:
        regime: The standing that caused the refusal, so a caller can print the
            reason rather than invent one.
    """

    def __init__(self, regime: Regime) -> None:
        super().__init__(regime.reason)
        self.regime = regime


def assess(lattice: Lattice, coupling: float, transverse_field: float = 1.0) -> Regime:
    r"""Decide how far the classical baseline may be trusted on this problem.

    The order is the claim: *can it run at all* before *can it be believed*, since a
    shape the sampler cannot represent is not a question about accuracy.

    Args:
        lattice: The spatial geometry asked for. This is
            :class:`src.physics.lattice.Lattice` -- the *physical* lattice -- not
            :class:`src.physics.classical.dual_lattice.Lattice`, which is what the
            sampler builds from it and carries imaginary time as its second axis.
        coupling: The Ising coupling :math:`J`. Negative is antiferromagnetic in
            this project's convention, where the term is
            :math:`-J \sum \sigma^z \sigma^z`.
        transverse_field: The field :math:`h`. Taken only for the one case where
            there is nothing to sample.

    Returns:
        The standing, with the reason attached.

    Examples:
        The calibration case, which is what the sampler was built for:

        >>> assess(Lattice("chain", 1, 12), coupling=1.0).standing
        'sound'

        A shape whose classical dual needs three axes, which is not built:

        >>> assess(Lattice("square", 4, 4), coupling=1.0).standing
        'unavailable'

        An odd ring, which the checkerboard sweep cannot update correctly:

        >>> assess(Lattice("chain", 1, 9, "periodic"), coupling=-1.0).standing
        'unavailable'

        An even ring is bipartite, so neither rule fires:

        >>> assess(Lattice("chain", 1, 8, "periodic"), coupling=-1.0).standing
        'sound'
    """
    if lattice.geometry != "chain" and lattice.rows > 1:
        return _needs_a_third_axis(lattice, coupling)
    if _is_an_odd_ring(lattice):
        return _breaks_the_checkerboard(lattice)
    if transverse_field == 0.0:
        return _nothing_to_sample()
    if lattice.frustrated_by(coupling):
        return _frustrated(lattice)
    return _calibrated(lattice)


def _is_an_odd_ring(lattice: Lattice) -> bool:
    """Whether the wrap closes an odd cycle, which the checkerboard cannot colour.

    Two sites are excluded rather than handled: a ring of two is degenerate, and
    :meth:`src.physics.lattice.Lattice.bonds` skips the wrap there anyway, so it is
    not a ring at all. The condition mirrors
    :func:`src.physics.method_catalogue.variational_imaginary_time_unsupported_reason`
    and a test pins the two together.
    """
    return lattice.boundary == "periodic" and lattice.n_sites > 2 and lattice.n_sites % 2 != 0


def _needs_a_third_axis(lattice: Lattice, coupling: float) -> Regime:
    """A two-dimensional shape, whose classical dual this module does not build."""
    return Regime(
        standing="unavailable",
        reason=(
            f"The classical method cannot be run on a {lattice.describe()}, so there "
            "is no classical number to compare against on this shape."
        ),
        detail=(
            "Its sampler works a classical lattice with one row of sites and "
            "imaginary time as the second axis. A two-dimensional lattice needs a "
            "third axis, which is not implemented -- and running it as a line of "
            f"{lattice.n_sites} sites instead would answer a different question "
            "while looking correct."
        )
        + (FRUSTRATION_CAVEAT if lattice.frustrated_by(coupling) else ""),
    )


def _breaks_the_checkerboard(lattice: Lattice) -> Regime:
    """An odd ring, where updating every other site at once is not a valid move."""
    return Regime(
        standing="unavailable",
        reason=(
            f"The classical method cannot be run on a ring of {lattice.n_sites}, so "
            "there is no classical number to compare against at this size."
        ),
        detail=(
            "Its sampler updates every other site at once, which is only correct if "
            "no two sites in the same group are connected. An odd ring closes back on "
            "itself so that two of them are, and the update would quietly correlate "
            "rather than fail."
        ),
    )


def _nothing_to_sample() -> Regime:
    """Zero transverse field, where the model is classical and the answer arithmetic."""
    return Regime(
        standing="unavailable",
        reason=(
            "With no field across the coupling direction this problem is not a "
            "quantum one, and the sampled method does not apply to it."
        ),
        detail=(
            "The estimator the sampler uses for the field term is multiplied by the "
            "field, so at zero field it measures nothing. The lowest energy is then "
            "found by arithmetic on the bond list rather than by sampling."
        ),
    )


def _frustrated(lattice: Lattice) -> Regime:
    """A runnable shape whose error bar cannot be believed. See the module docstring."""
    return Regime(
        standing="degraded",
        reason=(
            f"The classical comparison could not be made reliably on a "
            f"{lattice.describe()} with this coupling: the shape frustrates it, so "
            "the uncertainty the method reports is smaller than its real one."
        ),
        detail=FRUSTRATION_CAVEAT.strip(),
    )


def _calibrated(lattice: Lattice) -> Regime:
    """The regime the sampler was built and checked on."""
    return Regime(
        standing="sound",
        reason=(
            f"The classical method is inside the regime it was calibrated on "
            f"({lattice.describe()})."
        ),
        detail=(
            "One row of sites, and no frustration: every sampled weight is positive, "
            "successive measurements decorrelate within a bin, and the reported "
            "uncertainty means what it says."
        ),
    )
