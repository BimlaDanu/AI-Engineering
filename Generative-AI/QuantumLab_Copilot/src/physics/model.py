"""The problem specification every physics method consumes.

A single frozen dataclass is the contract between the agent layer and the
physics layer. The agent's job is to turn a sentence into one of these; the
physics layer's job is to turn one of these into a number. Keeping the two
sides apart behind an immutable, validated value object is what makes the
result cacheable, the run reproducible and the whole thing testable without a
language model in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

BoundaryCondition = Literal["periodic", "open"]
"""Lattice boundary condition. Periodic closes the chain into a ring."""

HAMILTONIAN_LATEX = (
    r"\hat H = -J \sum_{i=1}^{L} \hat\sigma^z_i \hat\sigma^z_{i+1} "
    r"- h \sum_{i=1}^{L} \hat\sigma^x_i"
)
r"""The project's Hamiltonian as LaTeX, with no delimiters around it.

One string, imported by the page caption, the agent's self-description and the
answer it composes, because the same equation written out three times drifts three
ways -- and the version a reader met first is the one they will trust.

Operators are written :math:`\hat\sigma^z_i`, not ``Z_i``: both are standard and
mean the same Pauli matrix, but the hatted form is what a physics reader expects to
see in a Hamiltonian, and the letter form reads as a *gate* to anyone coming from
the quantum-computing side. The sums carry their limits explicitly.

``J`` is kept even though it is 1 by default: it is a setting the user can change,
and an equation that hides the coupling cannot explain the number that comes back
when they do.
"""

HAMILTONIAN_INLINE = f"${HAMILTONIAN_LATEX}$"
"""The Hamiltonian for the middle of a sentence."""

HAMILTONIAN_DISPLAY = f"$$\n{HAMILTONIAN_LATEX}\n$$"
"""The Hamiltonian as its own centred line.

Streamlit renders ``$$`` as display math and ``$`` as inline, and nothing else:
``\\(...\\)`` and ``\\[...\\]`` are shown as literal backslashes and brackets. The
delimiters therefore live here rather than at each call site, where the wrong pair
is a rendering bug nobody notices until a screenshot.
"""

MAX_SITES_STATEVECTOR = 12
"""Hard cap on ``L`` for any method that materialises a ``2**L`` state vector.

At ``L = 12`` a dense real Hamiltonian is 4096 x 4096, about 134 MB, and
diagonalises in a second or two; every extra site multiplies the memory by four.
The cap is deliberately conservative so that a run can never take the machine
down, and it is enforced by each method's applicability check rather than here,
because methods whose cost is ``O(L)`` -- the free-fermion solver -- are
legitimately unbounded and must not be caught by it.
"""


@dataclass(frozen=True, slots=True)
class TFIMSpec:
    r"""A fully specified one-dimensional transverse-field Ising model.

    The Hamiltonian, in the convention used everywhere in this project, is

    .. math::

        H = -J \sum_i \sigma^z_i \sigma^z_{i+1} \; - \; h \sum_i \sigma^x_i

    where :math:`\sigma^z` and :math:`\sigma^x` are Pauli matrices. Published work
    is split between this convention and the mirrored one, which puts the coupling
    on :math:`\sigma^x` and the field on :math:`\sigma^z`. The two are related by a
    global basis rotation and give identical spectra, but mixing them silently is a
    classic source of wrong answers. This project commits to the form above and
    converts at the boundary rather than tolerating both internally.

    The instance is frozen so it can be hashed into a run cache: two identical
    specs must always produce the same answer.

    Attributes:
        n_sites: Number of spins in the chain, ``L``. At least 2.
        coupling: The Ising coupling ``J``. Strictly positive (ferromagnetic).
        field: The transverse field ``h``. Non-negative.
        boundary: ``"periodic"`` (a ring) or ``"open"`` (a segment).
    """

    n_sites: int
    coupling: float = 1.0
    field: float = 1.0
    boundary: BoundaryCondition = "periodic"

    def __post_init__(self) -> None:
        """Reject physically or numerically meaningless specifications.

        Raises:
            TypeError: If ``n_sites`` is not an integer.
            ValueError: If any field is outside its permitted range.
        """
        if not isinstance(self.n_sites, int) or isinstance(self.n_sites, bool):
            raise TypeError(f"n_sites must be an int, got {type(self.n_sites).__name__}")
        if self.n_sites < 2:
            raise ValueError(f"n_sites must be at least 2, got {self.n_sites}")
        if self.coupling <= 0.0:
            raise ValueError(f"coupling J must be strictly positive, got {self.coupling}")
        if self.field < 0.0:
            raise ValueError(f"field h must be non-negative, got {self.field}")
        if self.boundary not in get_args(BoundaryCondition):
            raise ValueError(
                f"boundary must be one of {get_args(BoundaryCondition)}, got {self.boundary!r}"
            )

    @property
    def ratio(self) -> float:
        """The dimensionless control parameter ``g = h / J``.

        The quantum critical point of the infinite chain sits at ``g = 1``,
        where Kramers-Wannier self-duality pins it exactly.
        """
        return self.field / self.coupling

    @property
    def n_bonds(self) -> int:
        """Number of Ising bonds: ``L`` for a ring, ``L - 1`` for a segment."""
        return self.n_sites if self.boundary == "periodic" else self.n_sites - 1

    def label(self) -> str:
        """A short human-readable identifier, used in logs and plot legends."""
        return f"TFIM L={self.n_sites} J={self.coupling:g} h={self.field:g} ({self.boundary[:4]})"
