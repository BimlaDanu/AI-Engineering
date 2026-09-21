r"""The quantum-to-classical mapping: one chain becomes one classical lattice.

The variational imaginary-time ansatz for the transverse-field Ising chain is

.. math::

    |\psi_P(\alpha, \beta)\rangle = \mathcal{N} \prod_{p=P}^{1}
        e^{-\beta_p \hat H_{\rm field}}\, e^{-\alpha_p \hat H_{\rm diag}}\, |+\rangle^{\otimes L},

with the two generators taken from the project\'s Hamiltonian with its overall
scales divided out,

.. math::

    \hat H_{\rm diag} = -\sum_{\langle ij \rangle} \hat\sigma^z_i \hat\sigma^z_j
               - \frac{g}{J} \sum_i \hat\sigma^z_i,
    \qquad
    \hat H_{\rm field} = -\sum_i \hat\sigma^x_i .

Sandwiching an operator between :math:`\langle\psi_P|` and :math:`|\psi_P\rangle`
and inserting a resolution of the identity in the :math:`\sigma^z` basis between
every exponential turns the quantum expectation value into a classical Ising
average on a lattice of ``2P + 1`` imaginary-time slices, with the operator on
the middle one:

.. math::

    H_{\rm cl}(s) = -\sum_t J_x(t) \sum_{\langle ij\rangle} s_{i,t} s_{j,t}
                    -\sum_t J_\tau(t) \sum_i s_{i,t} s_{i,t+1}
                    -\sum_t B(t) \sum_i s_{i,t},

sampled with weight :math:`e^{-H_{\rm cl}(s)}`, the couplings following from the
variational parameters as

.. math::

    J_x(p) = \alpha_p, \qquad
    B(p) = \frac{g}{J}\, \alpha_p, \qquad
    J_\tau(p) = \tfrac12 \ln \coth \beta_p .

Every weight is real and positive, so there is no sign problem and a classical
sampler can evaluate this variational family to arbitrary statistical precision
given enough sweeps. Any claim of quantum advantage on this model has to survive
that. The positivity is not special to the chain -- it follows from the model
being diagonal apart from the transverse field, so it holds on any lattice and
for either sign of :math:`J`; see :mod:`src.physics.classical.regime` for what
frustration costs instead.

``J`` and ``h`` are absent from the generators because :math:`\alpha_p` and
:math:`\beta_p` are free: scaling a generator and dividing its parameter by the
same constant gives an identical family. The ratio :math:`g/J` cannot be absorbed
that way, since one :math:`\alpha_p` drives both diagonal terms in the proportion
the Hamiltonian demands. Giving the longitudinal field its own parameter would be
more expressive and would break the correspondence with the quantum ansatz that
is the reason for using VITA as the baseline.

This module implements the one-dimensional chain only. The extra dimension in the
classical model is imaginary time, never a second spatial direction, so a
two-dimensional lattice would need a third axis that is not built here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.physics.model import BoundaryCondition

FloatArray = NDArray[np.float64]
SpinArray = NDArray[np.int8]

MIN_PARAMETER = 1e-6
r"""Floor on every variational parameter, applied before the coupling map.

Imaginary times are positive by construction, and :math:`J_\tau = \frac12 \ln
\coth\beta` diverges logarithmically as :math:`\beta \to 0`. This is a guard
against arithmetic overflow, not a modelling choice; the *statistical* floor
that keeps the transverse-field estimator well conditioned is
the ``min_parameter`` floor of
:class:`~src.physics.classical.variational_imaginary_time.OptimiserConfig`,
which is four orders of magnitude larger.
"""


@dataclass(frozen=True, slots=True)
class Couplings:
    r"""Classical Ising couplings of one VITA state.

    Attributes:
        spatial: Bond coupling :math:`J_x(t)` of each of the ``2P + 1`` slices.
            The middle slice always has ``0``: it carries no
            :math:`e^{-\alpha \hat H_{\rm diag}}` factor, because it is where the
            operator is inserted.
        temporal: Coupling :math:`J_\tau(t)` of the bond joining slice ``t`` to
            slice ``t + 1``. The last entry is ``0``, closing the open end of
            the imaginary-time direction.
        longitudinal: Site field :math:`B(t)` on each slice, ``g/J`` times
            ``spatial``. Zero throughout when the longitudinal field is off,
            which is the calibration case.
        temporal_derivative: :math:`\partial J_\tau / \partial \beta` for
            each bond. Carried here rather than recomputed because the gradient
            estimator needs it on every measurement.
    """

    spatial: FloatArray
    temporal: FloatArray
    longitudinal: FloatArray
    temporal_derivative: FloatArray

    @property
    def n_slices(self) -> int:
        """Number of imaginary-time slices, ``2P + 1``."""
        return int(self.spatial.size)

    @property
    def n_parameters(self) -> int:
        """Number of variational parameters, ``2P``."""
        return self.n_slices - 1

    @property
    def middle(self) -> int:
        """Index of the middle slice ``P``, where physical observables live."""
        return (self.n_slices - 1) // 2

    @property
    def has_longitudinal_field(self) -> bool:
        """Whether any slice carries a longitudinal field."""
        return bool(np.any(self.longitudinal != 0.0))


def split_parameters(theta: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Split the flat parameter vector into its ``alpha`` and ``beta`` halves.

    Args:
        theta: Flat array ``[alpha_1..alpha_P, beta_1..beta_P]`` of length ``2P``.

    Returns:
        The two halves, each of length ``P``.

    Raises:
        ValueError: If ``theta`` has odd length and so cannot be split.
    """
    if theta.size % 2 != 0:
        raise ValueError(f"theta must have even length 2P, got {theta.size}")
    half = theta.size // 2
    return np.asarray(theta[:half], dtype=float), np.asarray(theta[half:], dtype=float)


def join_parameters(alpha: FloatArray, beta: FloatArray) -> FloatArray:
    """Concatenate the two halves back into a flat parameter vector."""
    return np.concatenate([np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)])


def clip_parameters(theta: FloatArray) -> FloatArray:
    """Project parameters onto the physical domain ``theta > 0``.

    Imaginary times are positive by construction, so a negative parameter is
    read as its magnitude rather than rejected: an optimiser that overshoots
    should be reflected back into the domain, not stopped.
    """
    return np.maximum(np.abs(np.asarray(theta, dtype=float)), MIN_PARAMETER)


def total_imaginary_time(theta: FloatArray) -> float:
    r"""Total imaginary time :math:`\tau = \frac12 \sum_p (\alpha_p + \beta_p)`.

    The single number that says how far the ansatz has propagated, and the one
    to quote against circuit depth: it is what the quantum ansatz would have to
    reproduce, and how it grows with ``L`` is the substantive claim the method
    makes.
    """
    return 0.5 * float(np.sum(clip_parameters(theta)))


def build_couplings(theta: FloatArray, field_ratio: float = 0.0) -> Couplings:
    """Map variational parameters onto the couplings of the classical lattice.

    The slice ordering mirrors the ansatz read outwards from the operator in the
    middle. With ``a = alpha`` and ``b = ln(coth(beta)) / 2``::

        spatial  = [a_1, ..., a_P,  0,  a_P, ..., a_1]     (2P + 1 slices)
        temporal = [b_1, ..., b_P, b_P, ..., b_1,  0]      (2P + 1 bonds)

    The zero in ``spatial`` marks the middle slice, which carries no diagonal
    factor; the trailing zero in ``temporal`` closes the open time direction.
    The two arrays are offset by one because slices and bonds are different
    objects -- ``2P + 1`` slices are joined by ``2P`` bonds -- and getting that
    offset wrong is the mistake that shifts an energy by a plausible-looking
    constant.

    Args:
        theta: Flat parameter vector ``[alpha_1..alpha_P, beta_1..beta_P]``.
        field_ratio: The dimensionless ``g / J``. Zero switches the longitudinal
            field off entirely, which is the calibration case and the only one
            with a closed-form reference to be graded against.

    Returns:
        The spatial, temporal and longitudinal couplings, together with
        ``dJ_tau/dbeta``.
    """
    alpha, beta = split_parameters(clip_parameters(theta))

    spatial = np.concatenate([alpha, [0.0], alpha[::-1]])
    mirrored_beta = np.concatenate([beta, beta[::-1]])

    return Couplings(
        spatial=spatial,
        temporal=np.concatenate([0.5 * np.log(1.0 / np.tanh(mirrored_beta)), [0.0]]),
        longitudinal=field_ratio * spatial,
        # d/dbeta of 0.5 * ln(coth(beta)) = -1 / sinh(2 beta).
        temporal_derivative=np.concatenate([-1.0 / np.sinh(2.0 * mirrored_beta), [0.0]]),
    )


@dataclass(frozen=True, slots=True)
class Lattice:
    """Geometry of the classical lattice dual to a VITA state on the chain.

    Space is one-dimensional and inherits the chain's boundary condition;
    imaginary time is always **open**, because the first and last slices are the
    two ends of the path integral and are not joined to one another.

    Attributes:
        n_sites: Number of quantum spins, ``L``.
        n_slices: Number of imaginary-time slices, ``2P + 1``. Always odd:
            there has to be a middle one for the operator to sit on.
        boundary: The chain's boundary condition, ``"periodic"`` or ``"open"``.
    """

    n_sites: int
    n_slices: int
    boundary: BoundaryCondition = "periodic"

    def __post_init__(self) -> None:
        """Reject geometries the checkerboard sweep cannot handle.

        Raises:
            ValueError: If the chain is shorter than two sites, if the slice
                count is not an odd positive ``2P + 1``, or if a periodic chain
                has odd length.
        """
        if self.n_sites < 2:
            raise ValueError(f"n_sites must be at least 2, got {self.n_sites}")
        if self.n_slices < 1 or self.n_slices % 2 == 0:
            raise ValueError(f"n_slices must be an odd positive 2P+1, got {self.n_slices}")
        if self.boundary == "periodic" and self.n_sites > 2 and self.n_sites % 2 == 1:
            # The sweep updates a whole sublattice at once, which is only valid
            # if no two sites of the same parity share a bond. An odd ring
            # closes parity onto itself and silently correlates the update.
            raise ValueError(
                f"a periodic chain must have even length for the checkerboard sweep, "
                f"got {self.n_sites}; use boundary='open' or an even length"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the classical spin array, ``(L, 2P + 1)``."""
        return (self.n_sites, self.n_slices)

    @property
    def middle(self) -> int:
        """Index of the middle slice, where physical observables are measured."""
        return (self.n_slices - 1) // 2

    @property
    def n_bonds(self) -> int:
        """Spatial bonds per slice: ``L`` for a ring, ``L - 1`` for a segment."""
        return self.n_sites if self.boundary == "periodic" else self.n_sites - 1

    def random_spins(self, rng: np.random.Generator) -> SpinArray:
        """Draw a random configuration from the ``+-1`` alphabet."""
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=self.shape)

    def sublattice_mask(self, parity: int) -> NDArray[np.bool_]:
        """Checkerboard mask of the given ``parity``, 0 or 1.

        Sites of one parity share no bond with each other, in space or in time,
        so an entire sublattice can be proposed and accepted in one vectorised
        step. Two such steps update every spin exactly once.
        """
        site, slice_index = np.indices(self.shape)
        return ((site + slice_index) % 2) == parity


def spatial_neighbour_sum(spins: SpinArray, lattice: Lattice) -> FloatArray:
    """Sum of the in-slice neighbours of every site.

    Entry ``(i, t)`` holds ``s[i-1, t] + s[i+1, t]``, with terms beyond the ends
    dropped on an open chain. Every bond is therefore seen twice, once from each
    end, which is why the callers that want a bond sum carry a factor of one half.
    """
    total = np.zeros(spins.shape, dtype=np.float64)
    if lattice.boundary == "periodic":
        total += np.roll(spins, 1, axis=0)
        total += np.roll(spins, -1, axis=0)
    else:
        total[1:, :] += spins[:-1, :]
        total[:-1, :] += spins[1:, :]
    return total


def temporal_neighbour_field(spins: SpinArray, couplings: Couplings) -> FloatArray:
    """Weighted sum of the two imaginary-time neighbours of every site.

    Entry ``(i, t)`` is ``J_tau[t] * s[i, t+1] + J_tau[t-1] * s[i, t-1]``, with
    missing terms zero at the open ends of the time direction. The weight is
    inside the sum because, unlike the spatial coupling, ``J_tau`` differs from
    bond to bond.
    """
    field = np.zeros(spins.shape, dtype=np.float64)
    temporal = couplings.temporal[:-1]
    field[:, :-1] += temporal * spins[:, 1:]
    field[:, 1:] += temporal * spins[:, :-1]
    return field


def local_field(spins: SpinArray, lattice: Lattice, couplings: Couplings) -> FloatArray:
    r"""Effective field :math:`h_i` at every site, as the Metropolis sweep needs it.

    Defined so that flipping the spin at site :math:`i` changes the classical
    energy by exactly :math:`2 s_i h_i`. That is the only property the sweep
    uses, and it is why the bond terms are *not* halved here: each bond has to
    be seen once from each of its two ends for the flip cost to come out right.
    """
    spatial = couplings.spatial * spatial_neighbour_sum(spins, lattice)
    return spatial + temporal_neighbour_field(spins, couplings) + couplings.longitudinal


def classical_energy(spins: SpinArray, lattice: Lattice, couplings: Couplings) -> float:
    r"""Classical Ising energy :math:`H_{\rm cl}(s)`, with weight :math:`e^{-H_{\rm cl}}`.

    Bond terms are halved because :func:`local_field` counts each bond from both
    ends; the longitudinal term is not, because a site field is counted once.
    Conflating the two is a silent factor of two on the field, so the two
    contributions are written out separately rather than folded together.
    """
    bonds = couplings.spatial * spatial_neighbour_sum(spins, lattice)
    bonds = bonds + temporal_neighbour_field(spins, couplings)
    return float(-0.5 * np.sum(spins * bonds) - np.sum(spins * couplings.longitudinal))
