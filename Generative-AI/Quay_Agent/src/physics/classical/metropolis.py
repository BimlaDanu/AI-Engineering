r"""Metropolis sampling of the classical Ising model dual to the VITA state.

Sampling is done with a checkerboard sweep: sites of one sublattice share no
bond, in space or in imaginary time, so a whole sublattice can be proposed and
accepted at once as a single array operation. One *sweep* updates both
sublattices, that is, every spin exactly once.

Physical observables are measured on the middle slice :math:`m = P`, which is
where the operator was inserted when the expectation value was turned into a
classical average. A diagonal operator reads straight off the configuration,

.. math::

    \langle \hat\sigma^z_i \hat\sigma^z_j \rangle = \langle s_{i,m} s_{j,m}\rangle,
    \qquad
    \langle \hat\sigma^z_i \rangle = \langle s_{i,m} \rangle,

while the transverse field, which is off-diagonal, becomes a ratio of classical
weights across the bond that the operator straddles,

.. math::

    \langle \hat\sigma^x_i \rangle
      = \big\langle e^{-2 J_\tau(m)\, s_{i,m} s_{i,m+1}} \big\rangle .

That last estimator is the one to watch. It is unbiased, but its two possible
values differ by :math:`e^{4 J_\tau(m)}`, and :math:`J_\tau` diverges as
:math:`\beta \to 0`; a short chain then cannot average it, and the sampled
energy can come out *below* the true ground state -- which is impossible for a
variational state and is the signature of the estimator being under-sampled
rather than wrong. The floor in
the ``min_parameter`` floor of
:class:`~src.physics.classical.variational_imaginary_time.OptimiserConfig`
exists for this and nothing else.

Gradients come from the reweighting identity

.. math::

    \partial_\theta \langle \hat H \rangle
      = \langle \partial_\theta E_{\rm loc} \rangle
      - \langle E_{\rm loc}\, \partial_\theta H_{\rm cl} \rangle
      + \langle E_{\rm loc} \rangle \langle \partial_\theta H_{\rm cl} \rangle,

whose first term is present only because :math:`\beta_P` appears explicitly in
the transverse-field estimator as well as in the sampling weight. Dropping it is
a subtle and entirely plausible-looking bug: the optimiser still converges, just
to the wrong parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.physics.classical.dual_lattice import (
    Couplings,
    FloatArray,
    Lattice,
    SpinArray,
    local_field,
    spatial_neighbour_sum,
)
from src.physics.classical.regime import Regime, assess
from src.physics.lattice import Lattice as SpatialLattice


@dataclass(frozen=True, slots=True)
class Measurement:
    r"""Per-sample quantities needed for the energy, the gradient and the metric.

    Energies here are **extensive**, not per spin: the optimiser wants the
    gradient of the total energy, and dividing by ``L`` is left to the caller
    that reports the result.

    Attributes:
        energy: The local estimator ``E_loc`` of the quantum energy.
        magnetisation: ``|sum_i s_{i,m}|``, the order parameter of the middle
            slice. Taken in absolute value because the ferromagnetic phase
            breaks a symmetry the finite chain still respects, so the signed
            average is zero on both sides of the transition and says nothing.
        sigma_z_total: The *signed* ``sum_i s_{i,m}``, the estimator of
            :math:`\sum_i \langle\hat\sigma^z_i\rangle`, which is what the
            longitudinal term of the energy couples to.
        classical_gradient: ``d H_cl / d theta``, one entry per parameter.
        estimator_gradient: ``d E_loc / d theta``, the explicit parameter
            dependence of the transverse-field estimator. Non-zero in one
            component only.
    """

    energy: float
    magnetisation: float
    sigma_z_total: float
    classical_gradient: FloatArray
    estimator_gradient: FloatArray


class Sampler:
    """A single Markov chain on the classical lattice dual to the VITA state.

    The chain is stateful by design. Consecutive calls along an optimisation
    reuse the configuration left by the previous one, so each new set of
    parameters starts from a lattice that is already close to equilibrated and
    the warm-up cost is paid once rather than at every iteration.

    Args:
        lattice: Geometry of the classical lattice.
        transverse_field: The transverse field ``h`` of the quantum Hamiltonian.
        coupling: The Ising coupling ``J``.
        longitudinal_field: The longitudinal field ``g``. Zero by default,
            which is the calibration case.
        rng: Random source. Pass a seeded generator for reproducibility.
        geometry: The *physical* shape this chain of spins stands for, used only
            to state the sampler's own standing -- see :attr:`regime`. Defaults to
            the one-dimensional chain the classical lattice literally is, which is
            the honest reading of a dual lattice built without one.
    """

    def __init__(
        self,
        lattice: Lattice,
        transverse_field: float,
        coupling: float = 1.0,
        longitudinal_field: float = 0.0,
        rng: np.random.Generator | None = None,
        geometry: SpatialLattice | None = None,
    ) -> None:
        self.lattice = lattice
        self.transverse_field = float(transverse_field)
        self.coupling = float(coupling)
        self.longitudinal_field = float(longitudinal_field)
        self.rng = np.random.default_rng() if rng is None else rng
        self.spins: SpinArray = lattice.random_spins(self.rng)
        self._masks = (lattice.sublattice_mask(0), lattice.sublattice_mask(1))
        self.geometry = geometry or SpatialLattice("chain", 1, lattice.n_sites, lattice.boundary)

    @property
    def regime(self) -> Regime:
        """How far this chain's numbers may be trusted, and why.

        The sampler states its own standing rather than leaving the caller to work
        it out, because the two facts that decide it -- the shape and the sign of
        the coupling -- are both held here and nowhere else at measurement time.
        A caller obliged to ask separately is a caller that will one day not ask,
        and the failure is silent: the number comes back looking ordinary.

        Returns:
            The standing, from :func:`src.physics.classical.regime.assess`.
        """
        return assess(self.geometry, self.coupling, self.transverse_field)

    @property
    def field_ratio(self) -> float:
        """The dimensionless ``g / J`` that the coupling map needs.

        Held here rather than passed alongside every call because it is a
        property of the physical problem, not of a particular set of variational
        parameters, and separating the two is how the two get mismatched.
        """
        return self.longitudinal_field / self.coupling

    def sweep(self, couplings: Couplings) -> None:
        """Update every spin once, one checkerboard sublattice at a time."""
        for mask in self._masks:
            field = local_field(self.spins, self.lattice, couplings)
            delta = 2.0 * self.spins * field
            # Clipped from above so that a downhill move is always accepted and
            # the exponential never overflows on a large uphill barrier.
            probability = np.exp(np.minimum(-delta, 0.0))
            accepted = self.rng.random(self.spins.shape) < probability
            self.spins[mask & accepted] *= -1

    def warmup(self, couplings: Couplings, n_sweeps: int) -> None:
        """Advance the chain without measuring, to forget where it started."""
        for _ in range(n_sweeps):
            self.sweep(couplings)

    def measure(self, couplings: Couplings) -> Measurement:
        """Measure the energy, the order parameter and the gradient estimators."""
        spins = self.spins
        middle = self.lattice.middle

        neighbour_sum = spatial_neighbour_sum(spins, self.lattice)
        # Halved because the neighbour sum sees each bond from both of its ends.
        sigma_z_bonds = 0.5 * float(np.sum(spins[:, middle] * neighbour_sum[:, middle]))
        sigma_z_total = float(np.sum(spins[:, middle]))

        time_bond = spins[:, middle] * spins[:, middle + 1]
        sigma_x_local = np.exp(-2.0 * couplings.temporal[middle] * time_bond)
        sigma_x_total = float(np.sum(sigma_x_local))

        energy = (
            -self.coupling * sigma_z_bonds
            - self.transverse_field * sigma_x_total
            - self.longitudinal_field * sigma_z_total
        )

        # Bond sums per slice and per time bond, which is all the gradient needs.
        bonds_spatial = 0.5 * np.sum(spins * neighbour_sum, axis=0)
        bonds_temporal = np.sum(spins[:, :-1] * spins[:, 1:], axis=0)
        moments = np.asarray(np.sum(spins, axis=0), dtype=np.float64)

        classical_gradient = _classical_gradient(bonds_spatial, bonds_temporal, moments, couplings)

        # Only J_tau(middle) = 0.5 ln coth(beta_P) enters E_loc explicitly, and
        # beta_P is the last parameter, so exactly one component is non-zero.
        estimator_gradient = np.zeros_like(classical_gradient)
        estimator_gradient[-1] = (
            2.0
            * self.transverse_field
            * couplings.temporal_derivative[middle]
            * float(np.sum(time_bond * sigma_x_local))
        )

        return Measurement(
            energy=energy,
            magnetisation=abs(sigma_z_total),
            sigma_z_total=sigma_z_total,
            classical_gradient=classical_gradient,
            estimator_gradient=estimator_gradient,
        )


def _classical_gradient(
    bonds_spatial: FloatArray,
    bonds_temporal: FloatArray,
    moments: FloatArray,
    couplings: Couplings,
) -> FloatArray:
    """Assemble ``d H_cl / d theta`` from the per-slice bond and moment sums.

    Parameter ``alpha_p`` sets the spatial coupling -- and, when the
    longitudinal field is on, the site field -- of the mirror pair of slices
    ``p-1`` and ``2P-(p-1)``; parameter ``beta_p`` sets the temporal coupling of
    the mirror pair of bonds ``p-1`` and ``2P-p``. Since every term of ``H_cl``
    enters with a minus sign, so does every derivative, and ``beta`` picks up
    the chain-rule factor ``dJ_tau/dbeta`` on top.
    """
    n_slices = couplings.n_slices
    depth = (n_slices - 1) // 2
    ratio = couplings.longitudinal[0] / couplings.spatial[0] if couplings.spatial[0] else 0.0

    d_alpha = np.empty(depth)
    d_beta = np.empty(depth)
    for index in range(depth):  # index = p - 1, zero-based
        mirror = n_slices - 1 - index
        d_alpha[index] = -(bonds_spatial[index] + bonds_spatial[mirror]) - ratio * (
            moments[index] + moments[mirror]
        )
        d_beta[index] = -couplings.temporal_derivative[index] * (
            bonds_temporal[index] + bonds_temporal[2 * depth - 1 - index]
        )
    return np.concatenate([d_alpha, d_beta])


def log_derivatives(classical_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Log-derivatives :math:`O_j = \partial_j \ln\psi = -\frac12 \partial_j H_{\rm cl}`.

    The factor of one half is the whole content of this function and is easy to
    lose: the sampled weight is :math:`|\psi|^2 \sim e^{-H_{\rm cl}}`, so a
    derivative of the *log amplitude* is half a derivative of the classical
    action. It propagates squared into the metric, where getting it wrong
    rescales every step by four.
    """
    return -0.5 * classical_gradient
