r"""Does folding the reported angles preserve the state they describe?

A variational solver reports the angles it stopped at next to the energy it reached,
and a reader is entitled to assume the one produces the other. Folding the angles
into a tidier range is only allowed if the fold is by a whole period.

**The period is the generator's, not the circle's.** The diagonal layer applies
$e^{-i\gamma \hat H_\text{diag}}$, whose eigenvalues are spaced by $2J$, and the field
layer applies $e^{-i\beta h \sum_i \hat\sigma^x_i}$, spaced by $2h$. So the periods
are $\pi/J$ and $\pi/h$, and folding by $2\pi$ -- as this project once did -- is
state-preserving only when $2J$ and $2h$ are whole numbers. That holds at the
calibration point $J = h = 1$ and fails at most slider positions the interface
offers.

This script measures the energy a folded vector actually reports, both ways, at the
couplings the application uses. The `pi/J, pi/h` column must be zero everywhere; the
`2pi` column is what the defect looked like.
"""

import numpy as np

from src.physics.quantum.ansatz import AnsatzSpec, angle_periods, wrap_angles
from src.physics.quantum.statevector import diagonal_energies, energy, evolve

RNG = np.random.default_rng(0)

PAIRS = ((1.0, 1.0), (1.0, 0.85), (0.7, 1.0), (1.5, 0.5), (1.0, 0.5), (0.3, 1.7))


def energy_of(theta: np.ndarray, spec: AnsatzSpec, coupling: float, field: float) -> float:
    """Energy of the state the schedule ``theta`` prepares.

    Args:
        theta: The parameter vector.
        spec: The circuit it parameterises.
        coupling: The Ising coupling $J$.
        field: The transverse field $h$.

    Returns:
        The energy, evaluated exactly with no shot noise.
    """
    diagonal = diagonal_energies(spec.n_qubits, coupling=coupling)
    state = evolve(theta, spec, diagonal, field)
    return energy(state, diagonal, field)


def main() -> None:
    """Fold each vector both ways and report how far the energy moved."""
    spec = AnsatzSpec(n_qubits=6, depth=3)
    turn = 2.0 * np.pi
    print(f"{'J':>5} {'h':>5} {'2J':>5} {'2h':>5} {'energy':>12} {'pi/J, pi/h':>12} {'2pi':>12}")
    worst = 0.0
    for coupling, field in PAIRS:
        theta = np.asarray(RNG.uniform(-8.0, 8.0, size=spec.n_parameters))
        before = energy_of(theta, spec, coupling, field)
        periods = angle_periods(coupling, field)
        correct = energy_of(wrap_angles(theta, *periods), spec, coupling, field)
        # What the old fold did: one window of width 2*pi for both schedules.
        naive = energy_of((theta + np.pi) % turn - np.pi, spec, coupling, field)
        worst = max(worst, abs(correct - before))
        print(
            f"{coupling:5.2f} {field:5.2f} {2 * coupling:5.1f} {2 * field:5.1f} "
            f"{before:12.6f} {correct - before:12.2e} {naive - before:12.2e}"
        )
    assert worst < 1e-9, f"folding by the generator's period moved the energy by {worst:g}"
    print(f"\nfolding by pi/J and pi/h preserves the energy everywhere (worst {worst:.1e})")


if __name__ == "__main__":
    main()
