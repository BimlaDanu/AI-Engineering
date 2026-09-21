---
title: Ising machines, the commercial hardware whose native language is this model
source: "Mohseni, McMahon and Byrnes, Ising machines as hardware solvers of combinatorial optimization problems, Nature Reviews Physics 4, 363 (2022)"
arxiv: 2204.00276
topics: [ising-machines, hardware, business-applications, combinatorial-optimisation, applications]
---

# Ising machines, the commercial hardware whose native language is this model

Because so many decision problems reduce to minimising an Ising energy, a whole
class of special-purpose machines has been built to do only that. Mohseni, McMahon
and Byrnes (2022) review them together, which is the useful way to see them: they
compete on the same input format and differ in physics.

- **Superconducting quantum annealers.** Flux qubits with programmable couplings,
  driven by a transverse field that is ramped down. D-Wave's machines are the
  commercially available example, and the ones sold with a service contract.
- **Coherent Ising machines.** Networks of degenerate optical parametric
  oscillators, where the phase of each oscillator plays the role of a spin and the
  couplings are optical or measured-and-fed-back. Room temperature, all-to-all
  connectivity, no qubits in the computational sense.
- **Digital annealers and simulated-bifurcation machines.** Classical algorithms
  in custom silicon — ASIC, FPGA, GPU — that accept the same Ising input. Fujitsu's
  Digital Annealer is the best known.
- **Oscillator and memristor networks.** CMOS and analogue devices that relax into
  low-energy configurations of the same cost function.

## The distinction a buyer needs and a brochure blurs

Only some of these are quantum, and the classical ones are frequently the faster
answer today. A digital annealer is a heuristic — parallel tempering or a variant
of it — with the inner loop in hardware. It offers no quantum effect and no
asymptotic advantage, and it is often the best available solver for a mid-sized
industrial instance, precisely because it has none of the connectivity,
temperature or coherence constraints of the quantum machines.

"Runs an Ising model" therefore says nothing about whether a device is quantum.
The input format is shared; the mechanism is not.

## Where a solvable chain fits into a commercial stack

None of these machines is trusted on faith. A one-dimensional chain with a uniform
coupling and a transverse field is the standard shakedown instance: the spectrum
is known in closed form, the answer is not a matter of opinion, and a device that
gets it wrong is misconfigured rather than unlucky. Calibration, embedding checks
and schedule tuning are all done on instances of this kind before a customer
problem is loaded.

Yarkoni and colleagues (2022) reviewed the industrial deployments themselves and
found the same pattern across sectors: pilots, formulations and capability
building, with the hard performance comparisons still going the classical solvers'
way more often than not.
