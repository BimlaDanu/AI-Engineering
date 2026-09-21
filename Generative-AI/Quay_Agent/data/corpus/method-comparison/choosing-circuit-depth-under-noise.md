---
title: How deep should the circuit be? The optimum is interior, and that is the point
source: "Kandala et al., Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets, Nature 549, 242 (2017)"
arxiv: 1704.05018
topics: [circuit-depth, noise, barren-plateaus, ansatz, nisq, resource-estimation, quantum-computing, benchmarking]
---

# How deep should the circuit be? The optimum is interior, and that is the point

Every method in this comparison has a depth knob: the number of layers $p$ in
QAOA, the number of repetitions of an ansatz block in VQE, the number of steps in
an imaginary-time trajectory. On paper, more is better. On hardware it is not, and
the shape of that trade-off is the single most useful thing a feasibility study
can produce.

## Two curves that pull in opposite directions

**Expressibility rises with depth.** A shallow circuit can only reach a small set
of states, and if the ground state is not among them, no amount of optimising will
find it. This gap between the best reachable state and the true answer is the
*ansatz error*, and it falls as layers are added. For QAOA the limit is a theorem:
as $p \to \infty$ the alternating blocks reproduce an adiabatic sweep and the
answer becomes exact.

**Fidelity falls with depth.** Each two-qubit gate carries an error of roughly
$10^{-2}$ to $10^{-3}$ on current superconducting hardware, and the circuit also
has to finish before the qubits decohere. If a circuit has $G$ two-qubit gates
each with error $\epsilon$, the probability that none of them corrupts the run
falls off roughly as $(1-\epsilon)^{G}$ — so the useful depth is bounded by
$G \lesssim 1/\epsilon$, a few hundred gates at $\epsilon = 3\times10^{-3}$.
Separately, the whole circuit must run within the coherence time $T_2$, which is
a wall-clock constraint rather than a gate-count one.

The observed energy is the sum of the two errors, so it has a minimum at some
finite depth. Below it the circuit is too rigid; above it the answer is noise.
Kandala et al. (2017) chose their hardware-efficient ansatz around exactly this
constraint — using the device's native gates and connectivity so that depth was
spent on the state rather than on routing — and the paper is the standard
reference for the trade-off being real rather than theoretical.

## The third curve: trainability

There is a failure that is not about hardware at all. As a circuit gets deeper and
more expressive, the energy landscape it defines becomes flat almost everywhere:
gradients shrink exponentially with the number of qubits, so an optimiser sees
noise instead of slope and the number of shots needed to resolve a step becomes
prohibitive. This is the **barren plateau**, identified by McClean et al. (2018).

Two refinements matter for a real study:

- Cerezo et al. (2021) showed the effect depends on the observable. A **global**
  cost — one that measures all qubits at once — plateaus even at shallow depth,
  while a **local** cost built from few-qubit terms is trainable up to depths
  growing logarithmically with the qubit count. A Hamiltonian made of
  nearest-neighbour $\hat\sigma^z\hat\sigma^z$ and single-site $\hat\sigma^x$
  terms, as this chain is, is local in exactly that sense, which is favourable.
- Noise causes its own version of the problem. Wang et al. (2021) showed that
  hardware noise flattens the landscape independently of the ansatz, so the two
  effects compound: the depth that noise permits and the depth that trainability
  permits are separate ceilings, and a study should report both.

So there are three curves, and the workable depth is where all three allow it.

## Doing the arithmetic for this chain

For an open chain of $L$ sites, one layer of the standard alternating ansatz needs
$L-1$ two-qubit interactions — one per bond — plus one single-qubit rotation per
site. At depth $p$:

- two-qubit gates $\approx p\,(L-1)$, before transpilation;
- circuit duration $\approx p\,(L-1)\,t_{2q}$ if the bonds are done in sequence,
  or roughly $2\,p\,t_{2q}$ if even and odd bonds are done in parallel, which
  nearest-neighbour hardware permits;
- parameters $\approx 2p$ for QAOA, more for a free-angle ansatz.

Then two hard limits, one from gate error and one from coherence:

$$ p_{\text{error}} \;\lesssim\; \frac{1}{\epsilon\,(L-1)}, \qquad p_{\text{coherence}} \;\lesssim\; \frac{T_2}{(L-1)\,t_{2q}}\ \text{(sequential)} $$

The smaller of the two binds. With representative superconducting figures —
$\epsilon = 8\times10^{-3}$, $t_{2q} = 300$ ns, $T_2 = 100\ \mu$s — a 12-site
chain gives a gate-error ceiling near $p \approx 11$ and a coherence ceiling that
depends entirely on whether the bonds are parallelised. That comparison is the
actionable result: **which limit binds tells you what to buy.** A faster gate
fixes a coherence-bound circuit and does nothing for an error-bound one.

## Transpilation is where the estimate usually breaks

The counts above are for a circuit written against the problem's own geometry. A
real device supplies a fixed coupling map and a fixed native gate set, and the
compiler must bridge the difference — decomposing each interaction into native
gates and inserting SWAP operations wherever the circuit needs two qubits that the
chip does not connect.

For a chain on a line topology this costs almost nothing: the problem's bonds are
already the device's edges. For a chain on a sparser lattice, or for any problem
denser than a line, the SWAP overhead can multiply the gate count several times
over, and it lands entirely inside the depth budget computed above. An estimate
that skips transpilation is not conservative, it is wrong, and usually by a factor
rather than a margin.

## What a good answer to "how deep?" looks like

Not a number. A curve of energy error against depth, with the device's ceiling
drawn on it, and a statement of which constraint produced the ceiling. If the
curve's minimum sits below the ceiling, the method is depth-limited by physics and
a better device helps. If it sits above, the method is noise-limited and the
honest recommendation may be a different method — or a classical one.
