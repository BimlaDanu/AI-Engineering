r"""Physics and hardware: is the number right, and what would it cost to run?

Two questions, in the order that makes one page of them. The first is whether the
problem this project computes is the problem it means to compute. The second is what a
real quantum computer would make of the circuit that prepares it. They are one page
because the second is worthless without the first: a shot budget priced against a
Hamiltonian with a flipped sign is an exact answer about a problem nobody asked about.

Section one is the live cross-check. Every circuit, gradient, budget and verdict
downstream inherits the chain built in `src/physics/quantum/hamiltonians.py`, and a
factor of two or a flipped sign there propagates silently -- the run still converges,
the plots still look like physics, and the answer is wrong. Nothing else in the project
would catch it, because a variational method converges to the lowest energy of whatever
matrix it was handed and reports success. So the chain is rebuilt from its Pauli
representation on every interaction and held against two solvers that share no algebra
with it: the free-fermion closed form, and sparse exact diagonalisation.

Section two is the hardware arithmetic. Given a chain of this length and a circuit
of this depth, what would three different machines actually do with it? It is worth
having on a page rather than in a report because the answer moves under the controls,
and watching it move is the argument. Lengthen the chain and the depth ceiling falls.
Turn the segment into a ring and the wiring suddenly costs eight times the depth on both
real machines and nothing at all on the fictional one. Neither is surprising once
stated; both are much harder to believe from a sentence than from a slider.

Nothing here is a simulation and nothing here calls a language model. Section one is
three solvers; section two is arithmetic over a machine's wiring diagram and its
published error rates. That is why the page answers instantly, and why the same knob
position gives the same answer to everybody.

This page imports the exact solvers deliberately. It is a monitor on the grader's side
of the wall; `src/physics/quantum/` stays on the other side, and
the architecture test is the file that keeps it there -- it names this module in
an allow-list, so a second importer has to be a decision rather than an accident.

Formed by merging what were two pages, *Cross-check* and *Machines*. They shared the one
property that was least obvious about either -- both are driven only by the settings
knob, with no model in the loop -- and separately that property was a caption repeated
twice rather than the point of the page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse.linalg import eigsh

from src.hardware.devices import DEVICES, device_for
from src.hardware.export import Couplings, qasm3, run_card
from src.hardware.fidelity import USABLE_FIDELITY_FLOOR, depth_ceiling, estimate
from src.hardware.transpile import fits, transpile
from src.physics.model import BoundaryCondition, TFIMSpec
from src.physics.quantum.ansatz import AnsatzSpec
from src.physics.quantum.hamiltonians import ising_chain
from src.physics.reference import exact_diagonalisation, free_fermions
from src.ui import panels

SWEEP_POINTS = 61
"""Field values in the sweep, chosen so that the critical point is one of them.

The sweep runs to ``h/J = 3``, so sampling ``h/J = 1`` exactly needs the number of
*intervals* to divide by three -- not merely to be odd, which is what this said while
it was set to 81. Eighty intervals do not divide by three, so the nearest sample sat
at 1.0125 and the one field value the caption below points at was the one value the
chart never computed. Sixty do, giving a step of 0.05, which is also the step the
agent's own field sweep uses in :data:`src.agent.tools.DEFAULT_SWEEP_POINTS`.
"""

panels.header(
    "Physics and hardware",
    "Is the number right, and what would it cost to run? Two checks, neither of "
    "which asks a language model anything.",
    "src/physics/quantum/hamiltonians.py, src/physics/reference/ and src/hardware/",
)

setting = panels.current_setting()
physics = setting.physics
n_sites, coupling, field = physics.n_sites, physics.coupling, physics.field
boundary: BoundaryCondition = physics.boundary
longitudinal = physics.longitudinal
precision = physics.precision
depth = physics.depth

panels.model_has_no_say("any number on this page")

st.divider()
st.header("1 · Is the number right?")
st.info(
    "**Why this section exists.** Every circuit, budget and verdict downstream "
    "inherits the chain built here. A factor of two or a flipped sign in it would "
    "propagate silently -- the runs still converge, the charts still look like "
    "physics, and the answer is wrong. So the chain is rebuilt on every interaction "
    "and held against a closed-form solution and an independent numerical one. "
    "Agreement between three routes is evidence; a self-consistency check that "
    "passes is not."
)
panels.hamiltonian()


@st.cache_data(show_spinner=False)
def lowest_energy(
    n_sites: int,
    coupling: float,
    field: float,
    longitudinal: float,
    boundary: BoundaryCondition,
) -> float:
    """Ground-state energy of the Pauli-built matrix, by sparse eigensolve."""
    matrix = ising_chain(
        n_sites,
        coupling=coupling,
        transverse_field=field,
        longitudinal_field=longitudinal,
        boundary=boundary,
    ).to_matrix()
    if matrix.shape[0] <= 64:
        return float(np.linalg.eigvalsh(matrix.toarray()).real[0])
    start = np.ones(matrix.shape[0]) / np.sqrt(matrix.shape[0])
    return float(eigsh(matrix, k=1, which="SA", v0=start, tol=0.0)[0][0])


@st.cache_data(show_spinner=False)
def pfeuty_sweep(n_sites: int, coupling: float) -> pd.DataFrame:
    """Energy density and gap across the field, from the closed-form solution."""
    fields = np.linspace(0.0, 3.0, SWEEP_POINTS)
    specs = [
        TFIMSpec(n_sites=n_sites, coupling=coupling, field=float(h), boundary="periodic")
        for h in fields
    ]
    return pd.DataFrame(
        {
            "h / J": fields / coupling,
            "energy density E₀/N": [free_fermions.energy_density(spec) for spec in specs],
            "one quasiparticle ε(π/L)": [free_fermions.gap(spec) for spec in specs],
            "cheapest reachable pair": [free_fermions.parity_even_gap(spec) for spec in specs],
        }
    ).set_index("h / J")


chain = ising_chain(
    n_sites,
    coupling=coupling,
    transverse_field=field,
    longitudinal_field=longitudinal,
    boundary=boundary,
)
groups = chain.measurement_groups()
l1 = chain.coefficient_l1()
absolute_target = precision * n_sites
shots = (l1 / absolute_target) ** 2

st.subheader("What one energy evaluation costs, in measurements")
one, two, three, four = st.columns(4)
one.metric("Pauli terms", len(chain.terms))
two.metric("Measurement settings", len(groups), help="Qubit-wise commuting groups")
three.metric("Σ|cₐ|", f"{l1:.3g}", help="Sets the shot budget")
four.metric("Shots for ε", f"{shots:,.0f}", help="S ≈ (Σ|cₐ|)² / ε², optimally allocated")

st.caption(
    f"Every σᶻσᶻ and σᶻ term is diagonal in the same basis, and every σˣ term in another, "
    f"so this chain costs **{len(groups)}** circuits per evaluation rather than "
    f"{len(chain.terms)}. That factor comes from noticing a commutation structure, not "
    "from buying hardware — and the shot count is near-independent of N, because Σ|cₐ| "
    "grows like N while the absolute target ε = N·(ε/N) does too."
)

st.divider()
st.subheader("Cross-check against the exact solution")

pauli_energy = lowest_energy(n_sites, coupling, field, longitudinal, boundary)
rows: list[dict[str, object]] = [
    {"method": "Pauli representation → sparse matrix", "E₀": pauli_energy, "|Δ| vs Pfeuty": "—"}
]

spec = TFIMSpec(n_sites=n_sites, coupling=coupling, field=field, boundary=boundary)
if longitudinal > 0.0:
    st.warning(
        "The longitudinal field g is on. Neither reference solver in this repository "
        "carries it — `free_fermions.py` is the free-fermion TFIM and "
        "`exact_diagonalisation.py` builds the same Hamiltonian — so there is "
        "nothing here to check the number against. That is "
        "exactly why g ≠ 0 is where the feasibility question becomes honest, and it is "
        "also why the project calibrates at g = 0 first."
    )
else:
    ed_energy = exact_diagonalisation.ground_state_energy(spec)
    rows.append(
        {
            "method": "sparse exact diagonalisation (`physics/reference/exact_diagonalisation.py`)",
            "E₀": ed_energy,
            "|Δ| vs Pfeuty": "—",
        }
    )
    reason = free_fermions.unsupported_reason(spec)
    if reason is None:
        pfeuty = free_fermions.ground_state_energy(spec)
        rows.append(
            {
                "method": "free fermions (`physics/reference/free_fermions.py`)",
                "E₀": pfeuty,
                "|Δ| vs Pfeuty": 0.0,
            }
        )
        rows[0]["|Δ| vs Pfeuty"] = abs(pauli_energy - pfeuty)
        rows[1]["|Δ| vs Pfeuty"] = abs(ed_energy - pfeuty)
        agreement = max(abs(pauli_energy - pfeuty), abs(ed_energy - pfeuty))
        if agreement < 1e-9:
            st.success(f"Three routes, one number — they agree to {agreement:.2e}.")
        else:
            st.error(
                f"The three routes disagree by {agreement:.2e}. Something in the Pauli "
                "construction or the diagonalisation has moved; stop and find it before "
                "trusting anything downstream."
            )
    else:
        # Two routes rather than three, and the two are still worth reporting. This
        # branch used to say only that the closed form was unavailable, so an open
        # chain -- the boundary condition hardware actually has -- got no agreement
        # figure at all, even though the Pauli construction and the diagonalisation
        # had just produced one to fourteen digits. A cross-check between two
        # independent routes is weaker evidence than between three; it is not no
        # evidence, and reporting nothing was the one reading of it that is wrong.
        st.info(f"No closed form for this configuration: {reason}")
        pair = abs(pauli_energy - ed_energy)
        rows[0]["|Δ| vs the other route"] = pair
        rows[1]["|Δ| vs the other route"] = pair
        if pair < 1e-9:
            st.success(
                f"Two routes, one number — the Pauli construction and the "
                f"diagonalisation agree to {pair:.2e}. The closed form, which would "
                "make it three, does not cover this configuration."
            )
        else:
            st.error(
                f"The two routes disagree by {pair:.2e}. Something in the Pauli "
                "construction or the diagonalisation has moved; stop and find it "
                "before trusting anything downstream."
            )

st.dataframe(rows, hide_index=True, width="stretch")

st.divider()
st.subheader("Where the physics happens")
sweep = pfeuty_sweep(n_sites if n_sites % 2 == 0 else n_sites + 1, coupling)
left, right = st.columns(2)
with left:
    st.markdown("**Energy density**")
    st.line_chart(sweep[["energy density E₀/N"]], height=260)
with right:
    st.markdown("**Gap**")
    st.line_chart(
        sweep[["one quasiparticle ε(π/L)", "cheapest reachable pair"]],
        height=260,
    )
st.caption(
    "Both panels are the periodic ring of even length, which is the case the closed form "
    "covers. Two curves rather than one, because they are two different quantities and "
    "only the upper one is the cost that matters. Exciting the chain conserves the parity "
    "∏ᵢ σˣᵢ, so quasiparticles arrive in pairs: a method that starts from a parity "
    "eigenstate — which every method here does — has to suppress the *pair*, at twice the "
    "energy of one. Both dip at h/J = 1 and neither reaches zero, because the momentum "
    "that would close them, k = 0, is not one a finite ring allows. That non-zero minimum "
    "is what makes imaginary-time convergence expensive but finite; quoting the lower "
    "curve for it would halve every convergence estimate, always in the flattering "
    "direction."
)

st.divider()
with st.expander("The terms themselves"):
    st.dataframe(
        [
            {"pauli string": label, "coefficient": coefficient}
            for label, coefficient in chain.to_labels()
        ],
        hide_index=True,
        width="stretch",
    )
    for index, group in enumerate(groups, start=1):
        st.markdown(
            f"**Setting {index}** — basis `{''.join(group.basis)}`, "
            f"{len(group.operator.terms)} terms"
        )


st.divider()
st.header("2 · What would it cost to run?")
st.caption(
    "The same chain, now as a circuit somebody would have to run. Everything above "
    "asks whether the number is right; everything below asks what a real machine "
    "would charge to get it -- which is a question worth asking only in that order."
)

st.markdown(
    f"A chain of **{n_sites}** magnets, "
    f"{'in a line' if boundary == 'open' else 'joined into a ring'}, "
    f"prepared by **{depth}** repeated layer{'s' if depth != 1 else ''}."
)

rows = []
for device in DEVICES:
    if fits(n_sites, device) is not None:
        rows.append(
            {
                "machine": device.name,
                "qubits": device.n_qubits,
                "longest run of joined qubits": len(device.longest_path()),
                "extra depth the wiring costs": "—",
                "depth on this machine": "—",
                "how long it runs": "does not fit",
                "signal that survives": "—",
                "extra measurements": "—",
                "deepest layer count worth running": "—",
                "what stops it first": "the register is too small",
            }
        )
        continue
    compiled = transpile(AnsatzSpec(n_qubits=n_sites, depth=depth, boundary=boundary), device)
    surviving = estimate(compiled)
    ceiling = depth_ceiling(n_sites, device, boundary=boundary)
    inflation = surviving.shot_inflation()
    rows.append(
        {
            "machine": device.name,
            "qubits": device.n_qubits,
            "longest run of joined qubits": len(device.longest_path()),
            "extra depth the wiring costs": f"x{compiled.routing_overhead:.1f}",
            "depth on this machine": compiled.two_qubit_depth,
            "how long it runs": f"{compiled.duration_us:.2f} us "
            f"({compiled.duration_ns / device.t2_ns:.0%} of coherence)",
            "signal that survives": f"{surviving.total:.1%}",
            "extra measurements": "unbounded" if inflation == float("inf") else f"x{inflation:.1f}",
            "deepest layer count worth running": ceiling.limit,
            "what stops it first": ceiling.binding,
        }
    )

st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.caption(
    "**ideal** is a control rather than a machine: everything connected to everything "
    "and nothing ever wrong, so any error left on it belongs to the algorithm and not "
    "to the hardware. **linear** is a realistic row of qubits, which is the shape this "
    "problem already has, so it is the best case that is still honest. "
    "**heavy-hex-27** is the wiring pattern real superconducting processors use."
)

runnable = [
    device
    for device in DEVICES
    if fits(n_sites, device) is None and depth <= depth_ceiling(n_sites, device, boundary).limit
]
if runnable:
    st.success(
        f"{panels.PRESENT} {len(runnable)} of {len(DEVICES)} machines would carry this "
        f"circuit as asked: {', '.join(device.name for device in runnable)}."
    )
else:
    st.warning(
        f"{panels.WARNING} No machine here would carry this circuit as asked. The table "
        "says how many layers each one would take instead."
    )

st.divider()
st.subheader("Where each machine runs out")
st.caption(
    f"How much of the result is still the intended state, as the circuit deepens. "
    f"Below {USABLE_FIDELITY_FLOOR:.0%} most of what comes back is noise, and the cost "
    "of seeing through it has already quadrupled."
)

curve = pd.DataFrame(
    {
        device.name: [
            estimate(
                transpile(AnsatzSpec(n_qubits=n_sites, depth=layer, boundary=boundary), device)
            ).total
            for layer in range(1, 13)
        ]
        for device in DEVICES
        if fits(n_sites, device) is None
    },
    index=pd.Index(range(1, 13), name="circuit layers"),
)
st.line_chart(curve, height=280)

st.divider()
st.subheader("Where the spins go, and what gets sent")

# The machine comes from the knob rather than from a second picker here. Two controls
# bound to one value have a precedence rule nobody can see, and the loser silently
# discards a choice somebody has just made.
chosen = physics.device if fits(n_sites, device_for(physics.device)) is None else ""
st.caption(
    f"Showing **{chosen or 'nothing'}** — change the machine in the settings knob. "
    "Placement decides which physical qubit holds which magnet, and it is most of the "
    "engineering: a ring laid out naively turns one interaction into a queue that the "
    "whole circuit waits behind."
)
if chosen:
    machine = device_for(chosen)
    compiled = transpile(AnsatzSpec(n_qubits=n_sites, depth=depth, boundary=boundary), machine)
    # The rotation angles scale with J and h, so the exported file has to be written
    # against the knob's values. Left at the default it is a circuit for J = h = 1,
    # which is a different problem from the one the rest of the page just priced.
    strengths = Couplings(
        coupling=coupling, transverse_field=field, longitudinal_field=longitudinal
    )
    left, right = st.columns([2, 3])
    with left:
        panels.metrics(
            {
                "Layout": compiled.layout.strategy.replace("_", " "),
                "Shuffling moves": compiled.swaps,
                "Depth": compiled.two_qubit_depth,
            }
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "magnet": range(n_sites),
                    "sits on qubit": compiled.layout.sites,
                }
            ),
            hide_index=True,
            height=260,
        )
    with right:
        st.caption(
            "The circuit as the machine receives it, addressing physical qubits. Every "
            "operation the estimate above charged for is in this file -- a cost that "
            "the circuit does not contain is a number nobody can check."
        )
        st.code(qasm3(compiled, couplings=strengths)[:4000], language="text")

    card = run_card(
        compiled,
        shots=1_000_000,
        note="drawn from the controls on this page",
        couplings=strengths,
    )
    with st.expander("The run card that would go with it"):
        st.json(card.as_data(), expanded=False)

panels.footer("src/physics/quantum/hamiltonians.py, src/physics/reference/ and src/hardware/")
