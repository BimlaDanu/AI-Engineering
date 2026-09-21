r"""Quay Lab: the chain, the circuit, the search for its angles, and the true answer.

The part of this application no language model can touch. Every number and curve
comes from the physics layer driven by the settings knob\'s ``Physics`` tab, and
the ``Language model`` tab has no effect on any of it.

Eight numbered tabs. The numbering is the accessibility feature: a reader
arriving from the Chat page has been told a verdict and never told what the chain
is. The first five are the argument, each answering what the one before leaves
open; the last three are for checking it.

1. What it is -- the physics as pictures: where this chain sits between its two
   phases, and the ground state\'s actual arrangements read off the eigenvector.
2. What the machine does -- the variational loop as a subroutine inside an
   ordinary one, with the bound, the estimator, the shot arithmetic and the
   barren plateau. One equation per step, unfolded.
3. Which method -- four algorithms, two asking for an energy and two for a
   choice, each with what it prepares and what a machine reads back. Then the
   same chain from three opening positions, where all-zero angles never move.
4. The circuit -- gate arithmetic from integers before any circuit exists. The
   ratio worth reading is naive depth against scheduled: every
   :math:`\hat\sigma^z\hat\sigma^z` rotation commutes, so a chain needs two
   rounds however long it is, and missing that overprices by :math:`L/2`.
5. Tuning it -- the convergence curve, the stop reason, and what each added
   layer of the depth ladder bought.
6. The true answer -- imaginary time under :math:`e^{-\tau\hat H}` by exact
   matrix exponentiation. The grader\'s method, not the agent\'s; this page sits
   on the grader\'s side of the import wall and says so.
7. Why an ordinary computer competes -- the exact rewriting to a flat sheet of
   magnets, in five steps. Every weight is real and positive, so there is no
   sign problem and the classical baseline is not a straw man.
8. Report -- every number on the other seven tabs as copyable plain text.

Each method has its own size ceiling and each panel asks its own method whether
it applies, printing the refusal it gets back rather than raising: dense
operators stop at :data:`~src.physics.quantumness.MAX_SITES`, the statevector
simulator at :data:`~src.physics.model.MAX_SITES_STATEVECTOR`, the sparse exact
answer at :data:`~src.physics.model.MAX_SITES_SPARSE`, and the closed form wants
an even ring and has no limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse.linalg import expm_multiply

from src.hardware.devices import device_for
from src.hardware.fidelity import estimate
from src.hardware.transpile import fits, transpile
from src.physics import quantumness
from src.physics.classical import dual_lattice
from src.physics.model import BoundaryCondition, TFIMSpec
from src.physics.quantum.ansatz import AnsatzSpec
from src.physics.quantum.quantum_approximate_optimisation import depth_sweep
from src.physics.quantum.variational_eigensolver import solve
from src.physics.quantumness import Quantumness, Superposition
from src.physics.reference import exact_diagonalisation, free_fermions
from src.ui import figures, panels
from src.ui.starters import METHOD_QUESTIONS, STARTERS

IMAGINARY_TIME_STEPS = 40
# Points on the imaginary-time curve. Enough to show the shape of the descent.

MAX_IMAGINARY_TIME = 3.0
# How far to run :math:`\tau`.
#
# By three inverse energy units the gap has suppressed every excited state that a chain
# this size has, so the curve has visibly flattened onto the ground-state energy and
# there is nothing further to show.

SPREAD_RATIOS: tuple[float, ...] = (0.2, 0.7, 1.0, 2.0)
# The four fields the spreading strip is drawn at.
#
# Not evenly spaced, and deliberately: what the strip has to show is a transition at
# g = 1, so two points below it, the point itself and one well above it says more
# than four points at equal intervals, three of which would land in the same phase.

GAP_POINTS = 121
# Samples along the gap curve. Odd, so g = 1 is hit exactly and the cusp is real
# rather than an artefact of a grid that stepped over it.

MAX_GAP_RATIO = 2.5
# How far past the critical point the gap curve runs. Far enough that the linear
# rise on the disordered side is unmistakably linear.

MAPPING_SLICES = 5
# Imaginary-time slices in the drawing of the classical sheet, 2P + 1 for P = 2.
#
# Small on purpose. The drawing has to show that the rows are copies and that they
# are joined to their neighbours; five rows says both, and twenty says neither
# because the bonds stop being separable by eye.


# --------------------------------------------------------------------------
# The physics, computed once per knob position
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def chain_facts(
    n_sites: int, coupling: float, field: float, boundary: BoundaryCondition
) -> Quantumness:
    """Evaluate the operator identities on this chain.

    Cached on the knob's position rather than on a spec object, because Streamlit
    hashes arguments and four numbers hash more reliably than a dataclass.

    Args:
        n_sites: Chain length.
        coupling: The Ising coupling :math:`J`.
        field: The transverse field :math:`h`.
        boundary: Ring or segment.

    Returns:
        The measured identities.
    """
    return quantumness.measure(TFIMSpec(n_sites, coupling, field, boundary))


@st.cache_data(show_spinner=False)
def ground_state_picture(
    n_sites: int, coupling: float, field: float, boundary: BoundaryCondition
) -> Superposition:
    """Read the heaviest spin arrangements out of the ground state.

    Args:
        n_sites: Chain length.
        coupling: The Ising coupling :math:`J`.
        field: The transverse field :math:`h`.
        boundary: Ring or segment.

    Returns:
        The measured superposition.
    """
    return quantumness.superposition(TFIMSpec(n_sites, coupling, field, boundary))


@st.cache_data(show_spinner=False)
def spread_across_field(
    n_sites: int, coupling: float, boundary: BoundaryCondition, ratios: tuple[float, ...]
) -> tuple[Superposition, ...]:
    """Read the ground state at several fields, holding the coupling fixed.

    Args:
        n_sites: Chain length.
        coupling: The Ising coupling :math:`J`.
        boundary: Ring or segment.
        ratios: The values of :math:`h/J` to look at.

    Returns:
        One superposition per ratio, in the order given.
    """
    return tuple(
        quantumness.superposition(TFIMSpec(n_sites, coupling, coupling * ratio, boundary))
        for ratio in ratios
    )


@st.cache_data(show_spinner=False)
def gap_curves(
    n_sites: int, coupling: float, boundary: BoundaryCondition
) -> tuple[list[float], list[float], list[float] | None]:
    """Trace the energy gap against the field, for this chain and for an infinite one.

    The infinite chain is always available -- it is :math:`2|J - h|`, arithmetic on
    two numbers. This chain's own gap needs the closed form, which wants an even
    ring, so it is returned as ``None`` rather than substituted from a method of
    different standing when the chain is not one.

    Args:
        n_sites: Chain length.
        coupling: The Ising coupling :math:`J`.
        boundary: Ring or segment.

    Returns:
        The ratios sampled, the infinite chain's gap at each, and this chain's own
        gap at each or ``None``.
    """
    ratios = [MAX_GAP_RATIO * index / (GAP_POINTS - 1) for index in range(GAP_POINTS)]
    infinite = [free_fermions.gap_thermodynamic(coupling, coupling * ratio) for ratio in ratios]
    probe = TFIMSpec(n_sites, coupling, coupling, boundary)
    if free_fermions.unsupported_reason(probe) is not None:
        return ratios, infinite, None
    finite = [
        free_fermions.gap(TFIMSpec(n_sites, coupling, coupling * ratio, boundary))
        for ratio in ratios
    ]
    return ratios, infinite, finite


def phase_words(ratio: float) -> str:
    """Say which phase a chain is in, without using the word *phase*.

    Args:
        ratio: The field-to-coupling ratio :math:`h/J`.

    Returns:
        A short description a reader with no physics can act on.
    """
    if abs(ratio - 1.0) < 0.05:
        return "right on the critical point, where every approximate method has its hardest time"
    if ratio < 1.0:
        return "on the ordered side, where the magnets mostly line up with each other"
    return "on the disordered side, where the sideways push has knocked them over"


def no_exact_answer(reason: str) -> None:
    """Explain that nothing here can supply the true answer, and why that is the point.

    Args:
        reason: The refusal the method itself gave.
    """
    st.info(
        f"**No exact answer for this chain.** {reason}\n\n"
        "Nothing below is graded against a reference, so read the energies as what "
        "the circuit reached and not as how close it got. This is not a gap in the "
        "page: it is the regime the whole project is about — the question *is a "
        "quantum computer worth it* only becomes real above the size an ordinary "
        "computer can answer, and this chain is above it."
    )


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

setting = panels.current_setting()
physics = setting.physics

spec = TFIMSpec(
    n_sites=physics.n_sites,
    coupling=physics.coupling,
    field=physics.field,
    boundary=physics.boundary,
)
ansatz = AnsatzSpec(n_qubits=physics.n_sites, depth=physics.depth, boundary=physics.boundary)
# The exact answer comes from `panels.true_energy` rather than from a copy here, so
# that the reference line under this page's curves and the one under the Chat page's
# are the same number computed the same way.
#
# g has to be passed. The solves below run with it, and without it here the page grades
# them against the g = 0 energy, which is higher -- so a correct run lands underneath and
# the page reports its own variational bound as violated. There is no exact route at
# g != 0, so the right answer is the refusal `true_energy` returns, not a number from a
# neighbouring problem.
exact, exact_refusal = panels.true_energy(
    physics.n_sites,
    physics.coupling,
    physics.field,
    physics.boundary,
    physics.longitudinal,
)

panels.header(
    "Quay Lab",
    "The chain, the circuit, the search for its angles and the answer they are "
    "chasing — computed with no model anywhere in the loop.",
    "src/physics/ and src/hardware/",
)

# One short sentence and then a picture, in that order, before anything with an
# equation in it. A reader arriving from the Chat page has been handed a verdict
# about this chain and has never been shown what a chain is -- and this page used to
# open on five stacked paragraphs, which is the wrong way round: the drawing makes
# the two competing terms obvious in a glance and the Hamiltonian beneath it then
# says the same thing in symbols, which is far easier to read second.
st.markdown(
    f"**{physics.n_sites} small magnets in a "
    f"{'ring' if physics.boundary == 'periodic' else 'row'}.**"
)
# Written from the ratio, and never from the letter "g". The sidebar can still set a
# longitudinal field g from its advanced control, so the widespread literature habit
# of calling h/J "g" would put one symbol on two quantities on one screen. The
# equation below is the two-term form unless that control has been moved.
panels.figure(
    figures.chain_terms_figure(
        physics.n_sites,
        physics.coupling,
        physics.field,
        periodic=physics.boundary == "periodic",
    ),
    "Blue keeps neighbours pointing the same way; orange tries to knock each one "
    "sideways. No arrangement of magnets satisfies both, and that disagreement is "
    "the only reason any of this is quantum — everything the circuits do below is "
    "an attempt to settle it. "
    + (
        f"Only the ratio of the two matters: $h/J = {physics.ratio:.2f}$ puts this "
        f"chain {phase_words(physics.ratio)}."
        if physics.longitudinal == 0.0
        else f"Their ratio $h/J = {physics.ratio:.2f}$ would put this chain "
        f"{phase_words(physics.ratio)} — but a third field is switched on, "
        f"$g = {physics.longitudinal:g}$ along the coupling axis, which tilts the "
        "chain towards one direction and is exactly what stops the ratio from being "
        "the whole story."
    ),
    download="chain-terms",
)
panels.hamiltonian(longitudinal_field=physics.longitudinal)
panels.model_has_no_say("anything on this page")
if exact is not None:
    st.caption(
        "**This chain is small enough that the true answer is known.** It is worked "
        "out separately, by a method the circuit has no access to, and everything the "
        "circuit does below is marked against it — which is the only reason any of "
        "this can be called right or wrong. The number itself is on the **Report** tab."
    )
else:
    st.caption(
        "**This chain is too long for the true answer to be worked out at all**, which "
        "is the regime the whole project is about. Nothing below is marked; every tab "
        "says so where it matters."
    )
panels.glossary_expander()

with st.expander("**New here? Read the tabs in this order.**", expanded=False):
    st.markdown(
        "\n".join(
            [
                "Eight tabs is a lot to arrive at. They are numbered in the order "
                "somebody meeting this problem actually needs them, and **the first "
                "five are the whole story** — each one answers the question the one "
                "before it leaves you holding.",
                "",
                "| | tab | the question it answers | needs |",
                "|---|---|---|---|",
                "| 1 | **What it is** | What is this problem, and why can the answer "
                "not just be written down? | nothing |",
                "| 2 | **What the machine does** | So what does a quantum computer "
                "actually *do* about it? | tab 1 |",
                "| 3 | **Which method** | There are four of these algorithms. Which "
                "one, and what is each actually for? | tab 2 |",
                "| 4 | **The circuit** | What does the program look like, and would a "
                "real machine run it? | tab 3 |",
                "| 5 | **Tuning it** | Does the search actually find the answer, and "
                "how many layers is it worth? | tab 4 |",
                "",
                "The last three are for a reader who wants to check the claim rather "
                "than follow the argument. **6 · The true answer** computes the "
                "target by a method the circuit cannot see. **7 · Why an ordinary "
                "computer competes** is the algebra that turns this into an ordinary "
                "problem, and it is the reason the honest verdict here is usually "
                "*no*. "
                "**Report** is every number on the page as plain text.",
                "",
                "**Nothing here is worth less if the equations are skipped.** Every "
                "formula on this page sits behind a plain sentence saying what it "
                "means, or is folded away behind one, and the sentence is the part "
                "that carries the argument.",
            ]
        )
    )

(
    what_tab,
    loop_tab,
    method_tab,
    circuit_tab,
    optimise_tab,
    evolve_tab,
    mapping_tab,
    report_tab,
) = st.tabs(
    [
        "1 · What it is",
        "2 · What the machine does",
        "3 · Which method",
        "4 · The circuit",
        "5 · Tuning it",
        "6 · The true answer",
        "7 · Why an ordinary computer competes",
        "Report",
    ]
)

# --------------------------------------------------------------------------

with what_tab:
    st.markdown("##### Where this chain sits, and what it is made of")
    st.markdown(
        "**The two halves of the problem pull in different directions, and cannot "
        "both be satisfied.** The couplings want every magnet lined up along one "
        r"axis; the push wants each of them along another. Because those two "
        "operations do not commute — doing them in the other order gives a "
        "different result — no single arrangement of magnets can be the answer. "
        "That is the only reason any of this is quantum, and it is measured below "
        "rather than asserted."
    )

    ratios, infinite_gap, finite_gap = gap_curves(
        physics.n_sites, physics.coupling, physics.boundary
    )
    panels.figure(
        figures.phase_diagram_figure(
            spec,
            np.asarray(ratios),
            np.asarray(infinite_gap),
            None if finite_gap is None else np.asarray(finite_gap),
        ),
        "The star is this chain. The curve on the right is the cheapest way to "
        "disturb the chain at all — one ripple, and the smaller it is the harder "
        "the chain is to pin down. It falls to zero at the critical push and "
        "nowhere else, which is what makes that line a line. On a chain of finite "
        "length it never quite reaches zero, which is why a real chain crosses over "
        "smoothly instead of switching.",
        download="phase-diagram",
    )
    if finite_gap is None:
        st.caption(
            "Only the infinite chain is drawn. The closed form that would give this "
            "chain's own gap wants an even-length ring, and substituting a different "
            "method into the same axes would put two quantities of different standing "
            "on one plot."
        )

    st.divider()
    if physics.n_sites > quantumness.MAX_SITES:
        st.info(
            f"**The pictures below stop at {quantumness.MAX_SITES} magnets, and this "
            f"chain has {physics.n_sites}.** They are drawn from the whole state "
            "vector and the identities under them are whole-matrix norms, so both are "
            "built densely. Turn the chain down in the sidebar to see them; the "
            "physics they show does not change with length."
        )
    else:
        found = chain_facts(physics.n_sites, physics.coupling, physics.field, physics.boundary)
        picture = ground_state_picture(
            physics.n_sites, physics.coupling, physics.field, physics.boundary
        )

        st.markdown(
            "**So what is the lowest-energy state, then?** Several arrangements of "
            "the magnets at the same time. These are the ones it is mostly made of, "
            "labelled by how many neighbouring pairs in them disagree — each such "
            "broken bond costs energy, so the arrangements with fewest of them carry "
            "the most weight."
        )
        panels.figure(
            figures.superposition_figure(picture),
            f"Those rows are **{picture.shown_weight:.0%}** of the state between them. "
            f"Counted over all {picture.full_count} possible arrangements, this chain "
            f"is **{picture.effective_count:.1f} of them at once** — that is what "
            "*superposition* means here, and it is read off the answer rather than "
            "drawn by hand. One magnet's own arrow is only "
            f"**{picture.arrow_length:.2f}** long out of 1: the rest of it is not "
            "missing, it is stored in that magnet's correlations with its neighbours. "
            "That shortfall is entanglement, drawn.",
            download="ground-state-superposition",
        )

        st.markdown(
            "**Turn the push up and that mixture spreads.** Left to right is the "
            "transition itself: the weight starts on the two aligned arrangements, "
            "moves through the ones with a broken bond in them, and ends shared out "
            "evenly among every arrangement there is."
        )
        panels.figure(
            figures.spreading_figure(
                spread_across_field(
                    physics.n_sites, physics.coupling, physics.boundary, SPREAD_RATIOS
                ),
                SPREAD_RATIOS,
            ),
            "Blue is up and orange is down; the grey bar under each row is that row's "
            "share of the state. The caption under each panel is how many arrangements "
            f"that state is at once, out of {picture.full_count}.",
            download="superposition-across-field",
        )

        st.divider()
        expander = st.expander("The algebra underneath, checked on the matrices")
        # Folded away rather than removed. It is the premise every picture above
        # rests on, and it is also five rows of operator norms -- which is exactly
        # the material that makes a reader who does not do physics close the page.
        # Anybody who wants the check opens one line; nobody else meets it.
        rows = (
            (
                r"$[\hat H_{ZZ}, \hat H_X] \neq 0$ — the two halves compete",
                f"{found.terms_commutator:.6f}",
                "not zero, when both are switched on",
                panels.WARNING
                if (found.terms_commutator > 1e-12) == found.classical
                else panels.PRESENT,
            ),
            (
                r"$\{\hat\sigma^x_0, \hat\sigma^z_0\} = 0$ — one magnet, two axes",
                f"{found.onsite_anticommutator:.2e}",
                "$0$",
                panels.PRESENT if found.onsite_anticommutator < 1e-12 else panels.WARNING,
            ),
            (
                r"$[\hat\sigma^x_0, \hat\sigma^z_1] = 0$ — different magnets are separate",
                f"{found.offsite_commutator:.2e}",
                "$0$",
                panels.PRESENT if found.offsite_commutator < 1e-12 else panels.WARNING,
            ),
            (
                r"$[\hat H, \hat P] = 0$, with $\hat P = \prod_i \hat\sigma^x_i$ — a symmetry",
                f"{found.parity_commutator:.2e}",
                "$0$",
                panels.PRESENT if found.parity_commutator < 1e-9 else panels.WARNING,
            ),
            (
                r"$\langle \hat P \rangle$ in the lowest state",
                f"{found.parity_expectation:+.6f}",
                "$+1$",
                panels.PRESENT if found.parity_expectation > 0.999 else "—",
            ),
        )
        with expander:
            st.markdown(
                "\n".join(
                    [
                        "| what it says | measured | expected | |",
                        "|---|---|---|---|",
                        *(
                            f"| {claim} | `{value}` | {want} | {held} |"
                            for claim, value, want, held in rows
                        ),
                    ]
                )
            )
            st.caption(
                "Built here from scratch rather than taken from the solver's matrices, "
                "so a shared convention cannot hide a shared mistake. This module's own "
                f"lowest energy is {found.energy:.9f}"
                + (
                    f", which is {found.energy_disagreement:.1e} from the closed-form answer."
                    if found.energy_disagreement is not None
                    else ", and no closed form covers this chain, so nothing checks it."
                )
            )

# --------------------------------------------------------------------------

with loop_tab:
    st.markdown(
        "**The quantum computer is not the algorithm. It is a subroutine inside an "
        "ordinary one.** It is handed a set of dial positions, it runs a short "
        "circuit, it reports back a single number — an energy — and an ordinary "
        "computer decides what to try next. That happens thousands of times. Nothing "
        "on this page is a quantum computer solving a problem end to end, and no "
        "algorithm anyone runs today works that way."
    )
    panels.figure(
        figures.hybrid_loop_figure(),
        "Blue is the only part that needs the quantum machine, and it is the part "
        "that runs for microseconds. Orange is an ordinary computer, and it is where "
        "almost all the wall-clock time goes. A useful way to read the whole "
        "feasibility question: the loop is worth running only if the blue box does "
        "something the orange one could not have done for itself.",
        download="hybrid-loop",
    )

    st.divider()
    st.markdown("##### The four things worth knowing about that loop")
    st.caption(
        "All four in plain words first, then the algebra twice over: folded away "
        "immediately below them, and again at the foot of the tab as one equation per "
        "step of the loop. Skipping both costs a reader none of the argument."
    )

    st.markdown(
        """
**1. The answer can never come out too low, and that is what makes it checkable.**
Whatever the dials are set to, the energy the circuit reports is at or above the
true lowest energy — never below it. So *lower is always better*, the search never
has to wonder whether it has overshot, and a result that comes back **below** the
true answer is a bug rather than a discovery. That is exactly the check the
**Tuning it** tab performs and reports on every run.
"""
    )

    st.markdown(
        r"""
**2. The energy is never measured. It is assembled out of many separate
measurements.** The chain's energy is a sum of small pieces — one for each
neighbouring pair, one for each magnet on its own — and a machine can only read one
kind of piece at a time. So each piece is measured by itself, over and over, and
the results are averaged. One run of the circuit gives one $\pm 1$ reading per
piece, never an energy.
"""
    )

    # The bond count is the chain's own, not a literal. A ring has L bonds and a row
    # has L - 1, and writing "L - 1" beside a sidebar set to "a closed ring" states a
    # coefficient sum one bond short of the one the shot arithmetic actually uses.
    st.markdown(
        rf"""
**3. That is where the cost actually is.** Averaging those $\pm 1$ readings down to
a given accuracy needs a number of repeats that grows as the **square** of the
accuracy demanded: ten times the precision costs a hundred times the runs. And the
number of pieces grows with the chain — this one has **{spec.n_bonds}** neighbouring
pairs and **{physics.n_sites}** magnets — so the bill grows on both counts at once.
This is why a feasibility verdict here is usually settled by counting measurements
rather than by anything about circuits. A noisy machine multiplies the whole thing
by a further fixed factor, shown on the **The circuit** tab.
"""
    )

    st.markdown(
        """
**4. The classical half has its own failure mode.** On real hardware, working out
which way to turn one dial costs two more circuit runs — so a single step of the
search costs twice as many runs as there are dials. The danger is worse than the
price: for a deep circuit with no structure in it, the slope the optimiser is
following shrinks exponentially as the chain gets longer. That is a **barren
plateau**, and on one there is simply nothing left to follow. The circuit family
used here is built out of the problem's own two terms rather than from generic
gates, and it starts near the identity — the two standard defences.
"""
    )

    with st.expander("The algebra behind those four points"):
        st.markdown(
            r"""
**1 — the variational bound.** For any dial positions $\theta$ whatsoever,

$$
E(\theta) \;=\; \frac{\langle \psi(\theta) | \hat H | \psi(\theta)\rangle}
                    {\langle \psi(\theta) | \psi(\theta)\rangle}
        \;\ge\; E_0 ,
$$

where $E_0$ is the true lowest energy.

**2 — the energy as a sum of measurable pieces.** The Hamiltonian is written as a
sum of terms a machine can actually read off,

$$
\hat H \;=\; \sum_\alpha c_\alpha \hat P_\alpha ,
\qquad
\hat P_\alpha \in \{\hat\sigma^z_i \hat\sigma^z_{i+1},\; \hat\sigma^x_i\} ,
$$

and each $\hat P_\alpha$ is measured on its own and averaged.
"""
        )
        st.markdown(
            rf"""
**3 — the shot arithmetic.** To reach a precision $\epsilon$,

$$
N_{{\text{{shots}}}} \;\approx\;
    \left( \frac{{\sum_\alpha |c_\alpha|}}{{\epsilon}} \right)^{{\!2}} ,
\qquad
\sum_\alpha |c_\alpha| \;=\; J\,B \;+\; h\,L \;+\; g\,L
$$

where $B$ is the number of bonds — $L$ for a ring and $L-1$ for a row, so
**{spec.n_bonds}** for the chain in the sidebar.
"""
        )
        st.markdown(
            r"""
**4 — the parameter-shift rule.** On hardware the gradient is not a finite
difference but an exact identity. For one rotation $e^{-i\theta \hat P/2}$ about a
single Pauli $\hat P$,

$$
\frac{\partial E}{\partial \theta}
  \;=\; \tfrac{1}{2}\Big[ E\big(\theta + \tfrac{\pi}{2}\big)
                        - E\big(\theta - \tfrac{\pi}{2}\big) \Big] ,
$$

which costs two circuit runs. The dials above are not single rotations: one layer
angle drives every bond gate at once, so its derivative is the sum of one such
pair per gate — $2(B + L)$ runs for a layer, not two, and $2(B + 2L)$ once the
tilt $g$ is switched on, which adds a rotation per magnet. Shifting the shared
angle itself is not the rule and does not give the gradient.
"""
        )

    st.divider()
    st.markdown("##### What runs where")
    st.markdown(
        "\n".join(
            [
                "| step | runs on | what it costs | what goes wrong |",
                "|---|---|---|---|",
                "| run the circuit | **quantum machine** | microseconds | the state "
                "leaks away as the circuit gets deeper |",
                "| read the magnets | **quantum machine** | microseconds, times "
                "however many repeats the precision demands | this is where nearly "
                "all the cost is, and it grows as the square of the accuracy wanted |",
                "| average the readings | ordinary computer | negligible | too few "
                "repeats and the energy is indistinguishable from noise |",
                "| move the dials | ordinary computer | seconds | a flat landscape "
                "leaves the optimiser nothing to follow |",
            ]
        )
    )
    st.caption(
        "Only one row of that table needs hardware nobody can buy off a shelf, and it "
        "is the row this project prices. The other three run on an ordinary computer."
    )

    st.divider()
    st.markdown("##### The same four steps, in one equation each")
    st.caption(
        "One line of algebra per row of that table, in the same order, each under a "
        "sentence saying what it means. Read together they are the whole loop: prepare a "
        "state, read one bit from it, average, step. This is the only place on the page "
        "where the quantum mechanics is written down rather than described, and a reader "
        "who skips it loses none of the four conclusions above."
    )

    st.markdown(
        r"""
**1 · Run the circuit — a state is *prepared*, not calculated.** The dials
$\theta = (\gamma, \beta)$ are angles of rotation, and the machine applies the chain's
own two terms to an undecided start, alternately, $p$ times — $p$ is the sidebar's
**depth**:

$$
|\psi(\gamma, \beta)\rangle \;=\; \prod_{k=1}^{p}
    e^{-i\beta_k \hat H_{\text{field}}} \;
    e^{-i\gamma_k \hat H_{\text{diag}}} \;
    |+\rangle^{\otimes L}
$$

$\hat H_{\text{diag}}$ is the couplings and $\hat H_{\text{field}}$ the push, so the
gates *are* the two halves of the problem rather than a generic circuit borrowed from
somewhere else. **The circuit** tab draws this same product gate by gate.
"""
    )

    st.markdown(
        r"""
**2 · Read the magnets — this is the quantum mechanics, and it is one coin toss.**
Measuring one term of the energy returns one of its two eigenvalues, $+1$ or $-1$, at
random. Only the *bias* of that coin is set by the state:

$$
\operatorname{Pr}\big(m = \pm 1\big) \;=\; \langle \psi | \hat\Pi_\pm | \psi \rangle ,
\qquad
\hat\Pi_\pm \;=\; \tfrac{1}{2}\big( \hat I \pm \hat P_\alpha \big)
$$

That single line is why the cost of this whole enterprise sits where it does. The
machine cannot hand back a number; it hands back a toss whose bias is the number.
"""
    )

    st.markdown(
        r"""
**3 · Average the readings — and the shot count falls out of it.** Average $S_\alpha$
tosses of each term and add the terms up with their coefficients. Because every reading
is $\pm 1$, the error bar is fixed before anything is run:

$$
\hat E = \sum_\alpha c_\alpha \bar P_\alpha ,
\qquad
\bar P_\alpha = \frac{1}{S_\alpha} \sum_{s=1}^{S_\alpha} m^{(\alpha)}_s ,
\qquad
\operatorname{Var}\big[\bar P_\alpha\big]
    = \frac{1 - \langle \hat P_\alpha \rangle^2}{S_\alpha}
    \;\le\; \frac{1}{S_\alpha}
$$

Split a total budget $S = \sum_\alpha S_\alpha$ across the terms in proportion to
$|c_\alpha|$ and the error on the energy becomes
$\big(\sum_\alpha |c_\alpha|\big) / \sqrt{S}$ — which is point 3's shot count,
rearranged. The square in *ten times the precision costs a hundred times the runs* is
that square root read backwards.
"""
    )

    st.markdown(
        r"""
**4 · Move the dials — the one step an ordinary computer could have done alone.**
Downhill, by the slope:

$$
\theta^{(t+1)} \;=\; \theta^{(t)} \;-\; \eta \, \nabla_\theta E\big(\theta^{(t)}\big)
$$

On a device each component of $\nabla_\theta E$ is two more circuit runs, by the exact
identity in the expander above. In the simulation this page runs it is an adjoint
gradient instead, which costs about two energy evaluations for the *whole* vector rather
than two per dial. That difference is not cosmetic: a step count measured here is not a
shot count on hardware, and the **Physics and hardware** page prices the second.
"""
    )


with method_tab:
    st.markdown(
        "**There is not one quantum algorithm here, there are four, and they are "
        "not variations on each other.** Two of them ask *what is the lowest "
        "energy of this system* — that is a physics and chemistry question. Two of "
        "them ask *which of these billions of choices is best* — that is a "
        "scheduling and routing question. They share a chassis: a short circuit on "
        "the quantum machine inside an ordinary loop on a normal one. What they do "
        "not share is what they are aiming at."
    )
    panels.figure(
        figures.method_map_figure(),
        "The same chain can be handed to all four, which is exactly why it is used "
        "to test them. Nothing here is specific to this project's chain — this is "
        "the family, and the corpus shelf *Choosing between the methods* is the "
        "long version, published in full on the Knowledge base page.",
        download="method-map",
    )

    st.markdown(
        "\n".join(
            [
                "| method | what it is for | what breaks it |",
                "|---|---|---|",
                "| **VQE** | ground-state energies: molecules, catalysts, magnets |"
                " the search stalls, or the circuit cannot reach the answer |",
                "| **QAOA** | discrete choices: routing, scheduling, portfolios |"
                " classical solvers are very good at these already |",
                "| **imaginary time** | the same target as VQE, without a search |"
                " it needs many more measurements per step |",
                "| **annealing** | large discrete problems, on analogue hardware |"
                " the answer carries no certificate that it is right |",
            ]
        )
    )
    st.markdown(
        "\n".join(
            [
                "**Every row of that table is a question the Chat page will answer "
                "with citations.** These are its own buttons, quoted from the same "
                "list the page draws them from:",
                "",
                *(f"- *{STARTERS[key]}*" for key in METHOD_QUESTIONS),
            ]
        )
    )

    st.divider()
    st.markdown("##### The one that can be run here, three ways")
    st.markdown(
        "**Where a search starts changes how much work it costs, and sometimes "
        "whether it works at all.** The three runs below are the *same chain*, the "
        "*same circuit* and the *same optimiser*. Only the opening position of the "
        "dials differs. Press the button and watch what that alone is worth."
    )
    if exact is None and exact_refusal is not None:
        no_exact_answer(exact_refusal)
    if st.button("Race the three starts", type="primary"):
        with st.spinner("Running all three..."):
            st.session_state["race"] = tuple(
                (
                    label,
                    solve(
                        n_sites=physics.n_sites,
                        depth=max(physics.depth, 1),
                        coupling=physics.coupling,
                        transverse_field=physics.field,
                        longitudinal_field=physics.longitudinal,
                        boundary=physics.boundary,
                        family=family,
                        initialisation=initialisation,
                    ),
                )
                for label, family, initialisation in figures.RACE_STARTS
            )

    race = st.session_state.get("race")
    if race is None:
        panels.empty_state(
            "Not run yet.",
            "Press **Race the three starts**. It takes a second or two and needs no network.",
        )
    else:
        panels.figure(
            figures.method_race_figure(
                tuple((label, tuple(result.energy_history)) for label, result in race),
                exact,
            ),
            "Lower is better and the dashed line is the truth. The two informed "
            "starts land in the same place: what separates them is how many steps "
            "it took to get there. The cold start is the single dot — it never "
            "moved.",
            download="method-race",
        )
        st.markdown(
            "\n".join(
                [
                    "| start | energy reached | steps taken | stopped because |",
                    "|---|---|---|---|",
                    *(
                        f"| {label} | {result.energy:.6f} | "
                        f"{len(result.energy_history)} | {result.stop_reason} |"
                        for label, result in race
                    ),
                ]
            )
        )
        st.markdown(
            "**Three things to take from that, none of which needs any physics.** "
            "*One:* the informed starts reach the same energy, so what is left over "
            "is the circuit's own ceiling and no amount of further searching will "
            "remove it — that is what tab 5's depth ladder is for. *Two:* starting "
            "from the annealing schedule gets there in roughly half the steps, which "
            "on metered hardware is half the bill for the same answer. *Three:* the "
            "cold start is stationary. The optimiser looks around, sees no slope in "
            "any direction, and stops on its first step. That is the small, cheap "
            "version of the failure the literature calls a *barren plateau*, and it "
            "is the reason nobody initialises these circuits at zero."
        )

    with st.expander("The four, in two equations each: what it does, and what it reads"):
        st.markdown(
            r"""
Two lines per method. The first is what the method *is*; the second is what a machine
actually reads back, because that is where every one of these four is expensive and it
is the half usually left out. $\hat P_\alpha$ below is one measurable piece of the
energy and $m_s = \pm 1$ one reading of it, as in the previous tab.
"""
        )
        st.markdown(
            r"""
##### VQE — search the dials

*What it does.* Minimise the energy of the circuit's state over its angles. The answer
can never come out below the truth, so lower is simply better:

$$
E(\theta) = \langle \psi(\theta) | \hat H | \psi(\theta) \rangle \;\ge\; E_0 ,
\qquad
\theta^\star = \arg\min_\theta E(\theta)
$$

*What it reads.* Never $E$. Each piece is measured on its own and the pieces are added
up on an ordinary computer:

$$
E(\theta) = \sum_\alpha c_\alpha \, \langle \hat P_\alpha \rangle_\theta ,
\qquad
\langle \hat P_\alpha \rangle_\theta \;\approx\;
    \frac{1}{S} \sum_{s=1}^{S} m^{(\alpha)}_s
$$

For this chain **two** measurement settings are enough — every
$\hat\sigma^z \hat\sigma^z$ term can be read in one and every $\hat\sigma^x$ term in the
other. The **Physics and hardware** page lists both, term by term.
"""
        )
        st.markdown(
            r"""
##### QAOA — the same chassis, a different target

*What it does.* The identical circuit, described as two blocks alternating $p$ times: a
*cost* block that scores each answer and a *mixer* block that moves between answers.

$$
|\psi(\gamma, \beta)\rangle = \prod_{k=1}^{p}
    e^{-i\beta_k \hat H_M} e^{-i\gamma_k \hat H_C} \, |+\rangle^{\otimes L} ,
\qquad
\hat H_M = \sum_i \hat\sigma^x_i
$$

*What it reads.* $\hat H_C$ is diagonal — every candidate answer is a definite
arrangement with a definite score — so **one** setting does it. Each shot returns a whole
arrangement $z$, and its score is worked out classically:

$$
\langle \hat H_C \rangle
  = \sum_z \big| \langle z | \psi(\gamma, \beta) \rangle \big|^2 \, C(z)
  \;\approx\; \frac{1}{S} \sum_{s=1}^{S} C(z_s)
$$

Because every shot *is* a candidate answer, the best arrangement seen is usable even
when the average is poor. VQE has no such licence: an average is all it ever gets.
"""
        )
        st.markdown(
            r"""
##### Imaginary time — no search at all

*What it does.* Run the clock sideways. Every state except the lowest dies away
exponentially, at a rate set by how far above the ground state it sits:

$$
e^{-\hat H \tau} |\psi(0)\rangle
  = \sum_n c_n e^{-E_n \tau} |n\rangle
  \;\; \underset{\tau \to \infty}{\longrightarrow} \;\;
  c_0 e^{-E_0 \tau} |0\rangle
$$

*What it reads.* $e^{-\hat H \tau}$ is not a rotation — it shortens the state instead of
preserving its length — so no circuit performs it directly. The
variational version projects that motion onto the circuit's own dials and solves a small
linear system at every step:

$$
\sum_k A_{jk} \, \dot\theta_k = -\, b_j ,
\qquad
A_{jk} = \operatorname{Re} \langle \partial_j \psi | \partial_k \psi \rangle ,
\qquad
b_j = \operatorname{Re} \langle \partial_j \psi | \hat H | \psi \rangle
$$

$A$ has an entry for every *pair* of dials, and that is the whole of "many more
measurements per step": overlaps growing as the square of the parameter count, where VQE
needs a number of energies growing linearly with it.
"""
        )
        st.markdown(
            r"""
##### Annealing — no dials and no circuit

*What it does.* Change the Hamiltonian slowly from one whose answer is obvious to the one
you care about, and let the chain follow along:

$$
\hat H(s) = -A(s) \sum_i \hat\sigma^x_i
          - B(s) \sum_i \hat\sigma^z_i \hat\sigma^z_{i+1} ,
\qquad s = t / t_f \in [0, 1]
$$

*What it costs, and what it reads.* The runtime is set by the narrowest energy gap
anywhere along that path — a quantity nobody knows in advance, which is the method's
central difficulty:

$$
t_f \;\gg\; \frac{\max_s \big| \langle 1 | \partial_s \hat H | 0 \rangle \big|}
                  {\Delta_{\min}^{2}} ,
\qquad
\Delta_{\min} = \min_s \big( E_1(s) - E_0(s) \big)
$$

The readout is one arrangement per run, scored classically, repeated, best kept. There is
no variational bound here: a low answer is evidence and not proof, which is the *no
certificate* line in the table above. And $\Delta_{\min}$ is the same gap the **What it
is** tab plots: it closes at the critical push, so the one place this chain is most
interesting is the one place annealing is slowest.
"""
        )
        st.markdown(
            "Skipping this expander costs nothing. Every claim on this tab is "
            "made in the sentences above it."
        )

    st.divider()
    st.markdown("##### Where to go next, by what you want to know")
    st.markdown(
        "\n".join(
            [
                "| if you want to know… | go to |",
                "|---|---|",
                "| what a circuit for this actually looks like, gate by gate |"
                " **tab 4 · The circuit** |",
                "| whether more layers are worth paying for | **tab 5 · Tuning it** |",
                "| what the circuits are chasing, computed exactly | **tab 6 · The true answer** |",
                "| why an ordinary computer is a serious competitor here |"
                " **tab 7 · Why an ordinary computer competes** |",
                "| whether a real machine could run it — wiring, gate errors, coherence |"
                " the **Physics and hardware** page |",
                "| what the agent has read, note by note, with citations |"
                " the **Knowledge base** page |",
                "| any of the questions listed above, answered in prose with citations |"
                " the **Chat** page |",
            ]
        )
    )


with circuit_tab:
    described = ansatz.describe()
    st.markdown(
        "**A quantum computer is not programmed with instructions — it is wired up "
        "as a diagram and then run.** Each magnet in the chain gets one wire, time "
        "runs left to right, and the whole program is one short pattern repeated. "
        "This is that pattern, for the chain in the sidebar."
    )
    panels.figure(
        figures.circuit_figure(physics.n_sites, max(physics.depth, 1), ansatz.bonds, ansatz.rounds),
        "Each blue connector ties two neighbouring magnets together; each orange box "
        "nudges one magnet on its own. The connectors in a column share no wire, so "
        "they all happen at the same moment — and a chain only ever needs "
        f"**{ansatz.two_qubit_rounds_per_layer} such columns per layer, however long "
        "it is**. Doing them one after another instead would make the program "
        f"**{described['depth_saving_factor']} times longer** for exactly the same "
        "result, which is the single largest saving on this page and it comes from "
        "noticing the pattern rather than from better hardware.",
        download="ansatz-circuit",
    )
    st.caption(
        "The two dials the optimiser turns per layer are the same for every magnet, "
        "so **the search does not get harder as the chain gets longer**. That is the "
        "whole reason this circuit family is worth running rather than one with a "
        "dial per wire."
    )

    st.divider()
    st.markdown("##### Would a real machine run it?")
    machine = device_for(physics.device)
    refusal = fits(physics.n_sites, machine)
    if refusal is not None:
        st.warning(f"{panels.WARNING} {refusal}")
    else:
        compiled = transpile(ansatz, machine)
        surviving = estimate(compiled)
        stretched = tuple(bond.path for bond in compiled.bonds if bond.distance > 1)
        st.markdown(
            "**The chain wants every magnet next to the next one. A real machine is "
            "wired in a fixed pattern that nobody chose with this chain in mind.** "
            "So the chain has to be laid onto the machine's wiring, and wherever two "
            "neighbours land on two chips that are not wired together, their states "
            "have to be shuffled along until they meet."
        )
        panels.figure(
            figures.wiring_figure(
                machine.n_qubits,
                machine.coupling,
                compiled.layout.sites,
                stretched,
                machine.name,
            ),
            (
                "Every neighbouring pair landed on a pair of chips that are already "
                "wired together, so nothing has to travel — this chain fits the "
                "machine as it is."
                if not stretched
                else f"{len(stretched)} of the chain's neighbouring pairs landed apart "
                "and are drawn as orange dashes: those states have to be shuffled "
                "along the grey wiring to meet, and every shuffle is more gates and "
                "more time. That detour is what *wiring overhead* means."
            ),
            download="chain-on-machine",
        )
        st.markdown(
            f"**About {surviving.total:.0%} of the signal survives the trip.** A "
            "quantum state leaks away while a program runs, and every extra gate and "
            f"every extra microsecond costs some of it. The largest single loss here "
            f"is **{surviving.dominant_loss}**. What is lost is not wrong answers — it "
            "is faintness, and the cure is repetition: recovering the same accuracy "
            f"from this much signal takes **{surviving.shot_inflation():.1f} times** as "
            "many repeats as a perfect machine would need."
        )
        with st.expander("The numbers behind those two pictures"):
            panels.metrics(
                {
                    "Dials the optimiser turns": described["n_parameters"],
                    "Two-qubit gates": described["two_qubit_gates"],
                    "Depth, scheduled": described["two_qubit_depth"],
                    "Depth, naive": described["naive_two_qubit_depth"],
                }
            )
            panels.metrics(
                {
                    "Depth on this machine": compiled.two_qubit_depth,
                    "Wiring costs": f"x{compiled.routing_overhead:.1f}",
                    "Runs for": f"{compiled.duration_us:.2f} us",
                    "Signal that survives": f"{surviving.total:.1%}",
                }
            )
            st.caption(
                f"Coherence time {machine.t2_ns / 1000:.0f} microseconds. {machine.provenance}"
            )

with optimise_tab:
    st.markdown(
        "**The circuit has dials on it, and nobody knows in advance where they "
        "should be set.** So a search turns them, thousands of times, always "
        "downhill: try a setting, read the energy, adjust, repeat. Two questions "
        "decide whether that is worth doing, and both are answered on this tab — "
        "*does the search actually reach the answer*, and *how many layers of "
        "circuit is it worth paying for*."
    )
    st.caption(
        "Under the hood: L-BFGS on an exact gradient. While simulating, the gradient "
        "costs about two energy evaluations rather than one circuit run per dial, so "
        "there is nothing to gain from a gradient-free optimiser here -- real "
        "hardware is the opposite case and is served differently."
    )
    if exact is None and exact_refusal is not None:
        no_exact_answer(exact_refusal)
    if st.button("Run the optimiser", type="primary"):
        with st.spinner("Turning the dials..."):
            st.session_state["vqe"] = solve(
                n_sites=physics.n_sites,
                depth=max(physics.depth, 1),
                coupling=physics.coupling,
                transverse_field=physics.field,
                boundary=physics.boundary,
            )

    result = st.session_state.get("vqe")
    if result is None:
        panels.empty_state(
            "Not run yet.",
            "Press **Run the optimiser**. It takes a second or two and needs no network.",
        )
    else:
        if exact is None:
            st.info(
                f"Stopped because: {result.stop_reason}. With no exact answer for a "
                "chain this long there is nothing to grade this against, so it is "
                "reported as a number reached rather than as an error."
            )
        elif result.energy - exact < -1e-9:
            st.error(
                f"{panels.WARNING} The result is **below** the exact answer. This "
                "family gives an upper bound and can never legitimately do that, so "
                "it is a bug rather than a discovery."
            )
        else:
            st.success(
                f"{panels.PRESENT} Above the exact answer by {result.energy - exact:.2e}, "
                f"as a variational bound must be. Stopped because: {result.stop_reason}."
            )
        panels.figure(
            figures.convergence_figure(tuple(result.energy_history), exact),
            "The dashed line is the true answer, computed afterwards by a method the "
            "optimiser had no access to. The solid curve is what the circuit managed."
            if exact is not None
            else "No dashed line: nothing here knows the true answer for a chain this "
            "long, which is exactly the situation the project is about.",
            download="optimiser-convergence",
        )
        with st.expander("The numbers behind that curve"):
            panels.metrics(
                {
                    "Energy reached": f"{result.energy:.6f}",
                    "Exact answer": "not available" if exact is None else f"{exact:.6f}",
                    "Gap": "—" if exact is None else f"{result.energy - exact:.2e}",
                    "Circuit evaluations": result.n_energy_evaluations,
                }
            )

    st.divider()
    st.markdown("##### What each added layer bought")
    st.caption(
        "Each depth is warm-started from the one below, stretched onto a finer "
        "schedule, so the whole ladder costs about as much as one cold start at the "
        "deepest level."
    )
    if st.button("Climb the depth ladder"):
        with st.spinner("Adding a layer at a time..."):
            st.session_state["ladder"] = depth_sweep(
                n_sites=physics.n_sites,
                max_depth=max(physics.depth, 2),
                coupling=physics.coupling,
                transverse_field=physics.field,
                boundary=physics.boundary,
            )

    ladder = st.session_state.get("ladder")
    if ladder is None:
        panels.empty_state(
            "The ladder has not been climbed yet.",
            "Press **Climb the depth ladder** to see what each extra layer is worth.",
        )
    else:
        gains = ladder.marginal_gain()
        panels.figure(
            figures.ladder_figure(tuple(ladder.depths), tuple(ladder.energies), exact),
            "Each point is one more layer of circuit. What a depth decision turns on "
            "is how steeply this falls: once one more layer stops buying more accuracy "
            "than anybody needs, it is not worth the time it adds — a statement about "
            "the budget rather than about the physics. A curve that flattens early "
            "means the extra layers are being paid for and not used."
            if exact is not None
            else "Each point is one more layer of circuit. With no exact answer for a "
            "chain this long the distance cannot be plotted, so this is the raw energy "
            "each depth reached; lower is better.",
            download="depth-ladder",
        )
        with st.expander("The numbers behind that curve"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "layers": ladder.depths,
                        "energy": [round(value, 8) for value in ladder.energies],
                        "above exact": (
                            ["—"] * len(ladder.energies)
                            if exact is None
                            else [f"{value - exact:.2e}" for value in ladder.energies]
                        ),
                        "this layer bought": [f"{value:.2e}" for value in gains],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        if ladder.regressions:
            st.warning(
                f"{panels.WARNING} Depths {list(ladder.regressions)} came back worse "
                "than the depth below them. A deeper circuit strictly contains every "
                "shallower one, so that is a local minimum in the search rather than "
                "a limit of the family. It is reported rather than smoothed away."
            )

# --------------------------------------------------------------------------

with evolve_tab:
    st.markdown(
        "**What is the circuit actually aiming at, and how far short does it "
        "stop?** There is a way of grinding any starting state down to the "
        "lowest-energy one that has nothing to do with circuits and cannot be run "
        "on hardware at all. That makes it a clean target: it shows what the answer "
        "is, so the circuit's attempt can be held against it. The curve below is "
        "that grinding, watched as it happens."
    )
    st.caption(
        r"In the technical words: imaginary time drives any starting state towards "
        r"the lowest-energy one -- "
        r"under $e^{-\tau \hat H}$ every excited component decays faster than the "
        r"ground state, so the energy falls and flattens onto the true answer. It is "
        r"not something hardware can run -- it is not a physical evolution -- which "
        r"is exactly why it makes a clean reference for what the circuit is trying "
        r"to approximate."
    )
    evolution_refusal = exact_diagonalisation.unsupported_reason(spec)
    if evolution_refusal is not None:
        st.info(
            f"**Not computed for this chain.** {evolution_refusal}\n\n"
            "The curve is produced by exponentiating the full matrix, so it inherits "
            "the same ceiling as the exact answer itself. Turn the chain down in the "
            "sidebar to see it."
        )
    else:
        matrix = exact_diagonalisation.hamiltonian(spec)
        start = np.ones(2**physics.n_sites) / np.sqrt(2**physics.n_sites)
        times = np.linspace(0.0, MAX_IMAGINARY_TIME, IMAGINARY_TIME_STEPS)
        step = float(times[1] - times[0])
        energies: list[float] = []
        state = start
        for _ in times:
            normalised = state / float(np.linalg.norm(state))
            energies.append(float(normalised @ (matrix @ normalised)))
            state = expm_multiply(-step * matrix, normalised)

        reached = energies[-1]
        panels.figure(
            figures.imaginary_time_figure(times, tuple(energies), exact),
            "The starting state is the equal mixture of every arrangement, which is "
            "also where the circuit starts. How fast this falls is set by the energy "
            "gap, and the gap is smallest at the critical point — the same reason "
            "the critical point is where every approximate method struggles. Move "
            "the push to $h/J = 1$ in the sidebar and watch this curve take longer "
            "to flatten.",
            download="imaginary-time",
        )
        with st.expander("The numbers behind that curve"):
            panels.metrics(
                {
                    "Started at": f"{energies[0]:.6f}",
                    "Reached": f"{reached:.6f}",
                    "Exact answer": "not available" if exact is None else f"{exact:.6f}",
                    "Still above by": "—" if exact is None else f"{reached - exact:.2e}",
                }
            )

        st.divider()
        # The even-parity gap, not the single-quasiparticle energy. The starting
        # state above is the equal mixture of every arrangement, which is a parity
        # eigenstate, and every layer of the circuit preserves parity -- so what
        # sets the rate of this descent is the cheapest excitation *within that
        # sector*, which costs two quasiparticles. Quoting one of them would halve
        # every convergence-time estimate on the page, always optimistically.
        reachable = (
            free_fermions.parity_even_gap(spec)
            if free_fermions.unsupported_reason(spec) is None
            else None
        )
        if reachable is not None:
            st.markdown(
                "**What sets how fast that curve falls:** the cheapest excitation "
                f"this evolution can actually mix in, worth **{reachable:.4f}**."
            )
            st.caption(
                "Not the energy of a single quasiparticle, which is half this. The "
                "starting state is a parity eigenstate and nothing here breaks that "
                "symmetry, so excitations arrive in pairs and the pair is what the "
                "descent has to suppress. Computed from the closed-form solution, "
                "which shares no algebra with the matrix used for the curve above — "
                "two routes agreeing is evidence, one route checking itself is not."
            )
        else:
            st.caption(
                "The closed form does not cover this setting, so this is not shown "
                "rather than being computed a second way from the same matrix."
            )

# --------------------------------------------------------------------------

with mapping_tab:
    st.markdown(
        "**One row of magnets with a quantum push on it is the same problem as one "
        "flat sheet of ordinary magnets with nothing quantum in it at all.** That is "
        "not an analogy. It is an exact rewriting, and it is the reason an ordinary "
        "computer is a serious competitor here rather than a straw man: whatever the "
        "circuit is trying to do, a sampler can do to the sheet instead."
    )
    panels.figure(
        figures.dual_lattice_figure(
            min(physics.n_sites, 8), MAPPING_SLICES, periodic=physics.boundary == "periodic"
        ),
        "The chain on the left becomes the sheet on the right. Every row of the sheet "
        "is one copy of the chain; the extra direction is **imaginary time**, never a "
        "second row of real magnets. Blue bonds are the original couplings, repeated "
        "in every copy. Orange bonds join each magnet to itself in the next copy — "
        "they are what the quantum push turned into. The answer is read off the "
        "middle row.",
        download="quantum-to-classical-lattice",
    )

    st.divider()
    st.markdown("##### The rewriting, in five steps")
    st.markdown(
        "\n".join(
            [
                "In ordinary words first. Each step is a piece of bookkeeping that "
                "changes nothing about the problem and everything about who can solve "
                "it.",
                "",
                "1. **Stop describing the answer as a circuit.** Describe it instead "
                "as the starting state, damped repeatedly by the two halves of the "
                "problem in turn. The strengths of those dampings are the dials — the "
                "same job the circuit's angles do, one tab over.",
                "2. **Slip a complete list of every possible arrangement in between "
                "every damping.** Adding up over every arrangement changes nothing, "
                "the way multiplying by one changes nothing. What it buys is that the "
                "single quantum object becomes a **row of ordinary arrangements**, one "
                "per insertion. Those are the rows of the sheet in the picture.",
                "3. **The coupling half turns into ordinary weights inside each "
                "row.** It does not mix arrangements at all, so each row just picks up "
                "a number — an ordinary magnet-to-magnet coupling along the row.",
                "4. **The sideways push turns into couplings between one row and the "
                "next.** This is the step where the quantum part disappears. A push "
                "that knocked magnets over has become a plain coupling joining each "
                "magnet to *itself* in the following row — the orange bonds in the "
                "picture.",
                "5. **Collect everything.** What is left is a sheet of ordinary "
                "magnets that each point up or down, with ordinary couplings. No "
                "operators, no complex numbers, no superposition.",
            ]
        )
    )

    with st.expander("The same five steps, with the algebra"):
        st.markdown(
            r"""
**Step 1 — start from a family of trial states.** Rather than a circuit, write the
trial state as repeated damping by the two halves of the Hamiltonian:

$$
|\psi_P(\alpha, \beta)\rangle \;=\; \mathcal{N} \prod_{p=P}^{1}
    e^{-\beta_p \hat H_{\rm field}}\, e^{-\alpha_p \hat H_{\rm diag}}\,
    |+\rangle^{\otimes L},
$$

$$
\hat H_{\rm diag} = -\!\!\sum_{\langle ij \rangle}\! \hat\sigma^z_i \hat\sigma^z_j
        \;-\; \frac{g}{J} \sum_i \hat\sigma^z_i ,
\qquad
\hat H_{\rm field} = -\sum_i \hat\sigma^x_i .
$$

The $2P$ numbers $\alpha_p, \beta_p$ are the dials — the same role the circuit's
angles play, one tab over.
"""
        )

        st.markdown(
            r"""
**Step 2 — insert every possible arrangement of the magnets, between every factor.**
Writing $s \in \{-1, +1\}^L$ for one arrangement, the identity

$$
\sum_{s} |s\rangle\langle s| \;=\; \hat{\mathbb{1}}
$$

may be dropped in anywhere without changing anything. Dropping one in between
every exponential in Step 1 turns a single quantum object into a **sum over
$2P+1$ arrangements in a row** — one per insertion, with the quantity being
measured sitting on the middle one. Those are the rows of the sheet above.
"""
        )

        st.markdown(
            r"""
**Step 3 — the diagonal factors are just numbers.** $\hat H_{\rm diag}$ is diagonal
in this basis, so it does not mix arrangements at all:

$$
\langle s |\, e^{-\alpha_p \hat H_{\rm diag}} \,| s \rangle
  \;=\; \exp\!\Big( \alpha_p \!\!\sum_{\langle ij \rangle}\! s_i s_j
        \;+\; \alpha_p \frac{g}{J} \sum_i s_i \Big).
$$

Read the right-hand side as an ordinary Ising weight *within* one row: a coupling
$J_x(p) = \alpha_p$ along the row, and a plain magnetic field $B(p) = (g/J)\,\alpha_p$
on each magnet.
"""
        )

        st.markdown(
            r"""
**Step 4 — the push factors become couplings *between* rows.** $\hat H_{\rm field}$
acts on one magnet at a time, so it factorises, and each factor is a
$2 \times 2$ matrix that can be rewritten as an Ising weight:

$$
\langle s'_i |\, e^{\,\beta_p \hat\sigma^x_i} \,| s_i \rangle
  \;\propto\; \exp\big( J_\tau(p)\, s_i s'_i \big),
\qquad
J_\tau(p) \;=\; \tfrac{1}{2} \ln \coth \beta_p .
$$

This is the step where the quantum part disappears. A push that knocked magnets
sideways has become a plain coupling joining each magnet to *itself* in the next
row — the orange bonds in the picture.
"""
        )

        st.markdown(
            r"""
**Step 5 — collect the factors.** Everything is now a product of ordinary Ising
weights on an $L \times (2P+1)$ sheet:

$$
H_{\rm cl}(s) \;=\; -\sum_t J_x(t) \!\!\sum_{\langle ij\rangle}\! s_{i,t}\, s_{j,t}
              \;-\; \sum_t J_\tau(t) \sum_i s_{i,t}\, s_{i,t+1}
              \;-\; \sum_t B(t) \sum_i s_{i,t},
$$

sampled with weight $e^{-H_{\rm cl}(s)}$. No operators, no complex numbers, no
superposition — a sheet of magnets that each point up or down.
"""
        )

    st.success(
        f"{panels.PRESENT} **Every one of those weights is real and positive.** There "
        "is no sign problem here, so an ordinary sampler can evaluate this family to "
        "any precision you are willing to wait for. That is the whole reason this "
        "chain is the honest place to ask the feasibility question — the classical "
        "competitor is not handicapped, and any claim of quantum advantage has to "
        "beat it as it actually is."
    )
    st.caption(
        "Implemented in `src/physics/classical/dual_lattice.py`, which carries this "
        "derivation in its own docstring, and sampled by "
        "`src/physics/classical/variational_imaginary_time.py`. The mapping "
        "generalises to any lattice; this project implements the chain and nothing "
        "else, and the extra direction is always imaginary time."
    )

    with st.expander("The couplings this produces, for a two-layer example"):
        example = dual_lattice.build_couplings(
            np.array([0.4, 0.3, 0.5, 0.2]), field_ratio=physics.longitudinal / physics.coupling
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "slice": list(range(len(example.spatial))),
                    "along the row  J_x": [round(float(value), 6) for value in example.spatial],
                    "between rows  J_tau": [round(float(value), 6) for value in example.temporal],
                    "field  B": [round(float(value), 6) for value in example.longitudinal],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            r"From $\alpha = (0.4, 0.3)$ and $\beta = (0.5, 0.2)$, through the same "
            r"`build_couplings` the sampler calls. Five slices for two layers, because "
            r"$2P + 1$ is always odd — there has to be a middle one for the answer to "
            r"be read off."
        )


with report_tab:
    st.markdown("##### Everything on this page, as text")
    st.caption(
        "A page whose findings can only be screenshotted is a page whose findings do "
        "not travel. This is the same arithmetic the other six tabs draw, in a block "
        "that can be pasted into a message, an issue or a report."
    )
    described = ansatz.describe()
    per_magnet = "unavailable" if exact is None else f"{exact / physics.n_sites:.9f}"
    lines = [
        f"chain               {spec.label()}",
        f"h/J                 {physics.ratio:.4f}  ({phase_words(physics.ratio)})",
        f"exact energy        {'unavailable' if exact is None else f'{exact:.9f}'}",
        f"  per magnet        {per_magnet}",
        "",
        f"circuit layers      {physics.depth}",
        f"dials               {described['n_parameters']}",
        f"two-qubit gates     {described['two_qubit_gates']}",
        f"depth, scheduled    {described['two_qubit_depth']}",
        f"depth, naive        {described['naive_two_qubit_depth']}",
        f"depth saving        x{described['depth_saving_factor']}",
        "",
        f"machine             {physics.device}",
    ]
    machine_refusal = fits(physics.n_sites, device_for(physics.device))
    if machine_refusal is not None:
        lines.append(f"  refused           {machine_refusal}")
    else:
        compiled = transpile(ansatz, device_for(physics.device))
        surviving = estimate(compiled)
        lines += [
            f"  depth on it       {compiled.two_qubit_depth}",
            f"  wiring overhead   x{compiled.routing_overhead:.2f}",
            f"  duration          {compiled.duration_us:.2f} us",
            f"  signal surviving  {surviving.total:.4%}",
            f"  dominant loss     {surviving.dominant_loss}",
            f"  shot inflation    x{surviving.shot_inflation():.2f}",
        ]
    finished = st.session_state.get("vqe")
    if finished is not None:
        lines += [
            "",
            f"optimiser reached   {finished.energy:.9f}",
            f"  evaluations       {finished.n_energy_evaluations}",
            f"  stopped because   {finished.stop_reason}",
        ]
        if exact is not None:
            lines.append(f"  above exact by    {finished.energy - exact:.3e}")
    lines += [
        "",
        "computed by src/physics/ and src/hardware/, with no language model involved.",
    ]
    st.code("\n".join(lines), language="text")

panels.footer("src/physics/ and src/hardware/")
