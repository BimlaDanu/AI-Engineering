r"""The equations a variational circuit is a picture of, composed rather than recalled.

Two surfaces show a reader what a variational circuit is -- the drawing on the
Chat page and the answer on the code branch -- and the equation must be the same
on both. Written out twice it drifts twice; written by a language model it drifts
every run, and this is the part of a code answer the reader checks the code
against.

So every string here is composed from integers -- :attr:`AnsatzSpec.depth`,
:attr:`AnsatzSpec.n_qubits`, the bond list -- by the same specification that
prices the circuit everywhere else. Nothing here calls a model.

Three equations, for three questions:

*What does the circuit build?* A product of unitaries, one pair per layer, read
right to left. That ordering is stated in words beside the equation, being the
thing a reader new to circuits most often gets backwards.

*What comes back off the machine?* Not the state. A shot returns one :math:`\pm 1`
per magnet and the energy is an average of products of those. The gap between the
wavefunction and the tally of coin flips is invisible in code.

*What does that cost?* The shot count follows from the accuracy wanted and the
:math:`L_1` norm of the Hamiltonian\'s coefficients, quadratically in the
accuracy. It is composed from
:func:`~src.physics.quantum.hamiltonians.ising_chain`, the same call the agent
prices a run with, so the equation and the verdict cannot disagree.
"""

from __future__ import annotations

import math

from src.physics.lattice import Lattice
from src.physics.model import BoundaryCondition
from src.physics.quantum.ansatz import AnsatzSpec
from src.physics.quantum.hamiltonians import ising_chain

MAX_LAYERS_WRITTEN_OUT = 3
"""Layers spelled out in full before the middle is elided.

Three pairs of exponentials is a line a reader follows. Eight is a line they skip, and
a skipped equation teaches nothing -- so past this the middle becomes ``\\cdots`` and the
first and last layers carry the pattern.
"""


def _layer_factor(layer: int) -> str:
    r"""One layer as a pair of exponentials, field applied after the coupling.

    Args:
        layer: Which layer, counted from one, matching the subscript on its angles.

    Returns:
        The LaTeX for :math:`e^{-i\beta_k \hat H_\text{field}} e^{-i\gamma_k \hat
        H_\text{diag}}`, with no spacing around it.
    """
    return (
        rf"e^{{-i\beta_{{{layer}}} \hat H_\text{{field}}}}"
        rf"e^{{-i\gamma_{{{layer}}} \hat H_\text{{diag}}}}"
    )


def state_preparation_latex(spec: AnsatzSpec) -> str:
    r"""The state the circuit prepares, written out layer by layer.

    Written as an explicit product rather than as :math:`\prod_{k=1}^{p}` because the
    product notation hides the one thing a reader has to take from it: the factor
    written *last* acts *first*. A reader who takes the leftmost exponential as the
    first gate has the circuit backwards and will read every subsequent picture
    backwards too.

    Args:
        spec: The circuit. Its depth sets how many factors appear and its width sets
            the register the initial state is written over.

    Returns:
        The LaTeX, with no delimiters, ready to be wrapped in display maths.

    Examples:
        >>> "\\beta_{1}" in state_preparation_latex(AnsatzSpec(n_qubits=4, depth=1))
        True
        >>> "cdots" in state_preparation_latex(AnsatzSpec(n_qubits=4, depth=6))
        True
    """
    depth = max(spec.depth, 1)
    if depth <= MAX_LAYERS_WRITTEN_OUT:
        factors = [_layer_factor(layer) for layer in range(depth, 0, -1)]
    else:
        factors = [_layer_factor(depth), r"\cdots", _layer_factor(1)]
    return (
        rf"\lvert\psi_{{{depth}}}(\boldsymbol\gamma,\boldsymbol\beta)\rangle \;=\; "
        + r"\, ".join(factors)
        + rf"\, \lvert +\rangle^{{\otimes {spec.n_qubits}}}"
    )


def halves_latex(*, longitudinal: bool = False) -> str:
    r"""The Hamiltonian split into the two halves the circuit alternates between.

    The split is not a presentational convenience -- it is the design of the circuit.
    One half is diagonal and factors into commuting two-magnet gates; the other is
    single-magnet and costs no two-qubit depth. Every gate in every layer comes from one
    of the two.

    Args:
        longitudinal: Whether the chain carries a field along the coupling direction.
            It joins the diagonal half, which is why it costs no two-qubit depth.

    Returns:
        The LaTeX, with no delimiters.
    """
    diagonal = r"-J\sum_i \hat\sigma^z_i\hat\sigma^z_{i+1}"
    if longitudinal:
        diagonal += r" - g\sum_i \hat\sigma^z_i"
    return (
        rf"\hat H \;=\; \underbrace{{{diagonal}}}_{{\hat H_\text{{diag}}}} \;+\; "
        r"\underbrace{-\,h\sum_i \hat\sigma^x_i}_{\hat H_\text{field}}"
    )


def energy_latex(*, longitudinal: bool = False) -> str:
    r"""The quantity the optimiser minimises, term by term.

    Args:
        longitudinal: Whether the chain carries a field along the coupling direction.

    Returns:
        The LaTeX, with no delimiters.
    """
    terms = [r"-\,J\sum_i \langle \hat\sigma^z_i \hat\sigma^z_{i+1} \rangle"]
    if longitudinal:
        terms.append(r"g\sum_i \langle \hat\sigma^z_i \rangle")
    terms.append(r"h\sum_i \langle \hat\sigma^x_i \rangle")
    return (
        r"E(\boldsymbol\gamma,\boldsymbol\beta) \;=\; \langle\psi\rvert \hat H "
        r"\lvert\psi\rangle \;=\; " + r" \;-\; ".join(terms)
    )


ESTIMATOR_LATEX = (
    r"\langle \hat\sigma^z_i \hat\sigma^z_{i+1} \rangle \;\approx\; "
    r"\frac{1}{M}\sum_{m=1}^{M} s^{(m)}_i\, s^{(m)}_{i+1}, "
    r"\qquad s^{(m)}_i = \pm 1"
)
r"""How an expectation value is actually obtained, given a machine that only counts.

No quantum computer hands back :math:`\lvert\psi\rangle`. Each of :math:`M` shots
collapses the register to one arrangement of :math:`\pm 1`, and every number in the
energy is an average over those tallies. This equation is the bridge between the
wavefunction in :func:`state_preparation_latex` and the arithmetic a shot budget prices.
"""

SHOT_COST_LATEX = (
    r"\varepsilon \;\sim\; \frac{\lVert c \rVert_1}{\sqrt{M}} "
    r"\qquad\Longrightarrow\qquad "
    r"M \;\sim\; \left(\frac{\lVert c \rVert_1}{\varepsilon}\right)^{2}, "
    r"\qquad \lVert c \rVert_1 = \sum_\alpha \lvert c_\alpha \rvert"
)
r"""What accuracy costs in measurements, and why the cost is brutal.

The per-reading factor :func:`src.agent.graph._price_shots` evaluates, written out
rather than restated: the error of an average falls as :math:`1/\sqrt{M}`, so ten
times the accuracy costs a hundred times the shots. A campaign multiplies it by the
readings an optimisation takes and by what the machine's noise adds; every large
measurement count this project reports starts here.
"""

VARIATIONAL_BOUND_LATEX = (
    r"E(\boldsymbol\gamma,\boldsymbol\beta) \;\ge\; E_0 \quad "
    r"\text{for every } (\boldsymbol\gamma,\boldsymbol\beta)"
)
r"""Why a wrong answer here is still a usable answer.

No choice of angles can drive the energy below the true ground state, so the number the
circuit reports is an upper bound and the optimiser is only ever pushing it down. It is
what makes the check at the bottom of a drafted program meaningful: a value *below* the
closed form is a bug, not a better result.
"""


def coefficient_l1(
    n_qubits: int,
    coupling: float,
    transverse_field: float,
    longitudinal_field: float = 0.0,
    boundary: BoundaryCondition = "open",
    lattice: Lattice | None = None,
) -> float:
    r"""The :math:`L_1` norm of the Hamiltonian's coefficients, for the shot equation.

    Delegated to :func:`~src.physics.quantum.hamiltonians.ising_chain` rather than
    computed from :math:`J(L-1) + hL` here, so the number under the equation on the page
    is the number the agent priced the run with. Two routes to one quantity is one route
    too many when a reader is being invited to check the arithmetic.

    Args:
        n_qubits: Chain length.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field along the coupling direction.
        boundary: Ring or segment, which decides whether the last bond exists.
        lattice: The shape the sites sit on, or ``None`` for a chain. It belongs in
            the signature because the norm counts bonds: a 3x3 triangular cluster
            has sixteen where a nine-site chain has eight, and the shots go as the
            square of the difference.

    Returns:
        The sum of the absolute values of the Pauli coefficients.
    """
    return ising_chain(
        n_sites=n_qubits,
        coupling=coupling,
        transverse_field=transverse_field,
        longitudinal_field=longitudinal_field,
        boundary=boundary,
        lattice=lattice,
    ).coefficient_l1()


def circuit_mathematics(
    spec: AnsatzSpec,
    coupling: float | None = None,
    transverse_field: float | None = None,
    longitudinal_field: float = 0.0,
    target_error: float = 0.01,
) -> str:
    r"""The three equations behind a drafted program, as markdown a page can render.

    Placed **above** the code on the ``implement`` branch, because a reader who has
    already scrolled past eighty lines of Python has stopped asking what the program is
    for. The order is the order a circuit runs in: what state it builds, what is
    measured on it, what that measurement costs.

    Every ``$$`` sits alone on its own line -- Streamlit renders display maths no other
    way, and an equation that renders as literal dollar signs is worse than no equation.

    Args:
        spec: The circuit the question described. Supplies every integer here.
        coupling: The Ising coupling :math:`J`, when the question named a chain. Without
            it the worked shot count is omitted rather than computed from a default: a
            measurement count is the number a reader is most likely to quote, and one
            invented from a chain they did not ask about would be quoted wrongly.
        transverse_field: The transverse field :math:`h`, on the same terms.
        longitudinal_field: The field along the coupling direction. Non-zero adds its
            term to the diagonal half and to the energy.
        target_error: The energy accuracy the worked shot count is priced against, in
            the same units as the energy.

    Returns:
        The markdown block. Never empty: the equations hold whether or not a chain was
        named, and only the paragraph carrying real numbers depends on one.
    """
    longitudinal = longitudinal_field != 0.0
    angles = spec.n_parameters
    settings = 2
    parts = [
        "**What the circuit builds.** Every box in the diagram is a *unitary* -- a "
        "reversible rotation of the whole register, written "
        r"$e^{-i\theta\hat P}$, whose angle $\theta$ is a number the optimiser is free "
        "to change. The circuit is those rotations multiplied together, in layers:",
        "",
        "$$",
        state_preparation_latex(spec),
        "$$",
        "",
        f"Read it **right to left**: the rightmost factor acts first, so layer 1 is the "
        f"first thing that happens to the register and layer {max(spec.depth, 1)} is the "
        f"last. The starting state $\\lvert +\\rangle^{{\\otimes {spec.n_qubits}}}$ is "
        "every magnet pointing along the field at once -- the ground state of the field "
        "term on its own, and the state a row of Hadamard gates prepares. The two "
        "operators being exponentiated are the two halves of the Hamiltonian:",
        "",
        "$$",
        halves_latex(longitudinal=longitudinal),
        "$$",
        "",
        f"and that split is the whole design. $\\hat H_\\text{{diag}}$ is a sum of "
        f"commuting two-magnet terms, so its exponential factors into one gate per bond "
        f"and the bonds run **{spec.two_qubit_rounds_per_layer} rounds** at a time "
        f"however long the chain is. $\\hat H_\\text{{field}}$ is one rotation per wire "
        f"and costs no two-qubit depth at all. There are **{angles} angles** in total, "
        f"two per layer -- and that count does not grow with the {spec.n_qubits} "
        "magnets, which is the reason this family is worth running.",
        "",
        "**What is measured, which is not the state.** No quantum computer hands back "
        r"$\lvert\psi\rangle$. What the optimiser minimises is one number,",
        "",
        "$$",
        energy_latex(longitudinal=longitudinal),
        "$$",
        "",
        "and each bracket in it is an average over shots. One shot collapses the "
        "register to one arrangement of $\\pm 1$, and the estimate is the tally:",
        "",
        "$$",
        ESTIMATOR_LATEX,
        "$$",
        "",
        f"The $\\hat\\sigma^x$ terms need the basis rotated first -- a Hadamard on each "
        f"wire before the measurement -- because the machine only ever reads one basis. "
        f"All the $\\hat\\sigma^z$ terms commute with each other and so do all the "
        f"$\\hat\\sigma^x$ terms, so the whole energy takes **{settings} measurement "
        f"settings**, not one per term.",
        "",
        "**What that costs.** The error of an average falls as the square root of the "
        "number of samples, so the shots needed rise as its square:",
        "",
        "$$",
        SHOT_COST_LATEX,
        "$$",
        "",
    ]
    if coupling is not None and transverse_field is not None:
        norm = coefficient_l1(
            spec.n_qubits,
            coupling,
            transverse_field,
            longitudinal_field,
            spec.boundary,
            spec.lattice,
        )
        shots = math.ceil((norm / target_error) ** 2)
        parts.append(
            f"For this chain $\\lVert c \\rVert_1 = {norm:g}$, so **one** energy "
            f"reading to an accuracy of {target_error:g} costs about "
            f"**{shots:,} shots** -- and an optimisation needs hundreds of readings. "
            "That multiplication, not the gate count, is what usually decides whether a "
            "variational run is affordable."
        )
    else:
        parts.append(
            "The norm is fixed by the chain, so the shot count follows from $J$, $h$ "
            "and the length as soon as those are named."
        )
    parts += [
        "",
        "**Why an imperfect answer is still an answer.** No choice of angles can push "
        "the energy below the true ground state,",
        "",
        "$$",
        VARIATIONAL_BOUND_LATEX,
        "$$",
        "",
        "so whatever comes back is an upper bound on the right answer, and a value "
        "*below* the closed form at the bottom of the program is a bug rather than a "
        "better result.",
    ]
    return "\n".join(parts)
