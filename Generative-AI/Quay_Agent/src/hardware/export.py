r"""The circuit and its paperwork, as files somebody else can act on.

A feasibility study that ends in a paragraph is an opinion. This module is what
turns it into an artefact: the exact circuit in a language a machine accepts, the
numbers it was priced with, and a note saying what to check. Three things come out
of :func:`submission_package` and each has a different reader.

``circuit.qasm`` is for the machine. OpenQASM 3, written against the *physical*
qubits the layout chose, with every routing SWAP that the cost model charged for
present in the file. That correspondence is the point and it is asserted in the
tests: a run card claiming a two-qubit depth the circuit does not contain is a
number nobody can check, which is the failure this whole project is about.

``run_card.json`` is for whoever runs it later, including the same person in six
months. Problem, ansatz, device, layout, routing, fidelity, shot budget and the
assumptions behind each -- everything needed to say whether a result that comes back
is the one this study predicted.

``README.md`` is for a reader with no physics. It says in ordinary words what was
submitted, what the honest expectation is, and which two numbers would falsify it.

On writing QASM by hand
-----------------------

No SDK is used, for the same reason the transpiler does not use one: the file must
contain exactly what was priced, and a library that helpfully re-optimises would
break the correspondence the tests rely on. The subset emitted here is small and
entirely standard -- ``h``, ``rx``, ``rz``, ``cx``, ``swap``, ``measure`` -- and any
runtime that reads OpenQASM 3 accepts it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.figure_export import PROJECT_ROOT
from src.hardware.fidelity import Fidelity, estimate
from src.hardware.transpile import Transpiled
from src.physics.quantum.ansatz import initial_angles, split_angles

FloatArray = NDArray[np.float64]

SUBMISSION_DIRECTORY = PROJECT_ROOT / "reports" / "submissions"
"""Where packages are written.

Under ``reports`` because a submission is an output of the study, not data it reads,
and resolved from this file rather than from the working directory so that the same
command writes to the same place wherever it was run.
"""

QASM_HEADER = "OPENQASM 3.0;"
"""The version line. Version 3 because it has the ``input`` declaration this uses."""

INCLUDE_LINE = 'include "stdgates.inc";'
"""The standard gate library, which is where ``h``, ``rx``, ``rz``, ``cx`` come from."""


@dataclass(frozen=True, slots=True)
class Couplings:
    r"""The three numbers the emitted rotation angles are multiplied by.

    :class:`~src.physics.quantum.ansatz.AnsatzSpec` is deliberately a statement about
    *shape and price* -- how many gates, how deep, how wide -- and carries no coupling
    strengths at all. That is the right split for costing a circuit and the wrong one
    for writing it out, because the gate a device actually executes is
    :math:`R^z(2\gamma J)`, not :math:`R^z(2\gamma)`.

    So the strengths travel separately, and they travel *by value with a default*
    rather than being required. A required argument would break every caller that
    only wants a gate count; the default is the calibration point, and every field
    is written into the QASM header so that a file emitted from the default says so
    in its first five lines. A circuit that states its own assumptions can be
    checked; one that silently assumed :math:`J = h = 1` cannot.

    Attributes:
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis.
    """

    coupling: float = 1.0
    transverse_field: float = 1.0
    longitudinal_field: float = 0.0

    def describe(self) -> dict[str, float]:
        """Render as plain data for a run card.

        Returns:
            The three strengths, named as the rest of the project names them.
        """
        return {
            "coupling": self.coupling,
            "transverse_field": self.transverse_field,
            "longitudinal_field": self.longitudinal_field,
        }


def qasm3(
    compiled: Transpiled,
    angles: FloatArray | None = None,
    couplings: Couplings | None = None,
) -> str:
    r"""Write the compiled circuit as OpenQASM 3.

    Args:
        compiled: The circuit, already placed, routed and scheduled.
        couplings: The Hamiltonian strengths the rotation angles are scaled by. See
            :class:`Couplings`. ``None`` means the calibration point :math:`J = h =
            1`, :math:`g = 0`, which is written into the file's header like any
            other value so that a default is never a silent one.
        angles: The variational parameters, ordered as
            :func:`~src.physics.quantum.ansatz.split_angles` expects. ``None`` emits
            ``input`` declarations instead of numbers, which is what a variational
            submission usually wants: the runtime binds a fresh set of angles on
            every iteration of the optimiser, and re-emitting the circuit each time
            would be the same file with different literals in it.

    Returns:
        The program, ending in a newline.

    Raises:
        ValueError: If the angle array does not match the ansatz's parameter count.
    """
    ansatz = compiled.ansatz
    strengths = Couplings() if couplings is None else couplings
    if angles is not None and angles.size != ansatz.n_parameters:
        raise ValueError(
            f"the ansatz takes {ansatz.n_parameters} angles, got {angles.size}; "
            "a circuit written with the wrong number of parameters is not the circuit "
            "that was priced"
        )
    lines = [QASM_HEADER, INCLUDE_LINE, ""]
    lines.extend(_qasm_comments(compiled, strengths))
    lines.append("")
    if angles is None:
        lines.extend(_input_declarations(ansatz.depth))
        lines.append("")
    lines.append(f"qubit[{compiled.device.n_qubits}] q;")
    lines.append(f"bit[{ansatz.n_qubits}] c;")
    lines.append("")
    lines.append("// the equal superposition every layer starts from")
    for site in range(ansatz.n_qubits):
        lines.append(f"h q[{compiled.layout.physical(site)}];")
    for layer in range(ansatz.depth):
        lines.append("")
        lines.extend(_qasm_layer(compiled, layer, angles, strengths))
    lines.append("")
    lines.append("// one measurement per spin, in site order")
    for site in range(ansatz.n_qubits):
        lines.append(f"c[{site}] = measure q[{compiled.layout.physical(site)}];")
    return "\n".join(lines) + "\n"


def _qasm_comments(compiled: Transpiled, couplings: Couplings) -> list[str]:
    """Write the header that says what this circuit is and what it was priced at.

    The Hamiltonian strengths are named here even when they are the defaults. A
    reader handed a bare circuit cannot recover them -- the rotation angles have
    already absorbed them -- so a file that omits them is a file that can only be
    checked by someone who already knows the answer.

    Args:
        compiled: The circuit.
        couplings: The strengths the angles were scaled by.

    Returns:
        Comment lines, without a trailing blank.
    """
    ansatz = compiled.ansatz
    layout = ", ".join(f"{site}->q{qubit}" for site, qubit in enumerate(compiled.layout.sites))
    return [
        f"// transverse-field Ising {ansatz.geometry}, {ansatz.n_qubits} sites, "
        f"{ansatz.boundary} boundary",
        f"// H = -J sum sigma^z sigma^z - g sum sigma^z - h sum sigma^x, with "
        f"J={couplings.coupling:g}, g={couplings.longitudinal_field:g}, "
        f"h={couplings.transverse_field:g}",
        f"// ansatz: {ansatz.family}, depth {ansatz.depth}, {ansatz.n_parameters} parameters",
        f"// device: {compiled.device.name}, "
        f"native two-qubit gate {compiled.device.native_two_qubit}",
        f"// layout ({compiled.layout.strategy}): {layout}",
        f"// two-qubit depth {compiled.two_qubit_depth}, "
        f"{compiled.swaps} routing SWAPs, {compiled.duration_us:.2f} us",
        "// CX is written out rather than the native gate: the runtime synthesises it,",
        "// and the gate counts in the run card already include what that costs.",
    ]


def _input_declarations(depth: int) -> list[str]:
    r"""Declare the variational angles as runtime inputs.

    Args:
        depth: Number of ansatz layers.

    Returns:
        One declaration per parameter -- a :math:`\gamma` and a :math:`\beta` per
        layer.
    """
    lines = ["// bound by the optimiser on every iteration"]
    for layer in range(depth):
        lines.append(f"input float[64] gamma_{layer};")
        lines.append(f"input float[64] beta_{layer};")
    return lines


def _qasm_layer(
    compiled: Transpiled,
    layer: int,
    angles: FloatArray | None,
    couplings: Couplings,
) -> list[str]:
    r"""Write one ansatz layer.

    The layer is the coupling term followed by the field term, which is the order the
    ansatz defines. Every bond that needed routing is written with its SWAPs in
    front and the same SWAPs behind, exactly as the cost model charged for them --
    the reversal is what puts each spin back on the qubit the next layer expects.

    Every rotation carries its strength. The bond term generates
    :math:`e^{-i\gamma J \hat\sigma^z\hat\sigma^z}` and so compiles to
    :math:`R^z(2\gamma J)`; the field term gives :math:`R^x(2\beta h)`; the
    longitudinal term gives :math:`R^z(2\gamma g)` rather than another
    :math:`2\gamma`, which would silently set :math:`g = J`. Written into the angle
    rather than left to the reader because a QASM file has nowhere else to put it.

    Args:
        compiled: The circuit.
        layer: Which layer this is, counting from zero.
        angles: The parameters, or ``None`` to use the declared inputs.
        couplings: The strengths each rotation is scaled by.

    Returns:
        The lines for this layer.
    """
    ansatz = compiled.ansatz
    if angles is None:
        gamma_term, beta_term = f"gamma_{layer}", f"beta_{layer}"
    else:
        gamma, beta = split_angles(angles)
        gamma_term, beta_term = f"{gamma[layer]:.12g}", f"{beta[layer]:.12g}"
    bond_angle = _scaled(gamma_term, 2.0 * couplings.coupling)
    field_angle = _scaled(beta_term, 2.0 * couplings.transverse_field)
    tilt_angle = _scaled(gamma_term, 2.0 * couplings.longitudinal_field)
    lines = [f"// layer {layer + 1} of {ansatz.depth}: coupling, then field"]
    for bond in compiled.bonds:
        # Consecutive pairs along the route, minus the last one: the final pair is
        # where the interaction happens rather than another step towards it.
        forward = [(bond.path[step], bond.path[step + 1]) for step in range(len(bond.path) - 2)]
        for left, right in forward:
            lines.append(f"swap q[{left}], q[{right}];")
        near, far = bond.path[-2], bond.path[-1]
        lines.append(f"cx q[{near}], q[{far}];")
        lines.append(f"rz({bond_angle}) q[{far}];")
        lines.append(f"cx q[{near}], q[{far}];")
        for left, right in reversed(forward):
            lines.append(f"swap q[{left}], q[{right}];")
    for site in range(ansatz.n_qubits):
        lines.append(f"rx({field_angle}) q[{compiled.layout.physical(site)}];")
    if ansatz.longitudinal:
        lines.append("// longitudinal field: one Rz per site, no two-qubit cost")
        for site in range(ansatz.n_qubits):
            lines.append(f"rz({tilt_angle}) q[{compiled.layout.physical(site)}];")
    return lines


def _scaled(term: str, factor: float) -> str:
    """Multiply an angle -- a literal or a runtime input name -- by a constant.

    Args:
        term: The angle, either a number already formatted or an ``input`` name.
        factor: What to multiply it by.

    Returns:
        A QASM expression. A numeric term is folded to a single literal so the file
        stays readable; a named input keeps the multiplication, because the runtime
        supplies the name and the arithmetic has to happen there.

    Examples:
        >>> _scaled("gamma_0", 1.4)
        '1.4*gamma_0'
        >>> _scaled("0.5", 1.4)
        '0.7'
    """
    try:
        return f"{float(term) * factor:.12g}"
    except ValueError:
        return f"{factor:g}*{term}"


@dataclass(frozen=True, slots=True)
class RunCard:
    """Everything needed to run this circuit later and know what to expect.

    A record rather than a loose dictionary, so that a field cannot be added by one
    caller and missed by another, and so the type checker catches a package built
    from a circuit and a fidelity estimate that came from different runs.

    Attributes:
        compiled: The circuit as it was priced.
        fidelity: What is expected to survive.
        shots: The measurement budget this circuit is to be run with.
        note: Why this configuration was chosen, in one plain sentence. It is the
            field a reader looks at first and the one that is easiest to leave
            empty, so it is required.
        couplings: The Hamiltonian strengths the circuit's angles were scaled by.
            On the card because the circuit alone does not determine them and a
            reader comparing a returned energy against a prediction needs the
            problem, not only the shape of the circuit that attacked it.
    """

    compiled: Transpiled
    fidelity: Fidelity
    shots: int
    note: str
    couplings: Couplings = field(default_factory=Couplings)

    def as_data(self) -> dict[str, Any]:
        """Render the whole card as JSON-safe primitives.

        Returns:
            A nested mapping: the problem, the machine, the compiled circuit, the
            noise estimate, and the shot arithmetic that follows from it.
        """
        ansatz = self.compiled.ansatz
        inflation = self.fidelity.shot_inflation()
        return {
            "note": self.note,
            "problem": {
                "model": f"transverse-field Ising {ansatz.geometry}",
                "geometry": ansatz.geometry,
                "n_sites": ansatz.n_qubits,
                "boundary": ansatz.boundary,
                "has_longitudinal_field": ansatz.longitudinal,
                **self.couplings.describe(),
            },
            "ansatz": ansatz.describe(),
            "device": self.compiled.device.describe(),
            "layout": self.compiled.layout.describe(),
            "compiled": self.compiled.describe(),
            "fidelity": self.fidelity.describe(),
            "shots": {
                "requested": self.shots,
                "noise_inflation": None if inflation == float("inf") else round(inflation, 2),
                "equivalent_noiseless_shots": (
                    None if inflation == float("inf") else int(self.shots / inflation)
                ),
            },
        }


def run_card(
    compiled: Transpiled,
    shots: int,
    note: str,
    couplings: Couplings | None = None,
) -> RunCard:
    """Assemble a run card, computing the fidelity from the circuit itself.

    Args:
        compiled: The circuit, already placed, routed and scheduled.
        shots: The measurement budget.
        note: Why this configuration, in one sentence.
        couplings: The Hamiltonian strengths, or ``None`` for the calibration
            point. See :class:`Couplings`.

    Returns:
        The card.

    Raises:
        ValueError: If the shot budget is not positive, or the note is empty. Both
            are omissions rather than choices, and a package is the wrong place to
            discover one.
    """
    if shots <= 0:
        raise ValueError(f"a run needs at least one shot, got {shots}")
    if not note.strip():
        raise ValueError("a run card needs a note saying why this configuration was chosen")
    return RunCard(
        compiled=compiled,
        fidelity=estimate(compiled),
        shots=shots,
        note=note.strip(),
        couplings=Couplings() if couplings is None else couplings,
    )


def readme(card: RunCard) -> str:
    """Write the plain-language note that goes beside the circuit.

    Written for somebody who does not work on quantum computers and should not have
    to in order to judge whether this was a reasonable thing to submit. It states
    what the circuit is, what is expected of it, and the two numbers that would show
    the expectation was wrong.

    Args:
        card: The run card.

    Returns:
        The note as Markdown.
    """
    compiled = card.compiled
    ansatz = compiled.ansatz
    inflation = card.fidelity.shot_inflation()
    inflation_text = "unbounded" if inflation == float("inf") else f"{inflation:.1f} times"
    routing_text = (
        "no overhead at all"
        if compiled.swaps == 0
        else f"a factor of {compiled.routing_overhead:.1f} on the depth"
    )
    lines = [
        f"# {ansatz.n_qubits}-site {ansatz.geometry}, depth {ansatz.depth}, "
        f"on `{compiled.device.name}`",
        "",
        card.note,
        "",
        "## What is in here",
        "",
        "- `circuit.qasm` -- the circuit itself, written against the physical qubits",
        "  listed in the run card. Every gate the cost estimate charged for is in it.",
        "- `run_card.json` -- the numbers below, in full, plus the assumptions.",
        "",
        "## What was submitted",
        "",
        f"A {ansatz.geometry} of {ansatz.n_qubits} magnets with a {ansatz.boundary} boundary, "
        f"prepared by {ansatz.depth} repeated layer{'s' if ansatz.depth != 1 else ''} of the "
        f"same two operations, tuned by {ansatz.n_parameters} numbers the optimiser adjusts.",
        "",
        f"Each neighbouring pair pulls on the other with strength J = "
        f"{card.couplings.coupling:g}, and every magnet is tilted by a sideways field "
        f"h = {card.couplings.transverse_field:g}"
        + (
            f" and a field g = {card.couplings.longitudinal_field:g} along the coupling "
            "direction, which is what removes the shortcut an exact formula would offer"
            if card.couplings.longitudinal_field
            else ", with nothing along the coupling direction"
        )
        + ". Those three numbers are baked into the rotation angles in `circuit.qasm`, so "
        "they are stated in its header too.",
        "",
        f"The machine has {compiled.device.n_qubits} qubits wired in a "
        f"{compiled.device.topology.replace('_', '-')} pattern. Where the problem asked two "
        f"spins to interact that the wiring does not connect, the compiler moved them together "
        f"and back again: {compiled.swaps} such moves, which is {routing_text}.",
        "",
        "## What to expect, and what would falsify it",
        "",
        "| Quantity | Value | What it means |",
        "| --- | --- | --- |",
        f"| Two-qubit depth | {compiled.two_qubit_depth} | Sequential entangling steps. "
        "The thing coherence is spent on. |",
        f"| Duration | {compiled.duration_us:.2f} us | Against a coherence time of "
        f"{compiled.device.t2_ns / 1000:.0f} us. |",
        f"| Surviving signal | {card.fidelity.total:.1%} | The rest is noise. Largest loss: "
        f"{card.fidelity.dominant_loss}. |",
        f"| Shot cost of that | {inflation_text} | More measurements than a perfect machine "
        "would need, for the same precision. |",
        f"| Shots requested | {card.shots:,} | |",
        "",
        "Two numbers falsify this. If the measured energy sits further above the reference "
        "than the surviving-signal figure predicts, the noise model is optimistic. If the "
        "circuit takes materially longer than the duration above, the schedule is wrong and "
        "every fidelity figure here is too generous.",
        "",
        "The energy this returns is an **upper bound** on the true ground-state energy, and "
        "noise can only push it further up. A result that comes back *below* the reference is "
        "not a good result -- it is a bug.",
    ]
    return "\n".join(lines) + "\n"


def submission_package(
    card: RunCard,
    name: str,
    angles: FloatArray | None = None,
    directory: Path | None = None,
) -> Path:
    """Write the circuit, the run card and the note into one directory.

    Args:
        card: The run card.
        name: Directory name for this submission. Slashes and spaces are replaced,
            because the name is often built from a configuration label.
        angles: The variational parameters to bake into the circuit. ``None`` writes
            the circuit with ``input`` declarations and, separately, the ansatz's own
            starting angles into the run card, so that a reader can reproduce the
            first iteration exactly.
        directory: Where to write. Defaults to :data:`SUBMISSION_DIRECTORY`. A
            parameter rather than a constant read at call time so that a caller
            writing somewhere else does not have to reach into this module and
            change it, which two callers doing at once would silently interleave.

    Returns:
        The directory that was written.

    Raises:
        OSError: If the directory cannot be created or written.
    """
    folder = (directory or SUBMISSION_DIRECTORY) / _safe_name(name)
    folder.mkdir(parents=True, exist_ok=True)
    ansatz = card.compiled.ansatz
    starting = angles if angles is not None else initial_angles(ansatz.depth)
    payload = card.as_data()
    payload["angles"] = {
        "baked_into_circuit": angles is not None,
        "values": [round(float(value), 9) for value in starting],
    }
    (folder / "circuit.qasm").write_text(qasm3(card.compiled, angles), encoding="utf-8")
    (folder / "run_card.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (folder / "README.md").write_text(readme(card), encoding="utf-8")
    return folder


def _safe_name(name: str) -> str:
    """Reduce a label to something usable as a directory name.

    Args:
        name: The label, which may have come from a configuration string.

    Returns:
        The label with anything but letters, digits, dashes and underscores replaced
        by a dash.

    Raises:
        ValueError: If nothing usable is left, since writing to a directory called
            ``-`` would silently collide with the next such label.
    """
    cleaned = "".join(character if character.isalnum() else "-" for character in name.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    if not cleaned:
        raise ValueError(f"{name!r} leaves no usable directory name")
    return cleaned
