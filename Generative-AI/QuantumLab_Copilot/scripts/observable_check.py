"""Does the *model* pick the observable, or does the code?

The sweep tool now takes an ``observable`` -- ground-state properties, the
excitation spectrum, the energy's field derivatives, or both -- and the whole point
of putting it in the schema rather than in a branch is that the choice is the
agent's. That claim is not testable offline: the unit tests can only prove that a
chosen observable is honoured, never that a live model chooses it correctly from a
sentence.

So this asks. Six questions, each with an unambiguous right answer, and it prints
what the model actually requested. Several are near-misses on purpose: "how does the
magnetisation turn on" must **not** come back as a spectrum, "how far is the first
excited state" must not come back as a ground-state curve, and a question about the
first and second derivatives of the energy must come back as ``derivatives`` rather
than as the magnetisation -- which is the first derivative under another name and
carries nothing about the second. That last miss is the one this file was extended
for; it reached the user as a plot of the wrong quantity.

One model call per question, no full agent run:

    uv run python -m scripts.observable_check
"""

from __future__ import annotations

from src.logging_setup import configure_logging, get_logger
from src.physics.model import TFIMSpec
from src.settings import get_settings
from src.tools.calling import Toolbox, consult

LOG = get_logger("observable_check")

QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Plot the low-lying energy spectrum.", "spectrum"),
    ("How far above the ground state is the first excited state as h grows?", "spectrum"),
    ("Plot how the magnetisation turns on as the transverse field grows.", "ground_state"),
    ("How does the ground-state energy density vary with the field?", "ground_state"),
    (
        "Ground state energy and its first and second derivatives of the transverse "
        "field Ising model as a function of the magnetic field h/J for L = 6 using the "
        "free fermion approach.",
        "derivatives",
    ),
    ("Where does the curvature of the energy density dip, and how deep?", "derivatives"),
)
"""Question and the observable a correct call would carry."""

MATERIAL = """QUESTION: {question}

CHAIN SOLVED: a closed ring of 6 spins with J = 1 and h = 1.
COMPUTED AND CROSS-CHECKED:
  ground-state energy E0 = -7.727406610509 (pfeuty_exact and exact_diagonalisation agree)

NO PASSAGES WERE RETRIEVED. Do not cite the literature."""


def main() -> None:
    """Ask each question and report the observable the model asked for."""
    configure_logging(secrets=[get_settings().openrouter_api_key.get_secret_value()])
    box = Toolbox(spec=TFIMSpec(n_sites=6, coupling=1.0, field=1.0))
    for question, expected in QUESTIONS:
        runs = consult(MATERIAL.format(question=question), box)
        sweeps = [run for run in runs if run.tool == "SweepField"]
        if not sweeps:
            asked = "none: " + (", ".join(run.tool for run in runs) or "no tool at all")
        else:
            asked = str(sweeps[0].arguments.get("observable", "unset, so the default"))
        LOG.warning(
            "observable_choice",
            extra={
                "question": question,
                "expected": expected,
                "asked": asked,
                "verdict": "ok" if asked == expected else "MISS",
            },
        )


if __name__ == "__main__":
    main()
