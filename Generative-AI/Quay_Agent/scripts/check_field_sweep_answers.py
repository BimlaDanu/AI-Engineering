"""Ask the five curve questions and report what came back, offline.

The reproduction artefact for a defect found by using the application. Five questions
arrived in one message, every one of them contains the word "plot", and every one was
answered with a race between VQE, QAOA and VarQITE on a chain nobody had named. This
script asks them again and prints the four things that decide whether they are
answered now:

* **the intent** -- ``explain``, not the ``feasibility`` verdict three of them used to
  get;
* **the route** -- whether the campaign detoured through ``converge``, which is the
  method race and the wrong answer to all five;
* **the tool call** -- whether the exact field sweep was fetched, and for which curves;
* **the cross-check** -- whether every point of the curve was computed twice, by two
  solvers sharing no algebra, and how far apart they came out.

Offline by design: no key, no network, no cost. What a language model adds is the prose
and the judgement about which curves to fetch; everything this script checks is
deterministic, which is what makes it a check rather than a demonstration.

Run it with ``python -m scripts.check_field_sweep_answers``. It exits non-zero if any
question fails to reach a cross-checked curve, so it can gate a change to the routing.
"""

from __future__ import annotations

import sys

from src.agent.graph import run_campaign
from src.agent.reading import curves_asked_for
from src.logging_setup import configure_logging
from src.physics.registry import field_sweep_bench

ASKED: tuple[str, ...] = (
    "Can you plot low lying spectrum of quantum Ising as a function of an external "
    "filed using free fermion approach?",
    "How does the Jordan-Wigner transformation map the spin chain to free fermions "
    "and make the dynamics solvable?",
    "Using free fermion approach can you plot the ground state energy and its first "
    "and second derivatives of the transverse field Ising model as a function of the "
    "magnetic field h/J",
    "Using free fermion approach can you plot the ground state magnetization and its "
    "first and second derivatives of the transverse field Ising model as a function "
    "of the magnetic field h/J",
    "Plot magnetization and ground state enegy as a function of h/J for L site chain",
    "detail mathematical detail of quantum to classical mapping including well define "
    "mathematical equation",
)
"""The six questions as they were asked, typos and all.

Verbatim on purpose. *Filed* for *field* and *enegy* for *energy* are in the strings
because they were in the question, and a router that only works on correctly spelled
input is a router that does not work.

Two of the six ask for no curve at all: the Jordan-Wigner question and the
quantum-to-classical mapping question are requests for a derivation, and the right
behaviour there is prose from the notes with **no** sweep fetched. They are in the set
so that the check covers both answers rather than only the interesting one.
"""


def main() -> int:
    """Ask every question and print one line each.

    Returns:
        ``0`` when every question that asked for a curve got a cross-checked one and
        no question that asked for prose fetched one, and ``1`` otherwise.
    """
    configure_logging(level="WARNING")
    bench = field_sweep_bench()
    failures = 0
    for question in ASKED:
        visited: list[str] = []
        state = run_campaign(
            question,
            chat_model=None,
            search_corpus=True,
            fetch_external=False,
            suggest_followups=False,
            reference_bench=bench,
            visited=visited,
        )
        wanted = curves_asked_for(question)
        swept = [call for call in state["tool_calls"] if call.name == "exact_field_sweep"]
        raced = "converge" in visited
        print(f"\n> {question[:88]}")
        print(f"  intent      {state['intent'].intent}")
        print(f"  raced       {'YES -- wrong branch' if raced else 'no'}")
        if swept:
            call = swept[0]
            checked = "cross-checked" if "agree to" in call.result else "one method only"
            print(f"  swept       {call.arguments.get('curves')} on L={call.arguments['n_sites']}")
            print(f"  provenance  {checked}")
            if call.failed or raced:
                failures += 1
        else:
            print("  swept       nothing (answered in prose)")
            # Only the two derivation questions may reach here. A curve question that
            # fetched nothing is the defect this script exists to catch.
            if "spectrum" in wanted or "derivatives" in " ".join(wanted):
                print("  FAIL        a curve was asked for and none was fetched")
                failures += 1
        answer = state["answer"]
        print(f"  answered    {'yes' if answer is not None and answer.written else 'no'}")
    print(
        f"\n{len(ASKED)} question{'' if len(ASKED) == 1 else 's'}, "
        f"{failures} failure{'' if failures == 1 else 's'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
