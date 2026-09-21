"""Print the route each shape of question takes through the graph, and what it produced.

The answer to "is the LangGraph campaign actually doing anything?" -- which is a fair
question to ask of any agent framework, and one that a passing test suite does not
answer on its own. A route is easy to *claim* and easy to get wrong: a node that never
runs, a branch that always wins, a fan-out that turns out to be sequential, an
artefact computed and then dropped. All four look identical from the outside.

So this asks one question of each shape and prints, side by side, the intent that was
read, the nodes that actually ran in the order they ran, and every artefact the state
came back holding. Four things are then visible at a glance:

* **The branch is real.** A feasibility question runs ``plan``, ``solve``, ``analyse``
  and ``skeptic`` and spends measurements; an explanation runs none of them and spends
  none. If the branch were decorative, both would run the same nodes.
* **The loop is real.** ``plan → solve → analyse`` appears several times in one route,
  because the depth ladder climbs until the arithmetic refuses it -- and the refusals
  are counted.
* **The fan-out is real.** ``baseline`` appears between two rungs of the ladder rather
  than before or after them: it was issued in parallel with the planner and finished
  when it finished.
* **Nothing is claimed that was not computed.** An explanation carries no verdict, no
  runs and no shots, which is what the branch promises.

Offline, no key, no network, free. Every decision printed here is deterministic -- the
routing was never a language model's job -- so this is a check and not a demonstration.

Run with ``make routes``.
"""

from __future__ import annotations

from src.agent.graph import pipeline_nodes, run_campaign
from src.agent.state import CampaignState
from src.hardware.devices import device_for
from src.logging_setup import configure_logging
from src.physics.registry import field_sweep_bench

SHOT_BUDGET = 10**9
"""The budget the application itself ships with, so a refusal here is one a user sees."""

QUESTIONS: tuple[str, ...] = (
    # Names a chain and asks to watch three methods on it: the full campaign, plus
    # the race. Every node in the graph that can run, runs.
    "Loss/learning curve for VQE, QAOA and VarQITE: 8 spins at criticality?",
    # The same subject with no chain in the sentence. It races and answers in prose,
    # and must not invent a chain to issue a verdict about.
    "Of VQE, QAOA and VarQITE, which converges fastest and gets closest?",
    # A curve in the *field* rather than in an optimiser's epoch: answered by solving
    # the model exactly at every point, through a tool the grader lends.
    "Can you plot the exact energy levels as the field is turned up?",
    # A request for a file.
    "Can you write the QAOA circuit for 8 spins at depth 3?",
    # A request to be told something, with no chain and no curve.
    "What is a barren plateau?",
)
"""One question per shape the graph branches on, in the order this script prints them."""


def describe(question: str) -> None:
    """Run one campaign offline and print its route beside its artefacts.

    Args:
        question: The question to ask.
    """
    visited: list[str] = []
    state: CampaignState = run_campaign(
        question,
        shot_budget=SHOT_BUDGET,
        device=device_for("heavy-hex-27"),
        chat_model=None,
        search_corpus=False,
        fetch_external=False,
        suggest_followups=False,
        reference_bench=field_sweep_bench(),
        visited=visited,
    )
    race = state["race"]
    verdict = state["verdict"]
    classical = state["classical"]
    answer = state["answer"]
    print(f"\n{question}")
    print(f"  intent    {state['intent'].intent} ({state['intent'].decided_by})")
    # Deduplicated for the shape of the route, with the step count beside it so a
    # loop is visible as a loop rather than hidden by the deduplication.
    print(f"  route     {' -> '.join(dict.fromkeys(visited))}")
    print(f"  steps     {len(visited)} node runs, {len(set(visited))} distinct")
    print(
        f"  race      {[(run.method, round(run.energy, 4)) for run in race.runs] if race else '-'}"
    )
    print(f"  ladder    {[(run.label, round(run.energy, 4)) for run in state['runs']] or '-'}")
    print(f"  refused   {len(state['ruled_out'])} configuration(s) on arithmetic")
    print(f"  baseline  {round(classical.energy_per_site, 4) if classical else '-'} per site")
    print(f"  verdict   {(verdict.call, verdict.confidence) if verdict else '-'}")
    print(f"  shots     {state['shots'].spent:,}")
    print(f"  tools     {[call.name for call in state['tool_calls']] or '-'}")
    print(f"  prose     {'written' if answer and answer.written else '-'}")
    print(f"  document  {len(state['report'] or '')} characters")


def main() -> None:
    """Print the node list, then one block per question."""
    configure_logging(level="WARNING")
    nodes = pipeline_nodes()
    print(f"{len(nodes)} nodes compiled: {', '.join(nodes)}")
    for question in QUESTIONS:
        describe(question)


if __name__ == "__main__":
    main()
