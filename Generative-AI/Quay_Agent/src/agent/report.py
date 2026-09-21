"""Writing the campaign up for someone who does not work in physics.

The reader can read a table, check arithmetic, and tell when a claim has no
evidence under it. What they cannot do is supply the meaning of "ansatz", "shot"
or "variational bound" from memory, and a report that assumes they can is one
that cannot be checked.

Three rules hold throughout. Every domain word is glossed the first time it
appears. Every number that matters appears in a table beside the quantity it is
compared against, never alone in a sentence. And the verdict is stated first, so
a reader who stops after the first paragraph has the answer rather than the
methodology.

The report is composed here rather than written by a language model, because it
is a table of numbers the campaign already holds and generating those from a
prompt would risk the one error this project is built to avoid: a figure in the
summary that does not match the run record. A model may rephrase the verdict
summary; it never produces a number.

Mathematics is written in LaTeX between dollar delimiters, which is what the
project's rendering path expects -- single inline, double for a line of its own.
No other delimiter renders.
"""

from __future__ import annotations

from src.agent.state import CampaignState, snapshot
from src.physics.model import hamiltonian_display

GLOSSARY: tuple[tuple[str, str], ...] = (
    (
        "spin chain",
        "a row of magnetic elements, each of which can point one of two ways and each "
        "pulled on by its immediate neighbours",
    ),
    (
        "ground state",
        "the arrangement with the lowest possible energy -- the configuration the "
        "system settles into when left alone, and the thing being solved for",
    ),
    (
        "ansatz",
        "the shape of the trial answer, before its numbers are tuned. Choosing one is "
        "like deciding to fit a straight line rather than a curve: it fixes what "
        "answers are reachable at all",
    ),
    (
        "layer, or depth",
        "one repeat of the circuit's basic pattern. More layers can reach a better "
        "answer and take proportionally longer to run, which is the trade this whole "
        "assessment turns on",
    ),
    (
        "shot",
        "one run of the circuit followed by one measurement. A quantum computer gives "
        "a different reading each time, so an energy is an average over many shots, "
        "and shots are what the work is actually billed in",
    ),
    (
        "upper bound",
        "a number guaranteed to be at or above the true answer, never below. Every "
        "method used here reports one, which is why a lower number is a better one",
    ),
    (
        "coherence time",
        "how long the hardware holds its quantum state before it decays into noise. A "
        "circuit that runs longer than this returns an answer to a different question "
        "than the one asked",
    ),
)
"""Every domain term the report uses, with a plain-English gloss.

Written out as data rather than sprinkled through the prose so that the set is
finite and checkable: a test can assert that no term appears in the report without
appearing here, which is a property that comment-level discipline never sustains.
"""


def compose(state: CampaignState, decided_by: str) -> str:
    """Write the feasibility report as Markdown.

    Args:
        state: The finished campaign.
        decided_by: The name of the screen that reached the verdict, so the report
            can state which rule decided rather than leaving the reader to infer it.

    Returns:
        The report. Self-contained: a reader needs nothing but this string.
    """
    facts = snapshot(state)
    sections = [
        _verdict_section(state, decided_by),
        _problem_section(state),
        _quantum_section(state),
        _classical_section(state),
        _rejected_section(state),
        _crossover_section(state),
        _provenance_section(state, facts["shots_spent"]),
        _glossary_section(),
    ]
    return "\n\n".join(section for section in sections if section)


def _verdict_section(state: CampaignState, decided_by: str) -> str:
    """The answer, first, before any evidence.

    Args:
        state: The finished campaign.
        decided_by: Which screen decided.

    Returns:
        The opening section.
    """
    verdict = state["verdict"]
    if verdict is None:
        return (
            "# Feasibility assessment\n\nNo verdict. A feasibility call is a statement "
            "about one specific problem, and this question named none -- so what "
            "follows is what the runs measured, on the chain stated in the assumptions."
        )

    headline = {
        "go": "Yes -- worth using a quantum computer here",
        "no": "No -- a quantum computer is not worth using here",
        "conditional": "Not yet -- promising, but the case is not made",
    }[verdict.call]

    return (
        "# Feasibility assessment\n\n"
        f"## {headline}\n\n"
        f"{verdict.summary}\n\n"
        f"*Confidence: {verdict.confidence}. Decided by the `{decided_by}` rule.*"
    )


def _problem_section(state: CampaignState) -> str:
    """What the question was taken to mean, and what was assumed to get there.

    Args:
        state: The finished campaign.

    Returns:
        The section, or an empty string if nothing was formalised.
    """
    model = state["model"]
    if model is None:
        return ""

    lattice = model.lattice
    longitudinal = ""
    if model.longitudinal_field != 0.0:
        longitudinal = (
            rf", and $g = {model.longitudinal_field:g}$ a further field pulling it "
            r"along the same direction the neighbours do"
        )

    # The neighbour sum is described by what it runs over, and that differs by
    # shape: on a line it is each site and the next one, which is worth writing as
    # a count of spins; on a lattice it is a bond list, and the count that tells a
    # reader how much harder the problem is is the number of neighbours per site.
    if model.geometry == "chain":
        couplings = (
            rf"with {model.n_sites} spins, $J = {model.coupling:g}$ setting how "
            rf"strongly neighbours pull on each other, and "
            rf"$h = {model.transverse_field:g}$ the field pulling each spin sideways"
            f"{longitudinal}."
        )
    else:
        couplings = (
            rf"where $\langle ij \rangle$ runs over the {lattice.n_bonds} "
            rf"neighbouring pairs, $J = {model.coupling:g}$ sets how strongly "
            rf"neighbours pull on each other, and $h = {model.transverse_field:g}$ is "
            rf"the field pulling each spin sideways"
            f"{longitudinal}. Each spin away from the edge has "
            f"{lattice.neighbours_per_site()} neighbours rather than the two it would "
            "have on a line, which is most of why this shape costs more to solve."
        )

    lines = [
        "## The problem, as it was understood",
        "",
        f"The request was read as a {lattice.in_words()}, each interacting with its "
        "neighbours and pushed sideways by a magnetic field. Written out, the energy "
        "of the system is",
        "",
        hamiltonian_display(model.longitudinal_field, model.geometry),
        "",
        couplings,
    ]

    if model.assumptions:
        lines += [
            "",
            "**Assumptions made in getting here.** These were chosen, not given, and "
            "the verdict depends on them:",
            "",
        ]
        lines += [f"- {assumption}" for assumption in model.assumptions]

    if model.is_exactly_solvable:
        lines += [
            "",
            "**This problem has a closed-form solution.** With no field along the "
            "coupling direction, the chain reduces to a set of independent particles, "
            "and its lowest energy is a short sum that an ordinary computer evaluates "
            "instantly at any size. That fact alone settles the feasibility question, "
            "whatever the circuits below achieved.",
        ]

    return "\n".join(lines)


def _quantum_section(state: CampaignState) -> str:
    """Every quantum configuration tried, including the ones that went badly.

    Args:
        state: The finished campaign.

    Returns:
        The section, or a statement that nothing ran.
    """
    runs = state["runs"]
    if not runs:
        return "## What the quantum circuits achieved\n\nNo quantum configuration ran."

    lines = [
        "## What the quantum circuits achieved",
        "",
        "Every configuration tried is listed, including those that went badly. A list "
        "of only the successes cannot be told apart from a list of one lucky attempt.",
        "",
        "| Configuration | Layers | Two-qubit depth | Energy per spin | Shots | How the run went |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        lines.append(
            f"| {run.label} | {run.depth} | {run.two_qubit_depth} | "
            f"{run.energy_per_site:.6f} | {run.shots_spent:,} | "
            f"{run.diagnosis.signal.replace('_', ' ')} |"
        )

    lines += [
        "",
        "Energies are per spin so that chains of different lengths can be compared, "
        "and every one is an upper bound: the true answer is at or below each figure.",
    ]

    unhealthy = [run for run in runs if run.diagnosis.signal != "healthy"]
    if unhealthy:
        lines += ["", "**What went wrong, where it did:**", ""]
        lines += [
            f"- *{run.label}* -- {run.diagnosis.evidence}. {run.diagnosis.repair}."
            for run in unhealthy
        ]

    return "\n".join(lines)


def _classical_section(state: CampaignState) -> str:
    """The classical number, and the comparison against it.

    Args:
        state: The finished campaign.

    Returns:
        The section.
    """
    classical = state["classical"]
    if classical is None:
        return (
            "## The classical comparison\n\n"
            "**No classical result was produced.** Without one there is nothing to "
            "compare against, so no claim about a quantum computer being worthwhile "
            "can be supported by this campaign."
        )

    lines = [
        "## The classical comparison",
        "",
        "This is the number a quantum result has to beat. It comes from running "
        f"the {classical.method.replace('_', ' ')} method on the same problem -- "
        "the same layered structure the quantum circuit uses, run on an ordinary "
        "computer, which is what makes the comparison fair rather than a straw man.",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Energy per spin | {classical.energy_per_site:.6f} |",
        f"| Statistical uncertainty | ± {classical.energy_error:.6f} |",
        f"| Measurements taken | {classical.n_measurements:,} |",
        f"| How far it can be trusted | {classical.confidence} |",
        "",
        "The uncertainty is not decoration. Two energies differing by less than it "
        "are the same energy, so a quantum result only counts as better if it wins "
        "by more than that margin.",
    ]
    if not classical.is_trustworthy:
        # Its own headed subsection rather than a footnote on the table. The
        # failure this guards against is a reader taking the number at face
        # value, and a caveat set in small print beside a six-decimal figure is
        # read as a formality. The row above says "low"; this says what that cost.
        lines += [
            "",
            "### What this comparison could not check",
            "",
            "**The uncertainty in the table above is smaller than the real one.** "
            + (classical.caveat or "The method ran outside the regime it was calibrated on.")
            + " The consequence is specific rather than general: a quantum lead is "
            "measured in units of that uncertainty, so understating it makes a lead "
            "easier to claim than it should be. Any comparison below is reported for "
            "completeness and is not evidence either way.",
        ]
    return "\n".join(lines)


def _rejected_section(state: CampaignState) -> str:
    """Configurations that were priced and turned down before running.

    Args:
        state: The finished campaign.

    Returns:
        The section, or an empty string if nothing was rejected.
    """
    rejected = state["ruled_out"]
    if not rejected:
        return ""

    lines = [
        "## What was considered and rejected",
        "",
        "These were costed and turned down before any work was done on them. Rejecting "
        "a plan on arithmetic is free; discovering the same thing after spending the "
        "budget is not.",
        "",
    ]
    lines += [f"- **{item.label}** -- {item.reason}" for item in rejected]
    return "\n".join(lines)


def _crossover_section(state: CampaignState) -> str:
    """What would have to change for the answer to be different.

    Args:
        state: The finished campaign.

    Returns:
        The section, or an empty string if there is no verdict.
    """
    verdict = state["verdict"]
    if verdict is None:
        return ""
    return "\n".join(
        [
            "## What would change this answer",
            "",
            verdict.crossover_condition.capitalize() + ".",
            "",
            "This is stated so the assessment can be re-checked later without being "
            "re-read. Hardware improves; a verdict with no stated condition quietly "
            "expires without anyone noticing.",
        ]
    )


def _worst_case_row(state: CampaignState) -> list[str]:
    """State how loose the measurement price the runs were approved on was.

    A budget has to be priced before a state exists, so it charges every term the
    largest variance its outcomes allow. Once a run has finished, the state is there
    to be asked, and the honest thing is to report both figures rather than the one
    the decision was made on.

    Args:
        state: The finished campaign.

    Returns:
        One table row, or nothing when no run measured it.
    """
    runs = [run for run in state["runs"] if run.shots_at_true_variance > 0]
    if not runs:
        return []
    priced = sum(run.shots_spent for run in runs)
    measured = sum(run.shots_at_true_variance for run in runs)
    return [
        f"| Measurements the finished runs really needed | {measured:,}, "
        f"{priced / measured:.0f} times below the worst case they were priced at |"
    ]


def _provenance_section(state: CampaignState, shots_spent: int) -> str:
    """What the campaign spent, what it read, and what it could not see.

    Args:
        state: The finished campaign.
        shots_spent: Measurements consumed.

    Returns:
        The section.
    """
    ledger = state["shots"]
    coherence = state["coherence"]
    lines = [
        "## How this was produced",
        "",
        "| | |",
        "| --- | --- |",
        f"| Measurements spent | {shots_spent:,} of {ledger.budget:,} budgeted |",
        f"| Deepest circuit the hardware allows | two-qubit depth "
        f"{coherence.max_two_qubit_depth} |",
        f"| Configurations run | {len(state['runs'])} |",
        f"| Configurations rejected before running | {len(state['ruled_out'])} |",
        *_worst_case_row(state),
        "",
        "**What this assessment cannot see.** The quantum results come from an exact "
        "simulation of the circuit, not from hardware, so they show what the circuit "
        "would achieve on a perfect device. Real hardware adds errors that these "
        "numbers do not include, and those errors can only make the quantum side "
        "worse -- never better. Read every quantum figure here as the best case.",
    ]

    if state["citations"]:
        lines += ["", "**Sources consulted:**", ""]
        lines += [
            f"- {citation.title} (`{citation.identifier}`)" for citation in state["citations"]
        ]

    note = state["request"].screening_note
    if note:
        lines += ["", f"**Input screening:** {note}"]

    return "\n".join(lines)


def _glossary_section() -> str:
    """The domain vocabulary, defined.

    Returns:
        The closing section.
    """
    lines = ["## Terms used", ""]
    lines += [f"- **{term}** -- {gloss}" for term, gloss in GLOSSARY]
    return "\n".join(lines)
