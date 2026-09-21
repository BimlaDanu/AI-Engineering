"""What this session actually did, reduced to the shapes a chart needs.

Two rules hold this module together, and both are the reason it is a module rather
than a handful of comprehensions inside a page.

Nothing here is accumulated in a counter. Every figure is derived from the
campaigns themselves, so a total on the Analytics page cannot drift out of step
with the answers on the Chat page. A running tally kept beside the thing it counts
is a tally that will eventually disagree with it, and the disagreement is invisible
until somebody checks -- at which point neither number can be trusted.

Nothing here imports Streamlit. These are pure reductions over a tuple of
records, so they can be tested with a list built by hand rather than by driving an
interface. The page that draws them holds the session state; this module holds the
arithmetic.

What is worth charting, and why
-------------------------------

A dashboard of an application's own activity is usually decoration. Three of these
are not, because they are the evidence for the one claim about this system that a
screenshot of a good answer cannot support:

Which nodes ran. The pipeline is a graph that branches, and the honest way to
show that is unequal bars: ``baseline`` and ``solve`` fire only for a feasibility
question, ``explain`` only for a question about the subject, ``implement`` only
when code was asked for. A fixed pipeline would draw these all the same height.

Which branch each question took, and who decided. A route chosen by a model is
a different claim from a route chosen by a keyword, and both happen here.

Where the wall-clock went, per task. The tier system exists to keep the calls a
person waits on away from the expensive model. That is a design claim, and this is
the measurement that supports or refutes it -- if the fast tier is not fast, the
tiering bought nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.agent.usage import Usage

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from collections.abc import Sequence

    from src.agent.state import CampaignState

BRANCH_NODES: dict[str, str] = {
    "explain": "explain",
    "implement": "implement",
    "baseline": "feasibility",
    "solve": "feasibility",
    "plan": "feasibility",
}
"""Which branch a node's presence in the path implies.

Read off the path rather than off the intent, deliberately. The intent is what the
campaign *decided* to do and the path is what it *did*, and the two come apart on
exactly the runs worth looking at -- a feasibility question that was screened out,
or one whose formalisation failed and fell through to the conclusion.
"""


@dataclass(frozen=True, slots=True)
class Asked:
    """One question, and everything about how it was answered.

    Attributes:
        question: What was typed, as typed.
        framing: The voice it was asked in.
        branch: Which arm of the graph served it -- ``"feasibility"``,
            ``"explain"``, ``"implement"`` or ``"none"`` for a run that reached no
            arm at all. Derived from the path, so it cannot claim work that did not
            happen.
        decided_by: How the branch was chosen: by a model, by a keyword, or by the
            asker naming it.
        path: Every node the graph ran, in order, repeats included. Repeats are the
            interesting part -- a plan that was rejected and re-proposed appears
            twice, which is the retry loop made visible.
        seconds: Wall-clock for the whole campaign.
        usage: What it spent on model calls. Empty on a run with no model, which
            is a supported mode rather than a missing measurement.
        verdict: Go, no, conditional, or empty where the branch reached none.
        cited: Passages the answer leans on.
        written_by: ``"model"``, ``"notes"``, ``"refused"`` or empty.
        refused: Whether the campaign declined to answer rather than answering.
    """

    question: str
    framing: str = "neutral"
    branch: str = "none"
    decided_by: str = ""
    path: tuple[str, ...] = ()
    seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)
    verdict: str = ""
    cited: int = 0
    written_by: str = ""
    refused: bool = False

    @property
    def offline(self) -> bool:
        """Whether this answer was reached with no model call at all."""
        return self.usage.calls_made == 0


def branch_of(path: Sequence[str]) -> str:
    """Name the arm of the graph a run took, from the nodes it actually ran.

    Args:
        path: Node names in the order they ran.

    Returns:
        ``"feasibility"``, ``"explain"``, ``"implement"``, or ``"none"``. Where a
        run touched more than one -- which the graph permits, since the feasibility
        arm can also write prose -- the most specific wins, because "it explained
        something" is the weaker description of a run that also priced circuits.

    Examples:
        >>> branch_of(["screen", "interpret", "formalise", "baseline", "plan", "solve"])
        'feasibility'
        >>> branch_of(["screen", "interpret", "retrieve", "explain"])
        'explain'
        >>> branch_of(["screen"])
        'none'
    """
    named = {BRANCH_NODES[node] for node in path if node in BRANCH_NODES}
    for candidate in ("feasibility", "implement", "explain"):
        if candidate in named:
            return candidate
    return "none"


def record(
    question: str,
    state: CampaignState,
    path: Sequence[str],
    spent: Usage,
    seconds: float,
) -> Asked:
    """Reduce one finished campaign to the row the analytics read.

    Args:
        question: What was typed.
        state: The finished campaign.
        path: The nodes it ran, in order.
        spent: What it cost.
        seconds: Wall-clock it took.

    Returns:
        The record. Every field is read off the campaign rather than passed in
        separately, so a row cannot describe a run that did not produce it.
    """
    written = state["answer"]
    verdict = state["verdict"]
    return Asked(
        question=question,
        framing=state["request"].framing,
        branch=branch_of(path),
        decided_by=state["intent"].decided_by,
        path=tuple(path),
        seconds=round(seconds, 3),
        usage=spent,
        verdict=verdict.call if verdict is not None else "",
        cited=written.cited if written is not None else 0,
        written_by=written.written_by if written is not None else "",
        refused=written.refused if written is not None else False,
    )


def totals(asked: Sequence[Asked]) -> Usage:
    """Merge every question's spend into one.

    Args:
        asked: The session's questions.

    Returns:
        The combined usage. This is the session's only total; the page shows it in
        one place for the reason stated in this module's own docstring.
    """
    combined = Usage()
    for entry in asked:
        combined = combined.merge(entry.usage)
    return combined


def branch_counts(asked: Sequence[Asked]) -> dict[str, int]:
    """How many questions took each arm of the graph.

    Args:
        asked: The session's questions.

    Returns:
        Branch name to count, commonest first. Empty for an empty session, which
        the page draws as a prompt rather than as a chart of nothing.
    """
    counts: dict[str, int] = {}
    for entry in asked:
        counts[entry.branch] = counts.get(entry.branch, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: pair[1], reverse=True))


def node_counts(asked: Sequence[Asked], order: Sequence[str] = ()) -> dict[str, int]:
    """How many runs reached each node of the graph.

    Args:
        asked: The session's questions.
        order: The graph's own node order. Given one, nodes are reported in it and
            nodes nobody reached are dropped; the order comes from the compiled
            graph rather than from a list kept here, because a hand-kept list goes
            stale silently and the symptom is a chart in the wrong order.

    Returns:
        Node name to how many runs reached it, counted once per run however many
        times that run revisited it. Revisits are counted separately by
        :func:`retry_counts`, and mixing the two would let a single question with a
        three-attempt planning loop outweigh three questions.
    """
    counts: dict[str, int] = dict.fromkeys(order, 0)
    for entry in asked:
        for node in dict.fromkeys(entry.path):
            counts[node] = counts.get(node, 0) + 1
    return {node: count for node, count in counts.items() if count}


def retry_counts(asked: Sequence[Asked]) -> dict[str, int]:
    """How often a node ran more than once inside a single question.

    Args:
        asked: The session's questions.

    Returns:
        Node name to the number of *extra* visits, summed over questions. Empty
        when nothing looped, which is the common case and is worth being able to
        state: a planning loop that never retried is a budget that was never
        binding.
    """
    extra: dict[str, int] = {}
    for entry in asked:
        seen: dict[str, int] = {}
        for node in entry.path:
            seen[node] = seen.get(node, 0) + 1
        for node, count in seen.items():
            if count > 1:
                extra[node] = extra.get(node, 0) + count - 1
    return dict(sorted(extra.items(), key=lambda pair: pair[1], reverse=True))


@dataclass(frozen=True, slots=True)
class TaskTiming:
    """How long one kind of model call took, across the session.

    Attributes:
        task: What the call was for.
        tier: Which tier serves it.
        calls: How many were made.
        seconds: Total wall-clock inside those calls.
        tokens: Tokens in and out across them.
        cost_usd: What they cost at published rates. Unpriced models contribute
            nothing, so read this against the session's own priced/estimated
            label rather than as an invoice.
    """

    task: str
    tier: str
    calls: int
    seconds: float
    tokens: int = 0
    cost_usd: float = 0.0

    @property
    def mean_seconds(self) -> float:
        """Average wall-clock per call, or zero when there were none."""
        return self.seconds / self.calls if self.calls else 0.0


def timing_by_task(asked: Sequence[Asked]) -> tuple[TaskTiming, ...]:
    """Break the session's model calls down by what they were for.

    The measurement behind the tier system's central claim. Every call the design
    puts on the fast tier is one a person is waiting on, so if the fast tier's mean
    is not visibly below the strong tier's, the tiering is costing complexity and
    buying nothing.

    Args:
        asked: The session's questions.

    Tokens and cost are accumulated in the same pass rather than by a second
    reduction over the same records. :meth:`src.agent.usage.Usage.by_purpose`
    computes the same grouping, and two reductions over one list are two numbers
    that will eventually disagree about the same session.

    Returns:
        One entry per task that was called, slowest total first. Calls whose
        latency was never recorded contribute to the count and not to the seconds,
        which keeps a mean honest rather than dragging it towards zero.
    """
    from src.agent.model_selection import TASK_TIERS

    # Widened to plain strings deliberately: the purpose on a call record is
    # whatever the call site wrote, and a task this module has never heard of is
    # a row with no tier rather than a lookup that fails.
    tiers: dict[str, str] = {str(task): str(tier) for task, tier in TASK_TIERS.items()}

    calls: dict[str, int] = {}
    seconds: dict[str, float] = {}
    tokens: dict[str, int] = {}
    cost: dict[str, float] = {}
    for entry in asked:
        for call in entry.usage.calls:
            calls[call.purpose] = calls.get(call.purpose, 0) + 1
            tokens[call.purpose] = tokens.get(call.purpose, 0) + call.total_tokens
            cost[call.purpose] = cost.get(call.purpose, 0.0) + (call.cost_usd or 0.0)
            if call.latency_ms is not None:
                seconds[call.purpose] = seconds.get(call.purpose, 0.0) + call.latency_ms / 1000.0
    return tuple(
        sorted(
            (
                TaskTiming(
                    task=task,
                    tier=tiers.get(task, "—"),
                    calls=count,
                    seconds=round(seconds.get(task, 0.0), 3),
                    tokens=tokens.get(task, 0),
                    cost_usd=cost.get(task, 0.0),
                )
                for task, count in calls.items()
            ),
            key=lambda timing: timing.seconds,
            reverse=True,
        )
    )


def calls_by_model(asked: Sequence[Asked]) -> dict[str, int]:
    """How many calls each model served this session.

    Args:
        asked: The session's questions.

    Returns:
        Slug to call count, busiest first. The check on the tier configuration: a
        session showing one slug served every call has a deployment that pointed
        all three tiers at one model, whether or not anybody meant to.
    """
    counts: dict[str, int] = {}
    for entry in asked:
        for call in entry.usage.calls:
            counts[call.model] = counts.get(call.model, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: pair[1], reverse=True))
