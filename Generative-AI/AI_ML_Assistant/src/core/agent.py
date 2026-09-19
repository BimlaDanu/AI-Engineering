"""
Bounded plan->act->observe agent loop.

Handles multi-step questions that require multiple tool calls or refined retrieval
before producing a grounded answer. It uses a custom hard-capped ReAct-style loop
instead of a framework agent to stay aligned with the existing pipeline design.

Key design points:
- Uses structured :class:`~src.core.schemas.AgentAction` outputs for planning and
  validated dispatch, matching the router and grader approach.
- Reuses existing components: :meth:`KnowledgeBase.search`, chat ``@tool`` objects,
  and :func:`~src.generation.answer_from_agent` so citations, formatting, and token
  accounting remain consistent.
- Limits execution with ``settings.agent_max_steps`` and falls back to a simple
  retrieve-then-answer flow when no LLM is available.

The agent only collects evidence (retrieved passages and tool observations). A final
grounded response is generated in a single synthesis call after ``finish`` or when
the step limit is reached. Planning token usage is estimated with
:func:`~src.utils.estimate_tokens` and combined with synthesis usage in the returned
:class:`AnswerResult`, avoiding undercounting of agent overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.schemas import AgentAction
from src.generation import (
    AnswerResult,
    ToolEvent,
    _invoke_tool,
    answer_from_agent,
)
from src.utils import estimate_tokens

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.config import RagSettings
    from src.rag.retriever import KnowledgeBase, RetrievedChunk

_AGENT_SYSTEM = """You are a bounded reasoning agent for an AI/ML learning assistant. Break \
the user's question into a short sequence of steps. At each step choose EXACTLY ONE action:
- 'retrieve': search the knowledge base with a focused query. Use it to gather grounding \
passages; you may retrieve more than once with refined queries.
{tool_lines}- 'finish': stop once you have enough information to answer well.

You have at most {max_steps} steps. Be efficient: never repeat an action that already gave \
you what you need, and choose 'finish' as soon as you can answer. Do NOT write the final \
answer yourself — it is composed separately from the evidence you gather."""


@dataclass
class AgentStepRecord:
    """One executed step of the agent loop, for the trajectory shown in the Inspector."""

    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentOutcome:
    """The product of a full agent run: the answer, its context, and the trajectory."""

    result: AnswerResult
    chunks: list[RetrievedChunk] = field(default_factory=list)
    trajectory: list[AgentStepRecord] = field(default_factory=list)


def _first_line(text: str | None) -> str:
    """First sentence/line of a tool's docstring, for the compact action menu."""
    if not text:
        return "(no description)"
    return text.strip().splitlines()[0]


def _tool_menu(tools: list[Any]) -> str:
    """Render the available tools as ``- 'name': one-line description`` menu lines."""
    return "".join(f"- '{t.name}': {_first_line(t.description)}\n" for t in tools)


def _scratchpad(trajectory: list[AgentStepRecord]) -> str:
    """Render the steps taken so far so the planner can decide what to do next."""
    if not trajectory:
        return "(no steps taken yet)"
    return "\n".join(
        f"Step {i}: action={r.action}({r.action_input!r}) -> {r.observation[:400]}"
        for i, r in enumerate(trajectory, start=1)
    )


def _tool_args(tool_fn: Any, action_input: str) -> dict[str, Any]:
    """Map the agent's single string input onto the tool's primary argument.

    The built-in tools each take one main string argument (``query`` / ``expression`` /
    ``text``) with the rest optional, so binding the input to the first declared field
    covers them; other fields fall back to their defaults.
    """
    schema = getattr(tool_fn, "args", None) or {}
    if not schema:
        return {}
    first = next(iter(schema))
    return {first: action_input}


def _build_planner(llm: BaseChatModel | None) -> Any | None:
    """Return a structured-output runnable that emits :class:`AgentAction`, or None."""
    if llm is None:
        return None
    try:
        return llm.with_structured_output(AgentAction, method="json_schema")
    except Exception:
        return None


@dataclass
class _PlanStep:
    """One planning turn: the chosen action (or None) plus its estimated token usage.

    Structured-output planning does not return provider ``usage_metadata``, so the tokens
    are approximated — inputs from the prompt text, outputs from the emitted action — and
    accumulated across the loop so the reported cost reflects planning overhead.
    """

    action: AgentAction | None
    tokens_in: int = 0
    tokens_out: int = 0


def _planner_messages(
    system: str, question: str, trajectory: list[AgentStepRecord]
) -> list[tuple[str, str]]:
    """Build the (system, human) messages sent to the planner for one step."""
    human = (
        f"Question:\n{question}\n\n"
        f"Progress so far:\n{_scratchpad(trajectory)}\n\n"
        "Choose the next action."
    )
    return [("system", system), ("human", human)]


def _estimate_plan_tokens(messages: list[tuple[str, str]], action: AgentAction) -> tuple[int, int]:
    """Estimate one planning step's (input, output) tokens from its prompt and action."""
    prompt_text = "\n".join(text for _, text in messages)
    action_text = f"{action.thought} {action.action} {action.action_input}"
    return estimate_tokens(prompt_text), estimate_tokens(action_text)


def _next_action(
    planner: Any | None, system: str, question: str, trajectory: list[AgentStepRecord]
) -> _PlanStep:
    """Ask the planner for the next action; a None action ends the loop.

    Returns a :class:`_PlanStep` so the caller can accumulate the (estimated) planning-token
    usage even when the loop later ends — a None action (no LLM or a parse failure) simply
    carries zero tokens.
    """
    if planner is None:
        return _PlanStep(None)
    messages = _planner_messages(system, question, trajectory)
    try:
        action = planner.invoke(messages)
    except Exception:
        return _PlanStep(None)
    if not isinstance(action, AgentAction):
        return _PlanStep(None)
    tokens_in, tokens_out = _estimate_plan_tokens(messages, action)
    return _PlanStep(action, tokens_in, tokens_out)


def _do_retrieve(
    query: str, kb: KnowledgeBase | None, settings: RagSettings, subjects: list[str] | None
) -> tuple[str, list[RetrievedChunk]]:
    """Run one knowledge-base search and return a human-readable observation plus hits."""
    if kb is None:
        return ("Knowledge base unavailable — no passages retrieved.", [])
    hits = kb.search(query or "", subjects=subjects, k=settings.top_k, alpha=settings.hybrid_alpha)
    if not hits:
        return (f"No passages found for {query!r}.", [])
    titles = ", ".join(h.metadata.get("title", h.metadata.get("source", "?")) for h in hits)
    return (f"Retrieved {len(hits)} passage(s): {titles}.", hits)


def _execute_action(
    action: AgentAction,
    *,
    kb: KnowledgeBase | None,
    tools_by_name: dict[str, Any],
    settings: RagSettings,
    subjects: list[str] | None,
) -> tuple[str, list[RetrievedChunk], ToolEvent | None]:
    """Dispatch one action to retrieval or a tool; return (observation, chunks, tool_event)."""
    name = action.action.strip()
    if name == "retrieve":
        observation, hits = _do_retrieve(action.action_input, kb, settings, subjects)
        return (observation, hits, None)
    tool_fn = tools_by_name.get(name)
    if tool_fn is None:
        return (
            f"Unknown action {name!r}. Choose 'retrieve', a listed tool, or 'finish'.",
            [],
            None,
        )
    args = _tool_args(tool_fn, action.action_input)
    output = _invoke_tool(tool_fn, args)
    return (str(output), [], ToolEvent(name=name, args=args, output=str(output)))


def run_agent_loop(
    llm: BaseChatModel | None,
    question: str,
    *,
    kb: KnowledgeBase | None,
    tools: list[Any] | None,
    settings: RagSettings,
    level: str = "Beginner",
    style: str | None = None,
    history: list[tuple[str, str]] | None = None,
    subjects: list[str] | None = None,
) -> AgentOutcome:
    """Run the bounded plan->act->observe loop and synthesise a grounded, cited answer.

    Args:
        llm: The chat model used both for planning and the final synthesis. ``None`` yields
            a graceful no-op outcome (used only in offline paths without a model).
        question: The user's question.
        kb: The knowledge base for ``retrieve`` actions (may be ``None``).
        tools: Tools the agent may call (built-ins plus any remote MCP tools).
        settings: Supplies ``agent_max_steps`` and the retrieval knobs (``top_k``, ``alpha``).
        level, style, history, subjects: Passed through to retrieval and final generation so
            the agent answer matches the rest of the pipeline.
    """
    tools = list(tools or [])
    history = history or []
    tools_by_name = {t.name: t for t in tools}
    max_steps = max(1, int(getattr(settings, "agent_max_steps", 4)))

    chunks: list[RetrievedChunk] = []
    seen: set[tuple[Any, str]] = set()
    tool_events: list[ToolEvent] = []
    trajectory: list[AgentStepRecord] = []

    planner = _build_planner(llm)
    system = _AGENT_SYSTEM.format(tool_lines=_tool_menu(tools), max_steps=max_steps)
    plan_tokens_in = 0  # estimated planning usage, accumulated across steps
    plan_tokens_out = 0

    for _ in range(max_steps):
        step = _next_action(planner, system, question, trajectory)
        plan_tokens_in += step.tokens_in
        plan_tokens_out += step.tokens_out
        action = step.action
        if action is None:
            break
        if action.action.strip() == "finish":
            trajectory.append(
                AgentStepRecord(action.thought, "finish", "", "Enough information gathered.")
            )
            break
        observation, hits, event = _execute_action(
            action,
            kb=kb,
            tools_by_name=tools_by_name,
            settings=settings,
            subjects=subjects,
        )
        for chunk in hits:  # dedupe repeated retrievals so citations stay clean
            key = (chunk.metadata.get("source"), chunk.text[:80])
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)
        if event is not None:
            tool_events.append(event)
        trajectory.append(
            AgentStepRecord(action.thought, action.action, action.action_input, observation)
        )

    if llm is None:
        result = AnswerResult(
            text="The agent route needs a language model, which is unavailable.",
            tool_events=list(tool_events),
        )
    else:
        result = answer_from_agent(
            llm, question, chunks, tool_events, history, level=level, style=style
        )
    # Fold the estimated planning usage onto the exact synthesis usage so the reported
    # token/cost accounts for the whole agent run, not just the final generation.
    result.input_tokens += plan_tokens_in
    result.output_tokens += plan_tokens_out
    return AgentOutcome(result=result, chunks=chunks, trajectory=trajectory)
