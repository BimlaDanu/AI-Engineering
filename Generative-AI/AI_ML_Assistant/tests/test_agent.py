"""Offline tests for the bounded plan->act->observe agent loop (Phase 6, Option C).

The loop is built to be fully testable without a network or API key: the planner LLM, the
knowledge base, and the tools are all injectable, so these stubs drive deterministic
trajectories. They pin the behaviours that matter — the step cap is honoured, retrieval and
tool actions accumulate the right evidence, repeated retrievals are de-duplicated, unknown
actions fail safe, and a missing model degrades gracefully — plus the routing wiring that
sends the ``agent`` route to the ``agent`` step.
"""

from __future__ import annotations

from typing import ClassVar

from src.config import RagSettings
from src.core.agent import run_agent_loop
from src.core.schemas import AgentAction, RouteDecision
from src.core.steps import ROUTE_TO_STEP, after_route
from src.rag.retriever import RetrievedChunk


class _FakeReply:
    """A minimal chat reply: ``.content`` plus ``.usage_metadata`` like a real AIMessage."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": 12, "output_tokens": 8}


class _FakePlanner:
    """Yields a preset list of :class:`AgentAction` decisions, then defaults to 'finish'."""

    def __init__(self, actions: list[AgentAction]) -> None:
        self._actions = list(actions)

    def invoke(self, messages: object) -> AgentAction:
        if self._actions:
            return self._actions.pop(0)
        return AgentAction(thought="done", action="finish", action_input="")


class _FakeLLM:
    """A stub chat model: structured output drives planning; plain invoke is the synthesis."""

    def __init__(self, actions: list[AgentAction], answer: str = "Grounded answer [1].") -> None:
        self._planner = _FakePlanner(actions)
        self._answer = answer

    def with_structured_output(self, schema: object, method: str | None = None) -> _FakePlanner:
        return self._planner

    def invoke(self, messages: object) -> _FakeReply:
        return _FakeReply(self._answer)


class _FakeKB:
    """A knowledge base whose search always returns the same preset chunks."""

    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self._hits = hits

    def search(self, query: str, subjects=None, difficulty=None, k: int = 4, alpha: float = 0.7):
        return list(self._hits)


class _FakeTool:
    """A LangChain-tool-shaped stub: name, description, args schema, and a sync invoke."""

    name = "echo"
    description = "Echo the input back for testing.\nSecond line ignored."
    args: ClassVar[dict] = {"text": {"type": "string"}}

    def invoke(self, args: dict) -> str:
        return f"echoed: {args}"


def _chunk(source: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={"source": source, "title": source},
        vector_score=0.5,
        bm25_score=0.5,
        score=0.5,
    )


def _settings(**kw) -> RagSettings:
    return RagSettings(enable_agent=True, **kw)


def test_agent_finishes_immediately_and_synthesizes() -> None:
    llm = _FakeLLM([AgentAction(thought="easy", action="finish", action_input="")])
    outcome = run_agent_loop(llm, "What is attention?", kb=None, tools=[], settings=_settings())
    assert outcome.result.text == "Grounded answer [1]."
    assert outcome.chunks == []
    assert [s.action for s in outcome.trajectory] == ["finish"]


def test_agent_retrieves_then_finishes() -> None:
    kb = _FakeKB([_chunk("a.md", "alpha"), _chunk("b.md", "beta")])
    llm = _FakeLLM(
        [
            AgentAction(thought="need context", action="retrieve", action_input="attention"),
            AgentAction(thought="ready", action="finish", action_input=""),
        ]
    )
    outcome = run_agent_loop(llm, "Explain attention", kb=kb, tools=[], settings=_settings())
    assert len(outcome.chunks) == 2
    assert [s.action for s in outcome.trajectory] == ["retrieve", "finish"]
    assert "Retrieved 2 passage(s)" in outcome.trajectory[0].observation


def test_agent_calls_a_tool_and_records_the_event() -> None:
    llm = _FakeLLM(
        [
            AgentAction(thought="use tool", action="echo", action_input="ping"),
            AgentAction(thought="ready", action="finish", action_input=""),
        ]
    )
    outcome = run_agent_loop(
        llm, "echo something", kb=None, tools=[_FakeTool()], settings=_settings()
    )
    assert outcome.result.tool_events[0].name == "echo"
    assert outcome.result.tool_events[0].args == {"text": "ping"}
    assert "echoed" in outcome.trajectory[0].observation


def test_agent_honours_the_step_cap() -> None:
    kb = _FakeKB([_chunk("a.md", "alpha")])
    # The planner never chooses 'finish', so only the cap can end the loop.
    actions = [
        AgentAction(thought="again", action="retrieve", action_input=f"q{i}") for i in range(5)
    ]
    outcome = run_agent_loop(
        _FakeLLM(actions), "loop", kb=kb, tools=[], settings=_settings(agent_max_steps=2)
    )
    assert len(outcome.trajectory) == 2  # capped, never ran the 3rd step


def test_agent_dedupes_repeated_retrievals() -> None:
    kb = _FakeKB([_chunk("a.md", "alpha")])  # same hit every time
    actions = [
        AgentAction(thought="1", action="retrieve", action_input="x"),
        AgentAction(thought="2", action="retrieve", action_input="x"),
        AgentAction(thought="done", action="finish", action_input=""),
    ]
    outcome = run_agent_loop(_FakeLLM(actions), "q", kb=kb, tools=[], settings=_settings())
    assert len(outcome.chunks) == 1  # the duplicate passage is only kept once


def test_agent_unknown_action_fails_safe() -> None:
    llm = _FakeLLM(
        [
            AgentAction(thought="oops", action="banana", action_input="?"),
            AgentAction(thought="ok", action="finish", action_input=""),
        ]
    )
    outcome = run_agent_loop(llm, "q", kb=None, tools=[], settings=_settings())
    assert "Unknown action" in outcome.trajectory[0].observation
    assert outcome.result.text == "Grounded answer [1]."  # loop still produced an answer


def test_agent_without_llm_degrades_gracefully() -> None:
    outcome = run_agent_loop(None, "q", kb=None, tools=[], settings=_settings())
    assert "language model" in outcome.result.text
    assert outcome.trajectory == []


def test_agent_meters_planning_tokens() -> None:
    # The stub synthesis reports 12/8; the estimated planning tokens are added on top, so
    # the reported usage must exceed the synthesis-only baseline (#2: no more undercount).
    llm = _FakeLLM([AgentAction(thought="easy", action="finish", action_input="")])
    outcome = run_agent_loop(llm, "What is attention?", kb=None, tools=[], settings=_settings())
    assert outcome.result.input_tokens > 12
    assert outcome.result.output_tokens > 8


def test_agent_without_llm_reports_zero_planning_tokens() -> None:
    outcome = run_agent_loop(None, "q", kb=None, tools=[], settings=_settings())
    assert outcome.result.input_tokens == 0
    assert outcome.result.output_tokens == 0


def test_agent_route_is_wired_to_the_agent_step() -> None:
    assert ROUTE_TO_STEP["agent"] == "agent"
    state = {"decision": RouteDecision(route="agent", confidence=0.9, reason="multi-step")}
    assert after_route(state) == "agent"
