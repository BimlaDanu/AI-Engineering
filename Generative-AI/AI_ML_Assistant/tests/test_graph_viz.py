"""Offline tests for the graph engine's self-documentation (Mermaid diagram + path).

The linear engine is a behavioural twin of the graph engine (see ``test_engine_parity``);
what the graph engine adds is the ability to *draw itself* — a real ``StateGraph`` can emit
its own structure and light up the path a request took. These tests pin that capability down
without a network or an LLM: ``draw_mermaid`` is pure Python, and ``executed_nodes`` is a pure
reconstruction from trace flags.
"""

from __future__ import annotations

from src.core.graph import executed_nodes, pipeline_mermaid

_ALL_NODES = {
    "screen",
    "route",
    "retrieve",
    "grade",
    "augment",
    "generate",
    "tools_only",
    "meta",
    "agent",
    "refuse",
}


def test_pipeline_mermaid_contains_every_node() -> None:
    mermaid = pipeline_mermaid()
    assert "graph TD" in mermaid
    for node in _ALL_NODES:
        assert node in mermaid


def test_pipeline_mermaid_highlights_only_requested_nodes() -> None:
    # Boundary pseudo-nodes must never be styled (they are not real StateGraph nodes).
    mermaid = pipeline_mermaid(highlight=["__start__", "screen", "route", "meta", "__end__"])
    assert "classDef visited" in mermaid
    assert "class screen,route,meta visited" in mermaid
    assert "__start__ visited" not in mermaid


def test_pipeline_mermaid_no_highlight_block_when_empty() -> None:
    assert "classDef visited" not in pipeline_mermaid()
    # A path of only boundary nodes yields nothing to highlight.
    assert "classDef visited" not in pipeline_mermaid(highlight=["__start__", "__end__"])


def test_executed_nodes_injection_short_circuits() -> None:
    assert executed_nodes(injection=True, route="knowledge", augmented=False) == [
        "__start__",
        "screen",
        "refuse",
        "__end__",
    ]


def test_executed_nodes_knowledge_with_augment() -> None:
    assert executed_nodes(injection=False, route="knowledge", augmented=True) == [
        "__start__",
        "screen",
        "route",
        "retrieve",
        "grade",
        "augment",
        "generate",
        "__end__",
    ]


def test_executed_nodes_knowledge_without_augment() -> None:
    path = executed_nodes(injection=False, route="knowledge", augmented=False)
    assert "augment" not in path
    assert path == [
        "__start__",
        "screen",
        "route",
        "retrieve",
        "grade",
        "generate",
        "__end__",
    ]


def test_executed_nodes_single_step_routes() -> None:
    for route, node in [("tool", "tools_only"), ("meta", "meta"), ("agent", "agent")]:
        assert executed_nodes(injection=False, route=route, augmented=False) == [
            "__start__",
            "screen",
            "route",
            node,
            "__end__",
        ]


def test_executed_nodes_off_topic_refuses() -> None:
    assert executed_nodes(injection=False, route="off_topic", augmented=False) == [
        "__start__",
        "screen",
        "route",
        "refuse",
        "__end__",
    ]


def test_executed_nodes_unknown_route_falls_back_to_retrieve() -> None:
    # An unrecognised route mirrors the pipeline's safe default (the knowledge path).
    path = executed_nodes(injection=False, route="mystery", augmented=False)
    assert "retrieve" in path and "generate" in path
