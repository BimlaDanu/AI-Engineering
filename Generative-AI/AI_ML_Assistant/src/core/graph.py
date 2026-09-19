"""
Engine 2 — the LangGraph implementation of the routed pipeline.

This is the counterpart to :mod:`src.core.linear`. It connects the same step
functions from :mod:`src.core.steps` as LangGraph nodes with conditional edges,
so its behaviour matches the linear engine exactly—the orchestration is the only
difference. It is included both as a learning example and to support future
graph features such as streaming, checkpointing, and visualisation.

    START ─▶ screen ─┬─(injection)──────────────────────────▶ refuse ─▶ END
                     └─▶ route ─┬─▶ retrieve ─▶ grade ─┬─▶ generate ─▶ END   (sufficient)
                                │                      └─▶ augment ─▶ generate (insufficient)
                                ├─▶ tools_only ─▶ END             (tool task)
                                ├─▶ meta ─▶ END                   (about the app)
                                ├─▶ agent ─▶ END                  (multi-step, opt-in)
                                └─▶ refuse ─▶ END                 (off-topic)

Each step accepts ``(state, deps)``. :func:`functools.partial` binds ``deps``,
allowing each graph node to match LangGraph's single-argument callable while
keeping the graph state as pure data.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.core.steps import (
    ROUTE_TO_STEP,
    PipelineDeps,
    PipelineState,
    after_grade,
    after_route,
    after_screen,
    agent,
    augment,
    generate,
    grade,
    meta,
    refuse,
    retrieve,
    route,
    screen,
    tools_only,
)


def _assemble(deps: PipelineDeps | None) -> StateGraph:
    """Wire the shared steps into a ``StateGraph``. ``deps`` is bound into each node.

    Split out from :func:`build_pipeline` so the *structure* can be built (and drawn) without
    real dependencies — the node callables are never invoked during compilation or diagram
    rendering, only when the compiled graph is actually run.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("screen", partial(screen, deps=deps))
    graph.add_node("route", partial(route, deps=deps))
    graph.add_node("retrieve", partial(retrieve, deps=deps))
    graph.add_node("grade", partial(grade, deps=deps))
    graph.add_node("augment", partial(augment, deps=deps))
    graph.add_node("generate", partial(generate, deps=deps))
    graph.add_node("tools_only", partial(tools_only, deps=deps))
    graph.add_node("meta", partial(meta, deps=deps))
    graph.add_node("agent", partial(agent, deps=deps))
    graph.add_node("refuse", partial(refuse, deps=deps))

    graph.add_edge(START, "screen")
    graph.add_conditional_edges("screen", after_screen, {"route": "route", "refuse": "refuse"})
    graph.add_conditional_edges(
        "route",
        after_route,
        {
            "retrieve": "retrieve",
            "tools_only": "tools_only",
            "meta": "meta",
            "agent": "agent",
            "refuse": "refuse",
        },
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", after_grade, {"generate": "generate", "augment": "augment"}
    )
    graph.add_edge("augment", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("tools_only", END)
    graph.add_edge("meta", END)
    graph.add_edge("agent", END)
    graph.add_edge("refuse", END)
    return graph


def build_pipeline(deps: PipelineDeps):
    """Build and compile the routed pipeline as a LangGraph ``StateGraph`` for one request.

    ``invoke(initial_state)`` on the returned compiled graph runs the full flow and yields
    the final :class:`~src.core.steps.PipelineState`.
    """
    return _assemble(deps).compile()


def run_streaming(
    deps: PipelineDeps,
    initial: dict[str, Any],
    on_node: Callable[[str], None],
) -> PipelineState:
    """Run the graph via ``stream(stream_mode="updates")``, reporting each node as it fires.

    The graph-native twin of :func:`~src.core.linear.run_linear`: same steps, same final
    state, but progress is observable. ``stream_mode="updates"`` yields one
    ``{node_name: state_delta}`` mapping per node as it *completes*; ``on_node`` is called with
    each node's name so a caller (e.g. the Chat UI) can show live progress.

    The returned state is identical to ``build_pipeline(deps).invoke(initial)``: because
    :class:`~src.core.steps.PipelineState` is a plain ``TypedDict`` with no channel reducers,
    the graph's default last-write-wins merge is reproduced exactly by folding each delta into
    the accumulator with ``dict.update`` in visit order. Streaming is thus a pure add-on — it
    changes *when* the caller learns of progress, never *what* the pipeline computes.
    """
    state: PipelineState = dict(initial)  # type: ignore[assignment]
    for update in build_pipeline(deps).stream(initial, stream_mode="updates"):
        for node, delta in update.items():
            on_node(node)
            if delta:
                state.update(delta)
    return state


def executed_nodes(*, injection: bool, route: str | None, augmented: bool) -> list[str]:
    """Return the graph nodes the last run actually traversed, in visit order.

    Reconstructed from the trace flags rather than instrumented at run time so it stays a
    pure, testable function shared by any renderer. ``route`` is the router's route name
    (``knowledge`` / ``tool`` / ``meta`` / ``agent`` / ``off_topic``); ``injection`` short-
    circuits straight to refusal; ``augmented`` inserts the Corrective-RAG ``augment`` hop.
    """
    path = ["__start__", "screen"]
    if injection:
        return [*path, "refuse", "__end__"]
    path.append("route")
    step = ROUTE_TO_STEP.get(route or "knowledge", "retrieve")
    if step == "retrieve":
        path += ["retrieve", "grade"]
        if augmented:
            path.append("augment")
        path.append("generate")
    else:
        path.append(step)
    path.append("__end__")
    return path


def pipeline_mermaid(highlight: list[str] | None = None) -> str:
    """Return the compiled pipeline as a Mermaid ``graph`` definition, drawn by LangGraph.

    This is self-documentation the linear engine cannot produce: because the pipeline is
    modelled as a real ``StateGraph``, LangGraph renders its own structure via
    ``get_graph().draw_mermaid()``. When ``highlight`` lists node ids (see
    :func:`executed_nodes`), a ``visited`` class is appended so a renderer can light up the
    path the last question actually took.
    """
    mermaid = _assemble(None).compile().get_graph().draw_mermaid()
    visited = [n for n in (highlight or []) if n not in ("__start__", "__end__")]
    if visited:
        mermaid += (
            "\n    classDef visited fill:#7c5cff,color:#ffffff,"
            "stroke:#4b32c3,stroke-width:2px\n"
            f"    class {','.join(visited)} visited\n"
        )
    return mermaid
