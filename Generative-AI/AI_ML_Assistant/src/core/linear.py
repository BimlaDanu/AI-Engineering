"""
Engine 1 — the lightweight typed-router pipeline.

Connects the shared steps from :mod:`src.core.steps` using plain Python control
flow (`if/elif`) based on the typed
:class:`~src.core.schemas.RouteDecision`. This is the default engine: simple,
easy to follow, and straightforward to unit test. Its counterpart,
:mod:`src.core.graph`, runs the same pipeline using a LangGraph
``StateGraph``.

    screen ─▶ (injection?) ─▶ refuse
                 else       ─▶ route ─▶ retrieve ─▶ grade ─┬▶ generate          (sufficient)
                                     │                     └▶ augment ─▶ generate (insufficient)
                                     ├▶ tools_only          (tool task)
                                     ├▶ meta                (about the app)
                                     ├▶ agent               (multi-step, opt-in)
                                     └▶ refuse              (off-topic)
"""

from __future__ import annotations

from typing import Any

from src.core.steps import (
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

# Terminal single-step routes. The knowledge route ("retrieve") is handled separately
# because it runs the multi-step Corrective-RAG chain with a branch after grading. The
# ``agent`` step is single-step here too — its own bounded loop lives inside the step.
_PATHS: dict[str, tuple] = {
    "tools_only": (tools_only,),
    "meta": (meta,),
    "agent": (agent,),
    "refuse": (refuse,),
}


def run_linear(deps: PipelineDeps, initial: dict[str, Any]) -> PipelineState:
    """Run the routed pipeline with plain branching and return the final state."""
    state: PipelineState = dict(initial)  # type: ignore[assignment]

    state.update(screen(state, deps))
    if after_screen(state) == "refuse":
        state.update(refuse(state, deps))
        return state

    state.update(route(state, deps))
    target = after_route(state)
    if target == "retrieve":
        # Knowledge path: retrieve -> grade -> (augment if insufficient) -> generate.
        state.update(retrieve(state, deps))
        state.update(grade(state, deps))
        if after_grade(state) == "augment":
            state.update(augment(state, deps))
        state.update(generate(state, deps))
    else:
        for step in _PATHS[target]:
            state.update(step(state, deps))
    return state
