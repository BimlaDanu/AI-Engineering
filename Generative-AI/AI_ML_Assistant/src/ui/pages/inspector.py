"""🧪 Experiments workspace: the last-question pipeline trace plus a tool playground."""

from __future__ import annotations

import json

import streamlit as st

from src.core.graph import executed_nodes, pipeline_mermaid
from src.ui.pages.tools import render_tools_playground
from src.ui.registry import register_page


@register_page("🧪 Experiments", key="experiments", section="Analyse", order=65)
def render() -> None:
    """Experiments tab: the full pipeline trace of the last question, plus a tool playground."""
    st.subheader("🧪 Experiments")
    st.caption(
        "Inspect one question end-to-end — the full retrieval/routing trace of your last "
        "answer, plus a playground to fire each tool call standalone. "
        "Session-wide usage totals live in 📈 Analytics."
    )
    trace_tab, tools_tab = st.tabs(["🔬 Pipeline trace", "🛠️ Tool playground"])
    with trace_tab:
        _render_trace()
    with tools_tab:
        render_tools_playground()


def _render_trace() -> None:
    """Render the full RAG pipeline trace of the most recent question."""
    st.caption(
        "Tune the pipeline in the Chat header (**⚙️ Answer settings → Advanced RAG**) "
        "or in **⚙️ Settings**, and watch the effect here."
    )
    trace = st.session_state.trace
    if not trace:
        st.info("Ask something in **💬 AI Chat** first — the full trace will appear here.")
        return
    st.markdown(f"**Original question:** {trace['question']}")
    queries = trace.get("search_queries") or [trace["rewritten_query"]]
    if len(queries) > 1:
        # Multi-query fusion ran: show every variant that was retrieved and fused (RAG-Fusion).
        st.markdown(f"**Multi-query fusion ({len(queries)} queries, Reciprocal Rank Fusion):**")
        for q in queries:
            st.markdown(f"- {q}")
    else:
        st.markdown(f"**Rewritten search query:** {trace['rewritten_query']}")
    if trace.get("reranked"):
        st.markdown(
            "**Reranked:** ✅ an LLM reordered the candidate pool by relevance "
            "(listwise rerank) before the top-k cut."
        )
    if trace.get("diversified"):
        st.markdown(
            "**Diversified (MMR):** ✅ the candidate pool was reselected to balance "
            "relevance against novelty, reducing near-duplicate passages in the top-k."
        )
    st.markdown(
        f"**Subject filter:** `{trace['subject_filter']}` · "
        f"**Technique:** {trace.get('technique', 'Standard')} · "
        f"**Length:** {trace.get('response_length', 'Balanced')}"
    )
    _render_metadata_filters(trace)
    _render_graph(trace)
    _render_agent(trace)
    _render_corrective_rag(trace)
    _render_mcp(trace)
    if trace["chunks"]:
        st.markdown("**Retrieved chunks (vector vs BM25 vs blended score):**")
        st.table(trace["chunks"])
    else:
        st.warning("No chunks retrieved for this question.")
    _render_cost_breakdown(trace.get("tokens") or {})
    st.json(
        {
            "timings_s": trace["timings_s"],
            "tokens": trace["tokens"],
            "model": trace["model"],
        }
    )


def _render_cost_breakdown(tokens: dict) -> None:
    """Show the per-model token/cost split (answer model vs. cheaper classification model)."""
    breakdown = tokens.get("breakdown")
    if not breakdown:
        return
    answer = breakdown.get("answer", {})
    router = breakdown.get("router", {})
    st.markdown(
        f"**💰 Cost by model** — total **≈ ${tokens.get('cost_usd', 0):.5f}** "
        f"({tokens.get('input', 0)}+{tokens.get('output', 0)} tokens):  \n"
        f"• Answer `{answer.get('model', '?')}`: "
        f"{answer.get('input', 0)}+{answer.get('output', 0)} tokens "
        f"· ≈ ${answer.get('cost_usd', 0):.5f}  \n"
        f"• Routing `{router.get('model', '?')}` "
        f"(routing · injection · grading): "
        f"{router.get('input', 0)}+{router.get('output', 0)} tokens "
        f"· ≈ ${router.get('cost_usd', 0):.5f}"
    )


def _render_metadata_filters(trace: dict) -> None:
    """Show which knowledge-base facets (topic/source/year) narrowed this retrieval, if any."""
    filters = trace.get("metadata_filters") or {}
    parts = []
    if filters.get("topics"):
        parts.append(f"**topic** ∈ {{{', '.join(filters['topics'])}}}")
    if filters.get("sources"):
        parts.append(f"**source** ∈ {{{', '.join(filters['sources'])}}}")
    if filters.get("years"):
        parts.append(f"**year** ∈ {{{', '.join(filters['years'])}}}")
    if parts:
        st.markdown("**🔎 Knowledge-base filters:** " + " · ".join(parts))


def _render_graph(trace: dict) -> None:
    """Draw the pipeline as a LangGraph diagram with the last run's path highlighted.

    This is the one thing the graph engine gives that the linear engine cannot: because the
    pipeline is a real ``StateGraph``, LangGraph renders its own structure. The nodes the last
    question actually traversed are lit up, so the routing/Corrective-RAG decisions read off
    the diagram directly. Both engines run identical steps, so the picture is valid whichever
    one produced this trace (the engine that ran is noted).
    """
    injection = bool((trace.get("injection") or {}).get("is_injection"))
    visited = executed_nodes(
        injection=injection,
        route=trace.get("route"),
        augmented=bool(trace.get("augmented")),
    )
    mermaid = pipeline_mermaid(highlight=visited)
    engine = trace.get("engine", "linear")
    engine_label = "🕸️ LangGraph" if engine == "graph" else "⚡ Lightweight router"

    st.markdown("**🕸️ Execution graph** — drawn by LangGraph from the compiled `StateGraph`:")
    st.caption(
        f"Engine that ran: **{engine_label}**. Highlighted nodes are the path this question "
        "took. The linear engine runs the *same* steps but has no graph to draw itself."
    )
    html = (
        '<div class="mermaid" id="pipeline-graph"></div>'
        '<script type="module">'
        "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';"
        f"const src = {json.dumps(mermaid)};"
        "const el = document.getElementById('pipeline-graph');"
        "el.textContent = src;"
        "mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });"
        "mermaid.run({ nodes: [el] }).catch(e => { el.textContent = 'Diagram render "
        "failed — see the Mermaid source below.'; });"
        "</script>"
    )
    # ``st.iframe`` (Streamlit ≥1.55) auto-detects a raw HTML string and embeds it in the same
    # sandboxed iframe the old ``components.html`` used — it just replaces the API deprecated for
    # removal after 2026-06-01. The iframe scrolls its own overflow, so no ``scrolling`` flag.
    st.iframe(html, height=560)
    with st.expander("Mermaid source (renders on mermaid.live or GitHub, works offline)"):
        st.code(mermaid, language="text")


def _render_agent(trace: dict) -> None:
    """Show the bounded agent's plan->act->observe trajectory, if the agent route ran."""
    steps = trace.get("agent")
    if not steps:
        return  # the agent route only runs for multi-step questions when enabled
    st.markdown(f"**🤖 Agent trajectory** ({len(steps)} step(s)):")
    for i, step in enumerate(steps, start=1):
        label = step["action"]
        if step.get("action_input"):
            label += f" · `{step['action_input']}`"
        with st.expander(f"Step {i}: {label}"):
            st.markdown(f"**Thought:** {step['thought']}")
            st.markdown(f"**Observation:** {step['observation']}")


def _render_corrective_rag(trace: dict) -> None:
    """Show the Corrective-RAG sufficiency grade and any external augmentation."""
    grade = trace.get("grade")
    if not grade:
        return  # grading only runs on the knowledge route
    if grade["sufficient"]:
        st.success(
            f"🩹 **Corrective RAG:** knowledge base graded **sufficient** "
            f"(confidence {grade['confidence']:.0%}). {grade['reason']}"
        )
    else:
        used = trace.get("augment_sources_used") or []
        if trace.get("augmented") and used:
            st.info(
                f"🩹 **Corrective RAG:** graded **insufficient** — augmented from "
                f"**{', '.join(used)}**. {grade['reason']}"
            )
        else:
            st.warning(
                f"🩹 **Corrective RAG:** graded **insufficient**, but no external source "
                f"returned passages. {grade['reason']}"
            )


def _render_mcp(trace: dict) -> None:
    """Show whether remote MCP tools were connected for this question, and which."""
    mcp = trace.get("mcp")
    if not mcp:
        return  # MCP disabled for this request
    if mcp["tools"]:
        st.info(
            f"🔌 **Remote MCP:** connected to `{mcp['server']}` — "
            f"{len(mcp['tools'])} tool(s) available: **{', '.join(mcp['tools'])}**. "
            "Any the model actually called are listed under 🛠️ Tool calls in the Chat tab."
        )
    else:
        st.warning(
            f"🔌 **Remote MCP:** enabled for `{mcp['server']}`, but no tools loaded — the "
            "`langchain-mcp-adapters` dependency may be missing or the server unreachable."
        )
