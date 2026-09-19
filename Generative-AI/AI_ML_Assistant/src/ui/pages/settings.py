"""⚙️ Settings workspace: model selection, LLM generation params, RAG tuning, and status.

This page is the app's control center. It replaces the old sidebar panel: the sidebar is
now purely navigation, and everything global lives here — the active **Model**, the
**generation parameters** (temperature, top-p, max tokens), the developer **RAG tuning**
knobs, a live **System status** panel, and **Session** usage/export. Every widget binds to
``st.session_state`` by ``key`` (or mutates the shared :class:`RagSettings`), so choices
made here take effect on the next answer without any extra plumbing.

The page is laid out as a set of small, single-purpose section renderers (one per ``###``
heading) wired together by :func:`render`. Each reads and writes the shared settings object
directly, so sections stay independent and easy to read top-to-bottom.
"""

from __future__ import annotations

import html

import streamlit as st

from src.config import (
    API_EMBEDDING_MODEL,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL_NAME,
    RagSettings,
    google_api_key,
    openrouter_api_key,
)
from src.core.sources import SOURCE_REGISTRY
from src.ui.exporters import export_controls
from src.ui.models import model_cost_hint, model_picker
from src.ui.registry import register_page
from src.ui.state import load_kb


def _status_row(label: str, value: str, ok: bool | None = None) -> None:
    """Render one ``label → value`` status line, with an optional 🟢/🔴 health dot."""
    dot = "" if ok is None else ("🟢 " if ok else "🔴 ")
    st.markdown(
        f'<div class="sb-status"><span>{html.escape(label)}</span>'
        f"<span>{dot}{html.escape(value)}</span></div>",
        unsafe_allow_html=True,
    )


def _model_and_generation() -> None:
    """Pick the active model and set how it writes answers (temperature, top-p, length)."""
    settings: RagSettings = st.session_state.settings
    st.markdown("### 🤖 Model & generation")
    st.caption("Choose the model and how it answers.")

    col_model, col_temp = st.columns([2, 1])
    with col_model:
        # The same picker the 💬 Chat answer-settings popover draws, on the same
        # session key — one setting, seen from wherever the reader happens to be.
        model_picker()
    with col_temp:
        settings.temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            settings.temperature,
            0.05,
            help="Higher = more creative, lower = more predictable.",
        )

    model_cost_hint(st.session_state.model)

    col_top_p, col_max = st.columns(2)
    with col_top_p:
        settings.top_p = st.slider(
            "Top-p",
            0.0,
            1.0,
            settings.top_p,
            0.05,
            help="Another way to control variety. Leave at 1.0 unless you know you need it.",
        )
    with col_max:
        settings.max_tokens = st.number_input(
            "Max response length",
            min_value=0,
            max_value=32000,
            value=settings.max_tokens,
            step=256,
            help="Longest answer allowed, in tokens. 0 = let the model decide.",
        )


def _rag_tuning() -> None:
    """Controls for how the assistant finds and picks passages before answering.

    Also reachable from the Chat header popover — same settings object, so changes here
    and there stay in sync.
    """
    settings: RagSettings = st.session_state.settings
    st.markdown("### 🔧 Retrieval")
    st.caption("How the assistant looks things up. Also on the Chat header.")

    col_k, col_alpha = st.columns(2)
    with col_k:
        settings.top_k = st.slider(
            "Passages to use",
            1,
            10,
            settings.top_k,
            help="How many knowledge-base passages to feed the model per answer.",
        )
    with col_alpha:
        settings.hybrid_alpha = st.slider(
            "Meaning ↔ exact words",
            0.0,
            1.0,
            settings.hybrid_alpha,
            0.05,
            help="1.0 = match by meaning, 0.0 = match exact keywords.",
        )
    settings.multi_query = st.toggle(
        "🔀 Multi-query fusion",
        settings.multi_query,
        help="Ask the question a few different ways and merge the results. Finds more on "
        "vague questions; a bit slower.",
    )
    if settings.multi_query:
        settings.multi_query_count = st.slider(
            "How many versions",
            2,
            5,
            settings.multi_query_count,
            help="How many rephrasings to try.",
        )
    settings.rewrite_query = st.toggle(
        "Rewrite the question",
        settings.rewrite_query,
        disabled=settings.multi_query,
        help="Tidy the question before searching. Turned off while multi-query is on.",
    )
    settings.rerank = st.toggle(
        "🎯 Re-rank results",
        settings.rerank,
        help="Pull a bigger set, then let the model reorder it by relevance before the cut. "
        "Sharper picks; one extra step.",
    )
    if settings.rerank:
        settings.rerank_candidates = st.slider(
            "How many to re-rank",
            settings.top_k,
            20,
            settings.rerank_candidates,
            help="Size of the set to reorder before trimming to your top-k.",
        )
    settings.mmr = st.toggle(
        "🌈 Prefer varied results",
        settings.mmr,
        disabled=settings.rerank,
        help="Avoid near-duplicate passages so the results cover more ground. No extra "
        "model call. An alternative to re-ranking — off while that's on.",
    )
    if settings.mmr and not settings.rerank:
        settings.mmr_candidates = st.slider(
            "How many to pick from",
            settings.top_k,
            20,
            settings.mmr_candidates,
            help="Size of the set to choose your top-k from.",
        )
        settings.mmr_lambda = st.slider(
            "Focused ↔ varied",
            0.0,
            1.0,
            settings.mmr_lambda,
            step=0.05,
            help="Higher = stick to the closest matches; lower = more variety.",
        )
    settings.filter_difficulty = st.toggle(
        "Match my level",
        settings.filter_difficulty,
        help="Only use passages tagged for the learner level you picked.",
    )
    settings.compare_no_rag = st.toggle(
        "🆚 Show a no-lookup answer too",
        settings.compare_no_rag,
        help="Also answer without the knowledge base, so you can see what it adds.",
    )

    _kb_filters(settings)
    _corrective_rag(settings)
    _agent(settings)
    _mcp(settings)

    # Same steps, two ways of wiring them together — switch to compare identical behaviour.
    engine_labels = {"linear": "⚡ Lightweight", "graph": "🕸️ LangGraph"}
    current = settings.pipeline_engine if settings.pipeline_engine in engine_labels else "linear"
    settings.pipeline_engine = st.radio(
        "How the steps run",
        list(engine_labels),
        index=list(engine_labels).index(current),
        format_func=lambda key: engine_labels[key],
        horizontal=True,
        help="Same steps, two ways of running them. They give the same answer — "
        "Experiments shows which one ran.",
    )


def _kb_filters(settings: RagSettings) -> None:
    """Scope retrieval to specific knowledge-base facets (topic / source / year).

    The choices are read live from the index via :meth:`KnowledgeBase.facets`, so only facets
    that actually exist are offered. An empty box means "everything" (the default), and any
    stale selection that no longer exists in the index is silently dropped.
    """
    st.markdown("#### 🔎 Search within")
    st.caption("Limit the search to a topic, document, or year. Leave empty for everything.")
    kb = load_kb()
    if kb is None:
        st.info("Nothing indexed yet — run `make ingest` to enable this.")
        return
    facets = kb.facets()
    col_topic, col_year = st.columns([2, 1])
    with col_topic:
        settings.filter_topics = st.multiselect(
            "Topics",
            facets["topics"],
            default=[t for t in settings.filter_topics if t in facets["topics"]],
            help="Only retrieve chunks tagged with one of these topics.",
        )
    with col_year:
        settings.filter_years = st.multiselect(
            "Years",
            facets["years"],
            default=[y for y in settings.filter_years if y in facets["years"]],
            help="Only retrieve chunks from these publication years.",
        )
    settings.filter_sources = st.multiselect(
        "Source documents",
        facets["sources"],
        default=[s for s in settings.filter_sources if s in facets["sources"]],
        help="Only retrieve chunks from these source files.",
    )
    if settings.filter_topics or settings.filter_sources or settings.filter_years:
        st.caption("🔎 Search is limited to your picks for every new question.")


def _corrective_rag(settings: RagSettings) -> None:
    """Corrective-RAG (CRAG) controls: sufficiency gate + external augmentation."""
    st.markdown("#### 🩹 When the answer is thin")
    st.caption("If the knowledge base comes up short, pull in outside sources with citations.")
    settings.enable_augmentation = st.toggle(
        "Look elsewhere when needed", settings.enable_augmentation
    )
    settings.min_similarity = st.slider(
        "When to look elsewhere",
        0.0,
        0.6,
        settings.min_similarity,
        0.05,
        help="How weak the local results must be before reaching out. Higher = reach out more.",
        disabled=not settings.enable_augmentation,
    )
    available = list(SOURCE_REGISTRY)
    settings.augment_sources = st.multiselect(
        "Where to look",
        available,
        default=[s for s in settings.augment_sources if s in available],
        help="`arxiv` is free; `web` does a live web search (small cost, no new key).",
        disabled=not settings.enable_augmentation,
    )
    if "web" in settings.augment_sources and settings.enable_augmentation:
        st.caption(
            "🌐 Web search is on — great for recent topics, small cost per lookup. "
            "Turn it off to stay free and offline."
        )
    settings.augment_k = st.slider(
        "Passages per source",
        1,
        6,
        settings.augment_k,
        disabled=not settings.enable_augmentation,
    )


def _agent(settings: RagSettings) -> None:
    """Bounded agent route (Phase 6, Option C): a capped plan->act->observe loop."""
    st.markdown("#### 🤖 Agent mode")
    st.caption(
        "For questions that need several steps, let the assistant search and use tools a "
        "few times before answering (e.g. *find a recent paper on X and estimate its cost*). "
        "Off by default. You can watch each step in **🧪 Experiments**."
    )
    settings.enable_agent = st.toggle(
        "Handle multi-step questions",
        settings.enable_agent,
        help="Simple questions still take the direct route; safety checks always run first.",
    )
    settings.agent_max_steps = st.slider(
        "Max steps",
        1,
        8,
        settings.agent_max_steps,
        help="How many steps it may take before it has to answer.",
        disabled=not settings.enable_agent,
    )


def _mcp(settings: RagSettings) -> None:
    """Remote MCP (Model Context Protocol) tools: connect an external tool server."""
    st.markdown("#### 🔌 Remote tools (MCP)")
    st.caption(
        "Connect an outside tool server so the assistant can use extra tools while it "
        "answers — beyond the built-in arXiv, calculator, and token tools. Off by default: "
        "needs the `langchain-mcp-adapters` package and network access."
    )
    settings.enable_mcp = st.toggle("Use remote tools", settings.enable_mcp)
    settings.mcp_server_url = st.text_input(
        "MCP server URL",
        value=settings.mcp_server_url,
        disabled=not settings.enable_mcp,
        help="Default: DeepWiki — a public, no-auth, read-only server that answers "
        "questions about public GitHub repositories.",
    )
    if not settings.enable_mcp:
        return
    from src.core.mcp_client import clear_cache, load_mcp_tools

    url = settings.mcp_server_url.strip()
    # Evict cached tools when the target server changes, so the per-URL cache can't grow
    # unbounded across experimentation and the next probe reconnects to the new server.
    if st.session_state.get("mcp_probed_url") not in (None, url):
        clear_cache()
        st.session_state.pop("mcp_probe_status", None)

    # Probe on demand only: automatically the first time a URL is seen, and thereafter only
    # when the user clicks "Test connection". This avoids a "Connecting…" spinner on every
    # unrelated Settings rerun — the result is remembered in session state between reruns.
    should_probe = st.button("🔌 Test connection") or (
        st.session_state.get("mcp_probed_url") != url
    )
    if should_probe:
        with st.spinner("Connecting to the MCP server…"):
            tools = load_mcp_tools(settings)
        st.session_state.mcp_probed_url = url
        st.session_state.mcp_probe_status = [t.name for t in tools] if tools else None

    names = st.session_state.get("mcp_probe_status")
    if names:
        st.success(
            f"Connected — {len(names)} tool(s) available: {', '.join(names)}. "
            "The model will call them when useful."
        )
    elif st.session_state.get("mcp_probed_url") == url:
        st.warning(
            "No tools loaded. Install `langchain-mcp-adapters` (add it to `pyproject.toml`, "
            "then run `make sync`) and confirm the server URL is reachable."
        )
    else:
        st.info("Click **🔌 Test connection** to probe the server for available tools.")


def _limits_and_observability() -> None:
    """Per-session rate limiting and structured run logging (medium optionals #9/#10)."""
    settings: RagSettings = st.session_state.settings
    st.markdown("### 🔒 Limits & logging")
    st.caption("Cap how fast a session can spend, and keep a local log of each request.")
    settings.rate_limit_enabled = st.toggle(
        "Limit questions per session",
        settings.rate_limit_enabled,
        help="Slows things down to avoid surprise cost and runaway loops.",
    )
    col_burst, col_rate = st.columns(2)
    with col_burst:
        settings.rate_limit_burst = st.slider(
            "Quick questions",
            1,
            20,
            settings.rate_limit_burst,
            help="How many go through instantly before pacing starts.",
            disabled=not settings.rate_limit_enabled,
        )
    with col_rate:
        settings.rate_limit_per_min = st.slider(
            "Per minute after that",
            1,
            120,
            settings.rate_limit_per_min,
            help="Steady rate once the quick ones are used up.",
            disabled=not settings.rate_limit_enabled,
        )

    settings.enable_run_log = st.toggle(
        "Keep a request log",
        settings.enable_run_log,
        help="Save one line per request (route, cost, timings) to logs/runs.jsonl. "
        "No secrets; never changes the answer.",
    )
    if settings.enable_run_log:
        st.caption("Logging to `logs/runs.jsonl` (created on first request).")


def _system_status() -> None:
    """Live system-status panel from real config and knowledge-base state."""
    st.markdown("### 📡 System status")

    if EMBEDDING_BACKEND == "api":
        emb = API_EMBEDDING_MODEL.split("/")[-1]
    else:
        emb = EMBEDDING_MODEL_NAME.split("/")[-1]
    _status_row("Embeddings", f"{emb} ({EMBEDDING_BACKEND})")
    _status_row("Vector DB", "Chroma")

    kb = load_kb()
    if kb is None:
        _status_row("Knowledge base", "not indexed", ok=False)
    else:
        docs = len(kb.stats()["by_source"])
        _status_row("Knowledge base", f"{docs} docs · {kb.size} chunks", ok=True)

    has_or = bool(openrouter_api_key())
    has_google = bool(google_api_key())
    _status_row("OpenRouter", "ready" if has_or else "no key", ok=has_or)
    _status_row("Google (Gemini)", "ready" if has_google else "optional", ok=has_google or None)
    _status_row("arXiv", "public", ok=True)


def _session() -> None:
    """Session usage totals plus new-chat and export actions."""
    st.markdown("### 💾 Session")
    totals = st.session_state.totals
    _status_row("Tokens in / out", f"{totals['input']} / {totals['output']}")
    _status_row("Est. cost", f"${totals['cost']:.4f}")

    if st.button("🆕 New chat", width="stretch"):
        st.session_state.history = []
        st.session_state.trace = None
        st.rerun()

    st.caption("Export this conversation:")
    export_controls(st.session_state.history, key_prefix="settings")


@register_page("⚙️ Settings", key="settings", section="System", order=90)
def render() -> None:
    """Settings workspace: model, generation params, RAG tuning, status, and session."""
    st.subheader("⚙️ Settings")
    st.caption("The control center — everything global to the app lives here.")

    col_left, col_right = st.columns(2)
    with col_left:
        _model_and_generation()
        _rag_tuning()
    with col_right:
        _system_status()
        st.divider()
        _limits_and_observability()
        st.divider()
        _session()
