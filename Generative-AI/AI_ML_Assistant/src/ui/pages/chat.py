"""💬 Chat workspace: the conversation, sources/tool expanders, and re-ask controls."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import streamlit as st

from src.config import (
    LEVELS,
    PROMPT_TECHNIQUES,
    RESPONSE_LENGTHS,
    SUBJECTS,
    RagSettings,
)
from src.core.service import AnswerBundle, AnswerRequest
from src.rag.promote import external_candidates, promotion_id
from src.ratelimit import TokenBucket
from src.ui.exporters import export_controls
from src.ui.models import model_cost_hint, model_label, model_picker
from src.ui.registry import register_page
from src.ui.state import get_service, kb_status_notice, load_kb
from src.ui.theme import NAV_TITLE

# Friendly labels for the graph engine's live node-by-node progress (Track A). Keys are the
# LangGraph node names from :mod:`src.core.graph`; anything unmapped falls back to its raw name.
_NODE_LABELS = {
    "screen": "🛡️ Screening for prompt injection…",
    "route": "🧭 Routing the question…",
    "retrieve": "📚 Retrieving from the knowledge base…",
    "grade": "🩹 Grading retrieval sufficiency…",
    "augment": "🌐 Augmenting from external sources…",
    "generate": "✍️ Composing the grounded answer…",
    "tools_only": "🛠️ Running tools…",
    "meta": "💬 Answering about the assistant…",
    "agent": "🤖 Running the agent loop…",
    "refuse": "🚫 Declining (off-domain or unsafe)…",
}

# Shown as one-click starter questions when the chat is empty.
EXAMPLE_QUESTIONS = [
    "What is an embedding, and how does cosine similarity compare two of them?",
    "RAG vs fine-tuning — when should I use which?",
    "How does backpropagation train a neural network?",
    "How do transformers use attention?",
    "What is the Model Context Protocol (MCP)?",
    "What is an API, and how do apps call LLMs through one?",
]


def _rate_bucket() -> TokenBucket:
    """Return this session's request token-bucket, rebuilt if the limit config changed.

    Keyed on the (burst, per-minute) settings so tuning the limit in ⚙️ Settings takes
    effect on the next question rather than being frozen at first use.
    """
    ss = st.session_state
    settings: RagSettings = ss.settings
    sig = (settings.rate_limit_burst, settings.rate_limit_per_min)
    if ss.get("_rate_sig") != sig or "rate_bucket" not in ss:
        ss.rate_bucket = TokenBucket.per_minute(
            settings.rate_limit_burst, settings.rate_limit_per_min
        )
        ss._rate_sig = sig
    return ss.rate_bucket


def _rate_limited(question: str) -> bool:
    """Charge one token for this question; if the bucket is empty, post a friendly wait.

    Returns True when the request was throttled (and a notice was appended to history), so
    the caller skips the expensive pipeline call entirely — no tokens spent, no API cost.
    """
    ss = st.session_state
    if not ss.settings.rate_limit_enabled:
        return False
    bucket = _rate_bucket()
    if bucket.try_consume():
        return False
    wait = max(1, math.ceil(bucket.retry_after()))
    ss.history.append({"role": "user", "content": question})
    ss.history.append(
        {
            "role": "assistant",
            "content": (
                f"⏳ You're asking a little fast. Please wait about **{wait} second(s)** and "
                "try again — this per-session rate limit guards against runaway API usage and "
                "cost. You can adjust or disable it in ⚙️ Settings."
            ),
        }
    )
    return True


def process_question(
    question: str,
    level: str,
    ctx: dict[str, Any],
    on_node: Callable[[str], None] | None = None,
) -> None:
    """Run the pipeline for one question and append the result to chat history.

    ``on_node`` is forwarded to the service for live graph progress; ``None`` on the linear
    engine (see :func:`_run_with_progress`).
    """
    ss = st.session_state
    if _rate_limited(question):
        return
    ss.history.append({"role": "user", "content": question})
    pairs = [
        (m["role"], m["content"]) for m in ss.history[:-1] if m["role"] in ("user", "assistant")
    ]
    request = AnswerRequest(
        question=question,
        level=level,
        model=ctx["model"],
        subjects=ctx["subjects"],
        settings=ss.settings,
        history=pairs,
        technique=ss.get("technique", "Standard"),
        length=ss.get("response_length", "Balanced"),
        extra_instructions=ss.extra_instructions,
    )
    bundle = get_service().answer(request, on_node=on_node)
    if not bundle.ok:
        ss.history.append({"role": "assistant", "content": bundle.text})
        return

    totals = ss.totals
    totals["input"] += bundle.tokens_in
    totals["output"] += bundle.tokens_out
    totals["cost"] += bundle.cost
    ss.history.append(
        {
            "role": "assistant",
            "content": bundle.text,
            "question": question,
            "sources": bundle.sources,
            "tools": bundle.tools,
            "no_rag": bundle.no_rag,
            "meta": bundle.meta,
        }
    )
    ss.trace = bundle.trace
    _capture_promotable(bundle)


def _run_with_progress(question: str, ans_level: str, ctx: dict[str, Any]) -> None:
    """Answer one question, showing live node progress on the graph engine.

    The LangGraph engine can stream its nodes as they fire, so it gets an ``st.status`` panel
    that ticks each stage; the linear engine has nothing to stream, so it keeps the plain
    spinner. Either way the work runs through the same :func:`process_question`.
    """
    ss = st.session_state
    if getattr(ss.settings, "pipeline_engine", "linear") == "graph":
        with st.status(f"Answering at {ans_level} level…", expanded=True) as status:
            process_question(
                question,
                ans_level,
                ctx,
                on_node=lambda node: status.write(_NODE_LABELS.get(node, node)),
            )
            status.update(label="Answer ready", state="complete", expanded=False)
    else:
        with st.spinner(f"Answering at {ans_level} level…"):
            process_question(question, ans_level, ctx)


def _capture_promotable(bundle: AnswerBundle) -> None:
    """Accumulate any external (arXiv/web) passages from this answer as promotion candidates.

    Corrective-RAG augmentation cites external passages for one answer then discards them;
    here they are stashed (deduped by promotion id) in ``st.session_state`` so the 📄
    Knowledge Base page can offer to promote the good ones into the KB. UI-only — the core
    pipeline never touches Streamlit state.
    """
    store: dict[str, dict] = st.session_state.setdefault("promotable", {})
    for chunk in external_candidates(bundle.sources, bundle.contexts):
        store[promotion_id(chunk)] = {"text": chunk.text, "metadata": chunk.metadata}


def _kb_filter_controls(settings: RagSettings) -> None:
    """Compact topic/source/year knowledge-base filters (mirrors ⚙️ Settings → 🔎 filters).

    Options come from the live index via :meth:`KnowledgeBase.facets`; an empty box means
    "everything" and any selection no longer present in the index is dropped. Rendered only
    when a knowledge base is indexed and actually exposes at least one facet value.
    """
    kb = load_kb()
    if kb is None:
        return
    facets = kb.facets()
    if not (facets["topics"] or facets["sources"] or facets["years"]):
        return
    st.caption("🔎 Filter the knowledge base (empty = everything):")
    settings.filter_topics = st.multiselect(
        "Topics",
        facets["topics"],
        default=[t for t in settings.filter_topics if t in facets["topics"]],
    )
    settings.filter_sources = st.multiselect(
        "Source documents",
        facets["sources"],
        default=[s for s in settings.filter_sources if s in facets["sources"]],
    )
    settings.filter_years = st.multiselect(
        "Years",
        facets["years"],
        default=[y for y in settings.filter_years if y in facets["years"]],
    )


def _answer_settings() -> None:
    """Popover with every answer-shaping control, next to where it takes effect.

    All widgets bind to ``st.session_state`` by ``key`` (seeded in ``init_state``); the
    entry point reads those keys to build the per-run ``ctx``. Basics are always visible;
    the developer RAG knobs sit behind an *Advanced* expander (progressive disclosure).

    **Model** leads the popover because it is the choice that changes an answer the most —
    and, until it was here, the one control you had to leave the conversation to change.
    It is :func:`~src.ui.models.model_picker`, the same widget on the same session key that
    ⚙️ Settings draws, so the two are one setting rather than two that can disagree.
    """
    settings: RagSettings = st.session_state.settings
    with st.popover("⚙️ Answer settings", width="content"):
        model_picker("Model", help="Which LLM writes the answer. Same list as ⚙️ Settings.")
        model_cost_hint(st.session_state.model)
        col_subject, col_level = st.columns(2)
        with col_subject:
            st.selectbox(
                "Subject",
                list(SUBJECTS),
                key="subject_label",
                help="Scopes retrieval to that subject's documents (overlap docs are shared).",
            )
        with col_level:
            st.select_slider("Explain like I'm a…", LEVELS, key="level")
        col_technique, col_length = st.columns(2)
        with col_technique:
            st.selectbox(
                "Prompt technique",
                list(PROMPT_TECHNIQUES),
                key="technique",
                help="How the assistant structures its teaching (chain-of-thought, few-shot, …).",
            )
        with col_length:
            st.select_slider("Response length", list(RESPONSE_LENGTHS), key="response_length")

        with st.expander("⚙️ Advanced: RAG tuning"):
            settings.top_k = st.slider("Top-k chunks", 1, 10, settings.top_k)
            settings.hybrid_alpha = st.slider(
                "Hybrid weight — vector ↔ keyword",
                0.0,
                1.0,
                settings.hybrid_alpha,
                0.05,
                help="1.0 = pure vector similarity, 0.0 = pure BM25 keyword search.",
            )
            settings.enable_augmentation = st.toggle(
                "🩹 Corrective RAG (augment when KB is thin)",
                settings.enable_augmentation,
                help="Fetch citable external passages (arXiv, web search) when retrieval is "
                "insufficient, instead of answering from memory. Configure sources in "
                "⚙️ Settings → Corrective RAG.",
            )
            settings.min_similarity = st.slider(
                "Sufficiency floor (best retrieval score)",
                0.0,
                0.6,
                settings.min_similarity,
                0.05,
                help="Below this best hybrid score the context is graded insufficient "
                "and augmentation kicks in.",
                disabled=not settings.enable_augmentation,
            )
            settings.multi_query = st.toggle(
                "🔀 Multi-query fusion (RAG-Fusion)",
                settings.multi_query,
                help="Expand into several diverse queries, retrieve each, and fuse by "
                "Reciprocal Rank Fusion. Better recall; supersedes single-shot rewriting.",
            )
            settings.rewrite_query = st.toggle(
                "Query rewriting",
                settings.rewrite_query,
                disabled=settings.multi_query,
                help="Single-shot query translation. Ignored while multi-query fusion is on.",
            )
            settings.rerank = st.toggle(
                "🎯 Listwise reranking (RAG stage 2)",
                settings.rerank,
                help="Over-fetch a wider candidate pool, then let an LLM reorder it by "
                "relevance before the top-k cut. Sharper precision; one extra LLM call.",
            )
            if settings.rerank:
                settings.rerank_candidates = st.slider(
                    "Candidates to rerank",
                    settings.top_k,
                    20,
                    settings.rerank_candidates,
                )
            settings.mmr = st.toggle(
                "🌈 MMR diversity (RAG stage 2)",
                settings.mmr,
                disabled=settings.rerank,
                help="Select the top-k balancing relevance against novelty (fewer "
                "near-duplicates). No extra LLM call. Disabled while reranking is on.",
            )
            if settings.mmr and not settings.rerank:
                settings.mmr_candidates = st.slider(
                    "Candidates to diversify",
                    settings.top_k,
                    20,
                    settings.mmr_candidates,
                )
                settings.mmr_lambda = st.slider(
                    "Relevance ↔ diversity (λ)",
                    0.0,
                    1.0,
                    settings.mmr_lambda,
                    step=0.05,
                )
            settings.filter_difficulty = st.toggle(
                "Filter chunks by learner level", settings.filter_difficulty
            )
            _kb_filter_controls(settings)
            settings.compare_no_rag = st.toggle(
                "🆚 Compare with no-RAG answer", settings.compare_no_rag
            )
            st.text_area(
                "Extra system-prompt instructions",
                key="extra_instructions",
                placeholder="e.g. Always include a physics analogy.",
            )
            st.caption("Full pipeline trace of the last question: **🧪 Experiments**.")


def _render_message(msg: dict[str, Any]) -> None:
    """Render one history message: its text, plus (for assistant turns) the detail expanders."""
    st.markdown(msg["content"])
    if msg["role"] != "assistant":
        return
    if msg.get("no_rag"):
        with st.expander("🆚 Same question WITHOUT RAG (no retrieval, no tools)"):
            st.markdown(msg["no_rag"])
    if msg.get("sources"):
        with st.expander(f"📎 Sources ({len(msg['sources'])})"):
            for src_ in msg["sources"]:
                st.markdown(
                    f"**[{src_['ref']}] {src_.get('title', src_.get('source'))}** — "
                    f"{src_.get('topic', '')} · {src_.get('difficulty', '')} · "
                    f"subject: {src_.get('subject', '?')} · score {src_['score']}"
                )
    if msg.get("tools"):
        with st.expander(f"🛠️ Tool calls ({len(msg['tools'])})"):
            for tool_event in msg["tools"]:
                st.markdown(f"**{tool_event['name']}** `{tool_event['args']}`")
                st.markdown(tool_event["output"])
    if msg.get("meta"):
        meta = msg["meta"]
        st.caption(
            f"{meta['model']} · {meta['level']} level · "
            f"{meta['tokens_in']}+{meta['tokens_out']} tokens · ≈ ${meta['cost']:.5f}"
        )


@register_page("💬 AI Chat", key="chat", section=NAV_TITLE, order=20)
def render() -> None:
    """Render chat history, source/tool expanders, starter questions, and re-ask buttons."""
    ss = st.session_state
    ctx = ss.ctx
    level = ctx.get("level", "Beginner")

    col_settings, col_export, col_hint = st.columns([1, 1, 3])
    with col_settings:
        _answer_settings()
    # ``ctx`` was built by the entry point before this page ran, so take the model back from
    # session state afterwards: the picker above may have clamped it to something reachable,
    # and a question answered further down this same run must go to the model just shown.
    ctx["model"] = ss.model
    with col_export, st.popover("⬇️ Export", width="content"):
        st.caption("Download this conversation:")
        export_controls(ss.history, key_prefix="chat")
    with col_hint:
        # The model is named here too: it is set behind a popover, and a reader comparing
        # two answers needs to know which one wrote each without reopening anything.
        st.caption(
            f"**{ss.get('subject_label', 'All subjects')}** · {level} level · "
            f"{ss.get('technique', 'Standard')} · {ss.get('response_length', 'Balanced')} · "
            f"🤖 {model_label(ss.model)}"
        )

    prompt = st.chat_input("Ask about machine learning, deep learning, or AI…")
    # A question can arrive queued (a lesson/example/re-ask button set ``pending``) or freshly
    # typed. Resolve it before drawing, but *answer* it after the history renders (below).
    to_answer = None
    if ss.pending:
        to_answer = ss.pending
        ss.pending = None
    elif prompt:
        to_answer = (prompt, ss.get("level", "Beginner"))

    kb_status_notice(
        "The knowledge base is empty — run `make ingest` in a terminal (or use the "
        "Knowledge Base tab) to index the documents in `data/`. Chat still works, "
        "but answers won't be grounded or cited."
    )
    if not ss.history and to_answer is None:
        st.markdown("**✨ Try one of these to get started:**")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(example, key=f"ex_{i}", width="stretch"):
                ss.pending = (example, level)
                st.rerun()

    # Draw the existing conversation first, so arriving at Chat (from a lesson, a button, or
    # the sidebar) shows the page immediately instead of hanging on a blank screen while the
    # answer is generated.
    for msg in ss.history:
        with st.chat_message(msg["role"]):
            _render_message(msg)

    # Then answer any queued/typed question in place: the user's bubble and a "thinking"
    # spinner appear at once, and the reply fills in beneath — the familiar chat feel.
    if to_answer is not None:
        question, ans_level = to_answer
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            _run_with_progress(question, ans_level, ctx)
            _render_message(ss.history[-1])

    last = ss.history[-1] if ss.history else None
    if last and last["role"] == "assistant" and last.get("question"):
        idx = LEVELS.index(last["meta"]["level"])
        col_simpler, col_deeper, _ = st.columns([1, 1, 3])
        if idx > 0 and col_simpler.button(f"🪄 Simpler ({LEVELS[idx - 1]})"):
            ss.pending = (last["question"], LEVELS[idx - 1])
            st.rerun()
        if idx < len(LEVELS) - 1 and col_deeper.button(f"🔬 Deeper ({LEVELS[idx + 1]})"):
            ss.pending = (last["question"], LEVELS[idx + 1])
            st.rerun()
