"""Tool playground: run each LLM tool standalone (paper search, token cost, SymPy maths).

Rendered as a section inside the **🧪 Experiments** workspace rather than as its own
navigation page — call :func:`render_tools_playground` from there.

The paper-search panel is the flagship: it searches the **curated knowledge base** and
**live arXiv** together, and it builds a structured arXiv query from the fields you fill in
(rather than passing a raw sentence, which arXiv ranks poorly — see
:func:`src.tools.arxiv.build_arxiv_query`).
"""

from __future__ import annotations

import re

import streamlit as st

from src.config import MODEL_BY_ID, MODEL_PRICES
from src.rag.retriever import KnowledgeBase, RetrievedChunk
from src.tools import estimate_tokens_and_cost, math_calculator, search_arxiv
from src.tools.arxiv import ARXIV_CATEGORIES, build_arxiv_query
from src.ui.state import load_kb


def _model_label(model_id: str) -> str:
    """Friendly ``Label · id`` for a model picker, falling back to the raw id."""
    spec = MODEL_BY_ID.get(model_id)
    return f"{spec.label} ({model_id})" if spec else model_id


def render_tools_playground() -> None:
    """Render the standalone tool playground (paper search, token cost, SymPy maths)."""
    st.caption("The assistant calls these automatically in chat; try them standalone here.")
    _render_paper_search(load_kb())
    _render_token_estimator()
    _render_calculator()


def _render_paper_search(kb: KnowledgeBase | None) -> None:
    """Unified paper search: curated KB passages + live arXiv, from structured fields."""
    with st.expander("📄 Paper search — your knowledge base + live arXiv", expanded=True):
        st.caption(
            "The arXiv **tool call**, standalone: it builds a structured query from these "
            "fields and shows your knowledge base beside live arXiv. Put a person in "
            "**Author** (e.g. `Ng`, `Vaswani`) — a name typed as a keyword matches unrelated "
            "papers. To just browse the latest papers, see **📰 AI News**."
        )
        left, right = st.columns(2)
        with left:
            keywords = st.text_input(
                "Topic / keywords",
                key="arxiv_kw",
                placeholder="retrieval augmented generation",
            )
            author = st.text_input("Author", key="arxiv_au", placeholder="Lewis")
        with right:
            cat_label = st.selectbox("arXiv category", list(ARXIV_CATEGORIES), key="arxiv_cat")
            sort = st.radio(
                "Sort arXiv by", ["relevance", "recent"], key="arxiv_sort", horizontal=True
            )
        n_results = st.slider("Max arXiv results", 1, 10, 5, key="arxiv_n")

        if not st.button("🔍 Search papers", key="arxiv_go"):
            return
        query = build_arxiv_query(
            keywords=keywords, author=author, category=ARXIV_CATEGORIES[cat_label]
        )
        if not query:
            st.warning("Enter a topic or author, or pick a category, to search.")
            return

        kb_col, arxiv_col = st.columns(2)
        with kb_col:
            st.markdown("**📚 From your knowledge base**")
            # KB notes are indexed by topic, not author — searching them for a person's name
            # returns noise, so the KB column follows the Topic field only. Author → arXiv.
            _render_kb_hits(kb, keywords)
        with arxiv_col:
            st.markdown("**🌐 Live from arXiv**")
            st.caption(f"Query: `{query}`")
            with st.spinner("Searching arXiv…"):
                try:
                    result = search_arxiv.invoke(
                        {"query": query, "max_results": n_results, "sort": sort}
                    )
                except Exception as exc:
                    st.error(f"arXiv search failed: {exc}")
                    return
            st.markdown(result)


# A KB "match" needs genuine *semantic* relevance, not incidental keyword overlap. An
# off-domain query (e.g. the physics topic "spin chain metallic surfaces") can score highly
# on BM25 alone — its words ("chain", "surfaces") coincide with ML notes (chain rule, loss
# *surfaces*) while the vector similarity stays ~0. Gating on the blended hybrid score let
# that noise through (it read ~0.30, all from BM25); gating on the *vector* score requires the
# passage to actually be about the topic. Observed on the default embeddings: in-domain hits
# score vec ≳ 0.31, off-domain topics ≤ 0.22. Re-check if you switch embedding backends.
_KB_MIN_VECTOR_SCORE = 0.28


def _relevant_kb_hits(
    chunks: list[RetrievedChunk], *, min_vector_score: float = _KB_MIN_VECTOR_SCORE, limit: int = 4
) -> list[RetrievedChunk]:
    """Keep only chunks with real semantic relevance to the topic (pure, so it is testable).

    Filters on ``vector_score`` rather than the blended ``score`` so a keyword-only coincidence
    (high BM25, ~0 vector — an off-domain query sharing a word with the notes) is rejected.
    """
    return [c for c in chunks if c.vector_score >= min_vector_score][:limit]


def _render_kb_hits(kb: KnowledgeBase | None, topic: str) -> None:
    """List curated KB passages relevant to ``topic`` (hybrid search), with citations."""
    if kb is None:
        st.info("Knowledge base not loaded — run `make ingest` to enable KB search.")
        return
    if not topic.strip():
        st.caption(
            "Your notes are indexed by **topic**, not author — add a topic or keyword above "
            "to search them. An author-only search runs on arXiv (right) →"
        )
        return
    hits = _relevant_kb_hits(kb.search(topic, k=6))
    if not hits:
        st.caption(
            f"No passages in your knowledge base closely match “{topic}”. "
            "Try different keywords, or see the arXiv results on the right."
        )
        return
    for chunk in hits:
        st.markdown(_kb_hit_markdown(chunk))


# LaTeX commands (``\frac``, ``\hat``, ``\mathbb``, ``\qquad`` …) — dropped whole, since the
# leftover braces/letters read as gibberish. Applied before the control-char strip below.
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+\*?")
# Markdown/LaTeX control characters that make a raw KB passage render as a heading, bold
# text, math, code or a table when dropped into ``st.markdown``. Study notes are ``.md``
# files full of these — stripped from previews so a snippet reads as plain prose.
_MD_CONTROL = re.compile(r"[#*_`$~<>\[\]{}\\^|]")


def _plain(text: str) -> str:
    """Strip markdown + LaTeX so KB text renders as plain prose, not a heading or raw math."""
    text = _LATEX_COMMAND.sub(" ", text)
    text = _MD_CONTROL.sub("", text)
    return " ".join(text.split())


def _kb_hit_markdown(chunk: RetrievedChunk) -> str:
    """Render one KB hit as a markdown bullet: linked title, source, score, and a snippet."""
    meta = chunk.metadata
    source = meta.get("source", "?")
    title = _plain(meta.get("title") or meta.get("topic") or source).strip() or source
    url = meta.get("url") or meta.get("origin") or ""
    head = f"[{title}]({url})" if url else title
    snippet = _plain(chunk.text)[:220]
    return f"- **{head}** · `{source}` · score {chunk.score:.2f}  \n  {snippet}…"


def _render_token_estimator() -> None:
    """Standalone token & cost estimator for a chosen model's prices."""
    with st.expander("🔢 Token & cost estimator"):
        text = st.text_area("Text to measure", key="tok_text")
        price_model = st.selectbox(
            "Model", list(MODEL_PRICES), key="tok_model", format_func=_model_label
        )
        if st.button("Estimate", key="tok_go") and text:
            with st.spinner("Estimating…"):
                st.markdown(estimate_tokens_and_cost.invoke({"text": text, "model": price_model}))


def _render_calculator() -> None:
    """Standalone SymPy calculator (runs in the sandboxed, time-limited tool)."""
    with st.expander("🧮 Math & stats calculator (SymPy)"):
        expression = st.text_input(
            "Expression",
            key="math_expr",
            placeholder="integrate(exp(-x**2), (x, -oo, oo))",
        )
        operation = st.selectbox("Operation", ["evaluate", "simplify", "solve"], key="math_op")
        if st.button("Compute", key="math_go") and expression:
            with st.spinner("Computing…"):
                st.markdown(
                    math_calculator.invoke({"expression": expression, "operation": operation})
                )
