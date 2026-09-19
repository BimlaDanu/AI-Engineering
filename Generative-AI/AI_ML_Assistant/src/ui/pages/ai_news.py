"""📰 AI News workspace: live arXiv papers plus a curated directory of AI news sources.

Four tabs, from most to least "live":

* **Papers** — a live arXiv feed (relevance search or newest-first topic browsing).
* **Labs** — official newsrooms of the major AI labs (model releases, research, products).
* **Digests** — editorial aggregators and newsletters that summarise the day's AI news.
* **Voices** — high-signal researchers and founders to follow on X/Twitter.

Only the Papers tab hits the network (via :func:`src.tools.search_arxiv`); the other three
are curated link directories, so they always render offline and add no dependency. Each
directory entry is ``(title, url, blurb)`` — extend the lists below to add a source. A live
RSS-backed feed for the newsrooms is a natural future extension (it would need a feed parser
and network access to new hosts, so it is intentionally left out here).
"""

from __future__ import annotations

import html

import streamlit as st

from src.tools import search_arxiv
from src.ui.registry import register_page

# Preset topic buttons -> the arXiv query each fires.
_PRESETS: dict[str, str] = {
    "🧮 Machine learning": "machine learning",
    "🧠 Deep learning": "deep learning neural networks",
    "🔍 RAG": "retrieval augmented generation",
    "🤖 LLMs": "large language models",
    "👁️ Computer vision": "computer vision",
    "🖼️ Multimodal": "vision language models multimodal",
    "🗣️ NLP": "natural language processing",
    "🕹️ Agents & fine-tuning": "LLM agents tool use fine-tuning LoRA RLHF",
    "🌫️ Diffusion / GenAI": "diffusion models generative",
}

# Official newsrooms of the major AI labs: (title, url, blurb).
LABS: list[tuple[str, str, str]] = [
    (
        "OpenAI — News",
        "https://openai.com/news/",
        "Official announcements: GPT model releases, research, and product updates.",
    ),
    (
        "Anthropic — News",
        "https://www.anthropic.com/news",
        "Claude model releases, interpretability and safety research, and product news.",
    ),
    (
        "Google DeepMind — Blog",
        "https://deepmind.google/discover/blog/",
        "Gemini, AlphaFold, and frontier research from Google's combined AI lab.",
    ),
    (
        "Meta AI — Blog",
        "https://ai.meta.com/blog/",
        "Llama open-weight models and open research from Meta's FAIR / GenAI teams.",
    ),
    (
        "Mistral AI — News",
        "https://mistral.ai/news/",
        "Efficient open-weight model releases and updates from Mistral AI.",
    ),
    (
        "Hugging Face — Blog",
        "https://huggingface.co/blog",
        "Open-source models, datasets, and hands-on ML-engineering write-ups.",
    ),
]

# Editorial aggregators and newsletters that curate the day's/week's AI news.
DIGESTS: list[tuple[str, str, str]] = [
    (
        "The Batch — DeepLearning.AI",
        "https://www.deeplearning.ai/the-batch/",
        "Andrew Ng's weekly, curated AI-news digest — reliable signal over noise.",
    ),
    (
        "Import AI — Jack Clark",
        "https://importai.substack.com/",
        "Weekly deep dive on AI research, capabilities, and policy.",
    ),
    (
        "Readless — AI News Hub",
        "https://www.readless.ai/",
        "Editorial hub that summarises top AI stories and surfaces new tools.",
    ),
    (
        "Particle",
        "https://www.particle.news/",
        "Multi-source news summaries with balanced framing (strong tech/AI coverage).",
    ),
    (
        "Ground News",
        "https://ground.news/",
        "Compare how outlets across the spectrum cover the same AI story.",
    ),
    (
        "TLDR AI",
        "https://tldr.tech/ai",
        "Daily newsletter with concise summaries of the top AI news, tools, and papers.",
    ),
]

# High-signal accounts to follow on X/Twitter: (handle, url, blurb).
VOICES: list[tuple[str, str, str]] = [
    (
        "@OpenAI",
        "https://x.com/OpenAI",
        "Official account of the creators of ChatGPT — model and product rollouts.",
    ),
    (
        "@AnthropicAI",
        "https://x.com/AnthropicAI",
        "Official Anthropic account — Claude releases and safety research.",
    ),
    (
        "@sama",
        "https://x.com/sama",
        "Sam Altman, OpenAI CEO — product direction and announcements.",
    ),
    (
        "@AndrewNg",
        "https://x.com/AndrewNg",
        "Curated, opinionated insights on applied AI and the state of the industry.",
    ),
    (
        "@rowancheung",
        "https://x.com/rowancheung",
        "Daily summaries of the most important AI news and model releases.",
    ),
    (
        "@_akhaliq",
        "https://x.com/_akhaliq",
        "Fastest tracking of new research papers and arXiv releases.",
    ),
    (
        "@karpathy",
        "https://x.com/karpathy",
        "Andrej Karpathy — first-principles takes on LLMs, training, and tooling.",
    ),
    (
        "@ylecun",
        "https://x.com/ylecun",
        "Yann LeCun, Meta's chief AI scientist — research directions and field debates.",
    ),
]


def _source_card_html(title: str, url: str, blurb: str) -> str:
    """Build one source card's HTML, escaping every interpolated value (XSS guardrail)."""
    return (
        f'<div class="resource-card"><h4>'
        f'<a href="{html.escape(url, quote=True)}" target="_blank">{html.escape(title)}</a>'
        f"</h4><p>{html.escape(blurb)}</p></div>"
    )


def _source_card(title: str, url: str, blurb: str) -> None:
    """Render one styled, clickable source card (shared card style with Resources)."""
    st.markdown(_source_card_html(title, url, blurb), unsafe_allow_html=True)


def _card_grid(sources: list[tuple[str, str, str]]) -> None:
    """Lay out a list of ``(title, url, blurb)`` sources as a two-column card grid."""
    col_left, col_right = st.columns(2)
    for i, (title, url, blurb) in enumerate(sources):
        with col_left if i % 2 == 0 else col_right:
            _source_card(title, url, blurb)


def _fetch(query: str, sort: str) -> None:
    """Run an arXiv search and stash ``(query, markdown)`` in session state."""
    with st.spinner("Searching arXiv…"):
        try:
            st.session_state.arxiv_results = (
                query,
                search_arxiv.invoke({"query": query, "max_results": 8, "sort": sort}),
            )
        except Exception as exc:  # network/tool failure must surface, not crash the page
            st.error(f"arXiv search failed: {exc}")


def _render_papers() -> None:
    """The live arXiv feed: relevance search or newest-first topic browsing."""
    st.caption(
        "Browse or search the latest arXiv papers — relevance for finding one, the topic "
        "buttons for the newest. To see the arXiv **tool call** itself (structured query + "
        "KB comparison), use 🧪 Experiments."
    )

    sort_label = st.radio(
        "Ordering",
        ["Most relevant", "Most recent"],
        horizontal=True,
        help="Relevance is best for finding a paper; recent is best for a news feed.",
    )
    sort = "recent" if sort_label == "Most recent" else "relevance"

    query = st.text_input(
        "Search arXiv", placeholder="e.g. machine learning and the physical sciences"
    )
    if st.button("🔍 Search", type="primary") and query:
        _fetch(query, sort)

    st.markdown("**Or pick a topic:**")
    cols = st.columns(3)
    for i, (label, preset_query) in enumerate(_PRESETS.items()):
        if cols[i % 3].button(label, width="stretch"):
            _fetch(preset_query, sort)

    if st.session_state.arxiv_results:
        topic, results = st.session_state.arxiv_results
        st.markdown(f"### Results for _{topic}_")
        st.markdown(results)
    else:
        st.info("Search above or tap a topic to pull papers from arXiv.")


@register_page("📰 AI News", key="ai_news", section="Knowledge", order=70)
def render() -> None:
    """AI News tab: a live arXiv feed plus curated newsrooms, digests, and voices to follow."""
    st.subheader("📰 AI News")
    st.caption(
        "Stay **current** — live arXiv papers plus curated newsrooms, digests, and voices to "
        "follow. For evergreen courses and references, see **📚 Stacks**."
    )

    papers_tab, labs_tab, digests_tab, voices_tab = st.tabs(
        ["📄 Papers", "🏢 Labs", "📊 Digests", "🐦 Voices"]
    )
    with papers_tab:
        _render_papers()
    with labs_tab:
        st.caption("Official newsrooms — go straight to the source for model releases.")
        _card_grid(LABS)
    with digests_tab:
        st.caption("Editorial aggregators and newsletters that do the filtering for you.")
        _card_grid(DIGESTS)
    with voices_tab:
        st.caption("High-signal researchers and founders to follow on X/Twitter.")
        _card_grid(VOICES)
