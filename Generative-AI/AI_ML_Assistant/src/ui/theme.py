"""Shared visual theme: the sidebar brand mark, injected CSS, hero banner, and footer.

The app's single identity is **Synapse** — *where AI, ML & deep learning connect*. The name
appears as the vector brand mark at the top of the sidebar (:data:`SYNAPSE_LOGO`, rendered via
:func:`render_sidebar_brand`), as the hero kicker, and nowhere as a competing second name.
"""

from __future__ import annotations

import streamlit as st

from src.ui.account import TOP_BAR_NAME

# Sidebar brand glyph: a synapse — a presynaptic terminal firing neurotransmitters across the
# gap to a receiving dendrite — in the same blue→violet→pink gradient as the hero so it reads
# on both light and dark sidebars. Glyph only (no wordmark): the navigation heading carries the
# "Synapse" name, so this stays a clean icon above it. Rendered by ``st.logo``, which accepts a
# raw ``<svg …>`` string and places it above the navigation; if a viewer's build declines to
# render the inline SVG, the "🧠 Synapse" nav heading still shows the name.
SYNAPSE_LOGO = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 58 46" width="58" height="46" '
    'role="img" aria-label="Synapse">'
    '<defs><linearGradient id="syn" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#3b82f6"/>'
    '<stop offset="0.55" stop-color="#7c3aed"/>'
    '<stop offset="1" stop-color="#db2777"/></linearGradient></defs>'
    '<g stroke="url(#syn)" stroke-width="2.4" fill="none" stroke-linecap="round">'
    '<path d="M4 23 L12 23"/>'
    '<path d="M40 23 L47 23 M47 23 L52 18 M47 23 L52 28"/></g>'
    '<circle cx="16" cy="23" r="5.6" fill="url(#syn)"/>'
    '<circle cx="39" cy="23" r="4.2" fill="url(#syn)"/>'
    '<circle cx="24" cy="20.5" r="1.7" fill="url(#syn)"/>'
    '<circle cx="28.5" cy="24.5" r="1.4" fill="url(#syn)"/>'
    '<circle cx="32" cy="20" r="1.1" fill="url(#syn)"/>'
    "</svg>"
)

# The navigation heading — a plain, descriptive title at the top of the sidebar on every page
# (rendered by ``st.navigation`` itself, so it never depends on inline-SVG support). It names
# what the tool *is* for clarity; "Synapse" as the product name lives in the hero and the glyph.
NAV_TITLE = "🧭 AI/ML Research Assistant"

CSS = """
<style>
/* One place the accent is decided, so a colour change is a change to this block rather than
   a hunt through the rules below. Every fill and line is the accent at an alpha, never a
   flat hex: alpha composites over whatever background is actually behind it, so the same
   token reads on either of the two bases declared in .streamlit/config.toml, whichever one
   the reader's machine or the ⋮ menu has selected. Flat light-mode washes are what the
   earlier stylesheet used, and they all but vanished against a dark page. */
:root {
    --syn-accent: #8b5cf6;
    --syn-fill: rgba(139, 92, 246, 0.13);
    --syn-fill-strong: rgba(139, 92, 246, 0.26);
    --syn-line: rgba(139, 92, 246, 0.32);
    --syn-gradient: linear-gradient(120deg, #1e3a8a, #7c3aed 55%, #db2777);
}
/* Push content clear of Streamlit's fixed top toolbar so nothing is clipped behind it. */
.block-container { padding-top: 3.2rem; }
.hero {
    background: var(--syn-gradient);
    color: #fff; padding: 1.1rem 1.6rem; border-radius: 14px; margin-bottom: 0.9rem;
}
.hero .kicker {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.22em;
    text-transform: uppercase; opacity: 0.75; margin-bottom: 0.15rem;
}
.hero h1 { font-size: 1.6rem; margin: 0 0 0.3rem 0; color: #fff; }
.hero .tag {
    margin: 0; font-size: 0.95rem; font-weight: 600; letter-spacing: 0.02em;
}
.hero .tag span { opacity: 0.55; padding: 0 0.15rem; }
/* Five pages carry tab strips. The selected one used to differ by fill alpha alone — the
   difference between 0.08 and 0.22 of a translucent violet — which is a distinction that
   survives neither a dark background nor a colour-blind reader. The underline is the load-
   bearing signal now; the fill only supports it. */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: var(--syn-fill); border-radius: 10px 10px 0 0; padding: 8px 16px;
    border-bottom: 3px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--syn-fill-strong); }
.stTabs [aria-selected="true"] {
    background: var(--syn-fill-strong); font-weight: 600;
    border-bottom-color: var(--syn-accent);
}
.resource-card {
    border: 1px solid var(--syn-line); border-left: 5px solid var(--syn-accent);
    border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.7rem;
    background: var(--syn-fill);
}
.resource-card h4 { margin: 0 0 0.25rem 0; font-size: 1rem; }
.resource-card p { margin: 0; font-size: 0.88rem; opacity: 0.85; }
.sb-status {
    display: flex; justify-content: space-between; gap: 0.5rem;
    font-size: 0.82rem; padding: 0.12rem 0; line-height: 1.35;
}
.sb-status span:first-child { opacity: 0.7; }
.sb-status span:last-child { font-weight: 600; text-align: right; }
/* AI/ML Lab topic selector: force the four topics into a tidy 2×2 grid (two rows, not three). */
.st-key-lab_topic_row div[role="radiogroup"] { flex-wrap: wrap; row-gap: 0.3rem; }
.st-key-lab_topic_row div[role="radiogroup"] > label { flex: 0 0 47%; }
/* Sidebar footer: a pulsing synapse "firing" beside the tagline — the app's small signature. */
@keyframes syn-pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50%      { transform: scale(1.55); opacity: 1; }
}
.sb-foot {
    text-align: center; margin-top: 1.4rem; padding-top: 0.85rem;
    border-top: 1px solid var(--syn-line);
}
.sb-foot .dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;
    vertical-align: middle;
    background: linear-gradient(120deg, #3b82f6, #7c3aed, #db2777);
    animation: syn-pulse 2.4s ease-in-out infinite;
}
.sb-foot .line1 { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.03em; opacity: 0.9; }
.sb-foot .line2 { font-size: 0.7rem; opacity: 0.6; margin-top: 0.15rem; line-height: 1.3; }
</style>
"""


TOP_BAR_CSS = f"""
<style>
/* Lift the account controls out of the page body and onto Streamlit's header strip, to the
   left of its own Deploy button.

   They belong beside Deploy rather than in the page. Right-aligned inside a wide content
   column they landed a long way in from the window's edge, which reads as the middle of the
   screen rather than as a corner — and a control in the middle of a page is a control
   competing with the answer.

   Fixed rather than floated, because the header strip is not this element's parent and no
   amount of margin will move it there. Out of the flow, so the hero banner it used to sit
   above closes up behind it.

   The right offset has to clear everything Streamlit puts in that strip, and the list is
   longer than Deploy and the hamburger menu: while a script is running a **Stop** button
   appears to their left, and that is the one neighbour this bar must never cover — a reader
   watching a long retrieval has to be able to stop it. 17rem clears the group with a margin
   wide enough that the two sets of controls do not scan as one toolbar with an arbitrary
   break in it.

   It is a measured value rather than a computed one, because the header is Streamlit's and
   its contents are not ours to interrogate. If a future version widens that toolbar, this is
   the one number to change — and the thing to check is the whole strip at a wide window, not
   Deploy alone.

   Below 900px the header has no room to share, so the rule is dropped and the controls fall
   back into the page, where they still fit. */
@media (min-width: 900px) {{
    .st-key-{TOP_BAR_NAME} {{
        position: fixed;
        top: 0.55rem;
        right: 17rem;
        z-index: 999991;
        width: auto;
        margin: 0;
    }}
    /* With the bar up there, close the gap it left behind. The body is padded clear of the
       header strip by default, which is right when the first thing in the body is content —
       but here the strip is where this app's own controls now live, so the page would open
       with a band of nothing above the hero. */
    .block-container {{ padding-top: 1.4rem; }}
}}
/* The buttons sit on the header's own background, so they should read as part of it rather
   than as three cards dropped onto it. Cosmetic only: if a future Streamlit renames the
   test id, the rule stops applying and the bar renders plainly, which is how decoration
   ought to fail. */
.st-key-{TOP_BAR_NAME} [data-testid="stBaseButton-secondary"] {{
    border: 1px solid var(--syn-line);
    background: transparent;
}}
.st-key-{TOP_BAR_NAME} [data-testid="stBaseButton-secondary"]:hover {{
    border-color: var(--syn-accent);
}}
</style>
"""
"""Where the top-bar controls sit, and why it takes a stylesheet to put them there.

Separate from :data:`CSS` because that one only paints and this one *moves* an element. A
rule that repositions something can break a page; a rule that rounds a corner cannot, and
keeping them apart means the risky stylesheet stays short and reviewable.
"""


def render_sidebar_brand() -> None:
    """Render the Synapse brand mark at the very top of the sidebar (above the navigation).

    ``st.logo`` always paints in the upper-left, so this is order-independent — it does not
    matter where in the run it is called relative to the navigation widget.
    """
    st.logo(SYNAPSE_LOGO, size="large")


def render_sidebar_footer() -> None:
    """Render the small sidebar signature: a pulsing synapse and the app's one-line ethos.

    Appended after the navigation, so it sits at the end of the sidebar. Purely decorative —
    it states, in one breath, what makes Synapse different: grounded, cited, and honest.
    """
    st.sidebar.markdown(
        '<div class="sb-foot">'
        '<div class="line1"><span class="dot"></span>Grounded · Cited · Honest</div>'
        '<div class="line2">Answers from the knowledge base — or an honest "I don\'t know"</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the gradient hero banner: the Synapse kicker, a short domain label, verb strip."""
    st.markdown(
        '<div class="hero">'
        '<div class="kicker">🧠 Synapse</div>'
        "<h1>AI · ML · Deep Learning</h1>"
        '<p class="tag">Research<span>•</span>Learn<span>•</span>Analyze'
        "<span>•</span>Experiment</p>"
        "</div>",
        unsafe_allow_html=True,
    )
