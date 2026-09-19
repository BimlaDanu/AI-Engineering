"""Synapse — where AI, ML & deep learning connect. Streamlit entry point.

Structure: This module only wires the pieces together.
All business logic resides in``src/core`` (framework-agnostic pipeline) and
all presentation in ``src/ui`` (a page registry plus one module per workspace).
The sidebar navigation is built from that registry via ``st.navigation``,
so a new workspace is added by dropping a module in ``src/ui/pages`` with
a ``@register_page`` decorator. Run with ``make run``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# This streamlit's hot-reload watcher walks every imported module to decide what to watch.
# ``langchain_core`` pulls in ``transformers`` at import, whose image-processor submodules
# lazily import ``torchvision`` (not installed — we use the API embedding backend), so the
# watcher logs a benign traceback per submodule. It already catches the failure and keeps
# working, so drop just that logger to ERROR to keep the console clean without disabling
# hot-reload or muting any other Streamlit logs.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:  # `streamlit run src/app.py` puts src/ (not root) on sys.path
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="Synapse — where AI, ML & deep learning connect",
    page_icon="🧠",
    layout="wide",
)

import src.ui.pages  # noqa: E402,F401  -> import registers every workspace page
from src import auth  # noqa: E402  (identity: OIDC sign-in, own-key, optional password gate)
from src.config import SUBJECTS  # noqa: E402
from src.ui.account import account_bar  # noqa: E402
from src.ui.past_chats import autosave, new_chat_button, past_chats_panel  # noqa: E402
from src.ui.registry import get_sections, remember_nav_pages  # noqa: E402
from src.ui.state import init_state  # noqa: E402
from src.ui.theme import (  # noqa: E402
    CSS,
    TOP_BAR_CSS,
    render_sidebar_brand,
    render_sidebar_footer,
)


def _build_navigation() -> st.navigation:
    """Turn the registered pages into a **grouped** sidebar ``st.navigation``.

    The sidebar used to be one flat jump-list of thirteen links under a single heading, on the
    argument that the grouped-by-area view lived on 🏠 Home. Thirteen undifferentiated rows is
    not a list anybody reads, and it put ⚙️ Settings and 💬 AI Chat at the same visual weight.
    Both surfaces now read the same :func:`~src.ui.registry.get_sections` grouping, so the
    sidebar and the Home overview cannot drift apart.

    Group order comes from :data:`~src.ui.registry.SECTION_ORDER` and page order from each
    page's ``order`` field, so adding a workspace is still a one-line decorator in its own
    module. 🏠 Home is the default landing destination.
    """
    page_objs: dict[str, st.Page] = {}
    grouped: dict[str, list[st.Page]] = {}
    for section, pages in get_sections().items():
        for page in pages:
            obj = st.Page(
                page.render,
                title=page.label,
                url_path=page.key,
                default=(page.key == "home"),
            )
            page_objs[page.key] = obj
            grouped.setdefault(section, []).append(obj)
    # Let any page navigate to another by key (e.g. Tutor → Lab); see registry.switch_to_page.
    remember_nav_pages(page_objs)
    return st.navigation(grouped, position="sidebar")


def main() -> None:
    """Wire CSS, identity, the grouped navigation sidebar, and the active workspace page."""
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(TOP_BAR_CSS, unsafe_allow_html=True)

    # Publish this session's credential choice before anything can build a model. A visitor who
    # pasted their own key behind "Use a key" has their questions billed to it; everyone else
    # falls back to the deployment's environment. Re-stated every run on purpose — the override
    # is thread-local and a worker thread may be reused. See src.config.set_runtime_key.
    auth.apply_runtime_keys()

    render_sidebar_brand()  # Synapse logo, pinned above the navigation

    # Optional password gate (dormant unless a [password_auth] secret is configured). When
    # active and nobody is signed in, the login form is the only thing shown — no workspace
    # renders, and no account bar either, since its buttons would compete with the form.
    if not auth.ensure_authenticated():
        return

    # The top bar is the first thing drawn in the body so it lands on Streamlit's header strip
    # beside Deploy (TOP_BAR_CSS lifts it there on a wide window). New chat rides in the same
    # row rather than in the sidebar, where it would sit a centimetre under the 💬 AI Chat link.
    account_bar(lead=new_chat_button)

    # The hero banner is 🏠 Home's alone (it draws its own). It used to open every page, which
    # meant a hundred-odd pixels of gradient above the conversation on 💬 AI Chat and above the
    # controls on ⚙️ Settings — branding in the place the content should start. The sidebar
    # carries the identity continuously: the Synapse glyph above the navigation, the name in
    # its heading, the signature at its foot.
    page = _build_navigation()  # renders the grouped workspace links in the sidebar

    ss = st.session_state
    ss.ctx = {
        "model": ss.model,
        "level": ss.level,
        "subjects": SUBJECTS[ss.subject_label],
    }

    page.run()

    # After the page, so the panel lists a conversation this run has just added to. Both live
    # in the sidebar on every page: a reader who asks something and then walks over to
    # 📊 Evaluation has still had that conversation and should still find it.
    autosave()
    with st.sidebar:
        past_chats_panel()
    render_sidebar_footer()  # pulsing-synapse signature at the end of the sidebar


main()
