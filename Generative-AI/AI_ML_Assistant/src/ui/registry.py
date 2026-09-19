"""A tiny registry that lets workspace pages plug themselves into the app.

Each page module decorates its render function with :func:`register_page`. The entry point
(``app.py``) calls :func:`get_pages` to build the flat, ordered sidebar navigation, while the
🏠 Home dashboard calls :func:`get_sections` to render the same pages *grouped* by ``section``.
Adding, removing, or reordering a workspace touches only its own module — both the sidebar and
the Home overview follow automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.ui.theme import NAV_TITLE

# Top-to-bottom order of the sidebar's groups, and of the 🏠 Home overview's card blocks.
# Sections not listed here are appended alphabetically after these, so a new phase can
# introduce a group without editing this list (though listing it here pins its position).
#
# The shape is the one every chat application has settled on: what you came to do at the top,
# the machinery underneath, the knobs at the bottom. The first group carries the product's
# descriptive name and holds only the two pages a first-time visitor needs — everything else
# is grouped by what a reader is trying to *do*, not by which phase of the project built it.
SECTION_ORDER: list[str] = [
    NAV_TITLE,  # 🏠 Home, 💬 AI Chat
    "Learn",  # 🎓 Tutor, 🔬 Lab, 🎯 Trivia
    "Analyse",  # 📈 Analytics, 📊 Evaluation, 🆚 A/B testing, 🧪 Experiments
    "Knowledge",  # 📰 AI News, 📚 Stacks, 📄 Knowledge Base
    "System",  # ⚙️ Settings
]


@dataclass(frozen=True)
class Page:
    """A registered workspace: its nav label, render callback, section, and sort order."""

    key: str
    label: str
    render: Callable[[], None]
    section: str = "Workspaces"
    order: int = 100


_PAGES: list[Page] = []


def register_page(
    label: str,
    *,
    key: str | None = None,
    section: str = "Workspaces",
    order: int = 100,
) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Decorator: register a no-argument render function as a navigation page.

    Args:
        label: Nav label shown in the sidebar (may include a leading emoji).
        key: Stable identifier / URL path; defaults to the function's name.
        section: Sidebar group heading; ordered by :data:`SECTION_ORDER`.
        order: Ascending sort key *within* the section (lower appears first).
    """

    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        _PAGES.append(
            Page(key=key or fn.__name__, label=label, render=fn, section=section, order=order)
        )
        return fn

    return decorator


def get_pages() -> list[Page]:
    """Return all registered pages sorted by ``order`` then ``label``."""
    return sorted(_PAGES, key=lambda p: (p.order, p.label))


def _section_rank(section: str) -> tuple[int, str]:
    """Sort key placing known sections in :data:`SECTION_ORDER`, unknown ones after."""
    if section in SECTION_ORDER:
        return (SECTION_ORDER.index(section), "")
    return (len(SECTION_ORDER), section)


def get_sections() -> dict[str, list[Page]]:
    """Group registered pages into ``{section: [pages sorted by order]}``.

    Sections are ordered by :data:`SECTION_ORDER` (then alphabetically for any extra); the
    🏠 Home dashboard renders this as grouped cards. (The sidebar itself is a flat list built
    from :func:`get_pages`, so it stays uncluttered — Home is the grouped-by-area overview.)
    """
    sections: dict[str, list[Page]] = {}
    for page in get_pages():
        sections.setdefault(page.section, []).append(page)
    return {name: sections[name] for name in sorted(sections, key=_section_rank)}


# --- Programmatic cross-page navigation ----------------------------------------------------
# ``st.navigation`` identifies pages by the ``st.Page`` objects passed to it, and
# ``st.switch_page`` needs that same object to jump. The entry point builds those objects
# fresh each run and hands them here so any page's render function can navigate to another by
# its stable ``key`` (e.g. the 🎓 AI/ML Tutor's "Practise" button opening the 🔬 AI/ML Lab).
_NAV_PAGES: dict[str, Any] = {}


def remember_nav_pages(pages: dict[str, Any]) -> None:
    """Record ``{key: st.Page}`` for this run so :func:`switch_to_page` can resolve a key."""
    _NAV_PAGES.clear()
    _NAV_PAGES.update(pages)


def switch_to_page(key: str) -> bool:
    """Navigate to the registered page with ``key`` via ``st.switch_page``.

    Returns True if the jump was issued (the call reruns the app on success), False if the
    key is unknown — callers can then fall back to an in-place message instead of crashing.
    """
    page = _NAV_PAGES.get(key)
    if page is None:
        return False
    import streamlit as st

    st.switch_page(page)
    return True


def go_to_page(key: str, hint: str, *, level: str = "info") -> None:
    """Jump to the page registered as ``key``, or say where to find it if it is not there.

    Every cross-page button in ``src/ui/pages/`` needs the same two steps: attempt the jump,
    and on failure name the sidebar entry to open instead. Spelled out at each call site that
    was a nested ``if`` whose inner branch read like the normal path rather than the fallback,
    and the same two lines were repeated in five places.

    Args:
        key: The registry key of the destination page.
        hint: What to show when the destination is not registered. Name the sidebar entry —
            a reader who cannot be moved there needs to know where to go themselves.
        level: ``"info"`` (the default) or ``"warning"``, for a fallback worth more attention.
    """
    if switch_to_page(key):
        return
    import streamlit as st

    if level == "warning":
        st.warning(hint)
    else:
        st.info(hint)
