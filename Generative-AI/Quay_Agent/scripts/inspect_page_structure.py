"""Render every page of the interface and print what each one actually draws.

A page audit that reads the source tells you what the code says. This one runs the
pages and reports what came out, which is a different list: a heading inside a
branch nobody takes does not appear, a table that failed to build shows as an error
rather than as a table, and a page whose panels all landed in an ``st.info``
placeholder looks empty here even though its source is long.

Written for the question *does each page carry its weight, and does any of it say
the same thing twice* -- so the output is deliberately structural. Headings in
order, buttons by label, and counts of the element types that carry content. It is
not a substitute for :mod:`tests.test_ui_pages`, which asserts; this only reports.

Run with ``uv run python scripts/inspect_page_structure.py``. It calls no model and
needs no credential: every page is driven with the language-model switch held off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from src.ui import panels
from src.ui import setting as knob

RENDER_BUDGET_S = 120.0
"""Seconds a single page may take to render before the driver gives up.

Generous on purpose. Some pages solve a circuit or diagonalise a matrix while they
draw, and a budget tuned to the fastest page would report a slow one as broken.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root. ``AppTest`` resolves a relative path against its *caller*, so
every path handed to it here is made absolute first -- a lesson this script learnt
by looking for the pages inside a temporary directory."""


def drive(module: str) -> dict[str, Any]:
    """Render one page with no model available and collect what it drew.

    Args:
        module: The page's path relative to ``src/ui``, as held in
            :data:`src.ui.panels.PAGES`.

    Returns:
        A structural summary: any exception, the headings in order, the button
        labels, and counts of the content-bearing element types.
    """
    app = AppTest.from_file(
        str(PROJECT_ROOT / "src" / "ui" / module), default_timeout=RENDER_BUDGET_S
    )
    # Held off explicitly rather than trusting the default. `default_offline` opens
    # the switch by asking whether configuration loads, so a checkout that *has* a
    # credential would otherwise make this script call a real gateway nine times.
    app.session_state[panels.SETTING_KEY] = knob.Setting().with_model(offline=True)
    app.run()

    headings: list[str] = []
    for block in app.markdown:
        for line in str(block.value).splitlines():
            if line.startswith("#"):
                headings.append(line.strip())

    return {
        "module": module,
        "exception": [f"{item.type}: {item.message}" for item in app.exception],
        "headings": headings,
        "buttons": [str(item.label) for item in app.button],
        "markdown": len(app.markdown),
        "dataframes": len(app.dataframe),
        "infos": len(app.info),
        "errors": [str(item.value) for item in app.error],
        "warnings": [str(item.value) for item in app.warning],
    }


def main() -> None:
    """Render every page in the navigation and print the inventory."""
    for page in panels.PAGES:
        found = drive(page.module)
        print(f"\n=== {page.icon} {page.label}  ({found['module']})")
        print(
            f"    markdown={found['markdown']} dataframes={found['dataframes']} "
            f"infos={found['infos']} exceptions={len(found['exception'])}"
        )
        for label, items in (
            ("EXCEPTION", found["exception"]),
            ("ERROR", found["errors"]),
            ("WARNING", found["warnings"]),
        ):
            for item in items:
                print(f"    {label}: {item[:200]}")
        if found["buttons"]:
            print(f"    buttons: {found['buttons']}")
        for heading in found["headings"]:
            print(f"      {heading[:120]}")


if __name__ == "__main__":
    main()
