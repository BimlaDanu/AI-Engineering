"""Offline tests for the declared theme in ``.streamlit/config.toml``.

Declaring a theme has a side effect that is easy to undo by accident. Streamlit discards its
three built-in themes as soon as an app declares one of its own, and it renders the
System / Light / Dark control in the ⋮ menu only while **more than one** theme is on the list.
A single ``[theme]`` block therefore leaves exactly one, and the control vanishes: the app is
locked to whatever it declared, with no way back, including for a reader whose machine is set
the other way.

Splitting the palette into ``[theme.light]`` and ``[theme.dark]`` gives Streamlit two themes to
name (plus the "System" pairing it derives from them), which restores the control and makes
"follow the OS" the default for a reader who has never chosen. Nothing in the app reads these
values, so nothing in the app notices if they are merged back into one block — hence the test.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"

# The keys that decide a variant's surface. Present in both blocks, or the variant falls back
# to the shared [theme] section, which is the mistake this file exists to catch.
_SURFACE_KEYS = {"primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor"}


@pytest.fixture(scope="module")
def theme() -> dict:
    """The ``[theme]`` table as declared on disk, sub-tables included."""
    assert CONFIG_PATH.exists(), f"missing {CONFIG_PATH}"
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("theme", {})


def test_both_variants_are_declared(theme: dict) -> None:
    # The condition Streamlit's frontend actually tests: at least one of the two sub-themes
    # carries a value, or it builds a single "Custom Theme" and hides the appearance control.
    assert theme.get("light"), "no [theme.light] — the ⋮ appearance control will be hidden"
    assert theme.get("dark"), "no [theme.dark] — the ⋮ appearance control will be hidden"


def test_each_variant_sets_its_own_surface(theme: dict) -> None:
    for name in ("light", "dark"):
        missing = _SURFACE_KEYS - (theme.get(name) or {}).keys()
        assert not missing, f"[theme.{name}] inherits {sorted(missing)} from the shared section"


def test_the_shared_section_declares_no_colours(theme: dict) -> None:
    # A colour at the top level applies to both variants, so one of them gets a background it
    # was not designed for. Shared settings are for what a surface does not change, like font.
    shared = {k for k in theme if k != "base" and k.lower().endswith("color")}
    assert not shared, f"{sorted(shared)} in [theme] leaks across both variants"


def test_the_two_variants_do_not_share_a_background(theme: dict) -> None:
    light, dark = theme.get("light") or {}, theme.get("dark") or {}
    assert light.get("backgroundColor") != dark.get("backgroundColor"), "one background for both"
    assert light.get("textColor") != dark.get("textColor"), "one text colour for both"
