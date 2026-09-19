"""XSS-guardrail tests for the HTML card builders rendered via ``unsafe_allow_html``.

The card helpers all pass hardcoded constants today, so there is no live XSS. These tests
pin the guardrail so a future regression — feeding a live paper title, model text, or URL
into a card — cannot silently reintroduce an injection hole. Each builder is a pure function
returning a string, so no Streamlit runtime is needed.
"""

from __future__ import annotations

from src.ui.pages.ai_news import _source_card_html
from src.ui.pages.resources import _resource_card_html

_XSS_TITLE = "<script>alert(1)</script>"
_XSS_URL = 'javascript:alert(1)"><script>'
_XSS_BLURB = "a & b < c > d"


def _assert_escaped(markup: str) -> None:
    # The raw payload must not appear verbatim; escaped entities must.
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert "&amp;" in markup  # the "&" in the blurb is escaped


def test_resource_card_escapes_all_fields() -> None:
    _assert_escaped(_resource_card_html(_XSS_TITLE, _XSS_URL, _XSS_BLURB))


def test_source_card_escapes_all_fields() -> None:
    _assert_escaped(_source_card_html(_XSS_TITLE, _XSS_URL, _XSS_BLURB))


def test_card_preserves_benign_content() -> None:
    markup = _resource_card_html("StatQuest", "https://example.com", "Great intuition.")
    assert "StatQuest" in markup and "https://example.com" in markup
