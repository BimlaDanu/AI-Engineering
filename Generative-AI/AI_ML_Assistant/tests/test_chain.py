"""Tests for system-prompt style composition (no network, no LLM)."""

from src.config import PROMPT_TECHNIQUES, RESPONSE_LENGTHS
from src.generation import LEVEL_STYLES, compose_style


def test_default_style_is_level_only():
    assert compose_style("Beginner") == LEVEL_STYLES["Beginner"]


def test_technique_and_length_are_appended():
    style = compose_style("Practitioner", technique="Chain-of-Thought", length="Concise")
    assert LEVEL_STYLES["Practitioner"] in style
    assert PROMPT_TECHNIQUES["Chain-of-Thought"] in style
    assert RESPONSE_LENGTHS["Concise"] in style


def test_extra_instructions_are_appended():
    style = compose_style("Researcher", extra="  Always include a physics analogy.  ")
    assert style.endswith("Always include a physics analogy.")


def test_unknown_level_falls_back_to_beginner():
    assert compose_style("No-such-level") == LEVEL_STYLES["Beginner"]


def test_empty_presets_add_no_blank_lines():
    style = compose_style("Beginner", technique="Standard", length="Balanced", extra="")
    assert "\n\n" not in style and style == LEVEL_STYLES["Beginner"]
