"""Unit tests for JSON handling and the typed response models."""

from core import (
    CoachingResponse,
    JudgeFeedback,
    _split_numbered_lines,
    parse_json_response,
)

# ---------------------------------------------------------------------------
# parse_json_response (fallback parser)
# ---------------------------------------------------------------------------


def test_parses_plain_json_object():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parses_json_array():
    assert parse_json_response('["q1", "q2"]') == ["q1", "q2"]


def test_strips_json_fences():
    raw = '```json\n{"advice": "hi"}\n```'
    assert parse_json_response(raw) == {"advice": "hi"}


def test_strips_bare_fences():
    raw = '```\n["a", "b"]\n```'
    assert parse_json_response(raw) == ["a", "b"]


def test_malformed_json_returns_none():
    assert parse_json_response('{"advice": "unterminated') is None


def test_empty_and_none_return_none():
    assert parse_json_response("") is None
    assert parse_json_response(None) is None
    assert parse_json_response("   ") is None


def test_prose_returns_none():
    assert parse_json_response("Here is my advice: practice daily.") is None


# ---------------------------------------------------------------------------
# CoachingResponse.from_dict
# ---------------------------------------------------------------------------

VALID_COACHING = {
    "advice": "Practice out loud.",
    "action_items": ["Do a mock interview", "Prepare STAR stories"],
    "common_mistakes": ["Rambling"],
}


def test_coaching_valid():
    coaching = CoachingResponse.from_dict(VALID_COACHING)
    assert coaching is not None
    assert coaching.advice == "Practice out loud."
    assert coaching.action_items == ["Do a mock interview", "Prepare STAR stories"]
    assert coaching.common_mistakes == ["Rambling"]


def test_coaching_missing_field_returns_none():
    data = dict(VALID_COACHING)
    del data["action_items"]
    assert CoachingResponse.from_dict(data) is None


def test_coaching_wrong_types_return_none():
    data = dict(VALID_COACHING, advice=42)
    assert CoachingResponse.from_dict(data) is None
    data = dict(VALID_COACHING, common_mistakes="not a list")
    assert CoachingResponse.from_dict(data) is None


def test_coaching_non_dict_returns_none():
    assert CoachingResponse.from_dict(None) is None
    assert CoachingResponse.from_dict(["a list"]) is None
    assert CoachingResponse.from_dict("a string") is None


# ---------------------------------------------------------------------------
# JudgeFeedback.from_dict
# ---------------------------------------------------------------------------

VALID_JUDGE = {
    "score": 4,
    "verdict": "Solid answer with room to grow.",
    "strengths": ["Clear structure"],
    "improvements": ["Add a metric", "Mention the result"],
    "model_answer": "A strong answer would...",
}


def test_judge_valid():
    fb = JudgeFeedback.from_dict(VALID_JUDGE)
    assert fb is not None
    assert fb.score == 4
    assert fb.verdict.startswith("Solid")


def test_judge_score_clamped():
    fb = JudgeFeedback.from_dict(dict(VALID_JUDGE, score=9))
    assert fb is not None and fb.score == 5
    fb = JudgeFeedback.from_dict(dict(VALID_JUDGE, score=0))
    assert fb is not None and fb.score == 1


def test_judge_numeric_string_score_accepted():
    fb = JudgeFeedback.from_dict(dict(VALID_JUDGE, score="3"))
    assert fb is not None and fb.score == 3


def test_judge_non_numeric_score_returns_none():
    assert JudgeFeedback.from_dict(dict(VALID_JUDGE, score="great")) is None
    assert JudgeFeedback.from_dict(dict(VALID_JUDGE, score=None)) is None


def test_judge_missing_field_returns_none():
    data = dict(VALID_JUDGE)
    del data["model_answer"]
    assert JudgeFeedback.from_dict(data) is None


def test_judge_non_dict_returns_none():
    assert JudgeFeedback.from_dict("prose feedback") is None


# ---------------------------------------------------------------------------
# _split_numbered_lines (last-resort fallback)
# ---------------------------------------------------------------------------


def test_split_numbered_lines_basic():
    raw = """Here are your questions:
1. What is a Python list comprehension used for?
2) Explain the GIL and when it matters.
- Describe how you would reverse a linked list.
"""
    questions = _split_numbered_lines(raw)
    assert len(questions) == 3
    assert questions[0].startswith("What is a Python")


def test_split_numbered_lines_ignores_short_fragments():
    raw = "1. Too short\n2. This question is definitely long enough to keep."
    questions = _split_numbered_lines(raw)
    assert questions == ["This question is definitely long enough to keep."]


def test_split_numbered_lines_respects_limit():
    raw = "\n".join(f"{i}. Question number {i} padded to be long enough?" for i in range(1, 20))
    assert len(_split_numbered_lines(raw, n=10)) == 10


def test_split_numbered_lines_empty_input():
    assert _split_numbered_lines("") == []
    assert _split_numbered_lines("No list here, just prose.") == []
