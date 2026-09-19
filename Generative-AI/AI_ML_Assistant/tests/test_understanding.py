"""Offline tests for the 🎓 AI/ML Tutor's LLM-graded understanding check.

The two round-trips (pose a question, grade an answer) each go through
``llm.with_structured_output(schema, method=...).invoke(...)``, so a tiny fake chat model that
returns a preset Pydantic object for each schema exercises the whole flow with no network — the
same seam :mod:`src.eval` and :mod:`src.lab.seed` stub. This pins the prompt wiring, the
schema-typed returns, and the graceful fallback when the model yields nothing usable.
"""

from __future__ import annotations

from src.lab.schemas import UnderstandingGrade, UnderstandingQuestion
from src.lab.understanding import generate_question, grade_answer


class _FakeStructured:
    """A stand-in for ``with_structured_output(...)`` that echoes a preset object."""

    def __init__(self, result: object, sink: dict) -> None:
        self._result = result
        self._sink = sink

    def invoke(self, messages):
        # Capture the composed (system, human) messages so tests can assert on the prompt.
        self._sink["messages"] = messages
        return self._result


class _FakeModel:
    """A minimal chat model: returns a schema-keyed preset from with_structured_output."""

    def __init__(self, by_schema: dict, sink: dict | None = None) -> None:
        self._by_schema = by_schema
        self.sink = sink if sink is not None else {}
        self.last_method: str | None = None

    def with_structured_output(self, schema, method: str = "json_schema"):
        self.last_method = method
        return _FakeStructured(self._by_schema[schema], self.sink)


def test_generate_question_returns_typed_object_and_grounds_the_prompt():
    q = UnderstandingQuestion(question="Why do CNNs use pooling?", reference_answer="To ...")
    model = _FakeModel({UnderstandingQuestion: q})
    result = generate_question(model, lesson="How do CNNs process images?", level="Beginner")
    assert isinstance(result, UnderstandingQuestion)
    assert result.question == "Why do CNNs use pooling?"
    assert model.last_method == "json_schema"
    # The lesson and level are grounded into the human prompt.
    human = model.sink["messages"][-1][1]
    assert "How do CNNs process images?" in human
    assert "Beginner" in human


def test_generate_question_falls_back_when_model_returns_nothing_usable():
    # A non-schema return (e.g. a bare string) degrades to an empty question, not a crash.
    model = _FakeModel({UnderstandingQuestion: "oops not a schema object"})
    result = generate_question(model, lesson="x", level="Beginner")
    assert isinstance(result, UnderstandingQuestion)
    assert result.question == ""


def test_grade_answer_returns_typed_grade_and_passes_the_reference():
    grade = UnderstandingGrade(
        score=0.75, verdict="Partially correct", feedback="Good start.", missed=["pooling"]
    )
    model = _FakeModel({UnderstandingGrade: grade})
    result = grade_answer(
        model,
        lesson="CNNs",
        question="Why pooling?",
        reference="Downsampling.",
        answer="It shrinks the image.",
        level="Practitioner",
    )
    assert isinstance(result, UnderstandingGrade)
    assert result.score == 0.75
    assert result.missed == ["pooling"]
    human = model.sink["messages"][-1][1]
    assert "Downsampling." in human  # the reference is provided to the grader
    assert "It shrinks the image." in human  # so is the learner's answer


def test_grade_answer_falls_back_when_model_returns_nothing_usable():
    model = _FakeModel({UnderstandingGrade: None})
    result = grade_answer(
        model, lesson="x", question="q", reference="r", answer="a", level="Beginner"
    )
    assert isinstance(result, UnderstandingGrade)
    assert result.score == 0.0
    assert result.verdict == "Not graded"
