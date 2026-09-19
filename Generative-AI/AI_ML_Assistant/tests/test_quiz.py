"""Offline tests for the 🎯 Trivia core (src.quiz).

The LLM and the knowledge base are both injected, so the whole builder/grader runs with no
network and no Chroma index: a deterministic ``_StubWriter`` stands in for the model and a
plain list stands in for retrieval. Mirrors tests/test_llm_exercises.py.
"""

from __future__ import annotations

import pytest

from src.quiz import (
    QuizError,
    SourcePassage,
    build_quiz,
    grade_quiz,
)
from src.quiz.quiz import _to_question
from src.quiz.schemas import QuizItem


class _StubWriter:
    """A deterministic :class:`~src.quiz.quiz.QuizWriter` — returns fixed, valid items."""

    def __init__(self, items: list[QuizItem] | None = None) -> None:
        self.calls: list[dict] = []
        self._items = items

    def write(self, *, subject, level, passages, count):
        self.calls.append(
            {"subject": subject, "level": level, "passages": passages, "count": count}
        )
        if self._items is not None:
            return self._items
        # One well-formed item per requested question, each citing a real passage.
        return [
            QuizItem(
                question=f"Question {i} about {subject}?",
                options=["right", "wrong-1", "wrong-2", "wrong-3"],
                correct_index=0,
                explanation="Because option one matches the passage.",
                source_index=(i % len(passages)) + 1,
            )
            for i in range(1, count + 1)
        ]


def _passages(n: int = 4) -> list[SourcePassage]:
    return [SourcePassage(text=f"passage {i} text", source=f"doc_{i}.md") for i in range(1, n + 1)]


def _retrieve_from(passages):
    def retrieve(query, k):
        return passages[:k]

    return retrieve


def test_build_quiz_produces_requested_questions_with_sources():
    passages = _passages()
    writer = _StubWriter()
    quiz = build_quiz(
        "Machine Learning", "Beginner", 3, retrieve=_retrieve_from(passages), writer=writer
    )
    assert len(quiz.questions) == 3
    assert quiz.subject == "Machine Learning"
    # Every question cites one of the retrieved source documents.
    assert all(q.source.startswith("doc_") for q in quiz.questions)
    # The writer was asked for the requested count and given the passage *texts* only.
    assert writer.calls[0]["count"] == 3
    assert writer.calls[0]["passages"] == [p.text for p in passages]


def test_build_quiz_raises_when_kb_is_empty():
    with pytest.raises(QuizError):
        build_quiz("AI Engineering", "Beginner", 3, retrieve=lambda q, k: [], writer=_StubWriter())


def test_build_quiz_raises_when_writer_returns_nothing():
    with pytest.raises(QuizError):
        build_quiz(
            "Deep Learning",
            "Beginner",
            3,
            retrieve=_retrieve_from(_passages()),
            writer=_StubWriter(items=[]),
        )


def test_malformed_items_are_dropped_not_crashed():
    passages = _passages(2)
    good = QuizItem(
        question="Valid?",
        options=["a", "b", "c", "d"],
        correct_index=1,
        explanation="e",
        source_index=1,
    )
    too_few_options = QuizItem(
        question="Bad?", options=["only-one"], correct_index=0, explanation="e", source_index=1
    )
    correct_out_of_range = QuizItem(
        question="Bad2?", options=["a", "b"], correct_index=3, explanation="e", source_index=1
    )
    quiz = build_quiz(
        "ML",
        "Beginner",
        3,
        retrieve=_retrieve_from(passages),
        writer=_StubWriter(items=[good, too_few_options, correct_out_of_range]),
    )
    assert len(quiz.questions) == 1
    assert quiz.questions[0].correct_index == 1


def test_out_of_range_source_index_degrades_to_generic_citation():
    passages = _passages(1)
    item = QuizItem(
        question="Q?", options=["a", "b"], correct_index=0, explanation="e", source_index=99
    )  # no such passage
    q = _to_question(item, passages)
    assert q is not None and q.source == "knowledge base"


def test_grade_quiz_scores_and_marks_each_question():
    quiz = build_quiz(
        "ML", "Beginner", 3, retrieve=_retrieve_from(_passages()), writer=_StubWriter()
    )  # correct_index is 0 for every stub item
    result = grade_quiz(quiz, selections=[0, 1, None])
    assert result.total == 3
    assert result.score == 1  # only the first (index 0) matches
    assert result.graded[0].correct is True
    assert result.graded[1].correct is False
    assert result.graded[2].selected_index is None and result.graded[2].correct is False
    assert result.percentage == pytest.approx(100 / 3)


def test_grade_quiz_handles_short_selection_list():
    quiz = build_quiz(
        "ML", "Beginner", 2, retrieve=_retrieve_from(_passages()), writer=_StubWriter()
    )
    result = grade_quiz(quiz, selections=[0])  # second question left unanswered
    assert result.total == 2
    assert result.graded[1].selected_index is None


def test_quiz_sources_are_distinct_and_ordered():
    # Two questions citing the same doc collapse to one entry in Quiz.sources.
    items = [
        QuizItem(
            question="Q1?", options=["a", "b"], correct_index=0, explanation="e", source_index=1
        ),
        QuizItem(
            question="Q2?", options=["a", "b"], correct_index=0, explanation="e", source_index=1
        ),
    ]
    quiz = build_quiz(
        "ML", "Beginner", 2, retrieve=_retrieve_from(_passages()), writer=_StubWriter(items=items)
    )
    assert quiz.sources == ["doc_1.md"]


# --- The page's own failure handling ----------------------------------------------------------

_QUIZ_APP = """
from src.ui.pages.quiz import render
from src.ui.state import init_state

init_state()
render()
"""


def test_generate_reports_an_unavailable_model_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead provider must reach the reader as a message, not a traceback.

    ``_generate`` already wrapped ``build_quiz`` in a try, but built the model on the line
    *above* it. ``get_llm`` raises on a missing or malformed key, so the single most likely
    failure was the one the guard did not cover. Only reachable with a loaded knowledge base,
    which is why an audit against an empty index missed it.
    """
    from streamlit.testing.v1 import AppTest

    from src.ui import state
    from src.ui.pages import quiz as quiz_page

    class _KB:
        size = 12

        def search(self, *_a: object, **_k: object) -> list:
            return []

    monkeypatch.setattr(state, "load_kb_status", lambda: state.KBStatus(kb=_KB()))
    monkeypatch.setattr(quiz_page, "load_kb", _KB)

    def _no_provider(*_a: object, **_k: object):
        raise RuntimeError("no credentials / provider unreachable")

    monkeypatch.setattr(quiz_page, "get_llm", _no_provider)

    at = AppTest.from_string(_QUIZ_APP, default_timeout=60).run()
    generate = next(b for b in at.button if "Generate quiz" in b.label)
    generate.click().run()

    assert not at.exception, "an unreachable model must not surface as a traceback"
    assert at.error or at.warning, "an unreachable model must be reported"
