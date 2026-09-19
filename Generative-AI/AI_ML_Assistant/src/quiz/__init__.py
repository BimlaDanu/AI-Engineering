"""Knowledge-check quizzes: KB-grounded, structured-output multiple-choice questions.

The 🎯 Trivia workspace completes the learning loop that the rest of the app
already offers — 💬 **Chat** (ask), 🎓 **Tutor** (learn), 🔬 **Lab** (practice), and now
**test yourself**. Questions are generated *only* from passages retrieved from the curated
knowledge base (never free-invented), each carries a citation back to its source document,
and grading is pure/deterministic.

This package is framework-agnostic (no Streamlit) and offline-testable: the writer is a
:class:`~src.quiz.quiz.QuizWriter` protocol satisfied in production by
:class:`~src.quiz.quiz.LLMQuizWriter` (one ``with_structured_output`` round-trip) and in
tests by a tiny deterministic stub, exactly like the evaluation judge.
"""

from __future__ import annotations

from src.quiz.quiz import (
    GradedQuestion,
    LLMQuizWriter,
    Quiz,
    QuizError,
    QuizQuestion,
    QuizResult,
    QuizWriter,
    SourcePassage,
    build_quiz,
    grade_quiz,
)
from src.quiz.schemas import QuizItem, QuizItems

__all__ = [
    "GradedQuestion",
    "LLMQuizWriter",
    "Quiz",
    "QuizError",
    "QuizItem",
    "QuizItems",
    "QuizQuestion",
    "QuizResult",
    "QuizWriter",
    "SourcePassage",
    "build_quiz",
    "grade_quiz",
]
