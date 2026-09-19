"""Grounded quiz generation and deterministic grading — the core of 📝 Knowledge Check.

Design mirrors the evaluation harness: the LLM seam is a :class:`QuizWriter` *protocol*,
satisfied in production by :class:`LLMQuizWriter` (one ``with_structured_output`` round-trip)
and in tests by a deterministic stub, so the whole builder runs offline. Retrieval is injected
too (a ``retrieve`` callable), keeping this module free of any Streamlit or Chroma import.

Two responsibilities:

* :func:`build_quiz` — retrieve passages for a subject, ask the writer for multiple-choice
  items grounded in them, validate each item, and attach a source citation.
* :func:`grade_quiz` — pure, LLM-free scoring of the learner's selected options.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.quiz.schemas import QuizItem


class QuizError(Exception):
    """Raised when a quiz cannot be built (no KB content, or no valid item produced)."""


@dataclass(frozen=True)
class SourcePassage:
    """One retrieved knowledge-base passage: its text and the source document it came from."""

    text: str
    source: str


@dataclass(frozen=True)
class QuizQuestion:
    """A validated multiple-choice question ready to render, with its source citation."""

    prompt: str
    options: tuple[str, ...]
    correct_index: int
    explanation: str
    source: str

    @property
    def correct_option(self) -> str:
        """The text of the correct option (indexing is already validated in the builder)."""
        return self.options[self.correct_index]


@dataclass(frozen=True)
class Quiz:
    """A generated quiz: its subject/level, the questions, and the sources it drew from."""

    subject: str
    level: str
    questions: tuple[QuizQuestion, ...]

    @property
    def sources(self) -> list[str]:
        """Distinct source documents cited across the quiz, in first-seen order."""
        seen: list[str] = []
        for q in self.questions:
            if q.source not in seen:
                seen.append(q.source)
        return seen


@dataclass(frozen=True)
class GradedQuestion:
    """One graded answer: the question, what the learner picked, and whether it was right."""

    question: QuizQuestion
    selected_index: int | None
    correct: bool


@dataclass
class QuizResult:
    """The outcome of grading a whole quiz: per-question verdicts and the tally."""

    graded: list[GradedQuestion] = field(default_factory=list)
    score: int = 0
    total: int = 0

    @property
    def percentage(self) -> float:
        """Score as a 0..100 percentage (0.0 for an empty quiz)."""
        return 100.0 * self.score / self.total if self.total else 0.0


@runtime_checkable
class QuizWriter(Protocol):
    """The single generation primitive the builder needs — the seam tests stub out."""

    def write(self, *, subject: str, level: str, passages: list[str], count: int) -> list[QuizItem]:
        """Return up to ``count`` MCQ items grounded only in ``passages``."""
        ...


_SYSTEM = (
    "You are a computer-science exam writer creating multiple-choice questions to test a "
    "learner's understanding of AI, machine learning, and deep learning.\n"
    "Rules:\n"
    "- Base EVERY question strictly on the numbered context passages provided. Never invent "
    "facts that are not supported by them.\n"
    "- Each question has exactly four options: one correct answer and three plausible but "
    "clearly wrong distractors.\n"
    "- Questions must be self-contained: the learner does NOT see the passages, so never "
    "write 'according to the passage' or 'in the text above'.\n"
    "- Set source_index to the 1-based number of the passage a question is drawn from.\n"
    "- Write the explanation as concise feedback justifying the correct option."
)

_LEVEL_GUIDANCE: dict[str, str] = {
    "Beginner": (
        "Target a beginner: test core definitions and intuitions in plain language; avoid "
        "heavy mathematics."
    ),
    "Practitioner": (
        "Target a practitioner: test applied understanding, trade-offs, and when to use what."
    ),
    "Researcher": (
        "Target an advanced learner: test precise, subtle, or mathematical distinctions "
        "between related ideas."
    ),
}


def _format_passages(passages: list[str], limit: int = 1200) -> str:
    """Render passages as a compact numbered block for the writer prompt."""
    return "\n\n".join(f"[{i}] {p[:limit]}" for i, p in enumerate(passages, start=1))


class LLMQuizWriter:
    """A :class:`QuizWriter` backed by an OpenRouter chat model and structured outputs.

    One ``with_structured_output(QuizItems, method="json_schema")`` round-trip per quiz; the
    model is forced to emit JSON matching the schema, which is returned as validated
    :class:`~src.quiz.schemas.QuizItem` objects. Construct with a modest temperature for
    varied-but-grounded questions.
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    def write(self, *, subject: str, level: str, passages: list[str], count: int) -> list[QuizItem]:
        """Generate up to ``count`` grounded MCQ items (empty list if the model returns none)."""
        from src.quiz.schemas import QuizItems

        if not passages:
            return []
        human = (
            f"Subject: {subject}\n"
            f"{_LEVEL_GUIDANCE.get(level, '')}\n\n"
            f"Write {count} multiple-choice questions grounded in these passages:\n\n"
            f"{_format_passages(passages)}"
        )
        model = self._llm.with_structured_output(QuizItems, method="json_schema")
        result = model.invoke([("system", _SYSTEM), ("human", human)])
        return list(result.items) if isinstance(result, QuizItems) else []


def _to_question(item: QuizItem, passages: list[SourcePassage]) -> QuizQuestion | None:
    """Validate one raw :class:`QuizItem` and attach its source, or drop it if malformed.

    Guards against a model that ignores the schema hints: an item needs at least two options
    and a ``correct_index`` that actually lands inside them. ``source_index`` is 1-based into
    the passages; an out-of-range value degrades to a generic citation rather than crashing.
    """
    options = [o.strip() for o in item.options if o.strip()]
    if len(options) < 2 or not (0 <= item.correct_index < len(options)):
        return None
    idx = item.source_index - 1
    source = passages[idx].source if 0 <= idx < len(passages) else "knowledge base"
    return QuizQuestion(
        prompt=item.question.strip(),
        options=tuple(options),
        correct_index=item.correct_index,
        explanation=item.explanation.strip(),
        source=source,
    )


def build_quiz(
    subject: str,
    level: str,
    count: int,
    *,
    retrieve: Callable[[str, int], list[SourcePassage]],
    writer: QuizWriter,
) -> Quiz:
    """Build a grounded quiz for ``subject`` at ``level`` with up to ``count`` questions.

    Args:
        subject: The subject label shown to the learner (also steers retrieval).
        level: One of the app's difficulty levels; tunes question difficulty.
        count: How many questions to request (the writer may return fewer).
        retrieve: ``(query, k) -> passages`` — injected knowledge-base search.
        writer: The :class:`QuizWriter` that turns passages into MCQ items.

    Returns:
        A :class:`Quiz` of validated questions.

    Raises:
        QuizError: if the knowledge base yields no passages, or the writer produces no
            valid question — surfaced to the UI as a friendly message, never a crash.
    """
    query = f"key concepts, definitions, and important ideas in {subject}"
    passages = retrieve(query, max(count * 2, 6))
    if not passages:
        raise QuizError(
            "No knowledge-base content found for this subject. Build the index "
            "(📄 Knowledge Base → re-index, or `make ingest`) and try again."
        )
    items = writer.write(
        subject=subject, level=level, passages=[p.text for p in passages], count=count
    )
    questions = [q for item in items if (q := _to_question(item, passages)) is not None]
    if not questions:
        raise QuizError("The model did not return any usable questions — try again.")
    return Quiz(subject=subject, level=level, questions=tuple(questions))


def grade_quiz(quiz: Quiz, selections: list[int | None]) -> QuizResult:
    """Grade ``selections`` (one chosen option index per question, None = unanswered).

    Pure and deterministic — no LLM. A missing/short selections list treats the remaining
    questions as unanswered, so a partially-filled quiz still grades cleanly.
    """
    graded: list[GradedQuestion] = []
    score = 0
    for i, question in enumerate(quiz.questions):
        selected = selections[i] if i < len(selections) else None
        correct = selected == question.correct_index
        score += int(correct)
        graded.append(GradedQuestion(question=question, selected_index=selected, correct=correct))
    return QuizResult(graded=graded, score=score, total=len(quiz.questions))
