"""LLM-graded "check your understanding" for the 🎓 AI/ML Tutor's conceptual lessons.

Lessons with no runnable Lab exercise (CNNs, transformers, RAG vs fine-tuning) can still be
*practised* here: the model poses one short, level-scaled question about the lesson, the learner
answers in their own words, and the model grades that answer against its own reference — a
score, a verdict, targeted feedback, and the points that were missed.

Two structured-output round-trips, mirroring :mod:`src.lab.seed`: the model is forced to emit
JSON matching a Pydantic schema (``with_structured_output(..., method="json_schema")``) and we
consume the validated object — never parsed prose. The chat model is injected, so this module is
framework-agnostic (no Streamlit) and unit-testable offline with a stub. The page
(``src/ui/pages/ml_tutor.py``) wires the live model in and degrades gracefully without a key.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.lab.schemas import UnderstandingGrade, UnderstandingQuestion

_QUESTION_SYSTEM = (
    "You are an AI/ML tutor writing ONE short check-your-understanding question about a "
    "specific lesson, pitched at the learner's level. The question must probe the core idea "
    "(not trivia), be answerable in a few sentences, and be open-ended — never multiple-choice "
    "or yes/no. Also provide a concise correct reference answer used only for grading. Stay "
    "strictly within AI/ML (machine learning, deep learning, NLP, LLMs); never ask about other "
    "domains."
)

_GRADE_SYSTEM = (
    "You grade a learner's short free-text answer to an AI/ML question. Compare it against the "
    "reference answer and judge correctness and completeness at the learner's level (be "
    "generous about wording, strict about concepts). Return a score in 0.0–1.0, a very short "
    "verdict, two or three sentences of specific and encouraging feedback, and a list of key "
    "points from the reference the answer missed (empty if none). Never reveal that a reference "
    "answer exists; phrase feedback as a tutor would."
)


def generate_question(llm: BaseChatModel, *, lesson: str, level: str) -> UnderstandingQuestion:
    """Ask the model for one level-scaled understanding question (plus a grading reference).

    Args:
        llm: The chat model to drive (a modest temperature gives varied-but-focused questions).
        lesson: The lesson's seed question / title the check should probe.
        level: The learner level, so the question is pitched appropriately.

    Returns:
        A validated :class:`~src.lab.schemas.UnderstandingQuestion`; its ``question`` is empty
        only if the model returned nothing usable (the caller shows a friendly message).
    """
    model = llm.with_structured_output(UnderstandingQuestion, method="json_schema")
    human = (
        f"Lesson: {lesson}\n"
        f"Learner level: {level}\n\n"
        "Write one check-your-understanding question for this lesson at this level, with a "
        "concise reference answer for grading."
    )
    result = model.invoke([("system", _QUESTION_SYSTEM), ("human", human)])
    if isinstance(result, UnderstandingQuestion):
        return result
    return UnderstandingQuestion(question="", reference_answer="")


def grade_answer(
    llm: BaseChatModel,
    *,
    lesson: str,
    question: str,
    reference: str,
    answer: str,
    level: str,
) -> UnderstandingGrade:
    """Grade the learner's ``answer`` to ``question`` against the ``reference``, at ``level``.

    Args:
        llm: The chat model to drive (a low temperature gives stable, reproducible grades).
        lesson: The lesson the question came from, for context.
        question: The question the learner answered.
        reference: The model's own reference answer, used as the grading yardstick.
        answer: The learner's free-text answer.
        level: The learner level the grade is calibrated to.

    Returns:
        A validated :class:`~src.lab.schemas.UnderstandingGrade`.
    """
    model = llm.with_structured_output(UnderstandingGrade, method="json_schema")
    human = (
        f"Lesson: {lesson}\n"
        f"Learner level: {level}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer (for your grading only):\n{reference}\n\n"
        f"Learner's answer:\n{answer}\n\n"
        "Grade the learner's answer."
    )
    result = model.invoke([("system", _GRADE_SYSTEM), ("human", human)])
    if isinstance(result, UnderstandingGrade):
        return result
    return UnderstandingGrade(
        score=0.0,
        verdict="Not graded",
        feedback="The grader returned no result — please try again.",
        missed=[],
    )
