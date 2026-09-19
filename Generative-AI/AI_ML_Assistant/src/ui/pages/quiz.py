"""🎯 Trivia workspace: test yourself on the knowledge base.

The fourth pillar of the learning loop — 💬 **Chat** (ask), 🎓 **Tutor** (learn),
🔬 **Lab** (practise), and here **test yourself**. Questions are generated live *only* from
passages retrieved from the curated knowledge base (never free-invented), each cites the
source document it came from, and grading is instant and deterministic.

All the generation/grading logic lives in the framework-agnostic :mod:`src.quiz` package;
this module is the thin Streamlit shell that wires retrieval to it and renders the quiz.
It degrades gracefully: a missing index or API key shows a friendly message, and any model
failure surfaces as an error rather than crashing the page.
"""

from __future__ import annotations

import streamlit as st

from src.config import LEVELS, SUBJECTS, openrouter_api_key
from src.llm import get_llm
from src.quiz import (
    LLMQuizWriter,
    QuizError,
    SourcePassage,
    build_quiz,
    grade_quiz,
)
from src.ui.registry import go_to_page, register_page
from src.ui.state import kb_status_notice, load_kb


def _nav_row() -> None:
    """Links to the sibling learning workspaces, closing the learn → test loop."""
    c_learn, c_ask, _ = st.columns([2, 2, 5])
    if c_learn.button("🎓 Learn in AI/ML Tutor", width="stretch"):
        go_to_page("ml_tutor", "Open **🎓 AI/ML Tutor** from the sidebar.")
    if c_ask.button("💬 Ask in AI Chat", width="stretch"):
        go_to_page("chat", "Open **💬 AI Chat** from the sidebar.")


def _controls() -> tuple[str, str, int]:
    """Render the subject / level / length pickers and return the chosen values."""
    subject = st.selectbox("Subject", list(SUBJECTS), key="subject_label")
    st.select_slider("Difficulty", LEVELS, key="level")
    level = st.session_state.get("level", LEVELS[0])
    count = st.slider("Number of questions", min_value=3, max_value=8, value=5, key="quiz_count")
    return subject, level, count


def _generate(subject: str, level: str, count: int) -> None:
    """Build a fresh quiz from the knowledge base and stash it in session state.

    Retrieval is closed over the loaded KB and the subject filter; the writer uses the
    learner's chosen chat model with a modest temperature for varied-but-grounded questions.
    Any failure is shown in place — the previous quiz (if any) is left untouched.
    """
    kb = load_kb()
    settings = st.session_state.settings
    subjects = SUBJECTS[subject]

    def retrieve(query: str, k: int) -> list[SourcePassage]:
        hits = kb.search(query, subjects=subjects, k=k, alpha=settings.hybrid_alpha)
        return [SourcePassage(h.text, h.metadata.get("source", "knowledge base")) for h in hits]

    # Building the model is inside the guard, not above it. `get_llm` raises on a missing or
    # malformed key — the exact "bad key" case the handler below was written for — so leaving
    # it outside meant the one failure most likely to happen was the one that crashed the page.
    with st.spinner(f"Writing {count} questions on {subject} at {level} level…"):
        try:
            writer = LLMQuizWriter(get_llm(st.session_state.get("model"), temperature=0.4))
            quiz = build_quiz(subject, level, count, retrieve=retrieve, writer=writer)
        except QuizError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:  # network/model failure is shown, never crashes the page
            st.error(f"Could not generate the quiz: {type(exc).__name__}: {exc}")
            return
    st.session_state.quiz = quiz
    st.session_state.quiz_result = None
    st.session_state.quiz_nonce = st.session_state.get("quiz_nonce", 0) + 1


def _render_questions(quiz) -> list[int | None]:
    """Render each question as a radio group and return the selected option index per question."""
    nonce = st.session_state.get("quiz_nonce", 0)
    selections: list[int | None] = []
    for i, q in enumerate(quiz.questions):
        st.markdown(f"**{i + 1}. {q.prompt}**")
        choice = st.radio(
            "Select an answer",
            q.options,
            index=None,
            key=f"quizq_{nonce}_{i}",
            label_visibility="collapsed",
        )
        selections.append(q.options.index(choice) if choice is not None else None)
    return selections


def _render_result(result) -> None:
    """Render the graded outcome: a headline score plus per-question feedback and citations."""
    st.divider()
    pct = result.percentage
    emoji = "🏆" if pct >= 80 else "👍" if pct >= 50 else "📚"
    st.metric("Your score", f"{result.score} / {result.total}", f"{pct:.0f}%")
    st.caption(
        f"{emoji} "
        + (
            "Excellent — you've got this."
            if pct >= 80
            else "Good progress — review the misses below."
            if pct >= 50
            else "Worth revisiting — read the explanations, then learn more in 🎓 Tutor / 💬 Chat."
        )
    )

    for i, g in enumerate(result.graded):
        q = g.question
        if g.selected_index is None:
            head, icon = "Not answered", "⚪"
        elif g.correct:
            head, icon = "Correct", "✅"
        else:
            head, icon = "Incorrect", "❌"
        with st.expander(f"{icon} {i + 1}. {q.prompt}  —  {head}", expanded=not g.correct):
            for j, option in enumerate(q.options):
                marker = "✅" if j == q.correct_index else ("❌" if j == g.selected_index else "•")
                st.markdown(f"{marker} {option}")
            st.info(f"**Why:** {q.explanation}")
            st.caption(f"📄 Source: *{q.source}*")


@register_page("🎯 Trivia", key="quiz", section="Learn", order=40)
def render() -> None:
    """Trivia tab: generate a KB-grounded MCQ quiz, answer it, and get graded."""
    ss = st.session_state
    st.subheader("🎯 Trivia")
    st.caption(
        "Test yourself on the knowledge base. Pick a subject and difficulty, and the app "
        "writes multiple-choice questions **grounded in the indexed documents** — each with "
        "an explanation and a source. Complete the loop: **learn** in the Tutor, then **test**."
    )

    _nav_row()

    if load_kb() is None:
        kb_status_notice(
            "The knowledge base isn't indexed yet, so there's nothing to be quizzed on. "
            "Open **📄 Knowledge Base** to add documents and build the index, or run "
            "`make ingest` in a terminal."
        )
        return

    subject, level, count = _controls()

    if not openrouter_api_key():
        st.info(
            "Generating a quiz answers questions live, so it needs an OpenRouter key. Add "
            "`OPENROUTER_API_KEY` to `.env`, then reload this page."
        )
        return

    col_gen, col_new = st.columns([2, 2])
    if col_gen.button("✨ Generate quiz", type="primary", width="stretch"):
        _generate(subject, level, count)
    if ss.get("quiz") is not None and col_new.button("🔄 Clear", width="stretch"):
        ss.quiz = None
        ss.quiz_result = None

    quiz = ss.get("quiz")
    if quiz is None:
        return

    st.divider()
    st.caption(f"**{quiz.subject}** · {quiz.level} · {len(quiz.questions)} questions")
    selections = _render_questions(quiz)

    if st.button("✅ Submit answers", type="primary", width="stretch"):
        ss.quiz_result = grade_quiz(quiz, selections)

    if ss.get("quiz_result") is not None:
        _render_result(ss.quiz_result)
