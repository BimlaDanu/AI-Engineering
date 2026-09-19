"""🎓 AI/ML Tutor workspace: guided, level-aware learning paths through core AI topics.

The curriculum itself lives in :mod:`src.lab.curriculum` — the single source of truth shared
with the 🔬 AI/ML Lab. This page renders each lesson as two actions: **open it in 💬 Chat**
(answered at the chosen level, grounded and cited) and, when the lesson has a hands-on
mapping, **practise it in the 🔬 AI/ML Lab** (which preselects the matching recipe and dataset).
"""

from __future__ import annotations

import streamlit as st

from src.config import LEVELS, openrouter_api_key
from src.lab.curriculum import Lesson, get_track, track_names
from src.lab.schemas import UnderstandingGrade
from src.lab.understanding import generate_question, grade_answer
from src.llm import get_llm
from src.ui.registry import go_to_page, register_page


def _render_grade(grade: UnderstandingGrade) -> None:
    """Render an understanding grade: a headline score, a verdict, feedback, and missed points."""
    icon = "🏆" if grade.score >= 0.8 else "👍" if grade.score >= 0.5 else "📚"
    st.metric("Score", f"{grade.score:.0%}", grade.verdict)
    st.markdown(f"{icon} {grade.feedback}")
    if grade.missed:
        st.caption("Worth reviewing: " + "; ".join(grade.missed))


def _understanding_check(lesson: Lesson, level: str, slot: str) -> None:
    """A collapsible, LLM-graded understanding check for a conceptual (learn-only) lesson.

    The three states are driven by session-state keyed on ``slot`` (unique per lesson): no
    question yet → offer to generate one; a question is posed → take a free-text answer and
    grade it; a grade exists → show it, with the option to try a fresh question. Degrades to a
    friendly message when no API key is configured; any model failure is shown, never raised.
    """
    ss = st.session_state
    q_key, grade_key = f"uq_{slot}", f"ugrade_{slot}"
    with st.expander("🧠 Check your understanding"):
        if not openrouter_api_key():
            st.info(
                "This poses a question and grades your answer live, so it needs an OpenRouter "
                "key. Add `OPENROUTER_API_KEY` to `.env`, then reload."
            )
            return

        if ss.get(q_key) is None:
            st.caption("Answer one short question in your own words and get graded feedback.")
            if st.button("✨ Generate a question", key=f"gen_{slot}"):
                try:
                    model = get_llm(ss.get("model"), temperature=0.5)
                    with st.spinner("Writing a question…"):
                        ss[q_key] = generate_question(model, lesson=lesson.question, level=level)
                    ss.pop(grade_key, None)
                    st.rerun()
                except Exception as exc:  # model failure is shown, never crashes the page
                    st.error(f"Could not generate a question: {type(exc).__name__}: {exc}")
            return

        question = ss[q_key]
        if not question.question:
            st.info("The model didn't return a usable question — try again.")
            if st.button("🔄 Try again", key=f"regen_{slot}"):
                ss[q_key] = None
                st.rerun()
            return

        st.markdown(f"**{question.question}**")
        answer = st.text_area("Your answer", key=f"uans_{slot}", height=120)
        c_submit, c_new = st.columns([2, 2])
        if c_submit.button("✅ Submit answer", key=f"usub_{slot}", type="primary"):
            if not answer.strip():
                st.info("Write an answer first.")
            else:
                try:
                    model = get_llm(ss.get("model"), temperature=0.0)
                    with st.spinner("Grading your answer…"):
                        ss[grade_key] = grade_answer(
                            model,
                            lesson=lesson.question,
                            question=question.question,
                            reference=question.reference_answer,
                            answer=answer,
                            level=level,
                        )
                except Exception as exc:  # model failure is shown, never crashes the page
                    st.error(f"Could not grade the answer: {type(exc).__name__}: {exc}")
        if c_new.button("🔄 New question", key=f"unew_{slot}"):
            ss[q_key] = None
            ss.pop(grade_key, None)
            st.rerun()

        grade = ss.get(grade_key)
        if grade is not None:
            _render_grade(grade)


# Icon is 🎓, deliberately distinct from the app-wide 🧠 (Synapse's brand mark in the hero /
# browser tab) so the Tutor doesn't read as "the whole app". Key stays ``ml_tutor`` for
# navigation/tests; only the display label carries the AI/ML naming.
@register_page("🎓 AI/ML Tutor", key="ml_tutor", section="Learn", order=30)
def render() -> None:
    """AI/ML Tutor tab: pick a track and level, then send lessons to Chat or the Lab."""
    st.subheader("🎓 AI/ML Tutor")
    st.caption(
        "Structured learning paths. Pick a level and a track, then **learn** a lesson in "
        "**💬 AI Chat** or **practise** it hands-on in the **🔬 AI/ML Lab**."
    )

    st.select_slider("Explain everything at this level", LEVELS, key="level")
    level = st.session_state.get("level", "Beginner")

    track_name = st.selectbox("Learning track", track_names())
    track = get_track(track_name)
    if track is None:  # defensive: track_names() and get_track() share one source of truth
        st.error("That learning track could not be found.")
        return

    st.markdown(f"### {track.name}")
    st.caption(
        f"Lessons are answered at **{level}** level. **Learn** opens Chat; **Practise** opens "
        "the AI/ML Lab (where a lesson has a runnable exercise); conceptual lessons offer a "
        "**🧠 Check your understanding** instead."
    )

    for i, lesson in enumerate(track.lessons):
        col_num, col_title, col_learn, col_practice = st.columns([1, 6, 2, 2])
        col_num.markdown(f"**{i + 1}.**")
        col_title.markdown(lesson.title)

        if col_learn.button("💬 Learn", key=f"learn_{track.name}_{i}", width="stretch"):
            st.session_state.pending = (lesson.question, level)
            # Also expose the lesson to the 🔬 AI/ML Lab so it can seed a hands-on exercise.
            st.session_state.lab_lesson = lesson.question
            # Jump straight to Chat (one click, link-like) rather than a two-step detour; the
            # answer generates there with a visible spinner, not a blank hang.
            go_to_page("chat", "Open **💬 AI Chat** from the sidebar to see the answer.")

        # Lessons with a Practice mapping open the hands-on Lab; conceptual lessons (no runnable
        # exercise) instead offer an inline, LLM-graded understanding check — so no lesson is a
        # dead-end "Learn-only" link.
        if lesson.practice is not None:
            if col_practice.button(
                "🔬 Practise", key=f"practise_{track.name}_{i}", width="stretch", type="primary"
            ):
                st.session_state.lab_lesson = lesson.question
                st.session_state.lab_practice = lesson.practice
                # Breadcrumb origin so the Lab can show where the learner came from.
                st.session_state.lab_from = {"track": track.name, "title": lesson.title}
                go_to_page(  # falls back in place if the Lab isn't registered
                    "ml_lab",
                    "Open the **🔬 AI/ML Lab** from the sidebar to practise this.",
                    level="warning",
                )
        else:
            col_practice.caption("🧠 Concept")
            _understanding_check(lesson, level, f"{track.name}_{i}")
