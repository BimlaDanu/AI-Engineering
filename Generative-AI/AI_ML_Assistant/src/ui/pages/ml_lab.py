"""🔬 AI/ML Lab workspace: run real ML/DL/NLP on bundled datasets, seeded by the AI/ML Tutor.

Two ways to work, per topic (ML · NLP · lightweight DL):

* **Guided recipes** (the safe default) — the app runs its *own* vetted code against a
  bundled scikit-learn toy dataset and shows the result *and* the equivalent source. No
  arbitrary execution.
* **Advanced cell** (opt-in) — an editable Python cell, optionally seeded by the LLM from the
  current 🎓 AI/ML Tutor lesson, run through the restricted :mod:`src.lab.sandbox`. This executes
  code locally in-process behind an allow-list; it is a guardrail, not a hardened sandbox
  (see the caution rendered on the page and the module docstring).

Heavier topics (LLMs/RAG, Quantum ML/NQS) can't meaningfully execute in-browser — no code or
data-visualisation to offer — so they are deliberately absent here; explore them in 💬 Chat or
🎓 AI/ML Tutor instead. The page degrades gracefully when the optional data dependencies (pandas /
scikit-learn / matplotlib) are not yet installed.
"""

from __future__ import annotations

import streamlit as st

from src.config import LEVELS, PROMPT_TECHNIQUES, ROUTER_MODEL, openrouter_api_key
from src.eval.judge import LLMJudge
from src.lab.challenges import (
    CHALLENGES,
    Challenge,
    CheckResult,
    challenges_for_recipe,
    pick_for_level,
)
from src.lab.datasets import (
    DATASETS,
    TOPIC_DL,
    TOPIC_LLM,
    TOPIC_ML,
    TOPIC_NLP,
    LabDataset,
    datasets_for_topic,
)
from src.lab.llm_exercises import (
    LLM_EXERCISES,
    ExerciseResult,
    compare_prompt_techniques,
    evaluate_answer,
    tutor_system,
)
from src.lab.recipes import RECIPES, LabResult, ParamSpec, Recipe, recipes_for_topic
from src.lab.sandbox import ALLOWED_MODULES, run_sandboxed
from src.lab.seed import seed_snippet
from src.llm import get_llm
from src.ui.registry import go_to_page, register_page

# Selectable topics. ML/NLP/DL run guided code over bundled datasets; AI & LLM runs live-model
# exercises (prompt comparison + answer evaluation) that need an API key but no data deps.
# (Quantum ML/NQS has no runnable in-Lab exercise — explore it in 💬 Chat.)
_RUNNABLE_TOPICS = {
    TOPIC_ML: "Classic ML — classification, clustering, and exploration on toy datasets.",
    TOPIC_NLP: "Text — TF-IDF classification and cosine similarity on a small corpus.",
    TOPIC_DL: "Lightweight DL — a small neural net (sklearn MLP) with a live loss curve.",
    TOPIC_LLM: "AI & LLM — ask a question live, compare prompt techniques, and evaluate answers.",
}

# Optional Lab dependencies as (import name, pip/pyproject name).
_LAB_DEPS = (("pandas", "pandas"), ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"))


def _deps() -> tuple[bool, list[str]]:
    """Return ``(all_present, missing)`` for the optional Lab data dependencies."""
    missing = []
    for mod, pkg in _LAB_DEPS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return not missing, missing


def _param_value(p: ParamSpec, key: str):
    """Render one recipe parameter as its matching widget and return the chosen value."""
    if p.kind == "select":
        return st.selectbox(p.label, p.options, key=key)
    if p.kind == "slider_int":
        return st.slider(p.label, int(p.min), int(p.max), int(p.default), int(p.step), key=key)
    if p.kind == "slider_float":
        return st.slider(
            p.label, float(p.min), float(p.max), float(p.default), float(p.step), key=key
        )
    return st.checkbox(p.label, value=bool(p.default), key=key)


def _render_result(result: LabResult) -> None:
    """Render a recipe/sandbox :class:`LabResult` — error, summary, table, figure, code."""
    if result.error:
        st.error(result.error)
        return
    if result.summary:
        st.markdown(result.summary)
    if result.table is not None:
        st.dataframe(result.table, width="stretch", hide_index=True)
    if result.figure is not None:
        st.pyplot(result.figure)
    if result.code:
        with st.expander("📄 The code this recipe ran"):
            st.code(result.code, language="python")


def _render_progress() -> None:
    """Show a one-line tally of challenges passed this session (hidden until the first pass)."""
    progress = st.session_state.get("lab_progress", {})
    passed = sum(1 for done in progress.values() if done)
    if passed:
        st.caption(f"🏆 Practice progress: **{passed} / {len(CHALLENGES)}** challenges passed.")


def _record_challenge(challenge: Challenge, verdict: CheckResult) -> None:
    """Persist a challenge outcome in session progress — a pass sticks once it's earned."""
    progress: dict[str, bool] = st.session_state.setdefault("lab_progress", {})
    progress[challenge.key] = bool(progress.get(challenge.key)) or verdict.passed


def _render_challenge_goal(challenge: Challenge) -> None:
    """Show the active challenge's goal before the learner runs the recipe."""
    already = st.session_state.get("lab_progress", {}).get(challenge.key, False)
    badge = "✅ Challenge passed" if already else f"🎯 {challenge.level} challenge"
    st.info(f"**{badge}** — {challenge.goal}")


def _render_challenge_ladder(recipe_key: str) -> None:
    """List every tier for this recipe with a done/not-done marker, so harder goals are visible."""
    ladder = challenges_for_recipe(recipe_key)
    if len(ladder) <= 1:
        return
    progress = st.session_state.get("lab_progress", {})
    with st.expander("🏁 All challenge tiers for this recipe"):
        for c in ladder:
            mark = "✅" if progress.get(c.key) else "⬜"
            st.markdown(f"{mark} **{c.level}** — {c.goal}")


def _render_verdict(verdict: CheckResult) -> None:
    """Render the auto-check outcome of a graded run: passed (success) or not-yet (retry)."""
    if verdict.passed:
        st.success(f"✅ {verdict.message}")
    else:
        st.warning(f"🔁 {verdict.message}")


def _guided(topic: str, dataset: LabDataset, data) -> None:
    """Render the guided-recipe controls for a topic, run the recipe, and grade the challenge."""
    recipes = recipes_for_topic(topic)
    if not recipes:
        st.info("No guided recipe for this topic yet — try the advanced cell below.")
        return

    labels = {r.label: r for r in recipes}
    recipe: Recipe = labels[st.selectbox("Recipe", list(labels), key=f"recipe_{topic}")]
    st.caption(recipe.description)

    # The current difficulty selects which challenge tier the learner is attempting; not every
    # recipe has all three, so pick_for_level falls back to the nearest available tier.
    level = st.session_state.get("level", LEVELS[0])
    challenge = pick_for_level(recipe.key, level)
    if challenge is not None:
        _render_challenge_goal(challenge)
        _render_challenge_ladder(recipe.key)

    params: dict[str, object] = {}
    # The explore recipe picks its two scatter axes from the *live* dataset columns.
    if recipe.key == "explore":
        cols = list(data.df.columns)
        c1, c2 = st.columns(2)
        params["x"] = c1.selectbox("X axis", cols, index=0, key=f"x_{dataset.key}")
        params["y"] = c2.selectbox(
            "Y axis", cols, index=min(1, len(cols) - 1), key=f"y_{dataset.key}"
        )
    for p in recipe.params:
        params[p.name] = _param_value(p, key=f"{recipe.key}_{p.name}")

    if st.button("▶️ Run recipe", type="primary", key=f"run_{recipe.key}"):
        try:
            result = recipe.run(data, params)
        except Exception as exc:  # a recipe failure is shown, never crashes the page
            st.error(f"Recipe failed: {type(exc).__name__}: {exc}")
            return
        _render_result(result)
        # Grade the run against the active challenge (only when it produced a clean result — a
        # recipe error is already surfaced by _render_result, no need to repeat it as a verdict).
        if challenge is not None and result.error is None:
            verdict = challenge.check(result, params)
            _record_challenge(challenge, verdict)
            _render_verdict(verdict)


def _advanced(topic: str, dataset: LabDataset, data, lesson: str) -> None:
    """Render the opt-in advanced code cell: optional LLM seeding, then restricted execution."""
    st.caption(
        "Edit and run Python against this dataset. Pre-bound: `df`, `X`, `y`"
        + (", `texts`" if data.task == "text" else "")
        + " and the handles `np`, `pd`, `plt`."
    )
    st.warning(
        "⚠️ This runs your code **locally, in-process**. It's restricted to an allow-list "
        f"({', '.join(sorted(ALLOWED_MODULES))}) with no file/network access — a guardrail "
        "against accidents, **not** a hardened sandbox. Only run code you understand.",
        icon="⚠️",
    )

    code_key = f"lab_code_{topic}"
    if st.button("✨ Seed starter code from the current lesson", key=f"seed_{topic}"):
        if not openrouter_api_key():
            st.error("Set `OPENROUTER_API_KEY` in `.env` to let the assistant seed code.")
        else:
            try:
                model = get_llm(st.session_state.get("model"), temperature=0.1)
                snippet = seed_snippet(
                    model,
                    topic=topic,
                    lesson=lesson or f"Explore the {dataset.label} dataset.",
                    data=data,
                    feature_names=list(data.df.columns),
                )
                st.session_state[code_key] = snippet.code
                if snippet.explanation:
                    st.info(snippet.explanation)
            except Exception as exc:
                st.error(f"Could not seed code: {type(exc).__name__}: {exc}")

    st.session_state.setdefault(code_key, "print(df.head())\n")
    code = st.text_area("Python", key=code_key, height=220)

    if st.button("▶️ Run code", type="primary", key=f"exec_{topic}"):
        import matplotlib as mpl

        mpl.use("Agg")  # headless: figures are captured, never shown in a GUI
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        plt.close("all")
        ns = {
            "np": np,
            "pd": pd,
            "plt": plt,
            "df": data.df,
            "X": data.df.values,
            "y": None if data.target is None else data.target.values,
        }
        if data.task == "text":
            ns["texts"] = data.df["text"].tolist()

        result = run_sandboxed(code, ns)
        if result.stdout:
            st.code(result.stdout, language="text")
        if result.figure is not None:
            st.pyplot(result.figure)
        for message in result.warnings:
            st.warning(message, icon="⚠️")
        if result.error:
            st.error(result.error)
        elif not result.stdout and result.figure is None and not result.warnings:
            st.caption("Ran with no printed output and no figure.")


def _breadcrumb() -> None:
    """Render a compact breadcrumb of the 🎓 AI/ML Tutor lesson a Practise jump arrived from.

    The Tutor stashes ``lab_from`` (``{track, title}``) when its "Practise" button opens the
    Lab; showing it here closes the round-trip loop visually — the learner can see *which*
    lesson they're practising, and the 🎓 button in :func:`_nav_row` jumps back. Absent when
    the Lab is opened directly from the sidebar.
    """
    origin = st.session_state.get("lab_from")
    if not origin:
        return
    st.caption(f"🎓 **AI/ML Tutor** › {origin['track']} › *{origin['title']}*")


def _apply_practice() -> None:
    """Consume a Practice target queued by the 🎓 AI/ML Tutor, preselecting the exercise.

    Runs once per navigation: it seeds the topic / dataset / recipe widget-state keys so the
    learner lands on exactly the mapped exercise, stashes the hint for display, then clears the
    queued target so any later manual selection is respected.
    """
    practice = st.session_state.get("lab_practice")
    if practice is None:
        return
    st.session_state["lab_topic"] = practice.topic
    if practice.topic == TOPIC_LLM:
        # Live-model exercise: preselect it in the AI & LLM picker by its label.
        exercise = LLM_EXERCISES.get(practice.exercise_key)
        if exercise is not None:
            st.session_state["llm_exercise"] = exercise.label
    else:
        dataset = DATASETS.get(practice.dataset_key)
        if dataset is not None:
            st.session_state[f"ds_{practice.topic}"] = dataset.label
        recipe = next((r for r in RECIPES if r.key == practice.recipe_key), None)
        if recipe is not None:
            st.session_state[f"recipe_{practice.topic}"] = recipe.label
    st.session_state["lab_practice_hint"] = practice.hint
    st.session_state["lab_practice"] = None  # consume the navigation intent once


def _nav_row() -> None:
    """A compact toggle to move between the 🔬 Lab and its siblings: 🎓 Tutor and 💬 Chat."""
    c_learn, c_ask, _ = st.columns([2, 2, 5])
    if c_learn.button("🎓 Learn in AI/ML Tutor", width="stretch"):
        go_to_page("ml_tutor", "Open **🎓 AI/ML Tutor** from the sidebar.")
    if c_ask.button("💬 Ask in AI Chat", width="stretch"):
        lesson = st.session_state.get("lab_lesson", "")
        level = st.session_state.get("level", LEVELS[0])
        st.session_state.pending = (lesson or "Explain the concept behind this exercise.", level)
        go_to_page("chat", "Open **💬 AI Chat** from the sidebar to see the answer.")


# ------------------------------------------------------------------------------- AI & LLM
def _judge() -> LLMJudge:
    """Build the evaluation judge on the small, structured-output-capable router model.

    Kept independent of the learner's *answering* model: the router model is cheap and known
    to support ``json_schema`` responses, so judgements stay reliable whatever chat model the
    prompt-comparison or evaluation exercise is answering with.
    """
    return LLMJudge(get_llm(ROUTER_MODEL, temperature=0.0))


def _live_generate(model_id: str, temperature: float):
    """Return a ``(system, human) -> answer`` closure over a freshly built live chat model."""
    chat = get_llm(model_id, temperature=temperature)

    def generate(system: str, human: str) -> str:
        message = chat.invoke([("system", system), ("human", human)])
        return getattr(message, "content", str(message))

    return generate


def _prompt_compare_panel(lesson: str) -> None:
    """Answer one question under several prompt techniques and score each for relevancy."""
    st.markdown(
        "**Compare prompt techniques.** Ask a question, pick a few prompting strategies, and "
        "the app answers with each one **live** — then the evaluation judge scores every answer "
        "for relevancy so you can see what actually helps."
    )
    level = st.session_state.get("level", LEVELS[0])
    question = st.text_area(
        "Your question",
        value=lesson,
        key="pc_question",
        placeholder="e.g. Why does dropout reduce overfitting?",
    )
    techniques = st.multiselect(
        "Prompt techniques to compare",
        list(PROMPT_TECHNIQUES),
        default=["Standard", "Chain-of-Thought", "Analogy-first"],
    )
    if not st.button("▶️ Run comparison", type="primary", key="pc_run"):
        return
    if not question.strip():
        st.info("Type a question to compare techniques on.")
        return

    model_id = st.session_state.get("model", ROUTER_MODEL)
    with st.spinner(f"Answering with {len(techniques)} technique(s) and scoring…"):
        try:
            result = compare_prompt_techniques(
                question,
                techniques,
                level=level,
                model_id=model_id,
                generate=_live_generate(model_id, 0.3),
                judge=_judge(),
            )
        except Exception as exc:  # network/model failure is shown, never crashes the page
            st.error(f"Comparison failed: {type(exc).__name__}: {exc}")
            return
    _render_prompt_compare(result)


def _render_prompt_compare(result: ExerciseResult) -> None:
    """Render a prompt-comparison result: a compact score table plus per-technique answers."""
    if result.error:
        st.info(result.error)
        return
    st.markdown(result.summary)
    st.dataframe(
        [
            {
                "Technique": r["Technique"],
                "Relevancy": r["Relevancy"],
                "Output tokens": r["Output tokens"],
                "Est. cost": f"${r['Est. cost']:.5f}",
            }
            for r in result.rows
        ],
        width="stretch",
        hide_index=True,
    )
    for r in result.rows:
        with st.expander(f"{r['Technique']} — relevancy {r['Relevancy']:.2f}"):
            st.caption(f"Judge: {r['Why']}")
            st.markdown(r["Answer"])


def _llm_eval_panel(lesson: str) -> None:
    """Answer a question live, then score it with the RAGAs-style evaluation metrics."""
    st.markdown(
        "**Evaluate an answer.** Ask a question and get a **live** answer, then score it with "
        "the self-hosted RAGAs-style metrics. Paste reference context to also measure how well "
        "the answer is grounded in it."
    )
    level = st.session_state.get("level", LEVELS[0])
    question = st.text_area(
        "Your question",
        value=lesson,
        key="ev_question",
        placeholder="e.g. What is the bias–variance tradeoff?",
    )
    context = st.text_area(
        "Reference context (optional)",
        key="ev_context",
        placeholder="Passages the answer should be grounded in — separate them with blank lines.",
        help="Used for faithfulness and context metrics. Leave empty to score relevancy only.",
    )
    reference = st.text_input(
        "Reference (golden) answer (optional)",
        key="ev_reference",
        help="Enables context-recall: did the pasted context cover what a good answer needs?",
    )
    if not st.button("▶️ Answer & evaluate", type="primary", key="ev_run"):
        return
    if not question.strip():
        st.info("Type a question to answer and evaluate.")
        return

    model_id = st.session_state.get("model", ROUTER_MODEL)
    contexts = [block.strip() for block in context.split("\n\n") if block.strip()]
    with st.spinner("Answering and scoring…"):
        try:
            answer = _live_generate(model_id, 0.2)(tutor_system(level, ""), question)
            result = evaluate_answer(
                question,
                answer,
                contexts,
                reference.strip() or None,
                judge=_judge(),
            )
        except Exception as exc:  # network/model failure is shown, never crashes the page
            st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")
            return

    if result.error:
        st.info(result.error)
        return
    st.markdown("**Answer**")
    st.markdown(answer)
    st.divider()
    st.markdown(result.summary)
    st.dataframe(result.rows, width="stretch", hide_index=True)


def _ai_llm_lab(lesson: str) -> None:
    """Render the AI & LLM topic: pick a live-model exercise and run it (needs an API key)."""
    exercises = list(LLM_EXERCISES.values())
    labels = {e.label: e for e in exercises}
    choice = st.radio(
        "Exercise",
        list(labels),
        key="llm_exercise",
        captions=[e.description for e in exercises],
    )
    exercise = labels[choice]

    if not openrouter_api_key():
        st.info(
            "The AI & LLM exercises answer and evaluate in real time, so they need an "
            "OpenRouter key. Add `OPENROUTER_API_KEY` to `.env`, then reload this page."
        )
        return

    if exercise.key == "prompt_compare":
        _prompt_compare_panel(lesson)
    else:
        _llm_eval_panel(lesson)


@register_page("🔬 AI/ML Lab", key="ml_lab", section="Learn", order=35)
def render() -> None:
    """AI/ML Lab tab: guided ML/DL/NLP recipes, live AI & LLM exercises, and an opt-in cell."""
    st.subheader("🔬 AI/ML Lab")
    st.caption(
        "Do real work in the browser: run guided ML/DL/NLP recipes on toy datasets, or open the "
        "**AI & LLM** topic to ask a question live, compare prompt techniques, and evaluate "
        "answers — everything seeded by your current **🎓 AI/ML Tutor** lesson."
    )

    _breadcrumb()  # show where a Practise jump arrived from, closing the Tutor→Lab loop
    _apply_practice()  # honour a "Practise" jump from the Tutor before the widgets render

    _nav_row()

    # Keyed container → a `.st-key-lab_topic_row` class the theme CSS targets to lay the four
    # topics out as a 2×2 grid (two rows) instead of letting them wrap to three.
    with st.container(key="lab_topic_row"):
        topic = st.radio(
            "Topic",
            list(_RUNNABLE_TOPICS),
            key="lab_topic",
            horizontal=True,
            captions=list(_RUNNABLE_TOPICS.values()),
        )

    # Difficulty drives the challenge tier (and the live-exercise level). Keyed "level" so it
    # stays in sync with the 🎓 AI/ML Tutor's slider — a Practise jump lands at the same level.
    st.select_slider(
        "Difficulty",
        LEVELS,
        key="level",
        help="Sets the challenge tier: Beginner (run & read) → Practitioner (hit a target) → "
        "Researcher (a stiffer bar or a method constraint).",
    )
    _render_progress()

    lesson = st.session_state.get("lab_lesson", "")
    if lesson:
        st.info(f"📎 Current lesson from AI/ML Tutor: *{lesson}*")
    hint = st.session_state.get("lab_practice_hint", "")
    if hint:
        st.success(f"🔬 Try this: {hint}")

    # AI & LLM is live-model, dataset-free — it needs an API key, not the data-science deps.
    if topic == TOPIC_LLM:
        _ai_llm_lab(lesson)
        return

    ok, missing = _deps()
    if not ok:
        st.warning(
            "This topic needs a few extra libraries that aren't installed yet: "
            f"**{', '.join(missing)}**.\n\n"
            "Add them to `pyproject.toml` and run `make sync`, then reload this page. "
            "(The **AI & LLM** topic works without them.)"
        )
        return

    datasets = datasets_for_topic(topic)
    ds_labels = {d.label: d for d in datasets}
    dataset = ds_labels[st.selectbox("Dataset", list(ds_labels), key=f"ds_{topic}")]
    st.caption(dataset.blurb)

    try:
        data = dataset.loader()
    except Exception as exc:
        st.error(f"Could not load `{dataset.label}`: {type(exc).__name__}: {exc}")
        return

    tab_guided, tab_code = st.tabs(["🧭 Guided recipe", "🧪 Advanced (write code)"])
    with tab_guided:
        _guided(topic, dataset, data)
    with tab_code:
        _advanced(topic, dataset, data, lesson)
