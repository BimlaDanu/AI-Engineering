"""Iapp.py — Streamlit UI layer for the Interview Practice app.

Responsible only for widgets, session state, and rendering. All application
logic lives in core.py, all prompt/technique/role data in categories.py.

Every LLM request in the app is driven by ONE RequestSettings object built
from the sidebar, so the visible model settings (model, temperature, top-p,
frequency penalty, response length token cap) control every request —
coaching, guidelines, question generation, and judging alike.

Run with: streamlit run Iapp.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import auth
from categories import (
    BASE_TECHNIQUES,
    CATEGORIES,
    DIFFICULTY_LEVELS,
    PERSONA_HINTS,
    ROLE_PLAY_PERSONAS,
    ROLES,
    SENIORITY_LEVELS,
    TECHNIQUES,
)
from core import (
    FEATURES,
    CoachingResponse,
    JudgeFeedback,
    RequestSettings,
    estimate_cost,
    evaluate_category_answer,
    generate_category_questions,
    generate_concept_image,
    generate_cover_letter,
    generate_guidelines,
    generate_prep,
    get_last_usage,
    parse_cv,
    run_feature,
    summarize_cv,
    validate_input,
)

load_dotenv()

# Who is asking, and on whose budget? auth.header() draws the title with the
# account controls on the right of the same row, and returns None while the
# session has no key. Nothing below this point runs until it does, so a
# public deployment can never spend the deployer's credit unannounced.
access = auth.header("Interview Practice App")
if access is None:
    auth.welcome()
    st.stop()

client = auth.apply_quota(
    OpenAI(
        api_key=access.api_key,
        base_url="https://openrouter.ai/api/v1",
        # OpenRouter app attribution (see openrouter.ai/docs/app-attribution):
        # identifies this app in the OpenRouter dashboard and rankings.
        default_headers={"X-Title": "Interview Practice App"},
    ),
    access,
)


# ─────────────────────────────────────────────────────────────
# Rendering helpers for the typed structured responses
# ─────────────────────────────────────────────────────────────


def render_coaching(coaching: CoachingResponse) -> None:
    """Render a validated CoachingResponse (Structured output technique)."""
    st.markdown("**Advice**")
    st.write(coaching.advice)
    st.markdown("**Action items**")
    for item in coaching.action_items:
        st.markdown(f"- {item}")
    st.markdown("**Common mistakes to avoid**")
    for item in coaching.common_mistakes:
        st.markdown(f"- {item}")


def render_result(result: "str | CoachingResponse") -> None:
    """Render either a typed coaching result or a plain-text reply."""
    if isinstance(result, CoachingResponse):
        render_coaching(result)
    else:
        st.write(result)


def show_request_cost() -> None:
    """Show token usage and estimated USD price of the request just made.

    Prices come from the OpenRouter models endpoint (fetched once and
    cached); when unavailable, only the token counts are shown.
    """
    usage = get_last_usage()
    if not usage:
        return
    line = f"🧾 Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion"
    cost = estimate_cost(usage["model"], usage["prompt_tokens"], usage["completion_tokens"])
    if cost is not None:
        line += f" • estimated cost: ${cost:.6f}"
    else:
        line += " • price unavailable for this model"
    st.caption(line)


def render_judge_feedback(feedback: JudgeFeedback) -> None:
    """Render validated LLM-as-a-judge feedback with the star rubric."""
    st.markdown(
        f"**Score:** {'⭐' * feedback.score}{'☆' * (5 - feedback.score)}  ({feedback.score}/5)"
    )
    st.markdown(f"**Verdict:** {feedback.verdict}")
    st.markdown("**Strengths**")
    for item in feedback.strengths:
        st.markdown(f"- {item}")
    st.markdown("**Improvements**")
    for item in feedback.improvements:
        st.markdown(f"- {item}")
    with st.expander("💡 Model answer (click to reveal)"):
        st.write(feedback.model_answer)


# ─────────────────────────────────────────────────────────────
# Sidebar — developer settings (single source of every request)
# ─────────────────────────────────────────────────────────────

st.sidebar.header("Developer Settings")

st.sidebar.subheader("Prompt Technique")
selected_technique = st.sidebar.selectbox(
    "Choose a technique:",
    options=BASE_TECHNIQUES,
    help="Different techniques for interview prep coaching",
)

# Persona sub-selector — only appears when Role-play is chosen
if selected_technique == "Role-play":
    selected_persona = st.sidebar.selectbox(
        "Interviewer persona:",
        options=ROLE_PLAY_PERSONAS,
        help="Choose the interviewer's style",
    )
    if selected_persona in PERSONA_HINTS:
        st.sidebar.caption(PERSONA_HINTS[selected_persona])
    effective_technique = selected_persona
else:
    effective_technique = selected_technique

technique = TECHNIQUES[effective_technique]
structured_mode = selected_technique == "Structured output"

st.sidebar.subheader("Interview Role & Seniority")
selected_role = st.sidebar.selectbox(
    "Role:",
    options=ROLES,  # built from ROLE_CONTEXT keys — names cannot drift
    index=0,
    help="Select your target interview role",
)

selected_seniority = st.sidebar.selectbox(
    "Seniority Level:",
    options=SENIORITY_LEVELS,
    index=1,
    help="Select your experience level",
)

selected_difficulty = st.sidebar.selectbox(
    "Question Difficulty:",
    options=DIFFICULTY_LEVELS,
    index=1,
    help="How challenging the practice questions should be",
)

st.sidebar.subheader("Model Settings")
selected_model = st.sidebar.selectbox(
    "Model:",
    options=[
        "openai/gpt-5-mini",
        "openai/gpt-5-nano",
        "openai/gpt-5",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
    ],
    index=0,
    help="OpenRouter model to use (OpenAI and Google Gemini)",
)

temperature = st.sidebar.slider(
    "Temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Controls randomness: 0 = deterministic, 1 = random. "
    "Higher values (>0.8) make answers less predictable.",
)

if temperature > 0.8:
    st.sidebar.warning(
        "⚠️ High temperature makes answers wander. Lower it if the advice drifts off topic."
    )

top_p = st.sidebar.slider(
    "Top-p:",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help="Nucleus sampling: limits choices to top tokens. 1.0 = off. "
    "Usually tune this OR temperature, not both.",
)

frequency_penalty = st.sidebar.slider(
    "Frequency penalty:",
    min_value=-2.0,
    max_value=2.0,
    value=0.0,
    step=0.1,
    help="Higher values discourage repeating the same words. 0 = off.",
)

selected_length = st.sidebar.selectbox(
    "Response Length:",
    options=["Concise", "Detailed"],
    index=1,
    help="Concise = short focused answers. Detailed = in-depth. "
    "Length is set by the prompt; the token cap is only a safety limit.",
)

# The gpt-5 family spends completion tokens on reasoning BEFORE it writes
# anything you can see, and that reasoning is billed against max_tokens. A
# tight cap therefore does not buy a short answer — it buys an empty one.
# So Concise asks the model to think briefly and still leaves room to write;
# the actual brevity comes from the prompt, not from starving the budget.
CONCISE_MAX_TOKENS = 1600
DETAILED_MAX_TOKENS = 4000

# ONE settings object drives EVERY request (coaching, guidelines,
# question generation, judging, and CV summarisation).
settings = RequestSettings(
    model=selected_model,
    temperature=temperature,
    top_p=top_p,
    frequency_penalty=frequency_penalty,
    max_tokens=CONCISE_MAX_TOKENS if selected_length == "Concise" else DETAILED_MAX_TOKENS,
    reasoning_effort="low" if selected_length == "Concise" else None,
)
st.sidebar.caption(
    "ℹ️ These settings apply to every request in the app, including "
    "question generation and answer judging."
)


# ─────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────

st.header("Interview Preparation")

# CV upload stays outside tabs — it applies to all modes
uploaded_file = st.file_uploader("Upload your CV (PDF) — optional", type="pdf", key="cv_uploader")

cv_summary = ""
if uploaded_file:
    if (
        "current_cv_filename" not in st.session_state
        or st.session_state.current_cv_filename != uploaded_file.name
    ):
        st.session_state.current_cv_filename = uploaded_file.name
        st.session_state.cv_text = None
        st.session_state.cv_summary = None

    if "cv_text" not in st.session_state or st.session_state.cv_text is None:
        with st.spinner("Parsing CV..."):
            st.session_state.cv_text = parse_cv(uploaded_file)
            if not st.session_state.cv_text:
                st.warning("Failed to extract text from CV. Continuing without CV context.")

    if st.session_state.cv_text and (
        "cv_summary" not in st.session_state or st.session_state.cv_summary is None
    ):
        with st.spinner("Summarising CV..."):
            st.session_state.cv_summary = summarize_cv(client, settings, st.session_state.cv_text)
            if not st.session_state.cv_summary:
                st.warning("CV summary unavailable (API problem). Continuing without CV context.")

    cv_summary = st.session_state.get("cv_summary") or ""
    if cv_summary:
        with st.expander("📄 CV Summary (click to expand)"):
            st.write(cv_summary)

# Job description (optional) — grounds every request in the actual position
# the user is applying for (lightweight RAG: pasted text is retrieved into
# the USER message of each prompt, never into system messages).
with st.expander("📋 Job description (optional) — paste the position you're applying for"):
    job_description = st.text_area(
        "Job description text:",
        height=180,
        key="job_desc_input",
        help="When filled in, all advice and generated questions target this position.",
    )
    job_description = (job_description or "").strip()
    if job_description:
        jd_ok, jd_message = validate_input(job_description)  # same guard as other inputs
        if not jd_ok:
            st.warning(f"Job description ignored: {jd_message}")
            job_description = ""
        else:
            st.caption("✅ Job description will be used to tailor every request.")

# Five tabs: one per mode
tab_modes, tab_qa, tab_guidelines, tab_practice, tab_cover = st.tabs(
    [
        "🎯 Interview Prep Modes",
        "💬 Classic Q&A",
        "📋 Interviewer Guidelines",
        "🗂️ Category Practice",
        "✉️ Cover Letter",
    ]
)

# ─────────────────────────────────────────────────────────────
# Tab 1: multi-mode feature switcher
# ─────────────────────────────────────────────────────────────
with tab_modes:
    feature_name = st.radio(
        "What do you want to do?",
        list(FEATURES.keys()),
        horizontal=True,
        key="feature_radio",
    )

    mode_input = st.text_area(
        FEATURES[feature_name]["input_label"],
        height=150,
        key="mode_input",
    )

    if st.button("Go", key="feature_btn"):
        ok, message = validate_input(mode_input)
        if not ok:
            st.warning(message)
        else:
            with st.spinner("Thinking..."):
                result = run_feature(
                    client,
                    settings,
                    feature_name=feature_name,
                    user_text=mode_input,
                    technique=technique,
                    cv_summary=cv_summary,
                    job_description=job_description,
                    structured=structured_mode,
                )
            st.subheader(f"Result — {feature_name}  |  {effective_technique}")
            render_result(result)
            show_request_cost()

# ─────────────────────────────────────────────────────────────
# Tab 2: original role-based Q&A
# ─────────────────────────────────────────────────────────────
with tab_qa:
    user_question = st.text_input(
        "Ask an interview preparation question:",
        key="qa_input",
    )

    if st.button("Ask", key="qa_btn"):
        ok, message = validate_input(user_question)
        if not ok:
            st.warning(message)
        else:
            with st.spinner("Thinking..."):
                answer = generate_prep(
                    client,
                    settings,
                    technique=technique,
                    role=selected_role,
                    seniority=selected_seniority,
                    difficulty=selected_difficulty,
                    user_question=user_question,
                    cv_summary=cv_summary,
                    job_description=job_description,
                    response_length=selected_length,
                    structured=structured_mode,
                )
            st.subheader(
                f"Response ({effective_technique} | {selected_role} "
                f"({selected_seniority}) | {settings.model} | "
                f"T={settings.temperature})"
            )
            render_result(answer)
            show_request_cost()

# ─────────────────────────────────────────────────────────────
# Tab 3: Interviewer Guidelines
# ─────────────────────────────────────────────────────────────
with tab_guidelines:
    st.write(
        "Generate structured evaluation criteria an interviewer would use for your target role."
    )
    if st.button("Generate Interviewer Guidelines", key="guidelines_btn"):
        with st.spinner("Generating evaluation criteria..."):
            guidelines = generate_guidelines(client, settings, selected_role, selected_seniority)
        st.subheader(f"Interviewer Guidelines — {selected_role} ({selected_seniority})")
        st.write(guidelines)
        show_request_cost()

# ─────────────────────────────────────────────────────────────
# Tab 4: Category Practice (Advanced feature)
#
# Flow: pick category -> generate 10 questions -> pick one ->
#       type answer -> LLM-as-a-judge feedback (typed JudgeFeedback)
#
# Everything is stored in st.session_state under PER-CATEGORY keys
# ("questions::<category>", "feedback::<category>"), which gives:
#   a. category independence — switching never clobbers another
#      category's questions or feedback
#   b. results survive Streamlit reruns (rendered FROM state, not
#      inside the button's if-block)
# ─────────────────────────────────────────────────────────────
with tab_practice:
    st.write(
        "Pick one skill area and practise it: generate a set of "
        "questions, answer them one at a time, and get judged feedback."
    )

    category = st.radio(
        "Category:",
        list(CATEGORIES.keys()),  # dynamic — new categories appear automatically
        horizontal=True,
        key="cat_radio",
    )
    st.caption(CATEGORIES[category]["description"])

    # Per-category session-state keys -> category independence
    q_key = f"questions::{category}"
    fb_key = f"feedback::{category}"

    col_gen, col_reset = st.columns(2)
    with col_gen:
        if st.button("🎲 Generate 10 questions", key=f"gen_btn_{category}"):
            with st.spinner(f"Generating {category} questions..."):
                questions = generate_category_questions(
                    client,
                    settings,
                    category_name=category,
                    difficulty=selected_difficulty,  # reuses sidebar setting
                    seniority=selected_seniority,  # reuses sidebar setting
                    cv_summary=cv_summary,  # personalises if CV uploaded
                    job_description=job_description,  # targets the pasted position
                )
            show_request_cost()
            if questions:
                st.session_state[q_key] = questions
                st.session_state[fb_key] = {}  # fresh set -> clear old feedback
                st.session_state.pop(f"image::{category}", None)  # and old sketches
                # Drop the picked-question index too: a shorter new set would
                # otherwise leave the selectbox pointing past its last option.
                st.session_state.pop(f"pick_{category}", None)
                if len(questions) < 10:
                    st.info(f"Got {len(questions)} questions (model returned fewer than 10).")
            else:
                st.error("Couldn't generate questions — try again or switch model.")

    with col_reset:
        if st.button("🗑️ Reset this category", key=f"reset_btn_{category}"):
            st.session_state.pop(q_key, None)
            st.session_state.pop(fb_key, None)
            st.session_state.pop(f"image::{category}", None)
            st.session_state.pop(f"pick_{category}", None)

    # Rendered FROM session state -> survives reruns, independent per category
    questions = st.session_state.get(q_key, [])
    if questions:
        st.markdown(f"#### {category} — practice set ({selected_difficulty}, {selected_seniority})")
        for i, q in enumerate(questions, 1):
            st.markdown(f"**{i}.** {q}")

        st.divider()
        st.markdown("#### Answer a question")

        picked = st.selectbox(
            "Pick a question to answer:",
            options=range(len(questions)),
            format_func=lambda i: (
                f"Q{i + 1}: {questions[i][:80]}{'…' if len(questions[i]) > 80 else ''}"
            ),
            key=f"pick_{category}",
        )

        user_answer = st.text_area(
            "Your answer (as you would say it out loud):",
            height=180,
            key=f"answer_{category}_{picked}",  # per-question box
        )

        if st.button("⚖️ Get judged feedback", key=f"judge_btn_{category}"):
            ok, message = validate_input(user_answer)  # same security guard as other tabs
            if not ok:
                st.warning(message)
            else:
                with st.spinner("Judging your answer..."):
                    feedback = evaluate_category_answer(
                        client,
                        settings,
                        category_name=category,
                        question=questions[picked],
                        answer=user_answer,
                    )
                st.session_state.setdefault(fb_key, {})[picked] = feedback
                show_request_cost()

        # Render feedback for the currently selected question from state
        feedback = st.session_state.get(fb_key, {}).get(picked)
        if feedback:
            st.subheader(f"Feedback — Q{picked + 1}")
            if isinstance(feedback, JudgeFeedback):
                render_judge_feedback(feedback)
            else:
                # Fallback: validation failed or an error message; show as text
                st.write(feedback)

        # ── Concept sketch: image generation as a visual memory aid ──
        # Creative use of image generation (google/gemini-2.5-flash-image):
        # a whiteboard-style sketch of the concept behind the picked question,
        # for candidates who memorise visually. Stored per category+question.
        st.divider()
        st.markdown("#### 🎨 Concept sketch")
        st.caption(
            "Generate a whiteboard-style sketch that visualises the concept "
            "behind the selected question — a revision aid for visual learners."
        )
        img_key = f"image::{category}"
        if st.button("Sketch this concept", key=f"img_btn_{category}"):
            with st.spinner("Sketching (google/gemini-2.5-flash-image)..."):
                image_bytes = generate_concept_image(client, questions[picked])
            if image_bytes:
                st.session_state.setdefault(img_key, {})[picked] = image_bytes
            else:
                st.error(
                    "Couldn't generate an image — the image model may be "
                    "unavailable right now. Try again."
                )
        sketch = st.session_state.get(img_key, {}).get(picked)
        if sketch:
            st.image(sketch, caption=f"Concept sketch — Q{picked + 1}")

# ─────────────────────────────────────────────────────────────
# Tab 5: Cover Letter generator
#
# Complements interview prep: drafts a tailored cover letter from the
# uploaded CV (its summary) and the pasted job description above. The
# result is stored in session state so it survives reruns and can be
# edited in place before the user copies it out.
# ─────────────────────────────────────────────────────────────
with tab_cover:
    st.write(
        "Generate a personalised cover letter from your uploaded CV and the "
        "job description above. It's drafted to be tailored, professional, and "
        "ready to edit or submit."
    )

    # Both inputs live above the tabs; show the user what's currently in scope.
    st.markdown(
        f"- **CV:** {'✅ using your uploaded CV summary' if cv_summary else '⚠️ none uploaded'}\n"
        "- **Job description:** "
        f"{'✅ provided above' if job_description else '⚠️ none pasted above'}"
    )
    if not cv_summary or not job_description:
        st.info(
            "For the best result, upload a CV (PDF) and paste the target job "
            "description above. You can still generate with whatever is provided."
        )

    cover_notes = st.text_area(
        "Anything to emphasise? (optional)",
        height=90,
        key="cover_notes",
        help="e.g. 'stress my leadership experience' or 'mention I'm relocating to Berlin'.",
    )
    cover_notes = (cover_notes or "").strip()

    cl_key = "cover_letter_text"

    if st.button("✉️ Generate cover letter", key="cover_btn"):
        if not cv_summary and not job_description:
            st.warning("Please upload a CV or paste a job description above first.")
        else:
            notes_ok = True
            if cover_notes:
                notes_ok, notes_msg = validate_input(cover_notes)  # same guard as other inputs
                if not notes_ok:
                    st.warning(f"Notes ignored: {notes_msg}")
                    cover_notes = ""
            with st.spinner("Writing your cover letter..."):
                letter = generate_cover_letter(
                    client,
                    settings,
                    cv_summary=cv_summary,
                    job_description=job_description,
                    extra_notes=cover_notes,
                )
            st.session_state[cl_key] = letter
            # Streamlit ignores value= once a keyed widget exists, so the
            # edit box below would still show the PREVIOUS letter unless we
            # write the new one into its key here (this runs before the
            # widget is created, which is the only moment it is allowed).
            st.session_state["cover_letter_edit"] = letter
            show_request_cost()

    # Rendered FROM state -> survives reruns; editable in place.
    if st.session_state.get(cl_key):
        st.subheader("Your cover letter")
        st.text_area(
            "Edit before you send it:",
            value=st.session_state[cl_key],
            height=420,
            key="cover_letter_edit",
        )

        save_name = st.text_input(
            "File name:",
            value="cover_letter.txt",
            key="cover_filename",
            help="Saved into the app's working directory. Use a .txt name.",
        )

        # Keep only the basename so the file can never escape the working
        # directory (no paths, no traversal), and make sure it ends in .txt.
        # Download and save share this, so both produce the same file name.
        letter_filename = os.path.basename(save_name.strip()) or "cover_letter.txt"
        if not letter_filename.lower().endswith(".txt"):
            letter_filename += ".txt"

        col_dl, col_save = st.columns(2)
        with col_dl:
            st.download_button(
                "⬇️ Download as .txt",
                data=st.session_state["cover_letter_edit"],
                file_name=letter_filename,
                mime="text/plain",
                key="cover_download",
            )
        with col_save:
            if st.button("💾 Save to working directory", key="cover_save"):
                filename = letter_filename
                target = os.path.join(os.getcwd(), filename)
                try:
                    with open(target, "w", encoding="utf-8") as fh:
                        fh.write(st.session_state["cover_letter_edit"])
                    st.success(f"Saved to {target}")
                except OSError as exc:
                    st.error(f"Couldn't save the file: {exc}")
