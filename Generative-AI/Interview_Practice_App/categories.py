"""categories.py — pure data layer for the Interview Practice app.

WHAT THIS IS
------------
A pure-DATA module: it defines WHICH practice categories, prompt techniques,
and interview roles exist, and WHAT prompt text they use. It contains no API
calls and no Streamlit code, so it can be imported anywhere (core.py, tests,
standalone scripts) with zero side effects.

The three-layer architecture:

    categories.py  ->  data + prompt builders    (WHAT to ask)
    core.py        ->  API orchestration          (HOW to call the LLM)
    Iapp.py        ->  Streamlit UI               (WHERE the user interacts)

ADDING A NEW CATEGORY / TECHNIQUE / ROLE
----------------------------------------
Add ONE entry to the relevant registry dict below. The UI lists registries
dynamically and core.py builds prompts generically, so no other file changes.
"""

from typing import TypedDict

# Single knob for "how many questions per practice set".
QUESTIONS_PER_SET = 10


# ---------------------------------------------------------------------------
# (0) TYPED SPECS
# ---------------------------------------------------------------------------
# TypedDicts make the shape of each registry entry explicit, so core.py and
# the tests can rely on a checked contract instead of implicit dict keys.


class FewShotExample(TypedDict):
    """One demonstration exchange, sent as a real user/assistant message pair."""

    user: str
    assistant: str


class TechniqueSpec(TypedDict):
    """One prompt-engineering technique.

    ``system`` is the system message; ``examples`` are few-shot demonstrations
    sent as alternating user/assistant messages BEFORE the real user message
    (empty list for techniques that need none).
    """

    system: str
    examples: list[FewShotExample]


class CategorySpec(TypedDict):
    """One Category Practice entry (see field guide above CATEGORIES)."""

    description: str
    focus: str
    topics: str
    extra_rules: str
    example_questions: list[str]
    evaluation_focus: str


# ---------------------------------------------------------------------------
# (i) SHARED PROMPT TEMPLATES
# ---------------------------------------------------------------------------
# ONE generation wrapper and ONE evaluation template are shared by ALL
# categories. This is deliberate:
#   * consistency  —> every category returns the same output shape (a JSON
#                    object with a "questions" array), so core.py parses them
#                    all identically
#   * scalability  —> a new category only supplies data ({focus}, {topics},
#                    {extra_rules}, example questions); never a new format
#
# Per-category behaviour is injected through the {placeholders}.
# The JSON shape itself is ENFORCED by core.py via OpenRouter structured
# outputs (response_format + json_schema); the instructions below keep the
# model aligned with that schema.

GENERATION_TEMPLATE = """You are an expert interview question writer.

Generate exactly {n} {difficulty}-level interview questions for the category
"{category}", aimed at a {seniority} candidate.

Category focus: {focus}
Cover a broad mix of these sub-topics: {topics}

Examples of the expected question style (do NOT reuse these verbatim):
{examples}

Rules:
- Questions only — no answers, no explanations, no numbering commentary.
- Each question must be self-contained and answerable verbally in 2-4 minutes.
- Vary the sub-topics so the set covers the category broadly.
{extra_rules}{context_block}
Return ONLY a JSON object with a single key "questions" holding an array of
exactly {n} strings. No markdown fences, no text outside the JSON object."""


# The evaluation wrapper below is the LLM-as-a-judge component, and it is
# also the app's SECOND structured JSON output format. The first one is
# advice/action_items/common_mistakes, used by the "Structured output"
# technique.

EVALUATION_TEMPLATE = """You are a strict but fair interview answer judge (LLM-as-a-judge).

Category: {category}
Judge the answer against these criteria: {evaluation_focus}

Interview question:
\"\"\"{question}\"\"\"

Candidate's answer:
\"\"\"{answer}\"\"\"

Score strictly — a vague or generic answer should not score above 3.

Return ONLY a JSON object with exactly these fields:
- "score": integer from 1 to 5 (1 = poor, 5 = excellent)
- "verdict": one-sentence overall judgement (string)
- "strengths": list of 1-3 things the candidate did well (list of strings)
- "improvements": list of 2-3 specific, actionable fixes (list of strings)
- "model_answer": a concise strong answer to the same question, max 150 words (string)

No markdown fences, no text outside the JSON object."""


# ---------------------------------------------------------------------------
# (ii) THE CATEGORY REGISTRY
# ---------------------------------------------------------------------------
# Each entry is pure data. Field responsibilities:
#
#   description        -> shown to the USER under the category picker
#   focus              -> one line telling the LLM what this category tests
#   topics             -> sub-topics the question set must spread across
#   extra_rules        -> category-specific generation rules (may be "")
#   example_questions  -> 3 seed questions injected as FEW-SHOT style
#                         examples (ties into the course's few-shot
#                         technique) and serve as the "basic questions"
#                         every category is anchored on
#   evaluation_focus   -> the rubric the LLM-as-a-judge scores against

CATEGORIES: dict[str, CategorySpec] = {
    "Python Coding": {
        "description": "Core Python: data structures, algorithms, idioms, and reading/writing real code.",
        "focus": "practical Python programming ability, not trivia",
        "topics": (
            "lists/dicts/sets and their complexity, string manipulation, "
            "comprehensions and generators, OOP vs functions, error handling, "
            "iterators, decorators, typical algorithm patterns (two pointers, "
            "hashing, sorting), debugging"
        ),
        "extra_rules": (
            "- At least 3 questions must include a short code snippet the "
            "candidate has to read, predict, or fix.\n"
            "- At least 2 questions must ask the candidate to describe how "
            "they would implement something and state its time complexity."
        ),
        "example_questions": [
            "What is the difference between a list and a tuple in Python, and when would you choose each?",
            "This function is supposed to remove duplicates while preserving order but it fails — what's wrong and how would you fix it?",
            "How would you find the first non-repeating character in a string, and what is the time complexity of your approach?",
        ],
        "evaluation_focus": (
            "technical correctness, time/space complexity awareness, "
            "Pythonic idioms, handling of edge cases, clarity of explanation"
        ),
    },
    "AI Engineering": {
        "description": "LLMs in production: prompting, APIs, RAG, evaluation, safety, and system design.",
        "focus": "building real applications on top of large language models",
        "topics": (
            "prompt engineering techniques (zero-shot, few-shot, CoT), "
            "system vs user vs assistant roles, LLM settings (temperature, "
            "top-p, penalties), structured outputs, RAG and embeddings, "
            "hallucination and evaluation, prompt-injection security, "
            "cost/latency trade-offs"
        ),
        "extra_rules": (
            "- At least 2 questions must be scenario-based ('your LLM app "
            "does X wrong in production — what do you do?')."
        ),
        "example_questions": [
            "When would you use few-shot prompting instead of zero-shot, and what are its risks?",
            "Your RAG chatbot confidently answers questions with facts that are not in the retrieved documents. How do you diagnose and reduce this?",
            "How do temperature and top-p differ, and why is it usually recommended to tune one rather than both?",
        ],
        "evaluation_focus": (
            "understanding of LLM behaviour, practical production judgement, "
            "awareness of failure modes and security, correct terminology"
        ),
    },
    "Machine Learning": {
        "description": "Classical ML: models, training, evaluation, and the ML lifecycle end to end.",
        "focus": "applied machine learning fundamentals and engineering judgement",
        "topics": (
            "supervised vs unsupervised learning, bias-variance trade-off, "
            "overfitting and regularization, feature engineering, "
            "cross-validation, evaluation metrics (precision/recall/AUC), "
            "imbalanced data, model deployment and monitoring, "
            "gradient descent, tree ensembles vs neural networks"
        ),
        "extra_rules": (
            "- At least 2 questions must involve choosing and justifying an "
            "evaluation metric for a described business problem."
        ),
        "example_questions": [
            "Your model has 99% accuracy on a fraud-detection dataset — why might that be meaningless, and what would you look at instead?",
            "Explain the bias-variance trade-off and how you would detect which side of it your model is on.",
            "You deployed a model and its performance degrades over three months. What are the likely causes and how do you respond?",
        ],
        "evaluation_focus": (
            "conceptual correctness, appropriate metric choice, awareness of "
            "data leakage and overfitting, practical deployment thinking"
        ),
    },
    "Data Science": {
        "description": "Statistics, experimentation, SQL, and communicating insight to stakeholders.",
        "focus": "statistical reasoning and turning data into decisions",
        "topics": (
            "descriptive vs inferential statistics, hypothesis testing and "
            "p-values, A/B test design and pitfalls, confounding, "
            "SQL querying and joins, exploratory data analysis, "
            "data cleaning, visualization choices, storytelling with data"
        ),
        "extra_rules": (
            "- At least 2 questions must describe an A/B-testing or "
            "experiment-design scenario.\n"
            "- At least 1 question must be about explaining a technical "
            "result to a non-technical stakeholder."
        ),
        "example_questions": [
            "A product manager says our A/B test 'nearly reached significance' at p = 0.07 and wants to ship. How do you respond?",
            "What is a confounding variable? Give an example of one ruining an analysis and how you would control for it.",
            "How would you explain to a non-technical executive why more data doesn't automatically mean better conclusions?",
        ],
        "evaluation_focus": (
            "statistical rigour, experiment-design soundness, honesty about "
            "uncertainty, clarity when explaining to non-experts"
        ),
    },
    "Behavioural": {
        "description": "STAR-style questions on teamwork, conflict, failure, leadership, and motivation.",
        "focus": "past-behaviour evidence of soft skills, judged with the STAR framework",
        "topics": (
            "teamwork, conflict resolution, failure and learning, "
            "leadership and initiative, prioritisation under pressure, "
            "receiving and giving feedback, motivation and career goals, "
            "working with difficult stakeholders"
        ),
        "extra_rules": (
            "- Every question must be answerable with a specific past "
            "situation (STAR framework), e.g. 'Tell me about a time...' or "
            "'Describe a situation where...'.\n"
            "- No hypothetical 'what would you do' questions."
        ),
        "example_questions": [
            "Tell me about a time you disagreed with a teammate on a technical decision. How was it resolved?",
            "Describe a situation where a project you owned failed or missed a deadline. What did you learn?",
            "Tell me about a time you had to deliver results with incomplete information or shifting requirements.",
        ],
        "evaluation_focus": (
            "STAR structure (Situation, Task, Action, Result all present), "
            "specificity and personal ownership ('I' not 'we' for actions), "
            "measurable results, reflection and learning"
        ),
    },
}


# ---------------------------------------------------------------------------
# (iii) PROMPT BUILDERS
# ---------------------------------------------------------------------------
# Small pure functions: dict + arguments in, finished prompt string out.
# core.py calls these so it never needs to know how prompts are assembled;
# it just calls the builder with the right category and arguments.


def get_category(name: str) -> CategorySpec:
    """Look up a category by name.

    Raises:
        KeyError: with the list of available categories, instead of a bare
            KeyError, when ``name`` is not registered.
    """
    if name not in CATEGORIES:
        raise KeyError(f"Unknown category '{name}'. Available: {list(CATEGORIES.keys())}")
    return CATEGORIES[name]


def build_generation_prompt(
    category_name: str,
    difficulty: str,
    seniority: str,
    n: int = QUESTIONS_PER_SET,
    cv_summary: str = "",
    job_description: str = "",
) -> str:
    """Build the full 'generate N questions' user prompt for one category.

    Args:
        category_name: key into CATEGORIES (raises KeyError if unknown).
        difficulty: Easy / Medium / Hard.
        seniority: Junior / Mid / Senior.
        n: how many questions to request.
        cv_summary: optional CV summary; when present the model is asked to
            tailor 2-3 questions to it. It is embedded in this USER prompt,
            never in a system message.
        job_description: optional pasted job description; when present the
            model is asked to target the set at that position. Also embedded
            in the USER prompt only.

    Returns:
        The finished prompt string, ready to send as a user message.
    """
    cat = get_category(category_name)

    examples = "\n".join(f"- {q}" for q in cat["example_questions"])

    # Optional grounding context (CV / job description) — user-prompt only.
    context_block = "\n"
    if cv_summary:
        context_block += (
            f"\nCandidate background (tailor 2-3 of the questions to it):\n{cv_summary}\n"
        )
    if job_description:
        context_block += (
            f"\nTarget job description (tailor the set to this position):\n{job_description}\n"
        )

    return GENERATION_TEMPLATE.format(
        n=n,
        difficulty=difficulty,
        category=category_name,
        seniority=seniority,
        focus=cat["focus"],
        topics=cat["topics"],
        examples=examples,
        extra_rules=cat["extra_rules"],
        context_block=context_block,
    )


def build_evaluation_prompt(category_name: str, question: str, answer: str) -> str:
    """Build the LLM-as-a-judge user prompt for one (question, answer) pair.

    Raises:
        KeyError: if ``category_name`` is not registered.
    """
    cat = get_category(category_name)
    return EVALUATION_TEMPLATE.format(
        category=category_name,
        evaluation_focus=cat["evaluation_focus"],
        question=question,
        answer=answer,
    )


# ---------------------------------------------------------------------------
# (iv) PROMPT-TECHNIQUE REGISTRY
# ---------------------------------------------------------------------------
# Each technique = a system message + optional few-shot examples.
# The examples are sent by core.ask_llm() as REAL alternating user/assistant
# messages before the actual user message — not as prose inside the system
# prompt — so the boundary between system, user, and assistant roles stays
# explicit and inspectable.

TECHNIQUES: dict[str, TechniqueSpec] = {
    "Zero-shot": {
        "system": (
            "You are an experienced interview coach. The user is preparing "
            "for interviews. Help them by answering their questions directly "
            "and providing clear, actionable advice."
        ),
        "examples": [],
    },
    "Few-shot": {
        "system": (
            "You are an experienced interview coach. The user is preparing "
            "for interviews. The next messages contain example exchanges that "
            "demonstrate the coaching style, structure, and depth expected. "
            "Follow the same approach for the user's real question."
        ),
        "examples": [
            {
                "user": "How do I answer 'tell me about yourself'?",
                "assistant": (
                    "Use a simple four-part structure:\n"
                    "1. **Who you are** — name, current role, one line of background.\n"
                    "2. **Relevant experience** — the 1-2 achievements that best match this job.\n"
                    "3. **Why this opportunity** — connect your goals to their role.\n"
                    "4. **Closing hook** — one sentence on what you'd bring.\n"
                    "Keep it under 90 seconds and rehearse it out loud."
                ),
            },
            {
                "user": "What should I prepare in the week before the interview?",
                "assistant": (
                    "Prioritise four things:\n"
                    "- **Research the company** — product, recent news, how the team makes money.\n"
                    "- **Practise STAR stories** — prepare 4-5 covering conflict, failure, leadership, delivery.\n"
                    "- **Prepare questions to ask** — 3 specific ones that show research.\n"
                    "- **Do one mock interview** — out loud, timed, ideally with feedback."
                ),
            },
        ],
    },
    "Chain-of-thought": {
        "system": (
            "You are an experienced interview coach. The user is preparing "
            "for interviews. When answering, think step-by-step:\n"
            "1. Identify what type of question or concern the user has "
            "(technical, behavioral, or strategic)\n"
            "2. Consider the user's perspective and what they're likely worried about\n"
            "3. Break down the solution into actionable steps\n"
            "4. Provide specific, concrete advice they can implement\n"
            "Only then provide your response."
        ),
        "examples": [],
    },
    "Role-play (strict)": {
        "system": (
            "You are a strict, professional interviewer conducting a real "
            "interview. The user is the candidate. Your job is to ask tough "
            "but fair questions, listen carefully, and give direct feedback "
            "on their answers. Be encouraging but don't sugarcoat weaknesses. "
            "Help them prepare by role-playing realistic interview scenarios."
        ),
        "examples": [],
    },
    "Role-play (neutral)": {
        "system": (
            "You are a neutral, professional interviewer. Ask standard "
            "questions and give balanced, objective feedback without being "
            "harsh or overly encouraging."
        ),
        "examples": [],
    },
    "Role-play (friendly)": {
        "system": (
            "You are a warm, encouraging interviewer. Ask questions in a "
            "supportive way and give constructive feedback that builds "
            "confidence while still being honest."
        ),
        "examples": [],
    },
    "Structured output": {
        "system": (
            "You are an experienced interview coach. The user is preparing "
            "for interviews. Respond as a JSON object with these fields:\n"
            '- "advice": your main coaching tip (string)\n'
            '- "action_items": 2-3 concrete things they should do (list of strings)\n'
            '- "common_mistakes": 1-2 common mistakes to avoid (list of strings)\n'
            "Keep responses concise and focused."
        ),
        "examples": [],
    },
}

# UI grouping: the base choices shown in the sidebar, and the persona
# sub-choices revealed when "Role-play" is selected.
BASE_TECHNIQUES: list[str] = [
    "Zero-shot",
    "Few-shot",
    "Chain-of-thought",
    "Role-play",
    "Structured output",
]
ROLE_PLAY_PERSONAS: list[str] = [
    "Role-play (strict)",
    "Role-play (neutral)",
    "Role-play (friendly)",
]
PERSONA_HINTS: dict[str, str] = {
    "Role-play (strict)": "🔴 Tough questions, direct feedback, no sugarcoating.",
    "Role-play (neutral)": "🟡 Balanced, standard questions, objective feedback.",
    "Role-play (friendly)": "🟢 Supportive tone, confidence-building feedback.",
}


# ---------------------------------------------------------------------------
# (v) ROLE & SENIORITY REGISTRY
# ---------------------------------------------------------------------------
# Single source of truth for interview roles. The sidebar options are built
# from ROLE_CONTEXT's keys, so the visible role name and the role-context key
# can never drift apart.

ROLE_CONTEXT: dict[str, dict[str, str]] = {
    "AI Engineer": {
        "Junior": "Focus on fundamentals of building on LLMs: prompt engineering (zero-shot, few-shot, chain-of-thought), calling model APIs, system vs user messages, temperature/top-p, embeddings basics. Be ready for: how you'd build a simple chatbot or RAG prototype, structured (JSON) outputs, and how you'd spot and reduce hallucinations.",
        "Mid": "Demonstrate depth in production LLM apps: RAG pipeline design (chunking, retrieval quality, vector stores), evaluating LLM outputs (offline evals, LLM-as-a-judge), prompt-injection defence, function calling / tool use, cost and latency trade-offs. Expect: debugging bad generations, choosing RAG vs fine-tuning, monitoring quality in production.",
        "Senior": "Show system thinking for LLM platforms: model selection and routing across providers, guardrails and safety layers, eval-driven iteration, agentic system design, data privacy, scaling and cost governance. Expect: designing an end-to-end AI product, setting the evaluation strategy, mentoring engineers, judging build-vs-buy and prompt-vs-fine-tune trade-offs.",
    },
    "Machine Learning Engineer": {
        "Junior": "Focus on fundamentals: supervised/unsupervised learning, evaluation metrics, basic ML pipeline. Be ready for: common models, feature engineering basics, how you'd approach a new dataset.",
        "Mid": "Demonstrate depth: feature engineering, model evaluation, cross-validation, hyperparameter tuning. Expect: handling real-world data, debugging models, production concerns, project ownership.",
        "Senior": "Show system thinking: end-to-end ML solutions, stakeholder impact, mentoring. Expect: designing ML systems, data pipeline architecture, business metrics, leading initiatives.",
    },
    "Data Scientist": {
        "Junior": "Focus on fundamentals: statistics, EDA, basic modeling, SQL. Be ready for: interpreting results, A/B testing basics, visualization, communicating findings.",
        "Mid": "Demonstrate depth: statistical rigor, experiment design, complex queries, storytelling. Expect: A/B test design, handling confounding, SQL optimization, presenting to non-technical stakeholders.",
        "Senior": "Show system thinking: impact-driven analytics, strategy, mentoring. Expect: designing analytics from scratch, driving decision-making, cross-team influence, complex experimental design.",
    },
    "Research Scientist (AI/ML)": {
        "Junior": "Focus on fundamentals: research methodology, literature review, basic experiments. Be ready for: explaining your papers, reproducibility, research intuition.",
        "Mid": "Demonstrate depth: novel ideas, rigorous experiments, publishing. Expect: research contributions, literature knowledge, technical depth, pushing boundaries.",
        "Senior": "Show system thinking: research impact, leadership, vision. Expect: novel directions, mentoring researchers, high-impact papers, shaping the field.",
    },
    "Data Analyst": {
        "Junior": "Focus on fundamentals: SQL, basic analytics, visualization, Excel. Be ready for: writing queries, interpreting metrics, simple dashboards.",
        "Mid": "Demonstrate depth: advanced SQL, complex dashboards, self-service analytics, stakeholder management. Expect: designing analytics, query optimization, communicating insights.",
        "Senior": "Show system thinking: analytics strategy, mentoring, business impact. Expect: designing analytics infrastructure, driving insights, leading analytics initiatives.",
    },
}

ROLES: list[str] = list(ROLE_CONTEXT)
SENIORITY_LEVELS: list[str] = ["Junior", "Mid", "Senior"]
DIFFICULTY_LEVELS: list[str] = ["Easy", "Medium", "Hard"]
