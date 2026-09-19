"""Live-model exercises for the 🔬 AI/ML Lab's **AI & LLM** topic.

Unlike the tabular :mod:`src.lab.recipes` (which run vetted code over a bundled dataset), these
exercises are *interactive and model-driven*: the learner types their own question and the app
answers it in real time — no knowledge-base retrieval — then turns the app's **own** evaluation
machinery on the result. Two exercises, both reusing existing infrastructure rather than
duplicating it:

* **prompt_compare** — answer one question under several prompt-engineering techniques (from
  :data:`src.config.PROMPT_TECHNIQUES`) and score each with the :mod:`src.eval` judge, so the
  learner *sees* how prompting changes an answer's relevancy and cost.
* **llm_eval** — run the four RAGAs-style metrics (:func:`src.eval.metrics.evaluate_sample`)
  on a single (question, answer, context) sample, making abstract "LLM evaluation" concrete.

Framework-agnostic by design: no Streamlit here. The two collaborators that touch the network —
answer *generation* and *judging* — are injected (a ``generate`` callable and a
:class:`~src.eval.judge.Judge`), so the whole module is unit-testable offline with stubs, mirror-
ing how :mod:`src.eval` stubs its judge. The page (``src/ui/pages/ml_lab.py``) wires the live
model in.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.config import PROMPT_TECHNIQUES
from src.eval.judge import Judge
from src.eval.metrics import evaluate_sample
from src.utils import estimate_cost, estimate_tokens

# (system_prompt, user_prompt) -> answer text. Injected so this module never imports the LLM
# factory: the page passes a closure over the live chat model; tests pass a deterministic stub.
GenerateFn = Callable[[str, str], str]


@dataclass(frozen=True)
class LLMExercise:
    """Metadata for one AI & LLM exercise — the registry the page and curriculum share."""

    key: str
    label: str
    description: str


@dataclass
class ExerciseResult:
    """The output of an exercise for the page to render: a headline plus a comparison table.

    ``rows`` is a list of plain dicts (no pandas dependency) — each row one technique or one
    metric. The page decides which columns to tabulate and which to expand. ``error`` is set
    (and everything else empty) when the exercise could not run.
    """

    summary: str = ""
    rows: list[dict] = field(default_factory=list)
    error: str | None = None


# The two exercises, registered once and referenced by key from the curriculum's Practice
# mappings (validate_curriculum checks the keys resolve) and the Lab's exercise picker.
LLM_EXERCISES: dict[str, LLMExercise] = {
    "prompt_compare": LLMExercise(
        "prompt_compare",
        "Compare prompt techniques",
        "Answer one question several ways (standard, chain-of-thought, analogy-first, …) and "
        "let the evaluation judge score each for relevancy — see prompting change the answer.",
    ),
    "llm_eval": LLMExercise(
        "llm_eval",
        "Evaluate an answer (RAGAs metrics)",
        "Ask a question, get a live answer, then score it with faithfulness / answer-relevancy / "
        "context metrics — the same self-hosted RAGAs harness the 📊 Evaluation page uses.",
    ),
}


def tutor_system(level: str, technique_instruction: str) -> str:
    """Compose the tutor system prompt for a learner level, plus an optional technique rider."""
    base = (
        f"You are a concise, accurate AI/ML tutor. Explain for a {level}-level learner. "
        "Answer only from established knowledge; if unsure, say so."
    )
    return f"{base} {technique_instruction}".strip()


def compare_prompt_techniques(
    question: str,
    techniques: list[str],
    *,
    level: str,
    model_id: str,
    generate: GenerateFn,
    judge: Judge,
) -> ExerciseResult:
    """Answer ``question`` under each named technique and score every answer for relevancy.

    Args:
        question: The learner's own question (real-time, not knowledge-base grounded).
        techniques: Names from :data:`src.config.PROMPT_TECHNIQUES` to compare.
        level: Learner level, folded into the shared tutor system prompt.
        model_id: The active model id — used only to estimate each answer's cost.
        generate: ``(system, human) -> answer`` closure over the live model (injected).
        judge: A :class:`~src.eval.judge.Judge` used to score answer relevancy.

    Returns:
        An :class:`ExerciseResult` whose ``rows`` carry one entry per technique with its
        relevancy score, output-token count, estimated cost, and the full answer text.
    """
    if not question.strip():
        return ExerciseResult(error="Type a question to compare techniques on.")
    if not techniques:
        return ExerciseResult(error="Pick at least one prompt technique to compare.")

    rows: list[dict] = []
    for technique in techniques:
        system = tutor_system(level, PROMPT_TECHNIQUES.get(technique, ""))
        answer = (generate(system, question) or "").strip()
        relevancy = judge.score_relevancy(question, answer)
        out_tokens = estimate_tokens(answer)
        cost = estimate_cost(model_id, estimate_tokens(system + question), out_tokens)
        rows.append(
            {
                "Technique": technique,
                "Relevancy": round(relevancy.score, 2),
                "Output tokens": out_tokens,
                "Est. cost": cost,
                "Why": relevancy.reason,
                "Answer": answer or "_(the model returned an empty answer)_",
            }
        )

    best = max(rows, key=lambda r: r["Relevancy"])
    summary = (
        f"Compared **{len(rows)}** technique(s). Highest relevancy: **{best['Technique']}** "
        f"({best['Relevancy']:.2f}). Relevancy is judged 0–1; cost is an estimate for the "
        "active model."
    )
    return ExerciseResult(summary=summary, rows=rows)


def evaluate_answer(
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None,
    *,
    judge: Judge,
) -> ExerciseResult:
    """Score one (question, answer, contexts) sample with the four RAGAs-style metrics.

    Thin adapter over :func:`src.eval.metrics.evaluate_sample`: it turns the typed
    :class:`~src.eval.metrics.MetricResult` list into display rows. ``context_recall`` is only
    computed when a ``reference`` answer is supplied (it needs a golden to measure recall).
    """
    if not answer.strip():
        return ExerciseResult(error="There is no answer to evaluate yet.")

    results = evaluate_sample(judge, question, answer, contexts, reference)
    rows = [
        {
            "Metric": m.name.replace("_", " "),
            # Keep the column all-strings: a mix of "—" and floats can't serialise to Arrow
            # (Streamlit's st.dataframe) and warns loudly. Format the number as text instead.
            "Score": "—" if m.score is None else f"{m.score:.2f}",
            "Detail": m.detail,
        }
        for m in results
    ]
    scored = [m.score for m in results if m.score is not None]
    if scored:
        summary = (
            f"Evaluated the answer on **{len(rows)}** metric(s); mean of the "
            f"{len(scored)} defined score(s): **{sum(scored) / len(scored):.2f}** "
            "(higher is better, all in 0–1)."
        )
    else:
        summary = "No metric was defined for this sample (the answer asserts no checkable claim)."
    return ExerciseResult(summary=summary, rows=rows)
