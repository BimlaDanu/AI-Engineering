"""📊 Evaluation: measure RAG quality with RAGAs-style metrics over a golden set.

This page runs the real answer pipeline over the curated golden questions
(``data/eval/golden.jsonl``) and scores each answer with four LLM-judged metrics —
faithfulness, answer relevancy, context precision, and context recall (see
:mod:`src.eval`). It is **opt-in**: nothing runs until you click *Run evaluation*, because a
run makes several judge calls per question. Results show as aggregate metric cards, a
per-question table, and downloadable JSON/CSV reports.
"""

from __future__ import annotations

import csv
import io
import json

import streamlit as st

from src.config import OPENROUTER_MODELS, ROUTER_MODEL, SUBJECTS, openrouter_api_key
from src.core.service import AnswerRequest
from src.eval import LLMJudge, load_golden_set, run_evaluation
from src.eval.dataset import GoldenSample
from src.eval.metrics import METRICS
from src.eval.runner import EvalReport
from src.llm import get_llm
from src.ui.registry import register_page
from src.ui.state import get_service

# Human-friendly labels and one-line explanations for the metric cards.
_METRIC_LABELS: dict[str, tuple[str, str]] = {
    "faithfulness": ("Faithfulness", "Answer claims grounded in retrieved context."),
    "answer_relevancy": ("Answer relevancy", "How directly the answer addresses the question."),
    "context_precision": ("Context precision", "Relevant passages ranked first (avg precision)."),
    "context_recall": ("Context recall", "Reference-answer claims covered by retrieval."),
}


def _metrics_with_judge_errors(report: EvalReport) -> list[str]:
    """Metric labels whose score is undefined because the *judge* failed, not the input.

    A :class:`~src.eval.metrics.MetricResult` whose ``detail`` begins with ``"Judge error:"``
    means the judge model could not emit valid structured output for that metric's schema
    even after the json_schema→tool-calling fallback and retries — as opposed to a metric
    that is legitimately undefined for the input (e.g. an answer that asserts no claim).

    Weaker judge models routinely manage the single-score metric (answer relevancy) but not
    the *list-structured* ones (faithfulness, context precision, context recall), which need
    an array of JSON verdicts. Surfacing exactly which metrics failed turns a silent row of
    "—" into an actionable message. Returns the human labels in :data:`METRICS` order.
    """
    failed: list[str] = []
    for metric in METRICS:
        name = metric["name"]
        if any(
            m.name == name and m.detail.startswith("Judge error:")
            for r in report.results
            for m in r.metrics
        ):
            failed.append(_METRIC_LABELS.get(name, (name, ""))[0])
    return failed


def _answer_fn_factory(model: str):
    """Build an ``answer_fn`` that runs one golden sample through the real pipeline."""
    service = get_service()
    settings = st.session_state.settings

    def answer_fn(sample: GoldenSample) -> tuple[str, list[str]]:
        req = AnswerRequest(
            question=sample.question,
            level=sample.level,
            model=model,
            subjects=SUBJECTS.get(sample.subject),
            settings=settings,
            history=[],
        )
        bundle = service.answer(req)
        return bundle.text, bundle.contexts

    return answer_fn


def _run(
    samples: list[GoldenSample], answer_model: str, judge_model: str, workers: int
) -> EvalReport:
    """Execute an evaluation run with a live progress bar and return the report."""
    judge = LLMJudge(get_llm(judge_model, temperature=0.0))
    answer_fn = _answer_fn_factory(answer_model)
    bar = st.progress(0.0, text="Evaluating…")

    def progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Evaluating… {done}/{total}")

    # workers > 1 fans the samples out over a thread pool; because each sample is just
    # waiting on OpenRouter, this cuts wall-clock roughly linearly with no local CPU load.
    report = run_evaluation(
        answer_fn,
        judge,
        samples,
        progress=progress,
        max_workers=workers,
        parallel_metrics=workers > 1,
    )
    report.answer_model = answer_model
    report.judge_model = judge_model
    bar.empty()
    return report


def _render_report(report: EvalReport) -> None:
    """Render aggregate metric cards, the per-question table, and export buttons."""
    if not report.aggregates:
        st.warning(
            "The judge model returned no usable scores for any question — it likely does not "
            "support structured output. Try a different judge model (e.g. "
            "`openai/gpt-4o-mini`). Expand a question below to see the exact judge error."
        )
    elif failed := _metrics_with_judge_errors(report):
        st.warning(
            f"`{report.judge_model}` couldn't produce structured output for "
            f"**{', '.join(failed)}**, so they show “—”. Weaker or older judge "
            "models often manage the single-score metric (answer relevancy) but not the "
            "list-structured metrics, which need an array of JSON verdicts. For a complete "
            "run, pick a structured-output-capable judge such as `openai/gpt-4o-mini`. "
            "Expand a question below to see the exact judge error."
        )
    st.markdown("### Aggregate scores")
    st.caption(f"Answered by `{report.answer_model}` · judged by `{report.judge_model}`.")
    cols = st.columns(len(METRICS))
    for col, metric in zip(cols, METRICS, strict=True):
        name = metric["name"]
        label, help_text = _METRIC_LABELS[name]
        value = report.aggregates.get(name)
        col.metric(label, "—" if value is None else f"{value:.2f}", help=help_text)

    st.markdown("### Per-question scores")
    rows = report.as_rows()
    st.dataframe(rows, width="stretch", hide_index=True)

    with st.expander("🔍 Inspect answers and reasoning"):
        for r in report.results:
            st.markdown(f"**Q:** {r.sample.question}")
            for m in r.metrics:
                score = "—" if m.score is None else f"{m.score:.2f}"
                st.caption(f"{m.name}: {score} — {m.detail}")
            st.divider()

    payload = {
        "answer_model": report.answer_model,
        "judge_model": report.judge_model,
        "aggregates": {k: round(v, 4) for k, v in report.aggregates.items()},
        "samples": rows,
    }
    csv_buf = io.StringIO()
    writer = csv.DictWriter(
        csv_buf, fieldnames=["question", "subject", "level", *[m["name"] for m in METRICS]]
    )
    writer.writeheader()
    writer.writerows(rows)

    col_json, col_csv = st.columns(2)
    col_json.download_button(
        "⬇️ Report (JSON)",
        json.dumps(payload, indent=2),
        file_name="evaluation.json",
        width="stretch",
    )
    col_csv.download_button(
        "⬇️ Scores (CSV)",
        csv_buf.getvalue(),
        file_name="evaluation.csv",
        width="stretch",
    )


@register_page("📊 Evaluation", key="evaluation", section="Analyse", order=55)
def render() -> None:
    """Evaluation workspace: run RAGAs-style metrics over the curated golden set."""
    st.subheader("📊 Evaluation")
    st.caption(
        "Score **one** RAG configuration in absolute terms — RAGAs-style faithfulness, "
        "answer relevancy, context precision, and context recall over a curated golden set. "
        "To compare two configurations, see 🆚 A/B testing."
    )

    if not openrouter_api_key():
        st.warning("Set `OPENROUTER_API_KEY` in `.env` to run an evaluation (it needs LLM calls).")
        return

    try:
        samples = load_golden_set()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load the golden set: {exc}")
        return

    st.info(
        f"Golden set: **{len(samples)}** questions. A run answers each with the current "
        "model and settings, then makes ~6 judge calls per question — so it costs tokens. "
        "It runs only when you click below."
    )

    col_n, col_judge = st.columns(2)
    with col_n:
        n = st.slider("Questions to evaluate", 1, len(samples), min(5, len(samples)))
    with col_judge:
        default_judge = ROUTER_MODEL if ROUTER_MODEL in OPENROUTER_MODELS else OPENROUTER_MODELS[0]
        judge_model = st.selectbox(
            "Judge model",
            OPENROUTER_MODELS,
            index=OPENROUTER_MODELS.index(default_judge),
            help="A small, cheap model at temperature 0 is recommended for stable judgements.",
        )

    workers = st.slider(
        "⚡ Parallel requests",
        1,
        16,
        8,
        help="How many questions to evaluate at once. These calls only wait on OpenRouter, "
        "so higher is faster with no extra load on your machine — lower it if the judge "
        "model starts rate-limiting (1 = fully sequential).",
    )

    answer_model = st.session_state.get("model", OPENROUTER_MODELS[0])
    st.caption(
        f"Answers generated by the active model: `{answer_model}` (change it in ⚙️ Settings)."
    )

    if st.button("▶️ Run evaluation", type="primary"):
        try:
            st.session_state.eval_report = _run(samples[:n], answer_model, judge_model, workers)
        except Exception as exc:  # network / key / model failure must not crash the page
            st.error(f"Evaluation failed: {exc}")
            return

    report = st.session_state.get("eval_report")
    if report is not None:
        st.divider()
        _render_report(report)
