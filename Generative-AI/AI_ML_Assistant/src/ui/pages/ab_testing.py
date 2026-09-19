"""🆚 A/B testing: compare two RAG strategies head-to-head on the same golden set.

This page reuses the evaluation harness wholesale. It defines two *variants* — each a choice
of answer model plus a few retrieval-strategy knobs (top-k, hybrid α, query rewriting,
pipeline engine) — then runs both variants through :func:`~src.eval.run_evaluations` (one
shared thread pool that overlaps them) over the **same** golden subset scored by the **same**
judge, and diffs the aggregate metrics with the pure comparator in :mod:`src.eval.compare`.

Because both variants answer identical questions under an identical judge, the only thing
that varies is the strategy under test — so the per-metric deltas and the overall winner are
a fair comparison. Like 📊 Evaluation, it is **opt-in**: nothing runs (and no tokens are
spent) until you click *Run A/B test*, and a run costs roughly twice a single evaluation.
"""

from __future__ import annotations

import json
from dataclasses import replace

import streamlit as st

from src.config import OPENROUTER_MODELS, ROUTER_MODEL, SUBJECTS, openrouter_api_key
from src.core.service import AnswerRequest
from src.eval import LLMJudge, Variant, compare_reports, load_golden_set, run_evaluations
from src.eval.compare import TIE, WINNER_A, ABComparison
from src.eval.dataset import GoldenSample
from src.eval.runner import EvalReport
from src.llm import get_llm
from src.ui.registry import register_page
from src.ui.state import get_service

# The retrieval-strategy knobs a variant may override on top of the current settings. Kept
# small on purpose: these are the levers that most change retrieval/answer quality, so an A/B
# over them is meaningful without turning the page into a second Settings tab.
_ENGINES = ["linear", "graph"]


def _variant_settings(base, *, top_k: int, alpha: float, rewrite: bool, engine: str):
    """Return a copy of ``base`` settings with this variant's strategy overrides applied."""
    return replace(
        base,
        top_k=top_k,
        hybrid_alpha=alpha,
        rewrite_query=rewrite,
        pipeline_engine=engine,
    )


def _config_summary(model: str, *, top_k: int, alpha: float, rewrite: bool, engine: str) -> str:
    """One-line, human-readable description of a variant's configuration."""
    return (
        f"{model} · k={top_k} · α={alpha:.2f} · "
        f"rewrite={'on' if rewrite else 'off'} · engine={engine}"
    )


def _answer_fn_factory(model: str, settings):
    """Build an ``answer_fn`` running one golden sample through the pipeline with ``settings``."""
    service = get_service()

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


def _run_ab(
    samples: list[GoldenSample],
    cfg_a: dict,
    cfg_b: dict,
    base,
    judge_model: str,
    workers: int,
) -> tuple[EvalReport, EvalReport]:
    """Evaluate both variants over ``samples`` in one shared pool with a shared judge.

    Both variants and every sample within them are scheduled into a single thread pool, so
    the two variants overlap rather than running one fully before the other. The judge is
    built once and shared, which is exactly the fairness contract the comparator assumes.
    """
    judge = LLMJudge(get_llm(judge_model, temperature=0.0))
    jobs = [
        ("A", _answer_fn_factory(cfg_a["model"], _variant_settings(base, **_strategy(cfg_a)))),
        ("B", _answer_fn_factory(cfg_b["model"], _variant_settings(base, **_strategy(cfg_b)))),
    ]
    bar = st.progress(0.0, text="Running A/B…")

    def progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Running A/B… {done}/{total}")

    reports = run_evaluations(
        jobs,
        judge,
        samples,
        progress=progress,
        max_workers=workers,
        parallel_metrics=workers > 1,
    )
    bar.empty()
    report_a, report_b = reports["A"], reports["B"]
    report_a.answer_model, report_a.judge_model = cfg_a["model"], judge_model
    report_b.answer_model, report_b.judge_model = cfg_b["model"], judge_model
    return report_a, report_b


def _variant_controls(label: str, default_model_idx: int) -> dict:
    """Render one variant's config controls and return the chosen values."""
    st.markdown(f"**Variant {label}**")
    model = st.selectbox(
        "Answer model",
        OPENROUTER_MODELS,
        index=default_model_idx,
        key=f"ab_model_{label}",
    )
    top_k = st.slider("Top-k passages", 1, 10, 4, key=f"ab_topk_{label}")
    alpha = st.slider(
        "Hybrid α (vector ↔ BM25)",
        0.0,
        1.0,
        0.7,
        0.05,
        key=f"ab_alpha_{label}",
        help="1.0 = pure vector similarity; 0.0 = pure keyword (BM25).",
    )
    rewrite = st.checkbox("Rewrite query", value=True, key=f"ab_rewrite_{label}")
    engine = st.selectbox("Pipeline engine", _ENGINES, key=f"ab_engine_{label}")
    return {"model": model, "top_k": top_k, "alpha": alpha, "rewrite": rewrite, "engine": engine}


def _render_comparison(comparison: ABComparison) -> None:
    """Render the overall verdict, the per-metric table, config captions, and a JSON export."""
    a, b = comparison.variant_a, comparison.variant_b
    if comparison.overall_winner == TIE:
        st.info(f"**Result: tie** — {comparison.wins_a}–{comparison.wins_b} on decided metrics.")
    else:
        winner = a if comparison.overall_winner == WINNER_A else b
        st.success(
            f"**Winner: Variant {winner.label}** "
            f"({comparison.wins_a}–{comparison.wins_b} on decided metrics)."
        )

    st.caption(f"**A** — {a.config}")
    st.caption(f"**B** — {b.config}")

    st.markdown("### Per-metric scores")
    st.caption("All metrics are higher-is-better; Δ is B − A.")
    st.dataframe(comparison.as_rows(), width="stretch", hide_index=True)

    payload = {
        "variant_a": {"label": a.label, "config": a.config},
        "variant_b": {"label": b.label, "config": b.config},
        "overall_winner": comparison.overall_winner,
        "wins": {"A": comparison.wins_a, "B": comparison.wins_b},
        "metrics": comparison.as_rows(),
    }
    st.download_button(
        "⬇️ Comparison (JSON)",
        json.dumps(payload, indent=2),
        file_name="ab_comparison.json",
        width="stretch",
    )


@register_page("🆚 A/B testing", key="ab_testing", section="Analyse", order=60)
def render() -> None:
    """A/B workspace: compare two RAG strategies on the same golden set and judge."""
    st.subheader("🆚 A/B testing")
    st.caption(
        "Compare **two** RAG strategies head-to-head on the same golden set and judge, so "
        "only the strategy under test varies. To score just one, see 📊 Evaluation."
    )

    if not openrouter_api_key():
        st.warning("Set `OPENROUTER_API_KEY` in `.env` to run an A/B test (it needs LLM calls).")
        return

    try:
        samples = load_golden_set()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load the golden set: {exc}")
        return

    st.info(
        f"Golden set: **{len(samples)}** questions. A run answers each question **twice** "
        "(once per variant) and judges every answer — so it costs roughly double a single "
        "evaluation. It runs only when you click below."
    )

    col_n, col_judge = st.columns(2)
    with col_n:
        n = st.slider("Questions to evaluate", 1, len(samples), min(5, len(samples)))
    with col_judge:
        default_judge = ROUTER_MODEL if ROUTER_MODEL in OPENROUTER_MODELS else OPENROUTER_MODELS[0]
        judge_model = st.selectbox(
            "Judge model (shared)",
            OPENROUTER_MODELS,
            index=OPENROUTER_MODELS.index(default_judge),
            help="A small, cheap model at temperature 0 is recommended for stable judgements.",
        )

    workers = st.slider(
        "⚡ Parallel requests",
        1,
        16,
        8,
        help="How many answers to evaluate at once, across both variants combined. These "
        "calls only wait on OpenRouter, so higher is faster with no extra load on your "
        "machine — lower it if the judge model starts rate-limiting (1 = fully sequential).",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        cfg_a = _variant_controls("A", default_model_idx=0)
    with col_b:
        cfg_b = _variant_controls("B", default_model_idx=min(1, len(OPENROUTER_MODELS) - 1))

    base = st.session_state.settings
    if st.button("▶️ Run A/B test", type="primary"):
        try:
            report_a, report_b = _run_ab(samples[:n], cfg_a, cfg_b, base, judge_model, workers)
            st.session_state.ab_comparison = compare_reports(
                report_a,
                report_b,
                Variant("A", _config_summary(cfg_a["model"], **_strategy(cfg_a))),
                Variant("B", _config_summary(cfg_b["model"], **_strategy(cfg_b))),
            )
        except Exception as exc:  # network / key / model failure must not crash the page
            st.error(f"A/B test failed: {exc}")
            return

    comparison = st.session_state.get("ab_comparison")
    if comparison is not None:
        st.divider()
        _render_comparison(comparison)


def _strategy(cfg: dict) -> dict:
    """Strip the model out of a variant config, leaving only the strategy knobs."""
    return {
        "top_k": cfg["top_k"],
        "alpha": cfg["alpha"],
        "rewrite": cfg["rewrite"],
        "engine": cfg["engine"],
    }
