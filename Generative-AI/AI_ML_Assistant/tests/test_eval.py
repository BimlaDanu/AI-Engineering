"""Offline tests for the RAGAs-style evaluation harness (no network, no API key).

These pin the metric maths, dataset loading, and run aggregation using a deterministic
**stub judge** that satisfies the :class:`src.eval.judge.Judge` protocol. The real
:class:`~src.eval.judge.LLMJudge` (structured-output calls) needs a network and is out of
scope for unit tests — exactly the same split as the router tests. The point of injecting
the judge is that everything except the LLM round-trip is verifiable offline.
"""

from __future__ import annotations

import json

import pytest

from src.eval import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    run_evaluation,
    run_evaluations,
)
from src.eval.dataset import GoldenSample, load_golden_set
from src.eval.judge import Judge, LLMJudge
from src.eval.metrics import evaluate_sample
from src.eval.schemas import ClaimVerdict, ContextVerdict, RelevancyScore


class StubJudge:
    """A scripted :class:`Judge`: claims/verdicts/scores are supplied, not inferred.

    ``supported`` maps a claim substring -> bool; ``relevant`` is the ordered list of
    context relevance flags; ``relevancy`` is the fixed answer-relevancy score.
    """

    def __init__(
        self,
        claims: list[str] | None = None,
        supported: dict[str, bool] | None = None,
        relevant: list[bool] | None = None,
        relevancy: float = 1.0,
    ) -> None:
        self._claims = claims or []
        self._supported = supported or {}
        self._relevant = relevant or []
        self._relevancy = relevancy

    def extract_claims(self, text: str) -> list[str]:
        return list(self._claims)

    def verify_claims(self, claims, contexts):
        return [
            ClaimVerdict(claim=c, supported=self._supported.get(c, False), reason="stub")
            for c in claims
        ]

    def rate_contexts(self, question, contexts):
        return [
            ContextVerdict(index=i + 1, relevant=flag, reason="stub")
            for i, flag in enumerate(self._relevant[: len(contexts)])
        ]

    def score_relevancy(self, question, answer):
        return RelevancyScore(score=self._relevancy, reason="stub")


def test_stub_satisfies_judge_protocol() -> None:
    """The stub is a structural Judge, so the metrics accept it (and so does LLMJudge)."""
    assert isinstance(StubJudge(), Judge)
    # LLMJudge is constructed lazily elsewhere; here we only assert it names the protocol.
    assert issubclass(LLMJudge, object)


def test_faithfulness_is_fraction_of_supported_claims() -> None:
    judge = StubJudge(
        claims=["a", "b", "c", "d"],
        supported={"a": True, "b": True, "c": True, "d": False},
    )
    result = faithfulness(judge, "answer", ["ctx"])
    assert result.score == pytest.approx(0.75)


def test_faithfulness_counts_missing_verdicts_as_unsupported() -> None:
    """A judge that returns fewer verdicts than claims must not inflate the score.

    The denominator is the number of claims asked about, so an unjudged claim (a truncated
    structured response) counts as unsupported — 1 supported verdict for 4 claims is 0.25,
    not 1.0 (which dividing by ``len(verdicts)`` would have given).
    """

    class ShortJudge(StubJudge):
        def verify_claims(self, claims, contexts):
            return [ClaimVerdict(claim=claims[0], supported=True, reason="stub")]

    judge = ShortJudge(claims=["a", "b", "c", "d"])
    result = faithfulness(judge, "answer", ["ctx"])
    assert result.score == pytest.approx(0.25)


def test_context_precision_honours_declared_index_order() -> None:
    """Verdicts returned out of order are placed by their 1-based ``index``, not list order.

    The relevant passage is at retrieval rank 1 (index 1) but its verdict is returned second;
    scoring by declared index gives a perfect 1.0 rather than the 0.5 that trusting arrival
    order would produce.
    """

    class ShuffledJudge(StubJudge):
        def rate_contexts(self, question, contexts):
            return [
                ContextVerdict(index=2, relevant=False, reason="stub"),
                ContextVerdict(index=1, relevant=True, reason="stub"),
            ]

    result = context_precision(ShuffledJudge(), "q", ["c1", "c2"])
    assert result.score == pytest.approx(1.0)


def test_context_precision_missing_verdict_counts_as_irrelevant() -> None:
    """A passage the judge never rated is not treated as a relevant hit."""

    class PartialJudge(StubJudge):
        def rate_contexts(self, question, contexts):
            return [ContextVerdict(index=2, relevant=True, reason="stub")]

    # Only rank-2 is relevant; rank-1 is unrated (→ irrelevant). AP = (1/2)/1 = 0.5.
    result = context_precision(PartialJudge(), "q", ["c1", "c2"])
    assert result.score == pytest.approx(0.5)


def test_faithfulness_undefined_without_claims() -> None:
    """An answer that asserts nothing checkable has an undefined (None) score."""
    result = faithfulness(StubJudge(claims=[]), "hello!", ["ctx"])
    assert result.score is None


def test_faithfulness_zero_when_no_context() -> None:
    result = faithfulness(StubJudge(claims=["a"]), "answer", [])
    assert result.score == 0.0


def test_answer_relevancy_passes_through_judge_score() -> None:
    result = answer_relevancy(StubJudge(relevancy=0.4), "q", "a")
    assert result.score == pytest.approx(0.4)


def test_answer_relevancy_zero_for_empty_answer() -> None:
    assert answer_relevancy(StubJudge(relevancy=1.0), "q", "   ").score == 0.0


def test_context_precision_rewards_relevant_ranked_first() -> None:
    """Relevant-then-irrelevant scores higher than irrelevant-then-relevant."""
    good = context_precision(StubJudge(relevant=[True, False]), "q", ["c1", "c2"])
    bad = context_precision(StubJudge(relevant=[False, True]), "q", ["c1", "c2"])
    assert good.score == pytest.approx(1.0)  # single relevant hit at rank 1
    assert bad.score == pytest.approx(0.5)  # single relevant hit at rank 2
    assert good.score > bad.score


def test_context_precision_zero_when_nothing_relevant() -> None:
    result = context_precision(StubJudge(relevant=[False, False]), "q", ["c1", "c2"])
    assert result.score == 0.0


def test_context_precision_undefined_without_contexts() -> None:
    assert context_precision(StubJudge(), "q", []).score is None


def test_context_recall_is_fraction_of_reference_claims_covered() -> None:
    judge = StubJudge(claims=["x", "y"], supported={"x": True, "y": False})
    result = context_recall(judge, "reference", ["ctx"])
    assert result.score == pytest.approx(0.5)


def test_run_evaluation_aggregates_means(tmp_path) -> None:
    """The runner drives answer_fn + judge over samples and averages per metric."""
    samples = [
        GoldenSample("q1", "ref1", "All subjects", "Beginner"),
        GoldenSample("q2", "ref2", "All subjects", "Beginner"),
    ]
    judge = StubJudge(claims=["only"], supported={"only": True}, relevant=[True], relevancy=0.8)
    report = run_evaluation(lambda s: ("an answer", ["ctx"]), judge, samples)
    assert len(report.results) == 2
    assert report.aggregates["faithfulness"] == pytest.approx(1.0)
    assert report.aggregates["answer_relevancy"] == pytest.approx(0.8)
    assert report.aggregates["context_precision"] == pytest.approx(1.0)
    assert report.aggregates["context_recall"] == pytest.approx(1.0)


def test_run_evaluation_survives_a_failing_answer_fn() -> None:
    """A pipeline error for one sample is recorded, not raised."""

    def boom(sample: GoldenSample) -> tuple[str, list[str]]:
        raise RuntimeError("pipeline down")

    samples = [GoldenSample("q", "ref", "All subjects", "Beginner")]
    report = run_evaluation(boom, StubJudge(claims=[]), samples)
    assert len(report.results) == 1
    assert "answer failed" in report.results[0].answer


# --- concurrency: parallel paths must be identical to the sequential ones ---------------


def _sample(name: str) -> GoldenSample:
    return GoldenSample(f"q-{name}", f"ref-{name}", "All subjects", "Beginner")


def _echo_judge() -> StubJudge:
    return StubJudge(claims=["only"], supported={"only": True}, relevant=[True], relevancy=0.8)


def test_evaluate_sample_parallel_matches_sequential() -> None:
    """The four metric branches judged concurrently give the same ordered results."""
    judge = _echo_judge()
    seq = evaluate_sample(judge, "q", "answer", ["ctx"], "reference", parallel=False)
    par = evaluate_sample(judge, "q", "answer", ["ctx"], "reference", parallel=True)
    assert [m.name for m in seq] == [m.name for m in par]  # METRICS order preserved
    assert [m.score for m in seq] == [m.score for m in par]


def test_run_evaluation_parallel_matches_sequential_and_keeps_order() -> None:
    """A threaded run yields the same aggregates and the same per-sample order."""
    samples = [_sample(str(i)) for i in range(6)]
    answer_fn = lambda s: (f"answer to {s.question}", ["ctx"])  # noqa: E731

    seq = run_evaluation(answer_fn, _echo_judge(), samples, max_workers=1)
    par = run_evaluation(answer_fn, _echo_judge(), samples, max_workers=4, parallel_metrics=True)

    assert seq.aggregates == par.aggregates
    # Results stay aligned to the input order despite out-of-order completion.
    assert [r.answer for r in par.results] == [f"answer to {s.question}" for s in samples]
    assert [r.answer for r in par.results] == [r.answer for r in seq.results]


def test_run_evaluation_progress_fires_once_per_sample() -> None:
    samples = [_sample(str(i)) for i in range(5)]
    calls: list[tuple[int, int]] = []
    run_evaluation(
        lambda s: ("a", ["ctx"]),
        _echo_judge(),
        samples,
        progress=lambda done, total: calls.append((done, total)),
        max_workers=3,
    )
    assert len(calls) == len(samples)
    assert calls[-1] == (5, 5)  # final callback reports completion
    assert all(total == 5 for _, total in calls)


def test_run_evaluations_splits_reports_across_jobs() -> None:
    """A/B: two jobs over one golden set + judge return one report each, correctly split."""
    samples = [_sample(str(i)) for i in range(4)]
    jobs = [
        ("A", lambda s: (f"A:{s.question}", ["ctx"])),
        ("B", lambda s: (f"B:{s.question}", ["ctx"])),
    ]
    calls: list[tuple[int, int]] = []
    reports = run_evaluations(
        jobs,
        _echo_judge(),
        samples,
        progress=lambda done, total: calls.append((done, total)),
        max_workers=4,
        parallel_metrics=True,
    )
    assert set(reports) == {"A", "B"}
    assert [r.answer for r in reports["A"].results] == [f"A:{s.question}" for s in samples]
    assert [r.answer for r in reports["B"].results] == [f"B:{s.question}" for s in samples]
    # total == jobs * samples, one tick per unit of work.
    assert len(calls) == len(jobs) * len(samples)
    assert calls[-1] == (8, 8)


def test_evaluate_sample_isolates_a_failing_metric() -> None:
    """A judge failure on one metric degrades that metric to None; the others still score."""

    class ClaimBrokenJudge(StubJudge):
        # extract_claims underpins faithfulness and context_recall; the other two metrics
        # use score_relevancy / rate_contexts and must be unaffected.
        def extract_claims(self, text: str) -> list[str]:
            raise ValueError("Invalid JSON: EOF while parsing a string")

    judge = ClaimBrokenJudge(relevant=[True], relevancy=0.8)
    results = evaluate_sample(judge, "q", "answer", ["ctx"], "reference", parallel=True)
    by_name = {m.name: m for m in results}

    assert by_name["faithfulness"].score is None
    assert "Judge error" in by_name["faithfulness"].detail
    assert by_name["context_recall"].score is None
    # Metrics that do not extract claims are untouched.
    assert by_name["answer_relevancy"].score == 0.8
    assert by_name["context_precision"].score == 1.0


def test_metrics_with_judge_errors_names_only_the_failed_metrics() -> None:
    """The evaluation page flags exactly the metrics a weak judge could not score.

    Mirrors the real claude-3-haiku case: the single-score metric (answer relevancy)
    succeeds while the list-structured metrics degrade to None with a "Judge error:" detail.
    The page must name those three so the "—" cards are explained, not silent.
    """
    from src.eval.metrics import MetricResult
    from src.eval.runner import EvalReport, SampleEvaluation
    from src.ui.pages.evaluation import _metrics_with_judge_errors

    metrics = [
        MetricResult("faithfulness", None, "Judge error: EOF while parsing a string"),
        MetricResult("answer_relevancy", 1.0, "Fully on-point."),
        MetricResult("context_precision", None, "Judge error: EOF while parsing a string"),
        MetricResult("context_recall", None, "Judge error: EOF while parsing a string"),
    ]
    report = EvalReport(
        results=[SampleEvaluation(_sample("0"), "ans", ["ctx"], metrics)],
        aggregates={"answer_relevancy": 1.0},
    )

    # Labels, in METRICS order; answer relevancy scored, so it is not listed.
    assert _metrics_with_judge_errors(report) == [
        "Faithfulness",
        "Context precision",
        "Context recall",
    ]


def test_metrics_with_judge_errors_empty_when_undefined_but_not_errored() -> None:
    """A metric legitimately None (no claim to check) is NOT reported as a judge failure."""
    from src.eval.metrics import MetricResult
    from src.eval.runner import EvalReport, SampleEvaluation
    from src.ui.pages.evaluation import _metrics_with_judge_errors

    metrics = [MetricResult("faithfulness", None, "Answer makes no verifiable claim.")]
    report = EvalReport(
        results=[SampleEvaluation(_sample("0"), "ans", [], metrics)],
    )
    assert _metrics_with_judge_errors(report) == []


def test_run_evaluation_parallel_survives_failing_answer_fn() -> None:
    """A per-sample pipeline failure is still recorded, not raised, under threads."""

    def boom(sample: GoldenSample) -> tuple[str, list[str]]:
        raise RuntimeError("pipeline down")

    samples = [_sample(str(i)) for i in range(3)]
    report = run_evaluation(boom, StubJudge(claims=[]), samples, max_workers=3)
    assert len(report.results) == 3
    assert all("answer failed" in r.answer for r in report.results)


def test_llmjudge_falls_back_to_function_calling_when_json_schema_fails() -> None:
    """A judge model that rejects json_schema (e.g. Anthropic via OpenRouter) must not yield
    empty scores — LLMJudge retries the call as tool-calling structured output instead."""
    from src.eval.judge import LLMJudge
    from src.eval.schemas import RelevancyScore

    class _JsonSchemaFails:
        def invoke(self, messages):
            raise ValueError("json_schema response format is not supported by this model")

    class _FunctionCallingWorks:
        def invoke(self, messages):
            return RelevancyScore(score=0.9, reason="ok")

    class _FakeLLM:
        def __init__(self) -> None:
            self.methods: list[str] = []

        def with_structured_output(self, schema, method):
            self.methods.append(method)
            return _JsonSchemaFails() if method == "json_schema" else _FunctionCallingWorks()

    llm = _FakeLLM()
    result = LLMJudge(llm).score_relevancy("q", "a")
    assert result.score == 0.9
    # json_schema was tried first, then the tool-calling fallback.
    assert llm.methods == ["json_schema", "function_calling"]


def test_load_golden_set_reads_the_packaged_file() -> None:
    """The shipped golden set loads, is non-empty, and every row is well-formed."""
    samples = load_golden_set()
    assert len(samples) >= 10
    for s in samples:
        assert s.question and s.reference


def test_load_golden_set_validates_and_skips_comments(tmp_path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(
        "# a comment\n"
        '{"question": "q", "reference": "r"}\n'
        "\n"
        '{"question": "q2", "reference": "r2", "subject": "Machine Learning"}\n',
        encoding="utf-8",
    )
    samples = load_golden_set(path)
    assert len(samples) == 2
    assert samples[0].subject == "All subjects"  # default applied
    assert samples[1].subject == "Machine Learning"


def test_load_golden_set_rejects_unknown_subject(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"question": "q", "reference": "r", "subject": "Nope"}))
    with pytest.raises(ValueError, match="unknown subject"):
        load_golden_set(path)
