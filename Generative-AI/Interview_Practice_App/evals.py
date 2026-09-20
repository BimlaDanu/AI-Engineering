"""evals.py — fixed evaluation set for comparing prompt techniques or models.

Assesses PROMPT/MODEL performance rather than candidate answers: this
module holds a fixed set of 12 evaluation cases and a runner
that generates a coaching response for each case with every candidate
(technique or model), then scores every response with the SAME fixed judge
(fixed judge model, temperature 0, fixed rubric). Because the cases, rubric,
and judge are constant, runs are repeatable and score differences can be
attributed to the technique/model under test.

Usage (developer tool — makes 2 API calls per case per candidate):

    # Compare all prompt techniques on the first 3 cases
    uv run python evals.py --mode techniques --limit 3

    # Compare models using the Zero-shot technique on every case
    uv run python evals.py --mode models --limit 12

    # Save the raw per-case results as JSON
    uv run python evals.py --mode techniques --out eval_results.json

    # Serial run (easier to read the log), or more parallelism
    uv run python evals.py --mode techniques --limit 3 --workers 1
    uv run python evals.py --mode techniques --workers 12

    # Show the per-call INFO lines (latency, tokens, request id)
    uv run python evals.py --mode techniques --limit 3 --verbose

Speed/cost: cases run concurrently (--workers, default 6) and every request
asks for LOW reasoning effort. gpt-5* models spend completion tokens thinking
before emitting visible text, so low effort cuts both the token bill and the
latency. The cases, rubric, judge model and temperature stay fixed, so runs
remain comparable with each other.

Requires OPENROUTER_API_KEY in .env (same as the app). Never imports
Streamlit; results print as a markdown table.
"""

import argparse
import json
import logging
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from categories import TECHNIQUES
from core import (
    RequestSettings,
    ask_llm,
    generate_prep,
    json_schema_format,
    logger,
    parse_json_response,
)


class EvalCase(TypedDict):
    """One fixed evaluation case: a realistic interview-prep request."""

    id: str
    role: str
    seniority: str
    difficulty: str
    question: str


# ---------------------------------------------------------------------------
# The fixed eval set (12 cases). Do not edit between comparison runs —
# repeatability is what makes the scores comparable.
# ---------------------------------------------------------------------------

EVAL_CASES: list[EvalCase] = [
    {
        "id": "prep-plan",
        "role": "AI Engineer",
        "seniority": "Junior",
        "difficulty": "Easy",
        "question": "I have one week before my first AI engineer interview. What should I focus on?",
    },
    {
        "id": "tell-me-about-yourself",
        "role": "Data Analyst",
        "seniority": "Junior",
        "difficulty": "Easy",
        "question": "How should I structure my answer to 'tell me about yourself'?",
    },
    {
        "id": "transformers",
        "role": "AI Engineer",
        "seniority": "Mid",
        "difficulty": "Medium",
        "question": "How deep should I be able to explain transformer attention in an interview?",
    },
    {
        "id": "star-weakness",
        "role": "Machine Learning Engineer",
        "seniority": "Mid",
        "difficulty": "Medium",
        "question": "How do I talk about a failed ML project without sounding incompetent?",
    },
    {
        "id": "metrics",
        "role": "Machine Learning Engineer",
        "seniority": "Mid",
        "difficulty": "Medium",
        "question": "What evaluation-metric questions should I expect for a fraud-detection role?",
    },
    {
        "id": "ab-testing",
        "role": "Data Scientist",
        "seniority": "Mid",
        "difficulty": "Medium",
        "question": "What A/B testing pitfalls do interviewers usually probe for?",
    },
    {
        "id": "sql-prep",
        "role": "Data Analyst",
        "seniority": "Mid",
        "difficulty": "Medium",
        "question": "How should I prepare for a live SQL screening round?",
    },
    {
        "id": "salary",
        "role": "Data Scientist",
        "seniority": "Senior",
        "difficulty": "Medium",
        "question": "How do I answer questions about salary expectations without underselling myself?",
    },
    {
        "id": "system-design",
        "role": "AI Engineer",
        "seniority": "Senior",
        "difficulty": "Hard",
        "question": "How do I approach an ML system design interview for a recommendation system?",
    },
    {
        "id": "research-talk",
        "role": "Research Scientist (AI/ML)",
        "seniority": "Senior",
        "difficulty": "Hard",
        "question": "How should I present my research contributions to a mixed technical panel?",
    },
    {
        "id": "leadership",
        "role": "Machine Learning Engineer",
        "seniority": "Senior",
        "difficulty": "Hard",
        "question": "What leadership questions should I expect when moving from senior IC to team lead?",
    },
    {
        "id": "no-experience",
        "role": "AI Engineer",
        "seniority": "Junior",
        "difficulty": "Hard",
        "question": "How do I handle questions about production LLM experience when I only have side projects?",
    },
]

# The fixed rubric every response is judged against, regardless of which
# technique/model produced it.
# Deliberately strict. A lenient rubric scored every technique 5/5, which made
# the comparison useless — the point is to SEPARATE the techniques, so the
# anchors below make 5 hard to reach and push fluent-but-generic answers to 3.
# Changing this rubric invalidates comparisons with earlier runs; keep it fixed
# across the runs you intend to compare.
EVAL_DIMENSIONS = ("specificity", "actionability", "correctness", "insight")
# Short column labels for the results table.
DIMENSION_LABELS = {
    "specificity": "Spec",
    "actionability": "Action",
    "correctness": "Correct",
    "insight": "Insight",
}

# Deliberately strict, and scored per dimension rather than as one overall mark.
# History: a lenient overall 1-5 scale gave every technique 5.00; tightening the
# wording just moved every technique to 4.00. A single integer has too little
# resolution to separate seven fluent responses, so the judge now scores four
# dimensions independently and the reported score is their mean (0.25 steps).
# Changing this rubric invalidates comparison with earlier runs.
EVAL_RUBRIC = """Score the coaching response on FOUR dimensions, 1-5 each. Be strict.
Every response you see is fluent and competently written, so fluency earns
nothing — your job is to separate them. Score each dimension INDEPENDENTLY; do
not let a strong dimension lift a weak one, and do not give the same number to
every dimension unless the response genuinely deserves it.

- specificity: is this tailored to THIS role, seniority and difficulty, or would
  the same text serve any candidate applying anywhere?
- actionability: are the steps concrete enough to act on today (named topics,
  drills, time allocation, a worked structure), or filler like "practise coding"
  and "be confident"?
- correctness: is every technical and interview claim accurate and current?
- insight: does it say something a strong interviewer knows that an average blog
  post misses — the specific pitfall, what the interviewer is really probing for,
  the mistake most candidates make on this exact question?

Anchors, applied to each dimension separately:
5 = you can name no improvement on this dimension. If you can think of one thing
    to change, it is not a 5.
4 = strong, with one clear weakness.
3 = adequate but generic — would suit any candidate, any role, any level.
2 = vague, padded, or questionable.
1 = absent, wrong, or unusable.

In "rationale", name the single most important weakness in ONE sentence."""

EVAL_SCORE_FORMAT: dict[str, Any] = json_schema_format(
    "eval_score",
    {
        "type": "object",
        "properties": {
            **{d: {"type": "integer", "minimum": 1, "maximum": 5} for d in EVAL_DIMENSIONS},
            "rationale": {"type": "string"},
        },
        "required": [*EVAL_DIMENSIONS, "rationale"],
        "additionalProperties": False,
    },
)

DEFAULT_GENERATION_MODEL = "openai/gpt-5-mini"
DEFAULT_JUDGE_MODEL = "openai/gpt-5-mini"
DEFAULT_MODELS = ["openai/gpt-5-mini", "openai/gpt-5-nano"]


def judge_response(
    client: OpenAI, judge_model: str, case: EvalCase, response_text: str
) -> dict[str, Any]:
    """Score one generated response against the fixed rubric.

    Returns:
        {"score": int, "rationale": str}; score is None when judging failed.
    """
    # 400 was not enough: low reasoning effort still spends ~200 completion
    # tokens before the JSON starts, so the schema got cut off mid-object and
    # the case scored None. The visible output is tiny (four ints and one
    # sentence); the budget is for the thinking that precedes it.
    judge_settings = RequestSettings(
        model=judge_model, temperature=0.0, max_tokens=900, reasoning_effort="low"
    )
    system = "You are a strict, consistent evaluator of interview-coaching quality."
    prompt = f"""{EVAL_RUBRIC}

Candidate's request ({case["seniority"]} {case["role"]}, {case["difficulty"]} difficulty):
\"\"\"{case["question"]}\"\"\"

Coaching response to evaluate:
\"\"\"{response_text}\"\"\"

Return ONLY a JSON object with an integer 1-5 for each of specificity,
actionability, correctness and insight, plus "rationale": one sentence of at
most 20 words."""
    try:
        raw = ask_llm(
            client,
            judge_settings,
            system,
            prompt,
            response_format=EVAL_SCORE_FORMAT,
        )
    except Exception as exc:
        logger.error("eval_judge_failed case=%s error=%s", case["id"], exc)
        return {"score": None, "dimensions": {}, "rationale": f"judge error: {exc}"}

    parsed = parse_json_response(raw)
    if isinstance(parsed, dict) and all(isinstance(parsed.get(d), int) for d in EVAL_DIMENSIONS):
        dimensions = {d: max(1, min(5, parsed[d])) for d in EVAL_DIMENSIONS}
        return {
            # Mean of the four dimensions: the reported score moves in 0.25
            # steps, which is what makes near-identical responses separable.
            "score": round(statistics.mean(dimensions.values()), 2),
            "dimensions": dimensions,
            "rationale": str(parsed.get("rationale", "")),
        }
    logger.error(
        "eval_judge_unparsable case=%s chars=%d — raise the judge's max_tokens",
        case["id"],
        len(raw or ""),
    )
    return {"score": None, "dimensions": {}, "rationale": "judge returned unparsable output"}


def run_case(
    client: OpenAI,
    case: EvalCase,
    technique_name: str,
    generation_model: str,
    judge_model: str,
) -> dict[str, Any]:
    """Generate one response and judge it. Returns a flat result record."""
    # Fixed low temperature so generation differences come from the
    # technique/model under test, not sampling noise.
    #
    # max_tokens=900 truncated every reply (finish_reason=length) and sometimes
    # returned no visible text at all, which the judge then scored as a bad
    # answer — that measured the cap, not the technique. Low reasoning effort
    # keeps the token spend down; 1400 leaves room for a complete answer.
    gen_settings = RequestSettings(
        model=generation_model, temperature=0.2, max_tokens=1400, reasoning_effort="low"
    )
    response = generate_prep(
        client,
        gen_settings,
        technique=TECHNIQUES[technique_name],
        role=case["role"],
        seniority=case["seniority"],
        difficulty=case["difficulty"],
        user_question=case["question"],
        response_length="Concise",
        structured=False,
    )
    response_text = response if isinstance(response, str) else str(response)
    verdict = judge_response(client, judge_model, case, response_text)
    return {
        "case_id": case["id"],
        "technique": technique_name,
        "model": generation_model,
        "score": verdict["score"],
        "dimensions": verdict.get("dimensions", {}),
        "rationale": verdict["rationale"],
        "response": response_text,
    }


def summarize(results: list[dict[str, Any]], group_key: str) -> str:
    """Build a markdown comparison table, grouped by technique or model.

    Columns: the overall mean (mean of the four judged dimensions, averaged over
    cases), then one column per dimension so it is visible WHERE a technique
    wins or loses, not just that it does.
    """
    # Group EVERY result, scored or not. A candidate whose judge call failed
    # is averaged over fewer cases than the rest, and a mean over 2 cases must
    # not silently outrank a mean over 3 — so the N column reports scored/run.
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(r[group_key], []).append(r)

    def dimension_mean(records: list[dict[str, Any]], dimension: str) -> float | None:
        values = [
            r["dimensions"][dimension] for r in records if r.get("dimensions", {}).get(dimension)
        ]
        return statistics.mean(values) if values else None

    headers = [
        group_key.title(),
        "Mean",
        *(DIMENSION_LABELS[d] for d in EVAL_DIMENSIONS),
        "N",
        "Scores",
    ]
    lines = [f"| {' | '.join(headers)} |", "|" + "---|" * len(headers)]

    scored_by_group = {
        name: [r for r in records if r["score"] is not None] for name, records in groups.items()
    }
    ranked = sorted(
        (kv for kv in groups.items() if scored_by_group[kv[0]]),
        key=lambda kv: -statistics.mean([r["score"] for r in scored_by_group[kv[0]]]),
    )

    incomplete = False
    for name, records in ranked:
        scored = scored_by_group[name]
        scores = [r["score"] for r in scored]
        n_cell = str(len(scores))
        if len(scored) < len(records):
            n_cell = f"{len(scores)}/{len(records)}*"
            incomplete = True
        cells = [
            name,
            f"{statistics.mean(scores):.2f}",
            *(
                f"{m:.2f}" if (m := dimension_mean(scored, d)) is not None else "-"
                for d in EVAL_DIMENSIONS
            ),
            n_cell,
            ", ".join(f"{s:g}" for s in scores),
        ]
        lines.append(f"| {' | '.join(cells)} |")

    unscored = [name for name, records in groups.items() if not scored_by_group[name]]
    for name in unscored:
        lines.append(f"| {name} | - | " + "- | " * len(EVAL_DIMENSIONS) + "0 | (no scores) |")

    if incomplete or unscored:
        lines.append("")
        lines.append(
            "*Scored on fewer cases than the others, so its mean is not directly "
            "comparable. Re-run before drawing a conclusion from it."
        )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point. Parses args, runs the comparison, prints a table."""
    parser = argparse.ArgumentParser(
        description="Compare prompt techniques or models on the fixed eval set."
    )
    parser.add_argument(
        "--mode",
        choices=["techniques", "models"],
        default="techniques",
        help="Compare prompt techniques (one model) or models (one technique).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATION_MODEL,
        help="Generation model for --mode techniques.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to compare for --mode models.",
    )
    parser.add_argument(
        "--technique",
        default="Zero-shot",
        help="Technique used for --mode models.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Fixed judge model (keep constant across runs).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=len(EVAL_CASES),
        help="Use only the first N eval cases (cost control).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save raw per-case results as JSON.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Cases to run concurrently (default 6; use 1 for a serial run).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the per-call INFO log lines (latency, tokens, request id).",
    )
    args = parser.parse_args()

    # The engine logs one INFO line per LLM call, which buries this runner's own
    # progress output (2 lines per case), so the console stays at warnings and
    # errors — truncation and failures still show. --verbose turns the per-call
    # lines back on regardless of what APP_LOG_LEVEL says.
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not found — check your .env file.")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    cases = EVAL_CASES[: max(1, args.limit)]
    if args.mode == "techniques":
        candidates = [(t, args.model) for t in TECHNIQUES]
        group_key = "technique"
    else:
        candidates = [(args.technique, m) for m in args.models]
        group_key = "model"

    tasks = [
        (technique_name, model, case) for technique_name, model in candidates for case in cases
    ]
    total = len(tasks)
    workers = max(1, min(args.workers, total))
    print(
        f"Running {total} eval cases ({len(candidates)} candidates × "
        f"{len(cases)} cases, 2 API calls each) on {workers} worker(s)...\n"
    )

    # Results are stored by task index, so the saved JSON and the table stay in
    # a fixed order no matter which worker finishes first. The OpenAI client is
    # safe to share across threads.
    ordered: list[dict[str, Any] | None] = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_case, client, case, technique_name, model, args.judge_model): index
            for index, (technique_name, model, case) in enumerate(tasks)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            technique_name, model, case = tasks[index]
            try:
                record = future.result()
            except Exception as exc:
                # One bad case must not throw away the other N-1 results.
                logger.error("eval_case_failed case=%s error=%s", case["id"], exc)
                record = {
                    "case_id": case["id"],
                    "technique": technique_name,
                    "model": model,
                    "score": None,
                    "dimensions": {},
                    "rationale": f"case error: {exc}",
                    "response": "",
                }
            ordered[index] = record
            print(
                f"  [{done}/{total}] [{group_key}={record[group_key]}] "
                f"case={record['case_id']} score={record['score']}"
            )

    results: list[dict[str, Any]] = [r for r in ordered if r is not None]

    print(f"\n## Results (judge: fixed rubric, {args.judge_model}, temperature 0)\n")
    print(summarize(results, group_key))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nRaw results saved to {args.out}")


if __name__ == "__main__":
    main()
