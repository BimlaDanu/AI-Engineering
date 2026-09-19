"""Offline tests for the 🔬 ML Lab: the restricted sandbox, datasets, and recipes.

The sandbox tests need no optional dependency — they exercise the security guardrails with
plain Python. The dataset/recipe tests ``importorskip`` pandas / scikit-learn / matplotlib,
so the suite stays green before ``make sync`` installs them and exercises them once present.
"""

from __future__ import annotations

import pytest

from src.lab.sandbox import ALLOWED_MODULES, run_sandboxed


# --- sandbox (pure) -------------------------------------------------------------------
def test_sandbox_runs_allowed_code_and_captures_stdout():
    result = run_sandboxed("x = 2 + 3\nprint('value', x)")
    assert result.ok
    assert "value 5" in result.stdout
    assert result.error is None


def test_sandbox_allows_whitelisted_import():
    result = run_sandboxed("import math\nprint(round(math.sqrt(16)))")
    assert result.ok
    assert "4" in result.stdout


def test_sandbox_blocks_os_import():
    result = run_sandboxed("import os\nprint(os.getcwd())")
    assert not result.ok
    assert result.error is not None
    assert "blocked" in result.error.lower()


@pytest.mark.parametrize("builtin", ["open", "exec", "eval", "compile", "input", "__import__"])
def test_sandbox_removes_dangerous_builtins(builtin):
    # __import__ is replaced by the guard; the others are absent entirely. Calling any of
    # them by name fails (NameError for removed, ImportError for a blocked guarded import).
    result = run_sandboxed(f"{builtin}('nope')")
    assert not result.ok
    assert result.error is not None


def test_sandbox_error_is_captured_not_raised():
    result = run_sandboxed("1 / 0")
    assert not result.ok
    assert "ZeroDivisionError" in result.error


def test_sandbox_empty_code_is_ok():
    assert run_sandboxed("   ").ok


def test_sandbox_namespace_is_available():
    result = run_sandboxed("print(payload * 2)", {"payload": 21})
    assert result.ok
    assert "42" in result.stdout


def test_allowed_modules_excludes_dangerous_ones():
    for banned in ("os", "sys", "subprocess", "socket", "pathlib", "shutil"):
        assert banned not in ALLOWED_MODULES


def test_sandbox_captures_warning_raised_during_a_run():
    # ``warnings`` isn't importable in the sandbox, so warn via a preloaded helper — the same
    # path a real sklearn ConvergenceWarning takes when learner code fits a model.
    import warnings

    def emit():
        warnings.warn("model did not converge", stacklevel=1)

    result = run_sandboxed("emit()\nprint('done')", {"emit": emit})
    assert result.ok
    assert "done" in result.stdout
    assert any("did not converge" in w for w in result.warnings)


def test_format_warnings_deduplicates_and_drops_noise():
    import warnings

    from src.lab.sandbox import _format_warnings

    def msg(text: str, category=UserWarning):
        return warnings.WarningMessage(text, category, "<ml-lab>", 1)

    caught = [
        msg("did not converge"),
        msg("did not converge"),  # duplicate → collapsed
        msg("FigureCanvasAgg is non-interactive, and thus cannot be shown"),  # noise → dropped
    ]
    assert _format_warnings(caught) == ["UserWarning: did not converge"]


# --------------------------------------------------------------- seed-code grounding
# The advanced-cell seeder must describe the *actual* pre-bound variables per task, or the
# model invents columns that don't exist. These pin the environment description offline (no
# LLM, no pandas — LoadedData carries the task/target metadata the grounding reads).
from src.lab.datasets import LoadedData  # noqa: E402
from src.lab.seed import _environment  # noqa: E402


def _loaded(task: str, target_names):
    return LoadedData(df=None, target=None, task=task, description="d", target_names=target_names)


def test_environment_for_text_steers_away_from_a_label_column():
    # Regression for the reported NLP "KeyError: 'label'": the text task has no feature/label
    # column in df, so the grounding must say so and point at `texts` + the ready-made `y`.
    env = _environment(_loaded("text", ["weather", "finance"]), feature_names=["text"])
    assert "texts" in env
    assert "NO label column" in env
    assert "do not derive it from `df`" in env
    assert "vectorise" in env  # raw text must be vectorised before modelling


def test_environment_for_classification_lists_classes_and_no_target_column():
    env = _environment(_loaded("classification", ["setosa", "versicolor"]), ["petal_len"])
    assert "petal_len" in env
    assert "class labels" in env
    assert "setosa, versicolor" in env
    assert "NO target column" in env


def test_environment_maps_integer_coded_labels_to_names_not_bare_strings():
    # Regression for the reported "pos_label=malignant is not a valid label. It should be one
    # of [0 1]": the sklearn sets store y as integer codes, so the grounding must spell out the
    # code->name mapping and steer the model to use the code (e.g. pos_label=1), never the name.
    data = LoadedData(
        df=None,
        target=[0, 0, 1, 1],
        task="classification",
        description="d",
        target_names=["malignant", "benign"],
    )
    env = _environment(data, ["mean_radius"])
    assert "0 = malignant" in env and "1 = benign" in env
    assert "pos_label=1" in env  # use the integer code, not the name string


def test_environment_for_regression_says_continuous_targets():
    env = _environment(_loaded("regression", []), ["age", "bmi"])
    assert "continuous regression targets" in env
    assert "NO target column" in env


# ------------------------------------------------------------- challenges (pure, no deps)
# The challenge graders are pure functions of a LabResult + chosen params, so they test with
# hand-built results and never touch sklearn — these run before the importorskip below.
from src.lab.challenges import (  # noqa: E402
    BEGINNER,
    CHALLENGES,
    PRACTITIONER,
    RESEARCHER,
    challenges_for_recipe,
    pick_for_level,
    validate_challenges,
)
from src.lab.recipes import LabResult  # noqa: E402


def test_challenges_all_wire_to_a_real_recipe():
    # Mirrors validate_curriculum: no challenge may reference a missing recipe, a mismatched
    # topic, an unknown level, or a duplicate key. Empty problem list == the registry is sound.
    assert validate_challenges() == []


def test_challenge_ladder_is_ordered_by_level():
    ladder = challenges_for_recipe("classify")
    assert [c.level for c in ladder] == [BEGINNER, PRACTITIONER, RESEARCHER]


def test_pick_for_level_prefers_exact_then_falls_back_down():
    # Practitioner gets the Practitioner tier exactly.
    assert pick_for_level("classify", PRACTITIONER).level == PRACTITIONER
    # A recipe with only Beginner+Practitioner: a Researcher gets the hardest available.
    assert pick_for_level("cosine", RESEARCHER).level == PRACTITIONER
    # No challenges for an unknown recipe.
    assert pick_for_level("nope", BEGINNER) is None


def test_run_check_passes_on_a_clean_result_and_fails_on_error():
    run = next(c for c in CHALLENGES if c.key == "classify_run").check
    assert run(LabResult(summary="", metrics={"accuracy": 0.5}), {}).passed
    assert not run(LabResult(summary="", error="boom"), {}).passed
    # No metric produced → cannot confirm a run.
    assert not run(LabResult(summary="", metrics={}), {}).passed


def test_threshold_check_grades_against_the_bar():
    tune = next(c for c in CHALLENGES if c.key == "classify_tune").check
    passed = tune(LabResult(summary="", metrics={"accuracy": 0.96}), {})
    failed = tune(LabResult(summary="", metrics={"accuracy": 0.90}), {})
    assert passed.passed and passed.metric == 0.96
    assert not failed.passed and "below the target" in failed.message


def test_param_constraint_check_requires_the_named_method():
    knn = next(c for c in CHALLENGES if c.key == "classify_knn").check
    hit = LabResult(summary="", metrics={"accuracy": 0.97})
    # Right accuracy but wrong model → not passed (the challenge is about the method).
    assert not knn(hit, {"model": "LogisticRegression"}).passed
    # Right model and accuracy → passed.
    assert knn(hit, {"model": "KNeighborsClassifier"}).passed
    # Right model but under the bar → not passed.
    assert not knn(
        LabResult(summary="", metrics={"accuracy": 0.5}), {"model": "KNeighborsClassifier"}
    ).passed


def test_efficiency_check_caps_a_numeric_param():
    eff = next(c for c in CHALLENGES if c.key == "net_efficient").check
    good = LabResult(summary="", metrics={"accuracy": 0.96})
    assert eff(good, {"max_iter": 150}).passed
    assert not eff(good, {"max_iter": 400}).passed  # over the iteration budget


# ------------------------------------------------------------------- datasets & recipes
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from src.lab.datasets import DATASETS, datasets_for_topic  # noqa: E402
from src.lab.recipes import recipes_for_topic  # noqa: E402


def test_datasets_for_topic_are_scoped():
    assert {d.key for d in datasets_for_topic("ML")}
    assert {d.key for d in datasets_for_topic("NLP")} == {"mini_text"}


def test_iris_loads_as_classification():
    data = DATASETS["iris"].loader()
    assert data.task == "classification"
    assert data.df.shape == (150, 4)
    assert data.target is not None and len(data.target) == 150


def test_classify_recipe_produces_a_figure():
    data = DATASETS["wine"].loader()
    recipe = next(r for r in recipes_for_topic("ML") if r.key == "classify")
    result = recipe.run(data, {"model": "LogisticRegression", "test_size": 0.25, "C": 1.0})
    assert result.error is None
    assert result.figure is not None
    assert "%" in result.summary


def test_cosine_recipe_runs_on_text():
    data = DATASETS["mini_text"].loader()
    recipe = next(r for r in recipes_for_topic("NLP") if r.key == "cosine")
    result = recipe.run(data, {})
    assert result.error is None
    assert result.figure is not None


def test_tfidf_classify_recipe_runs_on_text():
    # Regression: the text corpus' string target is an Arrow-backed extension array under
    # pandas 3; scikit-learn's train/test split fancy-indexing choked on it until the recipe
    # coerced the target with .to_numpy(). Guard that the recipe runs end-to-end.
    data = DATASETS["mini_text"].loader()
    recipe = next(r for r in recipes_for_topic("NLP") if r.key == "tfidf_classify")
    result = recipe.run(data, {})
    assert result.error is None
    assert result.table is not None
    assert "%" in result.summary


def test_recipes_populate_machine_readable_metrics():
    # Challenges grade LabResult.metrics, so each measurable recipe must expose its headline
    # number there (not just in the prose summary). Exploration legitimately has none.
    from src.lab.recipes import RECIPES

    by_key = {r.key: r for r in RECIPES}
    expected = {
        "classify": ("wine", {}, "accuracy"),
        "cluster": ("iris", {}, "silhouette"),
        "tfidf_classify": ("mini_text", {}, "accuracy"),
        "cosine": ("mini_text", {}, "best_cosine"),
    }
    for recipe_key, (ds, params, metric) in expected.items():
        result = by_key[recipe_key].run(DATASETS[ds].loader(), params)
        assert result.error is None
        assert metric in result.metrics, f"{recipe_key} missing metric '{metric}'"


def test_end_to_end_challenge_grades_a_real_run():
    # The full loop: run the recipe, then grade its LabResult with the matching challenge.
    from src.lab.challenges import CHALLENGES
    from src.lab.recipes import RECIPES

    classify = next(r for r in RECIPES if r.key == "classify")
    result = classify.run(
        DATASETS["wine"].loader(), {"model": "LogisticRegression", "test_size": 0.25, "C": 1.0}
    )
    run_challenge = next(c for c in CHALLENGES if c.key == "classify_run")
    verdict = run_challenge.check(result, {})
    assert verdict.passed
    assert verdict.metric == result.metrics["accuracy"]


def test_sandbox_runs_pandas_over_dataset_namespace():
    data = DATASETS["iris"].loader()
    result = run_sandboxed("print(int(df.shape[0]))", {"df": data.df})
    assert result.ok
    assert "150" in result.stdout


# ----------------------------------------------------------------------------- curriculum
from src.lab.curriculum import TRACKS, get_track, track_names, validate_curriculum  # noqa: E402


def test_curriculum_practice_targets_all_resolve():
    # Every lesson's Practice must point at a real recipe + dataset for the right topic;
    # an empty problem list means no "Practise" button can lead to a dead exercise.
    assert validate_curriculum() == []


def test_track_lookup_matches_names():
    assert get_track(track_names()[0]) is not None
    assert get_track("no such track") is None


def test_at_least_one_lesson_is_learn_only():
    # Conceptual lessons (e.g. transformers) intentionally have no runnable exercise.
    lessons = [lesson for track in TRACKS for lesson in track.lessons]
    assert any(lesson.practice is None for lesson in lessons)
    assert any(lesson.practice is not None for lesson in lessons)


def test_runnable_coverage_is_eleven_of_fourteen():
    # 11 of the 14 lessons carry a runnable Practice (9 dataset recipes + 2 live AI & LLM
    # exercises); 3 stay conceptual/learn-only (CNNs, transformers, RAG vs fine-tuning).
    # Pinning it flags any accidental regression in the curriculum wiring.
    lessons = [lesson for track in TRACKS for lesson in track.lessons]
    runnable = [lesson for lesson in lessons if lesson.practice is not None]
    assert len(lessons) == 14
    assert len(runnable) == 11
    assert len(lessons) - len(runnable) == 3


def test_practice_recipe_runs_on_its_mapped_dataset():
    # The mapping is only useful if the recipe actually runs on the dataset it names. Live
    # AI & LLM exercises carry no dataset/recipe (they need a model), so skip them here —
    # they're covered offline by tests/test_llm_exercises.py.
    from src.lab.recipes import RECIPES

    by_key = {r.key: r for r in RECIPES}
    for track in TRACKS:
        for lesson in track.lessons:
            p = lesson.practice
            if p is None or not p.recipe_key:
                continue
            data = DATASETS[p.dataset_key].loader()
            result = by_key[p.recipe_key].run(data, {})
            assert result.error is None, f"{track.name} → {lesson.title}: {result.error}"
