"""Level-scaled, auto-checked challenges for the 🔬 AI/ML Lab.

A :class:`Challenge` turns a guided :mod:`src.lab.recipes` recipe from a *demo* into *practice*:
it states a concrete, level-appropriate **goal** ("tune C until test accuracy ≥ 0.95") and
carries a pure ``check`` that grades the actual :class:`~src.lab.recipes.LabResult` against that
goal — pass or fail, with a one-line reason. Grading reads the run's machine-readable
``LabResult.metrics`` (never the prose summary) plus the parameters the learner chose, so a
challenge like "do it with KNN" can verify *how* the result was reached, not just the number.

Three tiers, mapped to :data:`src.config.LEVELS` (``Beginner`` → run-and-read, ``Practitioner``
→ hit a threshold, ``Researcher`` → a stiffer bar or a method constraint). Not every recipe has
all three — :func:`challenges_for_recipe` returns whatever tiers exist, and
:func:`pick_for_level` chooses the best match for the learner's current level.

Framework-agnostic: pure data + pure functions, no Streamlit and no heavy dependencies, so it
imports and unit-tests fully offline. :func:`validate_challenges` mirrors
:func:`src.lab.curriculum.validate_curriculum` — it checks every challenge references a real
recipe whose topic agrees, at a known level — and is exercised by the test-suite so a typo can
never ship a broken challenge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.config import LEVELS
from src.lab.recipes import LabResult

# The three tiers, by name, in increasing difficulty (kept in step with src.config.LEVELS).
BEGINNER, PRACTITIONER, RESEARCHER = LEVELS

# A challenge's grader: given the run's result and the parameters the learner chose, decide
# whether the goal was met. Pure and deterministic — no I/O, no model — so it tests offline.
CheckFn = Callable[[LabResult, dict], "CheckResult"]


@dataclass(frozen=True)
class CheckResult:
    """The verdict of grading one run against a challenge's goal."""

    passed: bool
    message: str
    metric: float | None = None  # the headline number the check looked at, if any


@dataclass(frozen=True)
class Challenge:
    """One level-scaled, auto-checked practice task bound to a Lab recipe."""

    key: str  # stable, unique — also the progress-tracking id
    recipe_key: str  # a src.lab.recipes Recipe.key this challenge grades
    topic: str  # must equal the recipe's topic (validate_challenges enforces it)
    level: str  # one of src.config.LEVELS
    goal: str  # imperative, one line: what the learner must achieve
    check: CheckFn  # pure grader of (LabResult, chosen params) -> CheckResult


# --------------------------------------------------------------------------- check factories
def _ran(metric_key: str, *, label: str, fmt: str = "{:.3f}") -> CheckFn:
    """A Beginner check: the recipe ran without error and produced ``metric_key``."""

    def check(result: LabResult, params: dict) -> CheckResult:
        if result.error:
            return CheckResult(False, result.error)
        value = result.metrics.get(metric_key)
        if value is None:
            return CheckResult(False, "The run produced no result to read.")
        return CheckResult(
            True, f"Done — you ran it and read the {label} ({fmt.format(value)}).", value
        )

    return check


def _at_least(metric_key: str, threshold: float, *, label: str, fmt: str = "{:.3f}") -> CheckFn:
    """A threshold check: ``metric_key`` must reach ``threshold`` (higher is better)."""

    def check(result: LabResult, params: dict) -> CheckResult:
        if result.error:
            return CheckResult(False, result.error)
        value = result.metrics.get(metric_key)
        if value is None:
            return CheckResult(False, "The run produced no result to check.")
        if value >= threshold:
            return CheckResult(
                True,
                f"Goal met — {label} {fmt.format(value)} ≥ {fmt.format(threshold)}.",
                value,
            )
        return CheckResult(
            False,
            f"Not yet — {label} {fmt.format(value)} is below the target "
            f"{fmt.format(threshold)}. Adjust the settings and run again.",
            value,
        )

    return check


def _at_least_with_param(
    metric_key: str,
    threshold: float,
    *,
    param: str,
    equals: object = None,
    at_most: float | None = None,
    label: str,
    constraint_label: str,
    fmt: str = "{:.3f}",
) -> CheckFn:
    """A Researcher check: reach ``threshold`` *and* satisfy a constraint on a chosen param.

    ``equals`` pins a categorical parameter (e.g. use the KNN model); ``at_most`` caps a
    numeric one (e.g. ≤ 200 iterations). Exactly one of them is given. This lets a challenge
    verify *how* the bar was cleared — the method, not just the number.
    """

    def check(result: LabResult, params: dict) -> CheckResult:
        if result.error:
            return CheckResult(False, result.error)
        chosen = params.get(param)
        if equals is not None and chosen != equals:
            return CheckResult(False, f"Not yet — this challenge asks you to {constraint_label}.")
        if at_most is not None and (chosen is None or float(chosen) > at_most):
            return CheckResult(False, f"Not yet — this challenge asks you to {constraint_label}.")
        value = result.metrics.get(metric_key)
        if value is None:
            return CheckResult(False, "The run produced no result to check.")
        if value >= threshold:
            return CheckResult(
                True,
                f"Goal met — {label} {fmt.format(value)} ≥ {fmt.format(threshold)} "
                f"while you {constraint_label}.",
                value,
            )
        return CheckResult(
            False,
            f"Close — you {constraint_label}, but {label} {fmt.format(value)} is below "
            f"{fmt.format(threshold)}. Keep tuning.",
            value,
        )

    return check


# ------------------------------------------------------------------------------- the registry
# Thresholds are deliberately dataset-agnostic within a topic: the sklearn toy classification
# sets (iris/wine/breast-cancer/digits) all clear ~0.95 with a tuned linear model, so a single
# bar makes the challenge portable across whichever dataset the learner picks.
CHALLENGES: tuple[Challenge, ...] = (
    # --- ML · explore ------------------------------------------------------------------------
    Challenge(
        "explore_read",
        "explore",
        "ML",
        BEGINNER,
        "Pick two features, run the plot, and read the summary statistics — get a feel for the "
        "raw data a model learns from.",
        # Explore has no single metric; a clean run (no error) is the goal.
        lambda result, params: (
            CheckResult(False, result.error)
            if result.error
            else CheckResult(True, "Explored — you can see the data's shape and spread.")
        ),
    ),
    # --- ML · classify -----------------------------------------------------------------------
    Challenge(
        "classify_run",
        "classify",
        "ML",
        BEGINNER,
        "Train the classifier and read its hold-out accuracy off the confusion matrix.",
        _ran("accuracy", label="accuracy", fmt="{:.1%}"),
    ),
    Challenge(
        "classify_tune",
        "classify",
        "ML",
        PRACTITIONER,
        "Tune the regularisation C (and the test size) until test accuracy reaches 0.95.",
        _at_least("accuracy", 0.95, label="accuracy", fmt="{:.1%}"),
    ),
    Challenge(
        "classify_knn",
        "classify",
        "ML",
        RESEARCHER,
        "Switch the model to KNeighborsClassifier and still reach 0.95 accuracy — a different "
        "model family, same bar.",
        _at_least_with_param(
            "accuracy",
            0.95,
            param="model",
            equals="KNeighborsClassifier",
            label="accuracy",
            constraint_label="use the KNN model",
            fmt="{:.1%}",
        ),
    ),
    # --- ML · cluster ------------------------------------------------------------------------
    Challenge(
        "cluster_run",
        "cluster",
        "ML",
        BEGINNER,
        "Run K-Means and read its silhouette score (how cleanly the clusters separate).",
        _ran("silhouette", label="silhouette score"),
    ),
    Challenge(
        "cluster_tune",
        "cluster",
        "ML",
        PRACTITIONER,
        "Choose the number of clusters k that lifts the silhouette score to at least 0.45.",
        _at_least("silhouette", 0.45, label="silhouette"),
    ),
    Challenge(
        "cluster_best",
        "cluster",
        "ML",
        RESEARCHER,
        "Find the k that separates the data best — push the silhouette score to 0.50 or higher.",
        _at_least("silhouette", 0.50, label="silhouette"),
    ),
    # --- DL · small_net ----------------------------------------------------------------------
    Challenge(
        "net_run",
        "small_net",
        "DL",
        BEGINNER,
        "Train the small neural net and read its test accuracy once the loss curve settles.",
        _ran("accuracy", label="test accuracy", fmt="{:.1%}"),
    ),
    Challenge(
        "net_tune",
        "small_net",
        "DL",
        PRACTITIONER,
        "Widen the layers and/or raise the iterations until test accuracy reaches 0.92.",
        _at_least("accuracy", 0.92, label="test accuracy", fmt="{:.1%}"),
    ),
    Challenge(
        "net_efficient",
        "small_net",
        "DL",
        RESEARCHER,
        "Reach 0.95 test accuracy in at most 200 iterations — accuracy under a compute budget.",
        _at_least_with_param(
            "accuracy",
            0.95,
            param="max_iter",
            at_most=200.0,
            label="test accuracy",
            constraint_label="stay within 200 iterations",
            fmt="{:.1%}",
        ),
    ),
    # --- NLP · tfidf_classify ----------------------------------------------------------------
    Challenge(
        "tfidf_run",
        "tfidf_classify",
        "NLP",
        BEGINNER,
        "Vectorise the corpus with TF-IDF, train the classifier, and read its accuracy.",
        _ran("accuracy", label="accuracy", fmt="{:.0%}"),
    ),
    Challenge(
        "tfidf_score",
        "tfidf_classify",
        "NLP",
        PRACTITIONER,
        "Confirm the TF-IDF baseline separates the two topics at 0.75 accuracy or better.",
        _at_least("accuracy", 0.75, label="accuracy", fmt="{:.0%}"),
    ),
    # --- NLP · cosine ------------------------------------------------------------------------
    Challenge(
        "cosine_run",
        "cosine",
        "NLP",
        BEGINNER,
        "Build the TF-IDF cosine-similarity matrix and find the most similar pair of documents.",
        _ran("best_cosine", label="top similarity", fmt="{:.2f}"),
    ),
    Challenge(
        "cosine_pair",
        "cosine",
        "NLP",
        PRACTITIONER,
        "Confirm the closest pair scores above 0.30 cosine — the signal RAG uses to retrieve.",
        _at_least("best_cosine", 0.30, label="top similarity", fmt="{:.2f}"),
    ),
)


def challenges_for_recipe(recipe_key: str) -> list[Challenge]:
    """Return the challenges for a recipe, ordered easiest → hardest (by :data:`LEVELS`)."""
    order = {level: i for i, level in enumerate(LEVELS)}
    found = [c for c in CHALLENGES if c.recipe_key == recipe_key]
    return sorted(found, key=lambda c: order.get(c.level, len(LEVELS)))


def pick_for_level(recipe_key: str, level: str) -> Challenge | None:
    """Pick the best challenge for a recipe at ``level``.

    Prefers an exact level match; otherwise returns the hardest tier at or below ``level`` (so a
    Researcher still gets the toughest available), falling back to the easiest if none is lower.
    Returns None when the recipe has no challenges at all.
    """
    ladder = challenges_for_recipe(recipe_key)
    if not ladder:
        return None
    exact = next((c for c in ladder if c.level == level), None)
    if exact is not None:
        return exact
    order = {name: i for i, name in enumerate(LEVELS)}
    target = order.get(level, 0)
    at_or_below = [c for c in ladder if order.get(c.level, 0) <= target]
    return at_or_below[-1] if at_or_below else ladder[0]


def validate_challenges() -> list[str]:
    """Return a list of problems: challenges whose recipe/topic/level/key don't resolve.

    Empty list == every challenge is wired correctly. Imported lazily so this module stays free
    of the optional Lab dependencies at import time.
    """
    from src.lab.recipes import RECIPES

    recipe_topic = {r.key: r.topic for r in RECIPES}
    problems: list[str] = []
    seen: set[str] = set()
    for c in CHALLENGES:
        if c.key in seen:
            problems.append(f"{c.key}: duplicate challenge key")
        seen.add(c.key)
        if c.level not in LEVELS:
            problems.append(f"{c.key}: unknown level '{c.level}'")
        if c.recipe_key not in recipe_topic:
            problems.append(f"{c.key}: unknown recipe '{c.recipe_key}'")
        elif recipe_topic[c.recipe_key] != c.topic:
            problems.append(f"{c.key}: topic '{c.topic}' disagrees with recipe '{c.recipe_key}'")
    return problems
