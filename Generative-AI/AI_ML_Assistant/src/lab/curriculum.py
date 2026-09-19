"""Shared curriculum for the 🎓 AI/ML Tutor and 🔬 AI/ML Lab — the single source of truth.

The Tutor renders these tracks as guided learning paths (each lesson opens in 💬 Chat at the
chosen level); the Lab reads the *same* lessons to offer a hands-on exercise for the ones that
carry a runnable :class:`Practice` mapping. Defining a lesson once means the "learn it → then
practise it" round-trip between the two pages can never drift out of sync.

Framework-agnostic: pure data, no Streamlit and no heavy dependencies, so it imports and
tests offline. A :class:`Practice` references a recipe key from :mod:`src.lab.recipes` and a
dataset key from :mod:`src.lab.datasets`; :func:`validate_curriculum` checks they all resolve
(exercised by the test-suite so a typo can't ship a dead "Practice" button).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Practice:
    """A hands-on 🔬 AI/ML Lab target for a lesson.

    Two flavours, distinguished by ``topic``:

    * **Dataset recipes** (``topic`` in ``ML`` / ``NLP`` / ``DL``) preselect a
      :class:`~src.lab.recipes.Recipe` (``recipe_key``) over a bundled
      :class:`~src.lab.datasets.LabDataset` (``dataset_key``).
    * **Live-model exercises** (``topic == "AI & LLM"``) preselect an
      :class:`~src.lab.llm_exercises.LLMExercise` (``exercise_key``) — no recipe or dataset,
      so ``recipe_key``/``dataset_key`` stay empty.

    :func:`validate_curriculum` enforces that exactly the right keys resolve for each flavour.
    """

    topic: str  # one of src.lab.datasets.TOPIC_* ("ML" | "NLP" | "DL" | "AI & LLM")
    recipe_key: str = ""  # a src.lab.recipes Recipe.key (dataset recipes only)
    dataset_key: str = ""  # a key into src.lab.datasets.DATASETS (dataset recipes only)
    hint: str = ""  # one line shown in the Lab: what to try or notice
    exercise_key: str = ""  # a key into src.lab.llm_exercises.LLM_EXERCISES (AI & LLM only)


@dataclass(frozen=True)
class Lesson:
    """One curriculum lesson: its title, the seed question sent to Chat, and optional practice.

    ``practice`` is None for conceptual lessons that have no meaningful in-browser exercise
    (e.g. transformers, prompt engineering) — those stay learn-only, answered in 💬 Chat.
    """

    title: str
    question: str
    practice: Practice | None = None


@dataclass(frozen=True)
class Track:
    """A named, ordered sequence of lessons — one learning path in the Tutor."""

    name: str
    lessons: tuple[Lesson, ...]


# The curriculum. Lessons with a Practice map onto an existing Lab recipe + bundled dataset;
# the rest are conceptual and open in Chat only. Keep recipe/dataset keys in step with
# src.lab.recipes.RECIPES and src.lab.datasets.DATASETS — validate_curriculum() enforces it.
TRACKS: tuple[Track, ...] = (
    Track(
        "Foundations of ML",
        (
            Lesson(
                "What is machine learning?",
                "What is machine learning, and how is it different from traditional programming?",
                Practice(
                    "ML",
                    "explore",
                    "iris",
                    "Read `describe()` and the scatter — this is the raw data a model learns from.",
                ),
            ),
            Lesson(
                "Supervised vs unsupervised",
                "What is the difference between supervised and unsupervised learning?",
                Practice(
                    "ML",
                    "cluster",
                    "iris",
                    "K-Means groups the flowers with **no labels** — compare its clusters to the "
                    "true species.",
                ),
            ),
            Lesson(
                "Overfitting & regularization",
                "What is overfitting, and how do regularization techniques prevent it?",
                Practice(
                    "ML",
                    "classify",
                    "wine",
                    "Sweep the **C** slider: smaller C = stronger regularisation. Watch accuracy "
                    "respond.",
                ),
            ),
            Lesson(
                "Train/validation/test splits",
                "Why do we split data into train, validation, and test sets?",
                Practice(
                    "ML",
                    "classify",
                    "iris",
                    "Move the **Test size** slider to see how the hold-out fraction shifts the "
                    "reported score.",
                ),
            ),
            Lesson(
                "Evaluating a classifier",
                "How do we measure whether a classifier is any good — accuracy, precision, "
                "and recall?",
                Practice(
                    "ML",
                    "classify",
                    "breast_cancer",
                    "A binary task (benign vs. malignant). Change the **test size** and **C** and "
                    "watch the reported accuracy respond.",
                ),
            ),
        ),
    ),
    Track(
        "Deep Learning",
        (
            Lesson(
                "Neurons & activations",
                "What is an artificial neuron, and why do we need activation functions?",
                Practice(
                    "DL",
                    "small_net",
                    "digits",
                    "Change **units per layer** — more neurons add capacity (and cost). Note the "
                    "accuracy.",
                ),
            ),
            Lesson(
                "Backpropagation",
                "How does backpropagation train a neural network?",
                Practice(
                    "DL",
                    "small_net",
                    "digits",
                    "The loss curve **is** backprop at work: each step nudges the weights to lower "
                    "loss.",
                ),
            ),
            Lesson("CNNs", "How do convolutional neural networks process images?"),
            Lesson(
                "Transformers & attention",
                "How do transformers use attention to model sequences?",
            ),
        ),
    ),
    Track(
        "LLMs & RAG",
        (
            Lesson(
                "Embeddings",
                "What is an embedding, and how does cosine similarity compare two of them?",
                Practice(
                    "NLP",
                    "cosine",
                    "mini_text",
                    "The heatmap is cosine similarity between TF-IDF vectors — the same signal RAG "
                    "uses to retrieve.",
                ),
            ),
            Lesson(
                "Text classification with TF-IDF",
                "How does TF-IDF turn text into numeric features a classifier can learn from?",
                Practice(
                    "NLP",
                    "tfidf_classify",
                    "mini_text",
                    "Train on the weather-vs-finance corpus and read the accuracy — the classic "
                    "baseline that embeddings later improve on.",
                ),
            ),
            Lesson("RAG vs fine-tuning", "RAG vs fine-tuning — when should I use which?"),
            Lesson(
                "Prompt engineering",
                "What are the most useful prompt-engineering techniques and when do they help?",
                Practice(
                    "AI & LLM",
                    exercise_key="prompt_compare",
                    hint="Ask your question, then compare techniques — watch how chain-of-thought "
                    "or analogy-first shifts the judged relevancy.",
                ),
            ),
            Lesson(
                "Evaluating LLMs",
                "How do we evaluate the quality of an LLM's answers?",
                Practice(
                    "AI & LLM",
                    exercise_key="llm_eval",
                    hint="Get a live answer, then read its faithfulness and relevancy scores — "
                    "paste a reference context to also measure grounding.",
                ),
            ),
        ),
    ),
)


def track_names() -> list[str]:
    """Return the track names in curriculum order (for the Tutor's track selector)."""
    return [t.name for t in TRACKS]


def get_track(name: str) -> Track | None:
    """Return the :class:`Track` with this name, or None if there is no such track."""
    return next((t for t in TRACKS if t.name == name), None)


def validate_curriculum() -> list[str]:
    """Return a list of problems: Practice targets whose recipe/dataset/topic don't resolve.

    Empty list == the whole curriculum is wired correctly. Imported lazily so this module
    stays free of the optional Lab dependencies at import time.
    """
    from src.lab.datasets import DATASETS, TOPIC_LLM
    from src.lab.llm_exercises import LLM_EXERCISES
    from src.lab.recipes import RECIPES

    recipe_topic = {r.key: r.topic for r in RECIPES}
    problems: list[str] = []
    for track in TRACKS:
        for lesson in track.lessons:
            p = lesson.practice
            if p is None:
                continue
            where = f"{track.name} → {lesson.title}"
            if p.topic == TOPIC_LLM:
                # Live-model exercise: an exercise key must resolve, and no dataset/recipe.
                if p.exercise_key not in LLM_EXERCISES:
                    problems.append(f"{where}: unknown LLM exercise '{p.exercise_key}'")
                if p.recipe_key or p.dataset_key:
                    problems.append(
                        f"{where}: '{TOPIC_LLM}' practice must not set a recipe_key/dataset_key"
                    )
                continue
            # Dataset recipe: dataset + recipe must resolve and agree on the topic.
            if p.exercise_key:
                problems.append(f"{where}: exercise_key is only valid for '{TOPIC_LLM}' topics")
            if p.dataset_key not in DATASETS:
                problems.append(f"{where}: unknown dataset '{p.dataset_key}'")
            elif p.topic not in DATASETS[p.dataset_key].topics:
                problems.append(
                    f"{where}: dataset '{p.dataset_key}' doesn't serve topic '{p.topic}'"
                )
            if p.recipe_key not in recipe_topic:
                problems.append(f"{where}: unknown recipe '{p.recipe_key}'")
            elif recipe_topic[p.recipe_key] != p.topic:
                problems.append(f"{where}: recipe '{p.recipe_key}' is not a '{p.topic}' recipe")
    return problems
