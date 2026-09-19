"""Bundled toy datasets for the 🔬 AI/ML Lab — all offline, no downloads.

Each :class:`LabDataset` wraps a small, classic dataset (the scikit-learn toy sets, plus a
tiny hand-written text corpus for NLP) behind a lazy loader that returns a tidy
:class:`LoadedData` (a features ``DataFrame`` and an optional target ``Series``). Everything
ships with scikit-learn or is embedded here, so a Lab session never touches the network.

Framework-agnostic: this module imports pandas / scikit-learn lazily *inside* the loaders, so
importing it never fails even when those optional Lab dependencies are absent — the page
checks availability and degrades gracefully.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Lab topics. ML/NLP/DL run guided recipes over the bundled datasets below. TOPIC_LLM is
# different in kind: it drives *live-model* exercises (prompt comparison, answer evaluation)
# that take no dataset — see src.lab.llm_exercises and the page's AI & LLM panel. "Quantum" is
# intentionally absent — it points the learner back to 💬 Chat rather than executing code.
TOPIC_ML = "ML"
TOPIC_NLP = "NLP"
TOPIC_DL = "DL"
TOPIC_LLM = "AI & LLM"


@dataclass(frozen=True)
class LoadedData:
    """One materialised dataset: a features frame, an optional target, and its task type."""

    df: Any  # pandas.DataFrame of features (or a single text column for NLP)
    target: Any | None  # pandas.Series aligned to df.index, or None
    task: str  # "classification" | "regression" | "text"
    description: str
    target_names: list[str]


@dataclass(frozen=True)
class LabDataset:
    """A selectable dataset: display metadata plus a lazy, offline loader."""

    key: str
    label: str
    topics: tuple[str, ...]
    task: str
    blurb: str
    loader: Callable[[], LoadedData]


# --- A tiny, embedded text corpus so NLP recipes run with zero downloads -------------------
# Two clearly separable classes (weather vs. finance) — enough to demonstrate TF-IDF,
# classification, and cosine similarity without fetching 20-newsgroups over the network.
_MINI_CORPUS: list[tuple[str, str]] = [
    ("The storm brought heavy rain and strong winds across the coast.", "weather"),
    ("Forecasters expect a sunny weekend with mild temperatures.", "weather"),
    ("A cold front will drop temperatures and bring morning frost.", "weather"),
    ("Humidity rose sharply before the afternoon thunderstorm.", "weather"),
    ("Snowfall closed several mountain roads overnight.", "weather"),
    ("Clear skies and a gentle breeze are forecast for tomorrow.", "weather"),
    ("The central bank raised interest rates to curb inflation.", "finance"),
    ("Shares of the tech firm rose after strong quarterly earnings.", "finance"),
    ("Investors sold bonds as yields climbed to a yearly high.", "finance"),
    ("The startup raised a funding round led by a venture capital firm.", "finance"),
    ("Oil prices fell on concerns about weakening global demand.", "finance"),
    ("The company reported record revenue but shrinking profit margins.", "finance"),
]


def _load_sklearn_bunch(name: str) -> LoadedData:
    """Load a scikit-learn toy dataset by name into a tidy :class:`LoadedData`."""
    import pandas as pd
    from sklearn import datasets as skds

    loaders = {
        "iris": skds.load_iris,
        "wine": skds.load_wine,
        "breast_cancer": skds.load_breast_cancer,
        "digits": skds.load_digits,
        "diabetes": skds.load_diabetes,
    }
    bunch = loaders[name]()
    df = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
    target_names = [str(t) for t in getattr(bunch, "target_names", [])]
    if name == "diabetes":  # the only regression set here
        target = pd.Series(bunch.target, name="target")
        task = "regression"
    else:
        target = pd.Series(bunch.target, name="target")
        task = "classification"
    return LoadedData(
        df=df,
        target=target,
        task=task,
        description=bunch.DESCR.split("\n", 1)[0] if getattr(bunch, "DESCR", "") else name,
        target_names=target_names,
    )


def _load_mini_corpus() -> LoadedData:
    """Materialise the embedded weather/finance text corpus as a :class:`LoadedData`."""
    import pandas as pd

    texts = [t for t, _ in _MINI_CORPUS]
    labels = [c for _, c in _MINI_CORPUS]
    df = pd.DataFrame({"text": texts})
    target = pd.Series(labels, name="category")
    return LoadedData(
        df=df,
        target=target,
        task="text",
        description="A small labelled corpus (weather vs. finance) for NLP demos.",
        target_names=sorted(set(labels)),
    )


DATASETS: dict[str, LabDataset] = {
    "iris": LabDataset(
        "iris",
        "Iris (flowers)",
        (TOPIC_ML,),
        "classification",
        "150 flowers, 4 measurements, 3 species — the classic starter classification set.",
        lambda: _load_sklearn_bunch("iris"),
    ),
    "wine": LabDataset(
        "wine",
        "Wine",
        (TOPIC_ML,),
        "classification",
        "178 wines, 13 chemical features, 3 cultivars.",
        lambda: _load_sklearn_bunch("wine"),
    ),
    "breast_cancer": LabDataset(
        "breast_cancer",
        "Breast cancer (diagnostic)",
        (TOPIC_ML,),
        "classification",
        "569 samples, 30 features, benign vs. malignant — a binary classification set.",
        lambda: _load_sklearn_bunch("breast_cancer"),
    ),
    "digits": LabDataset(
        "digits",
        "Handwritten digits",
        (TOPIC_ML, TOPIC_DL),
        "classification",
        "1,797 8×8 digit images (64 features, 10 classes) — great for a small neural net.",
        lambda: _load_sklearn_bunch("digits"),
    ),
    "diabetes": LabDataset(
        "diabetes",
        "Diabetes (regression)",
        (TOPIC_ML,),
        "regression",
        "442 patients, 10 features, a continuous disease-progression target.",
        lambda: _load_sklearn_bunch("diabetes"),
    ),
    "mini_text": LabDataset(
        "mini_text",
        "Weather vs. finance (text)",
        (TOPIC_NLP,),
        "text",
        "A tiny embedded labelled corpus for TF-IDF classification and cosine similarity.",
        _load_mini_corpus,
    ),
}


def datasets_for_topic(topic: str) -> list[LabDataset]:
    """Return the datasets applicable to ``topic``, in registry order."""
    return [d for d in DATASETS.values() if topic in d.topics]
