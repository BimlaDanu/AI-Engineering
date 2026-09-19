"""Guided, parameterised ML/DL/NLP recipes for the 🔬 AI/ML Lab.

Each :class:`Recipe` is the *safe* default path: the app runs its **own** vetted code (no
arbitrary execution) against a bundled :class:`~src.lab.datasets.LoadedData`, and returns a
:class:`LabResult` — a short summary, an optional table, an optional matplotlib figure, and
the **equivalent source code** it ran. That last field is the teaching bridge to the opt-in
advanced cell: "here's exactly what just happened — now edit it."

Recipes are pure functions of ``(data, params)``. scikit-learn / matplotlib are imported
lazily inside each so this module imports even when those optional deps are absent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.lab.datasets import TOPIC_DL, TOPIC_ML, TOPIC_NLP, LoadedData


@dataclass(frozen=True)
class ParamSpec:
    """One tunable knob a recipe exposes, rendered as a widget by the page."""

    name: str
    label: str
    kind: str  # "select" | "slider_int" | "slider_float" | "checkbox"
    default: Any
    options: tuple[Any, ...] = ()
    min: float = 0.0
    max: float = 1.0
    step: float = 1.0


@dataclass
class LabResult:
    """The output of a recipe (or the equivalent of a sandbox run) for the page to render.

    ``metrics`` carries the run's headline numbers (e.g. ``accuracy``, ``silhouette``) as
    machine-readable floats, so a :class:`~src.lab.challenges.Challenge` can grade the outcome
    against a goal *without scraping the prose summary*. It is empty for recipes that produce
    no single checkable number (e.g. exploration).
    """

    summary: str
    code: str = ""
    figure: Any | None = None  # matplotlib Figure
    table: Any | None = None  # pandas DataFrame
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Recipe:
    """A guided exercise: metadata, its tunable params, and a pure run function."""

    key: str
    topic: str
    label: str
    description: str
    params: tuple[ParamSpec, ...]
    run: Callable[[LoadedData, dict[str, Any]], LabResult]


def _new_fig(figsize: tuple[float, float] = (5.0, 3.5)):
    """Create a standalone matplotlib figure/axes (Agg backend set by the caller/page)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# --------------------------------------------------------------------------------------- ML
def _explore(data: LoadedData, params: dict[str, Any]) -> LabResult:
    """Summary statistics plus a scatter of two chosen features, coloured by target."""
    df = data.df
    cols = list(df.columns)
    x_col = params.get("x", cols[0])
    y_col = params.get("y", cols[1] if len(cols) > 1 else cols[0])

    fig, ax = _new_fig()
    color = None if data.target is None else data.target
    scatter = ax.scatter(df[x_col], df[y_col], c=color, cmap="viridis", s=18, alpha=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs. {y_col}")
    if color is not None:
        fig.colorbar(scatter, ax=ax, label="target")
    fig.tight_layout()

    code = (
        "import matplotlib.pyplot as plt\n"
        f"df.plot.scatter(x={x_col!r}, y={y_col!r}, c=y, cmap='viridis')\n"
        "print(df.describe())\n"
    )
    return LabResult(
        summary=f"Showing `describe()` and a scatter of **{x_col}** vs **{y_col}**.",
        code=code,
        figure=fig,
        table=df.describe().T.reset_index(names="feature"),
    )


def _classify(data: LoadedData, params: dict[str, Any]) -> LabResult:
    """Train a classifier, report accuracy, and plot the confusion matrix."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    if data.target is None or data.task != "classification":
        return LabResult(summary="", error="This recipe needs a classification dataset.")

    model_name = params.get("model", "LogisticRegression")
    test_size = float(params.get("test_size", 0.25))
    x_tr, x_te, y_tr, y_te = train_test_split(
        data.df.values,
        data.target.values,
        test_size=test_size,
        random_state=0,
        stratify=data.target.values,
    )
    x_tr_s = StandardScaler().fit(x_tr)
    x_tr, x_te = x_tr_s.transform(x_tr), x_tr_s.transform(x_te)

    if model_name == "KNeighborsClassifier":
        clf = KNeighborsClassifier(n_neighbors=int(params.get("k", 5)))
    else:
        clf = LogisticRegression(max_iter=1000, C=float(params.get("C", 1.0)))
    clf.fit(x_tr, y_tr)
    preds = clf.predict(x_te)
    acc = accuracy_score(y_te, preds)

    fig, ax = _new_fig((4.5, 4.0))
    ConfusionMatrixDisplay.from_predictions(
        y_te,
        preds,
        ax=ax,
        colorbar=False,
        display_labels=data.target_names or None,
    )
    ax.set_title(f"{model_name} — accuracy {acc:.3f}")
    fig.tight_layout()

    code = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        f"from sklearn.linear_model import LogisticRegression\n"
        f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size={test_size}, "
        "random_state=0, stratify=y)\n"
        "sc = StandardScaler().fit(X_tr)\n"
        f"clf = LogisticRegression(max_iter=1000, C={float(params.get('C', 1.0))}).fit("
        "sc.transform(X_tr), y_tr)\n"
        "preds = clf.predict(sc.transform(X_te))\n"
        "print('accuracy', (preds == y_te).mean())\n"
    )
    return LabResult(
        summary=f"**{model_name}** reached **{acc:.1%}** accuracy on a {test_size:.0%} "
        "hold-out split (features standardised).",
        code=code,
        figure=fig,
        metrics={"accuracy": float(acc)},
    )


def _cluster(data: LoadedData, params: dict[str, Any]) -> LabResult:
    """K-Means clustering visualised on the first two principal components."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    k = int(params.get("k", 3))
    x = StandardScaler().fit_transform(data.df.values)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
    coords = PCA(n_components=2, random_state=0).fit_transform(x)
    sil = silhouette_score(x, labels) if k > 1 else float("nan")

    fig, ax = _new_fig()
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"K-Means (k={k}) on 2-D PCA — silhouette {sil:.3f}")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()

    code = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.decomposition import PCA\n"
        "Xs = StandardScaler().fit_transform(X)\n"
        f"labels = KMeans(n_clusters={k}, n_init=10, random_state=0).fit_predict(Xs)\n"
        "coords = PCA(n_components=2).fit_transform(Xs)\n"
    )
    return LabResult(
        summary=f"**K-Means** with k={k} — silhouette score **{sil:.3f}** "
        "(higher = better-separated clusters).",
        code=code,
        figure=fig,
        metrics={"silhouette": float(sil)},
    )


# --------------------------------------------------------------------------------------- DL
def _small_net(data: LoadedData, params: dict[str, Any]) -> LabResult:
    """Train a small multi-layer perceptron and plot its training loss curve."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if data.target is None or data.task != "classification":
        return LabResult(summary="", error="This recipe needs a classification dataset.")

    width = int(params.get("hidden", 64))
    max_iter = int(params.get("max_iter", 150))
    x_tr, x_te, y_tr, y_te = train_test_split(
        data.df.values,
        data.target.values,
        test_size=0.25,
        random_state=0,
        stratify=data.target.values,
    )
    sc = StandardScaler().fit(x_tr)
    clf = MLPClassifier(
        hidden_layer_sizes=(width, width),
        max_iter=max_iter,
        random_state=0,
    )
    clf.fit(sc.transform(x_tr), y_tr)
    acc = accuracy_score(y_te, clf.predict(sc.transform(x_te)))

    fig, ax = _new_fig()
    ax.plot(clf.loss_curve_)
    ax.set_xlabel("iteration")
    ax.set_ylabel("training loss")
    ax.set_title(f"MLP ({width}×2) — test accuracy {acc:.3f}")
    fig.tight_layout()

    code = (
        "from sklearn.neural_network import MLPClassifier\n"
        "from sklearn.preprocessing import StandardScaler\n"
        f"clf = MLPClassifier(hidden_layer_sizes=({width}, {width}), max_iter={max_iter}, "
        "random_state=0)\n"
        "clf.fit(StandardScaler().fit_transform(X), y)\n"
        "plt.plot(clf.loss_curve_)\n"
    )
    return LabResult(
        summary=f"A 2-layer MLP ({width} units each) hit **{acc:.1%}** test accuracy over "
        f"{len(clf.loss_curve_)} iterations. Watch the loss curve flatten as it converges.",
        code=code,
        figure=fig,
        metrics={"accuracy": float(acc), "iterations": float(len(clf.loss_curve_))},
    )


# -------------------------------------------------------------------------------------- NLP
def _tfidf_classify(data: LoadedData, params: dict[str, Any]) -> LabResult:
    """Vectorise text with TF-IDF and train a linear classifier over it."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    if data.task != "text" or data.target is None:
        return LabResult(summary="", error="This recipe needs the text dataset.")

    texts = data.df["text"].tolist()
    # .to_numpy() (not .values): a string target is an Arrow-backed extension array under
    # pandas 3, which scikit-learn's fancy indexing can't slice — coerce to a real ndarray.
    y = data.target.to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(texts, y, test_size=0.34, random_state=0)
    vec = TfidfVectorizer(stop_words="english")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(x_tr), y_tr)
    acc = accuracy_score(y_te, clf.predict(vec.transform(x_te)))

    import pandas as pd

    vocab = vec.get_feature_names_out()
    top = (
        pd.DataFrame({"term": vocab, "tfidf_weight_class_0": clf.coef_[0]})
        .sort_values("tfidf_weight_class_0")
        .tail(8)
        .reset_index(drop=True)
    )

    code = (
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "vec = TfidfVectorizer(stop_words='english')\n"
        "clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(texts), y)\n"
    )
    return LabResult(
        summary=f"TF-IDF + logistic regression classified the held-out texts at "
        f"**{acc:.0%}** accuracy. The table shows the terms most indicative of one class.",
        code=code,
        table=top,
        metrics={"accuracy": float(acc)},
    )


def _cosine(data: LoadedData, params: dict[str, Any]) -> LabResult:
    """Compute pairwise TF-IDF cosine similarity across the corpus and show it as a heatmap."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if data.task != "text":
        return LabResult(summary="", error="This recipe needs the text dataset.")

    texts = data.df["text"].tolist()
    mat = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sim = cosine_similarity(mat)

    n = len(texts)
    best, bi, bj = -1.0, 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > best:
                best, bi, bj = sim[i, j], i, j

    fig, ax = _new_fig((4.5, 4.0))
    im = ax.imshow(sim, cmap="magma", vmin=0, vmax=1)
    ax.set_title("TF-IDF cosine similarity")
    ax.set_xlabel("document")
    ax.set_ylabel("document")
    fig.colorbar(im, ax=ax, label="cosine")
    fig.tight_layout()

    code = (
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.metrics.pairwise import cosine_similarity\n"
        "M = TfidfVectorizer(stop_words='english').fit_transform(texts)\n"
        "sim = cosine_similarity(M)\n"
    )
    return LabResult(
        summary=f"Most similar pair (cosine **{best:.2f}**):\n\n- *{texts[bi]}*\n- *{texts[bj]}*",
        code=code,
        figure=fig,
        metrics={"best_cosine": float(best)},
    )


# Ordered recipe registry. The page filters by topic and renders each recipe's params.
RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "explore",
        TOPIC_ML,
        "Explore & plot",
        "Summary statistics and a scatter of two features, coloured by the target.",
        (),  # feature axes are chosen from the live dataset by the page
        _explore,
    ),
    Recipe(
        "classify",
        TOPIC_ML,
        "Train a classifier",
        "Fit a classifier on a train/test split and inspect its confusion matrix.",
        (
            ParamSpec(
                "model",
                "Model",
                "select",
                "LogisticRegression",
                ("LogisticRegression", "KNeighborsClassifier"),
            ),
            ParamSpec("test_size", "Test size", "slider_float", 0.25, min=0.1, max=0.5, step=0.05),
            ParamSpec(
                "C", "Regularisation C (LogReg)", "slider_float", 1.0, min=0.01, max=10.0, step=0.01
            ),
            ParamSpec("k", "Neighbours k (KNN)", "slider_int", 5, min=1, max=15, step=1),
        ),
        _classify,
    ),
    Recipe(
        "cluster",
        TOPIC_ML,
        "Cluster (K-Means + PCA)",
        "Cluster the features with K-Means and visualise them on 2-D PCA.",
        (ParamSpec("k", "Clusters k", "slider_int", 3, min=2, max=8, step=1),),
        _cluster,
    ),
    Recipe(
        "small_net",
        TOPIC_DL,
        "Train a small neural net",
        "Fit a 2-layer MLP and watch its training-loss curve converge.",
        (
            ParamSpec("hidden", "Units per layer", "slider_int", 64, min=8, max=128, step=8),
            ParamSpec("max_iter", "Max iterations", "slider_int", 150, min=50, max=400, step=25),
        ),
        _small_net,
    ),
    Recipe(
        "tfidf_classify",
        TOPIC_NLP,
        "TF-IDF text classification",
        "Vectorise text with TF-IDF and classify it with logistic regression.",
        (),
        _tfidf_classify,
    ),
    Recipe(
        "cosine",
        TOPIC_NLP,
        "Cosine similarity",
        "Build a TF-IDF cosine-similarity matrix and find the most similar documents.",
        (),
        _cosine,
    ),
)


def recipes_for_topic(topic: str) -> list[Recipe]:
    """Return the recipes registered for ``topic``, in registry order."""
    return [r for r in RECIPES if r.topic == topic]
