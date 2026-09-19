"""LLM-backed starter-code seeding for the AI/ML Lab's opt-in advanced cell.

Given the current 🎓 AI/ML Tutor lesson plus the chosen dataset, the model writes a short,
runnable snippet the learner can then edit and run in the sandbox. Structured output only —
we validate into a :class:`~src.lab.schemas.SeededSnippet`, never parse prose. This is a thin
wrapper over an injected chat model, so it stays framework-agnostic and unit-testable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.lab.datasets import LoadedData
from src.lab.schemas import SeededSnippet

_SEED_SYSTEM = (
    "You write short, correct, runnable Python teaching snippets for a data-science lab. "
    "You may use only numpy, pandas, matplotlib.pyplot, and scikit-learn. The handles `np`, "
    "`pd`, `plt` and the data variables described under 'Pre-bound variables' are already "
    "defined — use them exactly as described. `y` is ALREADY the target; never rebuild it from "
    "`df` (there is no label/target column inside `df`). Do not reload the data, read files, "
    "access the network, or import os/sys. Keep it under ~25 lines, print results, and draw at "
    "most one matplotlib figure."
)


def _distinct_labels(target: Any) -> list:
    """Return the distinct target values as plain Python scalars, first-seen order.

    ``.tolist()`` coerces a pandas Series / numpy array to native Python ints or strings (so an
    ``int64`` code becomes a real ``int``); a plain list (used by tests) is handled too.
    """
    if target is None:
        return []
    try:
        values = target.tolist()
    except AttributeError:
        try:
            values = list(target)
        except TypeError:
            return []
    seen: set = set()
    out: list = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _class_hint(data: LoadedData) -> str:
    """Describe the classification target's *actual* encoding for the prompt.

    The sklearn toy sets store ``y`` as integer codes (0, 1, …) while ``target_names`` holds
    the human-readable names ("malignant"). Advertising the names alone led the model to write
    ``pos_label="malignant"`` against an integer array — a guaranteed ``ValueError``. So when
    ``y`` is integer-coded we spell out the code→name mapping and tell the model to use the
    codes; when ``y`` already holds string labels (the text corpus) we list those verbatim.
    """
    labels = _distinct_labels(data.target)
    names = data.target_names
    if labels and all(isinstance(v, int) for v in labels):
        pairs = ", ".join(
            f"{code} = {names[code]}" if 0 <= code < len(names) else str(code) for code in labels
        )
        return (
            f"integer class codes: {pairs}. Use the integer code where a label is needed "
            f"(e.g. `pos_label={labels[-1]}`), never the name string"
        )
    if labels:
        listed = ", ".join(repr(v) for v in labels)
        return f"string class labels: {listed}. Use these strings directly where a label is needed"
    return ", ".join(names) or "n/a"


def _environment(data: LoadedData, feature_names: list[str]) -> str:
    """Describe the exact pre-bound variables for this dataset's task, to ground the model.

    The advanced cell binds different variables per task — for the text corpus ``X`` is raw
    strings (not a numeric matrix) and the only ``df`` column is ``text`` — so a generic
    "df of features / X matrix / y target" description misleads the model into inventing a
    label column. This spells out the real contract so the snippet uses what actually exists.
    """
    if data.task == "text":
        return (
            "Pre-bound variables (use these, do not redefine):\n"
            "- `texts`: list[str], the raw documents — the natural input for this task.\n"
            "- `df`: a pandas DataFrame with a single column `text` (the same raw documents). "
            "It has NO feature columns and NO label column.\n"
            "- `X`: the raw text as an array, NOT numeric — you must vectorise it first "
            "(e.g. `sklearn.feature_extraction.text.TfidfVectorizer`) before any model.\n"
            f"- `y`: the target ({_class_hint(data)}), already prepared and aligned to "
            "`texts`. Use it directly; do not derive it from `df`."
        )
    target_kind = "continuous regression targets" if data.task == "regression" else "class labels"
    cols = ", ".join(feature_names[:20]) or "the feature columns"
    return (
        "Pre-bound variables (use these, do not redefine):\n"
        f"- `df`: a pandas DataFrame of numeric feature columns ({cols}). It has NO target "
        "column.\n"
        "- `X`: the numeric feature matrix (`df.values`).\n"
        f"- `y`: the target array ({target_kind}"
        + (f"; {_class_hint(data)}" if data.task == "classification" else "")
        + "), already prepared and aligned to `X`. Use it directly; do not derive it from `df`."
    )


def seed_snippet(
    llm: BaseChatModel,
    *,
    topic: str,
    lesson: str,
    data: LoadedData,
    feature_names: list[str],
) -> SeededSnippet:
    """Ask the model for a runnable starter snippet for this lesson and dataset.

    Args:
        llm: The chat model to drive (a low temperature is recommended for stable code).
        topic: The Lab topic ("ML" / "NLP" / "DL").
        lesson: The ML-Tutor lesson question seeding the exercise.
        data: The materialised dataset (its task type and description guide the snippet).
        feature_names: The dataset's feature column names, to ground the code.

    Returns:
        A validated :class:`SeededSnippet` (code + explanation).
    """
    model = llm.with_structured_output(SeededSnippet, method="json_schema")
    human = (
        f"Topic: {topic}\n"
        f"Lesson to illustrate: {lesson}\n"
        f"Dataset ({data.task}): {data.description}\n\n"
        f"{_environment(data, feature_names)}\n\n"
        "Write a snippet that helps the learner explore this lesson hands-on with this data."
    )
    result = model.invoke([("system", _SEED_SYSTEM), ("human", human)])
    if isinstance(result, SeededSnippet):
        return result
    return SeededSnippet(code="", explanation="The model returned no snippet.")
