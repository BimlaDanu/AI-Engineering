"""The report's figures.

Each function renders one figure and returns the path it was written to.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src import config
from src.log import get_logger
from src.utils import as_int
from src.viz.style import (
    SEQUENTIAL_CMAP,
    TEXT_MUTED,
    apply_style,
    colour_for,
    strip_chrome,
)

logger = get_logger(__name__)


def _save(figure: plt.Figure, name: str) -> Path:
    """Write a figure into the report's figure directory.

    Args:
        figure: The rendered figure.
        name: File name, including the extension.

    Returns:
        The path written.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    figure.savefig(path)
    plt.close(figure)
    logger.info("wrote %s", path.name)
    return path


def top_entities_per_category(aggregated: pd.DataFrame, n: int = 8) -> Path:
    """Bar chart of the most reported entities in each category, one panel each.

    Args:
        aggregated: Output of ``aggregate_entities``.
        n: Entities to show per category.

    Returns:
        The path written.
    """
    apply_style()
    english = aggregated[aggregated["lang"] == "en"]
    categories = [c for c in config.CATEGORIES if c in set(english["category"])]

    figure, axes_grid = plt.subplots(2, 2, figsize=(12, 7.5))
    for slot, (category, axes) in enumerate(zip(categories, axes_grid.flat, strict=False)):
        subset = english[english["category"] == category].nlargest(n, "articles").iloc[::-1]
        bars = axes.barh(subset["surface"], subset["articles"], color=colour_for(slot), height=0.68)

        for bar, value in zip(bars, subset["articles"], strict=True):
            axes.text(
                bar.get_width() + max(subset["articles"]) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                str(int(value)),
                va="center",
                fontsize=8,
                color=TEXT_MUTED,
            )

        axes.set_title(category, loc="left")
        axes.set_xlabel("articles mentioning the entity")
        axes.grid(axis="y", visible=False)
        axes.margins(x=0.14)
        strip_chrome(axes)

    for unused in axes_grid.flat[len(categories) :]:
        unused.set_visible(False)

    figure.suptitle(
        "Most reported entities by news category (English)",
        x=0.5,
        y=1.0,
        fontsize=13,
        weight="bold",
    )
    figure.tight_layout()
    return _save(figure, "entities_by_category.png")


def entity_timeline(timeline: pd.DataFrame, n_series: int = 6) -> Path:
    """Line chart tracing the leading entities year by year.

    Args:
        timeline: Years as rows, entities as columns, article counts as values.
        n_series: How many entities to draw; the rest are dropped.

    Returns:
        The path written.
    """
    apply_style()
    leading = timeline[timeline.sum().nlargest(n_series).index]

    figure, axes = plt.subplots(figsize=(11, 5))
    for slot, column in enumerate(leading.columns):
        axes.plot(
            leading.index,
            leading[column],
            linewidth=2,
            color=colour_for(slot),
            label=str(column),
        )

    # Years are integers; left alone, matplotlib offers ticks like "2007.5".
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlabel("year of publication")
    axes.set_ylabel("articles mentioning the entity")
    axes.set_title("When the leading entities were in the news", loc="left")
    axes.legend(ncols=3, loc="upper right")
    axes.grid(axis="x", visible=False)
    strip_chrome(axes)
    figure.tight_layout()
    return _save(figure, "entity_timeline.png")


def similarity_distribution(scored: pd.DataFrame, threshold: float = 0.8) -> Path:
    """Box plot of the similarity scores per category and summarizer.

    Args:
        scored: Output of ``score``.
        threshold: The level to mark.

    Returns:
        The path written.
    """
    apply_style()
    methods = sorted(scored["method"].unique())
    categories = [c for c in config.CATEGORIES if c in set(scored["category"])]

    figure, axes = plt.subplots(figsize=(11, 5.5))
    width = 0.36

    for slot, method in enumerate(methods):
        positions = np.arange(len(categories)) + (slot - (len(methods) - 1) / 2) * width
        data = [
            scored[(scored["method"] == method) & (scored["category"] == category)][
                "similarity"
            ].to_numpy()
            for category in categories
        ]
        box = axes.boxplot(
            data,
            positions=positions,
            widths=width * 0.82,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.6},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": colour_for(slot),
                "markeredgecolor": "none",
                "alpha": 0.55,
            },
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colour_for(slot))
            patch.set_edgecolor("white")
            patch.set_linewidth(1.2)
        for element in ("whiskers", "caps"):
            for line in box[element]:
                line.set_color(colour_for(slot))
        axes.plot([], [], color=colour_for(slot), linewidth=6, label=method)

    axes.axhline(threshold, color=TEXT_MUTED, linestyle="--", linewidth=1.2)
    axes.text(
        len(categories) - 0.5,
        threshold + 0.008,
        f"threshold {threshold}",
        ha="right",
        fontsize=8,
        color=TEXT_MUTED,
    )

    axes.set_xticks(np.arange(len(categories)))
    axes.set_xticklabels([c.replace(" and ", " &\n") for c in categories])
    axes.set_ylabel("cosine similarity to the source article")
    axes.set_title("Summary-to-source similarity by category and summarizer", loc="left")
    axes.legend(title="")
    axes.grid(axis="x", visible=False)
    strip_chrome(axes)
    figure.tight_layout()
    return _save(figure, "similarity_distribution.png")


def similarity_vs_overlap(scored: pd.DataFrame) -> Path:
    """Scatter of semantic similarity against how much text was reused.

    Extractive summaries sit at the right-hand edge, where overlap is 1.0 by
    construction.

    Args:
        scored: Output of ``score``.

    Returns:
        The path written.
    """
    apply_style()
    figure, axes = plt.subplots(figsize=(8.5, 5.5))

    for slot, method in enumerate(sorted(scored["method"].unique())):
        subset = scored[scored["method"] == method]
        axes.scatter(
            subset["lexical_overlap"],
            subset["similarity"],
            s=34,
            color=colour_for(slot),
            alpha=0.72,
            edgecolor="white",
            linewidth=0.8,
            label=method,
        )

    axes.set_xlabel("lexical overlap with the source (share of summary words reused)")
    axes.set_ylabel("cosine similarity to the source")
    axes.set_title("Lexical reuse does not determine semantic similarity", loc="left")
    axes.legend()
    strip_chrome(axes)
    figure.tight_layout()
    return _save(figure, "similarity_vs_overlap.png")


def ner_recall_by_language(recall: pd.DataFrame) -> Path:
    """Bar chart of the cross-lingual recall proxy per language.

    Args:
        recall: Output of ``recall_by_language``.

    Returns:
        The path written.
    """
    apply_style()
    figure, axes = plt.subplots(figsize=(7, 4.2))

    ordered = recall.sort_values("recall_proxy")
    bars = axes.bar(ordered["lang"], ordered["recall_proxy"], color=colour_for(0), width=0.55)
    for bar, value, checked in zip(bars, ordered["recall_proxy"], ordered["checked"], strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{value:.3f}\nn={checked}",
            ha="center",
            fontsize=8,
            color=TEXT_MUTED,
        )

    axes.set_ylim(0, 1.12)
    axes.set_ylabel("share of English-named people also tagged")
    axes.set_xlabel("language")
    axes.set_title("Person names present in the text but missed by the model", loc="left")
    axes.grid(axis="x", visible=False)
    strip_chrome(axes)
    figure.tight_layout()
    return _save(figure, "ner_recall_by_language.png")


def topic_confusion(confusion: pd.DataFrame) -> Path:
    """Heatmap of the topic classifier's confusion matrix.

    Args:
        confusion: Square matrix with labelled rows and columns.

    Returns:
        The path written.
    """
    apply_style()
    figure, axes = plt.subplots(figsize=(7.5, 6))

    image = axes.imshow(confusion.to_numpy(), cmap=SEQUENTIAL_CMAP)
    labels = [c.replace(" and ", " &\n") for c in confusion.index]
    axes.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axes.set_yticks(range(len(labels)), labels)

    peak = confusion.to_numpy().max()
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            value = as_int(confusion.iat[row, column])
            axes.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > peak * 0.55 else "#1a1a19",
            )

    axes.set_xlabel("predicted category")
    axes.set_ylabel("true category")
    axes.set_title("Topic classifier confusion matrix (English, held-out set)", loc="left")
    axes.grid(visible=False)
    figure.colorbar(image, ax=axes, shrink=0.75, label="articles")
    figure.tight_layout()
    return _save(figure, "topic_confusion.png")
