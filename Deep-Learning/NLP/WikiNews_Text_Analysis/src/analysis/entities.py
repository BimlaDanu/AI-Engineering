"""Aggregate, trace and cluster the extracted entities.

Three views on the mention table: which entities dominate each category, when
each entered and left the news, and whether the leading entities fall into
coherent semantic groups.

An entity's weight is the number of articles it appears in, not the number of
mentions, so one long article cannot outvote a recurring topic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src import config
from src.analysis.embeddings import encode
from src.log import get_logger
from src.nlp.labels import LOCATION, display_form
from src.utils import as_int

logger = get_logger(__name__)


def aggregate_entities(entities: pd.DataFrame) -> pd.DataFrame:
    """Collapse mentions into one row per entity, language and category.

    The coarse label is derived here rather than used as a grouping key. A name
    the pipeline tagged two ways in one article would otherwise split into two
    rows and be counted as two articles.

    Args:
        entities: The mention table from the NER stage.

    Returns:
        One row per ``(lang, category, entity_key)`` with mention and article
        counts, the years it spans, its dominant coarse label and a display
        name.
    """
    grouped = entities.groupby(["lang", "category", "entity_key"], observed=True)
    aggregated = grouped.agg(
        mentions=("pageid", "size"),
        articles=("pageid", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        label=("label", lambda values: values.mode().iat[0]),
    ).reset_index()

    # The display name is chosen per language, not per category, so one entity
    # never appears as "U.S" in one panel and "US" in the next.
    aggregated = aggregated.merge(display_surfaces(entities), on=["lang", "entity_key"], how="left")

    return aggregated.sort_values(["lang", "category", "articles"], ascending=[True, True, False])


def display_surfaces(entities: pd.DataFrame) -> pd.DataFrame:
    """Choose one display name per entity per language.

    Args:
        entities: The mention table.

    Returns:
        ``lang``, ``entity_key`` and the ``surface`` to display it under: the
        most frequent spelling, ties broken towards the longer form, tidied by
        ``display_form``.
    """
    counts = (
        entities.groupby(["lang", "entity_key", "entity"], observed=True)
        .size()
        .reset_index(name="n")
    )
    counts["length"] = counts["entity"].str.len()
    best = counts.sort_values(["n", "length"], ascending=[False, False]).drop_duplicates(
        ["lang", "entity_key"]
    )
    best = best.assign(
        surface=[
            display_form(str(entity), str(lang))
            for entity, lang in zip(best["entity"], best["lang"], strict=True)
        ]
    )
    return best[["lang", "entity_key", "surface"]]


def rank_entities(aggregated: pd.DataFrame, lang: str = "en", n: int = 20) -> pd.DataFrame:
    """Rank a language's entities by the articles they appear in.

    Each article carries one primary category, so summing the per-category
    article counts gives the entity's article total without double counting.

    Args:
        aggregated: Output of ``aggregate_entities``.
        lang: Language to rank.
        n: How many entities to return.

    Returns:
        The top ``n`` entities with their totals, display name and most common
        coarse label.
    """
    subset = aggregated[aggregated["lang"] == lang]
    totals = subset.groupby("entity_key", as_index=False).agg(
        articles=("articles", "sum"),
        mentions=("mentions", "sum"),
        first_year=("first_year", "min"),
        last_year=("last_year", "max"),
        surface=("surface", "first"),
        label=("label", lambda values: values.mode().iat[0]),
    )
    return totals.nlargest(n, "articles").reset_index(drop=True)


def top_entities(
    aggregated: pd.DataFrame,
    lang: str = "en",
    label: str | None = None,
    n: int = 15,
) -> pd.DataFrame:
    """Take the leading entities per category for one language.

    Args:
        aggregated: Output of ``aggregate_entities``.
        lang: Language to report on.
        label: Restrict to one coarse label, or ``None`` for all of them.
        n: How many entities to keep per category.

    Returns:
        The top ``n`` entities of each category, ranked by article count.
    """
    subset = aggregated[aggregated["lang"] == lang]
    if label is not None:
        subset = subset[subset["label"] == label]
    return (
        subset.sort_values("articles", ascending=False)
        .groupby("category", observed=True)
        .head(n)
        .reset_index(drop=True)
    )


def entity_timeline(entities: pd.DataFrame, keys: list[str], lang: str = "en") -> pd.DataFrame:
    """Count how many articles mention each entity, year by year.

    Args:
        entities: The mention table.
        keys: Normalised entity keys to trace.
        lang: Language to restrict to.

    Returns:
        Years as rows, entities as columns, article counts as values. Missing
        years are filled with zero so a gap reads as a gap.
    """
    subset = entities[(entities["lang"] == lang) & (entities["entity_key"].isin(keys))]
    subset = subset.dropna(subset=["year"])

    counts = (
        subset.groupby(["year", "entity_key"], observed=True)["pageid"]
        .nunique()
        .unstack(fill_value=0)
    )
    if counts.empty:
        return counts

    full_years = range(int(counts.index.min()), int(counts.index.max()) + 1)
    return counts.reindex(full_years, fill_value=0).sort_index()


def geographic_focus(entities: pd.DataFrame, lang: str = "en", n: int = 10) -> pd.DataFrame:
    """Rank the places the news concentrates on, per category.

    The places are read back out of the LOCATION entities rather than matched
    against a country list. No gazetteer is used, so a heavily reported city
    appears alongside the states.

    Args:
        entities: The mention table.
        lang: Language to report on.
        n: Places to keep per category.

    Returns:
        The leading places per category, with their article counts, the years
        they span and the name they display under.
    """
    subset = entities[entities["lang"] == lang]
    places = subset[subset["label"] == LOCATION]
    ranked = (
        places.groupby(["category", "entity_key"], observed=True)
        .agg(
            articles=("pageid", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .reset_index()
    )
    # The same display name as every other table, so one place is not "U.S"
    # in one category and "US" in the next.
    names = display_surfaces(subset).set_index("entity_key")["surface"]
    ranked["place"] = ranked["entity_key"].map(names)
    return (
        ranked.sort_values("articles", ascending=False)
        .groupby("category", observed=True)
        .head(n)
        .reset_index(drop=True)
    )


def cluster_entities(surfaces: list[str], n_clusters: int = 6) -> tuple[np.ndarray, float]:
    """Group entity names into semantic clusters.

    Args:
        surfaces: Entity display names to cluster.
        n_clusters: Number of clusters to fit.

    Returns:
        The cluster assignment per entity, and the silhouette score. On strings
        as short as entity names the score is usually modest, so it is reported
        with the clusters.

    Raises:
        ValueError: If there are fewer entities than requested clusters.
    """
    if len(surfaces) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} entities to fit {n_clusters} clusters")

    vectors = encode(surfaces)
    model = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED, n_init=10)
    assignments = model.fit_predict(vectors)

    score = float(silhouette_score(vectors, assignments, metric="cosine"))
    logger.info(
        "clustered %d entities into %d groups (silhouette %.3f)", len(surfaces), n_clusters, score
    )
    return assignments, score


def describe_clusters(
    surfaces: list[str], assignments: np.ndarray, labels: list[str]
) -> pd.DataFrame:
    """Summarise what each cluster contains.

    Args:
        surfaces: The clustered entity names.
        assignments: Cluster index per entity.
        labels: Coarse entity label per entity.

    Returns:
        One row per cluster with its size, its dominant entity label and a
        sample of members.
    """
    frame = pd.DataFrame({"surface": surfaces, "cluster": assignments, "label": labels})
    rows = []
    for cluster, group in frame.groupby("cluster"):
        rows.append(
            {
                "cluster": as_int(cluster),
                "size": len(group),
                "dominant_label": group["label"].mode().iat[0],
                "members": ", ".join(group["surface"].head(8)),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
