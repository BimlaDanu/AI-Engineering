"""Stage 4: analyse the extracted entities.

Four views on the mention table:

1. Aggregated: the entities that dominate each news category.
2. Over time: when those entities entered and left the news.
3. Geographic focus: the places each category concentrates on.
4. Semantic groups: whether the leading entities cluster coherently.

Run with: make ner-analysis
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.analysis.entities import (
    aggregate_entities,
    cluster_entities,
    describe_clusters,
    entity_timeline,
    geographic_focus,
    rank_entities,
    top_entities,
)
from src.log import get_logger

logger = get_logger(__name__)

# Entities taken into the clustering and the timeline. Enough to show structure,
# small enough that the tables stay readable.
_N_FOR_CLUSTERING = 120
_N_FOR_TIMELINE = 8
_N_CLUSTERS = 6


def load_entities() -> pd.DataFrame:
    """Read the mention table written by the NER stage.

    Returns:
        The mention table.

    Raises:
        FileNotFoundError: If the NER stage has not run.
    """
    if not config.ENTITIES_FILE.exists():
        raise FileNotFoundError(f"{config.ENTITIES_FILE} missing. Run `make ner` first.")
    return pd.read_parquet(config.ENTITIES_FILE)


def build_clusters(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Cluster the most widely reported English entities.

    English only: clustering four languages at once mostly rediscovers the
    languages rather than the semantics.

    Args:
        aggregated: Output of ``aggregate_entities``.

    Returns:
        One row per entity with its cluster assignment.
    """
    english = rank_entities(aggregated, lang="en", n=_N_FOR_CLUSTERING)

    assignments, score = cluster_entities(english["surface"].tolist(), n_clusters=_N_CLUSTERS)
    english["cluster"] = assignments
    english["silhouette"] = score

    description = describe_clusters(
        english["surface"].tolist(), assignments, english["label"].astype(str).tolist()
    )
    logger.info("semantic clusters:\n%s", description.to_string(index=False))
    return english


def main() -> None:
    """Run the entity analyses and write their tables."""
    config.ensure_directories()
    entities = load_entities()

    # 1. Aggregated view.
    aggregated = aggregate_entities(entities)
    aggregated.to_parquet(config.ENTITY_AGGREGATE_FILE, index=False)
    logger.info("aggregated to %d entity/category rows", len(aggregated))

    leaders = top_entities(aggregated, lang="en", n=8)
    logger.info(
        "top English entities per category:\n%s",
        leaders[["category", "surface", "label", "articles", "mentions"]].to_string(index=False),
    )

    # 1b. Where the news is concentrated, read back out of the entities.
    places = geographic_focus(entities, lang="en", n=6)
    config.GEOGRAPHY_FILE.parent.mkdir(parents=True, exist_ok=True)
    places.to_parquet(config.GEOGRAPHY_FILE, index=False)
    logger.info(
        "geographic focus per category:\n%s",
        places[["category", "place", "articles", "first_year", "last_year"]].to_string(index=False),
    )

    # 2. Dynamics over time.
    leaders = rank_entities(aggregated, lang="en", n=_N_FOR_TIMELINE)
    traced = leaders["entity_key"].tolist()
    timeline = entity_timeline(entities, traced, lang="en")
    # Column names are matching keys; relabel them with the display name every
    # other table uses, so the figure reads "UTC" rather than "Utc".
    timeline = timeline.rename(
        columns=dict(zip(leaders["entity_key"], leaders["surface"], strict=True))
    )
    timeline.to_parquet(config.ENTITY_TIMELINE_FILE)
    logger.info(
        "articles per year for the %d most reported entities:\n%s",
        len(traced),
        timeline.to_string(),
    )

    # 3. Semantic clusters.
    clusters = build_clusters(aggregated)
    clusters.to_parquet(config.ENTITY_CLUSTERS_FILE, index=False)

    logger.info(
        "wrote %s, %s, %s",
        config.ENTITY_AGGREGATE_FILE.name,
        config.ENTITY_TIMELINE_FILE.name,
        config.ENTITY_CLUSTERS_FILE.name,
    )


if __name__ == "__main__":
    main()
