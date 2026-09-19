"""Stage 10: render every figure into reports/figures.

Run with: make figures
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.analysis.topics import evaluate_language
from src.data.loader import load_articles
from src.log import get_logger
from src.viz import plots

logger = get_logger(__name__)


def _require(path: Path) -> pd.DataFrame:
    """Read a stage output, with a message naming the stage that writes it.

    Args:
        path: The parquet file to read.

    Returns:
        The table.

    Raises:
        FileNotFoundError: If the stage that writes it has not run.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing. Run the earlier pipeline stages first.")
    return pd.read_parquet(path)


def main() -> None:
    """Render all report figures."""
    config.ensure_directories()

    plots.top_entities_per_category(_require(config.ENTITY_AGGREGATE_FILE))
    plots.entity_timeline(_require(config.ENTITY_TIMELINE_FILE))
    plots.ner_recall_by_language(_require(config.NER_RECALL_FILE))

    scored = _require(config.SIMILARITY_FILE)
    plots.similarity_distribution(scored)
    plots.similarity_vs_overlap(scored)

    # The confusion matrix is not stored as a table, so it is refitted here.
    plots.topic_confusion(evaluate_language(load_articles(), "en").confusion)

    logger.info("figures written to %s", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
