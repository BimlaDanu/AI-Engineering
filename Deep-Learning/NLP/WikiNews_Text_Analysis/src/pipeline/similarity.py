"""Stage 8: similarity between summaries and source articles (requirement 5).

Run with: make similarity
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.analysis.similarity import distribution, drivers, extremes, score, threshold_check
from src.log import get_logger

logger = get_logger(__name__)

_THRESHOLD = 0.8


def main() -> None:
    """Score every summary against its source and write the scored table."""
    config.ensure_directories()

    if not config.SUMMARIES_FILE.exists():
        raise FileNotFoundError(f"{config.SUMMARIES_FILE} missing. Run `make summarize` first.")
    summaries = pd.read_parquet(config.SUMMARIES_FILE)

    scored = score(summaries)

    logger.info(
        "similarity by method and category:\n%s",
        distribution(scored, _THRESHOLD).to_string(index=False),
    )
    logger.info(
        "share at or above %.1f (%%):\n%s",
        _THRESHOLD,
        threshold_check(scored, _THRESHOLD).to_string(),
    )
    logger.info("what the score tracks:\n%s", drivers(scored).to_string())

    highest, lowest = extremes(scored)
    logger.info("highest scoring summaries:\n%s", highest.to_string(index=False, max_colwidth=45))
    logger.info("lowest scoring summaries:\n%s", lowest.to_string(index=False, max_colwidth=45))

    scored.to_parquet(config.SIMILARITY_FILE, index=False)
    logger.info("wrote %s", config.SIMILARITY_FILE)


if __name__ == "__main__":
    main()
