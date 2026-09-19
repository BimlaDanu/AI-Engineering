"""Stage 7: grammar, readability and style findings for the summaries.

Run with: make grammar
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.analysis.quality import assess, fault_rates
from src.log import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Assess every summary and write the quality table."""
    config.ensure_directories()

    if not config.SUMMARIES_FILE.exists():
        raise FileNotFoundError(f"{config.SUMMARIES_FILE} missing. Run `make summarize` first.")
    summaries = pd.read_parquet(config.SUMMARIES_FILE)

    assessed = assess(summaries)

    logger.info(
        "readability by method and language (Flesch reading ease, higher is easier):\n%s",
        assessed.pivot_table(index="method", columns="lang", values="flesch", aggfunc="median")
        .round(1)
        .to_string(),
    )
    logger.info(
        "mean words per sentence:\n%s",
        assessed.pivot_table(
            index="method", columns="lang", values="words_per_sentence", aggfunc="mean"
        )
        .round(1)
        .to_string(),
    )
    logger.info("style faults (%% of summaries):\n%s", fault_rates(assessed).to_string())

    assessed.to_parquet(config.GRAMMAR_FILE, index=False)
    logger.info("wrote %s", config.GRAMMAR_FILE)


if __name__ == "__main__":
    main()
