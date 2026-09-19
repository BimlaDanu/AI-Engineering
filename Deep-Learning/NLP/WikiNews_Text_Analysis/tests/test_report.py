import pandas as pd

from src.analysis.quality import assess
from src.pipeline.report import section_similarity, section_summaries


def _scored(methods):
    source = "The council approved the budget on Monday. Housing stays flat."
    return pd.DataFrame(
        [
            {
                "pageid": 1,
                "lang": "en",
                "category": "Politics and conflicts",
                "method": method,
                "title": "Budget",
                "summary": "The council approved the budget on Monday.",
                "text": source,
                "source_chars": len(source),
                "summary_chars": 41,
                "compression": 0.6,
                "similarity": 0.9,
                "lexical_overlap": 1.0,
            }
            for method in methods
        ]
    )


def test_sections_build_with_both_summarizers():
    scored = _scored(["extractive", "abstractive"])
    assert "## 6." in section_similarity(scored)
    assert "### 5.1" in section_summaries(scored, assess(scored))


def test_sections_build_when_only_one_summarizer_ran():
    # `make summarize SUMMARIZER=extractive` leaves one method, and the report
    # must still build rather than raise on the missing one.
    scored = _scored(["extractive"])
    assert "## 6." in section_similarity(scored)
    assert "## 5." in section_summaries(scored, assess(scored))
