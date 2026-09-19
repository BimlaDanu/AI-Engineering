"""Shared fixtures.

The fixtures build small in-memory frames with the same shape as the real
tables. Nothing here loads a model or reads the corpus, so the unit tests stay
fast enough to run on every edit.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_records() -> list[dict[str, object]]:
    """Four raw records shaped like the WikiNews JSONL."""
    return [
        {
            "pageid": 1,
            "lang": "en",
            "title": "Flood hits the capital",
            "categories": ["Disasters and accidents", "United Kingdom", "June 3, 2010"],
            "text": ["Heavy rain flooded the capital.", "Two bridges were closed."],
            "url": "https://en.wikinews.org/wiki/a",
            "date": "2010-06-03",
            "type": "title",
        },
        {
            "pageid": 1,
            "lang": "es",
            "title": "Inundación en la capital",
            "categories": ["Disasters and accidents", "United Kingdom"],
            "text": ["La lluvia inundó la capital."],
            "url": "https://es.wikinews.org/wiki/a",
            "date": "jueves 3 de junio de 2010",
            "type": "interlang link",
        },
        {
            "pageid": 2,
            "lang": "en",
            "title": "Parliament debates the budget",
            "categories": ["Politics and conflicts"],
            "text": ["Parliament debated the budget on Tuesday."],
            "url": "https://en.wikinews.org/wiki/b",
            "date": "2011-02-01",
            "type": "title",
        },
        {
            "pageid": 3,
            "lang": "en",
            "title": "Empty article",
            "categories": ["Sports"],
            "text": [],
            "url": "https://en.wikinews.org/wiki/c",
            "date": "2012-01-01",
            "type": "title",
        },
    ]


@pytest.fixture
def entities() -> pd.DataFrame:
    """A small mention table with the columns the NER stage writes."""
    return pd.DataFrame(
        [
            {
                "pageid": 1,
                "lang": "en",
                "entity": "Barack Obama",
                "entity_key": "barack obama",
                "label": "PERSON",
                "label_raw": "PERSON",
                "category": "Politics and conflicts",
                "year": 2008,
                "sentence": "Barack Obama spoke.",
                "title": "t",
            },
            {
                "pageid": 1,
                "lang": "en",
                "entity": "the United States",
                "entity_key": "united states",
                "label": "LOCATION",
                "label_raw": "GPE",
                "category": "Politics and conflicts",
                "year": 2008,
                "sentence": "In the United States.",
                "title": "t",
            },
            {
                "pageid": 1,
                "lang": "de",
                "entity": "Vereinigte Staaten",
                "entity_key": "vereinigte staaten",
                "label": "PERSON",
                "label_raw": "PER",
                "category": "Politics and conflicts",
                "year": 2008,
                "sentence": "In den Vereinigten Staaten.",
                "title": "t",
            },
            {
                "pageid": 2,
                "lang": "en",
                "entity": "barack obama",
                "entity_key": "barack obama",
                "label": "PERSON",
                "label_raw": "PERSON",
                "category": "Crime and law",
                "year": 2009,
                "sentence": "obama again.",
                "title": "t2",
            },
        ]
    )


@pytest.fixture
def summaries() -> pd.DataFrame:
    """A small summary table with the columns the summarize stage writes."""
    source = (
        "The council approved the budget on Monday. The budget covers transport and housing. "
        "Councillors said transport spending would rise by ten percent. Housing stays flat."
    )
    return pd.DataFrame(
        [
            {
                "pageid": 1,
                "lang": "en",
                "category": "Politics and conflicts",
                "method": "extractive",
                "summary": "The council approved the budget on Monday. Housing stays flat.",
                "text": source,
                "title": "Budget",
                "source_chars": len(source),
                "summary_chars": 61,
                "compression": 0.3,
            },
            {
                "pageid": 1,
                "lang": "en",
                "category": "Politics and conflicts",
                "method": "abstractive",
                "summary": "Councillors backed a budget raising transport spending by a tenth.",
                "text": source,
                "title": "Budget",
                "source_chars": len(source),
                "summary_chars": 66,
                "compression": 0.33,
            },
        ]
    )
