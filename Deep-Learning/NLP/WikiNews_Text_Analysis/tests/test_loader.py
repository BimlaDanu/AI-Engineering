import pandas as pd

from src.data.loader import extract_topics, join_paragraphs, load_articles


def test_extract_topics_keeps_only_real_labels():
    categories = ["Disasters and accidents", "United Kingdom", "June 3, 2010", "Sports"]
    assert extract_topics(categories) == ["Disasters and accidents", "Sports"]


def test_extract_topics_returns_empty_when_no_label_present():
    assert extract_topics(["United Kingdom", "June 3, 2010"]) == []


def test_join_paragraphs_separates_with_blank_lines_and_drops_empties():
    assert join_paragraphs(["One.", "  ", "Two."]) == "One.\n\nTwo."


def test_join_paragraphs_on_empty_input():
    assert join_paragraphs([]) == ""


def test_load_articles_drops_records_without_text(tmp_path, raw_records):
    frame = _write_and_load(tmp_path, raw_records)
    assert 3 not in set(frame["pageid"]), "the article with empty text should be dropped"
    assert len(frame) == 3


def test_load_articles_backfills_date_onto_translations(tmp_path, raw_records):
    frame = _write_and_load(tmp_path, raw_records)
    spanish = frame[frame["lang"] == "es"].iloc[0]
    assert spanish["date"] == pd.Timestamp("2010-06-03"), (
        "the English page's ISO date should carry over"
    )
    assert spanish["year"] == 2010


def _write_and_load(tmp_path, records) -> pd.DataFrame:
    import json

    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return load_articles(path)
