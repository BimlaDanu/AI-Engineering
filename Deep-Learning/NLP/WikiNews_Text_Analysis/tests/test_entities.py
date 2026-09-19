import pandas as pd

from src.analysis.entities import (
    aggregate_entities,
    display_surfaces,
    entity_timeline,
    geographic_focus,
    top_entities,
)


def test_aggregate_counts_articles_not_mentions(entities):
    aggregated = aggregate_entities(entities)
    obama = aggregated[
        (aggregated["entity_key"] == "barack obama")
        & (aggregated["category"] == "Politics and conflicts")
    ]
    assert int(obama["articles"].iloc[0]) == 1


def test_aggregate_picks_a_display_surface_for_each_entity(entities):
    aggregated = aggregate_entities(entities)
    assert aggregated["surface"].str.len().gt(0).all()


def test_aggregate_records_the_year_span(entities):
    aggregated = aggregate_entities(entities)
    assert {"first_year", "last_year"} <= set(aggregated.columns)


def test_top_entities_can_be_restricted_to_one_label(entities):
    aggregated = aggregate_entities(entities)
    persons = top_entities(aggregated, lang="en", label="PERSON", n=5)
    assert set(persons["label"]) == {"PERSON"}


def test_timeline_fills_missing_years_with_zero(entities):
    timeline = entity_timeline(entities, ["barack obama"], lang="en")
    assert list(timeline.index) == [2008, 2009]
    assert timeline.notna().all().all()


def test_timeline_is_empty_for_an_entity_that_was_never_mentioned(entities):
    assert entity_timeline(entities, ["nobody at all"], lang="en").empty


def test_geographic_focus_ranks_only_locations(entities):
    from src.analysis.entities import geographic_focus

    places = geographic_focus(entities, lang="en", n=5)
    assert "barack obama" not in set(places["entity_key"]), "people are not places"


def test_geographic_focus_reports_the_years_a_place_spans(entities):
    from src.analysis.entities import geographic_focus

    places = geographic_focus(entities, lang="en", n=5)
    assert {"first_year", "last_year", "articles", "place"} <= set(places.columns)


def test_display_surface_is_chosen_once_per_entity_not_per_category(entities):
    from src.analysis.entities import aggregate_entities

    aggregated = aggregate_entities(entities)
    obama = aggregated[aggregated["entity_key"] == "barack obama"]
    assert obama["surface"].nunique() == 1, "one entity must display under one name everywhere"


def test_display_surface_breaks_ties_towards_the_longer_form(entities):
    from src.analysis.entities import display_surfaces

    surfaces = display_surfaces(entities).set_index(["lang", "entity_key"])["surface"]
    assert surfaces[("en", "barack obama")] == "Barack Obama"


def test_display_surfaces_strips_the_article_from_the_printed_name():
    entities = pd.DataFrame(
        {
            "lang": ["en", "en", "en"],
            "entity_key": ["gaza strip", "gaza strip", "nasa"],
            "entity": ["the Gaza Strip", "the Gaza Strip", "NASA"],
        }
    )
    surfaces = display_surfaces(entities).set_index("entity_key")["surface"]
    assert surfaces["gaza strip"] == "Gaza Strip"
    assert surfaces["nasa"] == "NASA"


def test_rank_entities_sums_articles_across_categories():
    from src.analysis.entities import rank_entities

    aggregated = pd.DataFrame(
        {
            "lang": ["en", "en", "en"],
            "category": ["Crime and law", "Politics and conflicts", "Crime and law"],
            "entity_key": ["united states", "united states", "parliament"],
            "label": ["LOCATION", "LOCATION", "ORGANISATION"],
            "articles": [8, 40, 20],
            "mentions": [10, 60, 25],
            "first_year": [2008, 2006, 2010],
            "last_year": [2010, 2012, 2011],
            "surface": ["United States", "United States", "Parliament"],
        }
    )
    ranked = rank_entities(aggregated, lang="en", n=2)

    # 8 + 40 beats 20; ranking on a single category's row would invert this.
    assert list(ranked["entity_key"]) == ["united states", "parliament"]
    assert int(ranked.loc[0, "articles"]) == 48
    assert int(ranked.loc[0, "first_year"]) == 2006
    assert int(ranked.loc[0, "last_year"]) == 2012


def test_rank_entities_is_restricted_to_one_language():
    from src.analysis.entities import rank_entities

    aggregated = pd.DataFrame(
        {
            "lang": ["en", "de"],
            "category": ["Crime and law", "Crime and law"],
            "entity_key": ["parliament", "bundestag"],
            "label": ["ORGANISATION", "ORGANISATION"],
            "articles": [5, 99],
            "mentions": [5, 99],
            "first_year": [2010, 2010],
            "last_year": [2011, 2011],
            "surface": ["Parliament", "Bundestag"],
        }
    )
    assert list(rank_entities(aggregated, lang="en", n=5)["entity_key"]) == ["parliament"]


def test_aggregate_counts_one_article_once_when_labels_disagree():
    """One article, one entity, two labels: still one article."""
    entities = pd.DataFrame(
        {
            "pageid": [7, 7, 8],
            "lang": ["en", "en", "en"],
            "entity": ["Washington", "Washington", "Washington"],
            "entity_key": ["washington", "washington", "washington"],
            "label": ["LOCATION", "ORGANISATION", "LOCATION"],
            "category": ["Politics and conflicts"] * 3,
            "year": [2010, 2010, 2011],
        }
    )
    aggregated = aggregate_entities(entities)

    assert len(aggregated) == 1
    assert int(aggregated["articles"].iloc[0]) == 2
    assert int(aggregated["mentions"].iloc[0]) == 3
    assert aggregated["label"].iloc[0] == "LOCATION"


def test_geographic_focus_names_a_place_the_same_way_in_every_category():
    entities = pd.DataFrame(
        {
            "pageid": [1, 2, 3],
            "lang": ["en", "en", "en"],
            "entity": ["U.S", "US", "US"],
            "entity_key": ["united states"] * 3,
            "label": ["LOCATION"] * 3,
            "category": ["Crime and law", "Politics and conflicts", "Politics and conflicts"],
            "year": [2010, 2011, 2012],
        }
    )
    places = geographic_focus(entities, lang="en", n=5)

    assert places["place"].nunique() == 1, "one place, one spelling"
