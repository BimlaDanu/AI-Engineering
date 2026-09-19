"""Find wrongly predicted entities without a labelled test set.

The corpus ships no gold NER annotation, but articles sharing a ``pageid``
describe the same event in different languages. Three checks build on that:

1. **Missed entities.** A person named in the English article, appearing
   verbatim in the other language's text, that the other pipeline did not tag.
   Person names are largely invariant across these four languages; place names
   are not (London / Londres), so places are excluded.
2. **Label disagreement.** The same name tagged PERSON in one language and
   LOCATION in another. One of the two models is wrong; which one is not
   decided here.
3. **Surface heuristics.** Boundary and spurious-span errors visible from the
   span itself.
"""

from __future__ import annotations

import re

import pandas as pd

from src.log import get_logger
from src.nlp.labels import LOCATION, PERSON, leading_articles
from src.utils import as_int

logger = get_logger(__name__)

# A surname shorter than this matches too much text by accident ("Li", "Ng").
_MIN_NAME_LENGTH = 4


def _mentions_name(text: str, name: str) -> bool:
    """Check whether an article's text contains a person's name.

    Matches the full name or the surname alone, since translations often drop
    the given name after first use. Word boundaries stop "Ross" matching inside
    "Crossing".

    Args:
        text: The article text to search.
        name: The person name as found in the English article.

    Returns:
        ``True`` if the name or its surname appears in the text.
    """
    haystack = text.casefold()
    if name.casefold() in haystack:
        return True

    surname = name.split()[-1] if name.split() else ""
    if len(surname) < _MIN_NAME_LENGTH:
        return False
    return re.search(rf"\b{re.escape(surname.casefold())}\b", haystack) is not None


def missed_entities(
    entities: pd.DataFrame,
    corpus: pd.DataFrame,
    target_lang: str,
) -> pd.DataFrame:
    """Find people named in English whose translation failed to tag them.

    Args:
        entities: The mention table for every language.
        corpus: The working corpus, supplying the target language's raw text.
        target_lang: Language to evaluate against English.

    Returns:
        One row per checked person name, with ``found`` saying whether the
        target pipeline tagged it. Only names present in the target text are
        included, so every ``found = False`` row is a real miss.
    """
    english = entities[(entities["lang"] == "en") & (entities["label"] == PERSON)]
    target = entities[entities["lang"] == target_lang]

    texts = corpus[corpus["lang"] == target_lang].set_index("pageid")["text"].to_dict()
    tagged: dict[int, set[str]] = {}
    if not target.empty:
        for page, keys in target.groupby("pageid")["entity_key"].agg(set).items():
            tagged[as_int(page)] = set(keys)

    rows: list[dict[str, object]] = []
    for page, group in english.groupby("pageid", observed=True):
        pageid = as_int(page)
        text = texts.get(pageid)
        if text is None:
            continue

        for name in group["entity"].unique():
            if len(name) < _MIN_NAME_LENGTH or not _mentions_name(text, str(name)):
                continue
            name_key = str(name).casefold()
            surname = name_key.split()[-1]
            # Token-level match. A substring test would count "Ross" as found
            # inside "Crossing" and inflate the recall figure.
            found = any(
                key == name_key or surname in key.split() for key in tagged.get(pageid, set())
            )
            rows.append({"pageid": pageid, "lang": target_lang, "entity": name, "found": found})

    return pd.DataFrame(rows)


def recall_by_language(missed: pd.DataFrame) -> pd.DataFrame:
    """Summarise the missed-entity check into a recall proxy per language.

    Args:
        missed: Concatenated output of ``missed_entities``.

    Returns:
        Checked count, found count and the resulting recall proxy per language.
    """
    summary = missed.groupby("lang").agg(checked=("found", "size"), found=("found", "sum"))
    summary["recall_proxy"] = (summary["found"] / summary["checked"]).round(3)
    return summary.reset_index()


def label_disagreements(entities: pd.DataFrame) -> pd.DataFrame:
    """Find names given different coarse labels in different languages.

    Args:
        entities: The mention table.

    Returns:
        One row per entity key that at least two languages labelled
        differently, with the labels each language chose, sorted by how many
        articles the entity touches.
    """
    per_language = (
        entities.groupby(["entity_key", "lang"], observed=True)["label"]
        .agg(lambda values: values.mode().iat[0])
        .unstack()
    )
    disagreeing = per_language[per_language.nunique(axis=1, dropna=True) > 1].copy()

    # An article is a (lang, pageid): translations of one event share a pageid,
    # and this table spans every language, so counting pageids alone would
    # merge the four articles that disagree into one.
    weight = (
        entities.drop_duplicates(["lang", "pageid", "entity_key"])
        .groupby("entity_key", observed=True)
        .size()
    )
    disagreeing["articles"] = weight.reindex(disagreeing.index)
    return disagreeing.sort_values("articles", ascending=False).reset_index()


def flag_surface_errors(entities: pd.DataFrame) -> pd.DataFrame:
    """Label each mention with the surface issue it shows, if any.

    Three narrow rules:

    * ``determiner_included`` -- a multi-word span opening with an article of
      its own language. A one-word span is the name itself, so it is never
      flagged.
    * ``stray_punctuation`` -- a span that absorbed a comma, colon or slash.
    * ``lowercase_span`` -- a span with no capital letter anywhere. Requiring
      the whole span keeps "al-Qaeda", "eBay" and "bin Laden" out of the count.

    The last two are errors. ``determiner_included`` is an annotation
    convention, so it is assigned first and a genuine error overrides it.

    Args:
        entities: The mention table, with ``entity`` and ``lang`` columns.

    Returns:
        The mention table with an ``error_type`` column.
    """
    flagged = entities.copy()
    words = flagged["entity"].str.split()
    languages = flagged["lang"].astype(str)

    is_determiner = pd.Series(
        [
            len(parts) > 1 and parts[0].casefold() in leading_articles(lang)
            for parts, lang in zip(words, languages, strict=True)
        ],
        index=flagged.index,
    )
    # Comparing against the lowercased string handles the accented alphabets
    # without listing their characters.
    is_lowercase = flagged["entity"].str.lower() == flagged["entity"]
    has_punctuation = flagged["entity"].str.contains(r"[,;:/\\|]", regex=True, na=False)

    flagged["error_type"] = "ok"
    flagged.loc[is_determiner, "error_type"] = "determiner_included"
    flagged.loc[has_punctuation, "error_type"] = "stray_punctuation"
    flagged.loc[is_lowercase, "error_type"] = "lowercase_span"
    return flagged


def location_naming_note(entities: pd.DataFrame) -> pd.DataFrame:
    """Count locations per language, which the recall check skips.

    Place names are translated (London / Londres / Londra), so the verbatim
    match used for people would report misses that are really translations.

    Args:
        entities: The mention table.

    Returns:
        Distinct locations and location mentions per language.
    """
    locations = entities[entities["label"] == LOCATION]
    return (
        locations.groupby("lang", observed=True)
        .agg(distinct_locations=("entity_key", "nunique"), mentions=("entity_key", "size"))
        .reset_index()
    )
