"""Normalise entity labels and entity text across the four pipelines.

The pipelines do not share a label set. ``en_core_web_sm`` is trained on
OntoNotes and emits 18 labels; the Spanish, French and German pipelines are
trained on WikiNER and emit four. Everything is mapped onto one coarse scheme
before any cross-language comparison, otherwise English looks richer purely
because its tag set is finer.

OntoNotes' numeric and temporal labels are dropped, since the other three
pipelines cannot produce them. ``GPE`` and ``LOC`` both become ``LOCATION``.
"""

from __future__ import annotations

import re
from typing import Final

PERSON: Final[str] = "PERSON"
ORGANISATION: Final[str] = "ORGANISATION"
LOCATION: Final[str] = "LOCATION"
OTHER: Final[str] = "OTHER"

COARSE_LABELS: Final[tuple[str, ...]] = (PERSON, ORGANISATION, LOCATION, OTHER)

# Label as produced by a pipeline -> the shared coarse label.
_COARSE_MAP: Final[dict[str, str]] = {
    # OntoNotes (English)
    "PERSON": PERSON,
    "ORG": ORGANISATION,
    "GPE": LOCATION,
    "LOC": LOCATION,
    "FAC": LOCATION,
    "NORP": OTHER,
    "EVENT": OTHER,
    "PRODUCT": OTHER,
    "WORK_OF_ART": OTHER,
    "LAW": OTHER,
    "LANGUAGE": OTHER,
    # WikiNER (Spanish, French, German)
    "PER": PERSON,
    "MISC": OTHER,
}

# Quantities and dates. English-only, so excluded.
_NUMERIC_LABELS: Final[frozenset[str]] = frozenset(
    {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
)

# Leading articles per language, stripped so "the United States" and "United
# States" aggregate as one entity. The lists are separate because an article in
# one language is a name in another: "Los Angeles", "La Paz", "UN". Each
# language sees only its own articles, plus the English "the", which appears in
# every language when an English name is quoted ("The Guardian").
_ARTICLES_BY_LANGUAGE: Final[dict[str, frozenset[str]]] = {
    "en": frozenset({"the", "a", "an"}),
    "es": frozenset({"the", "el", "la", "los", "las", "un", "una", "unos", "unas"}),
    "fr": frozenset({"the", "le", "la", "les", "l'", "un", "une", "du", "des"}),
    "de": frozenset({"the", "der", "die", "das", "den", "dem", "des", "ein", "eine"}),
}

# Abbreviations mapped to their expanded form, so "U.S", "US" and "the United
# States" do not rank as three entities. A hand-checked list, not entity linking.
_ALIASES: Final[dict[str, str]] = {
    "u.s": "united states",
    "us": "united states",
    "usa": "united states",
    "u.s.a": "united states",
    "u.k": "united kingdom",
    "uk": "united kingdom",
    "un": "united nations",
    "u.n": "united nations",
    "eu": "european union",
    "usgs": "united states geological survey",
    "nasa": "national aeronautics and space administration",
    "iss": "international space station",
    "fbi": "federal bureau of investigation",
    "who": "world health organization",
    "nato": "north atlantic treaty organization",
}

_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)
_EDGE_NOISE = re.compile(r"^[\s\"'“”‘’(\[]+|[\s\"'“”‘’)\].,;:!?]+$")
_WHITESPACE = re.compile(r"\s+")


def to_coarse(label: str) -> str | None:
    """Map a pipeline's entity label onto the shared coarse scheme.

    Args:
        label: The label spaCy produced, for example ``GPE`` or ``PER``.

    Returns:
        One of ``COARSE_LABELS``, or ``None`` if the entity should be dropped
        because no cross-language comparison is possible for it.
    """
    if label in _NUMERIC_LABELS:
        return None
    return _COARSE_MAP.get(label, OTHER)


def clean_surface(text: str) -> str:
    """Tidy the raw entity string without changing its case.

    Removes surrounding quotes, brackets and trailing punctuation left by the
    tokenizer, and collapses internal whitespace.

    Args:
        text: The entity text exactly as spaCy found it.

    Returns:
        The cleaned string, which may be empty if nothing survived.
    """
    cleaned = _WHITESPACE.sub(" ", text).strip()
    cleaned = _EDGE_NOISE.sub("", cleaned)
    return cleaned.strip()


def leading_articles(lang: str) -> frozenset[str]:
    """Return the articles that may open an entity span in one language.

    Args:
        lang: Two-letter language code.

    Returns:
        That language's articles. An unknown language gets the English set.
    """
    return _ARTICLES_BY_LANGUAGE.get(lang, _ARTICLES_BY_LANGUAGE["en"])


def entity_key(text: str, lang: str) -> str:
    """Build the key that groups different mentions of the same entity.

    Lowercases, drops a leading article belonging to ``lang`` and an English
    possessive, then applies the alias table. A single-word name is never
    stripped, so "Die" stays "die".

    Args:
        text: The entity text, cleaned or raw.
        lang: Language of the article the mention came from.

    Returns:
        A normalised key, empty if nothing meaningful remains.
    """
    cleaned = _POSSESSIVE.sub("", clean_surface(text)).casefold()
    words = cleaned.split()
    if len(words) > 1 and words[0] in leading_articles(lang):
        words = words[1:]
    key = " ".join(words).strip(" .")
    return _ALIASES.get(key, key)


def display_form(text: str, lang: str) -> str:
    """Tidy an entity spelling for display without changing which entity it is.

    Strips the same leading article and possessive as the key, keeping the
    original capitalisation, so "the Gaza Strip" prints as "Gaza Strip".

    Args:
        text: The entity spelling to display.
        lang: Language of the article the mention came from.

    Returns:
        The display form, or the input unchanged if nothing could be removed.
    """
    cleaned = _POSSESSIVE.sub("", clean_surface(text))
    words = cleaned.split()
    if len(words) > 1 and words[0].casefold() in leading_articles(lang):
        words = words[1:]
    return " ".join(words) or text


def is_usable(text: str, key: str) -> bool:
    """Decide whether an entity mention is worth keeping.

    Drops empty strings, single characters and spans with no letter in them.

    Args:
        text: The cleaned entity text.
        key: The normalised key for that text.

    Returns:
        ``True`` if the mention should be kept.
    """
    return bool(key) and len(text) > 1 and any(character.isalpha() for character in text)
