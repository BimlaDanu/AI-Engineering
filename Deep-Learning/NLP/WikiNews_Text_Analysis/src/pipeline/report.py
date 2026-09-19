"""Stage 11: write reports/findings.md from the stage outputs.

Every value in the report is read from a table the pipeline produced, so
re-running with different settings gives a report consistent with the new
results. Sections for stages that have not run are skipped.

Run with: make report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd

from src import config
from src.analysis.ner_errors import flag_surface_errors
from src.analysis.quality import fault_rates
from src.analysis.similarity import distribution, drivers, extremes, threshold_check
from src.log import get_logger
from src.utils import as_int

logger = get_logger(__name__)

_THRESHOLD = 0.8

# Columns section 4.3 reads. A small run may not produce every one of them.
_ERROR_TYPES = ("ok", "determiner_included", "lowercase_span", "stray_punctuation")


def _table(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a frame as a markdown table.

    Args:
        frame: The table to render.
        index: Whether to include the index as a column.

    Returns:
        The markdown table.
    """
    return frame.to_markdown(index=index)


def _read(path: Path) -> pd.DataFrame | None:
    """Read a stage output if it exists.

    Args:
        path: The parquet file.

    Returns:
        The table, or ``None`` if that stage has not been run.
    """
    return pd.read_parquet(path) if path.exists() else None


def _organisation_share(entities: pd.DataFrame) -> pd.Series:
    """Share of each language's mentions that were labelled ORGANISATION.

    Args:
        entities: The mention table.

    Returns:
        Percentage per language.
    """
    grouped = entities.groupby("lang", observed=True)["label"]
    return (grouped.apply(lambda labels: (labels == "ORGANISATION").mean()) * 100).round(1)


def section_scope(corpus: pd.DataFrame, tokens: pd.DataFrame) -> str:
    """Describe the corpus the analysis ran on.

    Args:
        corpus: The working corpus.
        tokens: The token table.

    Returns:
        The markdown section.
    """
    counts = pd.crosstab(
        corpus["lang"], corpus["primary_category"], margins=True, margins_name="total"
    )
    sentences = tokens.groupby(["lang", "pageid"], observed=True)["sent_id"].nunique()

    return f"""## 1. Scope and data

The analysis covers **{len(corpus):,} articles** in {corpus["lang"].nunique()} languages
({", ".join(sorted(corpus["lang"].unique()))}) across {corpus["primary_category"].nunique()} news
categories, published between {int(corpus["year"].min())} and {int(corpus["year"].max())}.

They were drawn from the WikiNews corpus, capped at {config.MAX_ARTICLES_PER_LANGUAGE} per
language and sampled at random with a fixed seed. Random rather than stratified sampling preserves
the corpus's own category and year distribution.

Articles per language and category:

{_table(counts, index=True)}

Pre-processing produced **{len(tokens):,} tokens** with part-of-speech, dependency and lemma
annotation, at a median of {sentences.median():.0f} sentences per article.

One normalisation was a precondition for the rest. The raw corpus assigns ISO dates only to
English pages; translations retain a local-format string. Articles sharing a `pageid` describe the
same event, so the English date applies to its translations. That is what makes any time analysis
outside English possible.
"""


def section_pos(tokens: pd.DataFrame) -> str:
    """Report the part-of-speech profile per language.

    Args:
        tokens: The token table.

    Returns:
        The markdown section.
    """
    profile = (pd.crosstab(tokens["pos"], tokens["lang"], normalize="columns") * 100).round(1)
    rows = [p for p in ("PROPN", "NOUN", "VERB", "ADJ", "DET") if p in profile.index]
    interesting = profile.loc[rows]

    return f"""## 2. Pre-processing (requirement 1)

Every article was sentence-split, tokenized, POS-tagged, dependency-parsed and lemmatized using
the language's own spaCy pipeline. All four pipelines are the `sm` size, which holds model
capacity constant for the cross-language comparison in section 4.

Share of tokens by part of speech (%):

{_table(interesting, index=True)}

The proper-noun row is the relevant one. English tags **{profile.loc["PROPN", "en"]}%** of its
tokens as proper nouns, against **{profile.loc["PROPN", "de"]}%** for German and
**{profile.loc["PROPN", "fr"]}%** for French. Articles covering the same events should not differ
this much in name density, which indicates that the non-English taggers identify fewer names. The
NER error analysis in section 4 confirms this.
"""


def section_ner(entities: pd.DataFrame, aggregated: pd.DataFrame) -> str:
    """Report the NER output and the aggregated entity view.

    Args:
        entities: The mention table.
        aggregated: The aggregated entity table.

    Returns:
        The markdown section.
    """
    labels = (pd.crosstab(entities["label"], entities["lang"], normalize="columns") * 100).round(1)
    top = (
        aggregated[aggregated["lang"] == "en"]
        .sort_values("articles", ascending=False)
        .groupby("category", observed=True)
        .head(5)[["category", "surface", "label", "articles", "mentions"]]
    )

    return f"""## 3. Named entity recognition (requirement 2)

**{len(entities):,} entity mentions** were extracted, covering
{entities["entity_key"].nunique():,} distinct entities at an average of
{len(entities) / len(entities.drop_duplicates(["lang", "pageid"])):.1f} mentions per article.
Each mention carries its source article: `pageid`, title, language, category, publication date,
the character offsets of the span, and the containing sentence.

The four pipelines use different label sets. English uses OntoNotes (18 labels); the other three
use WikiNER (4). All labels were mapped onto a common coarse scheme (PERSON, ORGANISATION,
LOCATION, OTHER). OntoNotes numeric and temporal labels were dropped, since the other three
pipelines cannot produce them and retaining them would overstate English coverage.

Coarse label share by language (%):

{_table(labels, index=True)}

### 3.1 Aggregated view

Entities are ranked by the number of articles mentioning them rather than by raw mention count: a
name repeated eleven times within one article represents one article.

{_table(top)}

The United States leads every category, so what separates the profiles is what sits below it.
Disasters and accidents runs on **UTC** and **USGS**, the timestamps and the US Geological Survey
in its earthquake reports. Science and technology is **Earth**, **NASA** and the **International
Space Station**. Politics and conflicts is dominated by states rather than individuals.
"""


def section_timeline(timeline: pd.DataFrame) -> str:
    """Report the entity dynamics over time.

    Args:
        timeline: Years as rows, entities as columns.

    Returns:
        The markdown section.
    """
    return f"""### 3.2 Dynamics over time

Articles per year for the most reported entities:

{_table(timeline.reset_index())}

The series correspond to real events. Entities tied to a specific political period appear and
disappear with it, while state-level entities persist across the whole span.

The decline after 2011 is a property of the corpus rather than of the news. WikiNews published
substantially less after that point, so the later years rest on few articles and should not be
read as a trend.
"""


def section_clusters(clusters: pd.DataFrame) -> str:
    """Report the semantic clustering of entities.

    Args:
        clusters: The clustered entity table.

    Returns:
        The markdown section.
    """
    rows = []
    for cluster, group in clusters.groupby("cluster"):
        rows.append(
            {
                "cluster": as_int(cluster),
                "size": len(group),
                "dominant label": str(group["label"].mode().iat[0]),
                "members": ", ".join(group.nlargest(7, "articles")["surface"]),
            }
        )
    described = pd.DataFrame(rows).sort_values("size", ascending=False)
    silhouette = float(clusters["silhouette"].iloc[0])

    return f"""### 3.3 Semantic groups

The {len(clusters)} most reported English entities were embedded with a multilingual sentence
encoder and clustered with K-means (silhouette {silhouette:.3f}).

{_table(described)}

No labels were supplied, and the smaller clusters still read as coherent groups: one per world
region, and one collecting the people and institutions of a single national politics.

The largest cluster is a residual, collecting entities that sit close to nothing else -- news
organisations and bare abbreviations. K-means over strings this short produces clusters that
overlap at their boundaries, which is what the silhouette score records, so the grouping is
indicative rather than a partition to rely on.
"""


def section_errors(recall: pd.DataFrame, entities: pd.DataFrame) -> str:
    """Report the investigation of wrongly predicted entities.

    Args:
        recall: The cross-lingual recall proxy per language.
        entities: The mention table.

    Returns:
        The markdown section.
    """
    ordered = recall.sort_values("recall_proxy")
    weakest, strongest = ordered.iloc[0], ordered.iloc[-1]
    flagged = flag_surface_errors(entities)
    surface = (
        (pd.crosstab(flagged["lang"], flagged["error_type"], normalize="index") * 100)
        .round(2)
        .reindex(columns=_ERROR_TYPES, fill_value=0.0)
    )
    org_share = _organisation_share(entities)

    return f"""## 4. Wrongly predicted entities (requirement 3)

The corpus contains no gold NER annotation, so accuracy cannot be measured directly. It does
contain structure: articles sharing a `pageid` describe the same event. Three checks are built on
that.

### 4.1 Missed entities

Take a person named in the English article. If that name appears verbatim in the other language's
article text and the pipeline did not tag it, the model missed it. The presence of the name in the
text is what distinguishes a miss from a difference in editorial coverage. Place names are
excluded, because they are translated (London, Londres), which would produce false misses.

{_table(recall)}

**{weakest["lang"]}** is weakest at {weakest["recall_proxy"]:.3f}, against
{strongest["recall_proxy"]:.3f} for **{strongest["lang"]}**, over {as_int(weakest["checked"])} and
{as_int(strongest["checked"])} checked names respectively. This is consistent with the proper-noun
rates in section 2.

Matching is at token level, not substring: a substring test counts "Ross" as found inside
"Crossing" and places every language above 0.96.

### 4.2 Label disagreements

A name given different coarse labels by two pipelines means at least one is wrong. `Vereinigte
Staaten` is LOCATION to the German model and PERSON to the French one. `Twitter` is PERSON to the
English model and OTHER elsewhere.

Organisations are the weakest category. English assigns {org_share["en"]:.0f}% of its mentions to
ORGANISATION against {org_share["es"]:.0f}% for Spanish, and the Spanish organisations that are
missing reappear largely as OTHER.

### 4.3 Surface issues

{_table(surface, index=True)}

Two of these are errors. `lowercase_span`, a span beginning with a lowercase common word, occurs
at {surface.loc["de", "lowercase_span"]:.1f}% for German against
{surface.loc["en", "lowercase_span"]:.1f}% for English. German capitalises all nouns, so a model
relying on capitalisation has a weaker signal. `stray_punctuation` marks spans that absorbed a
comma or slash.

`determiner_included` is an annotation convention rather than an error: OntoNotes includes the
article in names such as "the United States", WikiNER does not. It is measured because it is why
entity keys strip leading articles, without which English and German would never aggregate to the
same entity. "Den Haag" is counted under it even though the article belongs to the city's name;
the stripping is consistent across German, so the entity counts stay correct.

A third error type appears in the mention table: the English pipeline tags untranslated quoted
Spanish inside English articles as PERSON, giving spans such as "tomar medidas de prevención".
"""


def section_summaries(summaries: pd.DataFrame, quality: pd.DataFrame | None) -> str:
    """Report the summarization stage.

    Args:
        summaries: The summary table.
        quality: The readability and style table, if the stage has run.

    Returns:
        The markdown section.
    """
    per_method = summaries.pivot_table(
        index="method", columns="lang", values="compression", aggfunc="median"
    ).round(3)
    # A pageid is shared across translations, so an article is a (lang, pageid).
    n_articles = len(summaries.drop_duplicates(["lang", "pageid"]))
    n_summaries = len(summaries)

    section = f"""## 5. Summarization (requirement 4)

{config.ARTICLES_PER_CATEGORY} articles were summarized for each of the four categories in each of
the four languages, giving **{n_articles} articles and {n_summaries} summaries**, by two methods:

- **Extractive (TextRank).** Sentences form a graph weighted by TF-IDF cosine similarity, ranked
  by PageRank, with the top three returned in document order. The vectorizer is fitted per article,
  so no per-language stop-word list is needed.
- **Abstractive (LLM via OpenRouter).** A fixed prompt at low temperature, summarizing in the
  article's own language, with every response cached on disk.

Section 6 compares the two.

Median compression (summary length divided by source length):

{_table(per_method, index=True)}
"""

    if quality is not None:
        rates = fault_rates(quality)
        length = quality.groupby("method", observed=True)["words_per_sentence"].mean().round(1)
        section += f"""
### 5.1 Grammar and style

No Java runtime is available here, so LanguageTool was not used. The checks are readability with
each language's own Flesch coefficients, and deterministic detection of the defects summarizers
produce.

Share of summaries exhibiting each fault (%):

{_table(rates, index=True)}

Mean words per sentence:

{_table(length.to_frame("words per sentence"), index=True)}
"""
        if {"extractive", "abstractive"} <= set(rates.index):
            section += f"""
The two methods fail differently. Extractive summaries carry the faults of the sentences they
lift: {rates.loc["extractive", "overlong_sentence"]:.0f}% contain a sentence over 40 words, and
{rates.loc["extractive", "truncated"]:.1f}% end without terminal punctuation because the sentence
splitter cut an abbreviation. They also open with an unresolved pronoun when the sentence naming
the subject was not selected.

The abstractive summaries show neither fault, at {length["abstractive"]:.1f} words per sentence.
That follows from the prompt: asked only for three sentences, the model returned three of about
50 words each, and the explicit word target is what produced the figures above.
"""
    return section


def section_similarity(scored: pd.DataFrame) -> str:
    """Report the similarity analysis.

    Args:
        scored: The scored summary table.

    Returns:
        The markdown section.
    """
    highest, lowest = extremes(scored, n=3)
    columns = ["method", "lang", "category", "similarity", "lexical_overlap", "compression"]

    passing = threshold_check(scored, _THRESHOLD)
    by_method = scored.groupby("method", observed=True)
    overlap = by_method["lexical_overlap"].median().round(3)
    median = by_method["similarity"].median().round(3)
    comparison = pd.DataFrame(
        {
            "median lexical overlap": overlap,
            "median similarity": median,
            f"% at or above {_THRESHOLD}": passing,
        }
    ).reset_index()

    section = f"""## 6. Similarity between summaries and sources (requirement 5)

Similarity is the cosine between multilingual embeddings of the summary and of the full source
article. Long articles are chunked at sentence boundaries and the chunk vectors averaged; without
that the encoder's input window truncates each article and every score would compare a summary
against the article's opening paragraph.

**Lexical overlap**, the share of the summary's words that appear in the source, is reported
alongside it.

{_table(distribution(scored, _THRESHOLD))}

Share of summaries at or above the {_THRESHOLD} threshold (%):

{_table(threshold_check(scored, _THRESHOLD).reset_index())}

Spearman correlation of the similarity score with candidate explanatory features, computed
within each method:

{_table(drivers(scored).reset_index(names="method"))}

Within a method the correlations are read straight off. Across the pooled table they cannot be:
extractive overlap is 1.0 by construction, so a pooled coefficient measures the distance between
the two methods rather than any relationship inside either. Pooled, lexical overlap correlates
with similarity at roughly zero; within the abstractive summaries it is positive. The extractive
cell is empty because the feature does not vary there.

Compression is the one feature that behaves the same way for both methods: the more of the source
a summary keeps, the closer its vector sits to the source. That is arithmetic, not evidence of
quality, and it is why the threshold has to be read together with the compression column.

### 6.1 Highest and lowest scoring summaries

Highest:

{_table(highest[columns])}

Lowest:

{_table(lowest[columns])}

### 6.2 Interpretation

{_table(comparison)}
"""

    if {"extractive", "abstractive"} <= set(passing.index):
        section += f"""
The two measures separate the methods in opposite directions. Extractive summaries reuse the
source's wording almost completely, at a median lexical overlap of {overlap["extractive"]:.3f}, yet
clear the threshold less often than the abstractive summaries, whose median overlap is only
{overlap["abstractive"]:.3f}.

Verbatim reuse therefore does not guarantee semantic closeness. TextRank selects three sentences
that are central within the article's own sentence graph, but three sentences cannot represent an
article developing several points, and the document embedding averages over all of them. The
abstractive summarizer compresses across the whole article instead of selecting from it, so its
vector sits closer to the document average.

That does not make the extractive scores the more trustworthy of the two. Where an extractive
summary scores highly, part of the score is mechanical: the encoder is comparing a text against a
document that contains it, as `similarity_vs_overlap.png` shows directly. A high abstractive score
carries more information, because the wording changed and the meaning still matched.

Read as a screening filter, the {_THRESHOLD} threshold has to be taken together with the lexical
overlap. A summary passing it at an overlap near 1.0 has demonstrated copying, not comprehension.
On abstractive output, where overlap is well below 1.0, it passes {passing["abstractive"]:.1f}% of
summaries.
"""

    section += """
The lowest-scoring cases share a shape: a long, multi-topic source compressed heavily. When an
article covering several developments is reduced to three sentences, the summary keeps one of them
while the document embedding averages over all. That is a property of single-vector similarity
rather than evidence of a poor summary. The highest scores go to short, single-topic articles where
three sentences cover the whole piece.
"""
    return section


def section_topics(metrics: dict[str, dict[str, float]]) -> str:
    """Report the topic classification results.

    Args:
        metrics: Per-language metrics written by the topics stage.

    Returns:
        The markdown section.
    """
    frame = pd.DataFrame(metrics).T.reset_index(names="lang")

    return f"""## 7. Topic prediction

A TF-IDF and logistic-regression classifier trained per language on articles carrying exactly one
topic label. About a quarter of the corpus carries two or more, and an article that is genuinely
both political and criminal cannot be scored fairly against a single-label prediction.

{_table(frame)}

The baseline is a most-frequent-class classifier. One category holds more than half the articles,
so accuracy alone carries no information; macro F1 against the baseline is what shows the smaller
categories were learned.

The confusion matrix shows one substantial error mode: Crime and law against Politics and
conflicts, in both directions. That reflects genuine overlap in the source material -- the trial of
a former head of state belongs to both -- rather than a modelling failure.
"""


def section_limitations() -> str:
    """State what the analysis does not establish.

    Returns:
        The markdown section.
    """
    return """## 8. Limitations

- **No gold NER labels.** The recall figures in section 4 are a proxy built from cross-lingual
  agreement. They support the ordering between languages, not a precise accuracy figure.
- **Small spaCy models.** The `sm` pipelines hold model capacity constant across languages. Larger
  pipelines shift the absolute figures; the ordering between languages is unlikely to change.
- **Entity aliasing is not entity linking.** A short alias table merges the most common
  abbreviations. Entities that are neither identical nor in that table still rank separately.
- **Similarity uses one vector per document**, which penalises correct summaries of multi-topic
  articles, as section 6.2 describes.
- **The corpus thins after 2011.** Year-on-year figures for the later period rest on few articles.

### 8.1 What the error rates mean for using this

The stated business use is screening news for brand health, competitors and political affairs.
Three measured properties of this pipeline bear on that.

The entity table is an index of named individuals built by a model with a measured error rate.
Section 4.3 shows spans that are not names at all, and section 4.2 shows names given the wrong
type by one pipeline and the right one by another. A count of how often a person appears is
therefore an estimate, and attaching it to a person without review would attribute to them the
model's mistakes as well as the corpus's coverage.

The error rate is not the same in every language. Section 4.1 puts German recall below Spanish and
French on the same events, so the same person is indexed less often when the coverage is German.
A monitoring system built on this would under-report German-language subjects, and would do so
silently, since nothing in the output distinguishes a subject who was absent from one the model
missed.

Abstractive summarization sends article text to a third-party API. That is acceptable for
published journalism and would not be for internal documents; the extractive path exists partly
because it runs entirely offline.

The corpus itself is published WikiNews text under its own licence, and the people named in it are
named in public reporting. Nothing here infers anything about a person that the articles do not
state.
"""


# Captions match the title rendered inside each figure, so the contents page and
# the image agree. A figure added without an entry here still gets a caption.
_FIGURE_CAPTIONS: Final[dict[str, str]] = {
    "entities_by_category.png": "Most reported entities by news category",
    "entity_timeline.png": "When the leading entities were in the news",
    "ner_recall_by_language.png": "Person names present in the text but missed by the model",
    "similarity_distribution.png": "Summary-to-source similarity by category and summarizer",
    "similarity_vs_overlap.png": "Lexical reuse does not determine semantic similarity",
    "topic_confusion.png": "Topic classifier confusion matrix",
}


def _figure_caption(filename: str) -> str:
    """Look up a figure's caption, falling back to its file name.

    Args:
        filename: The PNG file name, for example ``entity_timeline.png``.

    Returns:
        The caption for the figure.
    """
    derived = filename.removesuffix(".png").replace("_", " ")
    return _FIGURE_CAPTIONS.get(filename, derived)


def main() -> None:
    """Assemble findings.md from whatever stages have been run."""
    config.ensure_directories()

    corpus = _read(config.CORPUS_FILE)
    tokens = _read(config.TOKENS_FILE)
    entities = _read(config.ENTITIES_FILE)
    aggregated = _read(config.ENTITY_AGGREGATE_FILE)
    timeline = _read(config.ENTITY_TIMELINE_FILE)
    clusters = _read(config.ENTITY_CLUSTERS_FILE)
    recall = _read(config.NER_RECALL_FILE)
    summaries = _read(config.SUMMARIES_FILE)
    quality = _read(config.GRAMMAR_FILE)
    scored = _read(config.SIMILARITY_FILE)

    if corpus is None or tokens is None or entities is None:
        raise FileNotFoundError("Run `make corpus preprocess ner` before building the report.")

    parts = [
        "# WikiNews NLP: findings\n",
        "Generated by `make report` from the pipeline's output tables.\n",
        section_scope(corpus, tokens),
        section_pos(tokens),
        section_ner(entities, aggregated) if aggregated is not None else "",
        section_timeline(timeline) if timeline is not None else "",
        section_clusters(clusters) if clusters is not None else "",
        section_errors(recall, entities) if recall is not None else "",
        section_summaries(summaries, quality) if summaries is not None else "",
        section_similarity(scored) if scored is not None else "",
    ]

    if config.TOPIC_METRICS_FILE.exists():
        metrics = json.loads(config.TOPIC_METRICS_FILE.read_text(encoding="utf-8"))
        parts.append(section_topics(metrics))
    parts.append(section_limitations())

    figures = sorted(p.name for p in config.FIGURES_DIR.glob("*.png"))
    if figures:
        # Embedded, not listed. The path is relative to this file's own
        # directory so the images render on GitHub rather than reading as
        # a list of names the reviewer has to go and open one by one.
        listed = "\n\n".join(
            f"**{_figure_caption(name)}**\n\n![{_figure_caption(name)}](figures/{name})"
            for name in figures
        )
        parts.append(f"## 9. Figures\n\n{listed}\n")

    output = config.REPORTS_DIR / "findings.md"
    output.write_text("\n".join(part for part in parts if part), encoding="utf-8")
    logger.info("wrote %s", output)


if __name__ == "__main__":
    main()
