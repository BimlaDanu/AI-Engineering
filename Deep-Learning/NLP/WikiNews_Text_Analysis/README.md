# WikiNews NLP

![Python](https://img.shields.io/badge/python-3.11-3776ab)
![spaCy](https://img.shields.io/badge/spaCy-3.8%2B-09a3d5)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-3.0%2B-ffd21e)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-f7931e)
![pandas](https://img.shields.io/badge/pandas-2.2%2B-150458)
![matplotlib](https://img.shields.io/badge/matplotlib-3.9%2B-11557c)
![uv](https://img.shields.io/badge/deps-uv%20locked-de5fe9)
![Lint](https://img.shields.io/badge/lint-ruff-261230)
![Types](https://img.shields.io/badge/types-mypy-1f5082)
[![check](https://github.com/TuringCollegeSubmissions/bidanu-NLP.AI.2.5/actions/workflows/check.yml/badge.svg?branch=main)](https://github.com/TuringCollegeSubmissions/bidanu-NLP.AI.2.5/actions/workflows/check.yml)

Named entity recognition, text summarization and summary-to-source similarity over the
[WikiNews multilingual corpus](https://github.com/PrimerAI/WikiNews-multilingual), which holds
15,200 articles in 33 languages.

The analysis covers four news categories (Politics and conflicts, Crime and law, Disasters and
accidents, Science and technology) in four languages (English, Spanish, French, German). The
multilingual scope is what makes the cross-language NER error analysis possible.

`make all` runs eleven stages. Each reads the previous stage's parquet file and writes its own,
and the last one generates the written report from those tables, so no number in the prose is
typed by hand.

Specification: [`AE.NLP.2.5.md`](AE.NLP.2.5.md). The formal basis of every method is derived in
[`Mathematical_theoretical_analysis.md`](Mathematical_theoretical_analysis.md): the TextRank
stationary distribution, the geometry that caps a faithful summary's similarity score, and the
estimator behind the cross-lingual recall figures with its three biases. The sections below link
into it wherever a result depends on the derivation.

| Output | Contents |
|---|---|
| `reports/findings.md` | Written findings, generated from the pipeline's output tables |
| `reports/figures/` | Six figures |
| `notebooks/report.ipynb` | The same results as a notebook walk-through |
| [`Mathematical_theoretical_analysis.md`](Mathematical_theoretical_analysis.md) | The formal basis of each method, and what it predicts about the results |
| `data/processed/` | One parquet table per stage |
| `data/interim/llm_cache/` | Cached model responses, so the abstractive summaries reproduce without an API key |

---

## Contents

1. [Results](#1-results)
2. [Requirements coverage](#2-requirements-coverage)
3. [Technology stack](#3-technology-stack)
4. [Running project](#4-running-project)
5. [Pipeline stages](#5-pipeline-stages)
6. [Code organisation](#6-code-organisation)
7. [Data contracts](#7-data-contracts)
8. [Configuration](#8-configuration)
9. [Testing](#9-testing)
10. [Extending the analysis](#10-extending-the-analysis)
11. [Limitations and further work](#11-limitations-and-further-work)

---

## 1. Results

| Measure | Value |
|---|---|
| Corpus analysed | 3,200 articles, 4 categories × 4 languages, 2004 to 2022 |
| Pre-processing | 1,074,437 tokens with part of speech, dependency and lemma |
| Named entities | 75,855 mentions, 28,886 distinct entities |
| Summaries | 160 articles, each summarized twice, so 320 summaries |
| Similarity at or above 0.8 | abstractive 76.2%, extractive 66.9% |
| Cross-lingual NER recall proxy | Spanish 0.973, French 0.972, German 0.911 |
| Topic prediction macro F1 | 0.87 to 0.90, against a most-frequent baseline near 0.18 |

### 1.1 What the analysis found

- **Entity profiles separate by category below the top name.** The United States leads all four,
  from 168 articles in Politics and conflicts down to 23 in Science and technology, so the
  separation is in what follows it: UTC (39) and USGS (22) in Disasters and accidents, the
  timestamps and the US Geological Survey of its earthquake reports; Earth (17), NASA (16) and the
  International Space Station (14) in Science and technology; and in Politics and conflicts states
  rather than individuals — the UN (51), the UK (43), Iraq (42) and the EU (36).
- **Entity time series track real events.** Entities tied to a specific political period appear
  and disappear with it; state-level entities persist across the whole span. Article
  volume thins after 2011, which is a property of the corpus rather than of the news.
- **The leading entities cluster by world region.** K-means over multilingual embeddings of the
  120 most reported English entities recovers a Middle East group, an East Asia group, a Europe
  group, a UK and Commonwealth group and a US-politics group, with no labels supplied. The sixth
  group is not geographic: it collects the agencies and timestamp tokens that appear everywhere,
  led by UTC, Reuters, the UN and Wikinews itself.
- **NER accuracy is measurably lower outside English.** Measured on person names appearing
  verbatim in a translated article but left untagged, German recall is 0.911 against 0.973 for
  Spanish and 0.972 for French. This is consistent with German tagging 7.2% of its tokens as
  proper nouns against English's 11.6%.
- **Abstractive summaries score higher on similarity than extractive ones**, 76.2% at or above
  the 0.8 threshold against 66.9%, while reusing far less wording (median lexical overlap 0.84
  against 1.00). Verbatim reuse does not guarantee semantic closeness: TextRank picks three
  locally central sentences, which cannot represent an article developing several points, whereas
  the abstractive summarizer compresses across the whole article. Where an extractive summary does
  score highly, part of that score is mechanical, since the encoder is comparing a text against a
  document that contains it. The
  [coverage identity](Mathematical_theoretical_analysis.md#51-the-coverage-identity) derives this
  result from the geometry and predicts the sign of every correlation the pipeline measures.
- **Topic prediction reaches macro F1 between 0.87 and 0.90** across the four languages. The main
  confusion, Crime and law against Politics and conflicts, reflects genuine overlap in the source
  material rather than a modelling failure.

Full detail, with every table, in `reports/findings.md`.

### 1.2 Figures

| Figure | Shows |
|---|---|
| `entities_by_category.png` | The most reported entities in each category, one panel per category |
| `entity_timeline.png` | Articles per year for the leading entities |
| `ner_recall_by_language.png` | Share of English-named people that each language also tagged |
| `similarity_distribution.png` | Similarity per category and summarizer, against the 0.8 line |
| `similarity_vs_overlap.png` | Semantic similarity against how much wording was reused |
| `topic_confusion.png` | Topic classifier confusion matrix, English held-out set |

---

## 2. Requirements coverage

| Requirement | Implementation | Result |
|---|---|---|
| 1. Text pre-processing: units and grammatical roles | `src/pipeline/preprocess.py`, `src/nlp/pipelines.py` | `tokens.parquet`: sentence index, token, lemma, POS, tag, dependency; findings §2 |
| 2. NER, each entity tied to its article and metadata | `src/pipeline/ner.py`, `src/nlp/labels.py` | `entities.parquet`: one row per mention with `pageid`, title, language, category, date, character offsets, containing sentence; findings §3 |
| NER analysis, at least two criteria | `src/analysis/entities.py` | Four: aggregated view, dynamics over time, semantic clusters, geographic focus; findings §3.1 to §3.3 |
| 3. Investigate wrongly predicted entities | `src/analysis/ner_errors.py` | `ner_recall.parquet`, `ner_errors.parquet`: missed entities, label disagreements, surface issues; findings §4 |
| 4. Summarize 10 to 20 articles for 3+ categories | `src/summarize/extractive.py`, `src/summarize/abstractive.py` | `summaries.parquet`: 10 articles per category, 4 categories, 4 languages, 2 methods; findings §5 |
| Grammar and style findings | `src/analysis/quality.py` | `grammar.parquet`: readability and five style faults per method; findings §5.1 |
| 5. Similarity scores, visualized and explained | `src/analysis/similarity.py`, `src/analysis/embeddings.py` | `similarity.parquet`, two figures, highest and lowest cases explained; findings §6 |
| Predict the topic for a set of texts | `src/analysis/topics.py` | `topic_predictions.parquet`, `topic_metrics.json`; findings §7 |

---

## 3. Technology stack

Python 3.11.3, 197 packages resolved into `uv.lock`. Versions below are the locked ones;
`pyproject.toml` declares only the floors.

### 3.1 What the pipeline runs on

pandas 3.0.5 is the unit of work: every stage reads a DataFrame and writes one. Between stages that
DataFrame becomes a parquet file through pyarrow 25.0.1, which is what makes a single stage
re-runnable without the other ten. numpy 2.1.3 sits under the embedding and similarity arithmetic,
and tabulate 0.10.0 turns the final tables into the markdown of `reports/findings.md`.

spaCy 3.8.16 does the linguistic work behind requirements 1 to 3 — sentences, tokens, lemmas, parts
of speech, fine tags, dependencies and named entities — in one pass per language.

TextRank needs a PageRank and nothing else, which is the whole reason networkx 3.6.1 is here: the
sentence graph is built from scikit-learn's TF-IDF vectors and handed to `nx.pagerank`.
scikit-learn 1.9.1 earns its place several times over, though — the topic classifier's TF-IDF and
logistic regression, the K-means over entity embeddings, the train/test split and every metric.

Abstractive summarization goes through the `openai` client 3.13.0 pointed at OpenRouter's base URL
rather than OpenAI's, which keeps the model a configuration value (`NLP_LLM_MODEL`) instead of a
library choice. `python-dotenv` 1.2.3 reads the key.

sentence-transformers 6.0.1 loads `paraphrase-multilingual-MiniLM-L12-v2`, roughly 470 MB on the
CPU. It has to be multilingual: a Spanish summary is scored against its Spanish source, and a
monolingual encoder would make those numbers meaningless.

textstat 0.7.13 computes Flesch reading ease with each language's own coefficients. The obvious
alternative for the grammar requirement was LanguageTool, and it was rejected for needing a JVM —
this project stays inside Python and uv. The style faults in `src/analysis/quality.py` are
deterministic checks written against that constraint. matplotlib 3.11.2 renders the six figures.

### 3.2 Models

| Model | Version | Language | Trained on |
|---|---|---|---|
| `en_core_web_sm` | 3.8.0 | English | OntoNotes 5, 18 entity labels |
| `es_core_news_sm` | 3.8.0 | Spanish | WikiNER, 4 entity labels |
| `fr_core_news_sm` | 3.8.0 | French | WikiNER, 4 entity labels |
| `de_core_news_sm` | 3.8.0 | German | WikiNER, 4 entity labels |
| `paraphrase-multilingual-MiniLM-L12-v2` | — | 50+ | Fetched by sentence-transformers, cached locally |
| `openai/gpt-5-mini` | — | all four | Remote via OpenRouter, cached in `data/interim/llm_cache/` |

The label-count column is the reason `src/nlp/labels.py` exists: English reports 18 labels and the
other three report 4, so nothing cross-lingual is comparable until both are mapped onto one scheme.
The four spaCy pipelines are pinned as direct URLs for the reason given in section 4.

### 3.3 Tooling

uv resolves and locks everything, and every Make target runs through `uv run`, so there is no
activate step and no ambient interpreter. ruff 0.16.7 lints and formats; its `D` and `ANN` rule sets
are what actually enforce the docstrings and type hints this project's style asks for. mypy 2.3.1
type-checks `src/` and `tests/` under `disallow_untyped_defs`, with pandas-stubs supplying the types
pandas does not ship. pytest 9.1.1 carries the suite. JupyterLab 4.6.3 and ipykernel 7.3.0 exist for
`notebooks/report.ipynb` and nothing else. GitHub Actions runs lint, type check and the fast tests
on every push.

### 3.4 What comes in underneath

None of these are declared anywhere, but they are what the work actually runs on.

Behind sentence-transformers sits the whole HuggingFace stack: torch 2.14.0 as the tensor backend,
transformers 5.17.0 to assemble the encoder, tokenizers 0.23.2 for sub-words, huggingface-hub 1.31.0
to fetch and cache the weights, safetensors 0.8.0 as the format they arrive in, and filelock 3.32.6
with fsspec 2026.7.0 managing that cache. torch itself pulls sympy 1.14.0 and jinja2 3.1.6 for its
kernel machinery. This is the heaviest part of the install by a wide margin, and it is why
`src/__init__.py` caps the thread variables before anything imports it.

spaCy brings its own world. thinc 8.3.13 is Explosion's neural network library, and beneath it are
the Cython primitives: blis 1.3.3 for matrix kernels, cymem 2.0.13 for memory pools, murmurhash
1.0.15 and preshed 3.0.13 for hashing and hash tables. Around those sit srsly 2.5.3 for
serialisation, catalogue 2.0.10 for the registry, confection 1.3.3 for config parsing, wasabi 1.1.3
for console output, weasel 1.0.0 for project workflows, spacy-legacy 3.0.12 and spacy-loggers 1.0.5
for compatibility, smart-open 8.0.1 with cloudpathlib 0.25.0 for reading pipeline archives, and
typer 0.27.2 for the `spacy` command line.

scikit-learn adds scipy 1.17.1, which is where the sparse TF-IDF matrices actually live, plus
joblib 1.6.0 and threadpoolctl 3.6.0 — the last being the mechanism `NLP_THREAD_LIMIT` works
through. textstat adds nltk 3.10.3 and pyphen 0.18.1, since counting syllables is hyphenation and
pyphen is where the per-language dictionaries come from. matplotlib adds pillow 12.3.0, fonttools
4.65.0, kiwisolver 1.5.1, contourpy 1.3.3, cycler 0.12.1 and pyparsing 3.3.2.

A few are shared: pydantic 2.13.5, which spaCy, thinc and openai all validate against; httpx 0.28.1
and requests 2.34.2 for HTTP; regex 2026.9.10 for the Unicode-aware patterns transformers and nltk
need; tqdm 4.70.1 for the progress bars during tagging and encoding; and python-dateutil
2.9.0.post0 with tzdata 2026.4 behind every pandas date.

### 3.5 Standard library

Nothing is added on top of the standard library where it already does the job. Paths are `pathlib`,
stage output is `logging`, the raw corpus and `topic_metrics.json` go through `json`, and the
environment overrides in `src/config.py` are plain `os.environ`. Three uses are less obvious:
`hashlib` builds the LLM cache key out of the article text, `concurrent.futures` runs the four
parallel API calls, and `functools.lru_cache` keeps the spaCy pipelines and the sentence encoder
loaded once per process rather than once per call. `re`, `dataclasses` and `argparse` do what their
names suggest.

---

## 4. Running project

```bash
make setup     # dependencies, spaCy pipelines, corpus
make all       # full pipeline, writes reports/findings.md
make check     # lint, type check, tests
```

`make help` lists every target. Python 3.11 and uv are the only prerequisites; no GPU and no Java
runtime. Threads are capped in `src/__init__.py` and the sentence encoder is pinned to the CPU.

Abstractive summarization reads `OPENROUTER_API_KEY` from `.env`, and only for an article whose
summary is not already cached. The cache is committed, so reproducing the published run needs no
key. Everything else runs offline, and every later stage handles a single-method run:

```bash
make summarize SUMMARIZER=extractive
```

Runtime for `make all` on a laptop CPU: about 6 minutes for the spaCy pass, 1 minute for the rest,
plus API latency for any abstractive summary not already in the committed cache.

Two things that are specific to this project rather than to the tooling:

- The spaCy pipelines are declared as direct-URL entries in the `models` dependency group and
  pinned in `uv.lock`. One fetched with `spacy download` is declared nowhere, so the next sync
  prunes it and the run dies on a missing model. Install them with `make sync`.
- Stages communicate through parquet, so only the changed stage needs re-running. `make ner` reads
  the cached parsed documents rather than parsing again.

---

## 5. Pipeline stages

| # | Stage | Command | Reads | Writes | Req. |
|---|---|---|---|---|---|
| 1 | corpus | `make corpus` | raw JSONL | `corpus.parquet` | |
| 2 | preprocess | `make preprocess` | `corpus` | `tokens.parquet`, DocBin cache | 1 |
| 3 | ner | `make ner` | DocBin cache | `entities.parquet` | 2 |
| 4 | ner-analysis | `make ner-analysis` | `entities` | `entity_aggregate`, `entity_timeline`, `entity_clusters`, `geographic_focus` | |
| 5 | ner-errors | `make ner-errors` | `entities`, `corpus` | `ner_recall`, `ner_errors` | 3 |
| 6 | summarize | `make summarize` | `corpus`, DocBin cache | `summaries.parquet` | 4 |
| 7 | grammar | `make grammar` | `summaries` | `grammar.parquet` | |
| 8 | similarity | `make similarity` | `summaries` | `similarity.parquet` | 5 |
| 9 | topics | `make topics` | raw JSONL | `topic_predictions`, `topic_metrics.json` | |
| 10 | figures | `make figures` | all above | `reports/figures/*.png` | |
| 11 | report | `make report` | all above | `reports/findings.md` | |

Stage 2 is the expensive one. It writes the parsed spaCy documents to a `DocBin` in
`data/interim/`, and stages 3 and 6 read that cache instead of re-running the models.

Stage 11 skips the sections whose tables are absent, so a partial run still produces a coherent
report.

---

## 6. Code organisation

`pipeline/` imports from the other packages and never the reverse. That one rule is what lets the
unit tests run without loading a model or reading 45 MB of JSONL: the computation lives in
`analysis/`, `nlp/` and `summarize/`, and the stage modules only wire it together. Everything
except the `slow` tests finishes in about five seconds.

```
src/
├── __init__.py          # caps the thread variables before torch or tokenizers are imported
├── config.py            # paths, scope, seed, batch sizes, the env-var overrides
├── log.py               # one logging format for every stage
├── utils.py             # narrows pandas scalars so int()/str() type-check
│
├── data/
│   ├── __init__.py
│   └── loader.py        # raw JSONL to one tidy table: paragraphs, topics, ISO dates
│
├── nlp/
│   ├── __init__.py
│   ├── pipelines.py     # loads the four pipelines, parses once, writes the DocBin
│   └── labels.py        # 18 OntoNotes labels and 4 WikiNER ones onto one scheme
│
├── analysis/
│   ├── __init__.py
│   ├── entities.py      # ranks entities by article, traces them by year, clusters them
│   ├── ner_errors.py    # the three checks that stand in for a gold test set
│   ├── embeddings.py    # multilingual vectors; long text chunked and averaged
│   ├── similarity.py    # cosine similarity and lexical overlap
│   ├── quality.py       # Flesch readability per language, five style faults
│   └── topics.py        # TF-IDF and logistic regression against a baseline
│
├── summarize/
│   ├── __init__.py
│   ├── extractive.py    # TextRank, vectorizer fitted per article
│   └── abstractive.py   # OpenRouter client, prompt, and the disk cache
│
├── viz/
│   ├── __init__.py
│   ├── style.py         # one palette, one set of axis defaults
│   └── plots.py         # the six figures, one function each
│
└── pipeline/            # one module per make target, each with main()
    ├── __init__.py
    ├── corpus.py        # stage 1
    ├── preprocess.py    # stage 2
    ├── ner.py           # stage 3
    ├── ner_analysis.py  # stage 4
    ├── ner_errors.py    # stage 5
    ├── summarize.py     # stage 6
    ├── grammar.py       # stage 7
    ├── similarity.py    # stage 8
    ├── topics.py        # stage 9
    ├── figures.py       # stage 10
    └── report.py        # stage 11, markdown templates filled from the result tables

tests/
├── __init__.py
├── conftest.py               # in-memory frames shaped like the real tables; loads no model
├── test_loader.py            # topics, paragraphs, dates
├── test_corpus.py            # scope selection, capping, primary category
├── test_pipelines.py         # truncation and the spaCy helpers
├── test_labels.py            # label mapping across both tag sets
├── test_entities.py          # aggregation, ranking, timeline, geographic focus
├── test_ner_errors.py        # the three checks and the recall proxy
├── test_extractive.py        # sentence selection and document order
├── test_abstractive.py       # prompt and cache key, no API call
├── test_embeddings.py        # chunking and vector averaging
├── test_similarity.py        # similarity and overlap arithmetic
├── test_quality.py           # readability and each style fault
├── test_topics.py            # training, baseline, metrics
├── test_report.py            # the templates against known tables
└── test_pipeline_outputs.py  # slow: the section 7 contracts against real output
```

What each stage reads and writes is in section 5; what those tables guarantee is in section 7.

---

## 7. Data contracts

Guarantees between stages, checked by the `slow` tests in `tests/test_pipeline_outputs.py`.

### 7.1 `corpus.parquet`

One row per article (3,200 at default settings).

```
pageid  lang  title  text  topics  url  date
n_paragraphs  n_chars  year  scope_categories  primary_category  is_single_topic
```

`date` is ISO format for every language, including translations; the raw corpus dates only
English pages that way. `primary_category` is deterministic for multi-topic articles.
`is_single_topic` marks the unambiguous labels the topic classifier trains on.

A `pageid` identifies an *event*, not an article: the English page and its three translations
share one. Anything counting articles must group by `(lang, pageid)`.

### 7.2 `tokens.parquet`

One row per token (1,074,437).

```
pageid  lang  sent_id  token  lemma  pos  tag  dep  is_stop  is_punct  is_alpha
```

### 7.3 `entities.parquet`

One row per entity mention (75,855).

```
pageid  lang  title  category  date  year
entity  entity_key  label_raw  label  start_char  end_char  sentence
```

`label_raw` is the pipeline's own label, `label` the shared coarse scheme, `entity_key` the
normalised grouping key. Aggregation is deferred to stage 4, which needs mention-level detail to
count articles as well as occurrences.

### 7.4 `entity_aggregate.parquet`

One row per `(lang, category, entity_key)`.

```
lang  category  entity_key  mentions  articles  first_year  last_year  label  surface
```

`label` is the dominant coarse label, derived rather than used as a grouping key: a name the
pipeline tagged two ways inside one article would otherwise split into two rows and be counted as
two articles. Each article carries exactly one primary category, so summing `articles` across
categories gives an entity's total without double counting.

### 7.5 `summaries.parquet`

One row per article and method: 160 articles × 2 methods, so 320 rows.

```
pageid  lang  category  method  summary  title  text
source_chars  summary_chars  compression
```

`similarity.parquet` is this table with `similarity` and `lexical_overlap` appended.

---

## 8. Configuration

All tunables live in `src/config.py`. These read an environment variable:

| Variable | Default | Effect |
|---|---|---|
| `NLP_MAX_ARTICLES` | 800 | Articles per language retained for tagging and NER |
| `NLP_ARTICLES_PER_CATEGORY` | 10 | Articles summarized per category per language |
| `NLP_THREAD_LIMIT` | 2 | BLAS and OpenMP threads |
| `NLP_SPACY_SIZE` | `sm` | Pipeline size; pair with matching URLs in the `models` dependency group |
| `NLP_LLM_MODEL` | `openai/gpt-5-mini` | OpenRouter model for the abstractive summaries |

```bash
NLP_MAX_ARTICLES=200 make corpus preprocess
NLP_ARTICLES_PER_CATEGORY=20 make summarize
```

`NLP_THREAD_LIMIT` must be set before the first `import src`, because the numerical libraries read
their thread settings at import time.

The categories, languages, random seed (42) and batch sizes are constants in `src/config.py`.

`_PROMPT_VERSION` in `src/summarize/abstractive.py` is part of the LLM cache key. Bump it when the
prompt changes, or the cache serves summaries written to the old wording.

---

## 9. Testing

```bash
make test        # full suite, 138 tests
make test-fast   # the 123 that need no pipeline output
```

Unit tests run on the fixtures in `conftest.py` and load no models. The 15 `slow` tests in
`test_pipeline_outputs.py` are the exception: they check the section 7 contracts against real
stage output, and they skip rather than fail when those tables are absent. CI runs `make
test-fast`, so it leaves them out — they assert on the output of a full pipeline run rather than
on the code under change.

A `live` marker is registered in `pyproject.toml` for tests that call the API, but none are
written, so `make test` needs no key. `test_abstractive.py` checks the prompt and the cache key
without reaching the network.

---

## 10. Extending the analysis

| Task | Steps |
|---|---|
| Add a language | Add the code to `LANGUAGES` in `src/config.py`, its spaCy model to the `models` group in `pyproject.toml`, and its articles to `_ARTICLES_BY_LANGUAGE` in `src/nlp/labels.py`. Nothing else is language-specific: TextRank fits its vectorizer per article and the encoder is multilingual. |
| Add a category | Add it to `CATEGORIES` in `src/config.py`. It must be one of the 13 labels in `TOPIC_LABELS`. |
| Replace a summarizer | Write a function taking the article and returning a string, then call it from `build_extractive` or `build_abstractive` in `src/pipeline/summarize.py`. Downstream stages key on the `method` column and need no change. |
| Replace the embedding model | Change `EMBEDDING_MODEL` in `src/config.py`. It must be multilingual, or the non-English similarity scores stop meaning anything. |
| Add a figure | Implement it in `src/viz/plots.py` using `apply_style` and `colour_for`, then call it from `src/pipeline/figures.py`. |

---

## 11. Limitations and further work

### 11.1 What limits the results

**Cross-lingual recall is a proxy.** With no gold annotation, findings §4 estimates recall from
cross-lingual agreement: a person named in English, present verbatim in the translated text, that
the other pipeline left untagged. English is the reference, so its own false positives count against
the others, and a name both models miss never enters the denominator. The ordering de < fr ≈ es is
corroborated independently by the proper-noun share, but the levels are not calibrated
([§11](Mathematical_theoretical_analysis.md#11-the-cross-lingual-recall-proxy)). A few hundred
hand-labelled articles per language would replace the proxy with measured precision and recall.
Repeating the run with `lg` pipelines would separate capacity from training data in the German
result, which `sm` everywhere cannot.

**Similarity measures coverage, moderated by coherence.** A faithful summary covering *k* of a
document's *m* chunks is capped at √(k/m) when those chunks are unrelated, however well it
summarises what it covers. Whether it is accurate never enters the calculation. The same geometry
predicts the sign of every correlation in the drivers table and the shape of the lowest-scoring
cases ([§5](Mathematical_theoretical_analysis.md#5-why-a-correct-summary-can-score-low)). Scoring
against each source chunk and taking a coverage-weighted maximum removes the cap; subtracting a
redundancy term from TextRank's objective would spread its three sentences across the article rather
than its densest region.

**The 0.8 threshold passes verbatim copies.** Read alone it credits an extractive summary for
reproducing its source, which is why lexical overlap is always reported beside it
([§7](Mathematical_theoretical_analysis.md#7-the-08-threshold-as-a-decision-rule)). Both are
unsupervised proxies; only a sample scored against human reference summaries would show whether
either tracks editorial judgement.

**The clusters are indicative.** A silhouette of 0.196 is the expected magnitude for one-to-three
token strings with no sentence context, not a defect, but the six groups overlap at their boundaries
and are not a partition to rely on. Embedding each name inside a sentence of context would sharpen
them, at the cost of one vector per mention rather than one per entity.

**Entity aliasing is not entity linking.** A hand-checked table merges the common abbreviations, so
two spellings that are neither identical nor in it still rank as separate entities. Resolving
mentions against a knowledge base would merge the spellings the table cannot reach and split the
homonyms it wrongly merges.

**The corpus thins after 2011.** Year-on-year figures for the later period rest on few articles, and
the decline visible in the timeline tracks WikiNews publishing volume rather than news volume.

### 11.2 What that means in use

The business case is screening news for brand health, competitors and political affairs. The entity
table is an index produced by a model with a measured error rate, so a count of how often a person
appears is an estimate, and one whose error rate varies by language. A monitoring system built on it
under-reports German-language subjects silently: nothing in the output distinguishes a subject who
was absent from one the model missed. Findings §8.1 states this in full.

Abstractive summarization sends article text to a third-party API. That is acceptable for published
journalism and would not be for internal documents; the extractive path runs entirely offline.
