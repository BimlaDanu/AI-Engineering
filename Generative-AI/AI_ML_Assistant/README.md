# 🧠 Synapse: *where AI, ML & deep learning connect*

[![check](https://github.com/BimlaDanu/AI-Engineering/actions/workflows/check-ai-ml-assistant.yml/badge.svg?branch=main)](https://github.com/BimlaDanu/AI-Engineering/actions/workflows/check-ai-ml-assistant.yml)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://docs.streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/docs/introduction/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Chroma](https://img.shields.io/badge/Chroma-FF6F61)](https://docs.trychroma.com/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-6566F1?logo=openrouter&logoColor=white)](https://openrouter.ai/docs/quickstart)
[![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)

Synapse is a **domain-specialised [RAG](https://arxiv.org/abs/2005.11401) chatbot** for machine
learning, deep learning and AI engineering questions. It retrieves answers from a curated
knowledge base and cites its sources. When it does not know something, it says so.

> **Grounded · Cited · Honest.** Every factual answer points at the passages it came from, or
> states its uncertainty. Off-topic questions and prompt-injection attempts are refused.

Each badge links to that tool's documentation. I wrote this for the Turing College Sprint 2
programme, against the brief in [125.md](125.md).

<details>
<summary><b>New to the terms?</b> A plain-English glossary with links (click to expand)</summary>

| Term | In simple terms | Go deeper |
|---|---|---|
| **RAG** (retrieval-augmented generation) | The app looks up relevant passages in its own library and answers *from* them. Grounded rather than guessed. | [Lewis et al., 2020](https://arxiv.org/abs/2005.11401) · [LangChain tutorial](https://python.langchain.com/docs/tutorials/rag/) |
| **Knowledge base (KB)** | The curated library of documents it answers from. | [`data/`](data/) |
| **Embedding** | Text turned into a list of numbers, so the app can measure which passages are closest *in meaning*. | [OpenAI guide](https://platform.openai.com/docs/guides/embeddings) |
| **Chunk** | A document sliced into smaller passages. The unit that gets retrieved and cited. | [LangChain splitters](https://python.langchain.com/docs/concepts/text_splitters/) |
| **Vector store** (Chroma) | The database holding those number-lists, which finds the closest matches. | [Chroma docs](https://docs.trychroma.com/) |
| **Hybrid search / BM25** | Meaning-based matching combined with keyword matching, so acronyms and names are not missed. | [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) |
| **Corrective RAG (CRAG)** | When the passages found look too weak, fetch more from outside instead of bluffing. | [Yan et al., 2024](https://arxiv.org/abs/2401.15884) |
| **Structured output** | The model replies in a fixed, checkable shape, so the app never guess-parses prose. | [OpenAI guide](https://platform.openai.com/docs/guides/structured-outputs) |
| **LangGraph** | Writing the pipeline as an explicit graph of steps you can draw and inspect. | [LangGraph docs](https://langchain-ai.github.io/langgraph/) |
| **MCP** (Model Context Protocol) | A standard way to borrow tools from a remote server. | [modelcontextprotocol.io](https://modelcontextprotocol.io/) |
| **RAGAs** | Scoring answer quality (faithfulness, relevancy, precision, recall) with an LLM as judge. | [RAGAs docs](https://docs.ragas.io/) |
| **Prompt injection** | A message that tries to trick the model into ignoring its rules. | [OWASP LLM01](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) |

</details>

---

## 📚 Contents
1. [Get started](#get-started)
2. [What makes Synapse different](#what-makes-synapse-different)
3. [How Synapse finds an answer](#how-synapse-finds-an-answer)
4. [Features at a glance](#features-at-a-glance)
5. [Accounts and personalisation](#accounts-and-personalisation)
6. [Explore the workspaces](#explore-the-workspaces)
7. [Project structure](#project-structure)
8. [Configuration](#configuration)
9. [Testing](#testing)
10. [Evaluation results](#evaluation-results)
11. [Technology stack](#technology-stack)
12. [Tasks checklist](#tasks-checklist)
13. [Current limitations](#current-limitations)
14. [App preview](#app-preview)

---

## Get started

**Prerequisites:** Python 3.11+, the [`uv`](https://docs.astral.sh/uv/) package manager, and an
[OpenRouter API key](https://openrouter.ai/keys).

```bash
make sync     # 1. build the uv-managed virtual environment
              # 2. put OPENROUTER_API_KEY=... in a .env file at the project root
make ingest   # 3. build the vector store from data/  (spends embedding tokens)
make run      # 4. launch the app
```

Optional `.env` keys: `GOOGLE_API_KEY` for the native Gemini model, and
`EMBEDDING_BACKEND=local` for free offline embeddings instead of the API default.

> **Step 3 is required before your first question**, because the app cannot search an empty
> index. Re-run `make ingest` after changing anything in `data/`, and after switching the
> embedding backend or model, since a different embedder produces number-lists of a different
> size. Stop the app first, or restart it afterwards.

| Command | Description |
|---|---|
| `make help` | List every target (also what bare `make` prints) |
| `make run` | Launch the Streamlit app |
| `make ingest` | Rebuild the Chroma vector store from `data/`. Confirms first, since it costs tokens |
| `make doctor` | Read-only readiness report: index drift, credentials, accounts. Spends nothing |
| `make check` | The CI gate: lint, format check, 455 offline tests |
| `make fix` | Auto-fix whatever `make check` would reject |
| `make test` | Just the tests. One test: `make test T=tests/test_x.py::test_y` |
| `make mcp-serve` | Optional. Publish Synapse's own tools as an MCP server |
| `make clean` | Remove caches, leaving `.venv` alone |

`make check` is exactly what CI runs, on every push to `main` and every pull request that
touches this directory
([`check-ai-ml-assistant.yml`](https://github.com/BimlaDanu/AI-Engineering/blob/main/.github/workflows/check-ai-ml-assistant.yml),
the badge at the top). This directory keeps its own [`check.yml`](.github/workflows/check.yml) with
the same steps, for when Synapse is pushed as a repository of its own. Neither needs a secret,
because the suite is fully offline with a fake LLM, KB and tools.

---

## What makes Synapse different

Ask it *"How do transformers use attention?"* or *"Explain LoRA to a beginner."* Three rules
govern every reply:

- **In the knowledge base.** Answered from the retrieved passages, with inline citations
  (`[1]`, `[2]`) you can expand to read the source.
- **On-topic but not in the knowledge base.** Answered from general knowledge, behind an explicit
  *"(Not covered by the knowledge base…)"* disclaimer.
- **Off-topic, or a prompt-injection attempt.** Politely refused.

You steer *how* it teaches from ⚙️ Settings:

| Dial | Options |
|---|---|
| **Subject** | Machine Learning · Deep Learning · AI Engineering · their shared Overlap · All |
| **Learner level** | Beginner · Practitioner · Researcher |
| **Prompt technique** | Standard · Chain-of-Thought · Few-shot · Socratic · Analogy-first |
| **Response length** | Concise · Balanced · Detailed |

Picking a subject filters which documents are in play. The shared-foundation notes in
`data/overlap/` show up whichever single subject you choose, because they underpin all of them.

---

## How Synapse finds an answer

```
User question
     │
     ▼
1. Input validation  ───────────────► reject empty / oversized input
     │
     ▼
2. Injection screening (classifier + regex patterns)  ──► refuse if malicious   ← always first
     │
     ▼
3. Route classification (structured output + confidence)
     │
     ├── knowledge → retrieve → grade sufficiency → [augment if weak] → generate (cited)
     ├── tool      → tool-calling answer (arXiv · calculator · token/cost)
     ├── meta      → answer about the app itself ("what can you do?")
     ├── agent     → bounded plan → act → observe loop  (opt-in)
     └── off_topic → polite domain refusal
     │
     ▼
Cited answer + sources + tool cards + full trace  →  rendered in Streamlit
```

Safety is screened before anything else, and a small cheap model does the routing so the larger
chat model is only used when it is needed. Retrieval rewrites follow-up questions into standalone
queries, and when the passages it finds are too weak it falls back to
[corrective RAG](https://arxiv.org/abs/2401.15884), fetching from arXiv or the web and citing
those too.

The pipeline lives in `src/core/` and never imports Streamlit, which is what makes it testable on
its own. Two engines run the same steps: **linear** (default) and **graph**
([LangGraph](https://langchain-ai.github.io/langgraph/), easier to visualise). A parity test keeps
their results identical. The routing model is independent of the chat model, so one model can
classify and another can write.

---

## Features at a glance

[125.md](125.md) lists the core requirements plus optional Easy, Medium and Hard tasks.

| Requirement | How the app supports it | Code |
|---|---|---|
| **1 · RAG: KB, embeddings, chunking, similarity search** | Front-matter-tagged docs, split by `RecursiveCharacterTextSplitter` (1200/200), stored in Chroma with cosine distance. Retrieval is hybrid: dense vectors blended with a hand-written BM25 scorer. | `src/rag/` |
| **2 · At least 3 domain tool calls** | arXiv search, a SymPy calculator (killable, timeout-guarded subprocess), and a token and cost estimator, in a bounded tool loop. | `src/tools/`, `src/generation.py` |
| **3 · Domain specialisation, prompts, security** | Curated AI/ML KB, a grounded system prompt with level, technique and length styling, and a layered security stack. | `src/generation.py`, `src/security.py` |
| **4 · LangChain + OpenRouter, errors, validation** | `ChatOpenAI` via OpenRouter, optional native Gemini, input validation, and graceful degradation that is enforced rather than incidental. | `src/llm.py`, `src/security.py` |
| **5 · Intuitive UI: context, sources, tools, progress** | Multi-page Streamlit studio with cited-source expanders and tool-call cards. No blocking call hangs silently: a spinner for single calls, live per-node status for the graph engine, a progress bar for batch runs. | `src/ui/`, `src/utils.py` |

<details>
<summary><b>Sub-requirement checklist</b>, each item pointing at the function that implements it</summary>

Symbol names rather than line numbers, because line numbers go stale on the next edit.

**RAG** · KB in `data/{ml,dl,ai,overlap}/` · embeddings in `src/rag/embeddings.py`
(`OpenAIEmbeddings` plus offline `LocalEmbeddings`) · chunking in `build_vector_store`
(`src/rag/ingest.py`) · similarity search in `src/rag/retriever.py` (Chroma cosine blended with
BM25)

**Tool calling** · `search_arxiv` (`src/tools/arxiv.py`) · `estimate_tokens_and_cost`
(`src/tools/tokens.py`) · `math_calculator` (`src/tools/calculator.py`) · bounded tool loop in
`src/generation.py`

**Domain specialisation** · subjects and techniques in `src/config.py` · prompt styling in
`src/generation.py` · injection patterns and domain vocabulary in `src/security.py`, screened by
`screen_injection` (`src/core/router.py`) and run first by `src/core/steps.py`, ahead of routing

**Technical** · `ChatOpenAI` with `base_url=OPENROUTER_BASE_URL` (`src/llm.py`) · error handling
in `src/core/service.py` and `src/generation.py` · `validate_input` (`src/security.py`), called
from `src/core/service.py`

**UI** · `src/app.py` and `src/ui/pages/` · cited sources in `chat.py` · tool cards and the
playground in `tools.py` · `st.spinner`, `st.status` and `st.progress` across all 10 pages that
make a slow call, with per-stage timings in 🧪 Experiments

</details>

---

## Accounts and personalisation

**Personalisation is always on and needs no account.** The ⚙️ Settings dials live in session
state, so each visitor gets an independent setup for the length of their visit, and the choices
travel with the conversation export. Nothing is stored per user on disk.

**Sign-in is real but delegated.** `st.login()` and `st.user` (Streamlit 1.42+, plus `authlib`)
put **Log in** and **Sign up for free** in the top bar and send the visitor to Google, Auth0,
Microsoft or Okta. Both buttons start the same flow, because with an identity provider there is
no separate sign-up. Synapse stores no passwords and never sees a credential. Configure it under
`[auth]`; `.streamlit/secrets.toml.example` has the walkthrough. With nothing configured the app
runs open, which is what `make run` on a laptop should do.

A `streamlit-authenticator` password gate is the fallback for a host with no provider. It reads
**`[password_auth]`**, not `[auth]`, because Streamlit reads an unrecognised `[auth]` sub-table as
the name of a provider. There is no self-service sign-up on that path, since `st.secrets` is
read-only on Community Cloud.

**Use a key.** A top-bar popover takes an OpenRouter key so a visitor's questions are billed to
them. It is validated against OpenRouter before being accepted, because a key with one character
missing would otherwise fail quietly: Synapse retrieves and cites before it generates, so the run
looks healthy right up until the answer. The key stays in that session's server-side store, never
on disk or in a log, and **Forget my key** drops it. Internally it is a thread-local override in
`src/config.py`, not a module global, since each Streamlit session runs in its own thread and a
global would let one visitor's key pay for another's question.

**The top bar** sits on the header strip beside Deploy, holding 🆕 New chat, the login buttons and
Use a key. Once someone is identified those collapse to their name with an account menu behind it.
Below 900px the row falls back into the page. The offset clears Streamlit's Stop button, which is
the one neighbour it must never cover.

**Themes.** Streamlit's ⋮ menu holds the System / Light / Dark switch. Synapse adds no toggle of
its own, but it has to earn that one: Streamlit discards its built-in themes as soon as an app
declares one, and draws the switch only while more than one theme is left. A single `[theme]`
block therefore removes the control entirely. `.streamlit/config.toml` declares the palette twice
instead, as `[theme.light]` and `[theme.dark]`, which restores the switch and makes System the
default. `tests/test_theme_config.py` pins that split.

---

## Explore the workspaces

The sidebar is built from four pieces: the Synapse glyph, grouped navigation over thirteen
workspaces, the 🕘 Past chats panel, and a one-line signature at the foot.

| Sidebar group | Workspace | Intended use |
|---|---|---|
| **🧭 AI/ML Research Assistant** | 🏠 Home | Landing dashboard: KB and session status, plus jump-in cards |
| | 💬 AI Chat | The main assistant: cited answers, tool cards, model settings, export |
| **Learn** | 🎓 AI/ML Tutor | Guided, level-aware learning paths |
| | 🔬 AI/ML Lab | Hands-on recipes, LLM exercises, and a live code cell |
| | 🎯 Trivia | KB-grounded quizzes with instant feedback |
| **Analyse** | 📈 Analytics | Session-wide token usage and cost. Per-question detail is in 🧪 Experiments |
| | 📊 Evaluation | Score **one** RAG configuration over the golden set |
| | 🆚 A/B testing | Compare **two** RAG strategies under one shared judge |
| | 🧪 Experiments | Inspect **one** question end to end, plus a playground for each tool call |
| **Knowledge** | 📰 AI News | Live arXiv papers, lab newsrooms, digests, voices to follow |
| | 📚 Stacks | Evergreen courses, tools and references per subject |
| | 📄 Knowledge Base | Inspect, upload and re-index docs; promote external passages |
| **System** | ⚙️ Settings | Model, generation, RAG tuning, engine, MCP, status |

Group order comes from `registry.SECTION_ORDER` and page order from each page's `order` field.
The sidebar and the 🏠 Home cards are the same grouping drawn twice, both from
`registry.get_sections()`, so they cannot fall out of step. Adding a workspace is one file in
`src/ui/pages/` with a `@register_page` decorator.

**🕘 Past chats** sits under the navigation on every page. The current thread is written out on
every run, so 🆕 New chat never destroys anything: what just left the screen becomes the top row.
The panel draws the 12 newest and says how many it is not showing, because every row is two
widgets rebuilt on every rerun. Both deletes take two presses, and the pending state is keyed per
thread so a rerun cannot arm the row below the one just pressed. Threads are one JSON file each,
in a directory per visitor keyed by a hash of their identity, which on a shared deployment is the
only thing keeping their questions apart.

---

## Project structure

A pure-Python core with a Streamlit UI on top. Everything under `src/core`, `src/rag`,
`src/tools`, `src/eval`, `src/lab` and `src/quiz` runs without Streamlit and is unit-tested
offline. `src/ui` is the only place the UI framework is imported.

| Package | What lives there |
|---|---|
| `src/core/` | The answer pipeline: routing, injection screening, the shared steps, both engines, the agent loop, external sources, the MCP client, saved chats, and `AssistantService` |
| `src/rag/` | Embeddings, ingestion, hybrid retrieval, and KB promotion |
| `src/tools/` | arXiv search, the SymPy calculator, token and cost estimation |
| `src/eval/` | Self-hosted RAGAs-style harness: judge, four metrics, dataset, runner, compare |
| `src/lab/` | 🔬 Lab and 🎓 Tutor: curriculum, six toy datasets, recipes, challenges, sandbox |
| `src/quiz/` | 🎯 Trivia: build items from KB passages, grade deterministically |
| `src/ui/` | Page registry, session state, theme, top bar, past chats, and one module per workspace |
| Top level | `config.py`, `llm.py`, `generation.py`, `security.py`, `auth.py`, `export.py`, `ratelimit.py`, `runlog.py`, `utils.py`, `doctor.py`, `mcp_server.py` |

<details>
<summary><b>Full file tree</b> with a one-line note per module (click to expand)</summary>

```
src/
├── app.py                # entry point: wires the top bar, grouped sidebar, and active page
├── config.py             # paths, the model + price registry, thread-local runtime keys
├── generation.py         # prompts, answer styling, and the RAG + tool-calling loop
├── llm.py                # chat-model factory: OpenRouter by default, native Gemini optional
├── security.py           # input validation, injection screening, the AI/ML domain gate
├── auth.py               # identity: OIDC sign-in, own-key, password-gate fallback
├── export.py             # a conversation as a JSON, CSV, or PDF transcript
├── ratelimit.py          # per-session token-bucket rate limiter
├── runlog.py             # append-only JSONL run log
├── utils.py              # token/cost estimation, stage timing, retry on transient failure
├── doctor.py             # `make doctor`: read-only readiness report
├── mcp_server.py         # optional: expose the app's own tools to any MCP client
│
├── core/                 # the answer pipeline. never imports Streamlit
│   ├── schemas.py        #   typed router / injection results
│   ├── router.py         #   pick a route, screen for injection
│   ├── steps.py          #   the shared steps, one source of truth for both engines
│   ├── linear.py         #   engine A, a lightweight if/elif router (the default)
│   ├── graph.py          #   engine B, the same steps as a LangGraph StateGraph
│   ├── agent.py          #   opt-in bounded plan -> act -> observe loop
│   ├── sources.py        #   pluggable external sources for CRAG (arXiv, web)
│   ├── mcp_client.py     #   borrow a remote MCP server's tools for the loop
│   ├── chats.py          #   saved conversations: one JSON per thread, one dir per visitor
│   └── service.py        #   AssistantService: runs one request, owns its cost and trace
│
├── rag/
│   ├── embeddings.py     #   pluggable backends (API default, local fallback)
│   ├── ingest.py         #   build the Chroma store from data/, via a staging swap
│   ├── retriever.py      #   hybrid BM25 + vector search, and query rewriting
│   └── promote.py        #   fold approved external passages into the curated KB
│
├── tools/                #   arxiv.py · calculator.py · tokens.py
├── eval/                 #   schemas · judge · metrics · dataset · runner · compare
├── lab/                  #   curriculum · datasets · recipes · challenges · sandbox · seed
├── quiz/                 #   quiz.py · schemas.py
│
└── ui/                   # the only place Streamlit is imported
    ├── registry.py       #   @register_page, get_sections, cross-page jumps
    ├── state.py          #   session state, and the KB cache keyed on the live collection
    ├── theme.py          #   brand mark, injected CSS, hero banner, footer
    ├── account.py        #   top bar: New chat, Log in, Sign up, Use a key, account menu
    ├── past_chats.py     #   🕘 Past chats panel, autosave, two-press deletes
    ├── exporters.py      #   shared JSON / CSV / PDF download buttons
    ├── models.py         #   the model picker, shared by Settings and the Chat popover
    └── pages/            #   home · chat · ml_tutor · ml_lab · quiz · resources · ai_news
                          #   analytics · evaluation · ab_testing · inspector
                          #   knowledge_base · tools · settings

data/
├── ml/ dl/ ai/ overlap/  # the knowledge base: 23 notes; the folder sets a doc's subject
├── eval/golden.jsonl     # 21-sample golden set
└── promoted/             # notes promoted from external sources

.streamlit/config.toml    # light and dark variants of the brand violet, and no telemetry
chroma_db/                # generated vector store (git-ignored; rebuilt by make ingest)
tests/                    # 455 offline tests; LLM, KB, tools and sources all injectable
```

</details>

---

## Configuration

Set through `.env`, loaded by `python-dotenv`, plus the ⚙️ Settings page. Keys are never
hard-coded, printed or committed.

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | none | **Required.** LLM and embedding access |
| `GOOGLE_API_KEY` | none | Optional. Enables the native `gemini-native` model |
| `EMBEDDING_BACKEND` | `api` | `api`, or `local` for free offline sentence-transformers |
| `API_EMBEDDING_MODEL` | `openai/text-embedding-3-large` | The API embedding model |
| `ROUTER_MODEL` | `openai/gpt-4o-mini` | Cheap model for routing and injection screening |
| `SYNAPSE_CHAT_DIR` | `outputs/chat_history` | Where saved conversations are written |

Chat models and per-1M-token prices live in the registry in `src/config.py`. Retrieval and
generation knobs live in `RagSettings` and are surfaced in ⚙️ Settings.

---

## Testing

- **`make test` runs 455 tests, all offline.** The LLM, knowledge base, tools and external
  sources are injectable, so every route is exercised deterministically with no network and no
  API key. Highlights: engine parity, security and injection, hybrid BM25, corrective RAG, the
  agent loop, evaluation metrics and concurrency, export, and MCP.
- **`make lint` and `make format`** run [Ruff](https://docs.astral.sh/ruff/) at line length 100,
  targeting py311, with rules `E F I UP B SIM C4 RET PERF PTH RUF ICN ERA S TRY`. Every exemption
  in `pyproject.toml` carries the reason it is exempt.
- **`make doctor`** reports index drift, credentials and account config without spending a token
  or printing a secret. The quickest way to find out why an answer looked wrong.
- **The root `conftest.py` is intentionally empty.** Under pytest's default `prepend` import mode
  its mere presence puts the project root on `sys.path`, so `from src…` resolves everywhere. No
  editable install, no `PYTHONPATH`.

---

## Evaluation results

The harness is self-hosted (`src/eval/`), so these come from running it on this repo rather than
from a published benchmark. Reproduce them by opening 📊 Evaluation, keeping the defaults, and
running the golden set.

<!-- TODO: run 📊 Evaluation on the default config and paste the four scores in, then delete
     this comment. Do not ship placeholder numbers. -->

| Metric | What it asks | Score |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved passages? | _TBD_ |
| **Answer relevancy** | Does the answer address the question asked? | _TBD_ |
| **Context precision** | Of the passages retrieved, how many were useful? | _TBD_ |
| **Context recall** | Of what was needed to answer, how much did retrieval find? | _TBD_ |

Setup: the 21-sample golden set, judge model `openai/gpt-4o-mini`, default `RagSettings`. Scores
are LLM-judged, so expect a few points of run-to-run variance.

<!-- TODO: run one A/B comparison (e.g. hybrid alpha=0.5 vs pure vector) and record which won. -->

| A/B comparison | Variant A | Variant B | Winner |
|---|---|---|---|
| _e.g._ hybrid `alpha=0.5` vs pure vector | _TBD_ | _TBD_ | _TBD_ |

---

## Technology stack

| Component | Technology |
|---|---|
| **Language** | [Python 3.11+](https://www.python.org/downloads/), managed with [uv](https://docs.astral.sh/uv/), linted with [Ruff](https://docs.astral.sh/ruff/) |
| **Interface** | [Streamlit](https://docs.streamlit.io/), multi-page and registry-driven |
| **Orchestration** | [LangChain](https://python.langchain.com/docs/introduction/) and [LangGraph](https://langchain-ai.github.io/langgraph/) |
| **Model access** | [OpenRouter](https://openrouter.ai/docs/quickstart), with optional native [Gemini](https://ai.google.dev/gemini-api/docs) |
| **Vector database** | [Chroma](https://docs.trychroma.com/), local and persistent |
| **Embeddings** | OpenRouter API by default, [sentence-transformers](https://www.sbert.net/) offline |
| **Documents** | [pypdf](https://pypdf.readthedocs.io/), for PDFs in `data/` and uploads in 📄 Knowledge Base |
| **Validation** | [Pydantic](https://docs.pydantic.dev/), for structured data and model outputs |
| **External tools** | [arXiv](https://arxiv.org/help/api), [SymPy](https://docs.sympy.org/), optional remote [MCP](https://modelcontextprotocol.io/) |
| **Export** | [fpdf2](https://py-pdf.github.io/fpdf2/), plus JSON and CSV |
| **Testing** | [pytest](https://docs.pytest.org/), fully offline |
| **AI assistance** | Claude Code and ChatGPT, for development, debugging and documentation |

---

## Tasks checklist

Numbered as in [125.md](125.md). Every core requirement is done, along with all ten Medium
tasks and four of the seven Hard ones, including both hard optionals I set out to do: the
self-hosted RAGAs evaluation and the agent route.

**Easy** · 1 conversation history and export ✅ (`chat.py`, `export.py`, 🕘 Past chats) ·
2 RAG-process visualisation ✅ (🧪 Experiments trace) · 3 source citations ✅ (inline `[n]`,
📎 Sources expander) · 4 interactive help ◑ partial (the `meta` route answers "what can you
do?", plus Home and tooltips, but no scripted tour)

**Medium**

| # | Task | Status | Where |
|---|---|---|---|
| 1 | Multi-model support | ✅ | registry in `config.py`, one shared picker (`ui/models.py`), `llm.py` |
| 2 | Real-time KB updates | ✅ | upload and re-index, plus live promotion (`knowledge_base.py`, `rag/promote.py`) |
| 3 | Prompt-injection protection | ✅ | patterns, LLM classifier, hardened prompt, domain gate ([OWASP LLM01](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)) |
| 4 | Authentication and personalisation | ✅ | OIDC plus password-gate fallback (`auth.py`), per-session dials in ⚙️ Settings |
| 5 | Token usage and cost display | ✅ | `utils.py`, `tools/tokens.py`, 📈 Analytics |
| 6 | Tool-result visualisation | ✅ | 🛠️ tool-call cards, tool playground |
| 7 | Export to PDF, CSV, JSON | ✅ | `export.py` (PDF via fpdf2; CSV formula-injection guard) |
| 8 | Remote MCP server tools | ✅ | `core/mcp_client.py`, DeepWiki by default, opt-in |
| 9 | Rate limiting and key management | ✅ | token-bucket limiter (`ratelimit.py`), live key status panel |
| 10 | Logging and monitoring | ✅ | append-only JSONL run log (`runlog.py`) with a Settings toggle |

**Hard**

| # | Task | Status | Where |
|---|---|---|---|
| 1 | Hybrid search | ✅ | vector and BM25 blended with an `alpha` dial (`rag/retriever.py`) |
| 2 | A/B testing of RAG strategies | ✅ | `eval/compare.py`, 🆚 A/B testing page |
| 3 | Automated KB updates | ◑ partial | CRAG auto-fetches passages, but promotion into the permanent KB is human-approved, not scheduled |
| 4 | Multi-language support | ✗ not attempted | the KB, prompts and domain gate are English-only |
| 5 | Advanced analytics dashboard | ◑ partial | session token and cost plus the JSONL log, but nothing historical or aggregated |
| 6 | Tools exposed as MCP servers | ✅ | `mcp_server.py`, FastMCP publishes every `ALL_TOOLS` entry (`make mcp-serve`) |
| 7 | RAG evaluation, RAGAs or otherwise | ✅ | self-hosted four-metric LLM-as-judge harness (`src/eval/`) |

**Extras beyond the spec:** the 🔬 AI/ML Lab (parameterised recipes, six toy datasets, LLM-seeded
exercises, and a restricted in-process code runner), 🎯 Trivia (KB-grounded MCQs written through
structured output, each with a citation, graded deterministically), and KB promotion, which
writes an approved external passage back as a curated note *and* adds it to the live index at
once.

---

## Current limitations

Areas I would look at next.

- **Run `make ingest` first**, and again after changing documents or the embedding backend. A
  different embedder produces differently-sized vectors, so the old index no longer matches.
- **`pyproject.toml` pins `streamlit>=1.36`, below what two features need.** Native sign-in wants
  1.42+, and the light/dark theme split wants 1.45+. Installing at the floor gives an app where
  both silently do nothing. Raising the floor is the fix.
- **Costs and token counts are estimates.** Cost comes from a static price table, so it needs
  updating as prices change, and token counts are heuristic wherever the provider returns no usage.
- **Web search and remote MCP cost money and need network**, and both are off by default. MCP also
  needs the optional `langchain-mcp-adapters` dependency and a reachable server, and fails soft to
  an empty tool list.
- **The domain gate is deliberately conservative.** It will sometimes refuse a borderline question
  rather than risk answering off-domain.
- **Prompt-injection defence is layered, not absolute.** The classifier, patterns, hardened prompt
  and domain gate reduce the risk without eliminating it.
- **The starter KB is 23 curated study notes.** For production use, add real papers through
  📄 Knowledge Base or the `data/` folders and re-index.
- **Evaluation and A/B testing make a lot of LLM calls**, roughly one answer plus about six judge
  calls per question, so an A/B run costs about double a single evaluation. They run concurrently
  over a thread pool (a ⚡ Parallel requests slider, default 8), with retry and backoff on every
  judge call. Set it to 1 for sequential behaviour, or lower if a cheap judge starts rate-limiting.
- **Accounts need a provider, saved chats need a volume.** With no `[auth]` provider the login
  buttons render disabled, and Community Cloud wipes `outputs/chat_history` on redeploy unless
  `SYNAPSE_CHAT_DIR` points at a mounted volume.
- **The Lab's code runner is a guardrail, not a hardened sandbox.** It restricts imports and
  builtins to catch accidents, not to contain a determined adversary.
- **Answers arrive complete rather than streamed**, because of the tool-calling loop.
- **English only.** The KB, the prompts and the domain gate all assume it.

---

## App preview

![Synapse, the AI Chat workspace](AI-ML-CHAT.png)
