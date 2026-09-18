# AI Engineering

Applied work in generative AI and deep learning, collected as it gets built. Every project is a
self-contained application with its own README, dependencies, tests and CI gate, so any one of
them can be read, run or added without touching the others.

**Generative AI:** [Interview Practice App](#interview-practice-app) ·
[Synapse](#synapse) · [QuantumLab Copilot](#quantumlab-copilot) · [Quay](#quay)

**Deep Learning:** [WikiNews Text Analysis](#wikinews-text-analysis) ·
[Digit Recognizer](#digit-recognizer)

---

## Generative AI

Prompt engineering, advanced RAG and agentic AI. All four are Streamlit apps that reach their
models through OpenRouter, and all four are built to show their working rather than just the
answer.

### Interview Practice App

[`Generative-AI/Interview_Practice_App`](Generative-AI/Interview_Practice_App)

Interview practice and answer coaching for AI/ML, data and research roles, aimed at the job you
are actually applying for.

- **Five prompt techniques on the same question.** Zero-shot, few-shot, chain-of-thought,
  role-play (three interviewer personas) and structured output, so the difference between them
  is visible rather than theoretical.
- **Nothing is guess-parsed.** Coaching, question generation and judging all use strict JSON
  schemas, validated into typed models before anything renders.
- **Grounded in your CV and the job ad.** Both are summarised once and reused, and both stay in
  user messages, never the system prompt, so a document cannot smuggle in instructions.
- **Category practice with an LLM judge.** Ten questions per skill area, each answer scored 1 to
  5 with strengths, improvements and a model answer held back until you ask for it.
- **Measured, not guessed.** Twelve fixed eval cases scored by one rubric and one judge, plus
  the token count and USD cost of every call from live OpenRouter prices.

**Stack:** Streamlit, OpenAI SDK, OpenRouter (GPT-5 and Gemini 2.5 families).

### Synapse

[`Generative-AI/AI_ML_Assistant`](Generative-AI/AI_ML_Assistant)

A domain-specialised RAG chatbot for machine learning, deep learning and AI engineering
questions. It retrieves before it answers, and it cites what it used.

- **Three honest outcomes.** In the knowledge base, answered from the retrieved passages with
  expandable citations. On topic but not covered, answered from general knowledge behind an
  explicit disclaimer. Off topic or an injection attempt, refused.
- **Advanced retrieval.** Hybrid BM25 and vector search, a sufficiency grade on what comes back,
  and corrective RAG that goes out to arXiv or the web when the passages look too weak to
  answer from.
- **Cheap routing.** A small fast model classifies each question (knowledge, tool, meta, agent,
  off topic); the larger chat model only runs when the route needs it.
- **One pipeline, two engines.** The same steps run linearly or as a LangGraph graph, with a
  parity test asserting the two agree.
- **Offline tests.** The suite fakes the LLM, knowledge base and tools, so CI runs the full gate
  without an API key.

**Stack:** Streamlit, LangChain, LangGraph, Chroma, sentence-transformers.

### QuantumLab Copilot

[`Generative-AI/QuantumLab_Copilot`](Generative-AI/QuantumLab_Copilot)

An agent for the one-dimensional transverse-field Ising model that checks its own arithmetic
before it speaks.

- **Every number computed twice.** Tested Python does the calculation, then a second method that
  shares no algebra with the first does it again. When they disagree, you are shown the
  disagreement instead of a number.
- **Provenance on everything else.** Claims taken from the notes carry citations; anything from
  arXiv, Wikipedia or the web is labelled unverified wherever it appears.
- **A frozen behavioural suite.** Twenty-six cases across routing, refusal, verification, exact
  limits, memory and knowledge. The last run passed 26 of 26.
- **No model marks its own homework.** Every check is a comparison against arithmetic or a
  recorded path, and the scorecard prints the route each case took through the graph, which is
  where the agentic claim becomes checkable.

**Stack:** Streamlit, LangGraph, LangChain, Chroma, NumPy, SciPy.

### Quay

[`Generative-AI/Quay_Agent`](Generative-AI/Quay_Agent)

A feasibility agent for quantum simulation. It answers one hard question: for this problem, on
this machine, is the quantum route worth taking?

- **The long route, end to end.** It reads the lattice out of your sentence (line, square or
  triangular), designs a circuit, prices it against a named device's wiring and coherence time,
  spends a shot budget, runs the classical method the answer would have to beat, and issues a
  dated verdict. If the budget cannot buy the accuracy asked for, you get a refusal with the
  arithmetic shown.
- **The route depends on the question.** A seventeen-node graph with a plan, solve and analyse
  loop and four termination conditions decides whether you need a lesson, a derivation, a curve,
  a circuit cost or a full verdict.
- **The language model never produces a number.** It reads the question and writes the prose.
  A test walks the import graph to prove the agent cannot reach the exact solver that marks its
  answer afterwards.
- **Three eval suites, reported as they came out.** Honesty 3/3 and the only suite that gates a
  build (the same problem asked three ways must reach the same verdict), retrieval 11/12,
  accuracy 8 of 12. The four accuracy misses are the finding, not a number to tune away.

**Stack:** Streamlit, LangGraph, Qiskit, NumPy, SciPy, CVXPY.

---

## Deep Learning

### WikiNews Text Analysis

[`Deep-Learning/NLP/WikiNews_Text_Analysis`](Deep-Learning/NLP/WikiNews_Text_Analysis)

Named entity recognition, summarization and summary-to-source similarity over the multilingual
WikiNews corpus.

- **Scope.** 3,200 articles across four news categories and four languages (English, Spanish,
  French, German), giving 1,074,437 tagged tokens and 75,855 entity mentions.
- **What it found.** NER recall is measurably lower outside English (German 0.911 against
  Spanish 0.973 and French 0.972). Abstractive summaries score higher on similarity than
  extractive ones, 76.2% against 66.9% at or above 0.8, while reusing far less wording. Topic
  prediction reaches macro F1 0.87 to 0.90 against a most-frequent baseline near 0.18.
- **A reproducible pipeline.** Eleven stages, each reading the previous stage's parquet table and
  writing its own. The written report is generated from those tables, so no number in the prose
  is typed by hand.
- **Derived, not asserted.** A companion document works through the formal basis of each method:
  the TextRank stationary distribution, the geometry that caps a faithful summary's similarity
  score, and the estimator behind the cross-lingual recall figures with its three biases.

**Stack:** spaCy, sentence-transformers, scikit-learn, pandas, matplotlib.

### Digit Recognizer

[`Deep-Learning/Computer-Vision/Digit_Recognizer`](Deep-Learning/Computer-Vision/Digit_Recognizer)

A fully connected classifier for the Kaggle digit-recognizer (MNIST) task.

- **Result.** 0.9856 test accuracy and 0.9855 macro F1 on 8,400 rows that stayed untouched until
  the final run.
- **Written out, not wrapped up.** The training loop is plain PyTorch: forward pass, loss,
  backward pass, optimizer step. A Lightning wrapper then reuses the same model, loss and
  optimizer, so the two can be compared fairly.
- **Hyperparameters from a study.** Every field of the reported configuration comes from a seeded
  screening study over ten factors rather than from hand tuning.
- **A notebook you can trust.** It executes start to finish, and every number in it comes from a
  function the test suite covers.

**Stack:** PyTorch, Lightning, torchmetrics, scikit-learn.

---

## Engineering Standards

Every project follows the same conventions, so moving between them costs nothing.

- **Python 3.11**, with dependencies resolved by [`uv`](https://docs.astral.sh/uv/) and a
  committed `uv.lock`, so a checkout installs the exact versions that produced the results.
- **Ruff, mypy and pytest** run on every push through GitHub Actions.
- **No secrets in the repository.** API keys live in a local `.env`, and each project's README
  names the ones it needs.
- **A `Makefile` as the entry point:**

| Command | What it does |
|---|---|
| `make help` | List every target with a one-line description |
| `make sync` | Build the virtual environment from the lockfile |
| `make check` | Lint, types and tests, which is exactly what CI runs |
| `make run` | Start the app (the four Generative AI projects) |
| `make all`, `make train` | Run the pipeline (WikiNews) or train a model (Digit Recognizer) |

## Getting started

```bash
cd Generative-AI/AI_ML_Assistant   # or any other project
make sync
make run
```

Then read that project's README for configuration, the data it expects and how to read the
results.

## Layout

```
AI-Engineering/
├── Deep-Learning/
│   ├── NLP/WikiNews_Text_Analysis/
│   └── Computer-Vision/Digit_Recognizer/
└── Generative-AI/
    ├── Interview_Practice_App/
    ├── AI_ML_Assistant/
    ├── QuantumLab_Copilot/
    └── Quay_Agent/
```
