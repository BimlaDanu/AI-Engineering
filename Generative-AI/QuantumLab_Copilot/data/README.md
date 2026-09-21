# `data/` — the knowledge bases

```
data/corpus/physics-notes/*.md        the model itself
data/corpus/quantum-computing/*.md    how quantum computing uses it
data/corpus/applications/*.md         what it is used for outside physics
```

That's all of it, and that's the rule: **everything in `data/` is committed.**
Nothing generated lives here. Generated artefacts get their own top-level folders
beside this one — `chroma_db/` for the index, `.checkpoints/` for conversation
state, both gitignored. So whether a file is version-controlled is answered by its
path, not by reading `.gitignore`.

```
data/          committed documents      ← the knowledge base
chroma_db/     generated index          ← gitignored, rebuild with make ingest
```

Nothing at question-time reads `data/`; once the index exists the corpus is only
consulted by the next ingest. Deleting `chroma_db/` is therefore always safe,
though re-embedding costs API calls.


## Three shelves, because choosing between them is a decision

One directory per shelf, declared in `src/rag/ingest.py` as `SHELVES`. Every chunk
carries its shelf, the router picks which shelf to search for each question, and
the interface shows which one answered. A directory nobody declared is **refused**
at ingest rather than indexed: notes nothing can search are worse than no notes.

| Shelf | What belongs on it |
| --- | --- |
| `physics-notes` | The model: the exact solution, the free-fermion mapping, the critical point at `h = J`, exponents, the transfer matrix and the quantum-to-classical mapping, and the practice and limits of exact diagonalisation. |
| `quantum-computing` | The model as the standard toy model of quantum computing: VQE, QAOA and state preparation, annealing and the minimum gap, Trotterised circuits, the hardware platforms that realise it, and how a solvable model verifies a device. |
| `applications` | What the model is used for outside physics: business and operational problems written as Ising cost functions, portfolio optimisation and finance, routing, scheduling and other NP-hard problems, the commercial Ising machines, machine learning applied to the model and built out of it, which quantum speedups are actually proven, and what industrial R&D does with a solvable instance. |

The split is not filing tidiness. A hardware result and a closed-form result are
different kinds of statement — one is an experiment on a device, the other is
arithmetic — and a reader is owed the difference. It also gives the router
something real to decide, and a decision that can be wrong is a decision worth
showing: if the chosen shelf comes back empty, the search widens to the whole
library and the trace says so.

**Adding a shelf** means four edits, and the third is the one that is easy to miss:

1. a `Shelf` in `SHELVES` (`src/rag/ingest.py`), and the directory beside the others;
2. its vocabulary in `SHELF_TERMS` (`src/agent/router.py`) so the offline router can
   pick it — the keys are validated against the registry at import, so a typo fails
   immediately instead of producing a filter that matches nothing;
3. its vocabulary in **`DOMAIN_TERMS`** as well. That list is the scope gate, not a
   ranking hint: a question carrying none of its words is refused offline before any
   model is called. A shelf whose subject is missing from it is a shelf of notes the
   agent refuses to answer from. `SHELF_TERMS` may be broader — it only chooses
   between shelves for a question that is already in scope, so `business` belongs
   there while only `business problem` belongs in the gate;
4. a label in `SHELF_CHOICE_LABELS` (`src/ui/panels.py`), so the dropdown that forces
   a shelf does not show a bare slug. A test holds that map to the registry's length.

Then `make ingest`, or the new notes are found by the keyword half of the search and
missed by the vector half.

**Filenames must be unique across shelves.** Chunk ids are derived from the
filename, so two notes called `benchmarking.md` on different shelves would
overwrite half of each other's chunks in the index, silently. `load_corpus`
refuses it.

## Writing a document

Notes *with citations* — the claim, the equation, and a pointer precise enough to
check against the original. Ingestion reads these frontmatter keys and attaches
them to every chunk as metadata:

```markdown
---
title: Exact solution of the transverse-field Ising chain
source: "Pfeuty, Annals of Physics 57, 79 (1970)"
arxiv: null            # arXiv id, DOI, or null for pre-arXiv work
topics: [exact-solution, free-fermions, critical-point]
---

Prose. LaTeX as `$...$` or `$$...$$`.
```

Two rules, both correctness rather than tidiness. `tests/test_corpus.py` enforces
them.

**Never write a computed number here.** Not one energy, not one critical exponent. The
agent's claim is that its numbers come from exact diagonalisation verified against
an independent solver — and a number sitting in the corpus is retrievable and
indistinguishable from one that was computed. Nothing downstream can tell them
apart. Qualitative statements are what this corpus is *for*: *the gap closes
linearly*, *the exponent is universal*. Writing `$\eta = 1/4$` as what CFT
**predicts** is fine — that is a fact about the theory, and it is what a computed
value gets checked against. Never write the output of a finite-`$L$` calculation.

**One idea per file.** Chunking splits on structure, so a file covering four
topics yields chunks that match four queries weakly instead of one query well.

## Paths

`corpus_path`, `vector_store_path` and `chroma_collection` in `src/settings.py`,
overridable from the environment — an evaluation run can build a throwaway index
without disturbing the one the app is serving.
