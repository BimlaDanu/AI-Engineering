# `data/` — the knowledge bases

Committed and reproducible: the Markdown corpus the retriever indexes, and a
pre-computed table of circuit costs. Nothing here needs a credential to rebuild.

```
data/corpus/physics-notes/*.md        38 notes   the model itself
data/corpus/quantum-computing/*.md    67 notes   how quantum computing uses it
data/corpus/method-comparison/*.md     8 notes   choosing between the methods
data/corpus/applications/*.md         13 notes   uses outside physics
data/resource-sweep.json                         circuit cost on every machine
```

Generated files live outside `data/` and are git-ignored: `chroma_db/` is the
vector index (`make ingest`), `.checkpoints/` is conversation state.

Neither file here is read while a question is being answered. The corpus is read
only by the next ingest, so deleting `chroma_db/` is always safe — rebuilding it
costs embedding calls. `resource-sweep.json` exists so a resource claim can be
checked against a table without running anything; rebuild it with `make sweep`,
which takes about a second and needs no network.

## The four shelves

One directory per shelf, declared as `SHELVES` in `src/rag/ingest.py`. Every chunk
carries its shelf, the router picks a shelf per question, and the interface shows
which one answered. An undeclared directory is refused at ingest, not indexed.

| Shelf | What belongs on it |
| --- | --- |
| `physics-notes` | The model: the exact solution, the free-fermion mapping, the critical point at $h = J$, critical exponents, the transfer matrix and the quantum-to-classical mapping, exact diagonalisation and its limits. |
| `quantum-computing` | The model as a toy model of quantum computing: VQE, QAOA and state preparation, annealing and the minimum gap, Trotterised circuits, hardware platforms, and how a solvable model verifies a device. |
| `method-comparison` | How the near-term methods compare and when to pick which: VQE, QAOA, variational imaginary time and annealing, how a classical objective becomes a cost operator and a mixer, how deep a circuit should be before noise wins. |
| `applications` | Uses outside physics: operational problems written as Ising cost functions, portfolio optimisation, routing and scheduling, commercial Ising machines, machine learning, and which quantum speedups are proven. |

**Filenames must be unique across shelves.** Chunk ids derive from the filename, so
two notes named `benchmarking.md` on different shelves would silently overwrite
half of each other's chunks. `load_corpus` refuses it.

Adding a shelf takes three edits: a `Shelf` in `SHELVES` with its directory; its
vocabulary in `SHELF_TERMS` (`src/agent/router.py`), which chooses between shelves;
and its vocabulary in `DOMAIN_TERMS`, which is the scope gate. A subject missing
from the gate is refused before any model is called, so a shelf absent from it is a
shelf the agent never answers from. Then `make ingest`.

## Writing a note

Every note is Markdown with a frontmatter fence. Ingestion reads these keys and
attaches them to each chunk:

```markdown
---
title: Exact solution of the transverse-field Ising chain
source: "Pfeuty, Annals of Physics 57, 79 (1970)"
arxiv: null            # arXiv id, DOI, or null for pre-arXiv work
topics: [exact-solution, free-fermions, critical-point]
---

Prose. LaTeX as $...$ or $$...$$.
```

`title`, `source` and `topics` are required; a note missing one is refused at
ingest.

**Never write a computed number.** Not one energy, not one critical exponent. A
number sitting in the corpus is retrievable and indistinguishable from one that was
computed, and nothing downstream can tell them apart. Qualitative statements are
what the corpus is for — *the gap closes linearly*, *the exponent is universal*.
Stating $\eta = 1/4$ as a prediction of the theory is fine; that is what a computed
value gets checked against. Never write the output of a finite-$L$ calculation.

**One idea per file.** Chunking splits on structure, so a file covering four topics
yields chunks that match four queries weakly instead of one query well.

## Paths

`corpus_path`, `vector_store_path` and `chroma_collection` in `src/settings.py`,
each overridable from the environment, so an evaluation run can build a throwaway
index without disturbing the one being served.
