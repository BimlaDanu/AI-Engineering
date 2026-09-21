"""Agentic retrieval over the physics literature.

Two halves. :mod:`src.rag.ingest` reads the committed Markdown corpus, chunks it,
embeds it and persists a Chroma collection -- run offline via ``make ingest``.
:mod:`src.rag.retrieve` runs inside the graph, and it is agentic rather than
naive: the router decides *whether* to retrieve at all, then retrieval grades
what comes back, rewrites the query once when nothing is worth keeping, and
returns empty so the caller refuses instead of answering from memory.

No paper is fetched here. The corpus is Markdown notes with citations in their
frontmatter, for the reasons in ``data/README.md`` -- chiefly that no PDF loader
is declared, so a PDF would be a file nothing can read.

Two invariants hold here and are enforced in code rather than in a prompt:

1. **Retrieved text is untrusted input.** It is delimited and carried in user
   messages, never concatenated into a system prompt.
2. **The vector store holds only prose.** Numerical results are keyed by an
   exact spec hash elsewhere; semantic similarity retrieving an ``L=12`` result
   for an ``L=13`` question would be a correctness disaster, not a near miss.
"""
