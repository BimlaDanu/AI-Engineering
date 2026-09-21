"""Tools: the work the model may ask for but never performs itself.

Six of them, and they fall into two kinds that are treated differently on purpose:

* **Inward** -- :func:`~src.tools.chains.compare_chain` and
  :func:`~src.tools.sweeps.sweep_field` run the same verified physics at a length
  or a field the question was not asked at. Their numbers are cross-checked
  exactly like the main answer's, so nothing arriving through a tool is a weaker
  result.
* **Outward** -- :func:`~src.tools.papers.find_papers` (arXiv),
  :func:`~src.tools.wiki.look_up` (Wikipedia), :func:`~src.tools.websearch.search_web`
  (a configured search endpoint) and :class:`~src.tools.mcp.McpServer` (whatever a
  connected MCP server advertises). Nothing here can check what comes back, so it
  is labelled unverified wherever it surfaces and may be offered only as further
  reading.

Two rules apply to every outward tool. **None of them takes a URL** -- the host is
a constant in the calling module and the model names only a search term, so a tool
argument cannot become a request to somewhere unexpected; :mod:`src.tools.http` is
the single way out to the network. And **an unconfigured tool is withheld** from
the schema list rather than offered and failed.

:mod:`src.tools.calling` holds the machinery: the schemas the model sees, the
validation its arguments face, and the single round in which they run. Keeping each
capability as a plain function in its own module means the physics is testable
without a model, and the tool description lives in exactly one place -- the schema
whose docstring the model reads.
"""
