"""The agent: state schema, graph, nodes, and the policy they apply.

This package holds what the project means by "the agent" --
the LangGraph state and its edge predicates, and the nodes that plan, select a
method, run it, verify the result and reflect on a failure. Nothing else.
Settings and logging live as flat modules at the top of ``src`` because
everything imports them and neither is big enough to be a package.

Facts versus policy is the line that keeps this package small. *Which methods
exist, when each applies and what each costs* are physics facts and live in
:mod:`src.physics`; *which applicable method to prefer, when to ask the human,
when to give up* is policy and lives here. That split is what lets a notebook
do method selection with no agent, no API key and no LangGraph.

The boundary rule, in one line: **`src/agent` never imports `src/ui`.** The
dependency points one way only, which is what makes the graph testable headless
and what would let a FastAPI front end replace Streamlit without touching a
single node.
"""
