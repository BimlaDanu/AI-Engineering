"""Pipeline page: the graph, with the last question's path lit up.

The page that answers *how did it decide?* -- and the one that is hardest to fake,
because the drawing is not a diagram somebody maintained. The nodes and edges are
read off the compiled graph itself, so a node added to the agent appears here with
no work, and a diagram that has drifted from the code is not a failure mode this
page has.

The path is recorded rather than inferred, and recorded where it happens
-------------------------------------------------------------------------

Asking a question on the Chat page records its route as a side effect, so this page
shows the last real question rather than asking for a second run of its own -- a
trace of a question nobody asked describes a path nobody took. Under the hood the
graph is *streamed* rather than invoked, and each node's name is appended as it
finishes. So the highlighted route is what actually
ran, in the order it actually ran, including the loop going round more than once --
not a plausible path reconstructed afterwards from the final state. That distinction
matters more than it sounds: the commonest way an agent disappoints is by taking a
shorter route than its diagram suggests, and a reconstruction would hide exactly
that.

What was searched for, and who wrote the query
---------------------------------------------

Retrieval is a loop rather than a lookup: it searches, grades what came back, and
rewrites the query when a round kept nothing. All of that was happening and none of
it was visible -- the citations survived and every round that produced them was
discarded, so a thin answer from a corpus with nothing on the subject looked exactly
like a thin answer from a query that never named the subject. The panel under the
question shows both sides: the question as it was typed, the query that actually
went to the index, and every round in between with its author.

The two sides differ on a feasibility question even when nothing was rewritten,
because the opening query there is *built* rather than quoted -- it deliberately
asks about the classical methods too, so the reading list behind a verdict is not one
in which quantum always looks promising. That is a decision, and this is where it
becomes checkable.

The working lives here too
--------------------------

The trail the campaign wrote as it went, the configurations it ran, the ones it
refused on arithmetic before spending anything, and the report in full are all
below the graph. They used to sit under every reply on the Chat page, which served
neither reader -- it pushed the answer out of sight for somebody who came to ask a question, and
scattered the audit trail across a thread for somebody checking the agent. Here it
is in one place, against the diagram the route ran through.

Reading the loop
----------------

Three nodes do the feasibility work and they run in a cycle, which is why their
counts are the largest numbers on the page:

``plan``
    Propose one configuration -- a circuit at a particular depth -- and price it
    against the machine's coherence time and the measurement budget *before* anything
    is spent. If it is unaffordable the campaign goes straight back to ``plan`` for a
    cheaper one, which is the small self-loop on the box.
``solve``
    Run that circuit and get an energy back. This is the step every other step exists
    to serve. It was called ``run`` until a reader pointed out that a graph reading
    plan, run, analyse never says where the problem is actually solved.
``analyse``
    Work out what the energy means and whether going deeper could do better. If it
    could, back to ``plan``; if not, on to the verdict.

So ``plan`` at six and ``solve`` at four is not six of anything abstract: it is six
configurations proposed, four of them run and two refused on arithmetic before they
cost a measurement. Those counts live in the table under the diagram and not on the
boxes. The label went through ``plan 6x`` -- read as a multiplier by everyone who
saw it -- and then ``plan / ran 6 times``, which said it plainly and put a second
line of text on every repeated box. The name alone leaves the picture to carry the
shape of the route, which is the one thing only the picture can do.

A run where ``plan`` appears and ``solve`` never does is a campaign that refused
everything before spending anything -- a legitimate outcome, and a completely
different one from a campaign that tried.

The branch that only appears sometimes
--------------------------------------

``converge`` runs when the question asked to *watch* something rather than be told
where it ended -- *plot the loss curve for these three methods*, *which of them
converges fastest*. It races VQE, QAOA and imaginary-time evolution on the chain the
campaign formalised and keeps their curves, then hands on to whichever branch answers
the question. It runs **before** that branch rather than beside it, so the answer can
quote the numbers instead of being written as though nothing had been computed.

Seeing it grey on the diagram is not a fault: it means this question asked for a
verdict or an explanation and not for a picture of a descent.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from src.agent.graph import pipeline_dot, pipeline_mermaid, pipeline_nodes, run_campaign
from src.hardware.devices import device_for
from src.ui import panels

NODE_PURPOSE: dict[str, str] = {
    "recall": "read what this thread has already been asked, so a follow-up inherits it",
    "screen": "check the question for an injection attempt, and whether it is in scope at all",
    "interpret": "decide what kind of answer this question wants, before any work is done",
    "formalise": "turn the sentence into a Hamiltonian, and record every assumption made",
    "retrieve": "find background worth citing, if the question warrants a search",
    "baseline": "run the ordinary-computer solver the quantum arm has to beat",
    "converge": "race the three methods on this chain and keep the curve each one traced",
    "plan": "propose the next configuration, and price it before spending anything",
    "solve": "run the circuit, get an energy back, and charge the measurement budget",
    "analyse": "work out what the run did, and whether trying again could do better",
    "consult": "offer the model its tools -- arXiv, symbolic algebra, costing -- and "
    "record every call it makes",
    "explain": "answer in prose from the notes, running no circuit and reaching no verdict",
    "implement": "write runnable code for this chain, and label it as code nobody here ran",
    "skeptic": "argue against the conclusion before it is written down",
    "scribe": "write up whichever answer the branch produced, for a non-physicist",
    "suggest": "propose what to ask next, and screen each suggestion before offering it",
    "remember": "write the exchange back to the thread, so the next question can use it",
}
# What each node is for, in one line.
#
# Kept here rather than in the graph because it is a description for a reader, not a
# docstring for a maintainer, and because a node whose purpose cannot be stated in one
# line is usually a node doing two things.

panels.header(
    "Pipeline trace",
    "The graph the agent runs, drawn from the compiled graph itself, with the last "
    "question's route through it.",
    "src/agent/graph.py",
)

setting = panels.current_setting()
nodes = pipeline_nodes()
# Read once, up here, because two blocks need it: the search trail at the top, which
# belongs beside the question it came from, and the evidence tabs at the bottom.
last_campaign = st.session_state.get("last_campaign")
path: list[str] = st.session_state.get("path", [])
asked: str = st.session_state.get("path_question", "")
visits = Counter(path)

request = st.session_state.get("path_request")

if asked:
    st.markdown("##### What was asked, and what was understood")
    left, right = st.columns(2)
    with left:
        st.caption("As it was typed")
        st.info(asked)
    with right:
        st.caption("As the agent restated it")
        restated = getattr(request, "clarified", "")
        if restated:
            st.info(restated)
        else:
            st.caption(
                "No restatement was made -- either the question was already clear, or "
                "there was no language model to ask. The original was used as it stood."
            )
    st.caption(
        "The restatement is stored **beside** the question, never in place of it. "
        "Replacing it would tidy away the pressure in the wording, and measuring "
        "whether a verdict moves when only the wording changes is only possible if "
        "the wording survives the trip. On a question that asks for an explanation "
        "or for code it is what the corpus is searched with -- the panel below shows "
        "the query that actually went to the index -- and it is shown here so a "
        "reader sees what was understood before seeing what was concluded."
    )
    if last_campaign is not None:
        # Below the restatement rather than beside it, because the three are one
        # chain and it reads in that order: what was typed, what that was taken to
        # mean, and what was therefore searched for. Split across two pages -- or
        # two columns of a page -- a reader has to reconstruct the chain themselves.
        st.divider()
        panels.query_trace(last_campaign)
    if request is not None and not request.in_scope:
        st.warning(
            f"{panels.WARNING} This question was judged to be outside what the "
            "assessment covers, so no chain was invented for it and the route below "
            "stops early. That is why only a few nodes are filled."
        )
    st.divider()
else:
    st.caption("Nothing asked yet in this session, so no route is marked.")

st.graphviz_chart(pipeline_dot(path), width="stretch")

if path and "solve" not in path:
    # The commonest thing a reader wonders at, and nothing on the page answered it:
    # half the graph is grey because the question was not a feasibility question, not
    # because anything went wrong. Named here rather than left to be inferred from
    # the colours.
    st.info(
        f"{panels.PRESENT} **The circuit loop did not run for this question, and that "
        "is the route working.** `plan -> solve -> analyse` prices a circuit, runs it "
        "and decides whether a deeper one is worth the measurements -- which answers "
        "*is this worth putting on quantum hardware?* and answers nothing else. A "
        "request to be taught something, or to be given code, is answered from the "
        "notes through `consult`, and spending a minute of optimisation on it would "
        "produce a number nobody asked for. Ask about a named chain -- *is a 10-spin "
        "critical chain worth running?* -- to fill the other half."
    )

ran, last, idle, ends, edges = st.columns(5)
ran.markdown(":blue-badge[blue] ran for this question")
last.markdown(":violet-badge[dark] where the run ended")
idle.markdown(":grey-badge[grey] in the graph, not run this time")
ends.markdown("**START / END** the graph's entry and exit")
edges.markdown("**→ always** · **⇢ a decision picks the branch**")
st.caption(
    "**A blue box may have run more than once, and the table below says how many "
    "times.** The counts are kept off the boxes on purpose: this picture is for the "
    "*shape* of the route, and a second line of text on every repeated box competes "
    "with it. Three nodes are the ones that repeat -- `plan` (propose a circuit and "
    "price it before spending anything), `solve` (run it and get an energy back) and "
    "`analyse` (decide whether going deeper could do better). The small loop on "
    "`plan` itself is a configuration priced, refused and replaced with a cheaper one."
)
st.caption(
    "Drawn by LangGraph from the compiled graph, not from a diagram anybody "
    "maintains -- so a node added to the agent appears here with no work, and this "
    "picture cannot drift from the code. The two ovals, the dashed edges and the "
    "words on them are LangGraph's own: dashed means the next node was chosen at run "
    "time, and the label is the name of the branch that was taken."
)
with st.expander("Diagram source (Mermaid, for pasting anywhere that reads it)"):
    st.caption(
        "The same graph in Mermaid, straight from LangGraph. Offered as text rather "
        "than rendered here because Mermaid has to be fetched from a content delivery "
        "network by your browser, and a diagram that needs the internet to appear is "
        "a diagram that is sometimes simply missing. The picture above renders "
        "natively and always."
    )
    st.code(pipeline_mermaid(path), language="text")

st.divider()

if not path:
    panels.empty_state(
        "No question has been asked yet in this session.",
        "Ask one on the **Chat** page and come back -- the route is recorded as the "
        "campaign runs, so nothing has to be re-run to see it. Or trace one here.",
    )
else:
    st.success(
        f"{panels.PRESENT} {len(path)} node visits across {len(visits)} distinct "
        f"nodes, recorded as the campaign ran rather than reconstructed afterwards."
    )
    st.code(" → ".join(path), language="text")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "node": name,
                    "ran": visits[name],
                    "what it does": NODE_PURPOSE.get(name, ""),
                }
                for name in nodes
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if visits["plan"] > visits["solve"]:
        st.info(
            f"**`plan` ran {visits['plan']} times and `solve` ran {visits['solve']}.** The "
            "difference is configurations refused on arithmetic before anything was "
            "spent on them -- too deep for the machine's coherence time, or too "
            "expensive for the measurement budget. That is what a practitioner does "
            "before booking machine time. The **What it refused** tab below names "
            "each one and the arithmetic that ruled it out."
        )
    if not visits["solve"]:
        st.warning(
            f"{panels.WARNING} **`solve` never fired.** Every configuration proposed was "
            "refused before it cost anything, so the verdict rests entirely on the "
            "ordinary-computer baseline and on arithmetic. That is a legitimate "
            "outcome and a completely different one from a campaign that tried."
        )

st.divider()
st.subheader("The working behind the last answer")
st.caption(
    "Everything the campaign did, kept here rather than under the reply on the Chat "
    "page. A reader who wants the answer wants the answer; a reader who wants to "
    "audit it wants the trail beside the graph it ran through, which is this page."
)
if last_campaign is None:
    panels.empty_state(
        "No campaign has finished in this session.",
        "Ask a question on the **Chat** page, or trace one below -- either fills this in.",
    )
else:
    panels.campaign_evidence(last_campaign)

with st.expander("Trace a different question from here"):
    st.caption(
        "Runs a whole campaign. Offline it takes a couple of seconds, costs nothing "
        "and needs no key -- and the route is the same one a live run takes, because "
        "which node runs next was never a language model's decision."
    )
    question = st.text_input(
        "question",
        value="Is quantum hardware worth it for a 10-spin critical Ising chain?",
        label_visibility="collapsed",
    )
    if st.button("Trace it", type="primary"):
        fresh: list[str] = []
        with st.spinner("Running, one node at a time..."):
            traced = run_campaign(
                question,
                shot_budget=setting.physics.shots,
                device=device_for(setting.physics.device),
                chat_model=None if setting.model.offline else "auto",
                # Searched even with no language model. The corpus half that needs no model
                # -- keyword retrieval over the notes -- returns citations perfectly well on
                # its own, and switching it off with the model was costing the offline
                # demonstration every citation it could have had. Which shelf to search is
                # then chosen by rule instead of by a model, which is a slightly worse
                # choice and a very much better outcome than no sources at all.
                search_corpus=True,
                fetch_external=False,
                visited=fresh,
            )
        st.session_state["path"] = fresh
        st.session_state["path_question"] = question
        st.session_state["last_campaign"] = traced
        st.session_state["path_request"] = traced["request"]
        st.rerun()

st.divider()
st.subheader("What the model calls cost, in money")
panels.session_cost()

panels.footer("src/agent/graph.py")
