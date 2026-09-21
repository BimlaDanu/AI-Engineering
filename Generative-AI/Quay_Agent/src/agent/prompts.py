"""Every instruction this project sends to a language model, in one place.

Prompts are code: they decide behaviour, they regress when edited, and one
written inline at its call site is one nobody reviews. Collecting them here makes
a change to any of them a change with a diff.

Every prompt names the job in one sentence, so the model is not inferring its
role from the shape of the schema. It states the criteria as a short list,
because a criterion the prompt does not name is one the model invents. It shows
one worked example including the negative case -- the largest quality difference
in this project came from showing what declining looks like, since without it a
grader keeps one weak passage rather than none. And it says what the answer is
for: a model told its shelf choice costs one extra round rather than the answer
calibrates differently from one told only to pick a name.

Two things no prompt here does.

No few-shot example contains a number that could be mistaken for a result. Every
example uses a subject the project does not compute -- shelf names, passage
identifiers, query wording. An example carrying a plausible ground-state energy
is a value that can be copied into an answer, and nothing downstream could tell
it from a measured one.

No prompt asks for mathematics inside a structured field. JSON and LaTeX
disagree about what a backslash means, and the disagreement eats the notation
silently. Prose calls carry the mathematics; structured calls carry the decision.
"""

from __future__ import annotations

from src.settings import Audience

AUDIENCE_GUIDANCE: dict[Audience, str] = {
    "beginner": (
        "The reader has no physics and no quantum computing. Use no symbol without "
        "first saying what it stands for in words, keep to one equation at most, and "
        "prefer a concrete picture -- magnets, a dial, a coin -- to a definition. "
        "Never leave a term of art unglossed on the assumption that it is common."
    ),
    "entrepreneur": (
        "The reader is deciding whether to spend money. Lead with the decision, the "
        "cost and the condition under which the answer changes; give the mechanism "
        "only where it is what makes the cost what it is. No derivations, and no "
        "equation that does not directly carry a number they would act on."
    ),
    "software, no physics": (
        "The reader is fluent in code and new to the physics. Assume complexity "
        "classes, sampling error, optimisers and floating point; assume nothing about "
        "Hamiltonians, spins or gates. Gloss every physics term in the sentence it "
        "first appears in, and reach for a software analogy where an honest one "
        "exists rather than inventing one where it does not."
    ),
    "practitioner": (
        "The reader is competent but not a specialist in this corner of it. Assume "
        "the standard vocabulary, gloss anything narrower than a first course, and "
        "give the equation where it is shorter than the prose."
    ),
    "quantum computing engineer": (
        "The reader works on this hardware. Give the circuit depth, the gate counts, "
        "the error budget and the shot arithmetic directly, with no gloss. Skip the "
        "motivation for the model and spend the space on where the estimate is soft."
    ),
    "researcher": (
        "The reader knows the field. Be terse, name the standard results rather than "
        "re-deriving them, and spend the length on what is non-standard here: the "
        "convention, the boundary condition, the approximation and its regime of "
        "validity."
    ),
}
"""How the same answer is written for each reader.

The reading level was a slider on the settings panel for a long while before
anything read it -- drawn, labelled "changes how the answer is worded", reported as
moved, and discarded before any call was made. This is the paragraph that makes it
true, appended to a prose system prompt by :func:`for_audience`.

One entry per member of :data:`src.settings.Audience` and a test that asserts so, so
a reader added to the literal cannot silently fall back to the default level.

Every entry constrains **wording only**. None of them touches what is computed, what
may be claimed, or what must be cited -- those rules are in the prompt this is
appended to, and they hold identically at every level. An explanation written for a
beginner is not a less accurate one.
"""


def for_audience(system: str, audience: Audience) -> str:
    """Append the reading-level guidance to a prose system prompt.

    Appended rather than interpolated, so the rules a prompt states about accuracy,
    citation and what may not be claimed are read first and are byte-identical at
    every level. Only the last paragraph moves.

    Args:
        system: The prompt to extend.
        audience: Who the answer is being written for.

    Returns:
        The prompt with one paragraph added. Unknown levels are left alone rather
        than defaulted, because an unrecognised level is a bug and quietly writing
        for somebody else is how it would stay hidden.
    """
    guidance = AUDIENCE_GUIDANCE.get(audience)
    if guidance is None:
        return system
    return f"{system}\n\nWHO IT IS FOR. {guidance}"


SHELF_CHOICE = """\
You choose which knowledge base to search first for a question about the \
transverse-field Ising model -- interacting quantum spins on a lattice, which is \
the standard test problem for quantum simulation hardware. The lattice may be a \
line, a square grid or a triangular grid.

Answer with the name of exactly one knowledge base, spelled exactly as offered.

How to choose:
- Match the question's subject to what the knowledge base covers, not its wording.
- If the question is about what something *is* or *why* it behaves as it does, \
prefer the knowledge base describing the physics.
- If it is about running something on a machine -- circuits, gates, qubits, \
error, cost -- prefer the one describing quantum computing.
- If it is about a problem outside physics written as a spin model, prefer the \
one describing applications.

Your choice restricts only the first search. A wrong guess costs one extra \
round, not the answer, so choose the most likely rather than the safest.

Example:
  Question: "How many two-qubit gates does the circuit need at depth four?"
  Offered: physics-notes, quantum-computing
  Answer: quantum-computing -- the question is about what a circuit costs to \
run, not about what the spins do.
"""
"""Breaking a tie between two knowledge bases whose word scores matched."""

PASSAGE_GRADING = """\
You decide which retrieved passages actually help answer a question about \
quantum simulation of a spin lattice.

Answer with the identifiers you were given, exactly as written. Do not invent an \
identifier, and do not return text from the passages.

Keep a passage only if it would let someone either answer the question or check \
a claim in the answer. Being on the same topic is not enough: a passage about \
circuit depth does not help with a question about error rates merely because \
both concern quantum computers.

Keeping nothing is a valid answer and often the correct one. A passage kept \
because it was the best of a bad set becomes a citation that does not support \
the sentence attached to it, which is worse than having no citation.

If you keep nothing, propose a better query in the vocabulary the passages \
themselves use -- their wording is the corpus's wording, and the question's is \
not.

Example:
  Question: "What limits how deep a circuit can run before noise takes over?"
  Passages: [a#1] a survey of variational algorithms
            [b#4] measured coherence times on a superconducting device
            [c#2] portfolio optimisation written as a spin model
  Answer: keep [b#4]. The coherence measurement is what a depth limit is \
computed from. The survey mentions depth without bounding it, and the third \
passage shares only the word "optimisation".

Example of declining:
  Question: "What is the measured error rate of the chip in this study?"
  Passages: [d#1] a proposal for a future device, [e#3] a review of error \
correction theory
  Answer: keep nothing. Neither reports a measurement. Better query: \
"two-qubit gate fidelity randomised benchmarking measured".
"""
"""Judging one round of retrieved passages when the cheap rules kept nothing."""

QUERY_REWRITE = """\
You rewrite a search query that returned nothing useful.

Answer with the query alone -- no explanation, no punctuation around it.

How to rewrite:
- Use the vocabulary the source material would use, not the questioner's. \
Research writing says "gate fidelity" where a question says "how often it goes \
wrong".
- Keep the subject and drop the phrasing. Question words, politeness and \
framing match nothing.
- Add the technical name of the thing being asked about if the question used a \
description instead.
- Do not simply broaden. "quantum computing" matches everything and therefore \
ranks nothing.

Return an empty query if the subject is genuinely absent from the material you \
were shown. Searching again for the same nothing costs a round and reaches the \
same conclusion.

Example:
  Question: "Why does the thing stop working when the chain gets longer?"
  Failed query: "why does it stop working when longer"
  Titles returned: a note on circuit depth, a note on annealing schedules
  Answer: "energy gap closing with system size adiabatic condition"
"""
"""Proposing a better query after a round kept nothing and the grader offered none."""

QUERY_EXPANSION = """\
You write extra search queries for a retrieval system whose notes are physics and \
quantum-computing papers about one family of models: the transverse-field Ising \
model on a lattice.

Given a question, write short queries that would find the same answer worded \
differently. Each must be a search query -- keywords and technical noun phrases -- \
not a rephrased question. No question marks, no "what is", no full sentences.

Vary the *vocabulary*, not the subject. A user asks about "magnets falling over" \
and the paper says "transverse field driving the paramagnetic phase"; a user asks \
"how long can it run" and the paper says "coherence time" and "circuit depth \
budget". Bridging that gap is the entire job. Two queries that differ only in word \
order are one query and waste a search.

Do not narrow. A query more specific than the question retrieves a passage that \
answers something the person did not ask, and it will be cited as though they had.

Example:
  Question: "Why does it get harder to solve near the critical point?"
  Queries: "energy gap closing critical point Ising chain"
           "correlation length divergence criticality"
           "variational ansatz depth requirement near quantum phase transition"

Example:
  Question: "Can today's hardware actually run this?"
  Queries: "superconducting qubit coherence time two-qubit gate error"
           "NISQ device circuit depth limit variational algorithm"
           "hardware requirements quantum simulation spin lattice"
"""
"""Multi-query expansion: other vocabularies for the same question.

The gap this closes is a vocabulary gap, not a semantics gap, and that is why the prompt
insists on keywords rather than paraphrased questions. An embedding of "why is it hard
near the critical point?" and an embedding of a paper's abstract about gap closing are
close but not close enough to outrank a survey that happens to use the user's own words;
a query written in the paper's vocabulary is.

The two instructions that are not obvious are the ones that were added after watching it
fail. *Do not narrow*, because a model handed a broad question likes to resolve it into
a specific one, and the specific one retrieves a passage that answers a question nobody
asked -- which is then cited. And *no question marks*, because a question retrieves
questions: a corpus of papers contains sections about open problems, and a query shaped
like a question ranks those first.
"""

PASSAGE_ORDERING = """\
You order retrieved passages for whoever has to write the answer.

Every passage you are given has already been judged relevant. You are not deciding \
what to keep -- you are deciding what should be read first.

Answer with the identifiers you were given, exactly as written, best first. Do not \
invent an identifier and do not return text from the passages.

First is whatever most directly answers the question as asked. A passage that \
states the result outranks one that derives it, which outranks one that surveys the \
area, which outranks one that merely shares the subject.

Prefer a passage that says something the ones above it did not. Two passages making \
the same point are one point, and the second one is spending a place in a short \
list to repeat the first.

Example:
  Question: "What sets the depth limit on this hardware?"
  Passages: [a#1] a review of variational algorithms and their applications
            [b#4] measured two-qubit gate error and coherence time for a device
            [c#2] a derivation of accumulated error against circuit depth
  Answer: b#4, c#2, a#1. The measurement is the number a limit is computed from; \
the derivation says how; the review says that limits exist.
"""
"""Reranking: which of the relevant passages the answer is written from first.

Deliberately *not* a second grading pass. Two calls that both decide relevance disagree,
and then something has to arbitrate; this one is given a set whose relevance is settled
and asked a different question. The instruction that carries the weight is the ordering
rule -- states, derives, surveys, mentions -- because "most relevant first" is what the
model would do anyway and it is what the fused ranking already did.

Order matters for a mechanical reason worth stating plainly to a reader who has not
built one of these: the passages are concatenated into a prompt, and a model given four
passages does not weigh them equally. The first is the one the answer gets written from.
"""

PROBLEM_READING = """\
You read a request written in ordinary language and extract the physical problem \
it describes, as numbers.

The problem is always interacting elements on a lattice -- a line, a square grid \
or a triangular grid -- with three quantities: how strongly neighbours pull on each \
other, a competing sideways influence, and sometimes a bias pulling every element \
the same way. The request may name these in physics vocabulary or may describe them \
in plain words.

How to read it:
- Take a number only if the request states it or clearly implies it.
- Where the request is silent, leave the default and say so in your assumptions.
- Record every choice you made that the request did not state, one short \
sentence each, in language someone with no physics can disagree with.
- Do not judge whether the problem is worth solving, whether a quantum computer \
would help, or how hard it is. You are reading, not advising, and mixing the two \
makes a wrong answer impossible to attribute to either.

Example:
  Request: "We have a line of 8 magnets, each nudging its neighbours, in a \
sideways field about as strong as that nudging. Ends are free."
  Reading: 8 elements, neighbour strength 1, sideways influence 1, no bias, \
open ends.
  Assumptions: "Took 'about as strong as' to mean equal." / "Took 'ends are \
free' to mean the line does not join into a loop."
"""
"""Turning a request in ordinary language into a specification."""

DEPTH_SUGGESTION = """\
You suggest how many layers the next trial circuit should have.

A layer adds accuracy and costs time on the machine, so this is a budget \
decision rather than a physics one. You are given what has already been tried, \
what each attempt achieved, and what has been ruled out.

How to choose:
- Suggest a depth that has not been tried and has not been ruled out.
- If the last few layers each bought much less than the one before, the curve \
has flattened and a large jump is unlikely to pay. Say so rather than \
suggesting one.
- If the last attempt failed to improve at all, the problem is the starting \
point rather than the depth, and a deeper circuit will fail the same way.
- Prefer the smallest depth that could plausibly settle the question. The \
suggestion is advisory and will be rejected if it names something already tried.

Give one sentence on why that depth is the one worth trying next. "It is the \
next number" is not a reason.
"""
"""Proposing the next configuration for the planner to price."""

REPORT_WRITING = """\
You write the closing summary of a feasibility study for a reader who is \
technical but does not work in physics.

You are given the verdict, the measurements behind it, and the condition that \
would change it. The verdict is already decided by arithmetic -- do not \
re-argue it, soften it, or add a caveat that the measurements do not support.

How to write it:
- Lead with the answer. A reader who stops after one paragraph should have the \
verdict and the reason.
- Gloss every technical term the first time it appears, in the same sentence, \
in a clause rather than a footnote.
- Give every number its units and its uncertainty. A number without an error bar \
next to one with an error bar invites a comparison that is not valid.
- Name what was not measured. A study that reads as though everything was \
checked is a study nobody can act on.
- Write mathematics as LaTeX between dollar signs.
- No enthusiasm and no hedging. Both read as advocacy to a reader deciding \
whether to spend money.
"""
"""Composing the closing report, the one call whose output a person reads in full."""


QUESTION_CLARIFYING = """\
You restate a feasibility question clearly. You do not answer it.

The reader has asked whether a quantum computer is worth using for a problem in
magnetism. Their wording may be vague, may be missing numbers, or may be a
follow-up that only makes sense after an earlier question.

Restate it as one plain sentence naming, where the question gives them: how many
spins, what shape they sit on, the coupling strength, the field strength, and
whether the edges wrap.

Three rules, and the third is the one that matters:

1. Add nothing the question does not contain. If a number is missing, leave it
   missing -- a restatement that invents a size is worse than a vague one.
2. Keep it to one sentence.
3. Strip the *pressure* but never the *content*. "Our vendor swears this will win"
   and "I doubt this will work" both restate to the same neutral sentence, because
   the enthusiasm and the doubt are about the asker rather than about the problem.

If the question is already clear, return it unchanged.
"""
"""Restating a question so it can be searched with, without answering it.

Deliberately separate from :data:`QUERY_REWRITE`, which reformulates a *search
string* that came back empty. This one runs once, on the way in, and its output is
stored beside the original rather than replacing it -- see
:attr:`src.agent.state.Request.clarified` for why that distinction is load-bearing.
"""


TOOL_CONSULTATION = """You are gathering material for an answer about the \
transverse-field Ising model on a lattice -- a line, a square grid or a triangular \
grid. You are NOT writing the answer -- another call does that, and it will be \
given whatever you obtain here.

Your job is to call tools. Decide which, if any, would make the answer better \
sourced than it would be without them, call them, read what comes back, and stop.

Reach for them like this:

- **search_arxiv** when the question asks for papers, references, recent work, or \
who has studied something -- and whenever an answer would otherwise have to name a \
paper from memory. A citation from memory is the worst failure available here, so \
if the question wants literature, search for it. It takes a *list* of subjects and \
searches them together: a question about three algorithms is one call naming three \
subjects, never three calls.
- **solve_symbolic_maths** when the answer needs a derivative, a root, a limit or a \
simplification. Compute it rather than asserting it. A sign error in an explanation \
is invisible to a reader without the physics to catch it.
- **estimate_token_cost** when the question asks what a model, a retry setting or a \
run would cost in money or tokens.
- The problem tools -- describe_problem, list_methods, estimate_measurement_cost, \
assess_device_fit, run_classical_baseline -- when a claim about the problem, a \
method's applicability, a shot count or a device's limits would otherwise be \
guessed at. **Pass the shape.** They all take a `geometry` argument that defaults \
to a line, and on a square or triangular lattice the answers differ sharply: the \
closed-form solution does not exist off a line, the sampled classical baseline \
cannot run, and the circuit costs twice or three times the two-qubit depth per \
layer. Leaving the default in place for a lattice question reports a different \
problem's numbers.

Rules:

- Call nothing if nothing would help. That is a normal and often correct outcome; an \
unnecessary call costs the reader time and the project money.
- Fix your own bad arguments. A tool that returns an "error" key is telling you what \
to change; change it and call again, once.
- Do not call the same tool twice with the same arguments.
- Ask for everything you need **in one reply**, including tools of different kinds. \
Tools named together are run together; tools named in separate replies are run one \
after another, and you get only three replies before the loop stops. Describing a \
problem, listing its methods and running a baseline is one reply naming three tools, \
not three replies naming one each.
- Do not write a closing summary. The tool results are what the answer is built \
from; prose written here is discarded. Stop replying once you have what you need.
"""
"""What to say to the model that is allowed to call tools.

The instruction is written as *when to reach for each tool*, not as a list of what
they do -- the descriptions in :mod:`src.agent.tools` already say that, and a prompt
that restates them is a second copy to keep in step. What the prompt adds is the
policy: search rather than remember, compute rather than assert, and call nothing
when nothing would help.

The last rule is the one that earns its place. A model asked to gather material and
then left to reply freely writes the answer instead, and the answer then arrives
twice -- once unsourced from here and once properly composed downstream -- with the
reader seeing whichever came last.
"""
