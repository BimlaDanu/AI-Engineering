"""The questions the chat page offers, in a module something can import.

They live here rather than inside the page because a Streamlit page runs when it
is imported, so a question written into one is a question no test can see. Here
the page test can check the two claims this set makes: that every
question is short enough to sit on a button, and that between them they take the
agent down different paths rather than one path twelve times.

The subject is the four near-term methods -- VQE, QAOA, variational imaginary
time and quantum annealing -- because that is what this project is for. The
transverse-field Ising chain is the instrument the four are tested on, not the
thing being asked about.

Each question is ordinary language rather than a specification. Reading a
sentence into a Hamiltonian is the agent's first job and the step where a
feasibility study most often goes quietly wrong, so it is the first thing worth
watching it do. A starter that handed over ``n_sites=12`` would skip it.
"""

from __future__ import annotations

STARTERS: dict[str, str] = {
    # The whole pipeline: read the sentence, pick a method, cost the circuit
    # against a machine, run the classical baseline, decide. The only starter
    # that opens every branch of the graph.
    #
    # It names three methods and no boundary, which is what makes it run rather than
    # only route: three names send it to the race, the length gives it a chain to
    # assess, and a sentence silent about the boundary is read as a line -- a ring on
    # the default machine costs sixteen routing moves per circuit and is refused at
    # every depth, so the planning loop would conclude without solving anything.
    "feasibility": "VQE, QAOA or VarQITE for a 10-spin chain?",
    # Names no method, no length and no field, and is answered anyway -- because
    # the conversation before it is in the prompt. Second on purpose: pressed
    # after the first it is the memory layer demonstrated in one click.
    "memory": "And if we double the circuit depth?",
    # The physics, and the reason the benchmark is interesting: at h = J the
    # energy gap closes and all four methods have their worst time.
    # "in this case" pointed at a question that may not have been asked: a button
    # cannot assume a turn before it.
    "criticality": ("Why is the quantum critical point at $h=J$ the hardest place to compute?"),
    # Depth is the knob every method shares, and the optimum is interior --
    # expressibility rises with depth, fidelity falls, and the answer is a curve.
    "circuit_depth": "How deep should the circuit be before noise wins?",
    # Asks to watch rather than to be told, which is the one thing a paragraph
    # answers badly. It sends the campaign down an extra branch of the graph that
    # runs all three methods on the chain named here and keeps their loss curves,
    # so the picture underneath is of this chain and not of a stock example.
    #
    # The field is left out and taken equal to the coupling, which is the critical
    # point this question used to name. Said in the assumptions either way, and four
    # words shorter on the button.
    "loss_curve": "Loss curves for VQE, QAOA, VarQITE on 8 spins?",
    # The same race on the other side of the transition: below criticality rather
    # than at it, so the two curve questions answer about different physics instead
    # of the same chain twice.
    #
    # It names its own length and ratio, and it did not always. It used to read
    # "which converges fastest and gets closest?" and inherit both from the press
    # before it -- the memory layer demonstrated in one click, which is a good
    # demonstration and a bad button: pressed first, it silently assessed an invented
    # six-element chain. A button has no "press me second". The memory demonstration
    # is still in the set, as `memory`, and deliberately off the buttons.
    #
    # It named a ring until the pricing was read back: on the default machine, a
    # linear one, closing a ring costs sixteen routing moves per circuit, and the
    # shallowest rung on the ladder is then refused on coherence -- which refuses
    # every rung above it too, so the button raced its three methods and then
    # concluded without ever running a configuration. The ratio is what puts this
    # question on the other side of the transition; the boundary was never the part
    # doing that work.
    #
    # The ratio is written without maths delimiters, and it is the only question in
    # the set where that is a display decision rather than a style one -- see
    # `LONGEST_ON_A_BUTTON`. The reader reads the sentence identically either way.
    "convergence": "Which converges fastest on 6 spins at h/J = 0.5?",
    # The QAOA translation step: a classical objective becomes a diagonal cost
    # operator and a mixer. This is the part a reader outside physics performs.
    "cost_and_mixer": "How does a scheduling cost become a QAOA mixer?",
    # Imaginary time is not unitary, so no circuit runs it directly. What is done
    # instead -- and what it costs in measurements -- is the method.
    "imaginary_time": "How can a circuit run imaginary time?",
    # Use cases for VQE: molecules are the target, this chain is the instrument
    # check. A starter that makes the agent say so is a starter worth having.
    "molecules": "Does VQE pay off on molecules but not on this spin chain?",
    # The fourth method. QAOA is a Trotterised anneal with the schedule learned
    # rather than assumed, and saying which you are buying is the answer.
    "annealing": "Why not just anneal it? What does QAOA add over annealing?",
    # The three methods introduced from nothing. The question a reader who knows none
    # of this arrives with, and the one the *explain* branch answers best -- prose,
    # from the corpus, with citations.
    #
    # It names no model, and does not need to. The chain is not a topic the answer
    # might wander onto: `EXPLAIN_SYSTEM` opens by stating it as the subject, and
    # `TEACHING_SHAPE` requires an *On this chain specifically* section under every
    # algorithm heading, giving the gate cost and which terms commute. Naming it in
    # the question as well bought nothing and cost twenty-seven characters, and
    # calling the three "quantum computing algorithms" cost eighteen more: the names
    # say which field they belong to.
    "methods_taught": "Could you teach me VQE, QAOA and VarQITE?",
    # Real machines: superconducting, trapped ion, neutral atom, annealer, and
    # which method each one suits. Answered from the corpus, with citations.
    # Names the method, the length and the depth. It used to ask about "these
    # methods", which points at a previous answer -- pressed first there were no
    # such methods, and the machines were priced against an invented default chain.
    # "Machines" rather than "quantum computers": nothing else in the sentence could
    # be a classical one, and it was the longest word on any button.
    "hardware": "Which machines run VQE on 12 spins at depth 4?",
    # Takes the `implement` branch rather than the feasibility fan-out: the ask
    # is a circuit, not a verdict.
    "implement": "Write the QAOA circuit for 8 spins, depth 3?",
    # A curve in the *field* rather than in an optimiser's epoch, which is a
    # different question answered by different machinery: the agent calls a tool
    # that solves the model exactly at every field in the range, and the levels are
    # drawn beneath the answer. It is also the starter that demonstrates the wall --
    # the solver is sealed from everything the agent imports and reaches it only as a
    # callable the interface lends to that one tool.
    "spectrum": "Can you plot the exact energy levels as the field is turned up?",
    # The same machinery asked for the hardest of the five curves. Three stacked
    # panels: a quantity with no feature at the transition, a first derivative that
    # bends, and a second that dips sharply -- which is what a continuous phase
    # transition looks like when you are only allowed to plot smooth things.
    "derivatives": ("Can you plot the ground-state energy and its first two derivatives vs $h/J$?"),
}
"""Opening questions, keyed by the capability each one exercises.

Twelve short ones rather than four long ones. The label *is* the question -- a
button reading "Anneal" that sent a different sentence would be a small lie -- so
the length limit falls on the questions, and the comment above each one says what
it is there to reach.
"""


LONGEST_ON_A_BUTTON = 50
"""Longest a question drawn as a button may be, in characters.

Tighter than :data:`LONGEST_BUTTON`, which bounds every question in the set, because a
button is not a line of prose: four of them to a row makes each box about a quarter of
the page, which wraps a label at roughly twenty-five characters. Two lines is about
fifty, and the line that matters is the third -- a question one word too long for two
lines puts that word alone on a line of its own, and eight buttons ragged like that
read as a wall rather than as a set of choices.

So the questions on the buttons are written to two lines. The ones that are only ever
typed are not, and :data:`LONGEST_BUTTON` still bounds those: a sentence nobody has to
fit in a box may spend the characters.

A question on a button carries no ``$...$`` either, which is not a style rule but the
same wrapping rule seen from the other side. Streamlit does render maths in a button
label, in KaTeX -- a larger serif face than the label around it, wider than the
characters it replaces and taller than the line it sits on. So a ratio written as maths
mismatched the seven buttons beside it and took the label onto the third line this
constant exists to prevent. Written plainly it costs four characters less and wraps
where the count says it will. Prose renders through ``st.markdown`` and keeps its
delimiters.
"""

LONGEST_BUTTON = 85
"""Longest a starter may be, in characters.

The label *is* the question and always will be -- a button reading "Anneal" that sent a
different sentence would be a small lie -- so the bound falls on the questions rather
than on the labels. Past this a button stops being a button and becomes a paragraph
nobody presses.

Eighty-five rather than the seventy-five it was, since the set went from twelve buttons
to eight: three rows of shorter questions and two rows of slightly longer ones take
about the same space, and the limit exists to bound the block rather than the sentence.
A question carrying inline maths -- ``$h=J$`` -- spends four of its characters on
delimiters that are not drawn.

This was briefly raised to 110 to admit one long question, and then not needed: the
question was shortened instead, from *...algorithms applied to the above model?* to
*...algorithms?*, once it was clear the prompt already forces the answer onto the chain
under every heading. Shortening the question was the better of the two moves and the
allowance came back out, which is the note worth keeping -- a ceiling raised for one
case is a ceiling raised for every case written afterwards.

The questions actually drawn as buttons are held to :data:`LONGEST_ON_A_BUTTON`, which
is half this, for a reason this figure cannot express: this one bounds a sentence, and
that one bounds how many lines it wraps to in a quarter-width box.
"""

METHOD_QUESTIONS: tuple[str, ...] = (
    "methods_taught",
    "cost_and_mixer",
    "imaginary_time",
    "hardware",
)
"""The subset the Lab's method tab quotes back, by key.

The Lab explains the four methods with a cartoon and a live comparison and then has
nowhere to send a reader who wants one of them in prose. It sends them here, printing
these questions verbatim so that a question shown on one page can be found on the
other. Keys and not sentences, because a question copied by hand is a question that
drifts, and a promise the application does not keep is worse than no promise.

**Every key here must be a key in :data:`OFFERED_KEYS`**, because the sentence above
the list on the Lab page calls them buttons. ``molecules`` and then ``annealing`` were
both on this list and are not, each for that reason: the Lab said "ask this over there"
about a question that is no longer over there. The page test now checks the
rule rather than leaving it to be rediscovered a third time.
"""

QUESTIONS: tuple[str, ...] = tuple(STARTERS.values())
"""Every question here as a sequence, for callers that only want to ask them.

All twelve, including the four not on a button. ``make live-check`` asks these, and
the four it would otherwise lose are among the most informative trajectories in the
set -- the full fan-out, a memory follow-up, a depth question, and a question that
invites the agent to talk about a system this project does not model. A question worth
watching the agent answer does not stop being worth watching because it is not on a
button.
"""

NOT_OFFERED: tuple[str, ...] = (
    "criticality",
    "memory",
    "circuit_depth",
    "molecules",
    "annealing",
    "spectrum",
    "derivatives",
)
"""Seven questions kept in the set and taken off the buttons, by key.

**Nothing about them changed except the display.** Each is still answered if it is
typed, still asked by ``make live-check``, and still reachable from a follow-up
suggestion. This is a decision about how much a reader is asked to choose between on
arrival, not about what the agent can do.

Thirteen buttons is four rows, and four rows of full sentences is a wall a visitor
reads none of. Eight is two rows, which is a set somebody actually picks from.

Which five, and the cost of each:

* ``criticality`` is the one that hurts to lose, and it lost on arithmetic rather
  than on merit. Four of these eight are fixed by the Lab quoting them, ``implement``
  is fixed by closing row one, and ``feasibility`` and the two races are what make the
  set show more than one route. That is eight. Criticality is answered inside several
  of the questions that remain, and typing it still works.

``feasibility`` was on this list and has been put back on the buttons. Taking it off
cost exactly what the note here predicted: five of the eight buttons then ran the same
``consult -> explain`` route, the ``plan -> solve -> analyse`` loop was reachable from
two of them, and pressing several in a row produced visibly the same shape of answer.
The full fan-out is the thing worth demonstrating, so it is back in first position.
* ``memory`` only works as the *second* thing pressed -- alone it names no chain -- so
  as a button it depended on the reader having pressed the right one first. The memory
  layer is demonstrated better by the 🧠 Memory panel, which is on every page.
* ``circuit_depth`` and ``molecules`` are the two whose subjects the other starters
  already reach: depth appears in the QAOA circuit question, and the applications
  shelf answers the molecule comparison when it is asked.
* ``annealing`` came off to make room for ``methods_taught``, which covers the same
  ground from further back: annealing is one of the four methods that question asks
  about, so a reader who presses the new button is told what annealing is on the way
  to the comparison, rather than being asked to already know.
* ``spectrum`` and ``derivatives`` are off the buttons for room rather than for
  merit: they are the two questions the exact field sweep was built for, and both
  produce a figure the reader can read a conclusion off. ``spectrum`` was briefly on
  the buttons in place of ``convergence`` and that was the wrong trade --
  ``loss_curve`` → ``convergence`` → ``criticality`` is a **sequence**: the second
  names no chain, no field and no length and is answered anyway, from what the first
  one established. Breaking a chain of three to display a fourth capability costs the
  memory layer its one-click demonstration, which is the harder thing to show.
  Both curve questions are still answered when typed, still asked by ``make
  live-check``, still checked by ``make curve-check``, and still reachable from a
  follow-up suggestion.
"""

OFFERED_KEYS: tuple[str, ...] = (
    # Row one is the pitch: the full feasibility verdict first -- the only starter
    # that opens every branch of the graph -- then what makes the chain hard, then the
    # two races that watch the methods rather than being told about them. Ordered so a
    # reader who presses only the first button has seen the thing the project is for.
    #
    # Every one of these eight stands on its own. None inherits a chain from the press
    # before it, because a button cannot say "press me second" and a visitor who
    # starts anywhere must get an answer about a problem they can see was theirs.
    # `convergence` races the same three methods as `loss_curve` and earns its place
    # by racing them on the other side of the transition -- a ring, below criticality.
    # The exact-curve questions and the memory follow-up are in the set and off the
    # buttons -- see `NOT_OFFERED`.
    "feasibility",
    "loss_curve",
    "convergence",
    "implement",
    # Row two is the prose half, and it is exactly the four the Quay Lab's method tab
    # quotes back by key. That is a constraint, not a coincidence: the Lab prints them
    # under a sentence calling them the Chat page's own buttons, so any of the four
    # taken off here makes the Lab point at something that is not there.
    "methods_taught",
    "cost_and_mixer",
    "imaginary_time",
    "hardware",
)
"""The buttons, in the order they are drawn.

Written out rather than derived from :data:`STARTERS`, because the order a reader
meets these in is a decision and the dictionary's order is a different one -- that is
grouped by the capability each question exercises, which is what a maintainer needs
and not what a visitor does.

Held to :data:`STARTERS` and :data:`NOT_OFFERED` by the page test, so a
question cannot be dropped from the display by being quietly left off this list.
"""

OFFERED: tuple[str, ...] = tuple(STARTERS[key] for key in OFFERED_KEYS)
"""The questions that appear as buttons, in the order they are drawn.

Looked up from :data:`STARTERS` rather than written out, so a button cannot end up
sending a sentence nothing else in the project has ever asked.
"""

REMOVED: tuple[str, ...] = (
    "Which business problems are these methods actually for?",
    "Our vendor says QAOA beats our solver. Are they right?",
)
"""Two starters that were offered and are not, recorded so they are not re-added.

Both were read as *feasibility* questions, which is what the intent reader does with
anything that does not clearly ask for prose or for code. Neither named a chain, so one
was invented for them, and both came back as a verdict on a two-spin default: correct
about that chain, and no answer at all to the question that was asked. A starter whose
reply is a confident assessment of a problem the asker never mentioned is worse than no
starter, because the reader has no way to see that the subject was substituted.

The subjects are still covered -- the business framing by the corpus's *applications*
shelf, and the vendor framing by the Evaluations page, which asks the same question
neutrally, as a vendor and as a sceptic and reports whether the verdict moved. That is
the measurement the second question was there to demonstrate, and it is a better
demonstration than one button.
"""
