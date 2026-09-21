"""The questions offered on the chat page, in a module that can be imported.

They used to live inside the page itself, which meant nothing could check them: a
Streamlit page draws when it is imported, so a test or a script that wanted to know
what the openers were would have had to render the application to find out. The
claim being made about them -- that between them they exercise the agent's
different paths rather than six variations of one path -- was therefore a claim
nobody could verify.

Here it can be verified two ways. :mod:`tests.test_ui_starters` checks offline that
they route to more than one place, and ``make live-check`` asks them for real and
prints the trajectory each one took.

**Chosen to take different routes, not to read well together.** One is answered
from the notes, one needs a computation and a plot, one is about the model's own
subject, one is the kind of open question that leaves the corpus behind and sends
the loop outside it, and one asks what any of it is for outside a physics
department. A set of six openers that all end in one search would show a visitor a
search engine.

Each is phrased to name what it wants, which is not politeness but how the router
works: "plot the magnetisation" names an observable and asks for a curve, so it
routes to a computation, while "tell me about magnetisation" names no quantity and
gets a search. A vague opener would show a new user the clarification path first,
which is the least interesting thing this agent does.

None of them asks the assistant to describe itself. "Who are you?" is a route and it
works when typed, but spending one of six openings on it would advertise the
machinery ahead of the physics.
"""

from __future__ import annotations

STARTERS: dict[str, str] = {
    "curious": "What makes the quantum Ising chain model 'quantum'?",
    "computing": "What does the quantum Ising chain teach us about building quantum technologies?",
    "student": "Why does the energy gap close exactly at h = J?",
    "practitioner": (
        "Plot the magnetisation against the transverse field, and mark where it changes fastest."
    ),
    "critical": "What is quantum criticality, and how would I see it in a finite chain?",
    "applied": (
        "How can the Quantum Ising Chain Model be applied to real-world business problems?"
    ),
}
"""Opening questions, keyed by who is being invited to ask.

The keys are labels for whoever is reading this file; the buttons show the
questions themselves.
"""

QUESTIONS: tuple[str, ...] = tuple(STARTERS.values())
"""The same questions as a sequence, for callers that only want to ask them."""
