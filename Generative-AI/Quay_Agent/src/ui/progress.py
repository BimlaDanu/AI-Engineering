"""What to show a person while a campaign is still running.

A campaign takes tens of seconds, and most of that is model calls made one after
another because each needs the one before it. That wait is not going away entirely,
so the question this module answers is what the person sees during it. One
motionless spinner for half a minute reads as a hang: there is no way to tell a
campaign that is climbing the depth ladder from one whose provider has stopped
answering, and the reader's only move is to reload and lose the answer.

So every node reports itself as it finishes, in words that say what was *done*
rather than which function ran. ``formalise`` is a node name; "read the chain out of
the question" is what happened. The node names stay available on the Pipeline trace
page, which is where somebody who wants them is looking.

Two rules the wording follows, both learned from watching it:

Past tense, because the line appears when the step is over. A label written in
the present -- "searching the notes" -- is shown at the moment searching stopped,
which is exactly one step out of date and reads as a stall on whatever comes next.

A repeated node counts itself. The depth ladder runs ``plan`` and ``solve`` once
per rung, and four identical lines look like a stuck loop rather than like four
configurations being tried. Numbering them is the difference between "this is
working" and "this is spinning".
"""

from __future__ import annotations

PHRASES: dict[str, str] = {
    "recall": "looked back over the conversation",
    "screen": "screened the question",
    "interpret": "worked out what kind of answer it wants",
    "formalise": "read the chain of magnets out of the question",
    "retrieve": "searched the notes for background",
    "converge": "raced the methods against each other",
    "baseline": "ran the classical baseline",
    "plan": "designed a configuration to try",
    "solve": "ran it",
    "analyse": "checked what came back",
    "consult": "decided which tools to call, and called them",
    "explain": "wrote the explanation",
    "implement": "wrote the code",
    "skeptic": "argued against its own verdict",
    "scribe": "wrote the report",
    "suggest": "worked out what to ask next",
    "remember": "wrote the exchange to memory",
}
"""One line per node, in the words of what it did.

Every node the graph can run is here. A name missing from this map would be shown
raw, which is why :func:`src.ui.progress.phrase` falls back to the name rather than
to nothing: an unfamiliar step is worth seeing, and a silently dropped one turns the
next step's line into a lie about what the campaign is doing.
"""

LADDER_NODES = frozenset({"plan", "solve", "analyse"})
"""The nodes that run once per rung of the depth ladder.

Held as a set rather than checked against a list of names at the call site, because
the numbering rule is about *these* nodes specifically: every other node runs at
most once per campaign, and a second line from one of those is a defect worth
seeing plainly rather than tidying into an ordinal.
"""


def phrase(node: str, seen: int = 1) -> str:
    """What to show for one finished node.

    Args:
        node: The node's name, as the graph reported it.
        seen: How many times this node has now run, counting this one. Only the
            ladder nodes use it; every other node runs once and reads oddly with an
            ordinal after it.

    Returns:
        A line for a person to read.

    Examples:
        >>> phrase("retrieve")
        'searched the notes for background'
        >>> phrase("solve", 3)
        'ran it (3rd configuration)'
    """
    said = PHRASES.get(node, node)
    if node in LADDER_NODES and seen > 1:
        return f"{said} ({_ordinal(seen)} configuration)"
    return said


def _ordinal(count: int) -> str:
    """Render a small count as an ordinal.

    Args:
        count: A positive number.

    Returns:
        ``"2nd"``, ``"3rd"``, and so on. Written out rather than pulled from a
        dependency because the only values it ever sees are the rungs of a ladder
        that stops well before the teens.

    Examples:
        >>> [_ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21)]
        ['1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st']
    """
    if count % 100 in (11, 12, 13):
        return f"{count}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(count % 10, "th")
    return f"{count}{suffix}"


class Trail:
    """The running commentary on one campaign, ready to hand to ``on_node``.

    Keeps the count each node has reached so that :func:`phrase` can number the
    ladder, and keeps every line written so far so a caller can redraw the whole
    list rather than only the newest line -- which is what a Streamlit rerun needs.

    Not a plain list of strings because the count has to survive between calls, and
    not a closure because a caller that wants to show the trail needs to read it
    back.
    """

    def __init__(self) -> None:
        """Start an empty trail."""
        self.lines: list[str] = []
        self._counts: dict[str, int] = {}

    def record(self, node: str) -> str:
        """Note that a node finished, and return the line for it.

        Args:
            node: The node's name.

        Returns:
            The line that was added.
        """
        self._counts[node] = self._counts.get(node, 0) + 1
        said = phrase(node, self._counts[node])
        self.lines.append(said)
        return said

    def ran(self, node: str) -> bool:
        """Whether a node ran at all.

        Args:
            node: The node's name.

        Returns:
            True once that node has reported itself. This is how a page knows
            whether it already drew a result, without keeping a second flag beside
            the trail that could disagree with it.
        """
        return node in self._counts

    def latest(self) -> str:
        """The newest line, or a starting line when nothing has run yet.

        Returns:
            What to put on the status header.
        """
        return self.lines[-1] if self.lines else "reading the question"


ANSWER_READY = "scribe"
"""The node after which there is an answer worth showing.

Named here rather than written as a bare string at the call site, because the fact
it encodes is not obvious: the report is complete when the scribe finishes, and the
two nodes after it -- the follow-up suggestions and the memory write -- add nothing
a reader is waiting for. Between them they are a model call and a file write, so
holding the answer back until the graph ends costs the reader seconds for work that
was never part of the answer.
"""
