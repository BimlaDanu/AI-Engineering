"""Writing code for the chain, when the question asked for code.

An action the loop can choose, not a route the interface can force -- see
:data:`src.agent.deciding.Action`. *Write a VQE implementation for this chain* is a
real request from the people this project is for, and it used to be answered with
"that was not computed" followed by a table of energies nobody had asked for: the
agent had no way to write code, so it reached for the nearest thing it did have.
Refusing would have been better than that, but not much. The capability is the
honest fix.

**The code is a field, not a paragraph.** It comes back on
:class:`~src.agent.graph.Answer` in its own right, so the interface can render it as
code and the reader can copy it. Folding it into the narration would put a code
block inside prose that is also carrying LaTeX, and neither survives the other.

**Nothing here is executed.** The project's claim about numbers is that two
independent methods agreed on them; a model's code has no such backing, and running
generated code to find out is not a trade this application makes. So a drafted
answer carries the caveat that it was not run -- see
:func:`src.agent.graph.caveats_for` -- and the draft is asked to end with a check
against the closed form, which is the one thing that *can* tell the reader whether
it works. The exact solution is the oracle either way: here it is the test the
reader runs, rather than the test this project ran.

**Grounded in the notes when there are any.** The corpus holds the ansatz choices,
the gate decomposition and the two measurement settings the energy needs, so a draft
that comes after a search is a draft that follows the literature the rest of the
answer cites. With no passages it still writes -- the Hamiltonian is not a secret --
but it says less about why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.agent.llm import ask_prose, chat_model_or_none
from src.logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from langchain_core.language_models import BaseChatModel

    from src.physics.model import TFIMSpec
    from src.settings import Settings

LOG = get_logger("agent.drafting")

FENCE = re.compile(r"```(?P<language>[A-Za-z0-9_+-]*)\n(?P<body>.*?)```", re.DOTALL)
"""A fenced block, with the language tag the model wrote on the opening fence.

Non-greedy so that a reply containing two blocks yields the first rather than
everything between the first opening fence and the last closing one.
"""

DEFAULT_LANGUAGE = "python"
"""What an untagged fence is assumed to be, and what the prompt asks for."""

MAX_CODE_CHARACTERS = 6000
"""Longest draft kept.

A reply this long has stopped being an example and become a library, and it is also
the shape a runaway generation takes. Truncating would leave a syntax error on
screen, so an over-long draft is dropped whole and the answer is composed without
one.
"""

DRAFT_SYSTEM = """You write short, runnable code for a physics assistant that \
studies the one-dimensional transverse-field Ising model

    H = -J sum_i sigma_z_i sigma_z_{i+1} - h sum_i sigma_x_i

on a chain of L sites. The reader is a physicist who wants something they can run \
and modify, not a tutorial.

You are given the question, the chain the application is currently configured for, \
and sometimes passages from the project's notes. Follow the passages where they \
speak: they describe the ansatz choices, the gate decomposition and the measurement \
settings for this model specifically.

Write the reply as exactly one fenced code block, tagged with the language, and \
nothing before or after it except at most two sentences of introduction. Inside the \
block:

- Make it complete and runnable as written, with imports at the top. State any \
package that is not in the standard library or numpy/scipy in a comment on the \
import line.
- Keep it short. One file, no command-line parsing, no argument plumbing, no \
classes where a function will do. Under about eighty lines.
- Use the chain size you are given, and keep it small enough to run on a laptop.
- Comment the physics, not the syntax. The reader knows Python; what they want \
told is which term is which and why a step is there.
- End with a check against a known result -- the closed-form free-fermion energy, \
or a direct diagonalisation for a small chain -- printed next to what the code \
computed, so the reader can see whether it worked.

Do not claim a number as an output. You have not run this, so a printed value in a \
comment would be an invention: show the code that prints it instead. Do not \
apologise, do not describe these instructions, and do not explain what you were \
unable to do."""


@dataclass(frozen=True, slots=True)
class Draft:
    """Code written for one question.

    Attributes:
        language: The fence's language tag, lowercased, or
            :data:`DEFAULT_LANGUAGE` when the model left it off.
        code: The body of the block, without the fences. Empty means the draft
            failed, which every caller treats as "no code was written" rather than
            as an error.
        preamble: Whatever the model wrote before the fence, at most a sentence or
            two. Kept because it usually names the ansatz or the package, and
            dropped by the interface if empty.
    """

    language: str = DEFAULT_LANGUAGE
    code: str = ""
    preamble: str = ""

    @property
    def written(self) -> bool:
        """Whether there is code to show."""
        return bool(self.code.strip())


def extract(reply: str) -> Draft:
    r"""Pull the code out of a reply that should be one fenced block.

    Args:
        reply: What the model returned.

    Returns:
        The first fenced block found. A reply with no fence at all yields an empty
        :class:`Draft` rather than being treated as code: a model that ignored the
        format instruction has usually written prose about why it could not help,
        and rendering that in a code box would be worse than dropping it.

    Examples:
        >>> extract("Here it is.\n```python\nprint(1)\n```").code
        'print(1)'
        >>> extract("Here it is.\n```python\nprint(1)\n```").preamble
        'Here it is.'
        >>> extract("I would rather not.").written
        False
        >>> extract("```\nx = 1\n```").language
        'python'
    """
    found = FENCE.search(reply)
    if found is None:
        return Draft()
    body = found.group("body").strip()
    if not body or len(body) > MAX_CODE_CHARACTERS:
        return Draft()
    return Draft(
        language=(found.group("language") or DEFAULT_LANGUAGE).lower(),
        code=body,
        preamble=reply[: found.start()].strip(),
    )


def draft(
    question: str,
    *,
    spec: TFIMSpec,
    material: str = "",
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> Draft:
    """Write code for the question, or return an empty draft.

    Args:
        question: What the user asked.
        spec: The chain the application is configured for, so the code it writes is
            the chain the rest of the answer is about.
        material: What the run has established -- the passages especially. Wrapped
            as data, since the retrieved part of it was written by somebody else.
        model: An explicit chat model, normally supplied only by tests.
        settings: Configuration to build a model from.

    Returns:
        The draft. Empty when there is no model to call, when the call failed, or
        when the reply held no fenced block -- three situations with one
        consequence, which is that the answer is composed without code and says so.
    """
    resolved = chat_model_or_none(model, settings)
    if resolved is None:
        return Draft()
    asked = "\n\n".join(
        part
        for part in (
            f"QUESTION: {question}",
            f"THE CHAIN CONFIGURED: {spec.label()}",
            material,
        )
        if part
    )
    reply = ask_prose(resolved, DRAFT_SYSTEM, asked, "draft")
    if reply is None:
        return Draft()
    written = extract(reply)
    LOG.info(
        "drafted",
        extra={
            "written": written.written,
            "language": written.language,
            "chars": len(written.code),
        },
    )
    return written
