"""Tests for the drafting action.

The one thing this module cannot test is whether the code a model writes is any
good -- nothing here runs it, by the same policy the application follows. What it
can test is everything around that: which replies become a draft and which are
dropped, what the model is shown, and that the failures all come back as one empty
:class:`~src.agent.drafting.Draft` rather than as three different exceptions.

Offline throughout: an autouse fixture makes building a real chat model raise, so
a test that forgets to pass a fake gets the no-model path instead of a live call
against the key in ``.env``.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from src.agent.drafting import (
    DEFAULT_LANGUAGE,
    MAX_CODE_CHARACTERS,
    Draft,
    draft,
    extract,
)
from src.physics.model import TFIMSpec

SPEC = TFIMSpec(n_sites=8, coupling=1.0, field=1.0)

FENCED = """Uses a hardware-efficient ansatz.

```python
import numpy as np

print(np.zeros(2))
```
"""


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an accidental live model call impossible for the whole module."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test tried to build a real chat model")

    monkeypatch.setattr("src.agent.llm.build_chat_model", refuse)


class Replies:
    """A chat model that returns one fixed reply, recording what it was shown.

    Attributes:
        reply: What it will answer with. ``None`` stands for a call that failed,
            which reaches :func:`~src.agent.drafting.draft` as a ``None`` from
            :func:`src.agent.llm.ask_prose`.
        shown: Every human message it received, so a test can assert on the
            material rather than trusting that it was passed.
        systems: Every system prompt it received.
    """

    model_name = "test/model"

    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.shown: list[str] = []
        self.systems: list[str] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Record the call and answer with the fixed reply."""
        self.systems.append(str(messages[0].content))
        self.shown.append(str(messages[-1].content))
        if self.reply is None:
            raise RuntimeError("the provider was unreachable")
        return AIMessage(
            content=self.reply,
            usage_metadata={"input_tokens": 40, "output_tokens": 12, "total_tokens": 52},
        )


# --------------------------------------------------------------------------
# What counts as code
# --------------------------------------------------------------------------


def test_the_fenced_block_becomes_the_code_and_the_rest_becomes_the_preamble() -> None:
    written = extract(FENCED)
    assert written.code.startswith("import numpy")
    assert written.code.endswith("print(np.zeros(2))")
    assert written.preamble == "Uses a hardware-efficient ansatz."
    assert written.language == "python"


def test_the_fences_themselves_are_not_part_of_the_code() -> None:
    # st.code renders what it is given verbatim, so a stray fence would be shown
    # as a line of the program.
    assert "```" not in extract(FENCED).code


def test_only_the_first_of_two_blocks_is_kept() -> None:
    # Non-greedy matching, and the test that proves it: a greedy pattern returns
    # the two programs with a fence line between them, which is not runnable.
    reply = "```python\nfirst()\n```\nand then\n```python\nsecond()\n```"
    assert extract(reply).code == "first()"


def test_a_reply_with_no_fence_is_not_treated_as_code() -> None:
    # A model that ignored the format instruction has usually written prose about
    # why it could not help, and prose in a code box is worse than no code.
    assert not extract("I would rather not write that.").written


def test_an_untagged_fence_is_assumed_to_be_the_language_that_was_asked_for() -> None:
    assert extract("```\nx = 1\n```").language == DEFAULT_LANGUAGE


def test_a_tag_in_capitals_still_matches_a_highlighter() -> None:
    assert extract("```Python\nx = 1\n```").language == "python"


def test_an_empty_fence_is_dropped_rather_than_shown_as_a_blank_block() -> None:
    assert not extract("Here:\n```python\n\n```").written


def test_a_draft_that_became_a_library_is_dropped_whole() -> None:
    # Truncating would leave a syntax error on screen, which reads as a bug in the
    # program rather than as a limit in the application.
    long = "x = 1\n" * MAX_CODE_CHARACTERS
    assert len(long) > MAX_CODE_CHARACTERS
    assert not extract(f"```python\n{long}```").written


def test_an_empty_draft_is_the_shape_every_failure_takes() -> None:
    assert Draft().written is False
    assert Draft().language == DEFAULT_LANGUAGE
    assert Draft().code == ""


def test_a_draft_is_immutable() -> None:
    with pytest.raises(AttributeError):
        Draft().code = "print(1)"  # type: ignore[misc]


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------

ASKED = "Write a VQE implementation."


def drafted(model: Replies | None = None, material: str = "") -> Draft:
    """Run the drafting action against a fake model.

    :class:`Replies` implements the two methods this path uses rather than
    subclassing ``BaseChatModel``, which is a pydantic model with a large surface;
    the cast lives here so it is written once.
    """
    return draft(
        ASKED,
        spec=SPEC,
        material=material,
        model=model,  # type: ignore[arg-type]
    )


def test_the_model_is_told_the_question_and_the_chain_it_is_writing_for() -> None:
    # The chain in the settings, not one the model invents: the code is meant to be
    # about the same chain the rest of the answer is about.
    model = Replies(FENCED)
    drafted(model)
    shown = model.shown[0]
    assert ASKED in shown
    assert SPEC.label() in shown
    assert "L=8" in shown


def test_the_passages_reach_the_draft_so_it_follows_the_notes() -> None:
    model = Replies(FENCED)
    drafted(model, material="FROM THE NOTES: the ansatz alternates ZZ and X layers.")
    assert "alternates ZZ and X layers" in model.shown[0]


def test_the_draft_is_asked_to_end_with_a_check_against_a_known_result() -> None:
    # The one thing that can tell a reader whether unrun code works, so it is part
    # of the instruction rather than left to the model.
    model = Replies(FENCED)
    drafted(model)
    assert "check against a known result" in model.systems[0]


def test_with_no_model_there_is_no_draft_and_no_error() -> None:
    # How the whole offline suite runs, and a supported mode rather than a fault.
    assert not drafted().written


def test_a_call_that_failed_comes_back_as_no_code_rather_than_as_an_exception() -> None:
    assert not drafted(Replies(None)).written


def test_a_model_that_wrote_only_prose_comes_back_as_no_code() -> None:
    assert not drafted(Replies("I cannot help with that.")).written


def test_a_written_draft_carries_the_code_the_model_wrote() -> None:
    written = drafted(Replies(FENCED))
    assert written.written
    assert "print(np.zeros(2))" in written.code
