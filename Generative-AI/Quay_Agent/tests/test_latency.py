"""The changes made because the chat took half a minute to answer.

Every test here guards a *shortcut*, and a shortcut is exactly the kind of change
that is right when it is written and wrong two months later. Each one therefore
pins the property that made the shortcut safe rather than the speed it bought:
timings belong in ``scripts/time_a_question.py``, which a person runs and reads,
because a test that asserts a wall-clock number fails on a slow laptop and passes on
a fast one that has quietly broken the same thing.

Four shortcuts:

**The depth ladder stops at its first refusal.** Every reason a configuration is
refused grows with depth, so once one is refused nothing deeper can be afforded --
and the loop used to climb on anyway, spending a model call per rung to be told the
same arithmetic again.

**A node reports itself with the state it produced.** The chat page draws the answer
when the scribe reports, so a hook handed the state as it was *before* the node ran
would draw an empty card.

The third shortcut -- searching every phrasing of a question at once instead of one
after another -- is tested in ``tests/test_rag_retrieve.py``, beside the fake store
that makes it testable without embedding anything.

The fourth -- **one consultation round asks for its tools together, and one arXiv
search carries every subject** -- is tested in ``tests/test_llm.py`` and
``tests/test_tools.py``, beside the fake model and the fake index that make them
testable without a provider. Both use a ``threading.Barrier`` rather than a
stopwatch, for the reason at the top of this file: a barrier of three parties can
only be crossed by three calls genuinely in flight, so it proves the property on a
loaded machine as well as an idle one. A prompt rule asking the model to batch its
tool calls was tried first and measured worse: a prompt cannot fix what the
signature forbids.
"""

from __future__ import annotations

from src.agent.graph import DEPTH_LADDER, _refused_above, run_campaign
from src.agent.model_selection import ModelPool
from src.agent.state import CampaignState, RuledOut, new_campaign
from src.agent.state import Request as CampaignRequest

QUESTION = "Is a ring of 8 spins worth running on real hardware?"


def _campaign(**kwargs: object) -> CampaignState:
    """Run one campaign offline.

    Args:
        **kwargs: Passed straight to :func:`src.agent.graph.run_campaign`.

    Returns:
        The finished campaign.
    """
    return run_campaign(
        QUESTION,
        shot_budget=10**9,
        models=ModelPool(offline=True),
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# The depth ladder stops where the arithmetic stopped it
# --------------------------------------------------------------------------


def test_nothing_is_refused_before_anything_has_been() -> None:
    fresh = new_campaign(request=CampaignRequest(text=QUESTION), shot_budget=10**9)
    assert _refused_above(fresh) > max(DEPTH_LADDER)


def test_the_shallowest_refusal_becomes_the_ceiling() -> None:
    state = new_campaign(request=CampaignRequest(text=QUESTION), shot_budget=10**9)
    state["ruled_out"] = (
        RuledOut(label="hva-p8", reason="too deep"),
        RuledOut(label="hva-p4", reason="too costly"),
    )
    assert _refused_above(state) == 4


def test_a_label_carrying_no_depth_excludes_nothing() -> None:
    # An unparseable label must not become a ceiling of zero, which would stop the
    # ladder before its first rung and report "nothing was affordable" about a
    # campaign that had never priced anything.
    state = new_campaign(request=CampaignRequest(text=QUESTION), shot_budget=10**9)
    state["ruled_out"] = (RuledOut(label="something-else", reason="unrelated"),)
    assert _refused_above(state) > max(DEPTH_LADDER)


def test_a_budget_that_affords_nothing_refuses_once_rather_than_five_times() -> None:
    # The behaviour the change is for. Priced at an accuracy nothing can afford,
    # every rung is unaffordable -- and the honest report is one refusal with the
    # arithmetic, not five of them saying the same thing at increasing depth.
    broke = _campaign(precision_per_site=1e-8)
    assert not broke["runs"]
    assert len(broke["ruled_out"]) == 1
    assert broke["verdict"] is not None


def test_a_generous_budget_still_climbs() -> None:
    # The other half of the same claim: the ceiling must not stop a ladder that has
    # somewhere to go. A change that made every campaign refuse once would pass the
    # test above and be worthless.
    rich = _campaign(precision_per_site=1e-1)
    assert len(rich["runs"]) > 1
    assert {run.depth for run in rich["runs"]} <= set(DEPTH_LADDER)


# --------------------------------------------------------------------------
# A node reports itself with the state it produced
# --------------------------------------------------------------------------


def test_a_node_is_reported_with_the_state_after_it_ran() -> None:
    # What the chat page depends on to draw the answer early. Were the state the one
    # from before the node, the report would be missing at the moment the page draws
    # it -- and the bug would look like a rendering fault, not a streaming one.
    seen: list[tuple[str, bool]] = []
    finished = _campaign(on_node=lambda node, so_far: seen.append((node, so_far["report"] != "")))
    assert seen, "no node reported itself"
    named = [node for node, _ in seen]
    assert named[0] == "recall"
    assert named[-1] == "remember"
    had_report = dict(seen)
    assert had_report["scribe"], "the report was not written yet when the scribe reported"
    assert finished["report"] != ""


def test_the_recorded_path_and_the_reported_nodes_agree() -> None:
    # Two callers of the same stream. A campaign whose progress display and whose
    # Pipeline trace disagreed about what ran would leave nobody able to say which
    # was right.
    path: list[str] = []
    reported: list[str] = []
    _campaign(visited=path, on_node=lambda node, _: reported.append(node))
    assert path == reported


def test_a_campaign_with_no_hook_still_runs() -> None:
    assert _campaign()["verdict"] is not None
