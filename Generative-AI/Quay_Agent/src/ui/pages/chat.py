r"""Chat page: one thread, and every answer with the evidence behind it.

A question goes in, a campaign runs, and a verdict comes out alongside the trail
that produced it. Everything else in this interface reports on the project.

Offline by default. With no language model the campaign still formalises the
question, climbs the depth ladder, prices configurations against a real machine
and reaches a verdict by rule; what is lost is the prose, since the question is
read by pattern rather than by comprehension.

The thread is backed by :mod:`src.agent.memory`, so "what about twenty?" is
answered in the light of the question before it. Unreachable memory is reported
rather than quietly forgotten.

Three pictures are drawn from this project\'s own physics when the question asked
for them, never from anything a model wrote: the circuit it named
(:func:`src.agent.graph.circuit_the_question_named`), the exact field sweep
(:func:`src.ui.panels.field_sweep_asked_for`) and the convergence of all three
methods on this campaign\'s chain (:func:`src.agent.graph.converge_methods`).
Convergence curves carry a warning that every energy on them is exact, so they
compare methods on a perfect device.

A reply carries the verdict, its assumptions and its sources. The step-by-step
trail, the configurations run and refused, and the full report are on the
``Pipeline trace`` page -- a split by audience, with nothing removed.

Two rating buttons write one reading level from a closed set to
:mod:`src.agent.memory`, no free text, so nothing typed can be replayed. Up to
three follow-up questions come from a graph node reading what this run
established; each is screened by :func:`src.security.screen` before it becomes a
button, and a blocked question gets no suggestions at all.

Framings are measured on the ``Evaluations`` page rather than offered here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal, cast

import streamlit as st
from pydantic import SecretStr

from src.agent.graph import SearchTuning, run_campaign
from src.agent.middleware import CallBudget
from src.agent.model_selection import ModelPool, Tier
from src.agent.state import CampaignState
from src.agent.usage import Usage, metered
from src.hardware.devices import device_for
from src.logging_setup import get_logger
from src.physics.registry import field_sweep_bench
from src.settings import Settings, get_settings
from src.ui import access, panels, progress, session_log
from src.ui import setting as knob
from src.ui.starters import OFFERED

STARTERS_PER_ROW = 4
# Starter buttons to a row.
#
# Four, which fits the questions at their current length without wrapping past two
# lines. The labels are the questions themselves and always will be: a button reading
# "Ring" that sends a different sentence is a small lie, and the point of showing the
# question is that pressing it asks exactly that.


def ask(
    question: str,
    framing: str,
    chosen: knob.Setting,
    on_node: Callable[[str, CampaignState], None] | None = None,
    prose_sink: Callable[[str], None] | None = None,
) -> CampaignState:
    """Run one campaign at the settings knob's current position.

    Args:
        question: The problem, in the asker's own words.
        framing: Which voice it was asked in.
        chosen: Both knobs. Only the machine and the budget change a number here;
            the model half changes whether prose is written and by whom.
        on_node: Called with each node's name and the campaign after it, as each one
            finishes, so the page can say what is happening while it is still
            happening and draw the answer as soon as the node that wrote it is done.
            ``None`` runs it silently, which is what a test wants.
        prose_sink: Somewhere to send the explanation as it is written.

    Returns:
        The finished campaign. The route it took through the graph is left in session
        state for the Pipeline page, because a trace nobody asked for separately is
        the only kind that describes the question they actually asked.
    """
    path: list[str] = []
    st.session_state["path"] = path
    st.session_state["path_question"] = question
    # Metered so the Pipeline page can report what the answer cost. Offline it costs
    # nothing and says so, which is worth showing rather than hiding: "no model was
    # called" is a fact about how the answer was reached, not an empty panel.
    started = time.perf_counter()
    with metered() as meter:
        finished = _campaign(question, framing, chosen, path, on_node, prose_sink)
    elapsed = time.perf_counter() - started
    # One row per question, holding everything the Analytics page charts. Appended
    # here rather than assembled there because the path and the wall-clock exist
    # only at this moment: by the time a page reads session state, `path` has been
    # overwritten by the next question.
    asked: list[session_log.Asked] = st.session_state.setdefault("asked", [])
    asked.append(session_log.record(question, finished, path, meter.usage, elapsed))
    _report_lost_model(meter.usage)
    return finished


def _report_lost_model(usage: Usage) -> None:
    """Say so, on the page, when every model call failed.

    The gap this closes. A campaign computes and cross-checks every number before a
    model is consulted, so when the gateway refuses *all* of them -- a wrong key, a
    spent balance, no network -- what arrives is still a correct verdict with correct
    arithmetic, assembled by the deterministic paths. That is the graceful
    degradation this project is built around and it is worth having; what is not
    worth having is it happening **without saying so**. A reader cannot tell a
    model-written explanation from a composed one by looking, so silence here means
    the interface quietly misrepresents how its own answer was reached.

    Not raised as an error, because nothing on screen is wrong. Shown as a warning
    naming the likely causes in the order they occur, and only when *every* call
    failed: one failure in ten is what the retry layer is for, and reporting it
    would train a reader to ignore this box.

    Args:
        usage: What the meter recorded for the campaign that just finished.
    """
    if not usage.every_call_failed:
        return
    st.warning(
        f"{panels.WARNING} **The language model could not be reached** — all "
        f"{usage.failed_calls} attempts failed, and this answer was composed without "
        "one. Every number in it is still computed and cross-checked, so the verdict "
        "and the arithmetic stand; only the wording is the deterministic version. "
        "Usually a key that is wrong or out of credit, or no network. The terminal "
        "running the app logs the reason for each attempt."
    )


def _campaign(
    question: str,
    framing: str,
    chosen: knob.Setting,
    path: list[str],
    on_node: Callable[[str, CampaignState], None] | None = None,
    prose_sink: Callable[[str], None] | None = None,
) -> CampaignState:
    """Run the campaign itself, with the meter already wrapped around it.

    Args:
        question: The problem, in the asker's own words.
        framing: Which voice it was asked in.
        chosen: Both knobs.
        path: A list the graph appends each node's name to as it runs.
        on_node: Called with each node's name and the state after it, as each one
            finishes.
        prose_sink: Somewhere to send the explanation as it is written.

    Returns:
        The finished campaign.
    """
    return run_campaign(
        question,
        on_node=on_node,
        prose_sink=prose_sink,
        framing=cast(Literal["neutral", "vendor", "skeptical"], framing),
        visited=path,
        shot_budget=chosen.physics.shots,
        device=device_for(chosen.physics.device),
        models=pool(chosen),
        # Searched even with no language model, unless the reader has said not to.
        # The corpus half that needs no model -- keyword retrieval over the notes --
        # returns citations perfectly well on its own, and switching it off with the
        # model was costing the offline demonstration every citation it could have
        # had. Which shelf to search is then chosen by rule instead of by a model,
        # which is a slightly worse choice and a very much better outcome than no
        # sources at all.
        search_corpus=chosen.search.corpus,
        fetch_external=chosen.search.external,
        # The dials the settings panel offers, actually connected. They were
        # hard-coded here while the panel showed three unrelated ones, which is the
        # worst arrangement of the three available: a knob that changes nothing is
        # read as a knob that did not work.
        search_tuning=SearchTuning(
            passages=chosen.search.passages,
            vector_share=chosen.search.vector_share,
            shelves=chosen.search.shelves,
            rounds=chosen.search.rounds,
        ),
        suggest_followups=chosen.search.followups,
        # Two more dials that were drawn and discarded. The reading level changes
        # the wording of an explanation and nothing else; the accuracy changes the
        # shot arithmetic, and through it the verdict -- which is why it sits on the
        # Physics tab rather than beside the model settings.
        audience=chosen.model.audience,
        precision_per_site=chosen.physics.precision,
        # The same depth the circuit under the answer is drawn at. Passed rather than
        # defaulted so that a question naming a circuit but no depth gets one number,
        # not one on the picture and another in the algebra above the code.
        circuit_depth=chosen.physics.depth,
        # And the rest of the panel's chain, for whatever the question left unsaid.
        # These five dials drew the circuit above the answer and ran the Lab, and
        # reached the campaign nowhere -- so the page could show a ten-spin ring while
        # the answer under it read *TFIM L=6 (open)*. What the sentence says still
        # wins, and so does what the conversation said before it; this is the floor
        # under both, and the assumptions name it wherever it was used.
        chain_defaults=chosen.physics.as_reading(),
        # Latency, and only latency. Three calls at the front of the graph do not read
        # each other's answers, so they are issued together; the switch is here so a
        # reader can put them back in order and watch every number stay where it was.
        parallel_model_calls=chosen.model.parallel_calls,
        # The exact solution, lent to one tool for the length of this campaign. This
        # is what lets *plot the low-lying spectrum against the field* be answered
        # with the spectrum rather than with a paragraph about the literature. It is
        # lent here, by the interface, because the solver is sealed from everything
        # under `src/agent/` and cannot be imported there -- and it reaches only the
        # consultation node, which is not on the branch that gets graded. See
        # `src.physics.registry.field_sweep_bench`.
        reference_bench=field_sweep_bench(),
        memory=panels.conversation(),
    )


def tuned(chosen: knob.Setting) -> Settings | None:
    """Apply the knob's transport dials to the process configuration.

    The pool reads temperature, reply cap, timeout, throttle and retry count off a
    :class:`~src.settings.Settings`, so overlaying the knob onto a copy is what
    connects five sliders at once. Handing the pool ``None`` leaves it reading the
    process configuration directly, which is the right behaviour on a checkout with
    no credential: there is nothing to configure a call that will never be made.

    Args:
        chosen: The knob's position.

    Returns:
        A configuration carrying the visitor's dials, or ``None`` when there is no
        configuration to be had.
    """
    try:
        return chosen.model.applied_to(get_settings())
    except Exception:
        return None


def pool(chosen: knob.Setting) -> ModelPool:
    """Build the set of models this campaign may call.

    A pool per campaign rather than one per process, because the call ceiling has to
    be spent by one campaign and start again for the next -- a long-running interface
    sharing one ceiling would refuse the fourth question of an afternoon for reasons
    belonging to the first three.

    Args:
        chosen: The knob's position.

    Returns:
        The pool. An offline pool answers every request with nothing, so the caller
        has one shape to handle rather than two.
    """
    # Who is asking decides what may be spent, and it is applied here rather than
    # inside the knob: the knob records what a visitor *chose*, and a guest's
    # entitlement is not a choice. Keeping the two apart means the panel can show
    # the selection it holds and disable it, instead of holding a selection that
    # has been quietly rewritten.
    who = access.visitor()
    pinned = access.overrides_for(who)
    chosen_slugs = pinned or cast("dict[Tier, str]", chosen.model.overrides())
    return ModelPool(
        settings=_credentialled(tuned(chosen), access.own_key()),
        offline=chosen.model.offline,
        overrides=chosen_slugs,
        budget=CallBudget(limit=access.call_ceiling_for(who, chosen.model.max_calls)),
    )


def _credentialled(settings: Settings | None, key: str) -> Settings | None:
    """Swap in a visitor's own credential, when they pasted one.

    The one field :meth:`src.ui.setting.Model.applied_to` deliberately refuses to
    touch -- "the credential is not a visitor's to set" -- which is right for a
    settings knob and wrong for this, because here the visitor is supplying their
    own rather than redirecting the host's. It is replaced at the last possible
    moment, on a copy, so the process configuration is never mutated and the key
    reaches nothing but the client that needs it.

    Args:
        settings: Configuration to copy, or ``None`` when there is none to be had.
        key: The pasted credential, or ``""``.

    Returns:
        A copy carrying the visitor's key, or the argument unchanged when there is
        no key or no configuration.
    """
    if settings is None or not key:
        return settings
    return settings.model_copy(update={"openrouter_api_key": SecretStr(key)})


# No `panels.header` here, and that is a decision rather than an omission. Every
# other page reports *on* the project and opens by naming the module its figures
# came from, because a monitor is only worth reading if a reader can go and check
# it. This page is the project. "Every number on this page is read from
# src/agent/graph.py" answers a question nobody has asked yet, and the provenance of
# each answer travels with the answer -- on the card, and on the Pipeline trace page
# where anybody who wants it is already looking.
#
# The heading is the product's name rather than the word "Chat", which is the
# navigation label immediately to the left of it.
panels.masthead()

# Read once, here, and passed to both the circuit above the box and the chat knob
# below it. `panels.current_setting()` draws the sidebar knob when nothing has drawn
# it yet -- which is how a test reaches this page -- so calling it twice on one run
# draws two sliders with one identity and Streamlit refuses the second.
sidebar_setting = panels.current_setting()
# The circuit rather than the Hamiltonian. An equation is the right thing to put in
# front of a reader who already knows what the terms are, and the wrong thing to hand
# somebody deciding whether to type a question -- and this one carried a symbol, `g`,
# that is zero by default and cannot be reached from this page at all. The strip says
# the same two competing terms as a picture, is drawn from the setting so the sidebar
# moves it, and is the thing the agent actually prices. The equation itself is on Quay
# Lab, directly under the drawing of the chain.
panels.ansatz_letterhead(sidebar_setting.physics)

# No guest caption here. There were three of them at one point -- this line, a
# sidebar panel and the corner buttons -- all saying the same thing about the same
# session, which is two more than the fact needs and two more places for it to go
# stale. It lives in the account popover the corner buttons open, next to the
# control that changes it.

setting = panels.chat_settings(sidebar_setting)
memory = panels.conversation()

# The thread controls -- new chat, past chats, and the memory panel -- used to be
# drawn here, which meant they existed on this page and nowhere else: opening Quay
# Lab made the conversation and the memory disappear from the sidebar. They are now
# drawn once by `src.ui.app`, on every page. See `panels.conversation_controls`.

history: list[tuple[str, CampaignState]] = st.session_state.setdefault("history", [])

if not history:
    st.caption("Ask below, or press one of these. The same pipeline answers all of them.")
    # `OFFERED`, not every question in the set: eight buttons is two rows, and the
    # four left off are still answered when typed and still asked by `make
    # live-check`. See `src.ui.starters.NOT_OFFERED` for which four and why.
    questions = list(OFFERED)
    for start in range(0, len(questions), STARTERS_PER_ROW):
        for column, text in zip(
            st.columns(STARTERS_PER_ROW), questions[start : start + STARTERS_PER_ROW], strict=False
        ):
            column.button(
                text,
                width="stretch",
                key=f"starter-{text[:24]}",
                on_click=lambda chosen=text: st.session_state.__setitem__("pending", chosen),
            )

# No voice selector above the box. Asking somebody to choose a tone before they have
# asked anything is a control that has to be explained before it can be used, and the
# answer to "which one do I want?" is always "neutral". The three voices are still
# measured -- the Evaluations page asks the same question in all three and reports
# whether the verdict moved -- but that is a measurement the application performs, not
# a decision it hands to a visitor.
framing = "neutral"

for position, (asked, answered) in enumerate(history):
    with st.chat_message("user"):
        st.markdown(asked)
    with st.chat_message("assistant"):
        panels.answer_card(answered)
        panels.field_sweep_asked_for(answered, position)
        panels.circuit_asked_for(answered, setting.physics.depth, position)
        panels.convergence_asked_for(answered, position)
        panels.what_it_assumed(answered)
        panels.tools_called(answered)
        panels.sources_used(answered)
        if position == len(history) - 1:
            panels.answer_feedback(
                memory, setting.model.audience, position, offline=setting.model.offline
            )
            panels.followups(answered, position)


@st.cache_resource(show_spinner=False)
def _warm_the_corpus() -> bool:
    """Open the searchable notes once, in the background, before anybody asks.

    The collection is opened per search, and the *first* open in a process costs
    about 1.8 seconds -- importing the vector store and building its client -- while
    every open after it costs twenty milliseconds. Left alone, the first question of
    every session pays that, on the critical path, with a person watching.

    So it happens here instead, on a daemon thread while the page is still drawing.
    Cached as a resource, so the thread is started once per process rather than once
    per rerun, and daemon so that a warm still in flight cannot keep the process
    alive after the server is asked to stop.

    This reads the embedding credential to *construct* a client; it sends no request
    and calls no model, and a checkout with no key logs one line and carries on
    searching the keyword index -- which needs no credential at all.

    Returns:
        Always ``True``. The value is what Streamlit caches; the work is the effect.
    """
    from threading import Thread

    from src.rag.retrieve import warm_corpus

    Thread(target=warm_corpus, name="warm-corpus", daemon=True).start()
    return True


@st.cache_resource(show_spinner=False)
def _warm_the_gateway() -> bool:
    """Build the chat clients once, in the background, before anybody asks.

    The same trick as :func:`_warm_the_corpus` and for a measured reason.
    ``make timing`` prints one line per model call, and the *first* call of a
    process is consistently about three seconds slower than the identical calls
    after it: 3.66 s for a query rewrite against 0.78 s for the next call to the
    same model on the same tier. That gap is not the gateway thinking, it is this
    process importing ``langchain_openai``, constructing the client and opening
    the TLS connection -- work that has nothing to do with the question and was
    being done with a person watching a spinner.

    Doing it here moves roughly three seconds off the first answer of every
    session and changes nothing about any answer, which is the only kind of
    latency fix worth taking without an argument.

    Constructing a client sends no request and calls no model, so this costs
    nothing and a checkout with no credential simply logs a line and carries on --
    the same contract :func:`_warm_the_corpus` keeps.

    Returns:
        Always ``True``. The value is what Streamlit caches; the work is the
        effect.
    """
    from threading import Thread

    def warm() -> None:
        # Built from process configuration and **nothing session-scoped**. The first
        # version of this called `panels.current_setting()`, which reads
        # `st.session_state` -- from a worker thread, where there is no script run
        # context, so every warm logged a Streamlit warning and the test suite filled
        # up with them. The knob's dials are temperature, timeouts and retries; none
        # of them changes which client class is imported or which connection is
        # opened, and that is the only thing this exists to pay for in advance.
        try:
            from src.agent.model_selection import ModelPool

            warming = ModelPool(settings=get_settings())
            warming.for_task("query_rewrite")
            warming.for_task("problem_reading")
        except Exception as error:
            get_logger("ui.chat").info(
                "gateway_warm_skipped", extra={"detail": type(error).__name__}
            )

    Thread(target=warm, name="warm-gateway", daemon=True).start()
    return True


_warm_the_corpus()
_warm_the_gateway()

typed = st.chat_input("Ask about a chain of magnets...")
pending = st.session_state.pop("pending", None)
question = typed or pending

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        # A live commentary rather than one spinner. A campaign runs tens of seconds
        # -- most of it model calls that have to happen one after another -- and a
        # motionless spinner over that wait is indistinguishable from a hang, which
        # invites the one action that loses the answer: a reload. Each node reports
        # itself as it finishes and the label becomes what just happened, so the
        # page is visibly working even while the next round trip is outstanding. The
        # panel collapses on its own once the answer is drawn underneath it.
        trail = progress.Trail()
        # Claimed before the status panel, so the answer lands *above* the commentary
        # once it arrives. The report is finished at `scribe`, but two model calls
        # follow it -- the follow-up suggestions and the memory write -- and waiting
        # for those before showing anything held a finished answer back by seconds
        # for no reason a reader could see.
        answer_early = st.empty()
        # Where the report's progress is shown while the model is still writing it.
        # It is the longest call the graph makes and the one whose output *is* the
        # answer, so something has to happen here: the alternative is six seconds of
        # nothing followed by four paragraphs at once.
        #
        # **What is shown is the progress and not the prose, and that is a reversal.**
        # This used to stream the text itself, token by token, into a markdown
        # placeholder -- and the comment that stood here admitted what a reader was
        # actually watching: raw output, "the sources section still attached, the
        # mathematics not yet repaired". Three things went wrong with it, and none of
        # them is a matter of taste. An inline equation renders as literal
        # dollar-signs and backslashes until its closing delimiter arrives, so every
        # formula in the report appears first as debris and then as mathematics. A
        # table is a column of pipes until its last row lands. And headings pop in
        # above text already on screen, so the page reflows under the reader on
        # almost every chunk. Then the whole thing is thrown away and replaced by the
        # finished card, which means the streamed copy was never the answer -- it was
        # a draft nobody asked to see, shown in the one state in which this project's
        # own notation convention is violated.
        #
        # A word count is honest about the same fact -- the report is being written,
        # here is how far it has got -- without putting anything on screen that has
        # to be retracted. The answer arrives once, rendered, in one piece. The
        # node-by-node trail below carries the rest of the progress, and it is more
        # informative than watching prose accumulate because it says which *step* is
        # running rather than only that something is.
        writing = st.empty()
        so_far: list[str] = []

        def written(piece: str) -> None:
            """Report how far the explanation has got, without showing it.

            Args:
                piece: The next piece of text. Counted, not displayed -- see the
                    note above on why the finished report is the only version of
                    itself a reader should see.
            """
            so_far.append(piece)
            words = len("".join(so_far).split())
            writing.caption(f"Writing the report -- {words} words so far.")

        with st.status("Reading the question...", expanded=True) as running:

            def note(node: str, state: CampaignState) -> None:
                """Show a finished node, and draw the answer as soon as one exists.

                Args:
                    node: The node that just finished.
                    state: The campaign as it stands after it.
                """
                said = trail.record(node)
                running.update(label=said)
                st.write(f"\u2014 {said}")
                # No guard against drawing twice is needed: the node that writes the
                # report runs once per campaign, and the trail is what the render
                # below asks afterwards rather than a second flag that could
                # disagree with it.
                if node == progress.ANSWER_READY:
                    # The preview goes as the real card arrives, so the answer is
                    # never on the page twice in two states of repair.
                    writing.empty()
                    with answer_early.container():
                        panels.answer_card(state)

            # A campaign that raises must not take the page with it. There was no
            # handler here at all, so anything the graph could raise -- and a fuzz of
            # 280 generated questions found three things it could, all of them
            # arithmetic about small lattices -- arrived as a red traceback with the
            # question lost from the thread. The failures themselves are fixed; this
            # is the floor under the next one, because a page that answers questions
            # from a text box will always have a next one.
            try:
                finished = ask(question, framing, setting, on_node=note, prose_sink=written)
            except Exception as error:  # the page must survive anything the graph can raise
                get_logger("ui.chat").exception(
                    "campaign_failed", extra={"detail": type(error).__name__}
                )
                running.update(
                    label="Stopped: this question could not be answered",
                    state="error",
                    expanded=True,
                )
                writing.empty()
                st.error(
                    f"{panels.WARNING} This question could not be answered, and nothing "
                    "is offered in place of an answer. The failure was "
                    f"`{type(error).__name__}: {error}` at the step named above. It has "
                    "been logged; the question is unchanged, so it can be asked again or "
                    "reworded."
                )
                st.stop()
            running.update(
                label=f"Answered in {len(trail.lines)} steps",
                state="complete",
                expanded=False,
            )
        # Handed to the Pipeline page so it can show what was asked beside what the
        # agent understood, which is the pair a reader needs to judge a restatement.
        st.session_state["path_request"] = finished["request"]
        # The working -- every step it took, every configuration it refused, the
        # report in full -- is handed to the Pipeline trace page rather than drawn
        # here. See the module docstring for why.
        st.session_state["last_campaign"] = finished
        history.append((question, finished))
        # Drawn now only if the early draw did not happen -- a request refused at the
        # door never reaches the node that writes a report, and an answer shown twice
        # is worse than one shown late.
        if not trail.ran(progress.ANSWER_READY):
            panels.answer_card(finished)
        panels.field_sweep_asked_for(finished, len(history) - 1)
        panels.circuit_asked_for(finished, setting.physics.depth, len(history) - 1)
        panels.convergence_asked_for(finished, len(history) - 1)
        panels.what_it_assumed(finished)
        panels.tools_called(finished)
        panels.sources_used(finished)
        # Only under the newest answer. Repeating them beneath every past turn
        # would fill a long thread with stale buttons, and the useful set is
        # always the one at the bottom of the page.
        panels.answer_feedback(
            memory, setting.model.audience, len(history) - 1, offline=setting.model.offline
        )
        panels.followups(finished, len(history) - 1)
        st.caption(
            "Every step this took, every configuration it refused and the report in "
            "full are on the **Pipeline trace** page."
        )
