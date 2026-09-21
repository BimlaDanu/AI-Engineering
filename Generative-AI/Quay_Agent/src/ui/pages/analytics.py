"""Analytics: what this session did, what a model swap changes, and what it all costs.

Four tabs, and they answer four different questions that a single page of charts
would blur together.

``This session`` is the agentic evidence. A screenshot of a good answer proves
nothing about a pipeline -- a single prompt behind a text box would produce one
too. What cannot be faked is the shape of the work: unequal bars over the graph's
nodes, because ``baseline`` and ``solve`` fire only for a feasibility question and
``explain`` only for a question about the subject; a planning loop that shows up as
a node visited three times in one run; and wall-clock per *kind* of call, which is
the measurement that either supports the tier design or refutes it.

``Model bake-off`` is the experiment this project's central claim invites. If the
numbers really are computed by deterministic code and only the prose is written by
a language model, then swapping the model must leave the numbers alone. So swap it
and look. Three checks, in the order that decides whether the rest of the table
means anything: did every model read the same problem, did the computed baseline
hold, did the verdict hold. Only then, efficiency.

``Hardware cost surface`` is the physics half, and it is not a dashboard of this
application's activity. What is worth analysing in a feasibility project is how
depth, duration, surviving signal and measurement count move as the chain
lengthens and the machine changes -- because that surface is what every verdict is
an evaluation of. The table behind it is committed to the repository, contains no
random numbers and no model, and so reproduces byte for byte with `make sweep`.

``Models`` is the inventory and the routing policy: which tier serves which call,
and every model the subscription can reach.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.agent.graph import pipeline_nodes
from src.agent.model_selection import DEFAULT_TIER_SLUGS, TASK_TIERS, selectable_slugs
from src.agent.usage import price_basis
from src.evals import bakeoff
from src.evals.sweep import SWEEP_PATH
from src.hardware.devices import device_for
from src.model_catalogue import CATALOGUE
from src.ui import about, panels, session_log
from src.ui import setting as knob

DEFAULT_BAKEOFF_QUESTION = "Is quantum hardware worth it for a 10-spin critical Ising chain?"
# The question the bake-off opens on.
#
# A feasibility question rather than an explanation, deliberately: it is the branch
# that computes, so it is the branch where "the model swap did not move the number"
# is a claim with a number behind it. An explanation would compare prose to prose.

MAX_BAKEOFF_MODELS = 4
# How many models one comparison may run.
#
# Each is a full campaign at live prices, and the marginal model past the fourth
# tells a reader nothing the fourth did not. A ceiling here is cheaper than a
# ceiling discovered on an invoice.

panels.header(
    "Analytics",
    "**What it costs.** Money spent on model calls this session, and — separately — "
    "what a circuit costs in measurements on each machine. Two different budgets, "
    "kept apart. For *how well it answers*, see **Evaluations**.",
    "src/ui/session_log.py · src/evals/bakeoff.py · data/resource-sweep.json",
)

# Read once, before the tabs. `current_setting` draws the sidebar itself when a page
# was opened on its own, so calling it inside two tabs would draw two of every slider.
setting = panels.current_setting()

session_tab, bakeoff_tab, surface_tab, models_tab, developer_tab = st.tabs(
    # "For developers" came from the About page, which is gone. This page was
    # already the instrumentation page, so versions, configuration and what a run
    # left on disk belong on it rather than in a navigation slot of their own.
    ["Model spend", "Model bake-off", "Hardware cost surface", "Models", "For developers"]
)

# --------------------------------------------------------------------------

with session_tab:
    asked = panels.asked_this_session()
    if not asked:
        panels.empty_state(
            "Nothing asked yet in this session.",
            "Ask something on the **Chat** page and this fills in. Every chart here "
            "is derived from the answers themselves rather than from a counter, so "
            "there is nothing to show before the first one.",
        )
    else:
        total = session_log.totals(asked)
        panels.metrics(
            {
                "Questions": len(asked),
                "Model calls": total.calls_made,
                "Tokens": f"{total.total_tokens:,}",
                "Estimated spend (USD)": total.cost_label(),
            }
        )
        if total.calls_made == 0:
            st.info(
                f"{panels.PRESENT} **No language model was called.** Every answer above "
                "was reached on the deterministic path. The charts below still work, "
                "because what they measure is which code ran -- not which model did."
            )
        elif basis := total.cost_basis():
            st.caption(f"{panels.WARNING} {basis}")

        st.divider()
        st.subheader("Which arm of the graph served each question")
        st.caption(
            "Read off the nodes that actually ran, not off what the router decided. "
            "The two come apart on exactly the runs worth looking at -- a question "
            "that was screened out, or one whose formalisation failed and fell "
            "through to the conclusion without pricing anything."
        )
        branches = session_log.branch_counts(asked)
        st.bar_chart(
            pd.DataFrame({"questions": list(branches.values())}, index=list(branches)),
            height=220,
            horizontal=True,
        )

        st.divider()
        st.subheader("Which nodes actually ran")
        st.caption(
            "**Unequal bars are the point.** `baseline`, `plan` and `solve` fire only "
            "when a question needs a number; `explain` only when it needs prose; "
            "`implement` only when somebody asked for code. A fixed pipeline behind a "
            "text box would draw these all the same height, and this is the chart "
            "that tells the two apart."
        )
        nodes = session_log.node_counts(asked, pipeline_nodes())
        st.bar_chart(
            pd.DataFrame({"runs that reached it": list(nodes.values())}, index=list(nodes)),
            height=380,
            horizontal=True,
        )
        retries = session_log.retry_counts(asked)
        if retries:
            st.caption(
                "**Some nodes ran more than once inside a single question.** That is "
                "the repair loop: a configuration was priced, refused by a budget, and "
                "the planner got another turn. Extra visits, by node: "
                + ", ".join(f"`{node}` +{count}" for node, count in retries.items())
                + "."
            )
        else:
            st.caption(
                "No node ran twice in any question, so no budget was binding enough to "
                "force a re-plan. That is a fact about this session's settings, not a "
                "claim that the loop does not exist."
            )

        st.divider()
        st.subheader("Where the wall-clock went, by kind of call")
        timings = session_log.timing_by_task(asked)
        if not timings:
            st.caption("No model calls to time.")
        else:
            st.caption(
                "**This is the tier design, measured.** Every call the design puts on "
                "the fast tier is one somebody is waiting on -- which shelf to search, "
                "whether a passage earned its place, how to restate a question. If the "
                "fast tier's mean is not visibly below the strong tier's here, the "
                "tiering is costing complexity and buying nothing, and this table is "
                "where that would show. The last two columns answer the other half of "
                "the question -- which step spent the money, not just the time. They "
                "count published rates only, so a step served by a model the vendor "
                "publishes no rate for reads as $0.0000 here; the tile above says "
                "whether this session had any."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "call": timing.task,
                            "tier": timing.tier,
                            "calls": timing.calls,
                            "total seconds": round(timing.seconds, 2),
                            "mean seconds": round(timing.mean_seconds, 2),
                            "tokens": timing.tokens,
                            "cost": f"${timing.cost_usd:.4f}",
                        }
                        for timing in timings
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

        st.divider()
        with st.expander("Question by question"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "question": entry.question,
                            "asked as": entry.framing,
                            "branch": entry.branch,
                            "routed by": entry.decided_by,
                            "verdict": entry.verdict or "—",
                            "written by": entry.written_by or "—",
                            "cited": entry.cited,
                            "seconds": entry.seconds,
                            "calls": entry.usage.calls_made,
                            "tokens": entry.usage.total_tokens,
                            "cost (USD)": round(entry.usage.cost_usd, 6),
                        }
                        for entry in asked
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "The path each question took through the graph is drawn on the "
                "**Pipeline trace** page for the most recent one."
            )

# --------------------------------------------------------------------------

with bakeoff_tab:
    st.markdown(
        "**The claim this page tests.** A language model here formalises the "
        "question, proposes a circuit depth and writes the document. Deterministic "
        "code computes every number and a sealed solver grades it. If that is true, "
        "**changing the model must not change the answer** — only how long it took, "
        "what it cost, and how it reads."
    )
    st.caption(
        "So change it and look. The same question, the same chain, the same budget "
        "and the same machine, run once per model with that model serving every "
        "tier. Three checks, in this order, because the later ones are meaningless "
        "if an earlier one fails."
    )
    st.markdown(
        "\n".join(
            [
                "| # | check | why it comes first |",
                "|---|---|---|",
                "| 1 | **Did every model read the same problem?** | The chain is "
                "recovered from ordinary language by a model call. It is the one "
                "place a model can move a number, and it moves all of them at once. "
                "Two models that disagree here answered two different questions. |",
                "| 2 | **Did the computed baseline hold?** | Run from a fixed seed at "
                "a fixed depth by code no model touches. For an identical reading it "
                "must come back identical. A difference here is an architecture bug, "
                "not a fact about the models. |",
                "| 3 | **Did the verdict hold?** | Decided by rule from the numbers. "
                "If it moves while the numbers do not, the rule is reading something "
                "a model wrote. |",
            ]
        )
    )
    st.caption(
        "Only then is it worth reading the efficiency columns. A bake-off that led "
        "with *model A was 300 ms faster* would invite a choice on latency without "
        "first establishing that the two models answered the same question."
    )

    st.divider()
    reachable = knob.credential_available()
    if not reachable:
        panels.empty_state(
            "No credential, so there is nothing to compare.",
            "The bake-off runs one live campaign per model. Set `OPENROUTER_API_KEY` "
            "in `.env` and reopen this page. Everything else in this application "
            "runs without one; this is the single feature that cannot.",
        )
    else:
        question = st.text_input(
            "The question every model is asked",
            value=DEFAULT_BAKEOFF_QUESTION,
            help="The same string for every model. Changing it restarts the comparison.",
        )
        offered = list(selectable_slugs())
        preset = [slug for slug in DEFAULT_TIER_SLUGS.values() if slug in offered]
        chosen_models = st.multiselect(
            "Models to compare",
            options=offered,
            default=preset[:MAX_BAKEOFF_MODELS],
            max_selections=MAX_BAKEOFF_MODELS,
            help=(
                "Each one runs a full campaign at live prices. The three defaults are "
                "this deployment's own fast, standard and strong tiers, which makes "
                "the first comparison the one worth having: is the expensive tier "
                "earning its place?"
            ),
        )
        st.caption(
            (
                "Pick at least two — a bake-off of one model has nothing to compare it "
                "against, which is why the button below is greyed out."
                if len(chosen_models) < 2
                else f"{len(chosen_models)} campaigns will run, one after another."
            )
            + " Sequentially rather than at once, because wall-clock is one of the "
            "columns and four campaigns sharing one machine would report each other's "
            "contention as their own latency."
        )
        if st.button("Run the bake-off", type="primary", disabled=len(chosen_models) < 2):
            with st.spinner(f"Running {len(chosen_models)} campaigns..."):
                st.session_state["bakeoff"] = {
                    "question": question,
                    "trials": bakeoff.compare(
                        question,
                        chosen_models,
                        shot_budget=setting.physics.shots,
                        device=device_for(setting.physics.device),
                    ),
                }
        if len(chosen_models) < 2:
            st.caption("Pick at least two models — one model is not a comparison.")

        measured = st.session_state.get("bakeoff")
        if measured:
            trials = measured["trials"]
            found = bakeoff.agreement(trials)
            st.divider()
            st.subheader("What survived the model swap")
            if found.read_alike and found.numbers_held and found.verdict_held:
                st.success(f"{panels.PRESENT} {found.headline()}")
            else:
                st.error(f"{panels.WARNING} {found.headline()}")
            panels.metrics(
                {
                    "Models compared": found.compared,
                    "Distinct readings": len(found.readings),
                    "Distinct baselines": len(found.baselines),
                    "Distinct verdicts": len(found.verdicts),
                }
            )
            if found.failures:
                st.warning(
                    f"{panels.WARNING} Did not finish: {', '.join(found.failures)}. A "
                    "model that cannot be reached from here is a result about that "
                    "model, so it keeps its row rather than raising."
                )

            st.markdown("##### The comparison")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "model": trial.slug,
                            "read as": trial.read_as or "—",
                            "baseline / magnet": (
                                "—"
                                if trial.baseline_energy_per_site is None
                                else f"{trial.baseline_energy_per_site:.9f}"
                            ),
                            "verdict": trial.verdict or "—",
                            "seconds": trial.seconds,
                            "calls": trial.calls,
                            "tokens": trial.tokens,
                            "cost (USD)": bakeoff.cost_label(trial),
                            "words": trial.words,
                            "cited": trial.cited,
                            "hedges / 100 words": round(trial.hedge_rate, 2),
                            "baseline stated at": (
                                "never"
                                if trial.baseline_at is None
                                else f"{trial.baseline_at:.0%} in"
                            ),
                            "failed": trial.failure or "",
                        }
                        for trial in trials
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "**Nothing in that table was graded by a language model.** Every "
                "quality column is arithmetic or a word count — passages cited, "
                "hedging words per hundred, how far into the document the classical "
                "comparison first appears. A judge model would score the same run "
                "differently on a different day, and a number that moves on its own "
                "cannot support a decision."
            )

            missing = bakeoff.unpriced(trials)
            if missing:
                estimated = [slug for slug in missing if price_basis(slug) == "estimated"]
                unknown = [slug for slug in missing if price_basis(slug) == "unknown"]
                lines = []
                if estimated:
                    lines.append(
                        f"No **published** rate for {', '.join(estimated)}. Those rows "
                        "are marked *est.* and are costed by reasoning from the "
                        "vendor's public price for the nearest model of the same size "
                        "and family."
                    )
                if unknown:
                    lines.append(f"No rate at all for {', '.join(unknown)}.")
                lines.append(
                    "Either way the row is excluded from the cheapest line below. A "
                    "catalogue gap is not a free model, and naming one 'cheapest' on "
                    "an extrapolation recommends the single row nobody has costed."
                )
                st.caption(f"{panels.WARNING} " + " ".join(lines))

            best_price = bakeoff.cheapest(trials)
            best_speed = bakeoff.fastest(trials)
            left, right = st.columns(2)
            if best_price is not None:
                left.info(
                    f"**Cheapest that finished:** `{best_price.slug}` at "
                    f"${best_price.cost_usd:.6f}."
                )
            if best_speed is not None:
                right.info(
                    f"**Fastest that finished:** `{best_speed.slug}` at {best_speed.seconds:.1f}s."
                )
            if found.read_alike and found.numbers_held:
                st.caption(
                    "With the numbers identical across every model, the cheapest row "
                    "is the defensible default and the expensive one has to justify "
                    "itself on the writing columns alone. That is the decision this "
                    "page exists to make out of evidence rather than out of habit — "
                    "and it is the argument for putting the strong model on the "
                    "report and nothing else."
                )

# --------------------------------------------------------------------------

with surface_tab:
    panels.model_has_no_say("any figure on this tab")
    if not SWEEP_PATH.exists():
        panels.empty_state(
            "The table has not been built yet.",
            "Run `make sweep`. It prices every combination of chain length, boundary, "
            "circuit depth and machine, takes about a second, and needs no network and "
            "no key.",
        )
    else:
        stored = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
        frame = pd.DataFrame(stored["rows"])
        boundary = setting.physics.boundary
        shaped = frame[frame["boundary"] == boundary]

        st.caption(
            "**What to look for, in three glances.** The ideal machine is flat and "
            "the real ones are not — every curve that bends bends because of "
            "hardware, not because of the algorithm, and that separation is what a "
            "feasibility study exists to make. Depth is not the binding constraint: "
            "on both real machines the circuits run out of *signal* long before they "
            "run out of *time*. And rings cost where lines do not, because no machine "
            "here wires the two ends of a chain together."
        )
        panels.metrics(
            {
                "Configurations priced": len(frame),
                "Inside their machine's ceiling": int(frame["within_ceiling"].sum()),
                "Machines": frame["device"].nunique(),
                "Shape shown": "line" if boundary == "open" else "ring",
            }
        )
        st.caption(
            "The shape comes from the settings knob, so this page and every other one "
            "are always describing the same problem."
        )

        st.divider()
        st.subheader("How deep each machine will go, as the chain lengthens")
        st.caption(
            "The deepest circuit still worth running. Where a curve reaches zero, that "
            "machine cannot usefully run that chain at any depth at all."
        )
        ceilings = shaped.groupby(["n_sites", "device"])["max_depth"].max().unstack().sort_index()
        st.line_chart(ceilings, height=280)

        st.divider()
        st.subheader("What noise costs in extra measurements")
        st.caption(
            "Multiples of what a perfect machine would need for the same accuracy. It "
            "is the inverse square of the surviving signal, which is why it climbs so "
            "much faster than the fidelity curve falls -- and it is the number that "
            "decides most feasibility questions."
        )
        available = sorted(shaped["depth"].unique())
        depth_shown = min(available, key=lambda value: abs(value - setting.physics.depth))
        st.caption(
            f"At **{depth_shown}** circuit layers, the nearest priced depth to the "
            f"knob's current {setting.physics.depth}."
        )
        inflation = (
            shaped[shaped["depth"] == depth_shown]
            .pivot_table(index="n_sites", columns="device", values="shot_inflation")
            .sort_index()
        )
        st.line_chart(inflation, height=280)

        st.divider()
        st.subheader("What stops each machine first")
        st.caption(
            "A *fidelity* limit asks for more accurate gates. A *coherence* limit asks "
            "for faster ones. They are different requests, and a report that said only "
            "'it does not work' would let a reader act on the wrong one."
        )
        st.dataframe(
            shaped.groupby(["device", "binding_constraint"]).size().unstack(fill_value=0),
            width="stretch",
        )

        with st.expander("The whole table"):
            st.caption(
                "Every row, as committed. Sort a column by clicking it. This is the "
                "file a resource claim in any report can be checked against."
            )
            st.dataframe(shaped, hide_index=True, width="stretch", height=420)
        st.caption(f"**How it was produced.** {stored['how']}")

# --------------------------------------------------------------------------

with models_tab:
    st.subheader("This session, and the models behind it")
    panels.session_cost()

    st.divider()
    st.subheader("Which model serves which call")
    st.caption(
        "Three tiers, and they are not three sizes of the same thing. The cheap tier "
        "serves everything a person is waiting on -- which shelf to search, whether a "
        "passage earned its place, how to restate a question -- and those are short "
        "classifications where a larger model buys latency and nothing else. The strong "
        "tier serves the written report and nothing else, because that is the one call "
        "whose output a human reads end to end. **The Model bake-off tab is where that "
        "policy is checked rather than asserted.**"
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"call": task, "tier": tier, "model unless overridden": DEFAULT_TIER_SLUGS[tier]}
                for task, tier in TASK_TIERS.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.divider()
    st.subheader(f"Every model this subscription can reach ({len(CATALOGUE)})")
    st.caption(
        "Transcribed from the subscription. A slug marked unconfirmed was reconstructed "
        "from a display name and never resolved against the live index -- a wrong one "
        "fails at the first call, several minutes into a run, which is why it is shown "
        "here rather than discovered there. Any of them can be put on any tier from the "
        "settings knob."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "model": card.display_name,
                    "slug": card.slug,
                    "role": card.role,
                    "tier": card.tier,
                    "slug checked": panels.tick(card.confirmed),
                    "note": card.note,
                }
                for card in CATALOGUE
            ]
        ),
        hide_index=True,
        width="stretch",
    )

with developer_tab:
    about.developer_section()

panels.footer("src/ui/session_log.py · src/evals/bakeoff.py · src/evals/sweep.py · src/ui/about.py")
