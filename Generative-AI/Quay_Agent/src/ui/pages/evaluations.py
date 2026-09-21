"""Evaluations: the held-out suite, what it concluded, and the check run live.

The page to read second, after seeing an answer. A system producing output is easy
to show; this is the evidence about whether the output is any good, and about the
one failure that is invisible from a single answer -- a verdict that depends on how
the question was worded.

Nothing here is graded by a language model. Every number is arithmetic against
a closed-form solution or a word count, which is what lets the suite be a gate
rather than a report: a judge model gives a different score to the same run on a
different day, and a number that moves on its own cannot fail a build. It also
means every figure here can be recomputed by hand, which is a stronger claim than a
calibrated score nobody can check.

Three tabs, and the split is about who is reading
--------------------------------------------------

``Results`` is the last run of the suite, drawn from the JSON summary
``make evals`` writes beside its Markdown. Drawn rather than dumped: a page that
rendered the whole scorecard as one block of Markdown made finding a single number
mean reading a document. The document is still there, folded away at the foot of
the tab, so that reading it costs nothing to run.
The suite is a command, never something this page runs when it is opened -- a page
that re-scored the agent on every visit would report a different number each time
somebody moved a slider, and would bill whoever clicked the tab.

``Try it yourself`` is the honesty measurement, live, on a question of the
reader's choosing. It is the one thing on this page that costs money, and it is
behind a button for that reason.

``What is measured`` says what the two suites defend and why the cases are the
cases they are. Kept apart from the numbers so that a reader who wants the result
is not made to read the methodology first, and a reader who doubts the result can
find the methodology without hunting.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.agent.graph import run_campaign
from src.agent.mathmarkup import demote_headings
from src.agent.state import Framing
from src.evals.cases import ACCURACY_CASES, HONESTY_CASES
from src.evals.metrics import TARGET_ERROR_PER_SITE
from src.evals.scorecard import SCORECARD_PATH, SUMMARY_PATH
from src.hardware.devices import device_for
from src.ui import about, panels

FRAMINGS: tuple[Framing, ...] = ("neutral", "vendor", "skeptical")

REBUILD = (
    "Run `make evals`. It answers every case with the real agent, so it makes live "
    "model calls and needs the gateway credential -- which is why it is a command "
    "somebody runs deliberately rather than something this page does when opened."
)

panels.header(
    "Evaluations",
    "**How well it answers.** Three suites it was not tuned against, graded by "
    "arithmetic against a reference it cannot see, plus an honesty check you can run "
    "live. For *what it cost to get there*, see **Analytics**.",
    "src/evals/ and reports/scorecard.md",
)

left, right = st.columns(2)
left.info(
    "**There is a right answer, and the agent cannot see it.** This chain can be "
    "solved with pencil and paper, so there is an exact answer to score against. The agent is "
    "sealed away from it by an import wall that the architecture test walks on "
    "every run -- it has to work the problem rather than look it up. Most "
    "language-model applications are graded by another language model; this one is "
    "graded by arithmetic."
)
right.info(
    "**Nothing here needs a credential to read.** Every decision that changes a "
    "number is arithmetic, so the whole system runs with the language model switched "
    "off -- what is lost is the prose and nothing else. That is why offline is a "
    "supported mode rather than a degraded one, and why anybody can check these "
    "claims without an account."
)

results_tab, live_tab, method_tab, ethics_tab = st.tabs(
    # "Ethics and limits" came from the About page, which is gone. Being honest
    # about the score and being honest about the limits are one subject, and a
    # reader who has just read the first is exactly the reader for the second.
    ["Results", "Try it yourself", "What is measured", "Ethics and limits"]
)

# --------------------------------------------------------------------------

with results_tab:
    if not SUMMARY_PATH.exists():
        panels.empty_state("No scorecard yet.", REBUILD)
        if SCORECARD_PATH.exists():
            st.caption(
                "There is a Markdown scorecard on disk from before the summary was "
                "added. It is folded away at the foot of this tab; re-run `make evals` "
                "to fill in the charts."
            )
    else:
        report = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        accuracy = report["accuracy"]
        honesty = report["honesty"]

        # The headline first, because four minutes on this page buys a number
        # before it buys an argument -- and the criterion beside it, because a
        # percentage whose pass condition is not stated is not a measurement.
        if report.get("scores"):
            st.subheader("The three scores")
            panels.metrics(
                {
                    row["name"]: row["percent"] if row["ran"] else "not run"
                    for row in report["scores"]
                }
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "suite": row["name"],
                            "passed": f"{row['passed']}/{row['total']}" if row["ran"] else "—",
                            "what a pass means": row["criterion"],
                            "fails a build": panels.tick(row["gates"]),
                        }
                        for row in report["scores"]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "**Three numbers and no fourth.** The suites measure things that "
                "are not commensurable — how close the physics got, whether the "
                "wording moved the verdict, whether the search found the right "
                "note — so there is no weighted total. Averaging them would let a "
                "good search paper over a wrong answer. Only the honesty suite "
                "fails a build; the argument for that is in the table above."
            )
            st.divider()

        retrieval = report.get("retrieval")
        if retrieval:
            st.subheader("Did the search find the right note?")
            st.caption(
                f"Searched with **{retrieval['searched_with']}**, keeping the best "
                f"{retrieval['top_k']} passages per question — the same number this "
                "application shows. Relevance is judged against the curated topic "
                "tags each note declares in its own frontmatter, which the ranking "
                "never reads. Scoring a search against keywords taken from the "
                "question would measure whether keyword search can find its own "
                "vocabulary, which it can."
            )
            panels.metrics(
                {
                    "Questions answered": f"{retrieval['passed']}/{retrieval['cases']}",
                    f"Precision@{retrieval['top_k']}": f"{retrieval['mean_precision']:.2f}",
                    "Mean reciprocal rank": f"{retrieval['mean_reciprocal_rank']:.2f}",
                    "Right shelf chosen": (
                        f"{retrieval['routed_correctly']}/{retrieval['answerable']}"
                    ),
                }
            )
            # `first hit at` is a rank, so it stays an integer column even where
            # there is no rank to report. It used to fall back to an em-dash, which
            # put a string in a column of numbers: Arrow could not convert the
            # column, Streamlit logged a traceback on every render of this page and
            # then silently cast the whole thing to text, which cost the column its
            # numeric sort and its right alignment. A nullable integer renders the
            # gap as an empty cell and keeps both.
            #
            # The condition is an explicit `is None` rather than a falsy test. Ranks
            # here are one-based, so `or` happens to work -- and would start hiding a
            # genuine rank of zero the day anybody made them zero-based.
            ranked = pd.DataFrame(
                [
                    {
                        "question": row["question"],
                        "found": panels.tick(row["passed"]),
                        "relevant": f"{row['relevant']}/{row['returned']}",
                        "first hit at": (
                            None
                            if row["first_relevant_rank"] is None
                            else int(row["first_relevant_rank"])
                        ),
                        "shelf chosen": ", ".join(row["routed_to"]) or "unrestricted",
                    }
                    for row in retrieval["cases_detail"]
                ]
            )
            ranked["first hit at"] = ranked["first hit at"].astype("Int64")
            st.dataframe(
                ranked,
                hide_index=True,
                width="stretch",
                column_config={
                    "first hit at": st.column_config.NumberColumn(
                        help="Where the first relevant note landed. Empty means none was found.",
                    )
                },
            )
            st.caption(
                "**The last row is the one worth reading.** It asks something the "
                "corpus holds nothing on, and the only way to pass it is to return "
                "nothing at all — so a search that answers everything confidently "
                "fails exactly one case here, which is the point of including it."
            )
            st.divider()

        st.subheader("Did the verdict depend on how the question was worded?")
        st.caption(
            "The finding, so it goes first. The same chain, the same budget and the "
            "same machine, described three ways: flatly, by somebody who wants a yes, "
            "and by somebody who assumes a no. Nothing in the agent acts on the voice. "
            "**A verdict that moves between them is a verdict about the wording**, and "
            "it is the one thing in this project that fails a build."
        )
        if not honesty["framings"]:
            st.caption("The last run did not include the honesty suite.")
        elif honesty["drifted"]:
            st.error(
                f"{panels.WARNING} **The verdict moved.** These framings disagreed with "
                f"the neutral one: {', '.join(honesty['drifted'])}."
            )
        else:
            st.success(
                f"{panels.PRESENT} **The verdict did not move.** All "
                f"{honesty['framings']} framings reached the same call."
            )
        if honesty["framings_detail"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "asked as": row["framing"],
                            "verdict": row["verdict"] or "—",
                            "confidence": row["confidence"] or "—",
                            "hedges / 100 words": row["hedge_rate"],
                            "words": row["words"],
                            "baseline first appears at": (
                                "never"
                                if row["baseline_position"] is None
                                else f"{row['baseline_position']:.0%} in"
                            ),
                        }
                        for row in honesty["framings_detail"]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "**The last column is the subtle failure this suite looks for.** The "
                "verdict can stay honest while the evidence for it drifts towards the "
                "end of the document. A comparison nobody reads is a comparison that "
                "was not made."
            )

        st.divider()
        st.subheader("How close did it get to the true answer?")
        if not accuracy["cases"]:
            st.caption("The last run did not include the accuracy suite.")
        else:
            mean = accuracy["mean_error_per_site"]
            worst = accuracy["worst_error_per_site"]
            panels.metrics(
                {
                    "Cases solved": f"{accuracy['solved']}/{accuracy['cases']}",
                    "Mean error / magnet": "—" if mean is None else f"{mean:.2e}",
                    "Worst error / magnet": "—" if worst is None else f"{worst:.2e}",
                    "Target": f"{accuracy['target_error_per_site']:.0e}",
                }
            )
            detail = accuracy["cases_detail"]
            errors = [row for row in detail if row["error_per_site"] is not None]
            if errors:
                st.caption(
                    "Error per magnet, by case, against the target. The bars that "
                    "stand out sit near $h/J = 1$, where the chain is critical and "
                    "every approximate method has its hardest time -- which is where "
                    "the suite deliberately puts most of its cases."
                )
                st.bar_chart(
                    pd.DataFrame(
                        {"error per magnet": [row["error_per_site"] for row in errors]},
                        index=[row["name"] for row in errors],
                    ),
                    height=260,
                    horizontal=True,
                )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "case": row["name"],
                            "h/J": row["ratio"],
                            "read as": row["read_as"] or "—",
                            "exact": row["exact_per_site"],
                            "reached": row["reached_per_site"],
                            "error": row["error_per_site"],
                            "layers": row["depth"],
                            "shots": row["shots_spent"],
                            "solved": panels.tick(row["solved"]),
                            "seconds": row["seconds"],
                            "failed": row["failure"] or "",
                        }
                        for row in detail
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "**`read as` is what the agent decided the question described.** A "
                "case that solved a different chain perfectly failed at reading rather "
                "than at physics, and the two have different fixes. Accuracy is "
                "deliberately **not** a gate: failing a build for an honest miss "
                "creates pressure to loosen the tolerance until it passes, which is "
                "how a measurement stops measuring."
            )
            failed = [row for row in detail if row["failure"]]
            for row in failed:
                st.warning(f"**{row['name']}** did not run — {row['failure']}")

        st.divider()
        st.caption(
            f"Measured {report['measured_at']}. Graded by a language model: "
            f"{'yes' if report['graded_by_a_model'] else 'no'}. {REBUILD}"
        )
        if SCORECARD_PATH.exists():
            with st.expander("The full report, as `make evals` wrote it"):
                # Demoted, because that file is a document in its own right and
                # opens with an `# Scorecard` H1. Embedded raw it rendered *larger*
                # than this page's own title, inside a collapsed expander -- the one
                # place on the page where a reader is least expecting the biggest
                # text on screen. The same shift the agent's answers already get.
                st.markdown(demote_headings(SCORECARD_PATH.read_text(encoding="utf-8")))

# --------------------------------------------------------------------------

with live_tab:
    st.subheader("Run the honesty check yourself")
    st.caption(
        "The same measurement the suite makes, on a question of your choosing. It "
        "asks three times in three voices -- plainly, from somebody who wants the "
        "answer to be yes, and from somebody who assumes it is no. Nothing in the "
        "agent acts on the voice; the point is to see whether the answer moves when "
        "only the wording does. Three full campaigns, so it takes a moment and, with "
        "a model switched on, costs tokens."
    )

    setting = panels.current_setting()
    question = st.text_input(
        "question",
        value="Is quantum hardware worth it for a 10-spin critical Ising chain?",
        label_visibility="collapsed",
    )
    if st.button("Ask it three ways", type="primary"):
        with st.spinner("Asking three times..."):
            st.session_state["framing_check"] = {
                voice: run_campaign(
                    question,
                    framing=voice,
                    shot_budget=setting.physics.shots,
                    device=device_for(setting.physics.device),
                    chat_model=None if setting.model.offline else "auto",
                    search_corpus=True,
                    fetch_external=False,
                )
                for voice in FRAMINGS
            }

    checked = st.session_state.get("framing_check")
    if not checked:
        panels.empty_state(
            "Not run yet.",
            "Type a question and press **Ask it three ways**. With the language model "
            "switched off it still runs, costs nothing, and still measures the thing "
            "worth measuring -- the verdict is decided by rule either way.",
        )
    else:
        calls = {voice: state["verdict"] for voice, state in checked.items()}
        reached = {verdict.call for verdict in calls.values() if verdict is not None}
        if len(reached) == 1:
            st.success(
                f"{panels.PRESENT} The verdict did not move. All three framings reached "
                f"**{reached.pop()}**."
            )
        else:
            st.error(
                f"{panels.WARNING} The verdict moved with the wording: "
                f"{', '.join(sorted(reached))}. That is a finding about the agent, not "
                "about the physics, and it is the one thing in this project that fails a "
                "build."
            )
        st.dataframe(
            [
                {
                    "asked as": voice,
                    "verdict": verdict.call if verdict else "—",
                    "confidence": verdict.confidence if verdict else "—",
                    "why": verdict.summary if verdict else "—",
                }
                for voice, verdict in calls.items()
            ],
            hide_index=True,
            width="stretch",
        )

# --------------------------------------------------------------------------

with method_tab:
    panels.metrics(
        {
            "Accuracy cases": len(ACCURACY_CASES),
            "Honesty cases": len(HONESTY_CASES),
            "Target error per magnet": f"{TARGET_ERROR_PER_SITE:.0e}",
            "Graded by a model": "no",
        }
    )
    st.info(
        "**Two evaluations, measuring two different things.** *Accuracy* asks whether "
        "the machinery works: given a chain, does a campaign reach an energy close to "
        "the true one? It is graded by arithmetic against a solution the agent could "
        "not reach. *Honesty* asks whether the answer depends on how the question was "
        "asked -- the same chain, the same budget, the same machine, described three "
        "ways. A verdict that moves between them is a verdict about the wording."
    )
    st.subheader("Why the cases are the cases they are")
    st.caption(
        "The accuracy suite is weighted towards a field about equal to the coupling, "
        "because that is where the chain is critical and every approximate method has "
        "its hardest time. Away from it the problem is easy in both directions, so a "
        "suite that sampled evenly would report a high score for machinery that fails "
        "exactly where it matters."
    )
    st.dataframe(
        [
            {
                "case": case.name,
                "magnets": case.spec.n_sites,
                "h/J": round(case.ratio, 2),
                "regime": "critical"
                if abs(case.ratio - 1.0) < 0.2
                else ("ordered" if case.ratio < 1.0 else "disordered"),
                "why it is in the suite": case.note,
            }
            for case in ACCURACY_CASES
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "**Where to check the other half of the claim.** The suite measures the agent "
        "against a fixed model configuration. Whether the *choice* of model moves any "
        "of these numbers is a separate experiment, and it is on the **Model bake-off** "
        "tab of the Analytics page."
    )

with ethics_tab:
    about.ethics_section()

panels.footer("src/evals/ · src/ui/about.py")
