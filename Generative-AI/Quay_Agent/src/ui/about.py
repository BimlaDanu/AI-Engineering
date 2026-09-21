"""The two sections the About page held that a reader still needs.

The About page is gone, and this is where its content went. It was the last
entry in a navigation of nine, and it was the only one that was not a question about
an answer -- so it competed with the product for a slot in the sidebar while being
the page least likely to be opened. Three tabs of it, redistributed by where a
reader is already standing when the question occurs to them:

- *What this is, who it is for* -- deleted from the application. It was a page-length
  version of what the README says properly and what the chat page's empty state says
  in two lines. A reader inside the application has already decided to use it.
- *Ethics and limitations* -- now a tab on the ``Evaluations`` page, which is the
  page about how well this thing does on questions it was not tuned against. Being
  honest about the score and being honest about the limits are the same subject, and
  a reader who has just read one is the reader for the other.
- *For developers* -- now a tab on the ``Analytics`` page, which is already the
  instrumentation page. It opens by saying that nothing in it is needed to use Quay,
  because that is true and because somebody who lands there by accident should be
  able to leave immediately.

No secret is ever rendered here. Credentials are held as ``SecretStr`` and this
reports only whether each one is present -- which matters more on a tab than it did
on a page, because a tab beside a scorecard is more likely to be in a screenshot.
"""

from __future__ import annotations

import sys

import streamlit as st

from src.model_catalogue import CATALOGUE, card_for, unconfirmed_slugs
from src.settings import Settings, get_settings
from src.ui import panels
from src.ui.status import (
    ARTEFACT_DIRECTORIES,
    ARTEFACT_PURPOSE,
    DEPENDENCIES,
    PROJECT_ROOT,
    artefacts,
    corpus_size,
    missing_dependencies,
)


def ethics_section() -> None:
    """Draw the ethics statement and the limitations.

    Written for somebody who will never open the code, and every claim in it is
    checkable on another page of the application rather than taken on trust.
    """
    st.markdown("##### Ethics and limitations")
    st.caption(
        "Written for a reader who will never open the code. Every claim here is "
        "something a reader can check on another page of this application."
    )

    st.markdown(
        '**A verdict is dated, and it is about today\'s machines.** "Not feasible" '
        "means not feasible on the devices and error rates this application knows "
        "about, at the budget you set. It is not a statement about quantum computing "
        "in general and it carries the date and the device it was reached on, so it "
        "cannot be quoted as a permanent finding."
    )
    st.markdown(
        "**Nothing here is graded by a language model.** Every quality number in "
        "*How well it does* is arithmetic or a word count — energies, error against "
        "the exact answer, passages cited, hedging words per hundred. A judge model "
        "would score the same run differently on a different day, and a number that "
        "moves on its own cannot support a decision."
    )
    st.markdown(
        "**The agent is told what it cannot do, and the code enforces it.** It "
        "cannot reach the exact solvers, it cannot skip the classical comparison, "
        "and it cannot exceed the measurement budget — the budget refuses on "
        "arithmetic rather than warning and continuing."
    )
    st.markdown(
        "**Retrieval is cited or it is refused.** Answers drawn from the literature "
        "name the passages they rest on, with a citable identifier for each. When "
        "the corpus does not support a claim the answer says so instead of filling "
        "the gap from the model's own memory."
    )
    st.markdown(
        "**Your data.** Questions and the agent's replies are kept in a local file "
        "on this machine so the next question can be answered in the light of the "
        "last. The sidebar shows exactly what is being held and clears it on one "
        "click — a memory nobody can inspect is a memory nobody agreed to. Nothing "
        "is sent anywhere except the model gateway, and only when you press Ask."
    )
    st.markdown(
        "**Credentials.** Held as secrets, masked in the logs at the formatter, and "
        "never rendered on any page of this application — including this one, which "
        "reports only whether each is present."
    )
    st.markdown(
        "**Prompt injection.** Text arriving from outside — retrieved passages, "
        "fetched abstracts, follow-up questions the agent proposes — is screened "
        "before it reaches the model or a button, by a deterministic check rather "
        "than by asking a model to police itself."
    )
    st.markdown(
        "**Cost is shown, and an estimate says it is one.** A price copied from a "
        "vendor's rate card and a price reasoned out from a neighbouring model are "
        "kept in separate tables and labelled differently on screen. A guess "
        "presented as an invoice is the failure this project exists to measure in "
        "other people's work."
    )


def developer_section() -> None:
    """Draw what a developer needs: versions, configuration, artefacts on disk.

    Nothing here is needed to use the application, which is the first thing it says.
    """
    st.caption(
        "**Nothing in this tab is needed to use Quay.** It is here for somebody "
        "checking that the results can be reproduced on their own machine: which "
        "versions this process is running, how it is configured, and what a run has "
        "written to disk. It used to be three separate items in the navigation, "
        "which put a dependency table in front of every visitor who only wanted an "
        "answer."
    )

    st.divider()
    st.markdown("##### Can the next piece of work run on this machine?")
    missing = missing_dependencies(panels.CURRENT_TIER)
    if missing:
        st.error(
            "Missing and needed now: **"
            + ", ".join(dependency.distribution for dependency in missing)
            + "**"
        )
        st.code("uv add qiskit qiskit-aer qiskit-ibm-runtime networkx cvxpy", language="bash")
        st.caption("`make sync` afterwards, then `make check`.")
    else:
        st.success(f"Everything needed up to tier {panels.CURRENT_TIER} is installed.")

    with st.expander("Every package, and what it is for"):
        st.dataframe(
            [
                {
                    "package": dependency.distribution,
                    "version": dependency.installed_version or "— not installed —",
                    "needed from tier": dependency.needed_from_tier,
                    "for": dependency.purpose,
                }
                for dependency in DEPENDENCIES
            ],
            hide_index=True,
            width="stretch",
        )

    st.divider()
    st.markdown("##### How this process is configured")
    st.caption(
        "Read through `Settings`, which is the only place configuration enters the "
        "process. Credentials are shown as present or absent and never as values."
    )

    try:
        settings = get_settings()
    except Exception as error:
        st.error(f"`Settings` would not load: {error}")
        st.caption(
            "That is the same failure the agent would hit at startup. Most often a missing "
            "`OPENROUTER_API_KEY` in `.env`."
        )
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Credentials**")
            st.markdown(
                f"- `OPENROUTER_API_KEY` — {panels.presence(bool(settings.openrouter_api_key))}"
            )
            st.markdown(
                f"- `LANGSMITH_API_KEY` — {panels.presence(bool(settings.langsmith_api_key))}"
            )
            st.markdown(f"- LangSmith tracing — {'on' if settings.langsmith_tracing else 'off'}")
        with right:
            st.markdown("**Model**")
            chat_card = card_for(settings.chat_model)
            st.markdown(
                f"- chat: `{settings.chat_model}`"
                + ("" if chat_card else "  ⚠️ not in the subscription catalogue")
            )
            st.markdown(f"- embeddings: `{settings.embedding_model}`")
            st.markdown(f"- temperature: `{settings.temperature}`")
            st.markdown(f"- call ceiling per run: `{settings.max_model_calls_per_run}`")

        st.markdown("**Where things are kept**")
        for label, value in (
            ("corpus", settings.corpus_path),
            ("vector store", settings.vector_store_path),
            ("checkpoints", settings.checkpoint_path),
            ("memory", settings.memory_path),
        ):
            if value is None:
                st.markdown(f"- {label}: *disabled*")
                continue
            exists = (PROJECT_ROOT / value).exists()
            st.markdown(
                f"- {label}: `{value}` — {panels.presence(exists, 'present', 'not created yet')}"
            )

    st.markdown(f"- corpus notes found: **{corpus_size()}**")

    st.divider()
    st.markdown("##### This process")
    st.markdown(f"- Python `{sys.version.split()[0]}`")
    st.markdown(f"- repository root `{PROJECT_ROOT}`")
    st.markdown(f"- `.env` read by settings: `{Settings.model_config.get('env_file')}`")
    st.caption(
        "The test suite switches that last line off at the model, so every test run sees "
        "the same configuration wherever it runs. Without it, a machine holding a local "
        "`.env` would be testing different settings from one that does not."
    )

    with st.expander(f"Every model this subscription can reach ({len(CATALOGUE)})"):
        st.caption(
            "The display names are what the subscription shows and are the part to "
            "trust; a slug marked *unconfirmed* has been reconstructed from the display "
            "name and never resolved against the live model list. A wrong slug fails at "
            "the first call, several minutes into a run — which is why it is shown here "
            "rather than discovered there."
        )
        unconfirmed = unconfirmed_slugs()
        if unconfirmed:
            st.warning(f"{len(unconfirmed)} of {len(CATALOGUE)} slugs have never been verified.")
        st.dataframe(
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
            ],
            hide_index=True,
            width="stretch",
        )

    st.divider()
    st.markdown("##### What a run has actually left on disk")
    st.caption(
        "Listed, not interpreted. Parsing a scorecard that does not exist yet would "
        "mean inventing its format, and a page that renders a plausible layout for "
        'absent data is worse than one that says "nothing yet": the first is '
        "indistinguishable from a working system, the second is a to-do list."
    )

    found = artefacts()

    if not found:
        panels.empty_state(
            "No run has written anything yet.",
            "These are the directories being watched, and what each is for:\n\n"
            + "".join(
                f"- `{directory}/` — {ARTEFACT_PURPOSE[directory]}\n"
                for directory in ARTEFACT_DIRECTORIES
            )
            + "\n`make evals` writes the first of those, `make campaign` the rest.",
        )
    else:
        st.caption(f"{len(found)} file(s), newest first.")
        st.dataframe(
            [
                {
                    "file": artefact.path,
                    "size": panels.humanise_bytes(artefact.size_bytes),
                    "modified (UTC)": artefact.modified.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for artefact in found
            ],
            hide_index=True,
            width="stretch",
        )

        chosen = st.selectbox(
            "Preview a text artefact",
            options=[a.path for a in found if a.path.endswith((".md", ".json", ".qasm", ".txt"))],
            index=None,
            placeholder="pick a file",
        )
        if chosen:
            text = (PROJECT_ROOT / chosen).read_text(encoding="utf-8", errors="replace")
            if chosen.endswith(".md"):
                st.markdown(text)
            else:
                st.code(text, language="json" if chosen.endswith(".json") else None)

    st.caption("Watching: " + " · ".join(f"`{directory}`" for directory in ARTEFACT_DIRECTORIES))


panels.footer("src/ui/status.py")
