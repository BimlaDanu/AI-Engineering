"""Filing an evaluation run with LangSmith.

Separated from :mod:`src.evals.run` so that everything which can fail because of
somebody else's service lives in one module, behind two functions that are
documented never to raise. The local scorecard is the authority; this is how a run
becomes comparable with the run before it.

What it does, in order:

*Names a project.* Evaluation traces go to ``<project>-evals`` rather than to the
project live questions are filed under. Mixing them would make both useless: a
regression hunt would be reading a stranger's afternoon, and a usage question would
be reading the suite.

*Uploads the suite as a dataset.* One example per case, keyed by name, created only
if it is not already there. The suite is frozen, so this is idempotent by
construction -- and if it ever is not, the mismatch is worth seeing in LangSmith
rather than papering over.

*Records the scores as an experiment.* ``langsmith.evaluation.evaluate`` files one
run per example, and the function it calls does not re-run the agent -- it looks up
the result the local suite already produced. That matters twice: the suite is not
executed twice for one report, and the score LangSmith stores is by construction the
**same** :func:`src.evals.scoring.grade` verdict the scorecard shows. Two graders
would eventually disagree, and then "did this commit regress?" would have two
answers.

Every step is optional and every step is guarded. A missing key, an unreachable
host, an API that changed shape between versions -- each becomes a note on the
scorecard, which is the honest outcome: the measurement happened, the filing did
not.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from src.evals.cases import CASES
from src.evals.scoring import Scorecard
from src.logging_setup import get_logger
from src.settings import Settings, get_settings

LOG = get_logger("evals.tracking")

DATASET_NAME = "quantumlab-tfim-suite"
"""Dataset the cases are uploaded to. One per project, reused across runs."""

PROJECT_SUFFIX = "-evals"
"""Appended to the configured project name for evaluation traces."""


def _settings() -> Settings | None:
    """Read the process settings, or ``None`` when there are none to read."""
    try:
        return get_settings()
    except Exception:  # pragma: no cover - unconfigured environment
        return None


def is_configured(settings: Settings | None = None) -> bool:
    """Whether a LangSmith run can be filed at all.

    Args:
        settings: Configuration to read, defaulting to the process settings.

    Returns:
        Whether a credential is present. Tracing being switched *off* does not
        matter here: asking for an evaluation is an explicit request to record one,
        which is different from tracing every question a user asks.
    """
    resolved = settings if settings is not None else _settings()
    return resolved is not None and resolved.langsmith_api_key is not None


def begin(settings: Settings | None = None) -> tuple[str, ...]:
    """Point tracing at the evaluation project before the suite runs.

    Args:
        settings: Configuration to read, defaulting to the process settings.

    Returns:
        Notes for the scorecard: empty when tracing was configured, and one line
        saying so when it was not. Never raises.
    """
    resolved = settings if settings is not None else _settings()
    if resolved is None or resolved.langsmith_api_key is None:
        return ("LangSmith: no credential configured, so this run was scored locally only.",)
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = resolved.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = f"{resolved.langsmith_project}{PROJECT_SUFFIX}"
    LOG.info("evals_tracing_on", extra={"project": os.environ["LANGSMITH_PROJECT"]})
    return ()


def publish(card: Scorecard, settings: Settings | None = None) -> Scorecard:
    """File the scored suite with LangSmith.

    Args:
        card: The local scorecard, already graded.
        settings: Configuration to read, defaulting to the process settings.

    Returns:
        The scorecard, with a note added saying what was filed or why nothing was.
        The results themselves are never modified -- this function cannot change a
        grade, which is the point of it being a separate step.
    """
    resolved = settings if settings is not None else _settings()
    if resolved is None or resolved.langsmith_api_key is None:
        return card
    try:
        note = _upload(card, resolved)
    except Exception as error:
        LOG.warning("evals_publish_failed", extra={"error_type": type(error).__name__})
        note = (
            f"LangSmith: the run was not filed ({type(error).__name__}). "
            "The scores below are the local ones and are unaffected."
        )
    # `replace` rather than a fresh Scorecard: rebuilding it field by field dropped
    # `model_calls`, which defaults to zero, and the scorecard then reported "No
    # model was called: every answer here came from the deterministic path" on a run
    # that had just made about a hundred live calls. Adding a note must not be able
    # to change what the card says about the run.
    return replace(card, notes=(*card.notes, note))


def _upload(card: Scorecard, settings: Settings) -> str:
    """Create the dataset and the feedback, returning what to note.

    Args:
        card: The scored suite.
        settings: Configuration carrying the credential and project name.

    Returns:
        One line for the scorecard.

    Raises:
        Exception: Anything the client raises. :func:`publish` is what turns that
            into a note; keeping this function free of its own error handling is
            what keeps the two concerns apart.
    """
    from langsmith import Client  # imported lazily: the extra may not be installed
    from langsmith.evaluation import evaluate

    key = settings.langsmith_api_key
    if key is None:  # pragma: no cover - `publish` has already checked
        raise RuntimeError("no LangSmith credential")
    client = Client(api_key=key.get_secret_value())
    _dataset(client)
    graded = {result.case.name: result for result in card.results}

    def replay(inputs: dict[str, Any]) -> dict[str, Any]:
        """Return what the local suite already found for this example.

        Not a second run of the agent. The suite has finished by the time this is
        called, and re-running it here would double the cost of a report and open
        the door to LangSmith recording a different verdict from the scorecard.
        """
        result = graded.get(str(inputs.get("case")))
        if result is None:
            return {"passed": False, "status": "not run", "failures": ["case not in this run"]}
        return {
            "passed": result.passed,
            "status": result.status,
            "nodes": list(result.nodes),
            "energy": result.energy,
            "failures": [f"expected {check.claim}; {check.detail}" for check in result.failures],
        }

    def passed(run: Any, example: Any) -> dict[str, Any]:
        """Score one example: one when every check held, zero otherwise."""
        outputs = run.outputs or {}
        return {"key": "passed", "score": int(bool(outputs.get("passed")))}

    evaluate(
        replay,
        data=DATASET_NAME,
        evaluators=[passed],
        client=client,
        experiment_prefix="quantumlab-suite",
        metadata={"model": card.model or "none", "cases": card.total},
    )
    project = f"{settings.langsmith_project}{PROJECT_SUFFIX}"
    return (
        f"LangSmith: filed as an experiment over dataset `{DATASET_NAME}`; "
        f"agent traces are under project `{project}`."
    )


def _dataset(client: Any) -> Any:
    """Fetch the suite's dataset, creating it and its examples on first use.

    Args:
        client: A LangSmith client.

    Returns:
        The dataset. Examples carry the case name in their metadata, which is what
        a later run matches on -- a name rather than an index, so inserting a case
        does not re-label every case after it.
    """
    try:
        return client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Held-out behavioural suite for the QuantumLab Copilot agent: routing, "
                "refusals, verification, exact limits and memory."
            ),
        )
        # The case name goes in the *inputs*, not only the metadata: it is what the
        # replay function matches on, and an experiment's target sees inputs.
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"case": case.name, "question": case.question} for case in CASES],
            outputs=[{"expectation": case.why} for case in CASES],
            metadata=[{"case": case.name, "family": case.family} for case in CASES],
        )
        LOG.info("evals_dataset_created", extra={"cases": len(CASES)})
        return dataset
