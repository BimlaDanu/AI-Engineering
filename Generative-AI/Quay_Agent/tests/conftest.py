"""Give every test run the same configuration, wherever it runs.

A `.env` file is read by pydantic-settings automatically, so a test that
constructs `Settings()` without passing a credential picks one up where such a
file exists and finds nothing where it does not. The test then passes in one place
and fails in another, and the difference is a file nobody thinks of as an input.

Two measures close that gap, and both are session-wide so no test can forget them:

* `.env` is switched off at the model. `Settings.model_config["env_file"] = None`
  leaves the process environment as the only source of configuration, and this
  file controls that completely.
* Every environment variable the settings model reads is deleted, then the
  credentials are set to obvious placeholders. Nothing here reaches a gateway.

A test that wants the *unconfigured* case asks for the `unconfigured_environment`
fixture rather than relying on the surrounding environment.

One consequence is worth knowing. A placeholder credential is enough to *build* a
client and not enough to get an answer from one, so code that reaches for a
network service stalls rather than failing. Tests for such code switch the service
off explicitly -- ``ModelPool(offline=True)``, or a double -- instead of relying on
the absence of a key.

`NO_WAITING` shortens the timeout, the retry count and the throttle session-wide,
so a test that forgets to switch a service off costs about a second instead of a
minute. Shortening rather than blocking: some tests are about what happens when a
call fails and need it to actually fail, and a blocked socket would take Chroma
and the tracing client with it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from tempfile import mkdtemp

import pytest

from src.logging_setup import LOG_PATH_VARIABLE
from src.settings import Settings

PLACEHOLDER = "not-a-real-key"

# Credentials the suite hands out by default. Placeholders: no call leaves the process.
CREDENTIALS = {
    "OPENROUTER_API_KEY": PLACEHOLDER,
    "LANGSMITH_API_KEY": PLACEHOLDER,
}

# Where a test that promotes a fetched note is allowed to write it: never the
# repository. A note written into `promoted/` survives the run, and `load_library`
# counts the corpus plus the promoted notes, so the next run of
# `test_the_corpus_is_published_whole` fails on a count nobody changed. Tests that
# take the fetching path stub the network call as well; this is the second line.
PROMOTED_ELSEWHERE = mkdtemp(prefix="quay-tests-promoted-")

# Settings that make a forgotten network call cheap. The field names are the
# variable names -- `Settings` sets no env prefix -- so these override the defaults
# for every test in the session.
NO_WAITING = {
    # A test may write a promoted note; it may not write one into the repository.
    "PROMOTED_PATH": PROMOTED_ELSEWHERE,
    # One second, not sixty. Nothing in the suite is supposed to be waiting on a
    # gateway, so any wait at all is a test that forgot to switch the model off.
    "REQUEST_TIMEOUT_S": "1.0",
    # No retries. A retry only helps a transient failure, and in the suite there is
    # nothing transient to recover from -- the credential is a placeholder every
    # time. Retrying it three times with backoff is what made one forgotten switch
    # cost a hundred seconds.
    "MAX_RETRIES": "0",
}
"""Two settings, and deliberately not a third.

The client-side throttle is left alone. It only sleeps when a call actually goes
out, which in this suite means only the calls that forgot to switch the model off
-- and those now cost a second each anyway. Overriding it here would also make
``test_a_throttle_is_installed_by_default`` fail, and that test's subject *is* the
default: a session-wide override that turns the thing under test off is not a
speed-up, it is a suite that stopped checking.
"""


# --------------------------------------------------------------------------
# Set at import, not in a fixture, and the difference is the whole point
# --------------------------------------------------------------------------

os.environ.setdefault(LOG_PATH_VARIABLE, "")
"""Switch the interface's log file off for the suite.

Same intent as ``PROMOTED_PATH`` above -- a test may log, it may not log into the
repository -- and a different mechanism, because this one has to happen *earlier*
than a fixture can. ``NO_WAITING`` is applied by ``_hermetic_configuration``, which
is a session fixture and therefore runs after collection. That is too late here:
``test_ui_pages`` imports a page module, importing a Streamlit page module *runs*
the page, and running it calls ``panels.current_setting`` -> ``start_logging``. So
the interface opens its log file while pytest is still collecting, before the first
fixture of the first test exists.

This was found the honest way. The first attempt was a ``monkeypatch`` fixture on the
path, it passed its own test, and the log file appeared in ``reports/`` anyway --
``pytest --collect-only`` alone was enough to create it, which is what named the
cause. A module-level statement in ``conftest.py`` runs when pytest imports this
file, which is before it imports any test module.

``setdefault`` rather than an assignment, so somebody debugging a suite run can point
the log at a real file from their shell and actually get one.
"""


def _variables_the_settings_model_reads() -> set[str]:
    """Every environment variable `Settings` would consult, derived from the model.

    Read off `model_fields` rather than hard-coded, so a field added later is
    neutralised without anyone remembering to update this list.
    """
    prefix = str(Settings.model_config.get("env_prefix") or "")
    names: set[str] = set()
    for name, field in Settings.model_fields.items():
        names.add(f"{prefix}{name}".upper())
        alias = getattr(field, "alias", None) or getattr(field, "validation_alias", None)
        if isinstance(alias, str):
            names.add(alias.upper())
    return names


@pytest.fixture(scope="session", autouse=True)
def _hermetic_configuration() -> Iterator[None]:
    """Shut out `.env` and the developer's shell for the whole session."""
    Settings.model_config["env_file"] = None

    saved = dict(os.environ)
    for name in _variables_the_settings_model_reads():
        os.environ.pop(name, None)
    os.environ.update(CREDENTIALS)
    os.environ.update(NO_WAITING)
    _forget_cached_settings()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        _forget_cached_settings()


@pytest.fixture(scope="session", autouse=True)
def _corpus_opened_once_per_worker(_hermetic_configuration: None) -> None:
    """Open the vector store once, on one thread, before any test searches it.

    The collection is opened per search rather than held open, and
    :func:`src.rag.retrieve.warm_corpus` exists because of what that costs when
    several searches start at once on a process that has not opened it yet: they
    race, the losers log ``retrieval_store_unavailable`` and answer without the
    corpus. The chat page calls it for exactly this reason; the suite did not.

    That was harmless while the suite ran one test at a time and stopped being
    harmless when it went parallel -- each worker is its own process with its own
    cold collection, so the first few corpus tests on each worker raced, and the
    ones that lost produced a campaign with **no search rounds at all**. Two
    different tests in ``test_ui_pages.py`` failed intermittently on that, roughly
    one run in three, which is worse than a test that fails: a suite that fails
    sometimes is a suite nobody reads.

    Ordered after :func:`_hermetic_configuration` by depending on it, so the warm
    happens against the suite's configuration rather than the developer's.

    Nothing is raised. A store that will not open at all -- which is every fresh
    checkout, since the index is built by ``make ingest`` and is not in the
    repository -- is a supported outcome, and the keyword half of the search
    answers without it.
    """
    from src.rag.retrieve import warm_corpus

    warm_corpus()


@pytest.fixture
def unconfigured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every credential, for tests that assert the no-key behaviour.

    A blank value is used rather than a deletion where the settings model rejects
    blanks, because that is what a CI runner with an unset secret actually produces.
    """
    for name in CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    _forget_cached_settings()


@pytest.fixture(autouse=True)
def _no_open_form_left_by_an_import() -> None:
    """Close the form Streamlit leaves open on its global container.

    Importing a page module runs the page, and a page run outside a script context
    has nowhere to put a block: ``st.form`` records itself on the global container
    instead of on the block it opened, and stays there. Every later rendered page
    then meets a form that is still open and refuses the first button in it.

    The running application always has a script context, so it never leaks. Only the
    suite does, and only because it imports a page module to read a constant.
    """
    import streamlit as st

    for generator in (st._main, st.sidebar):
        # Private because Streamlit offers no way to ask; a leak it has no API for
        # is a leak it has no API to clear.
        generator._form_data = None


def _forget_cached_settings() -> None:
    """Drop any memoised `Settings`, so the next read sees the current environment."""
    try:
        from src.settings import get_settings
    except ImportError:  # pragma: no cover - settings module always exists
        return
    cache_clear = getattr(get_settings, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
