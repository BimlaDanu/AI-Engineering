"""Switching LangSmith tracing on, or deliberately off.

LangChain looks for its tracing configuration in the process environment, not in
an argument, so "enable tracing" means "set four variables before the first call".
That is easy to do in a notebook and easy to get wrong in an application, because
the failure is silent in both directions: a missing key traces nothing, and a
stale ``LANGSMITH_TRACING=true`` left in a shell traces everything to whichever
project happens to be named.

So this module owns the decision and states it out loud. It reads
:attr:`src.settings.Settings.tracing_enabled` -- which is true only when the
switch is on *and* a credential exists -- and writes the environment to match.
When tracing is not enabled it writes ``false`` rather than leaving the variable
alone, because the common case for "on but not credentialed" is an environment
someone else configured, and a per-call authentication warning is a worse outcome
than no traces.

Tracing is off by default. This project should not ship a default that
sends every question anyone types to a third-party service.

Nothing here logs the credential. The value moves from
:class:`~pydantic.SecretStr` into ``os.environ`` and is never rendered.
"""

from __future__ import annotations

import os

from src.logging_setup import get_logger
from src.settings import Settings, get_settings

TRACING_VARIABLE = "LANGSMITH_TRACING"
"""The master switch LangChain reads at call time."""

_KEY_VARIABLE = "LANGSMITH_API_KEY"
_PROJECT_VARIABLE = "LANGSMITH_PROJECT"


def configure_tracing(settings: Settings | None = None) -> bool:
    """Point LangSmith tracing at the configured project, or turn it off.

    Safe to call as often as you like -- it writes the same values each time --
    which is what makes it callable from :func:`src.agent.graph.run_campaign` rather than
    from a start-up hook a caller could forget.

    Args:
        settings: Configuration to read. Defaults to the process settings.

    Returns:
        ``True`` if tracing is now on. ``False`` covers both "switched off" and
        "switched on but with no credential to authenticate it", which are the
        same outcome and are logged differently.
    """
    resolved = get_settings() if settings is None else settings
    logger = get_logger("tracing")

    if not resolved.tracing_enabled:
        os.environ[TRACING_VARIABLE] = "false"
        if resolved.langsmith_tracing:
            # Asked for, but unusable: worth a line, because the person who set
            # the switch is expecting traces and will otherwise wonder where
            # they went.
            logger.warning("tracing_disabled", extra={"reason": "no_langsmith_api_key"})
        return False

    key = resolved.langsmith_api_key
    if key is None:  # pragma: no cover - tracing_enabled already implies a key
        return False
    os.environ[TRACING_VARIABLE] = "true"
    os.environ[_KEY_VARIABLE] = key.get_secret_value()
    os.environ[_PROJECT_VARIABLE] = resolved.langsmith_project
    logger.info("tracing_enabled", extra={"project": resolved.langsmith_project})
    return True
