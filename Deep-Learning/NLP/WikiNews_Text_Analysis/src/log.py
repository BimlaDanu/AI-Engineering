"""Shared logging setup."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

# These libraries log every HTTP request at INFO, which buries the stage output.
_NOISY = ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers")


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing timestamped lines to stderr.

    Args:
        name: Logger name, normally the calling module's ``__name__``.

    Returns:
        A configured logger. Handlers are installed once per process.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        for noisy in _NOISY:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
