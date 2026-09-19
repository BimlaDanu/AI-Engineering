"""Scalar conversion helpers.

pandas types values read out of a DataFrame as a broad scalar union, so passing
one into ``int()`` or ``str()`` does not type-check. These narrow it in one
place instead of a ``# type: ignore`` at every call site.
"""

from __future__ import annotations

from typing import Any


def as_int(value: Any) -> int:
    """Narrow a pandas scalar to ``int``.

    Args:
        value: A value read out of a DataFrame.

    Returns:
        The value as a plain Python ``int``.

    Raises:
        TypeError: If the value is not integer-like.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Expected an integer-like value, got {value!r}") from error


def as_str(value: Any) -> str:
    """Narrow a pandas scalar to ``str``.

    Args:
        value: A value read out of a DataFrame.

    Returns:
        The value as a plain Python ``str``.
    """
    return str(value)
