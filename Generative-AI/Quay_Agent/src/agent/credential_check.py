"""Answer whether a credential will work, before a campaign is spent finding out.

The interface lets a visitor supply their own gateway key. An accepted-but-wrong
key returns 401 on every model call, and the campaign survives that -- every
number is computed and cross-checked before a model is consulted -- so the
browser shows a correct verdict that no language model contributed to. That is
the failure this prevents.

The key is checked once, at the moment it is offered, against ``GET /key``, which
returns the credential's own label and allowance and consumes no tokens.

Three answers, not two: accepted, refused, or unverifiable. Refusing a valid key
because the check itself failed would make a working deployment unusable from a
flaky connection, so an unverifiable key is accepted with the reason reported. A
spent balance is distinguished too, since it fails exactly as a wrong key does.

Nothing here logs, returns or stores the key. It goes into one Authorization
header and out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.logging_setup import get_logger
from src.settings import get_settings

_logger = get_logger("agent.credential_check")

KEY_PREFIX = "sk-or-"
"""What an OpenRouter key starts with.

Checked before the network is touched, so an obvious paste error -- a key for
another service, a truncated fragment, a whole URL -- is named as one instead of
being reported as a rejected credential. The prefix rather than the full
``sk-or-v1-`` so that a future key version is not refused by this project.
"""

TIMEOUT_S = 10.0
"""How long to wait for the check.

Short on purpose: this runs with somebody watching a form. A gateway that has not
answered in ten seconds has not refused the key, and the caller is told exactly
that.
"""


@dataclass(frozen=True, slots=True)
class KeyStatus:
    """What asking the gateway about a credential established.

    Attributes:
        usable: Whether a campaign may be run with this key. True for a key the
            gateway accepted *and* for one that could not be checked, because the
            alternative is refusing valid keys whenever the check itself fails.
        checked: Whether the gateway actually answered. Read alongside
            :attr:`usable`: the pair ``(True, False)`` is "no reason to refuse
            this", which is a weaker claim than "this works".
        reason: One sentence for a person, in plain words, always populated --
            including on success, because a form that says nothing on success
            looks like a form that did nothing.
        credit_remaining: The allowance left on the key in US dollars, when the
            gateway reports one. ``None`` for a key with no limit set, which is
            the common case and is not a problem.
    """

    usable: bool
    checked: bool
    reason: str
    credit_remaining: float | None = None

    @property
    def spent(self) -> bool:
        """Whether the key is valid but has no allowance left.

        Returns:
            True when the gateway reported a remaining allowance of zero or less.
            Kept separate from :attr:`usable`: the key is real and the deployment
            may well want to say so differently from "that key is wrong".
        """
        return self.credit_remaining is not None and self.credit_remaining <= 0.0


def check_key(key: str, base_url: str | None = None) -> KeyStatus:
    """Ask the gateway whether it will accept this credential.

    Args:
        key: The credential to check. Never logged and never returned.
        base_url: The gateway's API root. Defaults to the configured one, so a
            deployment pointed at a proxy checks the key against the proxy.

    Returns:
        The verdict. See :class:`KeyStatus` for why an unverifiable key comes back
        usable.

    Examples:
        >>> check_key("").usable
        False
        >>> check_key("not-a-key").reason
        "That does not look like an OpenRouter key -- they begin with 'sk-or-'."
    """
    trimmed = key.strip()
    if not trimmed:
        return KeyStatus(usable=False, checked=False, reason="No key was given.")
    if not trimmed.startswith(KEY_PREFIX):
        return KeyStatus(
            usable=False,
            checked=False,
            reason=(
                f"That does not look like an OpenRouter key -- they begin with '{KEY_PREFIX}'."
            ),
        )
    root = (base_url or get_settings().openrouter_base_url).rstrip("/")
    try:
        response = httpx.get(
            f"{root}/key",
            headers={"Authorization": f"Bearer {trimmed}"},
            timeout=TIMEOUT_S,
        )
    except Exception as error:
        # The key is not refused on this path, and the reason says why rather than
        # implying the key was the problem.
        _logger.warning(
            "credential_check_unreachable",
            extra={"detail": type(error).__name__, "endpoint": f"{root}/key"},
        )
        return KeyStatus(
            usable=True,
            checked=False,
            reason=(
                "The key could not be checked -- the gateway did not answer. "
                "It has been accepted unverified."
            ),
        )
    return _read_response(response, root)


def _read_response(response: httpx.Response, root: str) -> KeyStatus:
    """Turn the gateway's reply into a verdict.

    Args:
        response: What ``GET /key`` returned.
        root: The API root, for the log line only.

    Returns:
        The verdict.
    """
    if response.status_code in (401, 403):
        _logger.info("credential_check_refused", extra={"status": response.status_code})
        return KeyStatus(
            usable=False,
            checked=True,
            reason="The gateway did not accept that key. Check it and paste it again.",
        )
    if response.status_code != 200:
        _logger.warning(
            "credential_check_unexpected",
            extra={"status": response.status_code, "endpoint": f"{root}/key"},
        )
        return KeyStatus(
            usable=True,
            checked=False,
            reason=(
                f"The key could not be checked -- the gateway answered "
                f"{response.status_code}. It has been accepted unverified."
            ),
        )
    remaining = _remaining(response)
    if remaining is not None and remaining <= 0.0:
        return KeyStatus(
            usable=True,
            checked=True,
            reason=(
                "That key is valid but has no credit left, so model calls will fail. "
                "Answers will still be computed and cross-checked."
            ),
            credit_remaining=remaining,
        )
    return KeyStatus(
        usable=True,
        checked=True,
        reason="The gateway accepted that key.",
        credit_remaining=remaining,
    )


def _remaining(response: httpx.Response) -> float | None:
    """Read the allowance left on the key, if the gateway states one.

    Args:
        response: A successful ``GET /key`` reply.

    Returns:
        Dollars remaining, or ``None`` when the key has no limit or the field is
        missing or unreadable. Defensive throughout: a gateway that changes this
        field's shape must not turn a working key into a refused one.
    """
    try:
        data = response.json().get("data", {})
        limit = data.get("limit")
        used = data.get("usage")
        stated = data.get("limit_remaining")
        if stated is not None:
            return float(stated)
        if limit is None:
            return None
        return float(limit) - float(used or 0.0)
    except Exception:
        return None
