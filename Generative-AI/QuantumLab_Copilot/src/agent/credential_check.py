"""Is this key any good? Asked once, before a pasted credential is accepted.

A visitor who pastes their own OpenRouter key into the top bar is handing the
application the thing every model call depends on, and the cost of taking a bad
one is unusually high **here specifically**: this agent computes and cross-checks
every number before it consults a model, so a key that fails produces an answer
that is still numerically correct and merely written by the offline fallback. The
failure is invisible. Nothing errors, nothing is missing, the prose is just
quietly worse -- which is the hardest kind of defect to notice and the easiest to
prevent, by asking the gateway before storing anything.

The question is put to ``GET /api/v1/key``, which is the endpoint OpenRouter
publishes for exactly this and which costs nothing and charges nothing: it reports
on the key presented in the header rather than running a model.

**Unverifiable is not the same as bad.** A laptop that is offline, a gateway
having a bad minute and a proxy that swallows the request all produce "no answer",
and refusing the key in those cases would mean a visitor with a perfectly good
credential is told it is invalid by an application that never managed to ask.
So the three outcomes are kept apart -- accepted, refused, accepted-unchecked --
and :attr:`KeyStatus.checked` is the field that says which of the two acceptances
happened, so the interface can say so too.

Standard library only, deliberately. ``httpx`` is in the environment as somebody
else's transitive dependency and importing it here would make this module depend
on a package this project has never declared.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.logging_setup import get_logger
from src.settings import DEFAULT_BASE_URL, get_settings

CHECK_TIMEOUT_S = 8.0
"""Seconds to wait for the gateway before giving up and accepting unchecked.

Short, because this runs while somebody watches a spinner in a popover, and the
cost of giving up is small -- the key is taken anyway and labelled as unverified.
"""

MIN_KEY_CHARACTERS = 20
"""Shortest string worth spending a network round trip on.

An OpenRouter key is far longer than this. The check is not a format rule and is
not trying to be one -- gateways change their prefixes and a validator that knows
better than the gateway is a validator that eventually rejects a working key. It
only catches the paste that obviously went wrong.
"""

LOG = get_logger("credential_check")


@dataclass(frozen=True, slots=True)
class KeyStatus:
    """What was decided about a pasted credential.

    Attributes:
        usable: Whether to store it and answer with it.
        checked: Whether the gateway actually confirmed it. ``False`` with
            ``usable`` true is the honest middle case: nobody could be reached, so
            the key was taken on trust.
        reason: One sentence for the visitor. Never contains the key.
    """

    usable: bool
    checked: bool
    reason: str


def _gateway_url() -> str:
    """Where to put the question.

    Returns:
        The endpoint this deployment actually calls, and :data:`DEFAULT_BASE_URL`
        only when there is no configuration to read -- which is the case on a
        checkout with no credential, and therefore the case of the visitor most
        likely to be pasting one.

        Asking the configured gateway rather than the default one matters for two
        separate reasons, and the second is the important one. A deployment
        pointed at an OpenAI-compatible proxy holds keys that proxy issued, and
        checking those against openrouter.ai would refuse every one of them. It
        would also **send the pasted credential to a host this deployment never
        configured**, which is not ours to do with somebody else's key. A proxy
        that does not implement ``/key`` answers something other than 200 and the
        key is accepted unchecked, which is the right outcome: unverifiable, not
        bad.
    """
    try:
        configured = get_settings().openrouter_base_url
    except Exception:
        return DEFAULT_BASE_URL
    return configured.rstrip("/") if configured else DEFAULT_BASE_URL


def _probe(key: str) -> tuple[int | None, str | None]:
    """Ask the gateway what it thinks of this key.

    Args:
        key: The credential, already stripped.

    Returns:
        The HTTP status and the label OpenRouter has on the key, either of which
        may be ``None`` when the request did not complete or said nothing useful.
        Exceptions are swallowed on purpose: every one of them means the same
        thing to the caller, which is "no answer", and the distinction between a
        DNS failure and a timeout is not one a visitor can act on.
    """
    request = urllib.request.Request(
        f"{_gateway_url()}/key",
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=CHECK_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data") if isinstance(payload, dict) else None
            label = data.get("label") if isinstance(data, dict) else None
            return int(response.status), str(label) if label else None
    except urllib.error.HTTPError as refused:
        return int(refused.code), None
    except Exception:
        return None, None


def check_key(typed: str) -> KeyStatus:
    """Decide whether a pasted credential should be accepted.

    Args:
        typed: Exactly what was pasted, unstripped. Surrounding whitespace is a
            copy-and-paste artefact rather than part of the key, so it is removed
            before anything else looks at it.

    Returns:
        The verdict. The key itself never appears in
        :attr:`KeyStatus.reason` and is never logged -- what is logged is which
        of the four branches below was taken.

    Examples:
        >>> check_key("   ").usable
        False
        >>> check_key("short").reason
        'That does not look like a key — check the paste.'
    """
    key = typed.strip()
    if not key:
        return KeyStatus(False, True, "Paste a key first.")
    if len(key) < MIN_KEY_CHARACTERS or any(character.isspace() for character in key):
        return KeyStatus(False, True, "That does not look like a key — check the paste.")

    status, label = _probe(key)
    if status is None:
        LOG.info("key_check_unreachable")
        return KeyStatus(
            True,
            False,
            "Could not reach OpenRouter to check that key, so it has been accepted "
            "as it is. If answers stop sounding like the model, the key is the "
            "first thing to suspect.",
        )
    if status in (401, 403):
        LOG.info("key_check_refused", extra={"status": status})
        return KeyStatus(False, True, "OpenRouter rejected that key.")
    if status != 200:
        LOG.info("key_check_inconclusive", extra={"status": status})
        return KeyStatus(
            True,
            False,
            f"OpenRouter answered {status} rather than yes or no, so the key has "
            "been accepted unchecked.",
        )
    LOG.info("key_check_accepted")
    named = f" ({label})" if label else ""
    return KeyStatus(True, True, f"Key accepted{named}. Answers now run on it.")
