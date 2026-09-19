"""Who is visiting Synapse, and whose key pays for their questions.

Three independent doors, in the order :func:`visitor` prefers them:

===============  ===========================================  ===========================
Level            What it means                                How they got there
===============  ===========================================  ===========================
``signed_in``    Identified by an OpenID provider             pressed **Log in** / **Sign
                                                              up for free** in the top bar
``own_key``      Anonymous, spending their own credential     pasted an OpenRouter key
                                                              behind **Use a key**
``password``     Identified by the built-in username /        typed credentials into the
                 password gate                                full-page login form
``open``         No door is configured at all                 running ``make run`` locally
===============  ===========================================  ===========================

**Nothing is gated on any of this.** Every page, the whole knowledge base, the tools and
the evaluation harness are reachable at every level, because none of that spends a model
call the host is billed for. The levels exist so a deployed Synapse can *show whose
session this is* and let a visitor put their own credential behind their own questions —
not to withhold features from anybody.

Native OIDC (``st.login`` / ``st.user``) is the deployment path
------------------------------------------------------------------
Streamlit has spoken OpenID Connect since 1.42. It needs ``authlib`` installed and an
``[auth]`` section in secrets naming a provider (Google, Auth0, Microsoft, Okta). That is
the whole configuration: there is no user table to keep, no password hash to store, and
**sign-up is real** — a visitor without an account creates one at the provider and returns
here signed in. Both top-bar buttons therefore start the same flow, which is what pressing
*Sign up for free* on a site like this one has always done; two buttons because they answer
two different questions ("I have an account" and "I do not"), and a visitor shown only
*Log in* assumes they need one already.

The built-in password gate is the fallback, and it reads ``[password_auth]``
--------------------------------------------------------------------------------
It used to read ``[auth]``, which is now Streamlit's own. The two cannot share the section:
Streamlit reads any unrecognised sub-table of ``[auth]`` as the *name of a provider*, so a
leftover ``[auth.credentials]`` would be parsed as a provider called "credentials". Config
under ``[password_auth]`` is therefore preferred, and a legacy ``[auth]`` is still accepted
**only when it carries a ``credentials`` table** — which native OIDC config never does, so
the fallback cannot misfire on an OIDC deployment. Existing installs keep working.

Degrading when nothing is configured
------------------------------------
A checkout has no secrets and ``make run`` has to work on a laptop with none, so every
"is this configured?" question is answered defensively before anything is drawn, and the
account buttons render **disabled with the reason in the tooltip**. Disabled rather than
hidden (a deployment that looks like it has no accounts is indistinguishable from a broken
one) and rather than live (pressing one would raise).

Credentials are never written anywhere. A pasted key lives in one browser session's
server-side store, is never logged, never put in a URL, and is gone when the tab closes.
"""

from __future__ import annotations

import copy
import html
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import streamlit as st
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Session-state keys we own. The public one (``user``) is seeded in src.ui.state.init_state so
# the rest of the app can read it unconditionally; the private one caches the authenticator so
# login and logout share a single instance (as the library requires).
USER_KEY = "user"
_AUTHENTICATOR_KEY = "_synapse_authenticator"

OWN_KEY_STATE = "own_api_key"
"""Session state holding a credential the visitor pasted.

Session state and nothing else: it lives in one browser session's server-side store, is
never written to disk, never logged, and never put in a URL. A key pasted here is gone when
the tab closes, which is the only promise about somebody else's credential this application
is in a position to keep.
"""

KEY_CHECK_STATE = "own_api_key_verdict"
"""Session state holding the verdict on the last credential pasted — the verdict, not the key.

It exists because accepting a key reruns the script, and a message written before the rerun
is discarded with the rest of that run's output.
"""

# Keys Streamlit itself owns inside ``[auth]``. Everything else at that level is either a
# single provider's settings or, when it is a table, the name of one.
_OIDC_RESERVED = frozenset({"redirect_uri", "cookie_secret", "client_kwargs"})

# The two names below are TOML section headings, not credentials — S105 matches on the
# substring "password" in the variable name.
_PASSWORD_SECTION = "password_auth"  # noqa: S105
_LEGACY_PASSWORD_SECTION = "auth"  # noqa: S105  (pre-OIDC; honoured only with credentials)

_DEFAULT_COOKIE_NAME = "synapse_auth"
_DEFAULT_COOKIE_EXPIRY_DAYS = 30


class User(BaseModel):
    """The currently authenticated user (typed so pages never poke raw session-state strings)."""

    username: str
    name: str


@dataclass(frozen=True)
class AuthConfig:
    """Parsed, validated auth configuration.

    ``credentials`` is the nested mapping ``streamlit_authenticator`` expects
    (``{"usernames": {"<user>": {"name": ..., "password": ...}}}``); the cookie fields drive the
    signed session cookie that keeps a login alive across refreshes.
    """

    credentials: dict
    cookie_name: str
    cookie_key: str
    cookie_expiry_days: int


def _to_plain(obj: object) -> object:
    """Deep-convert Streamlit's ``AttrDict`` secrets (and any nested mappings) to plain dicts.

    ``streamlit_authenticator`` mutates the credentials it is given (auto-hashing passwords in
    place); handing it Streamlit's live secrets object could raise or corrupt it, so we always
    pass a plain, freely-copyable structure.
    """
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _read_raw() -> dict | None:
    """Return the raw password-gate config from the environment or ``st.secrets``, or ``None``.

    The ``SYNAPSE_AUTH_CONFIG`` environment variable (JSON) wins when set — it is the escape
    hatch for deploy targets without a secrets file. Any failure (missing file, malformed JSON,
    secrets backend unavailable) degrades to ``None`` so a misconfigured gate simply stays off
    rather than crashing the app on startup.
    """
    blob = os.environ.get("SYNAPSE_AUTH_CONFIG")
    if blob:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            logger.warning("SYNAPSE_AUTH_CONFIG is set but is not valid JSON; login disabled.")
            return None
        return parsed if isinstance(parsed, dict) else None
    try:
        if _PASSWORD_SECTION in st.secrets:
            return _to_plain(st.secrets[_PASSWORD_SECTION])  # type: ignore[arg-type]
        # Legacy location, from before Streamlit's own OIDC claimed ``[auth]``. Accepted only
        # when it actually carries credentials: an OIDC ``[auth]`` block never does, so an
        # OIDC-only deployment is not mistaken for a half-configured password gate.
        legacy = st.secrets.get(_LEGACY_PASSWORD_SECTION)
        if isinstance(legacy, Mapping) and "credentials" in legacy:
            return _to_plain(legacy)  # type: ignore[arg-type]
    except Exception:  # no secrets file, unreadable backend — treat as "not configured"
        return None
    return None


def load_config(raw: dict | None = None) -> AuthConfig | None:
    """Build an :class:`AuthConfig` from ``raw`` (or the live secrets), or ``None`` if unusable.

    Returns ``None`` — meaning "login feature off" — when no config is present or it lacks any
    users. ``raw`` is injectable so the parsing can be unit-tested without a Streamlit runtime.
    A present-but-missing ``cookie_key`` is *not* rejected here (the config is otherwise valid);
    :func:`ensure_authenticated` surfaces that as a visible configuration error instead.
    """
    raw = raw if raw is not None else _read_raw()
    if not raw:
        return None
    credentials = raw.get("credentials")
    if not isinstance(credentials, Mapping):
        return None
    usernames = credentials.get("usernames")
    if not isinstance(usernames, Mapping) or not usernames:
        return None
    return AuthConfig(
        credentials=_to_plain(credentials),  # type: ignore[arg-type]
        cookie_name=str(raw.get("cookie_name") or _DEFAULT_COOKIE_NAME),
        cookie_key=str(raw.get("cookie_key") or ""),
        cookie_expiry_days=int(raw.get("cookie_expiry_days") or _DEFAULT_COOKIE_EXPIRY_DAYS),
    )


def is_enabled() -> bool:
    """True when a usable ``[auth]`` config exists — i.e. the login gate should guard the app."""
    return load_config() is not None


def current_user() -> User | None:
    """The signed-in :class:`User`, or ``None`` when logged out / auth disabled."""
    user = st.session_state.get(USER_KEY)
    return user if isinstance(user, User) else None


def _get_authenticator(cfg: AuthConfig):
    """Create (once per session) and cache the ``streamlit_authenticator.Authenticate`` object.

    Login and logout must share one instance, so it is memoised in session state. The library is
    imported lazily *here* — only when the gate is actually active — so a disabled gate never
    incurs the import (and the app runs even before ``streamlit-authenticator`` is installed).
    """
    authenticator = st.session_state.get(_AUTHENTICATOR_KEY)
    if authenticator is None:
        import streamlit_authenticator as stauth  # lazy: only when login is active

        authenticator = stauth.Authenticate(
            copy.deepcopy(cfg.credentials),  # library auto-hashes passwords in place
            cfg.cookie_name,
            cfg.cookie_key,
            cfg.cookie_expiry_days,
        )
        st.session_state[_AUTHENTICATOR_KEY] = authenticator
    return authenticator


def ensure_authenticated() -> bool:
    """Gate the app: return ``True`` to proceed, ``False`` to render nothing but the login form.

    * **Feature off** (no config) → ``True`` immediately; the app is fully open, as before.
    * **Misconfigured** (config present but no ``cookie_key``) → a visible error and ``False``,
      so the problem is obvious rather than a silent crash or an unguarded app.
    * **Logged in** → records the typed :class:`User` in session state and returns ``True``.
    * **Not yet logged in / bad credentials** → renders the login widget (plus a hint or error)
      and returns ``False``.
    """
    cfg = load_config()
    if cfg is None:
        return True  # login feature not configured → open app (unchanged behaviour)

    if not cfg.cookie_key:
        st.error(
            "🔒 Login is enabled but its `cookie_key` is missing. Add a random `cookie_key` to "
            "your `[auth]` secrets (see `.streamlit/secrets.toml.example`) to activate login."
        )
        return False

    try:
        authenticator = _get_authenticator(cfg)
        authenticator.login(location="main")
    except Exception as exc:  # bad config shape, hashing failure, etc. — fail visibly, not hard
        logger.exception("Login widget failed to render")
        st.error(f"🔒 Login is temporarily unavailable: {exc}")
        return False

    status = st.session_state.get("authentication_status")
    if status:
        username = st.session_state.get("username") or ""
        st.session_state[USER_KEY] = User(
            username=username,
            name=st.session_state.get("name") or username,
        )
        return True

    st.session_state[USER_KEY] = None
    if status is False:
        st.error("❌ Incorrect username or password.")
    else:
        st.info("🔒 Please log in to use Synapse.")
    return False


def render_logout(location: str = "sidebar", *, show_name: bool = True) -> None:
    """Draw the password gate's logout control (a no-op when logged out or the gate is off).

    Args:
        location: Where ``streamlit_authenticator`` should place the button. The account menu
            in the top bar passes ``"main"`` so the control appears inside its popover.
        show_name: Whether to print "Signed in — <name>" above the button. The account menu
            already shows the name on the control that opens it, and repeating it there would
            be the same fact in two places a centimetre apart.

    One logout surface on purpose. This used to be called unconditionally from the entry
    point *and* the name shown again in the corner, which is two controls for one action and
    therefore two places for them to disagree about whether anyone is signed in.
    """
    user = current_user()
    authenticator = st.session_state.get(_AUTHENTICATOR_KEY)
    if user is None or authenticator is None:
        return
    if show_name:
        st.sidebar.markdown(
            f"<div class='sb-status'><span>Signed in</span>"
            f"<span>{html.escape(user.name)}</span></div>",
            unsafe_allow_html=True,
        )
    authenticator.logout(location=location)


# --- Native OpenID Connect (the deployment path) --------------------------------------------


def oidc_providers() -> tuple[str, ...]:
    """Names of the OpenID providers configured under ``[auth]``, newest-Streamlit style.

    Two shapes are legal and both are in Streamlit's own documentation. A *single-provider*
    config puts ``client_id`` / ``client_secret`` / ``server_metadata_url`` directly in
    ``[auth]``; a *multi-provider* one adds a sub-table per provider (``[auth.google]``,
    ``[auth.auth0]``) beside the shared ``redirect_uri`` and ``cookie_secret``.

    Returns:
        ``("",)`` for a single-provider config (``""`` means "call ``st.login()`` with no
        argument"), the provider names for a multi-provider one, or ``()`` when no usable
        OIDC config is present. Never raises: reading ``st.secrets`` on a checkout with no
        secrets file is the *normal* local case and must not be an error.
    """
    try:
        return providers_in(st.secrets.get("auth"))
    except Exception:
        return ()


def providers_in(section: object) -> tuple[str, ...]:
    """The provider names in an ``[auth]`` mapping — :func:`oidc_providers` without Streamlit.

    Split out so the shape-matching can be unit-tested against both documented layouts with
    no secrets file and no browser. A sub-table without a ``client_id`` is not a provider:
    that is what keeps a legacy ``[auth.credentials]`` password block from being reported as
    a provider called "credentials".
    """
    if not isinstance(section, Mapping):
        return ()
    if section.get("client_id"):
        return ("",)
    return tuple(
        str(name)
        for name, value in section.items()
        if name not in _OIDC_RESERVED and isinstance(value, Mapping) and value.get("client_id")
    )


def oidc_configured() -> bool:
    """Whether pressing **Log in** would actually reach an identity provider."""
    return bool(oidc_providers())


def oidc_identity() -> tuple[bool, str, str]:
    """Read the signed-in identity out of ``st.user``.

    Kept separate from :func:`visitor` because it is the one part whose shape belongs to
    somebody else: ``st.user`` is absent on older Streamlit, empty when no provider is
    configured, and dict-like rather than an object.

    Returns:
        Whether anyone is signed in, their display name, and their email — empty strings for
        anything the provider did not supply.
    """
    user = getattr(st, "user", None)
    if user is None:
        return False, "", ""
    try:
        if not bool(user.get("is_logged_in", False)):
            return False, "", ""
        return True, str(user.get("name", "") or ""), str(user.get("email", "") or "")
    except Exception:  # pre-1.42 Streamlit, or a provider that returned nothing usable
        return False, "", ""


def begin_login(provider: str = "") -> None:
    """Send the visitor to the identity provider.

    Args:
        provider: A name from :func:`oidc_providers`, or ``""`` for a single-provider config.

    Both **Log in** and **Sign up for free** call this, and that is not a shortcut standing in
    for a real registration flow: with an identity provider there *is* no separate sign-up —
    a visitor without an account creates one at the provider and arrives back here signed in.
    """
    if provider:
        st.login(provider)
    else:
        st.login()


def end_login() -> None:
    """Sign the visitor out at the provider and clear ``st.user``."""
    st.logout()


# --- Bring your own credential --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeyStatus:
    """The verdict on a pasted credential.

    Attributes:
        usable: Whether the key should be accepted into session state.
        checked: Whether the verdict came from actually asking the provider. ``False`` means
            the check could not be made (no network, provider down) and the key was accepted
            on trust rather than verified.
        reason: A sentence to show the visitor.
    """

    usable: bool
    checked: bool
    reason: str


_KEY_CHECK_URL = "https://openrouter.ai/api/v1/key"
_KEY_CHECK_TIMEOUT_S = 6.0


def check_openrouter_key(typed: str) -> KeyStatus:
    """Ask OpenRouter whether a pasted key is real, before storing it.

    The check matters because of *how* a bad key fails here. Synapse answers from retrieved
    documents, so a key with one character missing does not produce an obvious error — it
    produces a run where retrieval succeeded, citations were gathered, and only the final
    generation failed, which reads as the model having a bad day. Better to say so at the
    moment it is pasted.

    An *unverifiable* key is accepted rather than refused: if the check itself could not be
    made, the honest answer is "we could not ask", and refusing on that basis would lock out
    a visitor whose key is fine whenever OpenRouter is briefly unreachable.

    Args:
        typed: What was pasted, unstripped.

    Returns:
        The verdict. Never raises, and never puts the key in a log or a message.
    """
    key = typed.strip()
    if not key:
        return KeyStatus(False, True, "Paste a key first.")
    if not key.startswith("sk-or-"):
        return KeyStatus(
            False,
            True,
            "That does not look like an OpenRouter key — they begin with `sk-or-`.",
        )
    try:
        import httpx

        response = httpx.get(
            _KEY_CHECK_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=_KEY_CHECK_TIMEOUT_S,
        )
    except Exception:  # offline, DNS failure, timeout — cannot ask, so do not pretend to know
        return KeyStatus(True, False, "Key accepted, but it could not be verified just now.")
    if response.status_code == 401:
        return KeyStatus(False, True, "OpenRouter rejected that key.")
    if response.status_code >= 400:
        return KeyStatus(True, False, "Key accepted, but OpenRouter could not confirm it.")
    return KeyStatus(True, True, "Key verified — your questions are billed to it.")


def own_key() -> str:
    """The credential this visitor pasted, or ``""``.

    A function rather than a field on :class:`Visitor`, so the value never travels inside a
    record that gets logged, cached or rendered. Callers reach for it at the moment they need
    it and nowhere else.
    """
    value = st.session_state.get(OWN_KEY_STATE)
    return value if isinstance(value, str) else ""


def set_own_key(key: str) -> None:
    """Store a pasted credential for this browser session only."""
    st.session_state[OWN_KEY_STATE] = key.strip()


def forget_own_key() -> None:
    """Drop a pasted credential from session state.

    A button rather than only a tab close, because a visitor on a shared machine should not
    have to trust the tab.
    """
    st.session_state.pop(OWN_KEY_STATE, None)


def apply_runtime_keys() -> None:
    """Publish this session's chosen credential to :mod:`src.config` for the current run.

    Called once per script run from the entry point, *before* any page builds a model. It is
    also called when there is no pasted key, which is the point: the override is thread-local
    and a worker thread may be reused by a later run, so each run states its choice rather
    than inheriting the previous one's.
    """
    from src.config import clear_runtime_keys, set_runtime_key

    # Wipe first, then state this run's choice. Setting each key individually meant the
    # promise only covered the keys named here, and GOOGLE_API_KEY was never among them — so
    # the day something lets a visitor paste one, it would outlive their session on whatever
    # worker thread picked it up. Clearing makes that impossible to get wrong by omission.
    clear_runtime_keys()
    set_runtime_key("OPENROUTER_API_KEY", own_key() or None)


# --- Who is visiting ------------------------------------------------------------------------

Level = Literal["signed_in", "own_key", "password", "open"]
"""Which door the visitor came through. See this module's docstring for the table."""


@dataclass(frozen=True, slots=True)
class Visitor:
    """One visitor: how they were identified, and what to call them.

    Attributes:
        level: Which door they came through.
        name: Display name from the provider or the password gate, or ``""``.
        email: Address from the provider, or ``""``. Held only so the account menu can show
            which account is signed in; nothing keys off it, because Synapse has no per-user
            data to key.
    """

    level: Level = "open"
    name: str = ""
    email: str = ""

    @property
    def is_identified(self) -> bool:
        """Whether a name can be shown for this visitor rather than a pair of login buttons."""
        return self.level in ("signed_in", "password")

    @property
    def greeting(self) -> str:
        """The most specific thing actually known about this visitor, for the account menu."""
        if self.name:
            return self.name.split()[0]
        if self.email:
            return self.email
        return {"own_key": "Your key", "password": "Signed in"}.get(self.level, "Guest")


def visitor() -> Visitor:
    """Who is looking at this page.

    Signing in wins over a pasted key, on the grounds that it is the more deliberate of the
    two and the only one whose identity can be shown back. A pasted key wins over the
    password gate, because someone who has put their own credential behind their questions
    has said something more specific about this session than "I am allowed in".

    Returns:
        The current visitor. ``open`` when no door is configured — the local ``make run``
        case, where the person at the keyboard is the person who started the process.
    """
    signed_in, name, email = oidc_identity()
    if signed_in:
        return Visitor("signed_in", name, email)
    if own_key():
        return Visitor("own_key")
    gated = current_user()
    if gated is not None:
        return Visitor("password", gated.name)
    return Visitor("open")
