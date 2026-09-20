"""auth.py — access control for the deployed app (UI layer).

The app costs money to run: every question, every judgement and every cover
letter is a paid API call. When it runs on your laptop that is fine, because
the key in ``.env`` is yours. The moment it is deployed somewhere public,
anyone who finds the URL is spending your credit, so the app needs to know
who is asking before it calls anything.

Three ways in, shown as a row of buttons in the top right:

    Log in       — paste your own OpenRouter key; you pay for your own use.
                   If the deployment is running Streamlit 1.42+ with an
                   ``[auth]`` block in secrets, this also offers real Google
                   sign-in through ``st.login()``.
    Sign up      — free: a link to create an OpenRouter account and key,
                   then the same paste box. OpenRouter has free models, so
                   a new account can use this app without adding a card.
    Try it free  — the deployer's shared key, capped at a small number of
                   requests per session so a stranger cannot drain it.

What this module does NOT do is store passwords. There is no user table and
no password hashing here, deliberately: a Streamlit app on a public host is
the wrong place to keep credentials, and a half-built auth system is worse
than none. Identity, when you want it, comes from ``st.login()`` (OIDC), and
spending is controlled by whose API key is being used.

SECURITY NOTES
--------------
* A pasted key lives in ``st.session_state`` for that browser session only.
  It is never written to disk, never logged, and never leaves the process
  except in the Authorization header the OpenAI SDK sends to OpenRouter.
* The free-trial counter is per session. Someone who opens a new tab gets a
  new allowance. It stops casual overuse, not a determined person — a hard
  limit needs server-side state keyed to an identity, which is a bigger
  change than this file. Set the trial low and treat it as a demo budget.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any

import streamlit as st

OPENROUTER_KEYS_URL = "https://openrouter.ai/keys"
OPENROUTER_SIGNUP_URL = "https://openrouter.ai/sign-up"

# Session-state keys (prefixed so they cannot collide with widget keys).
_KEY_API = "_auth_api_key"
_KEY_MODE = "_auth_mode"
_KEY_USED = "_auth_requests_used"
# The verdict on the last key pasted, and not the key: accepting one calls
# st.rerun(), which throws away anything written during that run, so the
# message has to survive in state to be shown on the run that follows.
_KEY_VERDICT = "_auth_key_verdict"

MODE_OWN_KEY = "own_key"
MODE_TRIAL = "trial"
MODE_SIGNED_IN = "signed_in"
MODE_LOCAL = "local"

# Defaults; override in .streamlit/secrets.toml or the environment.
DEFAULT_TRIAL_REQUESTS = 10
DEFAULT_SIGNED_IN_REQUESTS = 50


@dataclass(frozen=True)
class Access:
    """Who is using the app and on whose budget."""

    mode: str
    api_key: str
    label: str
    limit: int | None  # None = unlimited (the user brought their own key)

    @property
    def shared_budget(self) -> bool:
        """True when requests are billed to the deployer, not to the user."""
        return self.limit is not None


# ---------------------------------------------------------------------------
# Configuration (secrets first, then environment)
# ---------------------------------------------------------------------------


# The two locations Streamlit itself looks in. Touching st.secrets when no
# file exists makes Streamlit print "No secrets files found..." into the app
# — once per access — so the file is checked before the secret is read.
_SECRETS_PATHS = (
    pathlib.Path(".streamlit/secrets.toml"),
    pathlib.Path.home() / ".streamlit" / "secrets.toml",
)


def _secrets_available() -> bool:
    """True when a secrets.toml exists, i.e. when st.secrets is safe to read."""
    return any(path.is_file() for path in _SECRETS_PATHS)


def _secret(name: str) -> str | None:
    """Read one Streamlit secret, quietly returning None when there are none."""
    if not _secrets_available():
        return None
    try:
        value = st.secrets.get(name)  # type: ignore[union-attr]
    except Exception:
        return None
    return str(value) if value else None


def _setting(name: str) -> str | None:
    """Secrets take priority over the environment; .env fills in locally."""
    return _secret(name) or os.getenv(name)


def _int_setting(name: str, default: int) -> int:
    raw = _setting(name)
    try:
        return max(0, int(raw)) if raw is not None else default
    except ValueError:
        return default


def deployer_key() -> str | None:
    """The key that pays for the free trial, or None when not configured."""
    return _setting("OPENROUTER_API_KEY")


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"false", "0", "no", "off", ""}


def trial_enabled() -> bool:
    """Free trial is on when a shared key exists and it wasn't switched off."""
    if not deployer_key():
        return False
    return _truthy(_setting("ALLOW_FREE_TRIAL"), default=True)


def local_owner_key() -> str | None:
    """The developer's own key from .env, when this is a local run.

    Running `make run` on your own machine shouldn't put a login screen
    between you and your own app, so a key that came from the environment
    (and not from secrets.toml) grants access straight away. On a deployment
    the key lives in secrets instead, so the gate stays up. Set
    REQUIRE_LOGIN=true to force the gate locally and test the sign-in flow.
    """
    if _truthy(_setting("REQUIRE_LOGIN")):
        return None
    if _secret("OPENROUTER_API_KEY"):
        return None
    return os.getenv("OPENROUTER_API_KEY")


def _login_available() -> bool:
    """True only on Streamlit 1.42+ WITH an [auth] block configured."""
    return hasattr(st, "login") and _secret("auth") is not None


def _signed_in_user() -> str | None:
    """The OIDC identity, when the deployment has sign-in configured."""
    user = getattr(st, "user", None)
    if user is None:
        return None
    try:
        if not getattr(user, "is_logged_in", False):
            return None
        return getattr(user, "email", None) or getattr(user, "name", None) or "signed in"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Current access
# ---------------------------------------------------------------------------


def _mask(key: str) -> str:
    """Show enough of a key to recognise it, never enough to use it."""
    return f"{key[:7]}…{key[-4:]}" if len(key) > 14 else "…"


def _valid_key_shape(key: str) -> bool:
    """Cheap sanity check. Only the API can say whether a key really works."""
    return key.startswith("sk-") and len(key) >= 20


def _grant(mode: str, api_key: str) -> None:
    st.session_state[_KEY_MODE] = mode
    st.session_state[_KEY_API] = api_key
    st.session_state.setdefault(_KEY_USED, 0)


def current_access() -> Access | None:
    """The access this session already has, or None when it has none yet."""
    user = _signed_in_user()
    if user and not st.session_state.get(_KEY_API):
        # Signed in through OIDC but no personal key: spend the shared one,
        # on a larger allowance than an anonymous visitor gets.
        key = deployer_key()
        if key:
            return Access(
                mode=MODE_SIGNED_IN,
                api_key=key,
                label=user,
                limit=_int_setting("SIGNED_IN_REQUESTS", DEFAULT_SIGNED_IN_REQUESTS),
            )

    key = st.session_state.get(_KEY_API)
    if not key:
        own = local_owner_key()
        if own:
            return Access(mode=MODE_LOCAL, api_key=own, label="Local key", limit=None)
        return None
    mode = st.session_state.get(_KEY_MODE, MODE_OWN_KEY)
    if mode == MODE_TRIAL:
        return Access(
            mode=MODE_TRIAL,
            api_key=key,
            label="Free trial",
            limit=_int_setting("TRIAL_REQUESTS", DEFAULT_TRIAL_REQUESTS),
        )
    return Access(mode=MODE_OWN_KEY, api_key=key, label=user or "Your API key", limit=None)


def requests_left(access: Access) -> int | None:
    """Requests remaining on a shared budget; None when the user pays."""
    if access.limit is None:
        return None
    return max(0, access.limit - int(st.session_state.get(_KEY_USED, 0)))


def sign_out() -> None:
    """Forget the key and the trial counter, then end any OIDC session."""
    for key in (_KEY_API, _KEY_MODE, _KEY_USED):
        st.session_state.pop(key, None)
    if hasattr(st, "logout") and _signed_in_user():
        st.logout()


# ---------------------------------------------------------------------------
# Quota enforcement
# ---------------------------------------------------------------------------


class QuotaExceeded(RuntimeError):
    """Raised instead of making a call the session is no longer allowed."""


def _consume(access: Access) -> None:
    """Count one request against a shared budget, or refuse it."""
    if access.limit is None:
        return
    used = int(st.session_state.get(_KEY_USED, 0))
    if used >= access.limit:
        # The leading ⚠️ marks this as already user-facing, so core.py's
        # error mapper passes the text through instead of rewriting it.
        raise QuotaExceeded(
            f"⚠️ You've used all {access.limit} free requests in this session. "
            "Add your own OpenRouter API key from the menu in the top right to keep going."
        )
    st.session_state[_KEY_USED] = used + 1


class _QuotaCompletions:
    def __init__(self, inner: Any, access: Access):
        self._inner = inner
        self._access = access

    def create(self, **kwargs: Any) -> Any:
        _consume(self._access)  # counted before the call, so retries cost too
        return self._inner.create(**kwargs)


class _QuotaChat:
    def __init__(self, inner: Any, access: Access):
        self.completions = _QuotaCompletions(inner.completions, access)


class QuotaLimitedClient:
    """Wraps the OpenAI client so every call goes through the trial counter.

    core.ask_llm() only ever touches ``client.chat.completions.create``, so
    wrapping that one method covers every feature without changing a single
    call site.
    """

    def __init__(self, inner: Any, access: Access):
        self._inner = inner
        self.chat = _QuotaChat(inner.chat, access)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def apply_quota(client: Any, access: Access) -> Any:
    """Return the client, wrapped only when someone else is paying."""
    return QuotaLimitedClient(client, access) if access.shared_budget else client


# ---------------------------------------------------------------------------
# The header row
# ---------------------------------------------------------------------------


def sign_in_configured() -> bool:
    """Whether pressing a sign-in button would actually reach a provider.

    Two things have to be true: the Streamlit version has ``st.login`` (1.42+,
    and this project pins 1.37.1), and the deployment has an ``[auth]`` block
    in its secrets. Anything unexpected counts as "not configured", because
    the only thing this answer decides is whether a button is live.
    """
    if not hasattr(st, "login"):
        return False
    if not _secrets_available():
        return False
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def _sign_in_blocked_reason() -> str:
    """Why the sign-in buttons are disabled, said plainly in a tooltip."""
    if not hasattr(st, "login"):
        return (
            "Accounts need Streamlit 1.42 or newer and this app pins 1.37.1. "
            "Use your own OpenRouter key instead."
        )
    return (
        "Accounts are not configured for this deployment — see "
        ".streamlit/secrets.toml.example. You can use your own key instead."
    )


def _report_last_key_check() -> None:
    """Show the verdict on the key just pasted, once.

    Accepting a key calls st.rerun(), and anything written before a rerun is
    thrown away with the rest of that run's output — so the verdict is parked
    in session state, then read and cleared on the run that follows.
    """
    remembered = st.session_state.pop(_KEY_VERDICT, None)
    if remembered is None:
        return
    ok, message = remembered
    (st.success if ok else st.error)(message)


def _own_key_form(form_key: str) -> None:
    """The bring-your-own-credential path.

    A form rather than a bare text input, so the key is submitted once instead
    of putting a partial credential through a rerun on every keystroke.
    """
    st.caption(
        "Paste an OpenRouter key. Held in this browser session only — never written "
        f"to disk, never logged. [Get one free]({OPENROUTER_KEYS_URL})."
    )
    with st.form(form_key, clear_on_submit=True, border=False):
        typed = st.text_input(
            "OpenRouter key",
            type="password",
            placeholder="sk-or-v1-…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Use this key", use_container_width=True, type="primary")
    if submitted:
        _accept_key(typed)
    _report_last_key_check()


def _accept_key(typed: str) -> None:
    """Take a pasted credential if there's no reason not to, then rerun.

    The shape is all this can check without spending a call; only OpenRouter
    can say whether a well-formed key actually works, and it will say so on
    the first request.
    """
    candidate = (typed or "").strip()
    if not candidate:
        st.session_state[_KEY_VERDICT] = (False, "Paste a key first.")
        return
    if not _valid_key_shape(candidate):
        st.session_state[_KEY_VERDICT] = (
            False,
            "That doesn't look like an OpenRouter key — they start with `sk-or-`.",
        )
        return
    st.session_state[_KEY_VERDICT] = (True, "Key accepted for this session.")
    _grant(MODE_OWN_KEY, candidate)
    st.rerun()


def _forget_key() -> None:
    """Drop a pasted credential, for somebody on a shared machine who would
    rather not trust the tab closing."""
    sign_out()


def _use_a_key_menu(access: Access | None) -> None:
    """The 'Use a key' popover: your own key, or the shared free trial.

    Also where a visitor who already has access but no account of their own —
    a local run, or someone on the free trial — is told what they're running
    on. One place for that fact, so it cannot disagree with itself.
    """
    if access is not None and access.mode == MODE_LOCAL:
        st.caption(
            f"Running on the key in .env · {_mask(access.api_key)}. "
            "Set REQUIRE_LOGIN=true there to see the signed-out screen."
        )
        st.divider()
    elif access is not None and access.mode == MODE_TRIAL:
        left = requests_left(access) or 0
        st.caption(f"**Free trial** — {left} of {access.limit} requests left this session.")
        st.progress(left / access.limit if access.limit else 0.0)
        if left == 0:
            st.warning("Trial used up. Paste your own key to carry on.")
        st.divider()

    _own_key_form("own_key_form")

    if access is None and trial_enabled():
        st.divider()
        limit = _int_setting("TRIAL_REQUESTS", DEFAULT_TRIAL_REQUESTS)
        st.caption(f"No key yet? Try the app on ours — {limit} requests this session.")
        if st.button("Try it free", key="start_trial", use_container_width=True):
            _grant(MODE_TRIAL, deployer_key() or "")
            st.rerun()


def _account_menu(access: Access) -> None:
    """What sits behind the visitor's own name once they're in."""
    if access.mode == MODE_SIGNED_IN:
        st.caption(access.label)
        left = requests_left(access)
        if left is not None:
            st.caption(f"{left} of {access.limit} requests left this session.")
            st.progress(left / access.limit if access.limit else 0.0)
        if st.button("Log out", key="log_out", use_container_width=True):
            sign_out()
            st.rerun()
        return

    st.caption(f"Running on your own key · {_mask(access.api_key)}")
    st.caption("Requests are billed to your OpenRouter account.")
    st.button("Forget my key", key="forget_key", use_container_width=True, on_click=_forget_key)


# How far in from the right edge the lifted bar sits. It has to clear
# everything Streamlit puts in that strip, and the list is longer than Deploy
# and the ⋮ menu: while a script is running a **Stop** button appears to their
# left, and covering that one is not cosmetic — you could no longer stop a
# long request. This is a measured guess, not a computed value, because the
# header is Streamlit's and its contents are not ours to interrogate. If a
# future version widens that toolbar, this is the one number to change.
_HEADER_RIGHT_OFFSET = "13rem"

# Below this width the header has no room to share, so the bar drops back into
# the page rather than landing on top of Streamlit's own controls.
_LIFT_MIN_WIDTH = "900px"

_TOP_BAR_ANCHOR = "auth-top-bar-anchor"

_TOP_BAR_CSS = f"""<style>
/* Lift the account controls out of the page body and onto the header strip, so
   they sit in the same row as Streamlit's Deploy button and ⋮ menu.

   Fixed rather than floated: the header strip is not this element's parent, and
   no amount of margin will move it there. Out of the flow, so the title below
   closes up behind it.

   The hook is an empty anchor div emitted immediately before the row. Streamlit
   1.49 gives containers a key and an st-key-* class to target directly; 1.37
   does not, so the row is reached as the sibling that follows the anchor's
   element container. That makes this the version-fragile part of the file: if
   the bar ever appears in the middle of the page, this selector stopped
   matching and the rule below is where to look. */
[data-testid="element-container"]:has(#{_TOP_BAR_ANCHOR}) {{
  display: none;
}}

@media (min-width: {_LIFT_MIN_WIDTH}) {{
  [data-testid="element-container"]:has(#{_TOP_BAR_ANCHOR})
    + [data-testid="stHorizontalBlock"] {{
    position: fixed;
    top: 0.3rem;
    right: {_HEADER_RIGHT_OFFSET};
    width: auto;
    gap: 0.4rem;
    /* Streamlit's header sits at 999990. One above it, so the controls are
       clickable rather than painted underneath. */
    z-index: 999991;
  }}

  /* Columns inside a fixed bar must size to their labels instead of to a
     share of the page, or the bar keeps the full content width and the
     spacer column pushes everything off the right edge. */
  [data-testid="element-container"]:has(#{_TOP_BAR_ANCHOR})
    + [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
    width: auto !important;
    flex: 0 0 auto !important;
    min-width: 0 !important;
  }}
}}

/* Tighter than the stock 6rem, but NOT tighter than the header: Streamlit's
   header is fixed and about 3.75rem tall, so anything below that slides
   underneath it and the first row on the page loses its top edge. 4rem is the
   floor, not a preference. */
.block-container {{ padding-top: 4rem; }}
</style>"""


def _lift_to_top() -> None:
    """Move the controls onto Streamlit's header strip, beside Deploy and ⋮.

    Emits the stylesheet and the anchor the selector keys off. The anchor is
    an empty, hidden div: it exists only so the row that follows it can be
    named without a container key, which this Streamlit version doesn't have.

    If the CSS ever stops matching, nothing breaks — the controls simply stay
    where they were drawn, at the top of the page body.
    """
    st.markdown(_TOP_BAR_CSS, unsafe_allow_html=True)
    st.markdown(f'<div id="{_TOP_BAR_ANCHOR}"></div>', unsafe_allow_html=True)


def header(title: str) -> Access | None:
    """Draw the account controls at the top right of the page, then the title.

    Shaped after the pattern every visitor already knows: a quiet *Log in* and
    a filled *Sign up for free* in the corner, which is where the eye already
    goes to answer "whose session is this?".

    Both buttons start the same flow, which isn't a shortcut — with an identity
    provider there is no separate sign-up: someone without an account creates
    one at Google and comes back signed in. Two buttons because they answer two
    different questions, and a visitor shown only *Log in* assumes they need an
    account already.

    When no provider is configured the two are drawn **disabled with the reason
    in the tooltip**, rather than hidden or left live. Hidden, and the app looks
    like it simply has no accounts, which is indistinguishable from broken to
    anyone who was told to sign in. Live, and pressing one raises. Disabled with
    a sentence is the only one of the three that tells the truth — and *Use a
    key* beside them always works.

    Returns:
        The session's Access, or None when it has none yet (the caller should
        stop rather than build a client).
    """
    access = current_access()
    _lift_to_top()

    # The controls are drawn BEFORE the title, in their own row, so they sit at
    # the very top of the page rather than beside an h1 that is itself a third
    # of the way down. The leading weight in each row is an empty column that
    # pushes them right; the trailing one keeps them off the edge.
    if access is not None and access.mode in (MODE_OWN_KEY, MODE_SIGNED_IN):
        _, menu_col, _edge = st.columns([4.2, 3.4, 0.4])
        with menu_col, st.popover(f"👤 {access.label}", use_container_width=True):
            _account_menu(access)
        st.title(title)
        return access

    configured = sign_in_configured()
    reason = "Unlocks the app on your account." if configured else _sign_in_blocked_reason()
    # Started further right and the labels had nowhere to go: a Streamlit
    # button never truncates, it wraps, so a column too narrow for "Sign up for
    # free" turns it into two lines and makes the whole row taller. The leading
    # spacer is what gives the three of them room — shrink it, not them.
    _, log_in, sign_up, use_key, _edge = st.columns([2.6, 1.3, 2.1, 1.6, 0.3])
    with log_in:
        st.button(
            "Log in",
            key="log_in",
            help=reason,
            disabled=not configured,
            use_container_width=True,
            on_click=st.login if configured else None,
        )
    with sign_up:
        st.button(
            "Sign up for free",
            key="sign_up",
            type="primary",
            help=reason,
            disabled=not configured,
            use_container_width=True,
            on_click=st.login if configured else None,
        )
    with use_key:
        label = "Use a key"
        if access is not None and access.mode == MODE_TRIAL:
            label = f"Trial · {requests_left(access)} left"
        with st.popover(label, use_container_width=True):
            _use_a_key_menu(access)

    st.title(title)
    return access


def welcome() -> None:
    """The 'you are not in yet' panel, shown in place of the app itself."""
    st.info("Use **Sign up for free** or **Use a key** in the top right to get started.")
    st.markdown(
        "#### What you can do once you're in\n"
        "- Generate role-specific interview questions and practise them one at a time\n"
        "- Get your answers scored with strengths, improvements and a model answer\n"
        "- Upload your CV and a job ad for advice written around your background\n"
        "- Draft a tailored cover letter you can edit and download"
    )
    if not sign_in_configured():
        st.caption(
            "Accounts aren't configured for this deployment, so the Log in and Sign up "
            "buttons are disabled. An OpenRouter key works just as well — it's free to "
            f"create one at [openrouter.ai]({OPENROUTER_SIGNUP_URL})."
        )
