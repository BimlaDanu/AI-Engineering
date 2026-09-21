"""Who is using this, once it is not only you.

On a laptop the answer is "me", and asking would be pure friction. Deployed, four
different questions appear, and they are worth keeping apart because they have
different answers:

*Who should be let in at all?* An access gate. Configured by an environment
variable, :data:`PASSPHRASE_VARIABLE`, and **absent by default** -- a project
running locally must not demand a password nobody set. When the variable is set,
nothing renders until it is matched, and it is compared with
:func:`hmac.compare_digest` so the comparison does not leak its answer through
timing.

*Who is this, verifiably?* The deployment's own login. Streamlit's ``st.user``
carries an identity when the host is configured for OIDC, and that identity is
worth more than anything this application could establish, so when it is present
it wins. Nothing here configures a provider: an identity provider belongs to
whoever deploys, and inventing our own would be a worse version of one.

*Who is this, for the sake of being addressed?* **Nobody, unless a provider said
so.** There is deliberately no name box. A text box asking who you are is friction
on a laptop, where the answer is always "me", and it is a lie waiting to happen on
a deployment, where an unverified name sitting beside verified physics reads like a
sign-in. So identity here is either asserted by a login or absent, and
:attr:`Identity.verified` is the field that says which.

*Whose budget is this being spent on?* The host's, unless a visitor pastes their
own OpenRouter key. That is a fourth question and not a restatement of the third:
a key says who pays, an identity says who is asking, and treating them as the same
thing would file a stranger's conversation under a credential. So a pasted key is
held in session state alone -- see :data:`OWN_KEY_STATE` -- read by
:func:`src.ui.panels.session_settings`, and it changes nothing about the thread.

All four are asked in one row at the top of the window, :func:`account_bar`, and
in exactly one place. The sidebar carried a *Sign in* button before this, which
made two account surfaces for one fact, and it was hidden entirely when no
provider was configured -- so a local checkout looked like an application with no
accounts at all rather than one whose deployment has not been given a provider.

What follows from identity is where a conversation is kept. A signed-in visitor has a
stable key -- the digest of the name their provider asserted -- so their history
outlives the session. Everyone else gets :func:`session_thread`, a token minted per
browser session, which makes follow-up questions work now and keeps nothing
afterwards. See :func:`src.ui.panels.session_store`, which pairs each of those with
the storage it deserves: a file, or memory that dies with the tab.

Everything degrades. No gate configured means open; no provider configured means an
anonymous session that works in every respect. **A passphrase is never logged,
never written to session state and never rendered** -- only the fact that one
matched.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import streamlit as st

from src.agent.credential_check import check_key
from src.logging_setup import get_logger

PASSPHRASE_VARIABLE = "APP_PASSPHRASE"
"""Environment variable holding the deployment's shared passphrase, if any.

Read from the environment rather than from :class:`src.settings.Settings`
deliberately. ``Settings`` requires a model credential and raises without one, so
a gate that went through it would stop gating on any deployment whose *unrelated*
credential was missing. An access control that fails open when something else
breaks is not an access control.
"""

THREAD_LENGTH = 16
"""Characters of digest kept for a conversation key. Ample for one deployment."""

_ACCESS_KEY = "access_granted"
_SESSION_KEY = "session_thread"

OWN_KEY_STATE = "own_api_key"
"""Session-state key holding a credential the visitor pasted.

Session state and nowhere else: one browser session's server-side store, never
written to disk, never logged, never put in a URL, and gone when the tab closes.
That is the only promise about somebody else's credential this application is in a
position to keep, so it is the only one made -- see :func:`_own_key_form`, which
says it out loud rather than leaving it to be assumed.
"""

_KEY_CHECK_STATE = "own_api_key_verdict"
"""Session-state key holding the verdict on the credential most recently pasted.

The verdict and not the credential: three primitives, no key. It exists because
accepting a key calls :func:`streamlit.rerun`, and anything written to the screen
before a rerun is thrown away with the rest of that run's output -- which is how a
form can validate a key and then show the visitor nothing at all.
"""

TOP_BAR_NAME = "quantumlab_top_bar"
"""The container key the top bar is drawn under.

Streamlit turns a container's ``key`` into an ``st-key-<key>`` class on the
element, which is the one documented hook for styling a *particular* container.
The rule that uses it is :data:`src.ui.panels.TOP_BAR_CSS`; the name lives here,
beside the container it names, because ``panels`` is what calls this function and
the other direction would be an import cycle.
"""

Source = Literal["platform", "anonymous"]
"""Where an identity came from.

Two values, not three. There was a ``"typed"`` source when a name box existed, and
removing the box removed the source with it: a value no code path can produce is a
distinction a reader has to work out is dead.
"""

LOG = get_logger("identity")


@dataclass(frozen=True, slots=True)
class Identity:
    """Who the current session belongs to.

    Attributes:
        name: What to call them, or ``""`` when nobody said.
        source: Where the name came from. See :data:`Source`.
    """

    name: str = ""
    source: Source = "anonymous"

    @property
    def verified(self) -> bool:
        """Whether an identity provider vouched for this, rather than a text box."""
        return self.source == "platform"

    @property
    def display(self) -> str:
        """A name to greet, falling back to something neutral rather than blank."""
        return self.name or "there"

    @property
    def thread(self) -> str:
        """A stable conversation key for this identity.

        Returns:
            A digest, or ``"anonymous"``. Hashed rather than used directly because
            this ends up as a checkpointer key and a log field: a name is personal
            data, a digest is an identifier, and the identifier is all either
            needs.

        Examples:
            >>> Identity("Ada", "platform").thread
            '99a563ab2f6e21e9'
            >>> Identity().thread
            'anonymous'
        """
        if not self.name:
            return "anonymous"
        return hashlib.sha256(self.name.encode("utf-8")).hexdigest()[:THREAD_LENGTH]

    def explain(self) -> str:
        """One line describing the identity and how much it is worth.

        Returns:
            A sentence for the sidebar.

        Examples:
            >>> Identity("Ada", "platform").explain()
            'Signed in as Ada.'
            >>> Identity().explain()
            'Not signed in — this conversation is kept for this session only.'
        """
        if self.source == "platform":
            return f"Signed in as {self.name}."
        return "Not signed in — this conversation is kept for this session only."


def configured_passphrase() -> str | None:
    """Read the deployment's passphrase from the environment.

    Returns:
        The passphrase, or ``None`` when the variable is unset or blank. Blank
        counts as unset: an empty string in a deployment configuration is
        somebody clearing the value, not somebody choosing an empty password.
    """
    value = os.environ.get(PASSPHRASE_VARIABLE, "").strip()
    return value or None


def matches(attempt: str, passphrase: str) -> bool:
    """Compare an attempt against the passphrase in constant time.

    Args:
        attempt: What was typed.
        passphrase: What was configured.

    Returns:
        Whether they are equal. ``compare_digest`` rather than ``==`` so the
        comparison takes the same time whatever the input -- the standard defence,
        cheap enough that skipping it is never worth the argument.

    Examples:
        >>> matches("open sesame", "open sesame")
        True
        >>> matches("open sesam", "open sesame")
        False
    """
    return hmac.compare_digest(attempt.strip().encode("utf-8"), passphrase.encode("utf-8"))


def platform_user() -> Identity | None:
    """Read the identity the deployment established, if it established one.

    Returns:
        The verified identity, or ``None`` when the host is not configured for
        login or nobody is logged in. Every access goes through ``getattr`` and a
        broad ``except``: ``st.user`` raises rather than returning empty when no
        provider is configured, and that is the ordinary local case, not an error.
    """
    try:
        user: Any = st.user
        if not getattr(user, "is_logged_in", False):
            return None
        name = getattr(user, "name", None) or getattr(user, "email", None) or "signed-in user"
    except Exception:
        return None
    return Identity(name=str(name).strip(), source="platform")


def login_configured() -> bool:
    """Whether the deployment has an identity provider set up.

    Returns:
        ``True`` when ``st.secrets`` carries an ``auth`` section, which is what
        ``st.login`` needs. Reading secrets raises when there is no secrets file
        at all, so that is caught and read as "not configured".
    """
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def gate() -> bool:
    """Require the deployment's passphrase, if one is configured.

    Draws nothing and returns ``True`` when no passphrase is set, which is the
    local case and must stay frictionless.

    Returns:
        Whether the session may see the application. A caller that gets ``False``
        must stop: the form has already been drawn.
    """
    passphrase = configured_passphrase()
    if passphrase is None or st.session_state.get(_ACCESS_KEY) is True:
        return True

    st.markdown("#### This deployment is private")
    st.caption("Ask whoever runs it for the passphrase.")
    with st.form("access"):
        attempt = st.text_input("Passphrase", type="password")
        submitted = st.form_submit_button("Enter", type="primary")
    if submitted:
        if matches(attempt, passphrase):
            # The flag, never the value: session state is readable by anything
            # else drawing this page, and it does not need the passphrase.
            st.session_state[_ACCESS_KEY] = True
            LOG.info("access_granted")
            st.rerun()
        st.error("That is not the passphrase.")
        LOG.warning("access_denied")
    return False


def session_thread() -> str:
    """The key this session's conversation is filed under.

    Returns:
        The signed-in identity's digest when there is one, and otherwise a random
        token minted once per browser session.

        Two different keys for two different lifetimes, which is the whole reason
        this is not simply :attr:`Identity.thread`. A digest of a name asserted by a
        provider is the same key tomorrow, so a signed-in visitor's history
        persists. A session token exists only in this session's state, so an
        unidentified visitor gets working follow-up questions and leaves nothing
        behind -- and, critically, is not given the constant ``"anonymous"``, which
        would file every unnamed visitor's questions into one shared history and
        read them back to the next one.
    """
    verified = platform_user()
    if verified is not None:
        return verified.thread
    existing = st.session_state.get(_SESSION_KEY)
    if isinstance(existing, str) and existing:
        return existing
    token = secrets.token_hex(THREAD_LENGTH // 2)
    st.session_state[_SESSION_KEY] = token
    return token


def current_identity() -> Identity:
    """Who is here, without drawing anything.

    Split from :func:`account_bar` because the two are needed at different points
    in the run and one of them must not draw. The entry point resolves the identity
    *before* the settings knob, since the thread it derives is what memory is keyed
    by and the register learned from that thread is the knob's starting position --
    and at that moment the page's title has not been written yet, so anything drawn
    would land above it.

    Returns:
        The platform's identity when it asserted one, and an anonymous identity
        otherwise. A pasted key is deliberately *not* an identity: it says whose
        budget pays, which is a different question from who is asking, and
        conflating them would file a stranger's conversation under a credential.
    """
    return platform_user() or Identity()


def own_key() -> str | None:
    """The credential this session pasted, if it pasted one.

    Returns:
        The key, or ``None``. Read by :func:`src.ui.panels.session_settings`,
        which is the only caller: everything else should ask *that* for settings
        rather than reach for the credential itself.
    """
    key = st.session_state.get(OWN_KEY_STATE)
    return key if isinstance(key, str) and key else None


def account_bar(lead: Callable[[], None] | None = None) -> None:
    """Draw the account controls in one row at the top of the window.

    Args:
        lead: An optional control to draw first, in the same row. The entry point
            passes ``New chat`` on the chat page, so that page has one top bar
            rather than a control at one height and an account row at another.

    Shaped after the pattern every visitor already knows. Somebody who is not
    signed in sees a quiet *Log in*, a filled *Sign up for free* and *Use a key*;
    somebody who is sees their own name with the account menu behind it. That
    corner is where a person's eye already goes to answer "whose session is this?",
    and putting it anywhere else costs them a hunt for a control they were not
    expecting to have to look for.

    Both sign-in buttons start the same flow, and that is not a shortcut: with an
    identity provider there is no separate sign-up -- a visitor without an account
    creates one at the provider and arrives back here signed in, which is exactly
    what pressing *Sign up for free* on a site like this one has always done. Two
    buttons rather than one because they answer two different questions ("I have an
    account" and "I do not"), and a visitor shown only *Log in* assumes they need
    one already.

    When no provider is configured the two buttons are drawn **disabled with the
    reason in the tooltip** rather than hidden or left live. Hidden, and the
    deployment looks like one that simply has no accounts, which is
    indistinguishable from broken to anybody who was told to sign in -- and it is
    what the sidebar panel this replaced actually did. Live, and pressing one
    raises. Disabled with a sentence is the only one of the three that tells the
    truth, and it is why *Use a key* sits beside them: on a checkout with no
    provider it is the one control in this row that still does something.

    A horizontal container rather than a row of spacer columns. The widths of
    columns have to be guessed, and ``width="stretch"`` inside a narrow one wraps
    "Sign up for free" onto two lines, which makes the whole row taller and pushes
    the page down. Here each control sizes to its own label and the container
    pushes the group to the edge, so the bar is one line high.
    """
    with st.container(key=TOP_BAR_NAME, horizontal=True, horizontal_alignment="right"):
        if lead is not None:
            lead()
        if own_key() is not None or platform_user() is not None:
            _account_menu()
        else:
            _signed_out_controls()
    # Outside the container, and outside the branch, because the verdict on a key
    # has to outlive the control that produced it. Accepting one reruns, and the
    # rerun draws the account menu instead of the form -- so a report made only
    # inside the popover is a report nobody ever sees, which is how "accepted, but
    # nobody could check it" went unsaid. Drawn here it lands in the page body under
    # the bar, on whichever of the two runs still has something to say.
    _report_last_key_check()


def _signed_out_controls() -> None:
    """Draw the two sign-in buttons and the key form, for a session with neither."""
    configured = login_configured()
    reason = (
        "Sign in to keep a conversation across sessions."
        if configured
        else "Accounts are not configured for this deployment. You can use your own key instead."
    )
    st.button(
        "Log in",
        key="log_in",
        disabled=not configured,
        help=reason,
        on_click=st.login if configured else None,
    )
    st.button(
        "Sign up for free",
        key="sign_up",
        type="primary",
        disabled=not configured,
        help=reason,
        on_click=st.login if configured else None,
    )
    with st.popover("Use a key"):
        _own_key_form()


def _account_menu() -> None:
    """Draw whoever is here, with what they can do about it behind their name.

    A popover rather than a permanently expanded block: once somebody is signed in
    or running on their own key, their account is the least interesting thing on
    the page and should take one line until they ask it to take more.
    """
    verified = platform_user()
    label = verified.display if verified is not None else "Your key"
    with st.popover(label):
        if verified is not None:
            st.caption(verified.explain())
            if hasattr(st, "logout"):
                st.button("Sign out", on_click=st.logout, width="stretch")
        if own_key() is not None:
            st.caption("Answers run on your own key. It is never written to disk.")
            st.button("Forget my key", on_click=_forget_key, width="stretch")


def _own_key_form() -> None:
    """Offer the bring-your-own-credential path.

    Worth offering even on a laptop that already has a key in ``.env``: somebody
    demonstrating this may want the calls billed to their own account rather than
    the host's, and on a deployment it is the only way a visitor spends their own
    budget instead of whoever deployed it.

    A form, so the key is submitted once rather than on every keystroke -- a text
    input outside one would put a partial credential through a rerun for each
    character typed.

    The key is checked before it is accepted, and skipping that check would be
    worse here than in most applications. Every number in an answer is computed and
    cross-checked before a model is consulted, so a key that fails every call still
    produces a correct answer written by the offline fallback: nothing errors and
    the degradation is invisible. See :mod:`src.agent.credential_check`.
    """
    st.caption(
        "Paste an OpenRouter key to have answers written on your own account. "
        "Held in this browser session only — never written to disk, never logged. "
        "[Get one](https://openrouter.ai/keys)."
    )
    with st.form("own_key", clear_on_submit=True, border=False):
        typed = st.text_input("OpenRouter key", type="password", label_visibility="collapsed")
        submitted = st.form_submit_button("Use this key", width="stretch")
    if submitted:
        _accept_key(typed)
    # A refused key does not rerun, so its verdict is still this run's to draw and
    # the place for it is here, under the box it was typed into. An accepted one
    # has already reraised past this line; :func:`account_bar` reports that one.
    _report_last_key_check()


def _accept_key(typed: str) -> None:
    """Check a pasted credential and store it if there is no reason not to.

    The verdict goes into session state rather than onto the screen, because
    storing the key calls :func:`streamlit.rerun` and anything drawn before that is
    discarded with the rest of the run.

    Args:
        typed: What was pasted, unstripped.
    """
    with st.spinner("Checking the key…"):
        status = check_key(typed)
    st.session_state[_KEY_CHECK_STATE] = (status.usable, status.checked, status.reason)
    if not status.usable:
        return
    st.session_state[OWN_KEY_STATE] = typed.strip()
    LOG.info("own_key_accepted", extra={"verified": status.checked})
    st.rerun()


def _report_last_key_check() -> None:
    """Show the verdict on the key most recently pasted, once.

    Read and cleared, so the sentence stays on screen for exactly one run and does
    not reappear beside an unrelated control later. Called from two places and it
    has to be: a refusal is reported in the same run, inside the popover, while an
    acceptance is reported by :func:`account_bar` on the run after the rerun --
    by which point the popover that would have shown it is no longer drawn.
    """
    remembered = st.session_state.pop(_KEY_CHECK_STATE, None)
    if remembered is None:
        return
    usable, checked, reason = remembered
    if not usable:
        st.error(reason)
    elif checked:
        st.success(reason)
    else:
        st.warning(reason)


def _forget_key() -> None:
    """Drop a pasted credential from session state.

    A button rather than only a tab close, because a visitor on a shared machine
    should be able to take their key back without trusting that closing the tab
    did it.
    """
    st.session_state.pop(OWN_KEY_STATE, None)
    LOG.info("own_key_forgotten")
