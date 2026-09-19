"""The account controls in the top-right corner, and the New chat button beside them.

Shaped after the pattern every visitor already knows. Someone who is not identified sees a
quiet **Log in**, a filled **Sign up for free** and a **Use a key** popover in the corner;
someone who is sees their own name with the account menu behind it. That corner is where a
person's eye already goes to answer "whose session is this?", so putting it anywhere else
costs them a hunt for a control they were not expecting to have to find.

**New chat** rides in the same row rather than in the sidebar. In the sidebar it would sit
directly under the navigation's own 💬 AI Chat link — two chat-shaped controls a centimetre
apart — and it is the one control here that acts on the conversation rather than reporting
something every page needs.

The row is one keyed container so that :data:`src.ui.theme.TOP_BAR_CSS` can find *this*
element and lift it onto Streamlit's header strip beside the Deploy button. Right-aligned
inside the page's wide content column it landed a long way in from the window edge, which
reads as the middle of the page rather than as its corner — and a control in the middle of a
page is a control competing with the answer.

What is *drawn* and what is *available* are separate questions, and coupling them is the
mistake this module is written to avoid. With no identity provider configured the login
buttons are still drawn, disabled, with the reason in the tooltip: hidden, and the deployment
looks like one that simply has no accounts, which is indistinguishable from broken to anybody
who was told to sign in.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from src import auth

TOP_BAR_NAME = "synapse_top_bar"
"""The container key the top-bar controls are drawn in.

Streamlit turns a container's ``key`` into an ``st-key-<key>`` class on the element, which is
the one documented hook for styling a *particular* container. The rule that uses it is
:data:`src.ui.theme.TOP_BAR_CSS`; the name lives here, beside the container it names, and the
stylesheet reads it — the other direction would be a cycle, since this module is what the
entry point calls.
"""

_NOT_CONFIGURED = (
    "Accounts are not set up for this deployment — see `.streamlit/secrets.toml.example`. "
    "You can use your own OpenRouter key instead."
)


def account_bar(lead: Callable[[], None] | None = None) -> None:
    """Draw the top-right controls, above everything else on the page.

    Args:
        lead: An optional control drawn immediately to their left in the same row. The entry
            point passes :func:`new_chat_button`, so the page carries one top bar rather than
            an action row and an account row at two different heights.
    """
    who = auth.visitor()
    # A right-aligned horizontal container rather than a row of spacer columns. Columns were
    # the obvious first attempt and they are wrong twice over: the widths have to be guessed,
    # so the controls sit somewhere in the middle of the page; and a stretched button inside a
    # narrow column wraps "Sign up for free" onto two lines, which makes the whole row taller
    # and pushes the page down. Here each button sizes to its own label and the container
    # pushes the group to the edge, so the bar is one line high.
    with st.container(key=TOP_BAR_NAME, horizontal=True, horizontal_alignment="right"):
        if lead is not None:
            lead()
        if who.is_identified or who.level == "own_key":
            _account_menu(who)
            return
        _login_buttons()
        with st.popover("Use a key"):
            own_key_form()


def _login_buttons() -> None:
    """Draw **Log in** and **Sign up for free**, live or disabled with the reason.

    Both start the same flow, and that is not a placeholder standing in for a registration
    form: with an identity provider there is no separate sign-up — a visitor without an
    account creates one at the provider and returns here signed in. Two buttons because they
    answer two different questions, and a visitor shown only *Log in* assumes they need an
    account already.
    """
    providers = auth.oidc_providers()
    configured = bool(providers)
    # A single-provider config is reached with a bare ``st.login()``; with several, the first
    # is the one the corner buttons use and the rest are offered inside the popover, so the
    # bar stays two buttons wide however many providers a deployment has wired up.
    primary = providers[0] if providers else ""
    help_text = (
        "Unlocks your saved conversations on this deployment." if configured else _NOT_CONFIGURED
    )

    st.button(
        "Log in",
        key="account_log_in",
        disabled=not configured,
        help=help_text,
        on_click=(lambda: auth.begin_login(primary)) if configured else None,
    )
    st.button(
        "Sign up for free",
        key="account_sign_up",
        type="primary",
        disabled=not configured,
        help=help_text,
        on_click=(lambda: auth.begin_login(primary)) if configured else None,
    )


def _account_menu(who: auth.Visitor) -> None:
    """Draw the identified visitor's name, with what they can do behind it.

    A popover rather than a permanently expanded block: once somebody is signed in, their
    account is the least interesting thing on the page and should take one line until they
    ask it to take more.
    """
    with st.popover(who.greeting):
        if who.email:
            st.caption(who.email)
        if who.level == "own_key":
            st.caption("Running on your own OpenRouter key. It is never written to disk.")
            st.button("Forget my key", width="stretch", on_click=auth.forget_own_key)
            return
        if who.level == "signed_in":
            st.caption("Your conversations are kept under this account.")
            st.button("Log out", width="stretch", on_click=auth.end_login)
        elif who.level == "password":
            auth.render_logout(location="main", show_name=False)
        st.divider()
        st.caption("Prefer to be billed for your own questions?")
        own_key_form()


def own_key_form() -> None:
    """Offer the bring-your-own-credential path.

    Kept as a form so the key is submitted once rather than on every keystroke, which would
    put a partial credential through a rerun for each character typed.

    The key is checked before it is accepted. A key with one character missing would
    otherwise fail *quietly*: Synapse retrieves and cites before it generates, so the run
    still looks like it worked right up until the answer, and the failure reads as the model
    having a bad day rather than as a typo in a credential.
    """
    st.caption(
        "Paste an OpenRouter key to have your questions billed to it. Held in this browser "
        "session only — never written to disk, never logged. "
        "[Get one free](https://openrouter.ai/keys)."
    )
    with st.form("own_key_form", clear_on_submit=True, border=False):
        typed = st.text_input(
            "OpenRouter key", type="password", label_visibility="collapsed", placeholder="sk-or-…"
        )
        submitted = st.form_submit_button("Use this key", width="stretch")
    if submitted:
        _accept_key(typed)
    _report_last_key_check()


def _accept_key(typed: str) -> None:
    """Check a pasted credential and store it if there is no reason not to.

    The verdict goes to session state rather than to the screen, because storing the key
    calls :func:`streamlit.rerun` and anything drawn before that is discarded — which is how
    a first attempt at this managed to validate the key and then show the visitor nothing.
    """
    with st.spinner("Checking the key…"):
        status = auth.check_openrouter_key(typed)
    st.session_state[auth.KEY_CHECK_STATE] = (status.usable, status.checked, status.reason)
    if not status.usable:
        return
    auth.set_own_key(typed)
    st.rerun()


def _report_last_key_check() -> None:
    """Show the verdict on the key most recently pasted, once.

    Read and cleared, so a reason stays on screen for exactly the rerun that follows the
    paste and does not reappear beside an unrelated control later.
    """
    remembered = st.session_state.pop(auth.KEY_CHECK_STATE, None)
    if remembered is None:
        return
    usable, checked, reason = remembered
    if not usable:
        st.error(reason)
    elif checked:
        st.success(reason)
    else:
        st.warning(reason)
