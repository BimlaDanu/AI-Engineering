"""Who is visiting, and what that entitles them to.

The problem this solves is a hosting problem rather than a security one. On a
hosted deployment every model call is billed to whoever deployed it. An open Ask
button on a paid model ladder hands a stranger the host's budget; removing the
button leaves a demo that demonstrates nothing.

So there are four kinds of visitor, and they differ in exactly one thing: **which
models they may spend.**

===============  ==================================  ==========================
Visitor          Models                              How they got there
===============  ==================================  ==========================
``operator``     the full tier ladder, on the        no identity provider is
                 host's key                          configured at all
``guest``        the cheapest one there is, on the   opened the link on a
                 host's key                          deployment that has one
``signed_in``    the full tier ladder, on the        signed in with an identity
                 host's key                          provider
``own_key``      the full tier ladder, on theirs     pasted an OpenRouter key
===============  ==================================  ==========================

Nothing else is gated, and that is the design. A guest reaches every page, every
figure, the whole corpus, the cross-checks and the device arithmetic -- because none
of that calls a model. Every number in an answer is computed by :mod:`src.physics`
and cross-checked before a model is consulted, so what a guest gives up is some
quality of *prose* and none of the arithmetic. Most applications cannot offer a free
tier without gutting the product; this one can, and the reason it can is the claim
the whole project is built to demonstrate.

A guest's model selector is disabled rather than overridden. Pinning the guest
slug behind a selector that still shows ``openai/gpt-4o`` would be the dead-knob
failure this repository has a whole document about: the dial moves, the label agrees
with the dial, and the process ignores both. Guests are told, once, in a sentence.

Degrading when there is no identity provider
--------------------------------------------
Sign-in needs ``[auth]`` in Streamlit's secrets and the ``authlib`` package. A
checkout has neither, and ``make run`` has to keep working on a laptop with no
configuration at all -- so :func:`sign_in_configured` answers before anything is
drawn, and the log-in buttons are rendered **disabled with the reason in the
tooltip**.

Disabled rather than either alternative, and both alternatives were tried. Hidden,
and the deployment looks like one that simply has no accounts -- indistinguishable
from broken to anybody who was told to sign in, and it is what the first version of
this module shipped. Live, and pressing one raises. The third state is the only one
that tells the truth.

This module holds policy and one thin Streamlit-facing layer, in that order. The
policy functions take plain values and are tested without a browser; only
:func:`visitor` and :func:`account_bar` touch ``st``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import streamlit as st

from src.agent.credential_check import check_key
from src.agent.model_selection import Tier, guest_overrides
from src.settings import GUEST_CHAT_MODEL, get_settings

Level = Literal["guest", "operator", "signed_in", "own_key"]
"""How much of the application a visitor has paid for, one way or another.

``operator`` is the local case and it is not a lesser ``signed_in``: it means the
deployment has no identity provider at all, so whoever has the page open is whoever
started the process. They are not asked to sign in to something that does not exist,
and nothing is withheld from them.
"""

KEY_CHECK_STATE = "visitor_api_key_verdict"
"""Session-state key holding the verdict on the last credential pasted.

The verdict and not the credential: three primitives, no key. It exists because
accepting a key reruns the script, and a message written before the rerun is thrown
away with the rest of that run's output.
"""

OWN_KEY_STATE = "visitor_api_key"
"""Session-state key holding a pasted credential.

Session state and nothing else: it lives in one browser session's server-side
store, it is never written to disk, never logged, and never put in a URL. A key
pasted here is gone when the tab closes, which is the only promise about somebody
else's credential this application is in a position to keep.
"""

GUEST_CALL_CEILING = 10
"""Model calls one guest campaign may make.

A cost control with a fairness one behind it: the guest model is the cheapest the
project ships, so one question costs a fraction of a cent and an *afternoon* of them
is unbounded -- this is what keeps "unbounded" out of it. And on a deployment that
has moved the guest tier to a free endpoint
(:data:`src.settings.GUEST_FREE_MODEL`) the upstream rate limit is shared between
every visitor at once, so one of them climbing a long ladder can exhaust it for
everybody else looking at the demo.

**Ten because a feasibility question measured nine.** This was six on the reasoning
that six covers "read the question, plan, solve, analyse, explain, suggest", and
then ``make timing`` was pointed at the guest model and ran the route: query
rewrite, problem reading, shelf choice, query expansion, passage ordering, three
depth suggestions and a follow-up. Nine. At six a guest would have lost the depth
suggestions and the follow-up buttons -- and lost them *quietly*, because a spent
budget is a refusal the campaign absorbs and keeps going from, which is exactly the
kind of degradation nobody notices until they compare two screenshots.

So the number is one measurement plus a call of headroom, not an argument about
what ought to be enough. Above it, the depth ladder still climbs -- it is
deterministic and the model only *suggests* where to jump -- so what a guest at the
wall loses is the suggestion and not the answer.
"""


@dataclass(frozen=True, slots=True)
class Visitor:
    """One visitor, and what they may spend.

    Attributes:
        level: Which of the four kinds this is.
        name: Display name from the identity provider, or ``""``.
        email: Address from the identity provider, or ``""``. Held only to show
            the visitor which account they are signed in as; nothing keys off it,
            because this application has no per-user data to key.
    """

    level: Level = "guest"
    name: str = ""
    email: str = ""

    @property
    def is_guest(self) -> bool:
        """Whether this visitor is confined to the guest model.

        Returns:
            ``True`` only for a guest, which requires an identity provider to be
            configured and nobody to have signed in. A local run has no provider,
            so the operator is not a guest and nothing is withheld from them.
        """
        return self.level == "guest"

    @property
    def may_choose_models(self) -> bool:
        """Whether the tier selectors should be live for this visitor.

        Returns:
            ``True`` when the visitor is spending a credential of their own or one
            they signed in for -- which is the only case where choosing a model is
            a choice rather than a wish.
        """
        return not self.is_guest

    @property
    def greeting(self) -> str:
        """How to name this visitor in the account menu.

        Returns:
            A first name, an email address, or a description of the access level --
            whichever is the most specific thing actually known.
        """
        if self.name:
            return self.name.split()[0]
        if self.email:
            return self.email
        return {"own_key": "your own key", "operator": "this machine"}.get(self.level, "guest")


def overrides_for(visitor: Visitor) -> dict[Tier, str]:
    """Tier slugs this visitor's campaigns must use, if any.

    Args:
        visitor: Who is asking.

    Returns:
        Every tier pinned to the guest model for a guest, and an empty mapping
        otherwise -- which leaves the configured ladder and the
        visitor's own selection alone.

    Examples:
        >>> overrides_for(Visitor("signed_in"))
        {}
        >>> sorted(overrides_for(Visitor()))
        ['fast', 'standard', 'strong']
    """
    return guest_overrides() if visitor.is_guest else {}


def call_ceiling_for(visitor: Visitor, configured: int) -> int:
    """How many model calls one campaign may make for this visitor.

    Args:
        visitor: Who is asking.
        configured: The ceiling the knob is set to.

    Returns:
        The lower of the configured ceiling and the guest allowance, for a guest;
        the configured ceiling otherwise. The *lower* rather than the guest figure
        outright, so a host who tightens the knob to three is not silently loosened
        back up to the guest allowance.

    Examples:
        >>> call_ceiling_for(Visitor(), 12)
        10
        >>> call_ceiling_for(Visitor(), 3)
        3
        >>> call_ceiling_for(Visitor("own_key"), 12)
        12
    """
    if visitor.is_guest:
        return min(configured, GUEST_CALL_CEILING)
    return configured


def sign_in_configured() -> bool:
    """Whether an identity provider has been set up for this deployment.

    Answered by looking for an ``[auth]`` section in Streamlit's secrets, and
    answered defensively: reading ``st.secrets`` on a checkout with no secrets file
    raises, which is the *normal* case for a developer running ``make run`` and must
    not be an error. Any failure means "not configured", because the only thing this
    answer is used for is whether to draw a button.

    Returns:
        ``True`` when pressing the sign-in button would reach a provider.
    """
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def _identity() -> tuple[bool, str, str]:
    """Read the signed-in identity out of ``st.user``.

    Separated from :func:`visitor` because it is the one part that is entirely
    somebody else's shape: ``st.user`` is absent on old versions, empty when no
    provider is configured, and dict-like rather than an object.

    Returns:
        Whether a visitor is signed in, their display name and their email --
        empty strings for anything the provider did not supply.
    """
    user = getattr(st, "user", None)
    if user is None:
        return False, "", ""
    try:
        if not bool(user.get("is_logged_in", False)):
            return False, "", ""
        return True, str(user.get("name", "")), str(user.get("email", ""))
    except Exception:
        return False, "", ""


def visitor() -> Visitor:
    """Who is looking at this page.

    Signing in wins over a pasted key, on the grounds that it is the more
    deliberate of the two and the one whose identity can be shown back.

    There is no such thing as a guest on a deployment with no way to sign in, and
    getting that wrong broke the application. The guest tier exists to stop a
    stranger on a public link spending the host's budget; it was applied whenever
    nobody was signed in, which on a laptop running ``make run`` is *always* --
    so the operator, whose own credential is in the environment, was treated as an
    anonymous visitor: every tier pinned to the cheapest model and all three model
    selectors disabled, on their own machine, with their own key.

    So the guest tier is now conditional on there being a door to walk through. If
    :func:`sign_in_configured` is false there is no sign-in to withhold anything
    behind, the person at the keyboard is the person who configured the process, and
    they get the ladder they configured. That is not a loosening of the control: on
    the deployment the control was written for, a provider is configured and an
    anonymous visitor is still a guest.

    Returns:
        The current visitor. A guest only when a provider is configured and nobody
        has signed in.
    """
    signed_in, name, email = _identity()
    if signed_in:
        return Visitor("signed_in", name, email)
    if st.session_state.get(OWN_KEY_STATE):
        return Visitor("own_key")
    if not sign_in_configured():
        return Visitor("operator")
    return Visitor()


TOP_BAR_NAME = "quay_top_bar"
"""The container key the account controls are drawn in.

Streamlit turns a container's ``key`` into an ``st-key-<key>`` class on the element,
which is the one documented hook for styling a *particular* container. The rule that
uses it is :data:`src.ui.panels.TOP_BAR_CSS`; the name lives here, beside the
container it names, and `panels` reads it -- the other direction would be a cycle,
since `panels` is what calls this function.
"""


def account_bar(lead: Callable[[], None] | None = None) -> None:
    """Draw the account controls at the top right of the page, above everything.

    Args:
        lead: Optional control to draw immediately to their left, in the same row.
            The chat page passes its ``New chat`` button, so the page has one top
            bar rather than a control row and an account row at different heights.


    Shaped after the pattern every visitor already knows. An unauthenticated
    visitor sees two buttons in the corner -- a quiet *Log in* and a filled *Sign up
    for free* -- and a signed-in one sees their own name with the account menu
    behind it. That is where a person's eye already goes to answer "whose session is
    this?", so putting it anywhere else costs them a hunt around the page for a
    control they were not expecting to have to find.

    Both buttons start the same flow, and that is not a shortcut: with an identity
    provider there is no separate *sign up* -- a visitor without an account creates
    one at Google and arrives back here signed in, which is exactly what pressing
    *Sign up for free* on a site like this one has always done. Two buttons rather
    than one because they answer two different questions ("I have an account" and
    "I do not"), and a visitor who sees only *Log in* assumes they need one already.

    When no provider is configured the buttons are drawn disabled with the reason
    in the tooltip, rather than hidden or left live. Hidden, and the deployment
    looks like it simply has no accounts -- which is indistinguishable from broken to
    anybody who was told to sign in. Live, and pressing one raises. Disabled with a
    sentence is the only one of the three that tells the truth.

    What is drawn and what is withheld are two different questions, and coupling
    them broke this twice. First these buttons were hidden whenever no provider was
    configured, so a local run showed no account controls at all and the feature
    looked absent. Then the fix for a worse bug -- an operator having their own model
    selectors disabled on their own machine -- was applied here too, and the buttons
    vanished again. They are separate: *entitlement* is
    :func:`overrides_for` and :func:`call_ceiling_for`, and an operator is
    unrestricted there; *appearance* is this function, and the account controls are
    part of the product's face whether or not this particular deployment has a door
    behind them.
    """
    who = visitor()
    # A right-aligned horizontal container rather than a row of spacer columns.
    # Columns were the first attempt and they were wrong twice over: the widths had
    # to be guessed, so the controls sat somewhere in the middle of the page rather
    # than in the corner; and `width="stretch"` inside a narrow column wrapped
    # "Sign up for free" onto two lines, which made the whole row taller and pushed
    # everything down. Here the buttons size to their own labels and the container
    # pushes them to the edge, so the bar is one line high and actually in the
    # corner -- which is the only place this belongs.
    # Keyed, so the stylesheet in `src.ui.panels` can find *this* container and lift
    # it onto the header strip beside Streamlit's Deploy button. Right-aligned inside
    # a wide content column, these landed a long way in from the window's edge, which
    # reads as the middle of the page rather than as its corner. The alignment below
    # is still what positions them when the window is too narrow for the header to be
    # shared and the rule is dropped.
    with st.container(key=TOP_BAR_NAME, horizontal=True, horizontal_alignment="right"):
        if lead is not None:
            lead()
        if who.level in ("signed_in", "own_key"):
            _account_menu(who)
            return

        configured = sign_in_configured()
        reason = (
            "Unlocks the full model ladder."
            if configured
            else "Accounts are not configured for this deployment — "
            "see .streamlit/secrets.toml.example. You can use your own key instead."
        )
        st.button(
            "Log in",
            disabled=not configured,
            help=reason,
            on_click=st.login if configured else None,
            key="log_in",
        )
        st.button(
            "Sign up for free",
            type="primary",
            disabled=not configured,
            help=reason,
            on_click=st.login if configured else None,
            key="sign_up",
        )
        with st.popover("Use a key"):
            _own_key_form()


def _account_menu(who: Visitor) -> None:
    """Draw the signed-in visitor's own name, with what they can do behind it.

    A popover rather than a permanently expanded block: once somebody is signed in,
    their account is the least interesting thing on the page and should take one
    line until they ask it to take more.

    Args:
        who: The signed-in or own-key visitor.
    """
    with st.popover(f"{who.greeting}"):
        if who.email:
            st.caption(who.email)
        if who.level == "own_key":
            st.caption("Running on your own key. It is never written to disk.")
            st.button("Forget my key", width="stretch", on_click=_forget_key)
        elif sign_in_configured():
            st.caption("The full model ladder is available.")
            st.button("Log out", width="stretch", on_click=st.logout)


def _own_key_form() -> None:
    """Offer the bring-your-own-credential path.

    Drawn inside the account popover in the page's corner, and **that is the whole
    account surface**. There were three at one point -- these buttons, a sidebar
    panel repeating the guest sentence, and an inline caption on the chat page --
    which is three places for one fact and therefore three places for it to
    disagree with itself. The corner is where a visitor already looks for "whose
    session is this?", so the corner is the only place that keeps it.

    Kept as a form so the key is submitted once rather than on every keystroke,
    which would put a partial credential through a rerun for each character typed.

    The key is checked before it is accepted, which the first version of this form
    did not do. It took any non-empty string, and a key with one character
    missing then failed every model call of the next campaign -- silently, because
    the campaign computes and cross-checks its numbers before consulting a model and
    so still produced a correct answer. See :mod:`src.agent.credential_check` for
    what the check asks and why an unverifiable key is accepted rather than refused.
    """
    # Only for an actual guest. An operator -- a local run with no identity provider
    # -- is not browsing as a guest, is not restricted to the cheap model, and
    # telling them otherwise describes a restriction that is not being applied. The
    # key form itself is still worth offering them: somebody demonstrating this may
    # want the calls billed to their own key rather than the host's.
    if visitor().is_guest:
        st.caption(
            f"**Browsing as a guest.** Answers are written by "
            f"`{guest_model()}`, the cheapest model this project ships. Every "
            "number is computed and cross-checked either way, so signing in "
            "changes the prose and not the physics."
        )
    st.caption(
        "Or paste an OpenRouter key. Held in this browser session only — never "
        "written to disk, never logged. [Get one free](https://openrouter.ai/keys)."
    )
    with st.form("own_key", clear_on_submit=True, border=False):
        typed = st.text_input("OpenRouter key", type="password", label_visibility="collapsed")
        submitted = st.form_submit_button("Use this key", width="stretch")
    if submitted:
        _accept_key(typed)
    _report_last_key_check()


def _accept_key(typed: str) -> None:
    """Check a pasted credential and store it if there is no reason not to.

    The verdict is kept in session state rather than written to the screen here,
    because storing the key calls :func:`streamlit.rerun` and anything drawn before
    that is discarded -- which is how the first version of this managed to validate
    the key and then show the visitor nothing at all.

    Args:
        typed: What was pasted, unstripped.
    """
    with st.spinner("Checking the key…"):
        status = check_key(typed)
    st.session_state[KEY_CHECK_STATE] = (status.usable, status.checked, status.reason)
    if not status.usable:
        return
    st.session_state[OWN_KEY_STATE] = typed.strip()
    st.rerun()


def _report_last_key_check() -> None:
    """Show the verdict on the key most recently pasted, once.

    Read and cleared, so a reason stays on screen for exactly the rerun that
    follows the paste and does not reappear beside an unrelated control later.
    """
    remembered = st.session_state.pop(KEY_CHECK_STATE, None)
    if remembered is None:
        return
    usable, checked, reason = remembered
    if not usable:
        st.error(reason)
    elif checked:
        st.success(reason)
    else:
        st.warning(reason)


def guest_model() -> str:
    """The model a guest is actually served, named as it appears on screen.

    Read from configuration rather than from :data:`GUEST_CHAT_MODEL` directly,
    because the constant is only the *default*: a deployment sets
    ``GUEST_CHAT_MODEL`` in the environment and
    :func:`~src.agent.model_selection.guest_overrides` honours the setting. Naming
    the constant here would tell a guest they were running on one model while a
    different one answered them, which is the class of defect this project spends a
    whole document on.

    Returns:
        The slug's last segment, which is the part a reader recognises.
    """
    try:
        configured = get_settings().guest_chat_model or GUEST_CHAT_MODEL
    except Exception:
        configured = GUEST_CHAT_MODEL
    return configured.split("/")[-1]


def _forget_key() -> None:
    """Drop a pasted credential from session state.

    A button rather than only a tab close, because a visitor on a shared machine
    should not have to trust the tab.
    """
    st.session_state.pop(OWN_KEY_STATE, None)


def own_key() -> str:
    """The credential this visitor pasted, if any.

    A function rather than a field on :class:`Visitor` so that the value does not
    travel inside a record that gets logged, cached or rendered. Callers reach for
    it at the moment they build a client and nowhere else.

    Returns:
        The pasted key, or ``""``.
    """
    return cast("str", st.session_state.get(OWN_KEY_STATE, ""))
