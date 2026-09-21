"""Who may spend which models, and what a visitor is told about it.

Two halves, deliberately separate. The policy in :mod:`src.ui.access` takes plain
values and is checked here without a browser -- which is the point of it being
written as functions over a record rather than as branches inside a Streamlit
callback. Then the rendered half: a guest is *told* they are a guest, and the model
selectors a guest cannot use are drawn disabled rather than left live and ignored.

The second half is the one that matters. A gate that quietly rewrites a visitor's
selection passes every unit test in the first half and is still the dead-knob defect
this repository keeps a document about, so the assertion that closes it is about the
widget's ``disabled`` flag and not about the value that reaches the pool.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.agent.model_selection import ALL_TIERS, DEFAULT_TIER_SLUGS, guest_overrides
from src.settings import GUEST_CHAT_MODEL, GUEST_FREE_MODEL
from src.ui.access import (
    GUEST_CALL_CEILING,
    Level,
    Visitor,
    call_ceiling_for,
    overrides_for,
)
from src.ui.status import PROJECT_ROOT

ENTRY_POINT = "src/ui/app.py"
"""The only route on which the shared sidebar -- and so the account block -- exists."""

RENDER_TIMEOUT = 180
"""Generous: a cold first render imports numpy, scipy and the physics layer."""


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------


def test_a_guest_is_pinned_to_the_free_endpoint_on_every_tier() -> None:
    # Every tier rather than the strong one alone. A ladder with one free rung
    # still bills the host for the other two, and the whole promise of the guest
    # path is that opening the demo link cannot cost anybody anything.
    pinned = overrides_for(Visitor())
    assert set(pinned) == set(ALL_TIERS)
    assert set(pinned.values()) == {GUEST_CHAT_MODEL}


def test_the_guest_model_is_the_cheapest_thing_the_project_ships() -> None:
    # Priced, not merely believed to be cheap: a guest tier whose cost cannot be
    # stated is one whose cost is discovered on an invoice. And no dearer than any
    # tier a paying visitor gets, since the entire purpose is that opening the demo
    # link cannot cost the host meaningfully.
    from src.agent.usage import price_for

    price = price_for(GUEST_CHAT_MODEL)
    assert price is not None, "the guest model must be priced, not merely cheap"
    paid = [price_for(slug) for slug in DEFAULT_TIER_SLUGS.values()]
    assert all(rate is not None for rate in paid)
    assert price.input_usd <= min(rate.input_usd for rate in paid if rate is not None)


def test_the_documented_free_endpoint_really_bills_nothing() -> None:
    # `GUEST_FREE_MODEL` is offered as the zero-cost upgrade once an account can
    # reach a free provider. The `:free` suffix is a naming convention and the
    # price is the fact, so the fact is what is asserted -- if a future edit points
    # this at a slug that bills, the promise in its docstring becomes false and
    # nothing else in the suite would notice.
    from src.agent.usage import price_for

    price = price_for(GUEST_FREE_MODEL)
    assert price is not None
    assert (price.input_usd, price.output_usd) == (0.0, 0.0)


def test_a_signed_in_visitor_is_left_on_the_configured_ladder() -> None:
    levels: tuple[Level, ...] = ("signed_in", "own_key")
    for level in levels:
        assert overrides_for(Visitor(level)) == {}


def test_a_guests_call_ceiling_is_lowered_but_never_raised() -> None:
    # The lower of the two, so a host who tightens the knob is not loosened back
    # up by the guest allowance -- a "limit" that can increase a limit is not one.
    assert call_ceiling_for(Visitor(), 12) == GUEST_CALL_CEILING
    assert call_ceiling_for(Visitor(), 3) == 3
    assert call_ceiling_for(Visitor("signed_in"), 12) == 12


MEASURED_FEASIBILITY_CALLS = 9
"""Model calls one live feasibility campaign made on the guest model.

Query rewrite, problem reading, shelf choice, query expansion, passage ordering,
three depth suggestions and one follow-up. Reproduce with::

    uv run python -m scripts.time_a_question \\
        --fast google/gemini-2.5-flash-lite \\
        --standard google/gemini-2.5-flash-lite \\
        --strong google/gemini-2.5-flash-lite \\
        "Is a 10-magnet chain at the critical point worth running on a quantum computer?"

A recorded measurement rather than a derivation, and deliberately so. The first
attempt at this test derived the figure from ``len(TASK_TIERS)``, which is a count
of the *kinds* of call the project can make -- not the number one question makes,
since a feasibility question skips several kinds and asks for a depth three times.
It disagreed with reality in both directions at once.
"""


def test_a_guest_can_afford_the_longest_route_the_graph_takes() -> None:
    # The ceiling was 6, on an argument about which steps matter. Then a live
    # campaign on the guest model made nine calls, so a guest silently lost their
    # depth suggestions and their follow-up buttons -- silently because a spent
    # budget is a refusal the campaign absorbs and carries on from, which is the
    # kind of degradation nobody notices until they compare two screenshots.
    assert GUEST_CALL_CEILING >= MEASURED_FEASIBILITY_CALLS, (
        f"a guest gets {GUEST_CALL_CEILING} calls and the route needs "
        f"{MEASURED_FEASIBILITY_CALLS}; re-measure with scripts/time_a_question.py "
        "before raising or lowering either number"
    )


def test_a_guest_may_not_choose_models_and_everyone_else_may() -> None:
    assert not Visitor().may_choose_models
    assert Visitor("signed_in").may_choose_models
    assert Visitor("own_key").may_choose_models


def test_a_visitor_is_named_as_specifically_as_the_provider_allows() -> None:
    assert Visitor("signed_in", "Ada Lovelace", "ada@example.com").greeting == "Ada"
    assert Visitor("signed_in", "", "ada@example.com").greeting == "ada@example.com"
    assert Visitor("own_key").greeting == "your own key"
    assert Visitor().greeting == "guest"


def test_the_guest_override_names_every_tier_the_ladder_does() -> None:
    # Two places list the tiers, and a tier missing from the guest map would be a
    # tier that silently kept billing.
    assert set(guest_overrides()) == set(DEFAULT_TIER_SLUGS)


# --------------------------------------------------------------------------
# What a visitor sees
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered() -> AppTest:
    """The entry point, rendered once, as it comes out of a checkout.

    No ``.streamlit/secrets.toml``, so no identity provider -- which means the
    visitor is the **operator**, not a guest. Module-scoped because a cold render
    imports the whole physics layer and every assertion below reads the same tree.

    Returns:
        The finished app.
    """
    return AppTest.from_file(str(PROJECT_ROOT / ENTRY_POINT), default_timeout=RENDER_TIMEOUT).run()


@pytest.fixture(scope="module")
def rendered_for_a_guest() -> AppTest:
    """The entry point, rendered as an anonymous visitor to a real deployment.

    An ``[auth]`` section is injected, because that is the only thing that makes a
    guest a guest: with no provider there is nothing to sign in to and nothing to
    withhold. Every guest assertion needs this, and the four that used to rely on
    the checkout's *absence* of a provider were asserting the bug -- they passed
    while the operator's own model selectors were disabled on their own laptop.

    Returns:
        The finished app.
    """
    app = AppTest.from_file(str(PROJECT_ROOT / ENTRY_POINT), default_timeout=RENDER_TIMEOUT)
    app.secrets["auth"] = {
        "redirect_uri": "http://localhost:8501/oauth2callback",
        "cookie_secret": "test-only-not-a-real-secret",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
    }
    return app.run()


def test_a_visitor_with_no_provider_to_sign_into_is_not_called_a_guest(
    rendered: AppTest,
) -> None:
    # What is *drawn* and what is *withheld* are two different questions, and
    # coupling them broke this application twice -- once by hiding the account
    # controls whenever no provider was configured, so the feature looked absent,
    # and once by disabling an operator's own model selectors on their own machine.
    #
    # So: the buttons are drawn, because they are part of the product's face and a
    # deployment with no accounts should say so rather than look accountless. They
    # are disabled with the reason, because live they would raise. And the guest
    # *entitlement* is not applied at all -- no guest sentence, and the model
    # selectors stay live, which the two tests below assert.
    offered = [
        button for button in rendered.button if button.label in {"Log in", "Sign up for free"}
    ]
    assert len(offered) == 2
    assert all(button.disabled for button in offered)
    assert all("not configured" in (button.help or "") for button in offered)
    said = " ".join(element.value for element in rendered.caption).lower()
    assert "browsing as a guest" not in said


def test_an_anonymous_visitor_to_a_deployment_is_told_they_are_a_guest(
    rendered_for_a_guest: AppTest,
) -> None:
    labels = {button.label for button in rendered_for_a_guest.button}
    assert "Log in" in labels
    assert "Sign up for free" in labels


def test_a_guest_is_offered_both_log_in_and_sign_up(rendered_for_a_guest: AppTest) -> None:
    # Two buttons rather than one, in the corner where a person already looks. A
    # visitor who sees only "Log in" assumes they need an account already, and one
    # who sees neither assumes the deployment has no accounts -- which is
    # indistinguishable from broken to anybody who was told to sign in.
    offered = [
        button
        for button in rendered_for_a_guest.button
        if button.label in {"Log in", "Sign up for free"}
    ]
    assert len(offered) == 2
    assert all(not button.disabled for button in offered), (
        "a configured provider should make these live; disabled with a reason is "
        "for the case where there is no provider, and that case now draws no "
        "buttons at all"
    )


def test_the_tier_selectors_a_guest_cannot_use_are_disabled(
    rendered_for_a_guest: AppTest,
) -> None:
    # The assertion that closes the dead-knob failure. Forcing the guest slug behind
    # a live selector still showing `openai/gpt-4o` would satisfy every policy test
    # above while lying on screen about what is running.
    tiers = [box for box in rendered_for_a_guest.selectbox if box.label.endswith("tier")]
    assert tiers, "the three tier selectors were not drawn"
    assert all(box.disabled for box in tiers)


def test_rendering_the_account_block_needs_no_identity_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The normal case for a checkout: no `.streamlit/secrets.toml`, no `authlib`,
    # and `make run` still has to work. A sign-in button that raises when pressed
    # is worse than an absent one, so the button is simply not drawn -- and this is
    # asserted by rendering with the credential removed as well, since the account
    # block is above every control and would take the whole page down with it.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app = AppTest.from_file(str(PROJECT_ROOT / ENTRY_POINT), default_timeout=RENDER_TIMEOUT).run()
    assert not app.exception


def test_a_deployment_with_no_provider_has_no_guests() -> None:
    # The regression this pins, and it broke the application in the most annoying
    # possible way: every tier was overridden to the cheapest model and all three
    # model selectors were disabled, on a laptop, for the person whose own key was
    # in the environment. The guest tier exists to stop a stranger on a public link
    # spending the host's budget. With no identity provider there is no link and no
    # stranger -- so there is nothing to withhold, and withholding it anyway just
    # takes the model choice away from the one person entitled to make it.
    from src.ui.access import Visitor, call_ceiling_for, overrides_for

    operator = Visitor("operator")
    assert not operator.is_guest
    assert operator.may_choose_models
    assert overrides_for(operator) == {}, (
        "an operator's tier choice is being overridden; the model selector in the "
        "sidebar would move and change nothing"
    )
    assert call_ceiling_for(operator, 40) == 40


def test_the_model_selectors_are_live_on_a_checkout_with_no_provider() -> None:
    # Through the rendered application rather than through the policy functions,
    # because the defect was visible only once the two were put together: the
    # policy said "guest" and the widget read the policy.
    rendered = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=120).run()
    tiers = [box for box in rendered.selectbox if box.label.endswith(" tier")]
    assert len(tiers) == 3, f"expected three tier selectors, found {len(tiers)}"
    for box in tiers:
        assert not box.disabled, (
            f"the {box.label} selector is disabled on a checkout with no identity "
            "provider, so nobody can choose a model on their own machine"
        )


def test_choosing_a_model_in_the_sidebar_changes_what_the_campaign_would_run() -> None:
    # The end-to-end version: move the dial, then read the position the campaign is
    # built from. A test that only checked the widget was enabled would have passed
    # throughout the period when the choice was being silently overridden.
    from src.ui import panels

    chosen = "openai/gpt-4o-mini"
    app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180)
    app.run()
    for box in app.selectbox:
        if box.label == "standard tier":
            box.set_value(chosen)
    app.run()
    setting = app.session_state[panels.SETTING_KEY]
    assert dict(setting.model.tiers)["standard"] == chosen
    assert setting.model.overrides()["standard"] == chosen
