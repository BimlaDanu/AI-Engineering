"""The models this OpenRouter subscription can actually reach.

Every model name in the project comes from here, and nothing else hard-codes a
slug. The reason is narrow and practical: a model name that is wrong fails at
the first call with an opaque provider error, usually several minutes into a run
and usually after the expensive part. Naming them in one typed place turns that
into a lookup that fails at import.

The display names were read off the account; the slugs are a claim. The names
below are transcribed from what the subscription itself shows. The OpenRouter
slug beside each one is this project's best reconstruction of the identifier the
API expects, and for the newer models it is a reconstruction rather than a fact. Each card therefore
carries :attr:`ModelCard.confirmed`, and the slug verifier
resolves the whole catalogue against the live ``/api/v1/models`` endpoint and
reports what does not exist. Run it once before trusting a slug in anger.

What the project actually needs
-------------------------------
Two roles, and the rest is inventory. A **chat** model that follows a schema
reliably and is cheap enough to run a full evaluation sweep against, and an
**embedding** model that is stable, since changing it invalidates every vector
already in the store. Everything else here -- transcription, image, rerank -- is
listed so that the agent's tool layer can see what exists, not because anything in
this project reaches for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelRole = Literal["chat", "embedding", "rerank", "transcription", "image"]
"""What a model is for. The agent only ever selects among ``chat`` models."""

SubscriptionTier = Literal["basic", "advanced", "both"]
"""Which tier of the subscription exposes a model."""


@dataclass(frozen=True, slots=True)
class ModelCard:
    """One model, as this project needs to know it.

    Attributes:
        display_name: The name shown by the subscription. Transcribed, not
            inferred, and therefore the field to trust.
        slug: The identifier sent to OpenRouter. Believed correct; see
            ``confirmed``.
        role: What the model is for.
        tier: Which subscription tier exposes it.
        confirmed: Whether this slug has been resolved against the live
            OpenRouter model list. ``False`` means "reconstructed from the
            display name and not yet checked", which is the honest default for
            every model released after this project's reference material.
        note: Why the project would or would not reach for it.
    """

    display_name: str
    slug: str
    role: ModelRole
    tier: SubscriptionTier
    confirmed: bool = False
    note: str = ""


CATALOGUE: tuple[ModelCard, ...] = (
    # --- chat, small and cheap: the working tier ----------------------------
    ModelCard(
        "GPT-4o-mini",
        "openai/gpt-4o-mini",
        "chat",
        "both",
        confirmed=True,
        note="Long-established slug and the safest default. Cheap enough to run "
        "the whole evaluation sweep repeatedly.",
    ),
    ModelCard(
        "GPT-4.1 Mini",
        "openai/gpt-4.1-mini",
        "chat",
        "both",
        confirmed=True,
        note="Better instruction following than 4o-mini at a similar price.",
    ),
    ModelCard("GPT-4.1 Nano", "openai/gpt-4.1-nano", "chat", "both", confirmed=True),
    ModelCard("GPT-3.5 Turbo", "openai/gpt-3.5-turbo", "chat", "both", confirmed=True),
    # `anthropic/claude-3.5-haiku` stood here, marked confirmed and offered on the
    # shortlist. The slug verifier found the gateway no longer serves
    # it: it had been retired upstream while this file still promised it, so the
    # model selector offered an option that fails at the first call. Removed rather
    # than marked unconfirmed, because `selectable_slugs` still offers unconfirmed
    # slugs -- only absence from the catalogue takes an option off the selector, and
    # that is exactly what `featured_slugs`' filter exists to notice.
    ModelCard("Claude 3 Haiku", "anthropic/claude-3-haiku", "chat", "both", confirmed=True),
    ModelCard(
        "Claude Haiku 4.5",
        "anthropic/claude-haiku-4.5",
        "chat",
        "both",
        confirmed=True,
        note="The newest generation this subscription reaches, and the strong "
        "tier's default. Confirmed the hard way rather than by listing: "
        "`make bakeoff` drove a whole campaign through this slug live, and it "
        "read the chain correctly and returned the baseline to nine decimal "
        "places. A successful call is "
        "stronger evidence than an entry in a model list. Rejects "
        "`reasoning_effort`; src.agent.llm.REASONING_MODELS keeps the two apart.",
    ),
    # --- chat, free: what a visitor is served on a public deployment ---------
    ModelCard(
        "Gemma 4 31B (free endpoint)",
        "google/gemma-4-31b-it:free",
        "chat",
        "both",
        confirmed=True,
        note="Billed at zero in both directions -- verified against the live "
        "index, not inferred from the `:free` suffix. Offered as the guest tier "
        "(src.settings.GUEST_FREE_MODEL) but NOT the default, because calling any "
        "free endpoint on this account returns 404: free providers are reached "
        "only by accounts that opted into training on prompts, and this one has "
        "not. Reachable by setting GUEST_CHAT_MODEL after opting in.",
    ),
    ModelCard("GPT-4o", "openai/gpt-4o", "chat", "advanced", confirmed=True),
    ModelCard(
        "GPT-4o (2024-11-20)",
        "openai/gpt-4o-2024-11-20",
        "chat",
        "advanced",
        confirmed=True,
        note="Pinned snapshot. Prefer this over the floating alias when a result "
        "has to be reproducible months later.",
    ),
    # --- chat, newer: slugs reconstructed and not yet resolved --------------
    ModelCard(
        "GPT-5 Mini",
        "openai/gpt-5-mini",
        "chat",
        "both",
        confirmed=True,
        note="The standard tier's default -- see src.settings.DEFAULT_CHAT_MODEL. "
        "Confirmed the same way Haiku 4.5 was, by a live `make bakeoff` campaign "
        "that read the chain correctly and returned the baseline to nine decimal "
        "places. It had been sitting here unconfirmed while serving every real "
        "run, which nothing caught because the tier table named a *different* "
        "slug for that tier and the test checked the one nobody used.",
    ),
    ModelCard("GPT-5 Nano", "openai/gpt-5-nano", "chat", "both", confirmed=True),
    ModelCard("GPT-5.2", "openai/gpt-5.2", "chat", "both", confirmed=True),
    ModelCard("GPT-5.2-Codex", "openai/gpt-5.2-codex", "chat", "both", confirmed=True),
    ModelCard("GPT-5.4", "openai/gpt-5.4", "chat", "both", confirmed=True),
    ModelCard("GPT-5.4 Mini", "openai/gpt-5.4-mini", "chat", "both", confirmed=True),
    ModelCard("GPT-5.4 Nano", "openai/gpt-5.4-nano", "chat", "both", confirmed=True),
    ModelCard("Gemini 2.5 Flash", "google/gemini-2.5-flash", "chat", "both", confirmed=True),
    ModelCard(
        "Gemini 2.5 Flash Lite", "google/gemini-2.5-flash-lite", "chat", "both", confirmed=True
    ),
    ModelCard("Gemma 4 31B", "google/gemma-4-31b-it", "chat", "both", confirmed=True),
    ModelCard("MiniMax M2.7", "minimax/minimax-m2.7", "chat", "both", confirmed=True),
    ModelCard("GLM 5.2", "z-ai/glm-5.2", "chat", "advanced", confirmed=True),
    ModelCard(
        "DeepSeek v4 Flash", "deepseek/deepseek-v4-flash", "chat", "advanced", confirmed=True
    ),
    ModelCard("DeepSeek v4 Pro", "deepseek/deepseek-v4-pro", "chat", "advanced", confirmed=True),
    # --- embeddings: changing one invalidates the whole vector store --------
    ModelCard(
        "Text Embedding 3 Small",
        "openai/text-embedding-3-small",
        "embedding",
        "both",
        confirmed=True,
        note="The project default. 1536 dimensions, cheap, and the collection "
        "name in Chroma is qualified by it so a swap cannot silently mix vectors.",
    ),
    ModelCard(
        "Text Embedding 3 Large",
        "openai/text-embedding-3-large",
        "embedding",
        "basic",
        confirmed=True,
        note="3072 dimensions. Better recall on the physics corpus, and roughly "
        "double the storage and the embedding bill.",
    ),
    ModelCard("Qwen3 Embedding 8B", "qwen/qwen3-embedding-8b", "embedding", "both"),
    # --- inventory: present on the subscription, not used by this project ---
    ModelCard("Rerank 4 Pro", "cohere/rerank-4-pro", "rerank", "advanced"),
    ModelCard("Whisper Large V3 Turbo", "openai/whisper-large-v3-turbo", "transcription", "both"),
    ModelCard(
        "Nano Banana (Gemini 2.5 Flash Image)",
        "google/gemini-2.5-flash-image",
        "image",
        "basic",
        confirmed=True,
    ),
)
"""Every model the subscription exposes, transcribed from what it lists."""


def models_for(role: ModelRole, confirmed_only: bool = False) -> tuple[ModelCard, ...]:
    """Every catalogued model of a given role.

    Args:
        role: The role to filter by.
        confirmed_only: Return only slugs that have been resolved against the
            live OpenRouter model list. Use this when picking a model
            programmatically; a run that fails on an unverified slug fails late
            and expensively.

    Returns:
        The matching cards, in catalogue order.
    """
    return tuple(
        card for card in CATALOGUE if card.role == role and (card.confirmed or not confirmed_only)
    )


def card_for(slug: str) -> ModelCard | None:
    """The card for a slug, or ``None`` if the subscription does not list it.

    Returning ``None`` rather than raising is deliberate: a slug set in ``.env``
    that is absent here is a warning worth surfacing, not a reason to refuse to
    start. The subscription can change without this file being updated, and a
    catalogue that could halt the application would be worse than the problem it
    guards against.
    """
    for card in CATALOGUE:
        if card.slug == slug:
            return card
    return None


def unconfirmed_slugs() -> tuple[str, ...]:
    """Slugs that have never been resolved against the live model list.

    Surfaced on the environment page of the monitor, so that "the model name is
    a guess" is visible before a run rather than after one.
    """
    return tuple(card.slug for card in CATALOGUE if not card.confirmed)
