"""Resolve every catalogued model slug against the live OpenRouter index.

:mod:`src.model_catalogue` transcribes what the subscription *displays* and then
reconstructs the identifier the API expects. The display name is ground truth;
the slug is a claim. This script is the thing that turns the claim into a fact,
and it existed as a promise in that module's docstring long before it existed as
a file -- which is exactly the sort of gap this project is meant to close rather
than accumulate.

Why it earns its place in the repository
----------------------------------------
A wrong slug fails at the *first* call of a run, several minutes in and usually
after the expensive part, with a provider error that names nothing useful. One
read-only ``GET`` here replaces that failure with a table. It also answers a
question no amount of local reasoning can: which models the gateway currently
serves **free**, which is what a publicly deployed demo can safely offer a
visitor who has no key of their own.

What it does not do
-------------------
It never writes to the catalogue. A slug that resolves is reported as *ready to
be confirmed*, and flipping :attr:`~src.model_catalogue.ModelCard.confirmed` is
a deliberate edit by a person -- because ``confirmed`` is a claim this project
makes to its own tests, and a claim a script can grant itself is not a claim.

Usage:
    uv run python -m scripts.verify_model_slugs
    uv run python -m scripts.verify_model_slugs --free
    uv run python -m scripts.verify_model_slugs --free --limit 40

Reaches the network and sends the gateway credential as a bearer token, which is
all it does with it: the key is read by name out of the environment through
:func:`src.settings.get_settings`, never printed, and never written anywhere.

Exits non-zero when a catalogued slug does not resolve, so it can gate a change
to the catalogue.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import httpx

from src.model_catalogue import CATALOGUE, ModelCard
from src.settings import get_settings

TIMEOUT_S = 30.0
"""How long to wait on the index. Generous: this is one request, run by hand."""

FREE_SUFFIX = ":free"
"""How OpenRouter marks a no-charge endpoint of an otherwise paid model.

Checked *as well as* the published price rather than instead of it. The suffix is
a naming convention and conventions drift; a price of zero is the fact. A slug
has to satisfy both to be reported as free here, because a guest tier built on a
model that turns out to bill is a guest tier that bills the person hosting it.
"""


@dataclass(frozen=True, slots=True)
class LiveModel:
    """One entry from the gateway's own index.

    Attributes:
        slug: The identifier to send as ``model``.
        prompt_usd: Published input price per token, as the index reports it.
        completion_usd: Published output price per token.
        context: Context window in tokens, or ``0`` when the index omits it.
    """

    slug: str
    prompt_usd: float
    completion_usd: float
    context: int

    @property
    def is_free(self) -> bool:
        """Whether this endpoint bills nothing for either direction.

        Returns:
            ``True`` only when the price of a prompt token and of a completion
            token are both exactly zero. Not "small": a tier offered to
            strangers on somebody else's key has to be zero or it is a bill
            waiting to be discovered.
        """
        return self.prompt_usd == 0.0 and self.completion_usd == 0.0


def fetch_index() -> tuple[LiveModel, ...]:
    """Read the gateway's model index.

    Returns:
        Every model the gateway currently serves, in the order it lists them.

    Raises:
        httpx.HTTPStatusError: If the gateway refuses the request -- which for a
            bad credential is a 401 and is the honest thing to surface, since
            the alternative is reporting every slug in the catalogue as missing.
    """
    settings = get_settings()
    response = httpx.get(
        f"{settings.openrouter_base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"},
        timeout=TIMEOUT_S,
    )
    response.raise_for_status()
    entries = response.json().get("data", [])
    return tuple(_read_entry(entry) for entry in entries)


def _read_entry(entry: dict[str, object]) -> LiveModel:
    """Turn one index entry into a :class:`LiveModel`.

    Defensive about shape rather than strict: the index is somebody else's
    schema, it gains fields, and a missing price is worth reporting as unknown
    instead of crashing a verification run.

    Args:
        entry: One object from the index's ``data`` array.

    Returns:
        The fields this script uses.
    """
    pricing = entry.get("pricing")
    prices = pricing if isinstance(pricing, dict) else {}
    return LiveModel(
        slug=str(entry.get("id", "")),
        prompt_usd=_as_float(prices.get("prompt")),
        completion_usd=_as_float(prices.get("completion")),
        context=int(_as_float(entry.get("context_length"))),
    )


def _as_float(value: object) -> float:
    """Read a price or a size that may arrive as a string, or not at all.

    Args:
        value: Whatever the index put in the field.

    Returns:
        The number, or ``0.0`` when the field is absent or unparseable.
    """
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def audit(catalogue: tuple[ModelCard, ...], live: tuple[LiveModel, ...]) -> tuple[int, int, int]:
    """Print the catalogue against the index and count what needs attention.

    Three outcomes worth distinguishing, and they are printed as three lists
    rather than one annotated one, because they call for three different
    actions: fix the slug, flip the flag, or do nothing.

    **Chat models only.** The index answers for what the completions endpoint
    serves, and nothing else: every embedding, rerank and transcription slug in
    the catalogue is absent from it *while working perfectly* -- retrieval in
    this project runs on ``openai/text-embedding-3-small``, which the index does
    not list. Auditing those roles here would report six healthy models as
    broken, and a checker that cries wolf gets ignored on the run where it is
    right. They are counted and named as out of scope instead.

    Args:
        catalogue: The project's cards.
        live: What the gateway serves.

    Returns:
        The number of chat slugs that did not resolve, the number that resolved
        but are still marked unconfirmed, and the number already correct.
    """
    index = {model.slug: model for model in live}
    chat = tuple(card for card in catalogue if card.role == "chat")
    skipped = tuple(card for card in catalogue if card.role != "chat")
    missing = [card for card in chat if card.slug not in index]
    unflagged = [card for card in chat if card.slug in index and not card.confirmed]
    settled = [card for card in chat if card.slug in index and card.confirmed]

    print(f"catalogue: {len(chat)} chat slugs against {len(live)} the gateway serves")
    print(f"not audited: {len(skipped)} embedding/rerank/transcription slugs -- the")
    print("completions index does not list them, and they work regardless.\n")

    if missing:
        print(f"DOES NOT RESOLVE ({len(missing)}) -- the slug is wrong or the model is gone:")
        for card in missing:
            print(f"  {card.slug:44} {card.display_name}")
        print()

    if unflagged:
        print(f"RESOLVES, STILL MARKED UNCONFIRMED ({len(unflagged)}) -- ready to flip:")
        for card in unflagged:
            model = index[card.slug]
            rate = f"{model.prompt_usd * 1e6:.2f}/{model.completion_usd * 1e6:.2f}"
            print(f"  {card.slug:44} {rate:>16} USD per Mtok")
        print()

    print(f"confirmed and resolving: {len(settled)}")
    return len(missing), len(unflagged), len(settled)


def report_free(live: tuple[LiveModel, ...], limit: int) -> None:
    """List the endpoints that bill nothing, largest context first.

    What a public deployment needs and cannot work out locally. Sorted by
    context window because that is the axis on which free models differ most for
    this application: a campaign's prompts carry a Hamiltonian, a device's wiring
    and several retrieved passages, and a 4k window cannot hold them.

    Args:
        live: What the gateway serves.
        limit: How many to print.
    """
    free = sorted(
        (model for model in live if model.is_free and model.slug.endswith(FREE_SUFFIX)),
        key=lambda model: model.context,
        reverse=True,
    )
    print(f"\nfree endpoints ({len(free)}), widest context first:")
    for model in free[:limit]:
        print(f"  {model.slug:52} {model.context:>9,} tokens")
    if len(free) > limit:
        print(f"  ... and {len(free) - limit} more")


def main() -> int:
    """Run the audit.

    Returns:
        ``0`` when every catalogued slug resolves, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--free",
        action="store_true",
        help="also list the endpoints that bill nothing, for a guest tier",
    )
    parser.add_argument("--limit", type=int, default=25, help="how many free endpoints to print")
    arguments = parser.parse_args()

    live = fetch_index()
    missing, unflagged, _ = audit(CATALOGUE, live)
    if arguments.free:
        report_free(live, arguments.limit)

    if missing:
        print(f"\n{missing} slug(s) do not resolve. Fix the catalogue before trusting a run.")
        return 1
    print(f"\nevery catalogued slug resolves; {unflagged} could be marked confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
