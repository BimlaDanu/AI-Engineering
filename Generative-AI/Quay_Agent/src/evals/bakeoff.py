"""Running one question through several language models and measuring the difference.

The question this answers is not "which model is best". It is the sharper one:
**how much of the answer does the choice of model actually decide?**

The project's whole claim is that a language model formalises, plans and writes,
while deterministic code computes and a sealed solver grades. If that claim is
true, then swapping the model must leave the numbers where they are. That is a
falsifiable statement, and this module is the experiment that falsifies it -- the
same question, the same chain, the same budget, the same machine, run once per
model, with three things checked in order of how much they matter:

1. **Did every model read the same problem?** The chain is recovered from ordinary
   language by a model call, so this is the one place a model can move a number,
   and it moves *all* of them at once. Two models that disagree here are answering
   two different questions and nothing downstream is comparable.
2. **Given the same reading, did the number move?** The classical baseline is run
   from a fixed seed at a fixed depth by code no model touches, so for an identical
   reading it must come back identical. A difference here is a bug in the
   architecture, not a fact about the models.
3. **Did the verdict move?** Decided by rule from the numbers. It should not move
   either, and if it does the rule is reading something a model wrote.

Only once those three are settled is it worth looking at what a model legitimately
does change: **wall-clock, calls, tokens, money, and how the prose is written.**
That ordering is deliberate. A bake-off that leads with "model A was 300ms faster"
invites a reader to choose on latency without first establishing that the two
models were answering the same question at all.

Nothing here is graded by a language model. Every quality column is arithmetic or
a word count -- how many passages the answer cites, how many hedging words per
hundred, how early the classical comparison appears -- which is the same rule the
held-out suite runs under and the reason a number on this page can be recomputed
by hand from the report.

This costs money and needs a credential, so it is a function somebody calls
deliberately rather than something a page does when it is opened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.agent.graph import DEFAULT_SHOT_BUDGET, run_campaign
from src.agent.model_selection import ModelPool
from src.agent.usage import Usage, metered
from src.evals.metrics import baseline_position, count_hedges
from src.logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from collections.abc import Sequence

    from src.agent.state import CampaignState
    from src.hardware.devices import Device

_log = get_logger("evals.bakeoff")

WORDS_PER_HEDGE_RATE = 100
"""The denominator hedging is reported against.

Per hundred words rather than as a raw count, because a model that hedges four
times in a two-paragraph answer and one that hedges four times in a two-page answer
are not doing the same thing, and the raw count cannot tell them apart.
"""

BASELINE_TOLERANCE = 1e-9
"""How far two classical baselines may differ and still count as the same number.

Not zero, and the reason is floating-point rather than physics: the baseline is
deterministic given a reading, but the reading arrives as parsed decimals and two
models can produce ``1.0`` and ``1.0000000000000002`` for the same stated coupling.
Anything larger than this is a real difference and is reported as one.
"""


@dataclass(frozen=True, slots=True)
class Trial:
    """One model's run of the shared question, measured.

    Attributes:
        slug: The model that served every call in this run. One model on all three
            tiers, which is the only configuration that makes the comparison a
            comparison -- a run that escalated to a second model would attribute
            that model's work to this one.
        read_as: The chain the campaign decided the question described, as a label.
            The first column to read: two different readings make every other
            column incomparable.
        seconds: Wall-clock for the whole campaign, model calls and arithmetic
            together. What a person waiting actually experiences.
        calls: Model calls made.
        tokens: Prompt and completion tokens, summed.
        cost_usd: Estimated spend at the catalogue's published rates.
        fully_priced: Whether every call had a price on record. False makes
            ``cost_usd`` a floor rather than a total, and saying so is the
            difference between an estimate and a wrong number.
        estimated_cost_usd: The same run repriced with the reasoned rates in
            :data:`src.agent.usage.HEURISTIC_PRICES` filling the gaps, or ``None``
            if even those do not cover it. Read only where ``fully_priced`` is
            false: it is what the table shows instead of the word "unpriced", and
            it is deliberately kept out of :func:`cheapest`, because a winner named
            on an extrapolation recommends the row nobody has costed.
        verdict: Go, no, or conditional. Empty when the branch reached none.
        baseline_energy_per_site: What the classical method returned. Untouched by
            any model, so identical readings must produce identical values here.
        best_energy_per_site: The lowest energy any circuit run reached, or ``None``
            if none ran. This one may legitimately differ between models, because
            the circuit depth is proposed by a model call.
        depth: The layer count behind ``best_energy_per_site``.
        cited: Passages the answer leans on. Taken from the written answer where
            there is one; the feasibility branch writes a report rather than an
            answer, so there it is the campaign's own citation count. Reporting
            zero for a feasibility run that cited four sources would look like a
            retrieval failure and be a reporting one.
        written_by: ``"model"``, ``"notes"`` or ``"refused"``. A run whose prose
            fell back to the notes did not exercise the model being measured, and
            its writing columns say nothing about that model.
        words: Length of the written document.
        hedges: Hedging words in it, counted on word boundaries.
        states_baseline: Whether the document mentions the classical comparison at
            all.
        baseline_at: Where it first appears, as a fraction of the document. Hiding
            the comparison at the end is the failure this looks for; a document
            that mentions it only in a closing appendix has technically mentioned
            it.
        failure: Why the run did not complete, if it did not. Empty on success. A
            model that cannot be reached is a result about that model, so it takes
            a row rather than an exception.
    """

    slug: str
    read_as: str = ""
    seconds: float = 0.0
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    fully_priced: bool = True
    estimated_cost_usd: float | None = None
    verdict: str = ""
    baseline_energy_per_site: float | None = None
    best_energy_per_site: float | None = None
    depth: int | None = None
    cited: int = 0
    written_by: str = ""
    words: int = 0
    hedges: int = 0
    states_baseline: bool = False
    baseline_at: float | None = None
    failure: str = ""

    @property
    def ok(self) -> bool:
        """Whether this run finished and can be compared with the others."""
        return not self.failure

    @property
    def hedge_rate(self) -> float:
        """Hedging words per hundred words, or zero for an empty document."""
        if not self.words:
            return 0.0
        return WORDS_PER_HEDGE_RATE * self.hedges / self.words

    @property
    def seconds_per_call(self) -> float:
        """Wall-clock divided by calls made, or zero when nothing was called.

        A rough figure and deliberately so: it includes the arithmetic between the
        calls, which is the honest thing to charge a model with when the question
        is how long a person waits.
        """
        if not self.calls:
            return 0.0
        return self.seconds / self.calls


@dataclass(frozen=True, slots=True)
class Agreement:
    """What survived the model swap, and what did not.

    The headline of a bake-off, and it is a headline about the architecture rather
    than about any model in it.

    Attributes:
        compared: Runs that finished and could be compared.
        readings: The distinct problems the models decided they had been given.
        baselines: The distinct classical energies, rounded to the tolerance.
        verdicts: The distinct verdicts reached.
        failures: Slugs that did not produce a run at all.
    """

    compared: int
    readings: tuple[str, ...]
    baselines: tuple[float, ...]
    verdicts: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def read_alike(self) -> bool:
        """Whether every model recovered the same chain from the question."""
        return len(self.readings) <= 1

    @property
    def numbers_held(self) -> bool:
        """Whether the computed baseline was the same for every model.

        Only meaningful when :attr:`read_alike`. Two models that were handed
        different chains are *supposed* to produce different numbers, and reporting
        that as a failure of determinism would be reporting the wrong finding.
        """
        return len(self.baselines) <= 1

    @property
    def verdict_held(self) -> bool:
        """Whether the verdict was the same for every model."""
        return len(self.verdicts) <= 1

    def headline(self) -> str:
        """The finding in one sentence, for a page or a log line.

        Returns:
            Plain words, stating what held and what did not. Written so that the
            interesting case -- something moved -- is the loud one.
        """
        if self.compared < 2:
            return "Fewer than two models finished, so there is nothing to compare."
        if not self.read_alike:
            return (
                f"The {self.compared} models did not agree on what was being asked: "
                f"{'; '.join(self.readings)}. Nothing below is comparable until that is."
            )
        if not self.numbers_held:
            return (
                f"Same question, {len(self.baselines)} different computed baselines. "
                "That is an architecture bug -- no model touches this number."
            )
        if not self.verdict_held:
            return (
                f"The numbers held but the verdict moved: {', '.join(self.verdicts)}. "
                "The rule is reading something a model wrote."
            )
        return (
            f"All {self.compared} models read the same chain, computed the same "
            f"baseline and reached the same verdict. What differed is the prose, "
            "the wall-clock and the bill."
        )


def measure(state: CampaignState, slug: str, seconds: float, spent: Usage) -> Trial:
    """Reduce one finished campaign to a comparable row.

    Kept separate from the run so it can be exercised on a campaign built by hand,
    which is how the columns are tested without spending anything.

    Args:
        state: The finished campaign.
        slug: The model that served it.
        seconds: Wall-clock the campaign took.
        spent: The meter's :class:`~src.agent.usage.Usage` for the run.

    Returns:
        One row of the comparison.
    """
    model = state["model"]
    classical = state["classical"]
    verdict = state["verdict"]
    written = state["answer"]
    document = state["report"] or (written.text if written is not None else "")
    best = min(state["runs"], key=lambda run: run.energy_per_site, default=None)
    position = baseline_position(document)
    return Trial(
        slug=slug,
        read_as=model.label() if model is not None else "",
        seconds=round(seconds, 3),
        calls=spent.calls_made,
        tokens=spent.total_tokens,
        cost_usd=spent.cost_usd,
        fully_priced=spent.fully_priced,
        estimated_cost_usd=spent.estimated_cost_usd,
        verdict=verdict.call if verdict is not None else "",
        baseline_energy_per_site=classical.energy_per_site if classical is not None else None,
        best_energy_per_site=best.energy_per_site if best is not None else None,
        depth=best.depth if best is not None else None,
        cited=written.cited if written is not None else len(state["citations"]),
        written_by=written.written_by if written is not None else "",
        words=len(document.split()),
        hedges=count_hedges(document),
        states_baseline=position is not None,
        baseline_at=position,
    )


def run_trial(
    question: str,
    slug: str,
    *,
    framing: Literal["neutral", "vendor", "skeptical"] = "neutral",
    shot_budget: int = DEFAULT_SHOT_BUDGET,
    device: Device | None = None,
    search_corpus: bool = True,
) -> Trial:
    """Run the shared question once, with one model serving every call.

    Every tier is pinned to the same slug on purpose. The pool exists so that a
    campaign can send a tie-break to a cheap model and the report to an expensive
    one, which is the right production behaviour and the wrong experimental one:
    a run that escalated would credit the fast model's latency and the strong
    model's prose to a single row.

    A model that cannot be built or that fails mid-run returns a row carrying the
    failure rather than raising. "This model is unreachable from here" is a result
    about that model and belongs in the table beside the ones that worked.

    Args:
        question: The problem, in the asker's own words. The same string for
            every model.
        slug: The model to pin to all three tiers.
        framing: The voice to ask in. Held constant across the comparison.
        shot_budget: Measurements the campaign may spend.
        device: The machine to price circuits against.
        search_corpus: Whether to look for background sources.

    Returns:
        One row of the comparison, with :attr:`Trial.failure` set if the run did
        not finish.
    """
    pool = ModelPool(overrides={"fast": slug, "standard": slug, "strong": slug})
    started = time.perf_counter()
    try:
        with metered() as meter:
            finished = run_campaign(
                question,
                framing=framing,
                shot_budget=shot_budget,
                device=device,
                models=pool,
                search_corpus=search_corpus,
                fetch_external=False,
                suggest_followups=False,
            )
    except Exception as error:
        elapsed = time.perf_counter() - started
        _log.warning("bakeoff_trial_failed", extra={"model": slug, "error": str(error)})
        return Trial(slug=slug, seconds=round(elapsed, 3), failure=str(error)[:200])
    elapsed = time.perf_counter() - started
    trial = measure(finished, slug, elapsed, meter.usage)
    _log.info(
        "bakeoff_trial",
        extra={
            "model": slug,
            "seconds": trial.seconds,
            "calls": trial.calls,
            "verdict": trial.verdict,
        },
    )
    return trial


def compare(
    question: str,
    slugs: Sequence[str],
    *,
    framing: Literal["neutral", "vendor", "skeptical"] = "neutral",
    shot_budget: int = DEFAULT_SHOT_BUDGET,
    device: Device | None = None,
    search_corpus: bool = True,
) -> tuple[Trial, ...]:
    """Run the same question once per model, in the order given.

    Sequentially rather than concurrently, and the reason is the measurement
    itself: wall-clock is one of the columns, and four campaigns sharing one machine
    and a rate limit would report each other's contention as their own latency.

    Args:
        question: The problem, in the asker's own words.
        slugs: The models to compare. Duplicates are run twice, which is a
            legitimate thing to want -- it measures the spread of a single model
            against the spread between models.
        framing: The voice to ask in.
        shot_budget: Measurements each campaign may spend.
        device: The machine to price circuits against.
        search_corpus: Whether to look for background sources.

    Returns:
        One row per slug, in the order given.
    """
    return tuple(
        run_trial(
            question,
            slug,
            framing=framing,
            shot_budget=shot_budget,
            device=device,
            search_corpus=search_corpus,
        )
        for slug in slugs
    )


def agreement(trials: Sequence[Trial]) -> Agreement:
    """Reduce a set of trials to what did and did not survive the model swap.

    Args:
        trials: The rows to compare. Failed rows are counted as failures and are
            excluded from every other field, because a run that did not happen
            cannot agree or disagree with one that did.

    Returns:
        The comparison. Distinct values are reported in the order first seen
        rather than sorted, so a reader can tell which model introduced the
        disagreement by matching against the table.
    """
    finished = [trial for trial in trials if trial.ok]
    baselines: list[float] = []
    for trial in finished:
        value = trial.baseline_energy_per_site
        if value is None:
            continue
        if not any(abs(value - seen) <= BASELINE_TOLERANCE for seen in baselines):
            baselines.append(value)
    return Agreement(
        compared=len(finished),
        readings=tuple(dict.fromkeys(trial.read_as for trial in finished if trial.read_as)),
        baselines=tuple(baselines),
        verdicts=tuple(dict.fromkeys(trial.verdict for trial in finished if trial.verdict)),
        failures=tuple(trial.slug for trial in trials if not trial.ok),
    )


def cheapest(trials: Sequence[Trial]) -> Trial | None:
    """The finished run that spent the least money, among those that can be priced.

    A run using a model with no price on record is excluded rather than counted at
    zero. A catalogue gap is not a free model, and calling one "cheapest" is the
    single most expensive mistake this table could invite -- it recommends the one
    row nobody has costed.

    Args:
        trials: The rows to choose from.

    Returns:
        The cheapest priced run, or ``None`` if none finished with a price. Ties go
        to the first, which is the order the caller asked for them in.
    """
    finished = [trial for trial in trials if trial.ok and trial.fully_priced]
    return min(finished, key=lambda trial: trial.cost_usd, default=None)


def cost_label(trial: Trial) -> str:
    """What one row's cost column should say, with its basis attached.

    Three cases, and they are genuinely different facts about the run rather than
    three ways of saying "about this much":

    - every call priced from a vendor rate card -- the figure, plain;
    - some call priced only by reasoning from a neighbouring model -- the figure
      with ``est.`` on it, so nobody mistakes it for spend;
    - a model neither table lists -- ``unpriced``, and no number at all, because a
      blank a reader notices beats a zero they do not.

    Args:
        trial: The row to label.

    Returns:
        The cell text.

    Examples:
        >>> cost_label(Trial("openai/gpt-4o", cost_usd=0.0123, fully_priced=True))
        '0.012300'
        >>> cost_label(Trial("z-ai/glm-5.2", fully_priced=False, estimated_cost_usd=0.0045))
        '0.004500 est.'
        >>> cost_label(Trial("nobody/knows", fully_priced=False))
        'unpriced'
    """
    if trial.fully_priced:
        return f"{trial.cost_usd:.6f}"
    if trial.estimated_cost_usd is not None:
        return f"{trial.estimated_cost_usd:.6f} est."
    return "unpriced"


def unpriced(trials: Sequence[Trial]) -> tuple[str, ...]:
    """Models whose calls had no price on record.

    Args:
        trials: The rows to inspect.

    Returns:
        The slugs, once each, in the order first seen. De-duplicated because a
        repeat run lists one model three times and the reader is being told which
        entry to add to the price table, not how many rows it affected.
    """
    return tuple(
        dict.fromkeys(trial.slug for trial in trials if trial.ok and not trial.fully_priced)
    )


def fastest(trials: Sequence[Trial]) -> Trial | None:
    """The finished run that took the least wall-clock.

    Args:
        trials: The rows to choose from.

    Returns:
        The fastest, or ``None`` if nothing finished.
    """
    finished = [trial for trial in trials if trial.ok]
    return min(finished, key=lambda trial: trial.seconds, default=None)
