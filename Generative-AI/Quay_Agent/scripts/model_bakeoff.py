"""Ask several models the same question and print what changed and what did not.

The command-line half of the **Model bake-off** tab. The tab is where a visitor
meets the experiment; this is where it is reproduced, and reproduction is the point
-- a measurement that exists only inside a Streamlit session is a screenshot.

What it is for
--------------

The project claims that a language model formalises, plans and writes while
deterministic code computes and a sealed solver grades. That claim is falsifiable:
if it is true, swapping the model must leave the numbers where they are. So this
runs the same question once per model and checks three things in order.

1. Did every model read the same chain out of the question? The reading is a model
   call, so it is the one place a model can move a number -- and it moves all of
   them at once.
2. Given the same reading, did the computed baseline hold? Nothing touches it, so
   for one reading it must come back identical. A difference is an architecture bug.
3. Did the verdict hold? It is decided by rule from the numbers.

Only then does the efficiency table mean anything, and only then is it safe to
choose a model on price or latency.

    make bakeoff                                        # the three configured tiers
    uv run python -m scripts.model_bakeoff --models openai/gpt-4o,openai/gpt-4.1-mini
    uv run python -m scripts.model_bakeoff --question "..." --models a,b

**Not a test, deliberately.** It calls the live gateway once per model -- a full
campaign each -- so it costs tokens and depends on a model's judgement on the day.
Everything under ``tests`` stays offline and deterministic. The pure reductions this
prints are tested there; the running is done here, on purpose, by a person.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.agent.model_selection import DEFAULT_TIER_SLUGS
from src.evals.bakeoff import Trial, agreement, cheapest, compare, fastest, unpriced
from src.hardware.devices import LINEAR
from src.logging_setup import configure_logging

DEFAULT_QUESTION = "Is quantum hardware worth it for a 10-spin critical Ising chain?"
"""The question every model is asked when none is named.

A feasibility question rather than an explanation. It is the branch that computes,
so it is the branch where "the model swap did not move the number" is a claim with a
number behind it; comparing two explanations would compare prose to prose.
"""

SHOT_BUDGET = 1_000_000_000
"""The measurement budget each campaign gets.

The same generous figure the live check uses, and for the same reason: a budget that
merely sounds large refuses the first configuration, and every model then comes back
with "we could not afford to try", which compares nothing.
"""


def default_models() -> tuple[str, ...]:
    """The models compared when the caller names none.

    Returns:
        This deployment's own fast, standard and strong tiers, de-duplicated in
        that order. It makes the first comparison the one actually worth having --
        is the expensive tier earning its place? -- rather than an arbitrary three.
    """
    return tuple(dict.fromkeys(DEFAULT_TIER_SLUGS.values()))


def row(trial: Trial) -> str:
    """Render one trial as a line of a fixed-width table.

    Args:
        trial: The measured run.

    Returns:
        One line. The reading comes before the price, in the order the checks are
        meant to be read.
    """
    if not trial.ok:
        return f"{trial.slug:<34} {'DID NOT RUN':<26} {trial.failure[:50]}"
    baseline = (
        "--" if trial.baseline_energy_per_site is None else f"{trial.baseline_energy_per_site:.9f}"
    )
    return (
        f"{trial.slug:<34} "
        f"{trial.read_as:<26} "
        f"{baseline:>13}  "
        f"{trial.verdict:<12} "
        f"{trial.seconds:>6.1f}s "
        f"{trial.calls:>3} calls "
        f"{trial.tokens:>7,} tok "
        f"{'unpriced' if not trial.fully_priced else f'${trial.cost_usd:.6f}':>9} "
        f"{trial.words:>5}w "
        f"{trial.cited:>2} cited "
        f"{trial.hedge_rate:>5.2f} hedge/100w"
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments to parse. ``None`` reads the real command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.model_bakeoff",
        description="Ask several models the same question and compare what changed.",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="the question every model is asked; the same string for all of them",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="ask each model this many times. With one model and --repeat 3 this "
        "stops being a bake-off and becomes a stability check: does one model read "
        "the same sentence the same way twice? A bad reading that only happens "
        "sometimes is invisible from a single run and is worse than a consistent one",
    )
    parser.add_argument(
        "--models",
        default="",
        help="comma-separated slugs. Defaults to the three configured tiers, which "
        "is the comparison worth having first",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and print it.

    Args:
        argv: Command-line arguments. ``None`` reads the real ones.

    Returns:
        Zero when the numbers held across every model, one when something moved.
        Non-zero here is a real finding rather than a crash -- either the models
        read the question differently, or a number a model is not supposed to touch
        moved -- and both are things a person should be told about by an exit code
        rather than by scrolling.
    """
    configure_logging()
    arguments = parse_arguments(argv)
    slugs = (
        tuple(slug.strip() for slug in arguments.models.split(",") if slug.strip())
        or default_models()
    )
    repeats = max(arguments.repeat, 1)
    # Repeated slug by slug rather than round by round, so a model's own runs sit
    # together in the table: the question a repeat answers is "did *this* model hold
    # steady", and interleaving would make that the one comparison a reader has to
    # do by eye.
    asked = tuple(slug for slug in slugs for _ in range(repeats))
    print(f"question: {arguments.question}")
    print(f"models:   {', '.join(slugs)}")
    if repeats > 1:
        print(f"repeats:  {repeats} per model ({len(asked)} campaigns)")
    print()

    trials = compare(arguments.question, asked, shot_budget=SHOT_BUDGET, device=LINEAR)
    for trial in trials:
        print(row(trial))

    found = agreement(trials)
    print(f"\n{found.headline()}")
    if repeats > 1:
        for slug in slugs:
            own = [trial for trial in trials if trial.slug == slug]
            steady = agreement(own)
            held = "held steady" if steady.read_alike else "READ IT DIFFERENTLY EACH TIME"
            print(f"  {slug}: {held} across {repeats} runs -- {'; '.join(steady.readings)}")
    missing = unpriced(trials)
    if missing:
        print(
            f"no price on record for: {', '.join(missing)} -- excluded from the "
            "cheapest line below, because a catalogue gap is not a free model"
        )
    best_price = cheapest(trials)
    best_speed = fastest(trials)
    if best_price is not None:
        print(f"cheapest that finished: {best_price.slug} at ${best_price.cost_usd:.6f}")
    if best_speed is not None:
        print(f"fastest that finished:  {best_speed.slug} at {best_speed.seconds:.1f}s")
    return 0 if (found.read_alike and found.numbers_held and found.verdict_held) else 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
