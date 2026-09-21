r"""Deciding go, no, or conditional -- by rule, not by persuasion.

The verdict is the one part of a campaign that must not be left to a language
model, and the reason is the failure this project sets out to measure. A model
asked "was this worth it?" after watching its own circuit converge beautifully
will tend to say yes, and it will say yes slightly more often when the question
arrived phrased as a vendor's pitch. Both are sycophancy, and neither is fixable
by asking the model to be careful.

So the call itself is arithmetic. Each rule below is a screen with a single
question and an objective answer -- was a classical baseline run at all, is the
problem one with a closed-form solution, did the quantum energy actually beat the
classical one by more than the classical error bar -- and the first screen that
fires decides. What a language model contributes is the *wording*: the summary a
non-specialist reads. It never contributes the call, and the call does not move
when the question is rephrased, because none of these screens can see the phrasing.

A screen answers about the problem, and the reader has often asked about a method.
Every screen below asks something about the chain -- was a baseline run, does a closed
form exist, did the circuit beat it -- so none can mention VQE, QAOA or imaginary time,
and a question listing all three comes back with a verdict naming none of them.
:func:`_which_method_won` supplies the missing sentence, prepended in :func:`judge`
from the race the campaign ran. It is arithmetic like everything else here and no
screen can see it: it reports what was measured and does not move the call.

The order of the screens is itself a claim. They run from "no result here can be
trusted" through "no result here can matter" to "the results were compared and here is
the comparison". A campaign whose simulator returned an impossible energy must not be
judged on how good that energy looked, and a problem an ordinary computer solves in
closed form must not earn a "go" for converging nicely. Putting those cases first is
what stops a good-looking run from outvoting them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from src.agent.state import BaselineResult, CampaignState, RunRecord, Verdict, best_run

ADVANTAGE_MARGIN = 1.0
"""How many classical error bars the quantum energy must win by to count as a win.

Set at one standard deviation, which is a deliberately generous bar and is stated
so that nobody mistakes it for a strict one. The classical baseline reports a
statistical uncertainty, and two numbers differing by less than that uncertainty
are the same number; a comparison that ignored the error bar would manufacture a
result out of Monte Carlo noise, which is the easiest way to fake an advantage
and the hardest to notice afterwards.
"""


TIE_MARGIN = 1e-6
"""Energy-per-spin difference below which two methods are reported as having tied.

All three raced methods drive the *same* circuit, so the energy they can reach is a
property of the ansatz rather than of the optimiser and a tie is the expected outcome.
Naming a winner on a difference of :math:`10^{-9}` would present the noise floor of
three different optimisers as a result, and it would present the one genuine finding --
that the circuit, not the method, sets the accuracy -- as though it had not happened.
"""


@dataclass(frozen=True, slots=True)
class Screen:
    """One objective question, asked of the finished campaign.

    Attributes:
        name: Short identifier, recorded so a report can say which rule decided.
        question: What the screen asks, in plain words.
        check: Returns a verdict when the screen fires, or ``None`` to pass the
            campaign down to the next screen.
    """

    name: str
    question: str
    check: Callable[[CampaignState], Verdict | None]


def _named_a_chain_too_long(state: CampaignState) -> Verdict | None:
    """The question named a chain, and it is longer than anything here can price.

    Ahead of :func:`_no_model` because both fire on a campaign with no model and
    only one of them is true. "The request could not be turned into a specific
    physics problem" is the wrong sentence to show somebody who named a perfectly
    specific problem and was told nothing about why it was declined -- and it was
    the sentence they got, because the two cases arrive here looking identical.

    Args:
        state: The finished campaign.

    Returns:
        A verdict naming the length asked for and the ceiling, or ``None`` when this
        is not what happened.
    """
    if state["model"] is not None:
        return None
    named = next(
        (belief for belief in state["beliefs"] if "past every method" in belief.claim),
        None,
    )
    if named is None:
        return None
    return Verdict(
        call="no",
        summary=(
            f"{named.claim.capitalize()}. So the honest answer is not a verdict about "
            "this chain but a statement about the tools: nothing here can size it, and "
            "a verdict computed on a shorter chain would be a confident answer to a "
            "different question. " + " ".join(f"{item.capitalize()}." for item in named.support)
        ),
        crossover_condition=(
            "ask about a chain this can actually price -- up to sixty-four elements -- "
            "or read the scaling argument, which is what a question about hundreds of "
            "elements is really asking and does not need a run to answer"
        ),
        confidence="high",
    )


def _no_model(state: CampaignState) -> Verdict | None:
    """Nothing was formalised, so there is no question to answer.

    Args:
        state: The finished campaign.

    Returns:
        A verdict if the campaign never produced a model, otherwise ``None``.
    """
    if state["model"] is not None:
        return None
    return Verdict(
        call="no",
        summary=(
            "The request could not be turned into a specific physics problem, so "
            "there is nothing here to assess. No quantum or classical work was "
            "worth starting until that is settled."
        ),
        crossover_condition=(
            "restate the problem with the number of interacting elements and the "
            "strength of the competing effect, and it can be assessed"
        ),
        confidence="high",
    )


def _impossible_energy(state: CampaignState) -> Verdict | None:
    """A run reported an energy no state can have, so the machinery is wrong.

    This outranks every other screen. The lower an erroneous energy is, the better
    it looks and the more likely it is to be quoted, so a campaign that produced
    one must not be judged on its numbers at all.

    Args:
        state: The finished campaign.

    Returns:
        A verdict if any run fell below the arithmetic floor, otherwise ``None``.
    """
    broken = [run for run in state["runs"] if run.diagnosis.signal == "below_variational_bound"]
    if not broken:
        return None
    return Verdict(
        call="no",
        summary=(
            f"{len(broken)} of {len(state['runs'])} runs reported an energy below the "
            "level that is arithmetically possible for this problem. That is a fault "
            "in the calculation, not a discovery, and no conclusion can be drawn from "
            "this campaign until it is found."
        ),
        crossover_condition=(
            "fix the fault and re-run; the campaign can be assessed once every run "
            "reports an energy above the arithmetic floor"
        ),
        confidence="high",
    )


BASELINE_REFUSAL_MARK = "no classical number to compare against"
"""Phrase by which a *refusal* to run the baseline is recognised downstream.

:func:`src.agent.graph.run_classical_baseline` records the refusal as a belief rather
than as
a result, so that no number reaches a reader who would reasonably compare against
it. :func:`_no_classical_comparison` then has to tell two cases apart -- the
baseline was skipped, or the baseline **cannot run on this problem** -- and a
constant is how the two modules agree on which is which without the verdict
importing the physics layer to ask.
"""


def _no_classical_comparison(state: CampaignState) -> Verdict | None:
    """No classical baseline was produced, so no claim of advantage can be made.

    A quantum result with nothing to compare it against is not a weak result -- it
    is not a result. This screen is what makes the classical arm structurally
    unskippable: even a campaign that ran perfect circuits cannot get past it.

    Two cases arrive here and they need different crossover conditions. *"Run the
    classical method on the same problem, then the two can be compared"* is sound
    advice for a campaign that skipped the step and impossible advice for a problem
    the method cannot represent. It is the defect :func:`_named_a_chain_too_long`
    handles one layer down: the right sentence and the wrong sentence arrive looking
    identical.

    Args:
        state: The finished campaign.

    Returns:
        A verdict if the baseline is missing, otherwise ``None``.
    """
    if state["classical"] is not None:
        return None
    refused = next(
        (belief for belief in state["beliefs"] if BASELINE_REFUSAL_MARK in belief.claim),
        None,
    )
    if refused is not None:
        return Verdict(
            call="no",
            summary=(
                f"**{refused.claim}** A quantum number on its own cannot show that a "
                "quantum computer was worth using, so this campaign cannot reach a "
                "verdict on that question -- and it is worth being clear that this is "
                "not the same thing as a quantum computer losing. " + " ".join(refused.support)
            ),
            crossover_condition=(
                "a classical method that covers this problem. Until one exists the "
                "honest answer is not that a quantum computer loses here, but that "
                "nobody has measured what it would have to beat"
            ),
            confidence="high",
        )
    return Verdict(
        call="no",
        summary=(
            "No classical result was produced, so there is nothing to compare the "
            "quantum result against. A quantum number on its own cannot show that a "
            "quantum computer was worth using."
        ),
        crossover_condition=(
            "run the classical method on the same problem, then the two can be compared"
        ),
        confidence="high",
    )


def _closed_form_available(state: CampaignState) -> Verdict | None:
    r"""The problem has a closed-form answer, so no computer of any kind is needed.

    The screen the whole project turns on, and the one an enthusiastic assessment
    skips. With no longitudinal field this chain reduces to free particles and its
    energy is a short sum, computable in microseconds at any size. Nothing a
    quantum computer does can improve on that -- so however well the circuit ran,
    the answer to "is a quantum computer worth using here?" is no.

    It fires before any comparison of numbers, because the numbers are irrelevant:
    a quantum result that beat the sampled classical baseline would still be beaten
    by a formula.

    Args:
        state: The finished campaign.

    Returns:
        A verdict if the model is exactly solvable, otherwise ``None``.
    """
    model = state["model"]
    if model is None or not model.is_exactly_solvable:
        return None
    # What *this* campaign found, ahead of the standing reason. The reason is the same
    # sentence for every exactly solvable chain, because the fact it states is the same
    # fact -- but a reader meeting it under three different questions with no numbers in
    # front of it cannot tell that three different campaigns ran. The evidence is what
    # differs between them, so the evidence goes first.
    return Verdict(
        call="no",
        summary=(
            f"{_what_this_run_found(state)}"
            f"This problem ({model.label()}) has an exact solution in closed form: "
            "with no field along the coupling direction, the chain reduces to "
            "independent particles and its lowest energy is a short sum that an "
            "ordinary computer evaluates instantly, at any size. A quantum computer "
            "cannot "
            "improve on an instant exact answer, so the circuit results below are "
            "evidence that the method works, not evidence that it is needed."
        ),
        crossover_condition=(
            "add a field along the coupling direction, or any other term that breaks "
            "the reduction to independent particles; the closed form is then gone and "
            "the question becomes worth asking"
        ),
        confidence="high",
    )


def _what_this_run_found(state: CampaignState) -> str:
    """Open a verdict with the numbers this campaign actually produced.

    The closed-form screen states a fact about the *problem*, so its wording is
    identical for every exactly solvable chain -- and a reader who meets that paragraph
    under three different questions, with nothing specific in front of it, reasonably
    concludes the application is printing a canned reply. It is not: three campaigns
    ran, priced different circuits and reached different energies. This says so, in one
    sentence, before the standing reason.

    Args:
        state: The finished campaign.

    Returns:
        A sentence ending in a space, or the empty string when nothing ran.
    """
    quantum = best_run(state)
    if quantum is None:
        return ""
    # Agreement, because a single run is the common case on a small chain and
    # "1 circuits were run here and 1 were refused" is the first thing a reader sees
    # under the verdict. A sentence that cannot count to one is not read as a slip in
    # the prose; it is read as doubt about the arithmetic beside it.
    ran = len(state["runs"])
    reached = f"**{ran} circuit{'' if ran == 1 else 's'} {'was' if ran == 1 else 'were'} run here"
    refused = len(state["ruled_out"])
    if refused:
        reached += (
            f" and {refused} {'was' if refused == 1 else 'were'} "
            "refused on arithmetic before running"
        )
    reached += (
        f"; the best, {quantum.label} at depth {quantum.depth}, reached "
        f"{quantum.energy_per_site:.6f} per spin"
    )
    classical = state["classical"]
    if classical is not None:
        reached += f" against the ordinary computer's {classical.energy_per_site:.6f}"
    return reached + ".** "


def _which_method_won(state: CampaignState) -> str:
    """Answer the question the reader actually asked, when they asked which method.

    *VQE, QAOA or imaginary time for a 10-spin critical chain?* names three methods and
    asks to be told one. The screens below cannot tell them apart -- every one of them
    asks about the *problem*, not about the method -- so the honest verdict on whether a
    quantum computer is worth it here arrived without the word VQE in it, and read as a
    canned reply because from the reader's side it was one: nothing in it could have
    differed had they asked about a different method.

    This is the missing sentence, and it is arithmetic like the rest of the module. The
    three methods were run on this campaign's own chain in
    :func:`src.agent.graph.converge_methods`; who reached lowest and who got there in
    fewest steps are two facts about that run, not a judgement about it.

    Both numbers are reported, because lowest energy and fewest steps are different
    questions with different winners, and quoting only the first turns "which
    converges fastest" into an answer about something else.

    Args:
        state: The finished campaign.

    Returns:
        A sentence ending in a space, or the empty string when no race was run.
    """
    race = state["race"]
    if race is None:
        return ""
    winner = race.best()
    if winner is None:
        return ""

    quickest = min(race.runs, key=lambda run: run.n_steps)
    spread = max(run.energy_per_site for run in race.runs) - min(
        run.energy_per_site for run in race.runs
    )
    by_steps = ", ".join(
        f"{run.method} in {run.n_steps}" for run in sorted(race.runs, key=lambda run: run.n_steps)
    )

    if spread < TIE_MARGIN:
        # The usual outcome at this depth, and the one worth stating as an outcome
        # rather than resolving into a winner by the twelfth decimal place. All three
        # drive the same circuit, so where they *land* is a property of the circuit and
        # agreement is the expected result; what differs is the route, and saying
        # "QAOA won by 1e-9" would hide the only real finding behind a false one.
        return (
            f"**All {len(race.runs)} methods reached the same energy on this chain -- "
            f"{winner.energy_per_site:.6f} per spin -- so the choice between them is not "
            f"about accuracy but about how long they take: {by_steps} steps.** They share "
            "the same circuit, so where they stop is a property of the circuit and "
            "agreement is the expected result; the step counts are the difference. That "
            "is from an exact simulation with no noise in it, so it says which method "
            "optimises better and nothing about which survives a real device. "
        )

    others = ", ".join(
        f"{run.method}'s {run.energy_per_site:.6f} in {run.n_steps}"
        for run in race.runs
        if run.method != winner.method
    )
    said = (
        f"**Of the {len(race.runs)} methods raced on this chain, {winner.method} got "
        f"closest: {winner.energy_per_site:.6f} per spin in {winner.n_steps} steps, "
        f"against {others}.** "
    )
    if quickest.method != winner.method:
        # Worth its own sentence rather than a parenthesis. A method that settles in a
        # third of the steps and stops slightly higher is the better choice under a shot
        # budget and the worse one under none, and a reader deciding between them needs
        # to see that the two rankings disagree.
        said += (
            f"{quickest.method} settled fastest -- {quickest.n_steps} steps against "
            f"{winner.method}'s {winner.n_steps} -- but stopped higher, at "
            f"{quickest.energy_per_site:.6f}. "
        )
    return said + (
        "That ranking is from an exact simulation with no noise in it, so it says which "
        "method optimises better and nothing about which survives a real device. "
    )


def _no_usable_quantum_run(state: CampaignState) -> Verdict | None:
    """Nothing ran on the quantum side, so there is nothing to weigh.

    Args:
        state: The finished campaign.

    Returns:
        A verdict if no trustworthy run exists, otherwise ``None``.
    """
    if best_run(state) is not None:
        return None
    rejected = len(state["ruled_out"])
    return Verdict(
        call="no",
        summary=(
            "No quantum configuration completed within the budget"
            + (
                f", and {rejected} were rejected before running as too expensive"
                if rejected
                else ""
            )
            + ". There is no quantum result to compare, so no case for using one."
        ),
        crossover_condition=(
            "raise the measurement budget or the allowed circuit depth until at least "
            "one configuration can run, then compare it against the classical result"
        ),
        confidence="high",
    )


def _baseline_not_trustworthy(state: CampaignState) -> Verdict | None:
    """The classical number exists but cannot carry a comparison, so none is made.

    This is the third verdict, and the reason the project needs one. Every other
    screen here answers yes or no about the campaign; this one answers neither. Both
    arms produced a number, and the classical one came from a regime where the
    quantity the comparison divides by -- its error bar -- is known to be
    under-reported. See :mod:`src.physics.classical.regime` for what causes that.

    Reporting a lead on such a baseline would be the project's cardinal sin dressed
    as arithmetic. :func:`_compare` measures a quantum lead in units of the classical
    uncertainty, so an under-reported uncertainty is a *lowered bar*, and it lowers
    it in the one direction that flatters the quantum arm.

    It does not fire when the classical arm won anyway, and that asymmetry is the
    substance of the screen. A degraded baseline fails in a known direction: looser
    and noisier, so easier to beat. A degraded baseline that still beats the circuit
    has therefore settled the question, since the true classical answer is lower than
    the one measured and the gap only widens; "cannot tell" there would be false
    modesty. Only a lead *over* a degraded baseline is unsupportable, so only that
    case is refused.

    Placed last, immediately before :func:`_compare`, because it is reached exactly
    when a comparison was about to be made and says that it cannot be.

    Args:
        state: The finished campaign.

    Returns:
        The third verdict when a lead would rest on an untrustworthy baseline,
        otherwise ``None``.
    """
    classical = state["classical"]
    quantum = best_run(state)
    if classical is None or quantum is None or classical.is_trustworthy:
        return None
    if _classical_arm_won_anyway(classical, quantum):
        return None

    return Verdict(
        call="conditional",
        summary=(
            "**The classical comparison could not be made reliably here, so this is "
            "not a verdict about whether a quantum computer helps -- it is a statement "
            "that the comparison is missing.** "
            + _why_the_comparison_is_missing(classical, quantum)
            + " Whatever the circuit reached, this campaign cannot say it beat an "
            "ordinary computer."
        ),
        crossover_condition=(
            "a classical baseline whose uncertainty can be believed on this problem -- "
            "either a sampler that moves more than one spin at a time, so that "
            "successive measurements genuinely decorrelate, or an uncertainty estimated "
            "from the measured correlation between them rather than assumed away. Until "
            "then the honest answer here is that nobody knows"
        ),
        confidence="low",
    )


def _classical_arm_won_anyway(classical: BaselineResult, quantum: RunRecord) -> bool:
    """Whether the baseline beat the circuit decisively despite not being trustworthy.

    The asymmetry described on :func:`_baseline_not_trustworthy`. A degraded baseline
    is looser and noisier than the truth, so believing its bar only *widens* a gap it
    already won -- which makes that conclusion safe to state.

    An **unusable** bar is a different case and returns ``False`` here whatever the
    energies are: with no estimate of the spread there is no margin to compare
    against, so there is no decisive win to let through.

    Args:
        classical: The baseline, known to be untrustworthy.
        quantum: The best quantum run.

    Returns:
        True when the comparison may be left to :func:`_compare`.
    """
    if not classical.uncertainty_is_usable:
        return False
    margin = classical.energy_per_site - quantum.energy_per_site
    return margin < -ADVANTAGE_MARGIN * classical.energy_error


def _why_the_comparison_is_missing(classical: BaselineResult, quantum: RunRecord) -> str:
    """State which of the two failures happened, and what it costs.

    Two quite different sentences, and printing the wrong one would misinform. An
    under-reported bar is a real number that is too small; an unusable one is not a
    number at all, and calling that "smaller than the real one" would invite a reader
    to imagine it as merely optimistic.

    Args:
        classical: The baseline, known to be untrustworthy.
        quantum: The best quantum run.

    Returns:
        A sentence or two, ending without a trailing space.
    """
    reached = (
        f"The ordinary-computer method reached {classical.energy_per_site:.6f} per spin "
        f"against the circuit's {quantum.energy_per_site:.6f}."
    )
    if not classical.uncertainty_is_usable:
        return (
            f"{reached} But it carries **no usable uncertainty at all** -- it was "
            "measured in a way that gives no estimate of its own spread, which is a "
            "different thing from having been measured precisely. A quantum lead is "
            "counted in multiples of that spread, so with no spread there is no lead "
            "to count."
        )
    caveat = f" {classical.caveat}" if classical.caveat else ""
    return (
        f"{reached} Its stated uncertainty is ± {classical.energy_error:.6f}, and on "
        f"this problem that is known to be smaller than the real one.{caveat} A quantum "
        "lead is counted in multiples of exactly that uncertainty, so understating it "
        "understates what a lead has to clear -- in the one direction that flatters "
        "the quantum result."
    )


def _compare(state: CampaignState) -> Verdict:
    """Weigh the best quantum run against the classical baseline.

    Reached only when every earlier screen passed, which means the problem is one
    where the question is genuinely open, both arms produced a number, and the
    numbers can be compared. Energies are compared per spin, because the total
    energy grows with the chain and would make the two arms incomparable if they
    ever ran at different lengths.

    Args:
        state: The finished campaign.

    Returns:
        The verdict. Never ``None`` -- this is the last word.
    """
    quantum = best_run(state)
    classical = state["classical"]
    # Both are guaranteed present: the screens above return before reaching here
    # if either is missing.
    assert quantum is not None and classical is not None

    margin = classical.energy_per_site - quantum.energy_per_site
    threshold = ADVANTAGE_MARGIN * classical.energy_error
    spent = state["shots"].spent

    if margin > threshold:
        return Verdict(
            call="conditional",
            summary=(
                f"The quantum circuit reached {quantum.energy_per_site:.6f} per spin "
                f"against the classical method's {classical.energy_per_site:.6f} "
                f"± {classical.energy_error:.6f}, a lead of {margin:.6f} that is larger "
                f"than the classical uncertainty. That is a real lead in simulation, "
                f"and simulation is where it was measured: these circuits were run "
                f"without hardware noise, on {spent:,} measurements. The case is worth "
                "pursuing; it is not yet a case for buying time on a device."
            ),
            crossover_condition=(
                f"the lead survives on hardware -- it needs a device whose two-qubit "
                f"error rate is small enough that a circuit of two-qubit depth "
                f"{quantum.two_qubit_depth} still returns a usable state, which is the "
                "measurement that has not been made here"
            ),
            confidence="medium",
        )

    if margin < -threshold:
        # The case that used to be misreported. A signed margin compared against a
        # positive threshold puts a decisive loss and a genuine tie in the same
        # branch, so a quantum arm beaten by sixteen error bars was described as
        # having "reached the same place". A loss is a cleaner result than a tie and
        # deserves to be stated as one.
        return Verdict(
            call="no",
            summary=(
                f"The classical method won, and not narrowly. It reached "
                f"{classical.energy_per_site:.6f} per spin against the quantum "
                f"circuit's {quantum.energy_per_site:.6f} -- lower is better, and the "
                f"gap of {abs(margin):.6f} is "
                f"{abs(margin) / max(classical.energy_error, 1e-12):.0f} times the "
                f"classical uncertainty of ± {classical.energy_error:.6f}. The circuit "
                f"was not close, on {spent:,} measurements and without any hardware "
                "noise to blame."
            ),
            crossover_condition=(
                f"a deeper circuit. At {quantum.depth} layer"
                f"{'s' if quantum.depth != 1 else ''} the ansatz cannot yet represent "
                "this state; the gap closes with depth, and the question is whether it "
                "closes before the depth stops fitting on a device"
            ),
            confidence="high",
        )

    return Verdict(
        call="no",
        summary=(
            f"The two methods agree. The quantum circuit reached "
            f"{quantum.energy_per_site:.6f} per spin against the classical method's "
            f"{classical.energy_per_site:.6f} ± {classical.energy_error:.6f}, and the "
            f"gap of {abs(margin):.6f} is inside the classical error bar -- so on this "
            "evidence they reached the same place, and the quantum one cost more to "
            "get there."
        ),
        crossover_condition=(
            "a quantum result that beats the classical one by more than the classical "
            "error bar, at a chain length where the classical method is genuinely "
            "struggling rather than comfortable"
        ),
        confidence="medium",
    )


SCREENS: tuple[Screen, ...] = (
    # Ahead of `no_model`, which it is a special case of. Both fire when nothing was
    # formalised; this one is reached first because it can say *why*, and "the request
    # could not be turned into a specific physics problem" is a false statement about
    # a question that named 200 magnets.
    Screen(
        name="chain_too_long",
        question="Did the question name a chain longer than anything here can price?",
        check=_named_a_chain_too_long,
    ),
    Screen(
        name="no_model",
        question="Was the request turned into a specific problem at all?",
        check=_no_model,
    ),
    Screen(
        name="impossible_energy",
        question="Did any run report an energy that cannot exist?",
        check=_impossible_energy,
    ),
    Screen(
        name="no_classical_comparison",
        question="Was a classical baseline actually run?",
        check=_no_classical_comparison,
    ),
    Screen(
        name="closed_form_available",
        question="Does this problem already have an exact answer in closed form?",
        check=_closed_form_available,
    ),
    Screen(
        name="no_usable_quantum_run",
        question="Did any quantum configuration finish?",
        check=_no_usable_quantum_run,
    ),
    # Last, and deliberately so: it fires exactly where a comparison was about to
    # be made and says that this one cannot be. Every screen above it decides on
    # something other than the two energies, so none of them is affected by how
    # far the classical number can be trusted.
    Screen(
        name="baseline_not_trustworthy",
        question="Would a quantum lead here rest on a baseline that cannot carry one?",
        check=_baseline_not_trustworthy,
    ),
)
"""The screens, in the order they are applied. First to fire decides the verdict.

An explicit tuple rather than a decorator registry, and for the same reason the
tool list is explicit: the order is load-bearing, and a registry would let a new
screen silently insert itself in the wrong place.
"""


def judge(state: CampaignState) -> tuple[Verdict, str]:
    """Reach the verdict, and say which rule reached it.

    Args:
        state: The finished campaign.

    Returns:
        The verdict, and the name of the screen that decided it. The name is
        returned rather than logged internally so that the caller can put it in the
        trace and in the campaign's own notes: "no, because the problem has a
        closed-form answer" and "no, because the circuit lost to the baseline" are
        two very different outcomes that the call alone cannot tell apart.
    """
    for screen in SCREENS:
        found = screen.check(state)
        if found is not None:
            verdict, decided_by = found, screen.name
            break
    else:
        verdict, decided_by = _compare(state), "comparison"

    # Prepended here rather than inside one screen, because which screen fires is a
    # fact about the problem and whether three methods were named is a fact about the
    # question -- and a reader who asked "which of these three?" is owed the answer
    # whichever screen turns out to decide the feasibility call. The call itself is
    # untouched: no screen can see this string and none of them is passed the race.
    opening = _which_method_won(state)
    if opening:
        verdict = replace(verdict, summary=opening + verdict.summary)
    return verdict, decided_by
