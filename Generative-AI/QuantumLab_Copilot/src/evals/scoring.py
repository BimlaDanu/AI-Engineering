"""Grading one run against one case, and the scorecard that adds them up.

Kept apart from :mod:`src.evals.run` so that grading is a pure function of a case
and an answer. That is what makes the grader itself testable, and it is the same
separation the rest of the project uses: deciding is one job, and doing the I/O
around it is another.

**No model grades anything here.** Every check is a comparison -- a status against
an expected status, a node list against a forbidden node, a number against
arithmetic. An LLM judge would introduce exactly the unverifiable step the project
exists to avoid, and it would make the scorecard unreproducible: the same suite on
the same code would score differently on Tuesday.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.graph import Answer, executed_nodes
from src.evals.cases import TOLERANCE, Case


@dataclass(frozen=True, slots=True)
class Check:
    """One thing that had to be true.

    Attributes:
        claim: What was required, in words a failure can be read from.
        passed: Whether it held.
        detail: What actually happened. Filled in only when it did not hold --
            a passing check needs no explanation, and a scorecard full of them is
            unreadable.
    """

    claim: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CaseResult:
    """How one case went.

    Attributes:
        case: What was asked.
        checks: Every requirement, in the order they were applied.
        status: The status the run ended in.
        nodes: The nodes it went through.
        energy: The number it produced, if it produced one.
        retrieval_grounded: Whether the answer rested on a retrieved passage. Kept
            so the scorecard can say whether the index was reachable, which the
            model-call count does not cover -- see
            :attr:`Scorecard.grounded_cases`.
    """

    case: Case
    checks: tuple[Check, ...]
    status: str
    nodes: tuple[str, ...]
    energy: float | None
    retrieval_grounded: bool = False

    @property
    def passed(self) -> bool:
        """Whether every check held."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        """The checks that did not hold."""
        return tuple(check for check in self.checks if not check.passed)


def grade(case: Case, answer: Answer, follow_up: Answer | None = None) -> CaseResult:
    """Check one finished run against what the case required.

    Args:
        case: The expectation.
        answer: What the agent returned.
        follow_up: The second run, for a case that asked a follow-up question.

    Returns:
        The result, carrying one :class:`Check` per requirement the case stated.
        A case that states nothing is vacuously passed, which is why
        :func:`src.evals.run.validate` refuses to let one into the suite.
    """
    nodes = executed_nodes(answer)
    checks: list[Check] = []

    if case.expect_status:
        checks.append(
            Check(
                f"ends {case.expect_status}",
                answer.status == case.expect_status,
                f"ended {answer.status}",
            )
        )
    for node in case.expect_nodes:
        checks.append(Check(f"runs {node}", node in nodes, f"path was {' → '.join(nodes)}"))
    for node in case.forbid_nodes:
        checks.append(
            Check(f"does not run {node}", node not in nodes, f"path was {' → '.join(nodes)}")
        )
    if case.expect_verified:
        checks.append(
            Check(
                "number corroborated by two methods",
                answer.is_verified,
                "no independent method agreed",
            )
        )
    if case.expect_unverified_notice:
        # The other half of the same claim, and the half a fluent agent fails: an
        # unverified number is acceptable, and an unverified number that does not
        # admit it is not.
        said = any("unverified" in text.lower() for text in answer.caveats)
        checks.append(
            Check(
                "says the number was not corroborated",
                said and not answer.is_verified,
                "claimed verification" if answer.is_verified else "no caveat mentioned it",
            )
        )
    if case.expect_energy is not None:
        got = answer.energy
        checks.append(
            Check(
                f"energy is {case.expect_energy:.9f}",
                got is not None and abs(got - case.expect_energy) <= TOLERANCE,
                "no energy was produced" if got is None else f"got {got:.9f}",
            )
        )
    if case.expect_route:
        route = answer.routing.route if answer.routing is not None else "none"
        checks.append(
            Check(f"routed {case.expect_route}", route == case.expect_route, f"routed {route}")
        )
    if case.expect_shelves:
        picked = answer.routing.shelves if answer.routing is not None else ()
        # ("",) is how a case says "the router must pick no shelf", which is the
        # decision to search the whole library rather than the absence of one.
        wanted = () if case.expect_shelves == ("",) else case.expect_shelves
        checks.append(
            Check(
                f"searches {', '.join(wanted) if wanted else 'every knowledge base'}",
                picked == wanted,
                f"picked {', '.join(picked) if picked else 'none'}",
            )
        )
    if case.require_shelves:
        picked = answer.routing.shelves if answer.routing is not None else ()
        # Containment, not equality: the case names the shelf that must answer and
        # says nothing about a second one the router also thought relevant. See
        # Case.require_shelves.
        missing = [name for name in case.require_shelves if name not in picked]
        checks.append(
            Check(
                f"searches at least {', '.join(case.require_shelves)}",
                not missing,
                f"picked {', '.join(picked) if picked else 'none'}",
            )
        )
    if case.forbid_shelves:
        picked = answer.routing.shelves if answer.routing is not None else ()
        # The claim with a wrong answer but no single right one. See
        # Case.forbid_shelves.
        wrong = [name for name in case.forbid_shelves if name in picked]
        checks.append(
            Check(
                f"does not search {', '.join(case.forbid_shelves)}",
                not wrong,
                f"picked {', '.join(picked) if picked else 'none'}",
            )
        )
    if case.expect_followups:
        checks.append(
            Check(
                "offers a follow-up question",
                answer.followups.any,
                f"offered none ({answer.followups.explain()})",
            )
        )
    if case.forbid_followups:
        checks.append(
            Check(
                "offers no follow-up",
                not answer.followups.any,
                f"offered {len(answer.followups.suggestions)}",
            )
        )
    if case.follow_up:
        checks.extend(_follow_up_checks(case, follow_up))
    return CaseResult(
        case=case,
        checks=tuple(checks),
        status=answer.status,
        nodes=nodes,
        energy=answer.energy,
        retrieval_grounded=answer.retrieval is not None and answer.retrieval.grounded,
    )


def _follow_up_checks(case: Case, follow_up: Answer | None) -> list[Check]:
    """Check the second turn of a memory case.

    Args:
        case: The expectation.
        follow_up: The second run, or ``None`` if it was never made.

    Returns:
        The checks. A missing follow-up is one failed check rather than an
        exception: a runner that could not complete a case should produce a red
        row, not a stack trace that hides the twenty rows after it.
    """
    if follow_up is None:
        return [Check("the follow-up was asked", False, "no second run was made")]
    checks: list[Check] = []
    if case.follow_up_expect_status:
        checks.append(
            Check(
                f"follow-up ends {case.follow_up_expect_status}",
                follow_up.status == case.follow_up_expect_status,
                f"ended {follow_up.status}",
            )
        )
    if case.forbid_follow_up_route:
        route = follow_up.routing.route if follow_up.routing is not None else "none"
        checks.append(
            Check(
                f"follow-up is not routed {case.forbid_follow_up_route}",
                route != case.forbid_follow_up_route,
                f"was routed {route}",
            )
        )
    if case.expect_status == "refused":
        # The memory rule with teeth: a blocked first turn must leave nothing for
        # the second to recall.
        checks.append(
            Check(
                "the blocked turn was not recalled",
                not follow_up.recall.has_history,
                f"{len(follow_up.recall.turns)} turn(s) were recalled",
            )
        )
    return checks


@dataclass(frozen=True, slots=True)
class FamilyScore:
    """How one family of cases did.

    Attributes:
        family: The family name.
        passed: Cases in it that passed every check.
        total: Cases in it.
    """

    family: str
    passed: int
    total: int

    @property
    def rate(self) -> float:
        """The pass rate, between 0 and 1."""
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Every result, and the arithmetic over them.

    Attributes:
        results: One per case, in suite order.
        model: The chat model the run was configured with, or ``""`` when there was
            none.
        model_calls: How many model calls the whole suite actually made. Measured
            rather than inferred from the configuration, because those differ: a
            configured key whose gateway is unreachable produces a complete run on
            the deterministic path, and a scorecard that named the model anyway
            would be claiming the run tested something it did not.
        notes: Anything that qualifies the run -- a suite that skipped a
            dependency, a model that was unreachable. Printed above the table,
            because a scorecard whose caveats are underneath is a scorecard whose
            caveats are unread.
    """

    results: tuple[CaseResult, ...]
    model: str = ""
    model_calls: int = 0
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> int:
        """Cases that passed every check."""
        return sum(1 for result in self.results if result.passed)

    @property
    def total(self) -> int:
        """Cases run."""
        return len(self.results)

    @property
    def rate(self) -> float:
        """Overall pass rate, between 0 and 1."""
        return self.passed / self.total if self.total else 0.0

    @property
    def grounded_cases(self) -> int:
        """How many cases were answered from a retrieved passage.

        Returns:
            The count. Read alongside :attr:`model_calls`, which does not cover it:
            embedding a query is not a chat completion, so a run with zero model
            calls can still have searched a live index. Two runs that differ here
            tested different things and their scores are not comparable.
        """
        return sum(
            1
            for result in self.results
            if result.retrieval_grounded  # set by the runner from the answer itself
        )

    def by_family(self) -> tuple[FamilyScore, ...]:
        """Break the run down by what each family defends.

        Returns:
            One score per family, in the order the families first appear in the
            suite -- which is the order the suite was written to be read in.
        """
        order: list[str] = []
        counts: dict[str, list[int]] = {}
        for result in self.results:
            family = result.case.family
            if family not in counts:
                counts[family] = [0, 0]
                order.append(family)
            counts[family][1] += 1
            counts[family][0] += int(result.passed)
        return tuple(FamilyScore(name, counts[name][0], counts[name][1]) for name in order)

    def markdown(self) -> str:
        """Render the scorecard as a report.

        Returns:
            Markdown: the headline, the per-family table, the per-case table, and
            every failure spelled out underneath. Markdown rather than JSON
            because the audience is a person deciding whether to trust this
            application, and a machine-readable scorecard nobody reads has
            measured nothing.
        """
        lines = [
            "# Evaluation scorecard",
            "",
            f"**{self.passed} of {self.total} cases passed** ({self.rate:.0%}).",
            "",
            self.provenance(),
            "",
        ]
        for note in self.notes:
            lines.append(f"> {note}")
            lines.append("")

        lines += ["| What it defends | Passed |", "| --- | --- |"]
        for score in self.by_family():
            lines.append(f"| {score.family} | {score.passed}/{score.total} ({score.rate:.0%}) |")
        lines += ["", "| Case | Outcome | Path |", "| --- | --- | --- |"]
        for result in self.results:
            mark = "pass" if result.passed else "**FAIL**"
            lines.append(
                f"| {result.case.name} | {mark} ({result.status}) | {' → '.join(result.nodes)} |"
            )

        failed = [result for result in self.results if not result.passed]
        if failed:
            lines += ["", "## Failures", ""]
            for result in failed:
                lines.append(f"**{result.case.name}** — {result.case.why}")
                lines.append("")
                for check in result.failures:
                    lines.append(f"- expected {check.claim}; {check.detail}")
                lines.append("")
        else:
            lines += ["", "Every case passed.", ""]
        return "\n".join(lines)

    def provenance(self) -> str:
        """Say what actually answered the questions.

        Returns:
            One line naming the model and how many calls it took, or saying plainly
            that none was called. The second case is the whole test suite's normal
            mode and a supported way to run the application, so it is stated rather
            than left to be inferred from a missing line.
        """
        # The corpus half is stated separately, and it has to be. `model_calls`
        # counts chat completions, and retrieval does not make one -- it embeds the
        # query, through a different client that the meter never sees. So a run can
        # truthfully report "no model was called" while every retrieval case was
        # answered from a live index behind a real credential, which is a different
        # test from the one the line appears to describe.
        corpus = (
            f"The corpus index answered {self.grounded_cases} of the retrieval cases."
            if self.grounded_cases
            else "No passage was retrieved: the index was unreachable, so every "
            "retrieval case was scored on the refusal it produced."
        )
        if not self.model_calls:
            return (
                "No model was called: every answer here came from the deterministic "
                "path — regex screening, the keyword router, the exact solvers and "
                f"the assembled reply. {corpus}"
            )
        return (
            f"Model: `{self.model or 'unknown'}`, {self.model_calls} calls across "
            f"the suite. {corpus}"
        )

    def summary(self) -> str:
        """One line, for a log record."""
        return f"{self.passed}/{self.total} cases passed ({self.rate:.0%})"
