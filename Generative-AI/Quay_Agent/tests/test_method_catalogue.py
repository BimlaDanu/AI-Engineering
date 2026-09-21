"""Tests for the agent-facing method catalogue.

Two things are under test here and they are not the same thing. The first is
bookkeeping -- names, applicability, cost facts, the prose that reaches a prompt
-- which is what the catalogue is *for*. The second is the seal: that the entries
describing an exact solver carry no way to call one. The second is the reason
this module exists separately from ``tests/test_registry.py``, and it is the
only test in the suite whose failure would invalidate every result the project
produces rather than merely reporting a bug.

Nothing here runs a solver on a chain longer than eight sites.
"""

from __future__ import annotations

import pytest

from src.physics.method_catalogue import (
    COST_ORDER,
    EXACT_DIAGONALISATION,
    PFEUTY_EXACT,
    VARIATIONAL_IMAGINARY_TIME,
    VARIATIONAL_QUANTUM_EIGENSOLVER,
    MethodFacts,
    MethodSurvey,
    Rejection,
    agent_methods,
    all_methods,
    free_fermion_unsupported_reason,
    get_method,
    method_names,
    survey,
    variational_imaginary_time_unsupported_reason,
)
from src.physics.model import MAX_SITES_SPARSE, TFIMSpec

SIZES = [2, 4, 6, 8]


# --------------------------------------------------------------------------
# The seal — the agent may read the menu and may not read the answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [PFEUTY_EXACT, EXACT_DIAGONALISATION])
def test_an_exact_method_is_described_but_not_callable(name: str) -> None:
    facts = get_method(name)
    assert facts.availability == "grader_only"
    assert facts.ground_state_energy is None
    assert not facts.runnable_by_agent


@pytest.mark.parametrize("name", [PFEUTY_EXACT, EXACT_DIAGONALISATION])
def test_running_a_grader_only_method_from_here_is_a_permission_error(name: str) -> None:
    # Deliberately not KeyError or TypeError. Reaching this line means something
    # tried to evaluate the number it is supposed to be graded against, and the
    # exception has to say that rather than look like a lookup slip.
    with pytest.raises(PermissionError, match=name):
        get_method(name).solve(TFIMSpec(n_sites=4))


def test_the_agent_may_run_both_variational_routes_and_no_exact_one() -> None:
    # The two the agent may run are the two that return a bound it can check for
    # itself. Every exact method is grader-only, and that is the wall.
    assert [facts.name for facts in agent_methods()] == [
        VARIATIONAL_IMAGINARY_TIME,
        VARIATIONAL_QUANTUM_EIGENSOLVER,
    ]
    assert all(facts.accuracy != "exact" for facts in agent_methods())


def test_an_exact_method_still_applies_even_though_it_cannot_be_run() -> None:
    # Applicability is a fact about the problem; permission is a fact about the
    # caller. Conflating them would leave the agent unable to say that an exact
    # answer exists for the chain it is working on, which is exactly the thing
    # an honest feasibility report has to state.
    facts = get_method(PFEUTY_EXACT)
    assert facts.applies_to(TFIMSpec(n_sites=4))
    assert not facts.runnable_by_agent


def test_the_survey_separates_what_applies_from_what_may_be_run() -> None:
    result = survey(TFIMSpec(n_sites=4))
    assert len(result.applicable) == len(all_methods())
    assert [facts.name for facts in result.runnable()] == [
        VARIATIONAL_IMAGINARY_TIME,
        VARIATIONAL_QUANTUM_EIGENSOLVER,
    ]


def test_the_description_marks_the_methods_the_agent_cannot_run() -> None:
    text = survey(TFIMSpec(n_sites=4)).describe()
    assert "grader only" in text


# --------------------------------------------------------------------------
# Catalogue integrity
# --------------------------------------------------------------------------


def test_the_catalogue_is_not_empty() -> None:
    assert len(all_methods()) >= 3


def test_catalogue_names_are_unique() -> None:
    names = method_names()
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("facts", all_methods(), ids=lambda facts: facts.name)
def test_every_method_declares_its_cost_and_accuracy_class(facts: MethodFacts) -> None:
    assert facts.cost in COST_ORDER
    assert facts.accuracy in {"exact", "variational_bound", "uncontrolled"}
    assert facts.availability in {"agent", "grader_only"}


@pytest.mark.parametrize("facts", all_methods(), ids=lambda facts: facts.name)
def test_every_method_explains_itself_in_prose(facts: MethodFacts) -> None:
    # These strings end up in a prompt and in the justification shown to the
    # user, so an empty or placeholder one is a real defect.
    assert len(facts.summary) > 40
    assert len(facts.when_to_use) > 40


def test_the_catalogue_cannot_be_extended_by_a_caller() -> None:
    assert isinstance(all_methods(), tuple)


def test_cost_order_ranks_cheapest_first() -> None:
    assert (
        COST_ORDER["constant"]
        < COST_ORDER["linear"]
        < COST_ORDER["polynomial"]
        < COST_ORDER["exponential"]
    )


def test_the_baseline_is_cheaper_than_exact_diagonalisation_and_dearer_than_the_closed_form() -> (
    None
):
    # The ordering the agent sorts by, asserted rather than assumed: the whole
    # point of the baseline is that it is the affordable route at sizes where
    # the 2**L matrix is not, and that it is still not free.
    baseline = COST_ORDER[get_method(VARIATIONAL_IMAGINARY_TIME).cost]
    assert COST_ORDER[get_method(PFEUTY_EXACT).cost] < baseline
    assert baseline < COST_ORDER[get_method(EXACT_DIAGONALISATION).cost]


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", method_names())
def test_get_method_round_trips_every_catalogued_name(name: str) -> None:
    assert get_method(name).name == name


def test_get_method_rejects_an_unknown_name_and_lists_the_valid_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_method("dmrg")
    message = str(excinfo.value)
    assert "dmrg" in message
    for name in method_names():
        assert name in message


# --------------------------------------------------------------------------
# Applicability
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
def test_an_even_periodic_ring_can_be_solved_every_way_there_is(n_sites: int) -> None:
    result = survey(TFIMSpec(n_sites=n_sites))
    assert set(result.names()) == set(method_names())
    assert result.rejected == ()


def test_an_open_chain_falls_outside_the_closed_form() -> None:
    result = survey(TFIMSpec(n_sites=6, boundary="open"))
    assert result.names() == (
        EXACT_DIAGONALISATION,
        VARIATIONAL_IMAGINARY_TIME,
        VARIATIONAL_QUANTUM_EIGENSOLVER,
    )
    assert result.rejected[0].method.name == PFEUTY_EXACT
    assert "periodic" in result.rejected[0].reason


def test_an_odd_ring_rules_out_the_closed_form_and_the_sampler() -> None:
    # Those two need an even ring, for reasons that share no algebra: the closed
    # form needs a symmetric half-Brillouin-zone, and the sampler needs a
    # bipartite lattice for its checkerboard sweep. The circuit needs neither,
    # so it survives alongside exact diagonalisation.
    result = survey(TFIMSpec(n_sites=7))
    assert result.names() == (EXACT_DIAGONALISATION, VARIATIONAL_QUANTUM_EIGENSOLVER)
    reasons = {item.method.name: item.reason for item in result.rejected}
    assert "even L" in reasons[PFEUTY_EXACT]
    assert "bipartite" in reasons[VARIATIONAL_IMAGINARY_TIME]


def test_only_the_baseline_survives_a_chain_too_long_to_diagonalise() -> None:
    # The reason the baseline is in the project at all, stated as a test.
    # Sized past the *sparse* cap, which is the higher of the two: a length that only
    # cleared the state-vector cap would leave exact diagonalisation available and
    # this test would pass while checking something weaker than it claims.
    result = survey(TFIMSpec(n_sites=MAX_SITES_SPARSE + 4, boundary="open"))
    assert [facts.name for facts in result.runnable()] == [VARIATIONAL_IMAGINARY_TIME]
    # The circuit route drops out for the same reason the exact one does: this
    # project simulates it, and a simulated circuit carries the whole state
    # vector. On a device it would be the one method that did not care.
    rejected = {item.method.name: item.reason for item in result.rejected}
    assert "amplitudes" in rejected[VARIATIONAL_QUANTUM_EIGENSOLVER]


@pytest.mark.parametrize("n_sites", SIZES)
def test_the_survey_partitions_the_catalogue(n_sites: int) -> None:
    result = survey(TFIMSpec(n_sites=n_sites, boundary="open"))
    surveyed = {facts.name for facts in result.applicable}
    surveyed |= {item.method.name for item in result.rejected}
    assert surveyed == set(method_names())
    assert len(result.applicable) + len(result.rejected) == len(all_methods())


@pytest.mark.parametrize("facts", all_methods(), ids=lambda facts: facts.name)
def test_applies_to_agrees_with_the_underlying_check(facts: MethodFacts) -> None:
    for boundary in ("periodic", "open"):
        for n_sites in (4, 5):
            spec = TFIMSpec(n_sites=n_sites, boundary=boundary)
            assert facts.applies_to(spec) == (facts.unsupported_reason(spec) is None)


def test_a_rejection_reason_is_a_sentence_not_a_flag() -> None:
    result = survey(TFIMSpec(n_sites=5))
    reason = result.rejected[0].reason
    assert isinstance(reason, str)
    assert len(reason.split()) > 5


def test_the_survey_reports_whether_anything_applies() -> None:
    assert survey(TFIMSpec(n_sites=4)).has_applicable_method
    empty = MethodSurvey(spec=TFIMSpec(n_sites=4), applicable=(), rejected=())
    assert not empty.has_applicable_method


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def test_the_closed_form_declares_no_memory_estimate() -> None:
    # O(L) arithmetic on an array of L/2 momenta needs no approval gate.
    assert get_method(PFEUTY_EXACT).memory_bytes(TFIMSpec(n_sites=8)) is None


@pytest.mark.parametrize("n_sites", SIZES)
def test_exact_diagonalisation_predicts_a_positive_footprint(n_sites: int) -> None:
    estimate = get_method(EXACT_DIAGONALISATION).memory_bytes(TFIMSpec(n_sites=n_sites))
    assert estimate is not None
    assert estimate > 0


def test_the_predicted_footprint_grows_at_least_exponentially() -> None:
    facts = get_method(EXACT_DIAGONALISATION)
    small = facts.memory_bytes(TFIMSpec(n_sites=6))
    large = facts.memory_bytes(TFIMSpec(n_sites=8))
    assert small is not None and large is not None
    # Two extra sites quadruple the Hilbert space; the estimate must see that.
    assert large >= 4 * small


def test_the_footprint_estimate_survives_a_spec_the_method_would_refuse() -> None:
    # The agent needs the number precisely to explain the refusal, so costing
    # an over-cap spec must not raise.
    # Past the cap *this method* is refused on, which is the sparse one. Against the
    # state-vector cap this spec would be inside the limit and the test would no
    # longer be costing a refused run at all.
    spec = TFIMSpec(n_sites=MAX_SITES_SPARSE + 4)
    assert get_method(EXACT_DIAGONALISATION).unsupported_reason(spec) is not None
    estimate = get_method(EXACT_DIAGONALISATION).memory_bytes(spec)
    assert estimate is not None
    assert estimate > 0


# --------------------------------------------------------------------------
# Description — the text that reaches a prompt or a log
# --------------------------------------------------------------------------


def test_the_description_names_the_problem_and_every_available_method() -> None:
    spec = TFIMSpec(n_sites=4)
    text = survey(spec).describe()
    assert spec.label() in text
    for name in method_names():
        assert name in text


def test_the_description_quotes_the_reason_a_method_declined() -> None:
    result = survey(TFIMSpec(n_sites=6, boundary="open"))
    text = result.describe()
    assert "unavailable:" in text
    assert result.rejected[0].reason in text


def test_the_description_reports_a_memory_footprint_where_one_is_known() -> None:
    text = survey(TFIMSpec(n_sites=8)).describe()
    assert "MB" in text or "KB" in text


def test_the_description_says_so_when_nothing_applies() -> None:
    spec = TFIMSpec(n_sites=4)
    reason = "no method covers this"
    empty = MethodSurvey(
        spec=spec,
        applicable=(),
        rejected=tuple(Rejection(method=facts, reason=reason) for facts in all_methods()),
    )
    text = empty.describe()
    assert "(none)" in text
    assert reason in text


def test_the_description_is_deterministic() -> None:
    spec = TFIMSpec(n_sites=6, boundary="open")
    assert survey(spec).describe() == survey(spec).describe()


# --------------------------------------------------------------------------
# The one callable the agent holds really does solve the problem
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_baseline_the_agent_can_run_returns_a_variational_bound() -> None:
    # Checked against the free-fermion energy, which this test may import and
    # the catalogue may not. Per spin, and one-sided: a variational method that
    # came out *below* the ground state would mean the sampler is wrong, and a
    # two-sided tolerance would hide it.
    from src.physics.reference import free_fermions

    spec = TFIMSpec(n_sites=6, field=1.0)
    sampled = get_method(VARIATIONAL_IMAGINARY_TIME).solve(spec)
    exact = free_fermions.ground_state_energy(spec) / spec.n_sites
    assert sampled >= exact - 1e-3
    assert sampled < exact + 0.2


# --------------------------------------------------------------------------
# The closed form is a property of a line, and the menu has to say so
# --------------------------------------------------------------------------
#
# The defect these pin. `list_methods(n_sites=16, boundary="periodic")` answered
# `available: ['pfeuty_exact', ...]` for every shape, because the predicate that
# refuses the closed form had only a spec to look at and no spec carried a shape.
# So a 4x4 periodic square was told a cheap exact answer existed for it -- and
# `pfeuty_exact` is the reference every accuracy claim in the project is measured
# against, so offering it for a shape it cannot solve does not merely mislabel a
# menu, it reports the wrong problem's answer as the right one's.


@pytest.mark.parametrize("boundary", ["open", "periodic"])
@pytest.mark.parametrize(
    ("geometry", "rows"),
    [("square", 4), ("square", 2), ("triangular", 4)],
)
def test_the_closed_form_refuses_every_shape_that_is_not_a_line(
    geometry: str, rows: int, boundary: str
) -> None:
    spec = TFIMSpec(n_sites=16, boundary=boundary, geometry=geometry, rows=rows)  # type: ignore[arg-type]
    reason = free_fermion_unsupported_reason(spec)
    assert reason is not None
    # The refusal names the shape, so a report can quote it rather than paraphrase.
    assert geometry in reason
    assert "Jordan-Wigner" in reason
    assert not get_method(PFEUTY_EXACT).applies_to(spec)
    assert PFEUTY_EXACT not in survey(spec).names()


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_the_closed_form_still_applies_to_the_line_it_was_derived_for(boundary: str) -> None:
    # The refusal above must not have been bought by refusing everything: an even
    # ring is exactly the case Pfeuty solved, and it stays available.
    ring = TFIMSpec(n_sites=16, boundary="periodic")
    assert free_fermion_unsupported_reason(ring) is None
    assert PFEUTY_EXACT in survey(ring).names()
    # An open line is refused for the reason it always was -- the boundary, not the
    # shape -- and the two reasons must not be confused for one another.
    segment = TFIMSpec(n_sites=16, boundary="open")
    open_reason = free_fermion_unsupported_reason(segment)
    assert open_reason is not None
    assert "periodic ring" in open_reason
    del boundary


def test_the_sampled_baseline_declines_a_shape_it_cannot_represent() -> None:
    # Its dual lattice is sites by imaginary-time slices, which is two-dimensional
    # for a line. A 2D quantum problem needs a 3D classical one, and sampling it as
    # a line of the same size would report the wrong problem's answer with a
    # convincingly small error bar -- the failure that is invisible from the number.
    square = TFIMSpec(n_sites=16, boundary="open", geometry="square", rows=4)
    reason = variational_imaginary_time_unsupported_reason(square)
    assert reason is not None
    assert "three-dimensional" in reason
    assert VARIATIONAL_IMAGINARY_TIME not in survey(square).names()


def test_the_two_methods_that_do_generalise_are_still_offered_on_a_lattice() -> None:
    # A refusal that swept up everything would be safe and useless. Both of these
    # build their operators from a bond list, so a shape costs them nothing but
    # bonds, and both must survive.
    square = TFIMSpec(n_sites=16, boundary="open", geometry="square", rows=4)
    offered = survey(square).names()
    assert EXACT_DIAGONALISATION in offered
    assert VARIATIONAL_QUANTUM_EIGENSOLVER in offered
