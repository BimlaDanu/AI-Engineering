"""Tests for the grader's bench -- the binding from a catalogue entry to a solver.

The catalogue's own bookkeeping is tested in ``tests/test_method_catalogue.py``.
What is under test here is narrower and it is the half the agent cannot see:
that every method described as ``grader_only`` really does have an
implementation behind it, that binding attaches the right one, and that the two
exact routes -- which share no algebra -- return the same number.

Nothing here runs a solver on a chain longer than
:data:`~src.physics.model.DEFAULT_SITES`, which is what keeps the bench cheap.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.method_catalogue import (
    EXACT_DIAGONALISATION,
    PFEUTY_EXACT,
    VARIATIONAL_IMAGINARY_TIME,
    MethodFacts,
    all_methods,
    get_method,
)
from src.physics.model import DEFAULT_SITES, TFIMSpec
from src.physics.reference import exact_diagonalisation, free_fermions
from src.physics.registry import bind, exact_methods, exact_survey, solver_for

SIZES = [2, 4, DEFAULT_SITES]

MODULES = {
    free_fermions.METHOD_NAME: free_fermions,
    exact_diagonalisation.METHOD_NAME: exact_diagonalisation,
}


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module", [free_fermions, exact_diagonalisation])
def test_every_solver_module_is_catalogued_under_its_own_method_name(module: object) -> None:
    # Guards against the catalogue and the implementation drifting apart, which
    # would surface as an agent naming a method that cannot be looked up.
    name = module.METHOD_NAME  # type: ignore[attr-defined]
    assert name in {facts.name for facts in all_methods()}


@pytest.mark.parametrize("name", [PFEUTY_EXACT, EXACT_DIAGONALISATION])
def test_binding_attaches_the_function_from_the_module_it_claims(name: str) -> None:
    bound = solver_for(name)
    assert bound.ground_state_energy is MODULES[name].ground_state_energy


@pytest.mark.parametrize("name", [PFEUTY_EXACT, EXACT_DIAGONALISATION])
def test_binding_leaves_the_facts_alone(name: str) -> None:
    # Only the callable is filled in. If binding could change a cost class or a
    # rejection reason, the grader and the agent would be reasoning about two
    # different methods that happen to share a name.
    facts, bound = get_method(name), solver_for(name)
    assert (bound.name, bound.cost, bound.accuracy) == (facts.name, facts.cost, facts.accuracy)
    assert bound.unsupported_reason is facts.unsupported_reason
    assert bound.summary == facts.summary


def test_binding_an_already_runnable_method_is_a_no_op() -> None:
    facts = get_method(VARIATIONAL_IMAGINARY_TIME)
    assert bind(facts) is facts


@pytest.mark.parametrize("facts", all_methods(), ids=lambda facts: facts.name)
def test_every_catalogued_method_can_be_bound_to_something(facts: MethodFacts) -> None:
    # A method described in the catalogue and wired to nothing would appear in
    # the agent's menu and fail only when the grader tried to score against it.
    assert bind(facts).ground_state_energy is not None


def test_binding_reports_an_unwired_method_rather_than_returning_a_hollow_entry() -> None:
    invented = MethodFacts(
        name="dmrg",
        summary="x" * 50,
        when_to_use="y" * 50,
        cost="polynomial",
        accuracy="variational_bound",
        availability="grader_only",
        unsupported_reason=lambda spec: None,
    )
    with pytest.raises(KeyError):
        bind(invented)


# --------------------------------------------------------------------------
# The exact survey — what the cross-check iterates over
# --------------------------------------------------------------------------


def test_the_exact_survey_holds_only_the_exact_routes() -> None:
    # The variational baseline is deliberately absent. It returns an upper bound
    # with an error bar, and comparing that against a closed form at 1e-9 would
    # report a contradiction where there is only a variational gap.
    assert {facts.name for facts in exact_methods()} == {PFEUTY_EXACT, EXACT_DIAGONALISATION}
    assert all(facts.accuracy == "exact" for facts in exact_methods())


@pytest.mark.parametrize("n_sites", SIZES)
def test_an_even_periodic_ring_is_solvable_both_exact_ways(n_sites: int) -> None:
    result = exact_survey(TFIMSpec(n_sites=n_sites))
    assert result.names() == (PFEUTY_EXACT, EXACT_DIAGONALISATION)
    assert result.rejected == ()


def test_an_open_chain_leaves_only_exact_diagonalisation() -> None:
    result = exact_survey(TFIMSpec(n_sites=6, boundary="open"))
    assert result.names() == (EXACT_DIAGONALISATION,)
    assert result.rejected[0].method.name == PFEUTY_EXACT
    assert "periodic" in result.rejected[0].reason


def test_an_odd_chain_leaves_only_exact_diagonalisation() -> None:
    result = exact_survey(TFIMSpec(n_sites=7))
    assert result.names() == (EXACT_DIAGONALISATION,)
    assert "even L" in result.rejected[0].reason


@pytest.mark.parametrize("n_sites", SIZES)
def test_the_exact_survey_partitions_the_exact_methods(n_sites: int) -> None:
    result = exact_survey(TFIMSpec(n_sites=n_sites, boundary="open"))
    surveyed = {facts.name for facts in result.applicable}
    surveyed |= {item.method.name for item in result.rejected}
    assert surveyed == {facts.name for facts in exact_methods()}


# --------------------------------------------------------------------------
# The bound callables really solve the problem
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", SIZES)
def test_every_applicable_exact_method_agrees_with_every_other(n_sites: int) -> None:
    # The headline claim, routed through the bench rather than by importing the
    # two solvers by hand: whichever route the grader takes, it gets the same
    # number to machine precision.
    spec = TFIMSpec(n_sites=n_sites, coupling=1.0, field=0.7)
    energies = [facts.solve(spec) for facts in exact_survey(spec).applicable]
    assert len(energies) == 2
    assert energies[0] == pytest.approx(energies[1], abs=1e-10)


def test_calling_a_method_the_survey_rejected_raises_with_its_name() -> None:
    spec = TFIMSpec(n_sites=5)
    rejected = exact_survey(spec).rejected[0]
    with pytest.raises(ValueError, match=f"{rejected.method.name} cannot solve"):
        rejected.method.solve(spec)


# --------------------------------------------------------------------------
# The gap, and which gap it is
# --------------------------------------------------------------------------
#
# `free_fermions.gap` returns one quasiparticle's energy, and for a long while the
# interface presented that as "the energy gap" of the chain. It is neither the
# spacing to the first excited state nor the rate that sets imaginary-time
# convergence, and reading it as either is wrong in the optimistic direction. These
# hold the two apart, in the house pattern: a closed form against a matrix
# diagonalisation, which share no algebra at all.


def even_parity_levels(spec: TFIMSpec) -> tuple[float, float]:
    r"""The two lowest energies in the even-parity sector, from the matrix.

    Written out here rather than taken from a level index, because the index moves.
    Below the critical field the ring's first excited state is the odd-parity
    partner of the ground state and its second is the cheapest quasiparticle pair;
    above it the odd sector fills in with single quasiparticles and the pair is
    pushed several levels up. Sorting by parity rather than by position is what
    makes one assertion cover both phases.

    Parity is :math:`\hat P = \prod_i \hat\sigma^x_i`, which flips every spin --
    in this module's basis, the permutation sending each index to its bitwise
    complement.

    Args:
        spec: The ring to diagonalise.

    Returns:
        The lowest two even-parity energies, ascending.
    """
    dense = exact_diagonalisation.hamiltonian(spec).toarray()
    dimension = dense.shape[0]
    flipped = np.arange(dimension) ^ (dimension - 1)
    values, vectors = np.linalg.eigh(dense)
    parity = np.einsum("ij,ij->j", vectors[flipped, :], vectors)
    even = values[parity > 0.5]
    return float(even[0]), float(even[1])


@pytest.mark.parametrize("n_sites", [4, DEFAULT_SITES, 8])
@pytest.mark.parametrize("field", [0.2, 0.7, 1.0, 1.6, 2.5])
def test_the_even_parity_gap_is_what_the_matrix_says_it_is(n_sites: int, field: float) -> None:
    # The number every convergence estimate in the project rests on, checked against
    # a route that knows nothing about momenta, sectors or Bogoliubov rotations.
    ring = TFIMSpec(n_sites=n_sites, coupling=1.0, field=field, boundary="periodic")
    lowest, next_one = even_parity_levels(ring)
    assert free_fermions.parity_even_gap(ring) == pytest.approx(next_one - lowest, abs=1e-9)


def test_one_quasiparticle_is_not_the_distance_to_the_first_excited_state() -> None:
    # A regression pinning the bug rather than the fix. Below the critical field the
    # ring's two lowest states are a doublet split by an amount that vanishes
    # exponentially with L, while `gap` returns a number of order 2(J - h). Anything
    # reading `gap` as "the gap" is wrong here by six orders of magnitude, so this
    # exists to make that impossible to reintroduce quietly.
    ordered = TFIMSpec(n_sites=10, coupling=1.0, field=0.2, boundary="periodic")
    levels = exact_diagonalisation.low_levels(ordered, count=3)
    assert float(levels[1] - levels[0]) < 1e-5
    assert free_fermions.gap(ordered) > 1.0


def test_the_finite_quasiparticle_energy_tends_to_the_infinite_one() -> None:
    # `gap` and `gap_thermodynamic` are the same quantity at finite and infinite
    # length, which is what makes it legitimate to draw them on one pair of axes.
    # Away from the critical point they converge, and the finite one is always the
    # larger: a ring forbids the momentum that would make the excitation cheapest.
    for length in (8, 32, 128):
        ring = TFIMSpec(n_sites=length, coupling=1.0, field=0.4, boundary="periodic")
        assert free_fermions.gap(ring) >= free_fermions.gap_thermodynamic(1.0, 0.4)
    far = TFIMSpec(n_sites=512, coupling=1.0, field=0.4, boundary="periodic")
    assert free_fermions.gap(far) == pytest.approx(
        free_fermions.gap_thermodynamic(1.0, 0.4), abs=1e-3
    )
