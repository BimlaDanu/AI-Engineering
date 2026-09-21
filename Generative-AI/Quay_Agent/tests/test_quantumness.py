"""Tests for the operator algebra behind the word "quantum".

These are identities, not recorded numbers, which is what makes them worth
asserting: ``{sigma^x, sigma^z} = 0`` holds at every field and every chain length,
so a test of it cannot go stale the way a golden value can.

``L <= 8`` throughout, and here the cap is the module's own: these operators are
dense.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.model import TFIMSpec
from src.physics.quantumness import (
    HBAR,
    MAX_SITES,
    PAULI,
    Configuration,
    anticommutator,
    commutator,
    configuration_spins,
    measure,
    norm,
    parity,
    site_operator,
    superposition,
    terms,
)
from src.physics.reference import free_fermions

RING = TFIMSpec(n_sites=4, coupling=1.0, field=1.0)


def test_the_pauli_algebra_is_what_it_claims_to_be() -> None:
    # [sigma^x, sigma^y] = 2i sigma^z, the identity the whole module rests on.
    found = commutator(PAULI["x"], PAULI["y"])
    assert np.allclose(found, 2j * PAULI["z"])


def test_two_axes_on_one_spin_anticommute() -> None:
    assert norm(anticommutator(PAULI["x"], PAULI["z"])) < 1e-12


def test_the_hamiltonian_this_module_builds_is_the_right_one() -> None:
    # Built from Kronecker products rather than from `ed`'s bit arithmetic, so
    # agreement with the closed form checks the convention rather than assuming it.
    ising, field = terms(RING)
    values = np.linalg.eigvalsh(ising + field)
    assert float(values[0]) == pytest.approx(free_fermions.ground_state_energy(RING), abs=1e-12)


def test_the_terms_do_not_commute_unless_the_model_is_classical() -> None:
    quantum = measure(RING)
    assert not quantum.classical
    assert quantum.terms_commutator > 1.0
    for classical in (
        TFIMSpec(n_sites=4, coupling=1.0, field=0.0),
        TFIMSpec(n_sites=4, coupling=1.0, field=0.0, boundary="open"),
    ):
        assert measure(classical).classical


def test_parity_is_an_exact_symmetry_at_every_field() -> None:
    for field in (0.0, 0.5, 1.0, 2.0):
        spec = TFIMSpec(n_sites=4, coupling=1.0, field=field)
        ising, transverse = terms(spec)
        assert norm(commutator(ising + transverse, parity(spec.n_sites))) < 1e-9


def test_the_ground_state_sits_in_the_even_sector_once_the_field_is_on() -> None:
    # At h = 0 the two lowest states are degenerate and any mixture is a ground
    # state, so parity is only pinned once the field separates the sectors.
    assert measure(RING).parity_expectation == pytest.approx(1.0, abs=1e-9)


def test_different_sites_are_independent_degrees_of_freedom() -> None:
    found = measure(RING)
    assert found.offsite_commutator < 1e-12
    assert found.onsite_anticommutator < 1e-12


def test_the_uncertainty_relation_holds_and_tightens_with_the_field() -> None:
    # Delta S^y Delta S^z >= (hbar/2)|<S^x>|, and the bound is set by the same
    # magnetisation the sweep plots. Deep in the disordered phase the ground state
    # approaches an eigenstate of sigma^x and the inequality is saturated.
    slacks = []
    for field in (0.5, 1.0, 2.0, 5.0):
        found = measure(TFIMSpec(n_sites=4, coupling=1.0, field=field))
        assert found.product >= found.bound - 1e-12
        assert found.slack >= -1e-12
        slacks.append(found.slack)
    assert slacks == sorted(slacks, reverse=True)
    assert slacks[-1] < 0.01


def test_the_spreads_are_measured_in_units_of_hbar() -> None:
    # Every Pauli matrix squares to the identity, so a spread can never exceed
    # hbar/2 -- which is what makes the bound reachable rather than academic.
    found = measure(RING)
    assert found.spread_y <= 0.5 * HBAR + 1e-12
    assert found.spread_z <= 0.5 * HBAR + 1e-12


def test_a_chain_too_long_for_dense_operators_is_refused_with_the_reason() -> None:
    with pytest.raises(ValueError, match="dense-operator cap"):
        measure(TFIMSpec(n_sites=MAX_SITES + 2))


# --------------------------------------------------------------------------
# The ground state as a picture: what the cartoons on the Lab page are drawn from
# --------------------------------------------------------------------------


def test_a_basis_index_reads_back_as_the_spins_the_operators_agree_it_is() -> None:
    # The cartoon draws an arrow per spin, so a wrong convention here is a figure
    # that is confidently wrong -- and nothing else in the module would notice.
    # Checked against the same diagonals the operators produce, for every state.
    for n_sites in (2, 3):
        diagonals = [np.diag(site_operator(n_sites, "z", site)).real for site in range(n_sites)]
        for index in range(2**n_sites):
            spins = configuration_spins(n_sites, index)
            assert spins == tuple(int(diagonal[index]) for diagonal in diagonals)


def test_a_domain_wall_is_a_bond_whose_spins_disagree() -> None:
    # A wall costs 2J, so this count is the configuration's classical energy and
    # the label the cartoon sorts by.
    assert Configuration((1, 1, 1, 1), 0.25).domain_walls() == 0
    assert Configuration((-1, -1, -1, -1), 0.25).domain_walls() == 0
    assert Configuration((1, -1, 1, -1), 0.25).domain_walls() == 4
    # A ring has one more bond than a segment, and it is the one that wraps.
    assert Configuration((1, 1, -1, -1), 0.25).domain_walls() == 2
    assert Configuration((1, 1, -1, -1), 0.25).domain_walls(periodic=False) == 1


def test_the_ordered_ground_state_is_the_two_aligned_arrangements() -> None:
    # The picture the page claims at small field: all up and all down, equally, and
    # nothing else close. Asserted as physics rather than as recorded numbers --
    # which arrangements are heaviest, not what their weights happen to be.
    picture = superposition(TFIMSpec(n_sites=6, coupling=1.0, field=0.2))
    first, second = picture.configurations[0], picture.configurations[1]
    assert {first.spins, second.spins} == {(1,) * 6, (-1,) * 6}
    assert first.probability == pytest.approx(second.probability, abs=1e-9)
    assert first.probability > 0.4
    # Everything after them has a wall in it, and costs 2J to make.
    assert all(other.domain_walls() >= 2 for other in picture.configurations[2:])


def test_the_field_spreads_the_state_over_more_arrangements() -> None:
    # The left-to-right story of the strip of cartoons, as a monotone sequence: one
    # arrangement in the classical limit, all 2**L once the field has won.
    counts = [
        superposition(TFIMSpec(n_sites=6, coupling=1.0, field=field)).effective_count
        for field in (0.0, 0.5, 1.0, 2.0, 8.0)
    ]
    assert counts == sorted(counts)
    assert counts[0] == pytest.approx(1.0, abs=1e-6)
    assert counts[-1] > 60.0  # 2**6 = 64


def test_the_cartoon_says_how_much_of_the_state_it_is_not_showing() -> None:
    # The rows are a selection, and a picture that hid most of the state without
    # saying so would be the same failure as an unchecked number.
    picture = superposition(TFIMSpec(n_sites=6, coupling=1.0, field=1.0), limit=4)
    assert len(picture.configurations) == 4
    assert 0.0 < picture.shown_weight <= 1.0
    assert picture.shown_weight == pytest.approx(
        sum(row.probability for row in picture.configurations), abs=1e-12
    )
    weights = [row.probability for row in picture.configurations]
    assert weights == sorted(weights, reverse=True)
    assert picture.full_count == 64


def test_a_single_spins_arrow_shrinks_where_the_chain_is_entangled() -> None:
    # The arrow length is the one-site Bloch vector, so it is a purity: 1 means the
    # spin has a state of its own, and short means its information is in the
    # correlations instead. It is never longer than 1, which would not be a state.
    lengths = [
        superposition(TFIMSpec(n_sites=6, coupling=1.0, field=field)).arrow_length
        for field in (1.0, 2.0, 20.0)
    ]
    assert lengths == sorted(lengths)
    assert lengths[0] < 1.0
    assert lengths[-1] == pytest.approx(1.0, abs=1e-2)
    for length in lengths:
        assert length <= 1.0 + 1e-12


def test_the_arrow_points_along_x_because_the_symmetry_forbids_anything_else() -> None:
    # <sigma^z> = 0 in any parity eigenstate, so the arrow can only lie along x.
    # This is why the cartoon draws a length rather than a direction.
    x, y, z = superposition(TFIMSpec(n_sites=6, coupling=1.0, field=1.0)).arrow
    assert x > 0.0
    assert y == pytest.approx(0.0, abs=1e-12)
    assert z == pytest.approx(0.0, abs=1e-12)


def test_the_picture_agrees_with_the_magnetisation_the_rest_of_the_project_reports() -> None:
    # Same expectation value, two modules: the cartoon's arrow and the closed form.
    spec = TFIMSpec(n_sites=6, coupling=1.0, field=0.8)
    assert superposition(spec).arrow[0] == pytest.approx(
        free_fermions.transverse_magnetisation(spec), abs=1e-9
    )


def test_a_chain_too_long_to_draw_is_refused_with_the_same_reason() -> None:
    with pytest.raises(ValueError, match="dense-operator cap"):
        superposition(TFIMSpec(n_sites=MAX_SITES + 2))


def test_an_operator_lands_on_the_site_it_was_asked_for() -> None:
    # sigma^z on site 0 of two spins is diag(1, 1, -1, -1); on site 1 it alternates.
    assert np.allclose(np.diag(site_operator(2, "z", 0)).real, [1, 1, -1, -1])
    assert np.allclose(np.diag(site_operator(2, "z", 1)).real, [1, -1, 1, -1])
