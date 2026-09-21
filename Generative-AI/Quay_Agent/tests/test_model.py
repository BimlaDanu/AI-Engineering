from typing import Any

import pytest

from src.physics.model import (
    MAX_SITES_SPARSE,
    MAX_SITES_STATEVECTOR,
    WORKING_SITES,
    TFIMSpec,
)


def test_defaults_are_the_critical_point() -> None:
    spec = TFIMSpec(n_sites=8)
    assert spec.coupling == 1.0
    assert spec.field == 1.0
    assert spec.ratio == 1.0
    assert spec.boundary == "periodic"


def test_ring_has_one_bond_per_site_and_segment_has_one_fewer() -> None:
    assert TFIMSpec(n_sites=10, boundary="periodic").n_bonds == 10
    assert TFIMSpec(n_sites=10, boundary="open").n_bonds == 9


def test_spec_is_frozen_and_hashable() -> None:
    spec = TFIMSpec(n_sites=6)
    with pytest.raises(AttributeError):
        spec.n_sites = 8  # type: ignore[misc]
    # Hashability is what lets an identical question hit the run cache.
    assert hash(spec) == hash(TFIMSpec(n_sites=6))
    assert spec == TFIMSpec(n_sites=6)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"n_sites": 1}, ValueError),
        ({"n_sites": 0}, ValueError),
        ({"n_sites": 4.0}, TypeError),
        ({"n_sites": True}, TypeError),
        ({"n_sites": 4, "coupling": 0.0}, ValueError),
        ({"n_sites": 4, "coupling": -1.0}, ValueError),
        ({"n_sites": 4, "field": -0.1}, ValueError),
        ({"n_sites": 4, "boundary": "helical"}, ValueError),
    ],
)
def test_invalid_specs_are_rejected_at_construction(
    kwargs: dict[str, Any], expected: type[Exception]
) -> None:
    with pytest.raises(expected):
        TFIMSpec(**kwargs)


def test_label_is_stable_and_readable() -> None:
    assert TFIMSpec(n_sites=12, coupling=1.0, field=0.5).label() == "TFIM L=12 J=1 h=0.5 (peri)"


# --------------------------------------------------------------------------
# The shape is part of the problem statement
# --------------------------------------------------------------------------
#
# It was not, and one case inverted a verdict. `TFIMSpec` was
# (n_sites, coupling, field, boundary), every method took an optional geometry
# alongside it, and the geometry defaulted to a line -- so a sixteen-site problem
# asked about a 4x4 square was told the closed form applied to it. The closed form
# exists for a line and nothing else. Its answer at sixteen sites is -20.40 where
# the square's is -34.01, so a genuine variational energy for the square came back
# far *below* the number reported as exact, which reads as a broken variational
# bound and is really a comparison between two different problems.


def test_a_spec_defaults_to_the_line_it_always_was() -> None:
    spec = TFIMSpec(n_sites=12, boundary="open")
    assert spec.geometry == "chain"
    assert spec.rows == 1
    assert spec.cols == 12
    assert spec.is_one_dimensional
    # A line takes its bonds from the chain generators, not from the shared edge
    # list, so the two independent generators stay two.
    assert spec.shape_to_solve is None


def test_a_spec_can_name_a_two_dimensional_shape() -> None:
    square = TFIMSpec(n_sites=16, boundary="open", geometry="square", rows=4)
    assert square.cols == 4
    assert not square.is_one_dimensional
    assert square.lattice.describe() == "4x4 square (open)"
    # 24 bonds on a 4x4 open square: 12 horizontal, 12 vertical. A line of sixteen
    # has 15, which is the arithmetic that made the two problems look comparable.
    assert square.n_bonds == 24
    assert TFIMSpec(n_sites=16, boundary="open").n_bonds == 15
    assert square.shape_to_solve is not None
    assert square.shape_to_solve.n_bonds == 24


def test_the_shape_appears_in_the_label_only_when_it_is_not_a_line() -> None:
    # Every label that existed before the field did still reads as it did, which is
    # what keeps a run and its grading lined up by name.
    assert TFIMSpec(n_sites=6).label() == "TFIM L=6 J=1 h=1 (peri)"
    assert TFIMSpec(n_sites=16, geometry="square", rows=4).label().startswith("TFIM 4x4 square")


def test_a_ragged_lattice_is_refused_rather_than_rounded() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        TFIMSpec(n_sites=15, geometry="square", rows=4)


def test_a_two_dimensional_shape_needs_two_rows_and_a_chain_needs_one() -> None:
    with pytest.raises(ValueError, match="at least 2 rows"):
        TFIMSpec(n_sites=8, geometry="square", rows=1)
    with pytest.raises(ValueError, match="a chain has one row"):
        TFIMSpec(n_sites=8, geometry="chain", rows=2)


def test_a_shape_too_narrow_to_be_two_dimensional_is_refused_where_it_is_written() -> None:
    # The mirror of the row check, and it was missing. Rows were validated and columns
    # were not, so this spec was accepted and then raised from `Lattice` the first time
    # anything touched `.lattice` -- which is a `describe_problem` call away from the
    # mistake, by which point the tool has lost its chance to answer with the
    # `{"error": ...}` that every other bad argument gets.
    with pytest.raises(ValueError, match="at least 2 columns"):
        TFIMSpec(n_sites=3, geometry="triangular", rows=3)
    with pytest.raises(ValueError, match="at least 2 columns"):
        TFIMSpec(n_sites=4, geometry="square", rows=4)


def test_every_shape_a_spec_accepts_can_actually_be_built() -> None:
    # The property the test above exists to protect, stated directly: validation lives
    # in two classes, and the halves are only useful if they agree. A spec that
    # constructs and then cannot produce its own lattice is the shape of that
    # disagreement, so the whole small grid is walked rather than two examples.
    for geometry in ("square", "triangular"):
        for rows in range(1, 7):
            for n_sites in range(2, 25):
                try:
                    spec = TFIMSpec(n_sites=n_sites, geometry=geometry, rows=rows)
                except ValueError:
                    continue
                assert spec.lattice.n_sites == n_sites
                assert spec.shape_to_solve is None or spec.shape_to_solve.n_sites == n_sites


def test_an_unknown_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="geometry must be one of"):
        TFIMSpec(n_sites=8, geometry="kagome", rows=2)  # type: ignore[arg-type]


def test_the_spec_is_still_hashable_with_a_shape_on_it() -> None:
    # Frozen and hashable is what lets a spec key a run cache, and two identical
    # specs must land on the same entry however many fields they grew.
    first = TFIMSpec(n_sites=16, geometry="square", rows=4)
    second = TFIMSpec(n_sites=16, geometry="square", rows=4)
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    assert first != TFIMSpec(n_sites=16, geometry="square", rows=2)


def test_the_size_ladder_holds_in_the_order_the_project_depends_on() -> None:
    """Twelve to work at, sixteen to stop at, and nothing above that.

    Three numbers in three modules that only mean something in relation to each
    other, so they are pinned together rather than each on its own. The knob is
    included because it is the one a person can move: a dial offering a size every
    panel refuses teaches only that the software is broken.

    The upper bound is the part worth stating plainly. Sixteen sites is 0.26 s and
    155 MB of exact diagonalisation; eighteen is 2.5 s and half a gigabyte; twenty
    is twelve seconds and 1.9 GB, which is what a test suite flaking on a laptop
    under load looks like from the inside.
    """
    from src.ui.setting import MAX_SITES as KNOB_CEILING

    assert WORKING_SITES == 12
    assert WORKING_SITES <= MAX_SITES_STATEVECTOR <= MAX_SITES_SPARSE == 16
    assert KNOB_CEILING == MAX_SITES_SPARSE
