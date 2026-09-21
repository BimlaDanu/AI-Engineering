from typing import Any

import pytest

from src.physics.model import TFIMSpec


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
