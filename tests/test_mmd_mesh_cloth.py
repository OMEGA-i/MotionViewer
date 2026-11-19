"""Reject ambiguous cloth surfaces instead of silently simulating both layers."""

from types import SimpleNamespace

import pytest

from motionviewer.blender.mmd_mesh_cloth import MeshClothSettings, _components, _outer_faces


def test_connected_pieces_remain_distinct_until_explicit_welding():
    assert {frozenset(c) for c in _components({0: {1}, 1: {0}, 2: set()})} == {
        frozenset({0, 1}), frozenset({2}),
    }


def _double_sheet():
    return SimpleNamespace(data=SimpleNamespace(
        materials=[SimpleNamespace(name="tail"), SimpleNamespace(name="skirt")],
        polygons=[SimpleNamespace(
            index=index, material_index=material, vertices=vertices, area=1,
            center=SimpleNamespace(x=1, y=0), normal=SimpleNamespace(x=normal, y=0),
        ) for index, material, vertices, normal in (
            (0, 0, (0, 1, 2), 1), (1, 0, (3, 4, 5), -1), (2, 1, (6, 7, 8), -1),
        )],
    ))


def test_only_explicit_double_layer_material_loses_its_inner_sheet():
    kept, report = _outer_faces(_double_sheet(), {0, 1, 2}, ("tail",))
    assert kept == {0, 2}
    assert [r["retained"] for r in report] == [True, False]


def test_unexpected_single_sheet_fails_instead_of_guessing_new_asset_topology():
    with pytest.raises(ValueError, match="distinct outward/inward"):
        _outer_faces(_double_sheet(), {0, 2}, ("tail",))


def test_pin_gradient_must_have_positive_width():
    with pytest.raises(ValueError, match="below transition"):
        MeshClothSettings(("skirt",), ("skirt",), pin_transition_weight=0.1, pin_full_weight=0.2)
