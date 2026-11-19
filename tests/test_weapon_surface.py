"""Calibrated grip dispatch and rejection of changed/unreviewed asset geometry."""

from types import SimpleNamespace

import numpy as np
import pytest

from motionviewer.blender import mmd_weapon_contact
from motionviewer.blender.mmd_props import close_hand, fit_weapon_to_hand
from motionviewer.blender.weapon_grip_profiles import PROFILES


def test_surface_dispatch_preserves_frame_and_hand_context(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mmd_weapon_contact,
        "fit_surface_weapon",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"profile": "checked"},
    )
    monkeypatch.setattr(
        mmd_weapon_contact,
        "close_surface_hand",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"keys": 45},
    )
    bpy, weapon, armature = object(), object(), object()
    assert fit_weapon_to_hand(bpy, weapon, armature, "右手首", grip="high", frame=45, mode="surface") == {
        "profile": "checked"
    }
    assert close_hand(
        bpy, armature, "右手首", weapon=weapon, grip="high", frame_start=45, frames=3, mode="surface"
    ) == {"keys": 45}
    assert calls == [
        ((bpy, weapon, armature, "右手首"), {"grip": "high", "frame": 45}),
        ((bpy, armature, "右手首"), {"weapon": weapon, "grip": "high", "frame_start": 45, "frames": 3}),
    ]


def test_surface_profile_rejects_unreviewed_geometry_and_hand(monkeypatch):
    profile = PROFILES[0]
    monkeypatch.setattr(mmd_weapon_contact, "hand_signature", lambda *_: profile["hand_signature"])
    monkeypatch.setattr(mmd_weapon_contact, "geometry_signature", lambda *_: profile["weapon_signature"])
    assert mmd_weapon_contact._profile(None, None, None, "右手首", "high") is profile
    # Explicit --weapon paths historically carry a 'low' hint. A verified
    # calibration owns the selected grip section rather than trusting that hint.
    assert mmd_weapon_contact._profile(None, None, None, "右手首", "low") is profile
    with pytest.raises(ValueError, match="no calibration"):
        mmd_weapon_contact._profile(None, None, None, "左手首", "high")
    monkeypatch.setattr(mmd_weapon_contact, "geometry_signature", lambda *_: "changed asset")
    with pytest.raises(ValueError, match="no calibration"):
        mmd_weapon_contact._profile(None, None, None, "右手首", "high")


def test_geometry_signature_covers_vertices_and_topology_but_not_labels():
    obj = SimpleNamespace(
        name="first job",
        data=SimpleNamespace(
            vertices=[SimpleNamespace(co=point) for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0))],
            polygons=[SimpleNamespace(vertices=(0, 1, 2))],
        ),
    )
    original = mmd_weapon_contact.geometry_signature(obj)
    obj.name = "different render label"
    assert mmd_weapon_contact.geometry_signature(obj) == original
    obj.data.vertices[1].co = (1.01, 0, 0)
    assert mmd_weapon_contact.geometry_signature(obj) != original
    obj.data.vertices[1].co = (1, 0, 0)
    obj.data.polygons[0].vertices = (0, 2, 1)
    assert mmd_weapon_contact.geometry_signature(obj) != original


def test_calibration_preserves_asset_scale_and_keys_the_thumb():
    profile = PROFILES[0]
    placement = np.asarray(profile["weapon_in_hand"])
    np.testing.assert_allclose(placement[3], (0, 0, 0, 1))
    np.testing.assert_allclose(placement[:3, :3].T @ placement[:3, :3], np.eye(3), atol=2e-6)
    assert np.linalg.det(placement[:3, :3]) == pytest.approx(1, abs=2e-6)
    poses = profile["finger_eulers_degrees"]
    assert len(poses) == 15
    assert all("右親指" + segment in poses for segment in "０１２")
    assert all(np.isfinite(angles).all() for angles in poses.values())


def test_component_selection_rejects_unknown_geometry_before_editing(monkeypatch):
    monkeypatch.setattr(mmd_weapon_contact, "geometry_signature", lambda *_: "unreviewed")
    with pytest.raises(ValueError, match="exact source geometry"):
        mmd_weapon_contact.select_calibrated_component(object(), PROFILES[0])


def test_component_selection_cannot_cut_a_polygon(monkeypatch):
    profile = PROFILES[0]
    monkeypatch.setattr(mmd_weapon_contact, "geometry_signature", lambda *_: profile["source_weapon_signature"])
    obj = SimpleNamespace(data=SimpleNamespace(
        vertices=[SimpleNamespace(co=point) for point in ((0, 0, 0), (0, 1, 0), (.4, 0, 0))],
        polygons=[SimpleNamespace(vertices=(0, 1, 2))],
    ))
    with pytest.raises(ValueError, match="cut through a polygon"):
        mmd_weapon_contact.select_calibrated_component(obj, profile)
