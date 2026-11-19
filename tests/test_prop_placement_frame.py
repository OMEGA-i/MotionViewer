"""Weapon placement must select its requested pose before inspecting the rig."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from motionviewer.blender.mmd_props import fit_weapon_to_hand


@pytest.mark.parametrize(("requested", "expected"), [(1, 1), (45, 45), (None, 180)])
def test_weapon_fit_selects_frame_before_reading_pose(monkeypatch, requested, expected):
    # An unavailable hand aborts placement at its first rig read. This isolates
    # frame selection without a real Blender install or licensed character.
    scene = SimpleNamespace(frame_current=180)
    calls = []

    def frame_set(frame):
        calls.append(("frame", frame))
        scene.frame_current = frame

    class MissingHand:
        def __contains__(self, name):
            calls.append(("read_pose", scene.frame_current))
            assert scene.frame_current == expected
            return False

    scene.frame_set = frame_set
    bpy = SimpleNamespace(context=SimpleNamespace(
        scene=scene,
        view_layer=SimpleNamespace(update=lambda: calls.append(("update", scene.frame_current))),
    ))
    armature = SimpleNamespace(pose=SimpleNamespace(bones=MissingHand()))
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Matrix=object, Vector=object))

    with pytest.raises(KeyError, match="右手首"):
        fit_weapon_to_hand(bpy, object(), armature, "右手首", frame=requested)

    assert calls[-2:] == [("update", expected), ("read_pose", expected)]
    if requested is None:
        assert all(kind != "frame" for kind, _ in calls)
    else:
        assert calls[0] == ("frame", requested)
