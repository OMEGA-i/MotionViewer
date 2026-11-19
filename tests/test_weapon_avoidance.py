"""Hermetic checks for temporal bounds of the reviewed arm adaptation."""

import numpy as np
import pytest

from motionviewer.blender.mmd_weapon_avoidance import clearance_weights
from motionviewer.blender.weapon_grip_profiles import PROFILES


def test_clearance_bump_is_symmetric_and_leaves_outside_frames_unchanged():
    frames = np.arange(1, 101)
    weights = clearance_weights(frames, 50, 20)
    assert weights[49] == 1
    assert np.all(weights[(frames <= 30) | (frames >= 70)] == 0)
    np.testing.assert_allclose(weights[29:70], weights[29:70][::-1], atol=1e-15)
    assert np.max(np.abs(np.diff(weights))) < np.pi / 40
    # Both boundary slopes approach zero; no hard switch at the window edge.
    assert weights[30] < .007
    assert weights[68] < .007


@pytest.mark.parametrize("width", [0, -1, np.inf, np.nan])
def test_clearance_requires_finite_positive_transition(width):
    with pytest.raises(ValueError, match="positive and finite"):
        clearance_weights(np.arange(10), 5, width)


def test_reviewed_seed_limits_are_explicit_and_axis_is_unit():
    seed = PROFILES[0]["left_arm_clearance"]
    assert seed["degrees"] <= seed["max_joint_degrees"] <= 20
    assert seed["max_offset_m"] <= .12
    np.testing.assert_allclose(np.linalg.norm(seed["axis_in_weapon_frame"]), 1, atol=2e-6)


@pytest.mark.parametrize("outcome", ["rejected", "exception", "applied"])
def test_clearance_transaction_preserves_source_action(monkeypatch, outcome):
    from types import SimpleNamespace

    from motionviewer.blender import mmd_weapon_avoidance as module

    class Action:
        name = "source"
        users = 0

        def __init__(self):
            self.keys = [1, 2, 3]

        def copy(self):
            clone = Action()
            clone.keys = self.keys.copy()
            return clone

    source = Action()
    armature = SimpleNamespace(animation_data=SimpleNamespace(action=source),
                               pose=SimpleNamespace(bones={}))
    removed, frames = [], []
    bpy = SimpleNamespace(context=SimpleNamespace(
        scene=SimpleNamespace(frame_current=7, frame_set=frames.append),
        view_layer=SimpleNamespace(update=lambda: None)),
        data=SimpleNamespace(actions=SimpleNamespace(remove=removed.append)))

    def attempt(*args, **kwargs):
        armature.animation_data.action.keys.append(99)
        if outcome == "exception":
            raise RuntimeError("key insertion failed")
        return {"applied": outcome == "applied"}

    monkeypatch.setattr(module, "_apply_weapon_left_avoidance", attempt)
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="key insertion failed"):
            module.apply_weapon_left_avoidance(bpy, armature, None, frame_start=1, frames=10, fps=50)
    else:
        report = module.apply_weapon_left_avoidance(bpy, armature, None, frame_start=1, frames=10, fps=50)
        assert report["render_action_copy"] == (outcome == "applied")
    assert source.keys == [1, 2, 3]
    if outcome == "applied":
        assert armature.animation_data.action is not source
        assert armature.animation_data.action.keys == [1, 2, 3, 99]
        assert not removed
    else:
        assert armature.animation_data.action is source
        assert len(removed) == 1
        assert frames == [7]
