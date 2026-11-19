import numpy as np
import pytest

from motionviewer.blender.mmd_contact import FootSamples, _support_metrics, source_feet
from motionviewer.core.smplx_fk import source_to_blender


@pytest.mark.parametrize("missing_shoes", [False, True])
def test_display_floor_uses_full_shoe_motion_and_restores_frame(monkeypatch, missing_shoes):
    from types import SimpleNamespace

    from motionviewer.blender import mmd_contact

    scene = SimpleNamespace(frame_current=17)
    scene.frame_set = lambda frame: setattr(scene, "frame_current", frame)
    bpy = SimpleNamespace(context=SimpleNamespace(
        scene=scene, view_layer=SimpleNamespace(update=lambda: None),
    ))

    def measure(bpy, actor, *, frame_start, frames):
        assert (frame_start, frames) == (1, 3)
        scene.frame_set(3)
        if missing_shoes:
            raise ValueError("Cannot identify reliable left/right shoe skin")
        return SimpleNamespace(sole_z=np.array([[.02, .03], [.01, .05], [.04, .02]]),
                               vertex_counts=(20, 25))

    monkeypatch.setattr(mmd_contact, "measure_feet", measure)
    result = mmd_contact.estimate_display_floor(
        bpy, object(), frame_start=1, frames=3, fallback_floor=-.1,
    )
    assert scene.frame_current == 17
    assert result["changes_motion"] is False
    if missing_shoes:
        assert result["floor_z"] == -.1
        assert result["method"] == "mesh_bounds_fallback"
        assert "shoe skin" in result["reason"]
    else:
        assert result["floor_z"] == .01  # Low hair/cloth bounds must not lower the floor.
        assert result["method"] == "full_clip_shoe_minimum"
        assert result["lowest_shoe_range_m"] == [.01, .02]


def test_soma_contact_uses_archive_joint_names_not_internal_rig_names(tmp_path):
    names = np.array(["Hips", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"])
    joints = np.arange(45, dtype=float).reshape(3, 5, 3)
    path = tmp_path / "source.soma.npz"
    np.savez(path, joints77=joints, joint_names=names, fps=50.)
    original = path.read_bytes()
    ankles, toes, fps = source_feet(path, 2)
    np.testing.assert_array_equal(ankles, source_to_blender(joints[:2, [1, 2]]))
    np.testing.assert_array_equal(toes, source_to_blender(joints[:2, [3, 4]]))
    assert fps == 50.
    assert path.read_bytes() == original


def test_contact_metric_does_not_count_toe_pivot_as_sliding():
    ankles = np.zeros((4, 2, 3))
    ankles[:, :, 0] = np.arange(4)[:, None] * 0.01
    zeros = np.zeros_like(ankles)
    sample = FootSamples(ankles, zeros, zeros, np.zeros((4, 2)), zeros, zeros,
                         np.zeros((4, 3)), (10, 10))
    modes = np.full((4, 2), "toe")
    modes[0] = "flat"
    report = _support_metrics(sample, np.ones((4, 2)), modes, 0., 50.)
    assert report["support_speed_pairs"] == 4
    assert report["support_speed_cm_s_mean"] == 0.
    assert report["ankle_contact_speed_cm_s_p90"] > 0.
