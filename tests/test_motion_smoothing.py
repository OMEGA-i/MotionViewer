"""Upper-body edits must settle tiny oscillations without moving the feet."""

from __future__ import annotations

import json

import numpy as np
import pytest

from motionviewer.core.motion_smoothing import (
    UPPER_BODY_JOINTS,
    angular_velocity_vectors,
    repose_joints,
    smooth_upper_body,
)
from motionviewer.core.smplx_fk import (
    SMPLX_BODY22_NAMES,
    SMPLX_BODY22_PARENTS,
    axis_angle_from_rotations,
    global_rotations,
    rodrigues,
)

INDEX = {name: index for index, name in enumerate(SMPLX_BODY22_NAMES)}


def _motion(frames=60):
    roots = np.zeros((frames, 3), dtype=np.float64)
    pose = np.zeros((frames, 21, 3), dtype=np.float64)
    translation = np.zeros((frames, 3), dtype=np.float64)
    return roots, pose, translation


def _fk(roots, pose, root_positions, offsets):
    rotations = global_rotations(roots, pose)
    joints = np.zeros((len(roots), 22, 3))
    joints[:, 0] = root_positions
    for joint, parent in enumerate(SMPLX_BODY22_PARENTS):
        if parent >= 0:
            joints[:, joint] = joints[:, parent] + np.einsum("tij,j->ti", rotations[:, parent], offsets[joint])
    return joints


def test_signed_velocity_tracks_direction_in_a_common_reference():
    steps = np.radians(np.array([[2., 0., 0.], [0., 2., 0.], [-2., 0., 0.]]))
    rotations = [rodrigues(np.array([0.3, -0.2, 0.1]))]
    for step in steps:
        rotations.append(rodrigues(step) @ rotations[-1])
    measured = angular_velocity_vectors(np.array(rotations), fps=50.0)
    assert np.allclose(measured, steps * 50, atol=1e-12)


def test_small_alternating_shake_is_reduced_and_reported():
    roots, pose, translation = _motion()
    pose[:, INDEX["spine1"] - 1, 0] = np.radians(np.tile([1.0, -1.0], 30))
    _, edited, _, report = smooth_upper_body(roots, pose, translation, fps=50)
    assert np.abs(edited[2:-2, INDEX["spine1"] - 1]).max() < 1e-12
    assert report["after"]["jitter_deg"]["mean"] < report["before"]["jitter_deg"]["mean"] * 0.1
    assert report["local_deviation_deg"]["max"] == pytest.approx(1.0)
    assert report["angular_velocity_change_deg_s"]["max"] > 90
    json.dumps(report, allow_nan=False)


def test_roots_lower_body_distal_angles_and_source_arrays_are_unchanged():
    rng = np.random.default_rng(42)
    roots, pose, translation = _motion()
    roots[:] = rng.normal(0, 0.1, roots.shape)
    pose[:] = rng.normal(0, 0.1, pose.shape)
    translation[:] = rng.normal(0, 0.3, translation.shape)
    original = [value.copy() for value in (roots, pose, translation)]
    new_roots, edited, new_translation, _ = smooth_upper_body(roots, pose, translation, fps=50)
    for actual, expected in zip((roots, pose, translation), original, strict=True):
        assert np.array_equal(actual, expected)
    assert np.array_equal(new_roots, roots)
    assert np.array_equal(new_translation, translation)
    untouched = [index - 1 for index, name in enumerate(SMPLX_BODY22_NAMES) if index and name not in UPPER_BODY_JOINTS]
    assert np.array_equal(edited[:, untouched], pose[:, untouched])
    assert np.array_equal(edited[[0, -1]], pose[[0, -1]])
    assert not np.shares_memory(edited, pose)


@pytest.mark.parametrize("window", [3, 5, 9])
def test_fast_constant_turn_is_preserved_including_axis_angle_wrap(window):
    roots, pose, translation = _motion()
    pose[:, INDEX["neck"] - 1, 2] = np.radians(170 + np.arange(60) * 5.0)
    pose[:] = axis_angle_from_rotations(rodrigues(pose))
    _, edited, _, report = smooth_upper_body(roots, pose, translation, fps=50, window=window)
    assert np.array_equal(edited, pose)
    assert report["local_deviation_deg"]["max"] < 1e-10


def test_intentional_sharp_beat_cannot_be_changed_beyond_the_requested_cap():
    roots, pose, translation = _motion()
    pose[30, INDEX["neck"] - 1, 0] = np.radians(30.0)
    _, edited, _, report = smooth_upper_body(roots, pose, translation, fps=50, max_deviation_deg=1.0)
    assert np.degrees(edited[30, INDEX["neck"] - 1, 0]) == pytest.approx(29.0)
    assert report["local_deviation_deg"]["max"] <= 1.0 + 1e-10
    assert report["limited_joint_frames"] == 3


def test_world_deviation_reports_accumulation_along_the_spine():
    roots, pose, translation = _motion()
    for name in ("spine1", "spine2", "spine3", "neck", "head"):
        pose[30, INDEX[name] - 1, 0] = np.radians(10.0)
    _, _, _, report = smooth_upper_body(roots, pose, translation, fps=50, max_deviation_deg=1.0)
    assert report["local_deviation_deg"]["max"] == pytest.approx(1.0)
    assert report["world_deviation_deg"]["max"] == pytest.approx(5.0)
    assert report["worst_world_deviation_joint"] == "head"


@pytest.mark.parametrize("options", [{"window": 1}, {"strength": 0.0}, {"max_deviation_deg": 0.0}])
def test_disabling_the_filter_returns_an_exact_copy(options):
    rng = np.random.default_rng(13)
    roots, pose, translation = _motion()
    pose[:] = rng.normal(0, 0.5, pose.shape)
    _, edited, _, report = smooth_upper_body(roots, pose, translation, fps=50, **options)
    assert np.array_equal(edited, pose)
    assert not np.shares_memory(edited, pose)
    assert report["enabled"] is False


def test_flat_float32_pose_preserves_layout_and_unmodified_data():
    roots, pose, translation = _motion()
    pose[:, INDEX["spine1"] - 1, 0] = np.radians(np.tile([1., -1.], 30))
    pose = pose.astype(np.float32).reshape(60, 63)
    _, edited, _, _ = smooth_upper_body(roots, pose, translation, fps=50)
    assert edited.dtype == np.float32
    assert edited.shape == (60, 63)
    assert np.array_equal(edited[[0, -1]], pose[[0, -1]])


def test_reposed_joints_match_fk_and_leave_lower_body_exactly_unchanged():
    rng = np.random.default_rng(5)
    roots, pose, translation = _motion()
    roots[:] = rng.normal(0, 0.1, roots.shape)
    pose[:] = rng.normal(0, 0.1, pose.shape)
    translation[:, 0] = np.arange(len(roots)) * 0.03
    offsets = rng.normal(0, 0.2, (22, 3))
    joints = _fk(roots, pose, translation, offsets)
    _, edited, _, _ = smooth_upper_body(roots, pose, translation, fps=50)
    moved = repose_joints(roots, pose, edited, joints)
    expected = _fk(roots, edited, translation, offsets)
    assert np.allclose(moved, expected, atol=1e-12)
    lower = [INDEX[name] for name in ("pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
                                    "left_ankle", "right_ankle", "left_foot", "right_foot")]
    assert np.array_equal(moved[:, lower], joints[:, lower])
    assert not np.array_equal(moved[:, INDEX["left_wrist"]], joints[:, INDEX["left_wrist"]])


def test_reposing_preserves_measured_per_frame_bone_lengths_instead_of_refitting():
    rng = np.random.default_rng(11)
    roots, pose, translation = _motion()
    pose[:, INDEX["spine1"] - 1, 0] = np.radians(np.tile([1., -1.], 30))
    joints = rng.normal(0, 0.1, (60, 22, 3))
    before = joints.copy()
    _, edited, _, _ = smooth_upper_body(roots, pose, translation, fps=50)
    moved = repose_joints(roots, pose, edited, joints)
    assert np.array_equal(joints, before)
    for joint, parent in enumerate(SMPLX_BODY22_PARENTS):
        if parent >= 0:
            old_length = np.linalg.norm(joints[:, joint] - joints[:, parent], axis=-1)
            new_length = np.linalg.norm(moved[:, joint] - moved[:, parent], axis=-1)
            assert np.allclose(old_length, new_length, atol=1e-12)


@pytest.mark.parametrize("options", [{"window": 0}, {"window": 4}, {"strength": 1.1},
                                     {"max_deviation_deg": -1}, {"fps": 0}])
def test_invalid_filter_settings_are_rejected(options):
    roots, pose, translation = _motion()
    arguments = {"fps": 50, **options}
    with pytest.raises(ValueError):
        smooth_upper_body(roots, pose, translation, **arguments)
