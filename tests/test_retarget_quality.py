"""Unit coverage for Mixamo retarget quality primitives."""

from __future__ import annotations

import numpy as np

from motionviewer.blender.retarget._quality import (
    detect_foot_contact_frames,
    morphology_aware_joint_targets,
    quaternion_continuity,
    smooth_root_trajectory,
)


def test_detects_grounded_stationary_feet() -> None:
    positions = np.zeros((10, 2, 3), dtype=np.float32)
    result = detect_foot_contact_frames(positions)
    assert result.contact_intervals == {"left_foot": [(0, 10)], "right_foot": [(0, 10)]}


def test_filters_short_contact() -> None:
    positions = np.zeros((10, 2, 3), dtype=np.float32)
    positions[:, :, 2] = 1.0
    positions[4:6, 0, 2] = 0.0
    result = detect_foot_contact_frames(positions, min_contact_frames=3)
    assert result.contact_intervals["left_foot"] == []


def test_slow_sliding_foot_does_not_create_repeated_contact_locks() -> None:
    positions = np.zeros((10, 2, 3), dtype=np.float32)
    positions[:, 0, 0] = np.arange(10, dtype=np.float32) * 0.002
    result = detect_foot_contact_frames(positions)
    assert result.contact_intervals["left_foot"] == [(0, 3)]


def test_root_smoothing_clamps_discontinuities() -> None:
    root = np.zeros((12, 3), dtype=np.float32)
    root[6:, 0] = 1.0
    result = smooth_root_trajectory(root, window=5, max_delta=0.1)
    assert float(np.max(np.linalg.norm(np.diff(result, axis=0), axis=1))) <= 0.10001


def test_quaternion_signs_are_continuous() -> None:
    values = np.array([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]], dtype=np.float64)
    result = quaternion_continuity(values)
    assert np.allclose(result[0], result[1])


def test_morphology_aware_targets_keep_target_rest_proportions() -> None:
    source_pose = np.array(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)))
    source_rest = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    target_rest = np.array(((0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (1.2, 0.0, 0.0)))

    result = morphology_aware_joint_targets(
        source_pose,
        source_rest,
        target_rest,
        np.array((-1, 0, 1)),
        direction_fit_mask=np.array((False, True, True)),
    )

    assert np.allclose(result, ((0.0, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 1.2, 0.0)))
