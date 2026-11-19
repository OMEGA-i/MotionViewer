from __future__ import annotations

import numpy as np
import pytest

from motionviewer.core.prop_fit import (
    box_signed_distance,
    box_triangle_intersections,
    finger_chain_points,
    fit_finger_cylinder,
    fit_two_hand_box,
    plan_box_elbows,
)


def test_box_distance_detects_faces_edges_and_interior():
    points = np.array([[0, 0, 0], [0.5, 0, 0], [0.6, 0, 0], [0.6, 1.2, 0]])
    np.testing.assert_allclose(box_signed_distance(points, np.array([1, 2, 3])),
                               [-0.5, 0, 0.1, np.hypot(0.1, 0.2)])


def test_box_triangle_intersection_catches_crossing_with_all_vertices_outside():
    triangles = np.array([[[-2, 0, 0], [2, 0, 0], [0, 2, 0]],
                          [[2, 2, 2], [3, 2, 2], [2, 3, 2]],
                          [[-.2, -.2, .5], [.2, -.2, .5], [0, .2, .5]],
                          [[.8, .3, 0], [.3, .8, 0], [.8, .8, 0]]])
    np.testing.assert_array_equal(box_triangle_intersections(triangles, [1, 1, 1], inset=.0001),
                                  [True, False, False, False])


def test_box_elbow_plan_preserves_lengths_and_bounded_temporal_changes():
    shoulders = np.tile([.3, -.3, .2], (12, 1))
    elbows = np.tile([.05, -.05, -.02], (12, 1))
    wrists = np.tile([.18, -.13, .1], (12, 1))
    result, angles = plan_box_elbows(shoulders, elbows, wrists, wrists, np.zeros((12, 3)),
                                    np.tile(np.eye(3), (12, 1, 1)), np.array([.32, .22, .15]))
    np.testing.assert_allclose(np.linalg.norm(result - shoulders, axis=1), np.linalg.norm(elbows - shoulders, axis=1))
    np.testing.assert_allclose(np.linalg.norm(result - wrists, axis=1), np.linalg.norm(elbows - wrists, axis=1))
    assert np.abs(angles).max() <= 30
    assert np.abs(np.diff(angles)).max() <= 4
    assert np.any(angles != 0)


def carry_positions(gaps):
    gaps = np.asarray(gaps, dtype=float)
    left = np.column_stack((-gaps / 2, np.zeros_like(gaps), np.ones_like(gaps)))
    right = left.copy()
    right[:, 0] *= -1
    heads = np.tile((0, -0.3, 1.6), (len(gaps), 1))
    torsos = np.tile((0, -0.3, 1.0), (len(gaps), 1))
    return left, right, heads, torsos


def test_two_hand_box_follows_both_palms_as_one_rigid_object():
    positions = carry_positions([0.32] * 10)
    translation = np.column_stack((np.arange(10) * 0.1, np.arange(10) * 0.03, np.zeros(10)))
    moved = [value + translation for value in positions]
    fit = fit_two_hand_box(*moved, side=0.32, depth=0.24, height=0.20)
    np.testing.assert_allclose(fit.centres - translation, np.tile((0, 0.11, 0.95), (10, 1)))
    np.testing.assert_allclose(fit.rotations @ np.transpose(fit.rotations, (0, 2, 1)),
                               np.tile(np.eye(3), (10, 1, 1)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(fit.rotations), 1)
    assert fit.report["side_contact_error_per_hand_mm"]["max"] == 0
    assert fit.report["source_motion_modified"] is False


def test_changing_hand_span_is_reported_instead_of_scaling_box_or_changing_motion():
    positions = carry_positions([0.3, 0.3, 0.3, 0.5])
    originals = [value.copy() for value in positions]
    fit = fit_two_hand_box(*positions, fit_range=(0, 3))
    assert fit.size[0] == pytest.approx(0.3)
    assert fit.report["side_contact_error_per_hand_mm"]["max"] == pytest.approx(100)
    assert fit.report["frames_over_30mm_contact_error"] == 1
    for before, after in zip(originals, positions, strict=True):
        np.testing.assert_array_equal(before, after)


def test_bottom_box_uses_fixed_height_with_base_above_palms():
    positions = carry_positions([0.3] * 4)
    fit = fit_two_hand_box(*positions, grip_mode="bottom", height=0.12)
    np.testing.assert_allclose(fit.centres[:, 2] - fit.size[2] / 2, 1.005)
    assert fit.report["head_point_box_distance_mm"]["min"] > 80


def test_torso_noise_cannot_flip_an_unmoving_two_hand_box():
    left, right, heads, torsos = carry_positions([0.32] * 4)
    torsos[:, 1] = (0.001, -0.001, 0.001, -0.001)
    fit = fit_two_hand_box(left, right, heads, torsos, depth=0.24)
    np.testing.assert_allclose(fit.centres, np.repeat(fit.centres[:1], 4, axis=0))
    np.testing.assert_allclose(fit.rotations, np.repeat(fit.rotations[:1], 4, axis=0))


def test_bottom_support_inside_wide_box_has_only_vertical_contact_gap():
    fit = fit_two_hand_box(*carry_positions([0.32] * 4), grip_mode="bottom", side=0.6)
    assert fit.report["contact_error_per_hand_mm"]["max"] == pytest.approx(5)
    assert fit.report["frames_over_30mm_contact_error"] == 0


@pytest.mark.parametrize("kwargs", [{"fit_range": (2, 9)}, {"height": 0}, {"grip_mode": "magic"}])
def test_invalid_box_fit_fails_before_emitting_geometry(kwargs):
    with pytest.raises(ValueError):
        fit_two_hand_box(*carry_positions([0.3] * 4), **kwargs)


def test_coincident_hands_do_not_invent_orientation():
    with pytest.raises(ValueError, match="30 mm"):
        fit_two_hand_box(*carry_positions([0.01] * 4))


def finger_chain():
    lengths = np.array((0.04, 0.025, 0.02))
    offsets = np.tile(np.eye(4), (3, 1, 1))
    offsets[1, 1, 3], offsets[2, 1, 3] = lengths[:2]
    return np.eye(4), offsets, lengths


def test_finger_fk_obeys_joint_rotation_and_lengths():
    eulers = np.zeros((3, 3))
    eulers[0, 0] = np.pi / 2
    points = finger_chain_points(*finger_chain(), eulers)
    np.testing.assert_allclose(points[-1], (0, 0, 0.085), atol=1e-12)


def test_bounded_finger_fit_uses_total_angles_and_can_touch_reachable_shaft():
    # The supplied relaxed pose already has flexion. Returned values are total
    # local joint angles; adding the relaxed pose again would violate this contract.
    reference = np.zeros((3, 3))
    reference[:, 0] = np.radians((16, 27, 19))
    fit = fit_finger_cylinder(*finger_chain(), reference, np.array((0, 0, 0.105)),
                             np.array((1, 0, 0)), 0.02)
    assert np.all(np.asarray(fit["degrees"]) >= 0)
    assert np.all(np.asarray(fit["degrees"]) <= (90, 110, 90))
    assert fit["degrees"][2] <= fit["degrees"][1] + 5
    assert fit["tip_surface_error_mm"] < 3
    assert fit["max_bone_sample_penetration_mm"] < 1


def test_unreachable_shaft_returns_residual_instead_of_exceeding_joint_limits():
    fit = fit_finger_cylinder(*finger_chain(), np.zeros((3, 3)), np.array((0, 0, 1.0)),
                             np.array((1, 0, 0)), 0.02)
    assert fit["tip_surface_error_mm"] > 800
    assert np.all(np.asarray(fit["degrees"]) <= (90, 110, 90))
