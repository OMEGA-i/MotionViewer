"""Pure NumPy coverage for the retarget solver interface."""

from __future__ import annotations

import math

import numpy as np
import pytest

from motionviewer.blender.retarget.solver import GROUND_CLEARANCE_M, RetargetDefinition, solve_retarget


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        ((cosine, -sine, 0, 0), (sine, cosine, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), dtype=np.float64
    )


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        ((1, 0, 0, 0), (0, cosine, -sine, 0), (0, sine, cosine, 0), (0, 0, 0, 1)), dtype=np.float64
    )


def _translated(x: float, y: float, z: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    return matrix


def test_solver_transfers_rest_delta_without_blender() -> None:
    definition = RetargetDefinition(
        source_names=("pelvis",),
        target_names=("Hips",),
        parent_indices=np.array([-1]),
        rest_delta=np.eye(4)[None, ...],
        target_rest_local=np.eye(4)[None, ...],
    )
    source = np.stack((_rotation_z(0.0), _rotation_z(math.pi / 2)))[:, None]
    result = solve_retarget(definition, source, np.zeros((2, 3)), mode="direct")
    assert np.allclose(result.local_quaternions_wxyz[0, 0], (1, 0, 0, 0))
    assert abs(result.local_quaternions_wxyz[1, 0, 3]) > 0.7


def test_solver_preserves_recorded_object_basis_without_baking() -> None:
    basis = _rotation_x(math.pi / 2)
    child_local = _translated(0.0, 0.0, 0.5)
    definition = RetargetDefinition(
        source_names=("pelvis", "spine1"),
        target_names=("Hips", "Spine"),
        parent_indices=np.array((-1, 0)),
        rest_delta=np.stack((basis, basis)),
        target_rest_local=np.stack((basis, child_local)),
    )
    source = np.repeat(np.eye(4)[None, None], 2, axis=1)

    result = solve_retarget(definition, source, np.zeros((1, 3)), mode="direct")

    assert np.allclose(result.local_quaternions_wxyz[0], ((1, 0, 0, 0), (1, 0, 0, 0)))
    assert np.allclose(result.target_matrices[0, 0, :3, :3], basis[:3, :3])
    assert np.allclose(result.target_matrices[0, 1, :3, 3], (0.0, -0.5, 0.0), atol=1e-7)


def test_solver_keeps_quaternion_sign_continuous() -> None:
    definition = RetargetDefinition(
        source_names=("pelvis",),
        target_names=("Hips",),
        parent_indices=np.array([-1]),
        rest_delta=np.eye(4)[None, ...],
        target_rest_local=np.eye(4)[None, ...],
    )
    source = np.stack((_rotation_z(math.pi), _rotation_z(-math.pi)))[:, None]
    result = solve_retarget(definition, source, np.zeros((2, 3)))
    assert float(np.dot(result.local_quaternions_wxyz[0, 0], result.local_quaternions_wxyz[1, 0])) >= 0.0


def test_quality_mode_preserves_source_angular_velocity_without_contact_constraints() -> None:
    definition = RetargetDefinition(
        source_names=("pelvis",),
        target_names=("Hips",),
        parent_indices=np.array([-1]),
        rest_delta=np.eye(4)[None, ...],
        target_rest_local=np.eye(4)[None, ...],
    )
    source = np.stack((_rotation_z(0.0), _rotation_z(math.pi / 2)))[:, None]
    direct = solve_retarget(definition, source, np.zeros((2, 3)), mode="direct")
    quality = solve_retarget(definition, source, np.zeros((2, 3)), mode="quality")

    def step(result) -> float:
        left, right = result.local_quaternions_wxyz[:, 0]
        return math.degrees(2.0 * math.acos(np.clip(abs(float(np.dot(left, right))), -1.0, 1.0)))

    assert step(direct) == pytest.approx(90.0)
    assert step(quality) == pytest.approx(90.0)


def test_solver_ignores_uncalibrated_contact_bones() -> None:
    definition = RetargetDefinition(
        source_names=("left_foot",),
        target_names=("LeftToeBase",),
        parent_indices=np.array([-1]),
        rest_delta=np.eye(4)[None, ...],
        target_rest_local=np.eye(4)[None, ...],
    )
    source = np.repeat(np.eye(4)[None, None], 3, axis=0)
    result = solve_retarget(
        definition,
        source,
        np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        contact_intervals={"left_foot": [(0, 3)]},
    )
    assert np.allclose(result.root_locations[:, 0], (0.0, 1.0, 2.0))


def test_quality_mode_preserves_baseline_limb_direction_without_changing_lengths() -> None:
    rest_local = np.stack((_translated(0.0, 0.0), _translated(0.6, 0.0), _translated(0.6, 0.0)))
    definition = RetargetDefinition(
        source_names=("left_hip", "left_knee", "left_ankle"),
        target_names=("LeftUpLeg", "LeftLeg", "LeftFoot"),
        parent_indices=np.array((-1, 0, 1)),
        rest_delta=np.repeat(np.eye(4)[None], 3, axis=0),
        target_rest_local=rest_local,
    )
    source = np.stack((_translated(0.0, 0.0), _translated(0.3, 0.4), _translated(0.6, 0.8)))[None]
    roots = np.zeros((1, 3))

    direct = solve_retarget(definition, source, roots, mode="direct")
    quality = solve_retarget(definition, source, roots, mode="quality")

    assert np.allclose(direct.target_matrices[0, 2, :2, 3], (1.2, 0.0))
    assert np.allclose(
        quality.target_matrices[0, 2, :2, 3],
        direct.target_matrices[0, 2, :2, 3],
        atol=1e-6,
    )
    positions = quality.target_matrices[0, :, :3, 3]
    assert np.allclose(np.linalg.norm(np.diff(positions, axis=0), axis=1), (0.6, 0.6))


def test_quality_mode_uses_transferred_baseline_as_the_pole_hint() -> None:
    rest_local = np.stack((_translated(0.0, 0.0), _translated(0.6, 0.0), _translated(0.6, 0.0)))
    definition = RetargetDefinition(
        source_names=("left_hip", "left_knee", "left_ankle"),
        target_names=("LeftUpLeg", "LeftLeg", "LeftFoot"),
        parent_indices=np.array((-1, 0, 1)),
        rest_delta=np.repeat(np.eye(4)[None], 3, axis=0),
        target_rest_local=rest_local,
    )
    source = np.stack((_translated(0.0, 0.0), _translated(0.0, 0.6), _translated(0.8, 0.6)))[None]

    result = solve_retarget(definition, source, np.zeros((1, 3)), mode="quality")

    direct = solve_retarget(definition, source, np.zeros((1, 3)), mode="direct")
    assert np.allclose(
        result.target_matrices[0, 1, :3, 3],
        direct.target_matrices[0, 1, :3, 3],
        atol=3e-4,
    )


def test_quality_mode_rebuilds_baseline_limb_with_target_segment_lengths() -> None:
    """A differently proportioned avatar must not chase raw SMPL world positions."""
    definition = RetargetDefinition(
        source_names=("left_hip", "left_knee", "left_ankle"),
        target_names=("LeftUpLeg", "LeftLeg", "LeftFoot"),
        parent_indices=np.array((-1, 0, 1)),
        rest_delta=np.repeat(np.eye(4)[None], 3, axis=0),
        target_rest_local=np.stack((_translated(0.0, 0.0), _translated(0.4, 0.0), _translated(0.8, 0.0))),
    )
    source = np.stack((_translated(0.0, 0.0), _translated(0.3, 0.4), _translated(0.6, 0.8)))[None]

    result = solve_retarget(definition, source, np.zeros((1, 3)), mode="quality")

    positions = result.target_matrices[0, :, :3, 3]
    direct = solve_retarget(definition, source, np.zeros((1, 3)), mode="direct")
    expected_upper = direct.target_matrices[0, 1, :3, 3] - direct.target_matrices[0, 0, :3, 3]
    expected_lower = direct.target_matrices[0, 2, :3, 3] - direct.target_matrices[0, 1, :3, 3]
    expected_upper /= np.linalg.norm(expected_upper)
    expected_lower /= np.linalg.norm(expected_lower)
    assert np.allclose(positions[1] - positions[0], 0.4 * expected_upper, atol=3e-4)
    assert np.allclose(positions[2] - positions[1], 0.8 * expected_lower, atol=3e-4)


def test_quality_mode_locks_both_feet_during_double_support() -> None:
    names = (
        "pelvis",
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_foot",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_foot",
    )
    parents = np.array((-1, 0, 1, 2, 3, 0, 5, 6, 7))
    rest_local = np.stack(
        (
            _translated(0.0, 0.0),
            _translated(-0.5, 0.0),
            _translated(0.0, 0.6),
            _translated(0.0, 0.6),
            _translated(0.0, 0.2),
            _translated(0.5, 0.0),
            _translated(0.0, 0.6),
            _translated(0.0, 0.6),
            _translated(0.0, 0.2),
        )
    )
    definition = RetargetDefinition(
        source_names=names,
        target_names=names,
        parent_indices=parents,
        rest_delta=np.repeat(np.eye(4)[None], len(names), axis=0),
        target_rest_local=rest_local,
    )
    frame0 = np.stack(
        (
            _translated(0.0, 0.0),
            _translated(-0.5, 0.0),
            _translated(-0.5, 0.5),
            _translated(-0.5, 1.0),
            _translated(-0.5, 1.2),
            _translated(0.5, 0.0),
            _translated(0.5, 0.5),
            _translated(0.5, 1.0),
            _translated(0.5, 1.2),
        )
    )
    frame1 = frame0.copy()
    frame1[3:, 0, 3] = (-0.47, -0.47, 0.5, 0.5, 0.47, 0.47)
    source = np.stack((frame0, frame1))

    result = solve_retarget(
        definition,
        source,
        np.zeros((2, 3)),
        mode="quality",
        contact_intervals={"left_foot": [(0, 2)], "right_foot": [(0, 2)]},
    )

    for foot_index in (4, 8):
        world = result.target_matrices[:, foot_index, :3, 3] + result.root_locations
        assert np.linalg.norm(world[1] - world[0]) < 0.01
    for bone in range(len(names)):
        left, right = result.local_quaternions_wxyz[:, bone]
        step = math.degrees(2.0 * math.acos(np.clip(abs(float(np.dot(left, right))), -1.0, 1.0)))
        assert step <= 10.0001


def test_quality_mode_places_profiled_sole_support_on_the_ground() -> None:
    rest_local = np.stack(
        (_translated(0.0, 0.0), _translated(0.0, 0.5), _translated(0.0, 0.5), _translated(0.0, 0.2))
    )
    definition = RetargetDefinition(
        source_names=("left_hip", "left_knee", "left_ankle", "left_foot"),
        target_names=("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"),
        parent_indices=np.array((-1, 0, 1, 2)),
        rest_delta=np.repeat(np.eye(4)[None], 4, axis=0),
        target_rest_local=rest_local,
        contact_bone_indices={"left_foot": 2},
        contact_points_local={"left_foot": np.array(((0.0, 0.0, -0.2),))},
    )
    source = np.stack(
        (_translated(0.0, 0.0), _translated(0.0, 0.5), _translated(0.0, 1.0), _translated(0.0, 1.2))
    )[None]

    result = solve_retarget(
        definition,
        source,
        np.zeros((1, 3)),
        mode="quality",
        contact_intervals={"left_foot": [(0, 1)]},
    )

    anchor = result.target_matrices[0, 2] @ np.array((0.0, 0.0, -0.2, 1.0))
    assert np.isclose(float(anchor[2] + result.root_locations[0, 2]), GROUND_CLEARANCE_M)


def test_quality_mode_raises_penetrating_support_without_grounding_airborne_support() -> None:
    definition = RetargetDefinition(
        source_names=("left_foot", "right_foot"),
        target_names=("LeftFoot", "RightFoot"),
        parent_indices=np.array((-1, -1)),
        rest_delta=np.repeat(np.eye(4)[None], 2, axis=0),
        target_rest_local=np.stack((_translated(0.0, 0.0), _translated(1.0, 0.0, 1.0))),
        contact_bone_indices={"left_foot": 0, "right_foot": 1},
        contact_points_local={
            "left_foot": np.array(((0.0, 0.0, -0.2),)),
            "right_foot": np.array(((0.0, 0.0, 0.0),)),
        },
    )
    source = np.stack((_translated(0.0, 0.0), _translated(1.0, 0.0, 1.0)))[None]

    result = solve_retarget(definition, source, np.zeros((1, 3)), mode="quality")

    assert np.isclose(result.root_locations[0, 2], 0.2 + GROUND_CLEARANCE_M)
    airborne = result.target_matrices[0, 1, 2, 3] + result.root_locations[0, 2]
    assert np.isclose(airborne, 1.2 + GROUND_CLEARANCE_M)


def test_quality_mode_restores_end_foot_orientation_after_limb_ik() -> None:
    rest_local = np.stack((_translated(0.0, 0.0), _translated(0.6, 0.0), _translated(0.6, 0.0)))
    definition = RetargetDefinition(
        source_names=("left_hip", "left_knee", "left_ankle"),
        target_names=("LeftUpLeg", "LeftLeg", "LeftFoot"),
        parent_indices=np.array((-1, 0, 1)),
        rest_delta=np.repeat(np.eye(4)[None], 3, axis=0),
        target_rest_local=rest_local,
    )
    ankle = _rotation_z(np.pi / 3)
    ankle[:3, 3] = (0.6, 0.8, 0.0)
    source = np.stack((_translated(0.0, 0.0), _translated(0.3, 0.4), ankle))[None]
    result = solve_retarget(definition, source, np.zeros((1, 3)), mode="quality")
    assert np.allclose(result.target_matrices[0, 2, :3, :3], source[0, 2, :3, :3], atol=1e-5)


def test_quality_mode_flattens_profiled_sole_during_contact() -> None:
    rest_local = np.stack(
        (
            _translated(0.0, 0.0),
            _translated(0.0, 0.5),
            _translated(0.0, 0.5),
            _translated(0.0, 0.2),
        )
    )
    definition = RetargetDefinition(
        source_names=("left_hip", "left_knee", "left_ankle", "left_foot"),
        target_names=("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"),
        parent_indices=np.array((-1, 0, 1, 2)),
        rest_delta=np.repeat(np.eye(4)[None], 4, axis=0),
        target_rest_local=rest_local,
        contact_bone_indices={"left_foot": 2},
        contact_points_local={"left_foot": np.array(((0.0, 0.0, -0.2), (0.1, 0.2, -0.2), (-0.1, 0.2, -0.2)))},
    )
    source = np.stack(
        (_translated(0.0, 0.0), _translated(0.0, 0.5), _translated(0.0, 1.0), _translated(0.0, 1.2))
    )[None]
    result = solve_retarget(
        definition,
        source,
        np.zeros((1, 3)),
        contact_intervals={"left_foot": [(0, 1)]},
    )
    matrix = result.target_matrices[0, 2]
    world = definition.contact_points_local["left_foot"] @ matrix[:3, :3].T
    _, _, vh = np.linalg.svd(world - np.mean(world, axis=0), full_matrices=False)
    assert abs(float(vh[-1, 2])) > 0.999


def test_quality_mode_locks_all_profiled_sole_anchors_after_foot_rotation() -> None:
    """A contact lock must not preserve only the sole centroid.

    The second frame rolls the ankle while the foot remains in contact.  The
    four support anchors should stay on their frame-zero world positions after
    the contact projection and grounding pass.
    """
    rest_local = np.stack(
        (
            _translated(0.0, 0.0),
            _translated(0.0, 0.5),
            _translated(0.0, 0.5),
            _translated(0.0, 0.2),
        )
    )
    definition = RetargetDefinition(
        source_names=("left_hip", "left_knee", "left_ankle", "left_foot"),
        target_names=("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"),
        parent_indices=np.array((-1, 0, 1, 2)),
        rest_delta=np.repeat(np.eye(4)[None], 4, axis=0),
        target_rest_local=rest_local,
        contact_bone_indices={"left_foot": 2},
        contact_points_local={
            "left_foot": np.array(
                (
                    (-0.14, -0.10, -0.20),
                    (0.14, -0.10, -0.20),
                    (-0.14, 0.20, -0.20),
                    (0.14, 0.20, -0.20),
                )
            )
        },
    )
    frame0 = np.stack(
        (
            _translated(0.0, 0.0),
            _translated(0.0, 0.5),
            _translated(0.0, 1.0),
            _translated(0.0, 1.2),
        )
    )
    frame1 = frame0.copy()
    frame1[2] = _rotation_x(np.deg2rad(25.0))
    frame1[2, :3, 3] = (0.0, 1.0, 0.0)
    frame1[3, :3, 3] = (0.0, 1.2, 0.0)
    result = solve_retarget(
        definition,
        np.stack((frame0, frame1)),
        np.zeros((2, 3)),
        contact_intervals={"left_foot": [(0, 2)]},
    )
    anchors = np.asarray(definition.contact_points_local["left_foot"])
    world = np.stack(
        [
            anchors @ result.target_matrices[frame, 2, :3, :3].T
            + result.target_matrices[frame, 2, :3, 3]
            + result.root_locations[frame]
            for frame in range(2)
        ]
    )
    assert np.max(np.linalg.norm(world[1, :, :2] - world[0, :, :2], axis=1)) < 0.005
    assert np.max(np.abs(world[1, :, 2] - world[0, :, 2])) < 0.005
