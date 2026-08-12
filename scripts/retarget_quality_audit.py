"""Measure full-motion FBX retarget quality inside Blender.

Run with:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python \
    scripts/retarget_quality_audit.py -- --motion data/examples/.../omegamotiongpt.smplx.npz \
    --asset assets/fbx/iron.fbx --output /tmp/retarget_quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--asset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("quality", "direct"), default="quality")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def _world_head(armature: Any, bone_name: str) -> np.ndarray:
    return np.asarray(
        (armature.matrix_world @ armature.pose.bones[bone_name].head).to_tuple(), dtype=np.float64
    )


def _profiled_support_world(
    armature: Any,
    rig_profile: dict[str, Any] | None,
    foot_name: str,
    fallback_bone: str,
) -> np.ndarray:
    points = _profiled_support_points_world(armature, rig_profile, foot_name, fallback_bone)
    return np.array((float(np.mean(points[:, 0])), float(np.mean(points[:, 1])), float(np.min(points[:, 2]))))


def _profiled_support_points_world(
    armature: Any,
    rig_profile: dict[str, Any] | None,
    foot_name: str,
    fallback_bone: str,
) -> np.ndarray:
    support = (rig_profile or {}).get("sole_support_points", {}).get(foot_name)
    if not support:
        return _world_head(armature, fallback_bone)[None, :]
    from mathutils import Vector  # type: ignore

    pose_bone = armature.pose.bones[support["bone"]]
    points = []
    for local in support.get("points_local_m", support["points_local"]):
        point = armature.matrix_world @ pose_bone.matrix @ Vector((*local, 1.0))
        points.append(np.asarray(tuple(point[:3]), dtype=np.float64))
    return np.stack(points)


def _profiled_sole_geometry_world(
    armature: Any,
    rig_profile: dict[str, Any] | None,
    foot_name: str,
    fallback_bone: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centroid, upward normal, and forward heading for audit."""
    support = (rig_profile or {}).get("sole_support_points", {}).get(foot_name)
    if not support:
        return _world_head(armature, fallback_bone), np.array((0.0, 0.0, 1.0)), np.array((0.0, 1.0, 0.0))
    from mathutils import Vector  # type: ignore

    pose_bone = armature.pose.bones[support["bone"]]
    points = [
        np.asarray(
            tuple((armature.matrix_world @ pose_bone.matrix @ Vector((*local, 1.0)))[:3]), dtype=np.float64
        )
        for local in support.get("points_local_m", support["points_local"])
    ]
    values = np.stack(points)
    centroid = np.mean(values, axis=0)
    centered = values - centroid
    if len(values) >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0.0:
            normal = -normal
    else:
        normal = np.array((0.0, 0.0, 1.0))
    # Calibration stores the anchors in semantic heel, toe, medial, lateral
    # order.  Do not rediscover "forward" from world Y extrema: a turning
    # character can make that choice jump between unrelated support points.
    forward = values[1] - values[0] if len(values) >= 2 else np.array((0.0, 1.0, 0.0))
    forward[2] = 0.0
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    return centroid, normal, forward


def _profiled_sole_rotation_world(
    armature: Any,
    rig_profile: dict[str, Any] | None,
    foot_name: str,
    fallback_bone: str,
) -> np.ndarray:
    """Return the calibrated virtual sole frame in world coordinates.

    A Mixamo foot bone's quaternion is not a stable cross-asset measure:
    imported bone roll is part of that quaternion.  The calibrated semantic
    sole frame is the common quantity that can be compared instead.
    """
    support = (rig_profile or {}).get("sole_support_points", {}).get(foot_name)
    if not support:
        pose = armature.matrix_world @ armature.pose.bones[fallback_bone].matrix
        return np.asarray(pose.to_3x3(), dtype=np.float64)
    pose = armature.matrix_world @ armature.pose.bones[support["bone"]].matrix
    rotation = np.asarray(pose.to_3x3(), dtype=np.float64)
    local = np.column_stack(
        (
            np.asarray(support["sole_lateral_local"], dtype=np.float64),
            np.asarray(support["sole_forward_local"], dtype=np.float64),
            np.asarray(support["sole_normal_local"], dtype=np.float64),
        )
    )
    world = rotation @ local
    # The profile is measured from skin vertices and can carry tiny numerical
    # non-orthogonality.  Project it back onto SO(3) before geodesic metrics.
    left, _, right = np.linalg.svd(world)
    return left @ np.diag((1.0, 1.0, np.linalg.det(left @ right))) @ right


def _rotation_error_degrees(reference: np.ndarray, actual: np.ndarray) -> float:
    relative = reference.T @ actual
    cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _angle_degrees(
    start: np.ndarray, end: np.ndarray, target_start: np.ndarray, target_end: np.ndarray
) -> float:
    source_direction = end - start
    target_direction = target_end - target_start
    source_norm = float(np.linalg.norm(source_direction))
    target_norm = float(np.linalg.norm(target_direction))
    if source_norm < 1e-8 or target_norm < 1e-8:
        return float("nan")
    cosine = float(
        np.clip(np.dot(source_direction, target_direction) / (source_norm * target_norm), -1.0, 1.0)
    )
    return float(np.degrees(np.arccos(cosine)))


def _angular_steps_degrees(quaternions_wxyz: np.ndarray) -> np.ndarray:
    if len(quaternions_wxyz) < 2:
        return np.zeros((0,), dtype=np.float64)
    values = quaternions_wxyz / np.maximum(np.linalg.norm(quaternions_wxyz, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(np.abs(np.sum(values[1:] * values[:-1], axis=1)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(cosine))


def _metric_for_asset(asset_path: Path, motion_path: Path, *, mode: str = "quality") -> dict[str, Any]:
    import bpy  # type: ignore
    from bl_ext.user_default.smplx_blender_addon.utils.pose import set_pose_from_rodrigues  # type: ignore

    from motionviewer.blender.retarget._ground import _mesh_lowest_z
    from motionviewer.blender.retarget._quality import (
        detect_foot_contact_frames,
    )
    from motionviewer.blender.retarget._resolve import resolve_bone_mapping
    from motionviewer.blender.retarget.calibration import inspect_mixamo_rig
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import clear_scene
    from motionviewer.blender.smplx_mesh import source_transl_to_blender
    from motionviewer.core.smplx_actor import BODY_POSE_BONES

    clear_scene()
    label = f"retarget_audit_{asset_path.stem}"
    actor = create_fbx_actor_from_npz(
        motion_path,
        label=label,
        fbx_path=asset_path,
        bone_map="auto",
        retarget_mode=mode,
    )
    # The direct transfer is the target-space reference for directional
    # metrics.  Comparing an FBX limb vector to raw SMPL-X world axes is a
    # category error whenever the assets have different object bases or rolls.
    baseline_actor = create_fbx_actor_from_npz(
        motion_path,
        label=f"{label}_direct_baseline",
        fbx_path=asset_path,
        bone_map="auto",
        retarget_mode="direct",
    )
    source = bpy.data.objects[f"{label}_SMPLX_Driver"]
    target = actor.armature
    baseline_target = baseline_actor.armature
    raw_solver_rotations = target.get("motionviewer_solver_first_frame_global_rotations", "{}")
    solver_first_frame_rotations = (
        json.loads(raw_solver_rotations) if isinstance(raw_solver_rotations, str) else {}
    )
    raw_baseline_solver_rotations = baseline_target.get(
        "motionviewer_solver_first_frame_global_rotations", "{}"
    )
    baseline_solver_first_frame_rotations = (
        json.loads(raw_baseline_solver_rotations) if isinstance(raw_baseline_solver_rotations, str) else {}
    )
    mapping = resolve_bone_mapping("auto", fbx_armature=target).smplx_to_fbx
    inspection = inspect_mixamo_rig(target)
    raw_profile = target.get("motionviewer_mixamo_profile")
    rig_profile = json.loads(raw_profile) if isinstance(raw_profile, str) else None
    profile_rest = (rig_profile or {}).get("rest_matrices", {})
    profile_lengths = (rig_profile or {}).get("bone_lengths", {})
    rest_z = [float(matrix[2][3]) for matrix in profile_rest.values() if matrix]
    head_top_z = float(profile_rest.get("Head", [[0] * 4] * 3)[2][3]) + float(
        profile_lengths.get("Head", 0.0)
    )
    target_height = max(rest_z + [head_top_z]) - min(rest_z) if rest_z else 0.0
    source_rest_world = {name: source.matrix_world @ source.data.bones[name].matrix_local for name in mapping}
    source_rest_z = [float(matrix.translation.z) for matrix in source_rest_world.values()]
    source_head_top_z = float(source_rest_world["head"].translation.z) + float(
        source.data.bones["head"].length
    )
    source_height = max(source_rest_z + [source_head_top_z]) - min(source_rest_z)
    source_to_target_scale = target_height / source_height if source_height > 1e-8 else 1.0
    actual_to_canonical = {
        actual: canonical for canonical, actual in (rig_profile or {}).get("bone_map", {}).items()
    }
    mapped_parent: dict[str, str | None] = {}
    for source_name in mapping:
        parent = source.data.bones[source_name].parent
        while parent is not None and parent.name not in mapping:
            parent = parent.parent
        mapped_parent[source_name] = None if parent is None else parent.name
    ordered_source_names: list[str] = []

    def visit_source(name: str) -> None:
        if name in ordered_source_names:
            return
        parent = mapped_parent[name]
        if parent is not None:
            visit_source(parent)
        ordered_source_names.append(name)

    for source_name in mapping:
        visit_source(source_name)
    ordered_index = {name: index for index, name in enumerate(ordered_source_names)}
    action = getattr(getattr(target, "animation_data", None), "action", None)
    location_key_counts = {
        str(curve.array_index): len(curve.keyframe_points)
        for curve in (action.fcurves if action is not None else [])
        if curve.data_path == "location"
    }
    with np.load(motion_path, allow_pickle=False) as payload:
        global_orient = np.asarray(payload["global_orient"], dtype=np.float32)
        body_pose = np.asarray(payload["body_pose"], dtype=np.float32).reshape(global_orient.shape[0], 21, 3)
        transl = np.asarray(payload["transl"], dtype=np.float32)

    segment_pairs = {
        "left_upper_arm": ("left_shoulder", "left_elbow"),
        "left_forearm": ("left_elbow", "left_wrist"),
        "right_upper_arm": ("right_shoulder", "right_elbow"),
        "right_forearm": ("right_elbow", "right_wrist"),
        "left_upper_leg": ("left_hip", "left_knee"),
        "left_lower_leg": ("left_knee", "left_ankle"),
        "right_upper_leg": ("right_hip", "right_knee"),
        "right_lower_leg": ("right_knee", "right_ankle"),
    }
    angle_samples: dict[str, list[float]] = {name: [] for name in segment_pairs}
    endpoint_samples: dict[str, list[float]] = {
        name: [] for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }
    raw_ankle_orientation_samples: dict[str, list[float]] = {
        name: [] for name in ("left_ankle", "right_ankle")
    }
    virtual_sole_rotations: dict[str, list[np.ndarray]] = {"left_foot": [], "right_foot": []}
    source_feet = np.zeros((len(global_orient), 2, 3), dtype=np.float32)
    target_feet: dict[str, list[np.ndarray]] = {"left_foot": [], "right_foot": []}
    target_anchor_points: dict[str, list[np.ndarray]] = {"left_foot": [], "right_foot": []}
    target_sole_normals: dict[str, list[np.ndarray]] = {"left_foot": [], "right_foot": []}
    target_sole_headings: dict[str, list[np.ndarray]] = {"left_foot": [], "right_foot": []}
    lowest_mesh_z = float("inf")
    lowest_mesh_frame = None
    lowest_profiled_sole_z = float("inf")
    root_offset_z = None
    first_source_support_z: dict[str, float] = {}
    root_locations: list[np.ndarray] = []
    quaternion_samples: dict[str, list[np.ndarray]] = {name: [] for name in mapping.values()}
    source_quaternion_samples: dict[str, list[np.ndarray]] = {name: [] for name in mapping}
    bone_length_error: dict[str, list[float]] = {name: [] for name in mapping.values()}
    joint_error_samples: list[float] = []
    joint_error_by_frame: list[float] = []
    joint_error_by_name: dict[str, list[float]] = {name: [] for name in mapping}
    joint_limit_violations: list[dict[str, Any]] = []
    joint_limits = (rig_profile or {}).get("joint_limits", {})
    first_frame_contact_diagnostic: dict[str, dict[str, float]] = {}

    for index in range(len(global_orient)):
        frame = index + 1
        bpy.context.scene.frame_set(frame)
        root_locations.append(np.asarray(target.location[:], dtype=np.float64))
        if index == 0:
            root_offset_z = float(target.location.z - transl[index, 1])
        set_pose_from_rodrigues(source, "pelvis", global_orient[index])
        for bone_index, bone_name in enumerate(BODY_POSE_BONES):
            if bone_index < body_pose.shape[1] and bone_name in source.pose.bones:
                set_pose_from_rodrigues(source, bone_name, body_pose[index, bone_index])
        bpy.context.view_layer.update()

        for metric_name, (src_start, src_end) in segment_pairs.items():
            target_start = mapping[src_start]
            target_end = mapping[src_end]
            angle_samples[metric_name].append(
                _angle_degrees(
                    _world_head(baseline_target, target_start),
                    _world_head(baseline_target, target_end),
                    _world_head(target, target_start),
                    _world_head(target, target_end),
                )
            )

        for source_name in ("left_ankle", "right_ankle"):
            source_rotation = (source.matrix_world @ source.pose.bones[source_name].matrix).to_3x3()
            source_rest_rotation = source_rest_world[source_name].to_3x3()
            target_rest_rotation = np.asarray(
                profile_rest[actual_to_canonical[mapping[source_name]]], dtype=np.float64
            )[:3, :3]
            expected_rotation = (
                np.asarray(source_rotation)
                @ np.linalg.inv(np.asarray(source_rest_rotation))
                @ target_rest_rotation
            )
            actual_rotation = np.asarray(
                (target.matrix_world @ target.pose.bones[mapping[source_name]].matrix).to_3x3()
            )
            relative = expected_rotation.T @ actual_rotation
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            raw_ankle_orientation_samples[source_name].append(float(np.degrees(np.arccos(cosine))))

        for target_name in mapping.values():
            pose_bone = target.pose.bones[target_name]
            quat = pose_bone.rotation_quaternion
            quaternion_samples[target_name].append(
                np.asarray((quat.w, quat.x, quat.y, quat.z), dtype=np.float64)
            )
            rest_length = float(pose_bone.bone.length)
            posed_length = float((pose_bone.tail - pose_bone.head).length)
            bone_length_error[target_name].append(abs(posed_length - rest_length) / max(rest_length, 1e-8))
            canonical = actual_to_canonical.get(target_name)
            limit = joint_limits.get(canonical, {}).get("maximum_rotation_degrees")
            if limit is not None:
                normalized_w = min(1.0, abs(float(quat.w)) / max(float(quat.magnitude), 1e-12))
                angle = float(np.degrees(2.0 * np.arccos(normalized_w)))
                if angle > float(limit) + 1e-4:
                    joint_limit_violations.append(
                        {"bone": target_name, "frame": frame, "degrees": angle, "limit_degrees": float(limit)}
                    )
        for source_name in mapping:
            # Compare joint-local rotations on both rigs.  Armature-space
            # matrices include parent motion and would make a still knee look
            # fast whenever its hip rotates.
            quat = source.pose.bones[source_name].matrix_basis.to_quaternion()
            source_quaternion_samples[source_name].append(
                np.asarray((quat.w, quat.x, quat.y, quat.z), dtype=np.float64)
            )

        # All positional gates live in target space.  The direct action is
        # the calibrated FK baseline for this exact FBX object basis, scale
        # and bone roll; raw SMPL-X world positions are only an input, never a
        # valid cross-rig positional oracle.
        expected_positions = np.stack(
            [_world_head(baseline_target, mapping[name]) for name in ordered_source_names]
        )
        actual_positions = np.stack([_world_head(target, mapping[name]) for name in ordered_source_names])
        root_index = ordered_index["pelvis"]
        expected_positions -= expected_positions[root_index]
        actual_positions -= actual_positions[root_index]
        frame_joint_errors: list[float] = []
        for source_name in ordered_source_names:
            joint_index = ordered_index[source_name]
            error = float(np.linalg.norm(actual_positions[joint_index] - expected_positions[joint_index]))
            joint_error_samples.append(error)
            frame_joint_errors.append(error)
            joint_error_by_name[source_name].append(error)
            if source_name in endpoint_samples:
                endpoint_samples[source_name].append(error)
        joint_error_by_frame.append(float(np.mean(frame_joint_errors)))

        root_translation = source_transl_to_blender(transl[index])
        root_translation[:2] *= source_to_target_scale
        if index == 0:
            for support_name in ("left_ankle", "right_ankle", "left_foot", "right_foot"):
                first_source_support_z[support_name] = float(
                    _world_head(source, support_name)[2] + root_translation[2]
                )
        for foot_index, source_name in enumerate(("left_foot", "right_foot")):
            source_feet[index, foot_index] = _world_head(source, source_name) + root_translation
            support = _profiled_support_world(target, rig_profile, source_name, mapping[source_name])
            target_feet[source_name].append(support)
            target_anchor_points[source_name].append(
                _profiled_support_points_world(target, rig_profile, source_name, mapping[source_name])
            )
            _, sole_normal, sole_heading = _profiled_sole_geometry_world(
                target, rig_profile, source_name, mapping[source_name]
            )
            target_sole_normals[source_name].append(sole_normal)
            target_sole_headings[source_name].append(sole_heading)
            virtual_sole_rotations[source_name].append(
                _profiled_sole_rotation_world(target, rig_profile, source_name, mapping[source_name])
            )
            if index == 0:
                target_name = mapping["left_ankle" if source_name == "left_foot" else "right_ankle"]
                quality_local = target.pose.bones[target_name].matrix_basis.to_quaternion()
                baseline_local = baseline_target.pose.bones[target_name].matrix_basis.to_quaternion()
                baseline_sole = _profiled_sole_rotation_world(
                    baseline_target, rig_profile, source_name, mapping[source_name]
                )
                sole_profile = rig_profile["sole_support_points"][source_name]
                local_sole = np.column_stack(
                    (
                        np.asarray(sole_profile["sole_lateral_local"], dtype=np.float64),
                        np.asarray(sole_profile["sole_forward_local"], dtype=np.float64),
                        np.asarray(sole_profile["sole_normal_local"], dtype=np.float64),
                    )
                )
                solver_baseline_sole = (
                    np.asarray(baseline_solver_first_frame_rotations[target_name], dtype=np.float64).reshape(
                        3, 3
                    )
                    @ local_sole
                )
                solver_baseline_rotation = np.asarray(
                    baseline_solver_first_frame_rotations[target_name], dtype=np.float64
                ).reshape(3, 3)
                expected_forward = solver_baseline_rotation @ local_sole[:, 1]
                expected_forward[2] = 0.0
                expected_forward /= max(float(np.linalg.norm(expected_forward)), 1e-12)
                expected_normal = np.array((0.0, 0.0, 1.0))
                expected_lateral = np.cross(expected_forward, expected_normal)
                expected_contact_rotation = (
                    np.column_stack((expected_lateral, expected_forward, expected_normal)) @ local_sole.T
                )
                first_frame_contact_diagnostic[source_name] = {
                    "quality_local_rotation_degrees": float(
                        np.degrees(2.0 * np.arccos(np.clip(abs(quality_local.w), -1.0, 1.0)))
                    ),
                    "baseline_local_rotation_degrees": float(
                        np.degrees(2.0 * np.arccos(np.clip(abs(baseline_local.w), -1.0, 1.0)))
                    ),
                    "virtual_sole_delta_from_baseline_degrees": _rotation_error_degrees(
                        baseline_sole, virtual_sole_rotations[source_name][-1]
                    ),
                    "solver_to_blender_global_rotation_error_degrees": _rotation_error_degrees(
                        np.asarray(solver_first_frame_rotations[target_name], dtype=np.float64).reshape(3, 3),
                        np.asarray(
                            (target.matrix_world @ target.pose.bones[target_name].matrix).to_3x3(),
                            dtype=np.float64,
                        ),
                    ),
                    "baseline_solver_to_blender_global_rotation_error_degrees": _rotation_error_degrees(
                        np.asarray(
                            baseline_solver_first_frame_rotations[target_name], dtype=np.float64
                        ).reshape(3, 3),
                        np.asarray(
                            (
                                baseline_target.matrix_world @ baseline_target.pose.bones[target_name].matrix
                            ).to_3x3(),
                            dtype=np.float64,
                        ),
                    ),
                    "baseline_solver_to_blender_virtual_sole_error_degrees": _rotation_error_degrees(
                        solver_baseline_sole, baseline_sole
                    ),
                    "solver_contact_rotation_error_degrees": _rotation_error_degrees(
                        expected_contact_rotation,
                        np.asarray(solver_first_frame_rotations[target_name], dtype=np.float64).reshape(3, 3),
                    ),
                }
            lowest_profiled_sole_z = min(lowest_profiled_sole_z, float(support[2]))
        for mesh in actor.mesh_objects:
            mesh_z = _mesh_lowest_z(mesh)
            if mesh_z < lowest_mesh_z:
                lowest_mesh_z = mesh_z
                lowest_mesh_frame = frame

    contacts = detect_foot_contact_frames(source_feet)
    drift: dict[str, float] = {}
    source_drift: dict[str, float] = {}
    drift_by_interval: dict[str, list[float]] = {}
    drift_peaks: dict[str, dict[str, float | int]] = {}
    for foot_name, intervals in contacts.contact_intervals.items():
        samples = target_anchor_points[foot_name]
        interval_drifts: list[float] = []
        peak_value = 0.0
        peak_frame = 1
        for start, end in intervals:
            values = [
                float(np.linalg.norm(samples[frame][:, :2] - samples[start][:, :2], axis=1).max(initial=0.0))
                for frame in range(start, end)
            ]
            interval_peak = max(values, default=0.0)
            interval_drifts.append(interval_peak)
            if interval_peak >= peak_value and values:
                peak_value = interval_peak
                peak_frame = start + int(np.argmax(values)) + 1
        drift[foot_name] = max(interval_drifts, default=0.0)
        source_values = source_feet[:, 0 if foot_name == "left_foot" else 1]
        source_drift[foot_name] = max(
            (
                max(
                    (
                        float(np.linalg.norm(source_values[frame, :2] - source_values[start, :2]))
                        for frame in range(start, end)
                    ),
                    default=0.0,
                )
                for start, end in intervals
            ),
            default=0.0,
        )
        drift_by_interval[foot_name] = interval_drifts
        drift_peaks[foot_name] = {"frame": peak_frame, "meters": peak_value}

    target_steps = {
        target_name: _angular_steps_degrees(np.asarray(values))
        for target_name, values in quaternion_samples.items()
    }
    sole_tilt: dict[str, float] = {}
    heading_error: dict[str, float] = {}
    sole_orientation_error: dict[str, float] = {}
    for foot_name in ("left_foot", "right_foot"):
        normals = np.asarray(target_sole_normals[foot_name], dtype=np.float64)
        headings = np.asarray(target_sole_headings[foot_name], dtype=np.float64)
        contact_frames = [
            frame
            for start, end in contacts.contact_intervals.get(foot_name, ())
            for frame in range(start, end)
        ]
        if len(normals) and contact_frames:
            contact_normals = normals[contact_frames]
            sole_tilt[foot_name] = float(
                np.degrees(np.arccos(np.clip(contact_normals[:, 2], -1.0, 1.0))).max()
            )
            interval_heading_errors: list[float] = []
            interval_orientation_errors: list[float] = []
            rotations = virtual_sole_rotations[foot_name]
            for start, end in contacts.contact_intervals.get(foot_name, ()):
                reference_heading = headings[start]
                reference_rotation = rotations[start]
                interval_heading_errors.extend(
                    float(
                        np.degrees(np.arccos(np.clip(np.dot(headings[index], reference_heading), -1.0, 1.0)))
                    )
                    for index in range(start, end)
                )
                interval_orientation_errors.extend(
                    _rotation_error_degrees(reference_rotation, rotations[index])
                    for index in range(start, end)
                )
            heading_error[foot_name] = max(interval_heading_errors, default=0.0)
            sole_orientation_error[foot_name] = max(interval_orientation_errors, default=0.0)
        else:
            sole_tilt[foot_name] = 0.0
            heading_error[foot_name] = 0.0
            sole_orientation_error[foot_name] = 0.0
    source_steps = {
        source_name: _angular_steps_degrees(np.asarray(values))
        for source_name, values in source_quaternion_samples.items()
    }
    maximum_target_name = max(target_steps, key=lambda name: float(target_steps[name].max(initial=0.0)))
    maximum_target_values = target_steps[maximum_target_name]
    maximum_target_frame = int(np.argmax(maximum_target_values)) + 2 if len(maximum_target_values) else None
    angular_violations: list[dict[str, Any]] = []
    reverse_mapping = {target_name: source_name for source_name, target_name in mapping.items()}
    for target_name, values in target_steps.items():
        source_values = source_steps[reverse_mapping[target_name]]
        for step_index, (target_step, source_step) in enumerate(zip(values, source_values)):
            relative_limit = 1.5 * float(source_step) + 10.0
            # The source motion is the temporal reference.  A target may not
            # introduce motion faster than the source-relative allowance, but
            # a legitimate source step above 45 degrees is not itself a
            # retarget defect and must not force a geometry-destroying clamp.
            if float(target_step) > relative_limit + 0.001:
                angular_violations.append(
                    {
                        "bone": target_name,
                        "frame": step_index + 2,
                        "target_degrees": float(target_step),
                        "source_degrees": float(source_step),
                        "relative_limit_degrees": relative_limit,
                    }
                )

    mirror_abnormalities: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for left, right in (rig_profile or {}).get("mirror_pairs", {}).items():
        pair = tuple(sorted((left, right)))
        if pair in seen_pairs or left not in profile_lengths or right not in profile_lengths:
            continue
        seen_pairs.add(pair)
        left_length = float(profile_lengths[left])
        right_length = float(profile_lengths[right])
        relative = abs(left_length - right_length) / max(left_length, right_length, 1e-8)
        if relative > 0.25:
            mirror_abnormalities.append(
                {"left": left, "right": right, "relative_length_difference": relative}
            )

    representative_frames: list[dict[str, Any]] = []
    representative_candidates = [
        (1, "first"),
        ((len(global_orient) + 1) // 2, "middle"),
        (len(global_orient), "last"),
        (int(np.argmax(joint_error_by_frame)) + 1, "maximum_joint_error"),
        (maximum_target_frame, "maximum_angular_velocity"),
        (lowest_mesh_frame, "minimum_mesh_height"),
        *((int(value["frame"]), f"maximum_{foot_name}_drift") for foot_name, value in drift_peaks.items()),
    ]
    reasons_by_frame: dict[int, list[str]] = {}
    for frame, reason in representative_candidates:
        if frame is not None:
            reasons_by_frame.setdefault(int(frame), []).append(reason)
    for frame in sorted(reasons_by_frame):
        representative_frames.append({"frame": frame, "reasons": reasons_by_frame[frame]})

    result = {
        "asset": str(asset_path),
        "bone_coverage": {"mapped": len(mapping), "expected": 22, "complete": len(mapping) == 22},
        "rig_preflight": {
            "valid": inspection.valid,
            "prefix": None if inspection.adapter is None else inspection.adapter.prefix,
            "errors": list(inspection.errors),
        },
        "rig_profile": rig_profile,
        "location_key_counts": location_key_counts,
        "minimum_mesh_z_m": lowest_mesh_z,
        "minimum_mesh_z_frame": lowest_mesh_frame,
        "minimum_profiled_sole_z_m": lowest_profiled_sole_z,
        "root_offset_z_m": root_offset_z,
        "first_source_support_z_m": first_source_support_z,
        "source_minimum_foot_z_m": float(np.min(source_feet[:, :, 2])),
        "target_height_m": target_height,
        "source_to_target_scale": source_to_target_scale,
        "joint_error_reference": "rest_delta_trunk_and_target_length_limb_directions",
        "segment_error_reference": "direct_target_space_baseline",
        "first_frame_contact_diagnostic": first_frame_contact_diagnostic,
        "mean_joint_error_m": float(np.mean(joint_error_samples)),
        "p95_joint_error_m": float(np.percentile(joint_error_samples, 95.0)),
        "joint_error_by_name_m": {
            name: {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95.0)),
                "maximum": float(np.max(values)),
            }
            for name, values in joint_error_by_name.items()
        },
        "maximum_contact_drift_m": drift,
        "source_contact_drift_m": source_drift,
        "maximum_sole_anchor_drift_m": drift,
        "contact_drift_by_interval_m": drift_by_interval,
        "contact_drift_peaks": drift_peaks,
        "contact_intervals": contacts.contact_intervals,
        "representative_frames": representative_frames,
        "segment_error_degrees": {name: float(np.nanmax(values)) for name, values in angle_samples.items()},
        "endpoint_error_m": {name: float(np.max(values)) for name, values in endpoint_samples.items()},
        "maximum_segment_error_degrees": float(
            max((np.nanmax(values) for values in angle_samples.values()), default=0.0)
        ),
        "maximum_ankle_endpoint_error_m": float(
            max(
                endpoint_samples[name] and max(endpoint_samples[name])
                for name in ("left_ankle", "right_ankle")
            )
        ),
        "maximum_sole_tilt_degrees": sole_tilt,
        "maximum_foot_heading_error_degrees": heading_error,
        "maximum_ankle_orientation_error_degrees": sole_orientation_error,
        "raw_ankle_orientation_error_degrees_diagnostic": {
            name: float(max(values, default=0.0)) for name, values in raw_ankle_orientation_samples.items()
        },
        "maximum_bone_length_drift_relative": float(
            max((max(values, default=0.0) for values in bone_length_error.values()), default=0.0)
        ),
        "maximum_root_speed_m_per_frame": float(
            np.linalg.norm(np.diff(np.asarray(root_locations), axis=0), axis=1).max(initial=0.0)
        ),
        "maximum_target_angular_step_degrees": float(maximum_target_values.max(initial=0.0)),
        "maximum_target_angular_step": {
            "bone": maximum_target_name,
            "frame": maximum_target_frame,
            "degrees": float(maximum_target_values.max(initial=0.0)),
        },
        "angular_velocity_violations": angular_violations,
        "joint_limit_violations": joint_limit_violations,
        "mirror_abnormalities": mirror_abnormalities,
        "maximum_quaternion_norm_error": float(
            max(
                (
                    np.abs(np.linalg.norm(np.asarray(values), axis=1) - 1.0).max(initial=0.0)
                    for values in quaternion_samples.values()
                ),
                default=0.0,
            )
        ),
    }
    from motionviewer.blender.retarget.quality_gate import evaluate_quality_report

    result["quality_gate"] = evaluate_quality_report(result)
    return result


def main() -> None:
    args = _parse_args()
    report = {
        "motion": str(args.motion),
        "retarget_mode": args.mode,
        "assets": [_metric_for_asset(asset, args.motion, mode=args.mode) for asset in args.asset],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
