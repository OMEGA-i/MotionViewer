"""FBX export and round-trip validation for retargeted actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _set_linear_interpolation(action: Any) -> None:
    for curve in getattr(action, "fcurves", ()):
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def _quaternion_geodesic_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the sign-invariant geodesic distance between WXYZ quaternions."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(first * second, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def compare_roundtrip_samples(
    reference_positions: dict[str, np.ndarray],
    actual_positions: dict[str, np.ndarray],
    reference_rotations: dict[str, np.ndarray],
    actual_rotations: dict[str, np.ndarray],
    reference_root: np.ndarray,
    actual_root: np.ndarray,
    *,
    foot_bone_names: tuple[str, ...] = (),
    position_tolerance_m: float = 2e-3,
    rotation_tolerance_degrees: float = 0.5,
) -> dict[str, Any]:
    """Compare absolute world samples from a scene and an imported FBX.

    This deliberately compares positions and quaternions at every frame. Frame
    to frame deltas alone can hide a constant offset or a persistent foot twist.
    """
    position_errors: dict[str, float] = {}
    rotation_errors: dict[str, float] = {}
    for name, reference in reference_positions.items():
        actual = actual_positions.get(name)
        if actual is None:
            raise ValueError(f"Round-trip sample is missing position bone {name!r}")
        if np.shape(reference) != np.shape(actual):
            raise ValueError(f"Round-trip position shape mismatch for {name!r}")
        position_errors[name] = float(
            np.linalg.norm(np.asarray(reference) - np.asarray(actual), axis=-1).max(initial=0.0)
        )
    for name, reference in reference_rotations.items():
        actual = actual_rotations.get(name)
        if actual is None:
            raise ValueError(f"Round-trip sample is missing rotation bone {name!r}")
        if np.shape(reference) != np.shape(actual):
            raise ValueError(f"Round-trip rotation shape mismatch for {name!r}")
        rotation_errors[name] = float(_quaternion_geodesic_degrees(reference, actual).max(initial=0.0))
    root_error = float(
        np.linalg.norm(np.asarray(reference_root) - np.asarray(actual_root), axis=-1).max(initial=0.0)
    )
    foot_positions = {name: position_errors[name] for name in foot_bone_names if name in position_errors}
    foot_rotations = {name: rotation_errors[name] for name in foot_bone_names if name in rotation_errors}
    maximum_position = max([root_error, *position_errors.values()], default=0.0)
    maximum_rotation = max(rotation_errors.values(), default=0.0)
    return {
        "passed": maximum_position <= position_tolerance_m and maximum_rotation <= rotation_tolerance_degrees,
        "maximum_absolute_position_error_m": maximum_position,
        "maximum_root_position_error_m": root_error,
        "maximum_bone_position_error_m": position_errors,
        "maximum_foot_position_error_m": foot_positions,
        "maximum_quaternion_error_degrees": maximum_rotation,
        "maximum_foot_quaternion_error_degrees": max(foot_rotations.values(), default=0.0),
        "bone_quaternion_error_degrees": rotation_errors,
        "position_tolerance_m": float(position_tolerance_m),
        "rotation_tolerance_degrees": float(rotation_tolerance_degrees),
    }


def export_fbx_animation(
    bpy: Any,
    armature: Any,
    mesh_objects: list[Any],
    output_path: str | Path,
    *,
    frame_start: int,
    frame_end: int,
    fps: float | None = None,
) -> Path:
    """Export one retargeted armature/action as a deterministic FBX."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_start = int(frame_start)
    scene.frame_end = int(frame_end)
    if fps is not None:
        scene.render.fps = int(round(float(fps)))
    action = getattr(getattr(armature, "animation_data", None), "action", None)
    if action is None:
        raise ValueError("Retargeted armature has no animation action to export")
    _set_linear_interpolation(action)
    # The retargeter writes quaternion curves. Keep the armature and every pose
    # bone in that mode so the FBX bake cannot silently convert them to Euler.
    if hasattr(armature, "rotation_mode"):
        armature.rotation_mode = "QUATERNION"
    for pose_bone in getattr(getattr(armature, "pose", None), "bones", ()):
        if hasattr(pose_bone, "rotation_mode"):
            pose_bone.rotation_mode = "QUATERNION"

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    for mesh in mesh_objects:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    result = bpy.ops.export_scene.fbx(
        filepath=str(output),
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        axis_forward="-Z",
        axis_up="Y",
    )
    if "CANCELLED" in set(result) or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"FBX export failed: {output}")
    return output


def _sample_armature(
    bpy: Any, armature: Any, bone_names: tuple[str, ...], frames: list[int]
) -> dict[str, np.ndarray]:
    samples: dict[str, list[np.ndarray]] = {name: [] for name in bone_names}
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for name in bone_names:
            pose_bone = armature.pose.bones.get(name)
            if pose_bone is None:
                raise ValueError(f"Round-trip armature is missing bone {name!r}")
            matrix = armature.matrix_world @ pose_bone.matrix
            samples[name].append(
                np.asarray(tuple(float(value) for row in matrix for value in row), dtype=np.float64)
            )
    return {name: np.stack(values) for name, values in samples.items()}


def _sample_local_animation(
    bpy: Any, armature: Any, bone_names: tuple[str, ...], frames: list[int]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rotations: dict[str, list[np.ndarray]] = {name: [] for name in bone_names}
    roots: list[np.ndarray] = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        roots.append(np.asarray(tuple(float(value) for value in armature.location), dtype=np.float64))
        for name in bone_names:
            pose_bone = armature.pose.bones.get(name)
            if pose_bone is None:
                raise ValueError(f"Round-trip armature is missing bone {name!r}")
            quat = pose_bone.rotation_quaternion
            rotations[name].append(np.asarray((quat.w, quat.x, quat.y, quat.z), dtype=np.float64))
    return {name: np.stack(values) for name, values in rotations.items()}, np.stack(roots)


def _sample_world_animation(
    bpy: Any,
    armature: Any,
    bone_names: tuple[str, ...],
    frames: list[int],
    *,
    root_bone_name: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    positions: dict[str, list[np.ndarray]] = {name: [] for name in bone_names}
    rotations: dict[str, list[np.ndarray]] = {name: [] for name in bone_names}
    roots: list[np.ndarray] = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        root_bone = armature.pose.bones.get(root_bone_name) if root_bone_name else None
        if root_bone is not None:
            root_matrix = armature.matrix_world @ root_bone.matrix
            roots.append(
                np.asarray(tuple(float(value) for value in root_matrix.translation), dtype=np.float64)
            )
        else:
            roots.append(
                np.asarray(
                    tuple(float(value) for value in armature.matrix_world.translation), dtype=np.float64
                )
            )
        for name in bone_names:
            pose_bone = armature.pose.bones.get(name)
            if pose_bone is None:
                raise ValueError(f"Round-trip armature is missing bone {name!r}")
            matrix = armature.matrix_world @ pose_bone.matrix
            positions[name].append(
                np.asarray(tuple(float(value) for value in matrix.translation), dtype=np.float64)
            )
            quat = matrix.to_quaternion()
            rotations[name].append(np.asarray((quat.w, quat.x, quat.y, quat.z), dtype=np.float64))
    return (
        {name: np.stack(values) for name, values in positions.items()},
        {name: np.stack(values) for name, values in rotations.items()},
        np.stack(roots),
    )


def validate_fbx_roundtrip(
    bpy: Any,
    reference_armature: Any,
    output_path: str | Path,
    *,
    bone_names: tuple[str, ...],
    frame_start: int,
    frame_end: int,
    root_bone_name: str | None = None,
    position_bone_names: tuple[str, ...] | None = None,
    foot_bone_names: tuple[str, ...] = (),
    expected_fps: float | None = None,
    expected_fps_base: float | None = None,
    position_tolerance_m: float = 2e-3,
    rotation_tolerance_degrees: float = 0.5,
) -> dict[str, Any]:
    """Import an exported FBX and compare sampled bone matrices."""
    frames = list(range(int(frame_start), int(frame_end) + 1))
    scene = bpy.context.scene
    reference_fps = float(scene.render.fps)
    reference_fps_base = float(getattr(scene.render, "fps_base", 1.0))
    reference_positions, reference_world_rotations, reference_world_root = _sample_world_animation(
        bpy, reference_armature, position_bone_names or bone_names, frames, root_bone_name=root_bone_name
    )
    reference_rotations, reference_roots = _sample_local_animation(
        bpy, reference_armature, bone_names, frames
    )
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(Path(output_path).resolve()), use_anim=True)
    imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "ARMATURE"]
    if len(imported) != 1:
        raise RuntimeError(f"Expected one imported round-trip armature, found {len(imported)}")
    candidate = imported[0]
    try:
        action = getattr(getattr(candidate, "animation_data", None), "action", None)
        if action is None:
            raise RuntimeError("Imported FBX armature has no animation action")
        imported_start = int(round(action.frame_range[0]))
        imported_end = int(round(action.frame_range[1]))
        expected_count = int(frame_end) - int(frame_start) + 1
        frame_count_error = imported_end - imported_start + 1 != expected_count
        candidate_frames = [frame - int(frame_start) + imported_start for frame in frames]
        actual_positions, actual_world_rotations, actual_world_root = _sample_world_animation(
            bpy, candidate, position_bone_names or bone_names, candidate_frames, root_bone_name=root_bone_name
        )
        actual_rotations, actual_roots = _sample_local_animation(bpy, candidate, bone_names, candidate_frames)
        reference_root_steps = np.diff(reference_roots, axis=0)
        actual_root_steps = np.diff(actual_roots, axis=0)
        max_position = float(
            np.linalg.norm(reference_root_steps - actual_root_steps, axis=1).max(initial=0.0)
        )
        max_rotation = 0.0
        for name in bone_names:
            ref = reference_rotations[name]
            got = actual_rotations[name]
            ref_steps = _quaternion_geodesic_degrees(ref[1:], ref[:-1])
            got_steps = _quaternion_geodesic_degrees(got[1:], got[:-1])
            max_rotation = max(max_rotation, float(np.abs(ref_steps - got_steps).max(initial=0.0)))
        absolute = compare_roundtrip_samples(
            reference_positions,
            actual_positions,
            reference_world_rotations,
            actual_world_rotations,
            reference_world_root,
            actual_world_root,
            foot_bone_names=foot_bone_names,
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_degrees=rotation_tolerance_degrees,
        )
        actual_fps = float(scene.render.fps)
        actual_fps_base = float(getattr(scene.render, "fps_base", 1.0))
        expected_fps = reference_fps if expected_fps is None else float(expected_fps)
        expected_fps_base = reference_fps_base if expected_fps_base is None else float(expected_fps_base)
        fps_error = abs(actual_fps / actual_fps_base - expected_fps / expected_fps_base)
        rotation_mode_violations = [
            name
            for name in bone_names
            if candidate.pose.bones.get(name) is not None
            and candidate.pose.bones[name].rotation_mode != "QUATERNION"
        ]
        absolute["passed"] = bool(
            absolute["passed"]
            and max_position <= position_tolerance_m
            and max_rotation <= rotation_tolerance_degrees
            and not frame_count_error
            and fps_error <= 1e-6
            and not rotation_mode_violations
        )
        return {
            **absolute,
            "maximum_position_step_error_m": max_position,
            "maximum_rotation_step_error_degrees": max_rotation,
            "frame_count": len(frames),
            "imported_frame_start": imported_start,
            "imported_frame_end": imported_end,
            "frame_count_error": frame_count_error,
            "expected_fps": expected_fps,
            "expected_fps_base": expected_fps_base,
            "imported_fps": actual_fps,
            "imported_fps_base": actual_fps_base,
            "fps_error": fps_error,
            "rotation_mode_violations": rotation_mode_violations,
        }
    finally:
        for obj in imported:
            bpy.data.objects.remove(obj, do_unlink=True)
