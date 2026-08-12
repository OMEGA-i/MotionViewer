"""Blender adapter for the pure Mixamo retarget solver."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ...core.smplx_actor import BODY_POSE_BONES, SmplxActor
from ..smplx_mesh import source_transl_to_blender
from ._bootstrap import BootstrapOutput
from ._ground import GroundPlan
from ._precompute import RetargetContext
from .solver import solve_retarget


def _sample_source_motion(
    bpy: Any, boot: BootstrapOutput, context: RetargetContext
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate SMPL-X once and return matrices/root locations for the solver."""
    from bl_ext.user_default.smplx_blender_addon.utils.pose import set_pose_from_rodrigues  # type: ignore

    definition = context.solver_definition
    if definition is None:
        raise RuntimeError("Retarget context has no pure solver definition")
    source = np.empty((boot.num_frames, len(definition.source_names), 4, 4), dtype=np.float64)
    roots = np.empty((boot.num_frames, 3), dtype=np.float64)
    for index in range(boot.num_frames):
        bpy.context.scene.frame_set(boot.frame_start + index)
        set_pose_from_rodrigues(boot.smplx_armature, "pelvis", boot.global_orient[index])
        for pose_index, bone_name in enumerate(BODY_POSE_BONES):
            if pose_index < boot.body_pose.shape[1] and bone_name in boot.smplx_armature.pose.bones:
                set_pose_from_rodrigues(boot.smplx_armature, bone_name, boot.body_pose[index, pose_index])
        bpy.context.view_layer.update()
        for bone_index, source_name in enumerate(definition.source_names):
            matrix = boot.smplx_armature.matrix_world @ boot.smplx_armature.pose.bones[source_name].matrix
            rigid = matrix.to_quaternion().to_matrix().to_4x4()
            rigid.translation = matrix.translation
            source[index, bone_index] = np.asarray(rigid, dtype=np.float64)
        raw_root = source_transl_to_blender(boot.transl[index], boot.unit_scale) + boot.offset
        roots[index] = (
            raw_root[0] * context.root_translation_scale,
            raw_root[1] * context.root_translation_scale,
            raw_root[2],
        )
    return source, roots


def animate_all_frames(
    bpy: Any,
    boot: BootstrapOutput,
    context: RetargetContext,
    ground: GroundPlan,
    *,
    retarget_mode: str = "quality",
) -> SmplxActor:
    """Sample source data, solve outside Blender, and write the result once."""
    definition = context.solver_definition
    if definition is None:
        raise RuntimeError("Retarget context has no pure solver definition")
    source_matrices, root_locations = _sample_source_motion(bpy, boot, context)
    root_locations[:, 2] += ground.vertical_offset_z
    result = solve_retarget(
        definition,
        source_matrices,
        root_locations,
        contact_intervals=ground.foot_contact_frames if retarget_mode == "quality" else None,
        mode=retarget_mode,
    )
    for index in range(boot.num_frames):
        frame = boot.frame_start + index
        bpy.context.scene.frame_set(frame)
        boot.fbx_armature.location = tuple(float(value) for value in result.root_locations[index])
        boot.fbx_armature.keyframe_insert(data_path="location", frame=frame)
        for bone_index, target_name in enumerate(definition.target_names):
            pose_bone = boot.fbx_armature.pose.bones[target_name]
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = tuple(
                float(value) for value in result.local_quaternions_wxyz[index, bone_index]
            )
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=target_name)
    action = getattr(getattr(boot.fbx_armature, "animation_data", None), "action", None)
    if action is not None:
        for curve in action.fcurves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"
    # Keep a compact, inspectable stage boundary for the audit and exported
    # asset.  This catches any disagreement between NumPy FK and Blender's
    # pose-channel interpretation before a visually plausible action escapes.
    boot.fbx_armature["motionviewer_solver_first_frame_global_rotations"] = json.dumps(
        {
            target_name: result.target_matrices[0, bone_index, :3, :3].reshape(-1).tolist()
            for bone_index, target_name in enumerate(definition.target_names)
        }
    )
    bpy.context.view_layer.update()
    return SmplxActor(label=boot.label, armature=boot.fbx_armature, mesh_objects=boot.fbx_meshes)
