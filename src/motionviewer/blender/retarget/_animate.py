"""Blender adapters for the pure retarget solvers.

Two rig families, two solvers, one rule: the solve happens in NumPy and Blender
only receives channels.  ``animate_all_frames`` drives Mixamo through the
rest-delta path; ``animate_mmd_frames`` drives MMD through its own plan, which
models cancel bones, 捩 twist bones and the D-bones that carry the leg mesh.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ...core.smplx_actor import BODY_POSE_BONES, SmplxActor
from ..smplx_mesh import source_transl_to_blender
from ._bootstrap import BootstrapOutput
from ._ground import GroundPlan
from ._mmd_plan import build_mmd_plan
from ._precompute import RetargetContext
from ._resolve import BoneMapping
from .mmd import MmdPolishOptions
from .mmd_hands import relax_hands
from .mmd_solve import (
    MmdRetargetPlan,
    MmdSolveResult,
    audit_plan,
    polish_source_frames,
    solve_mmd_retarget,
)
from .solver import solve_retarget


def _sample_source_motion(
    bpy: Any, boot: BootstrapOutput, context: RetargetContext
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate source pose matrices once for the solver."""
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
            quaternion = result.local_quaternions_wxyz[index, bone_index]
            pose_bone.rotation_quaternion = tuple(float(value) for value in quaternion)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=target_name)
    _linearize(boot.fbx_armature)
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


def _action_fcurves(action: Any) -> list[Any]:
    """Curves of an action under either the legacy or the slotted layout.

    Blender 4.4 moved curves into layer/strip channel bags and dropped
    ``Action.fcurves``.  Reading only one of the two shapes silently leaves
    every keyframe on its default easing.
    """
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return list(legacy)
    curves: list[Any] = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                curves.extend(channelbag.fcurves)
    return curves


def _linearize(armature: Any) -> None:
    """Retargeted samples are one per source frame; Bezier easing invents motion."""
    action = getattr(getattr(armature, "animation_data", None), "action", None)
    if action is None:
        return
    for curve in _action_fcurves(action):
        for keyframe in curve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def animate_mmd_frames(
    bpy: Any,
    boot: BootstrapOutput,
    mapping: BoneMapping,
    *,
    vertical_offset_z: float = 0.0,
    polish: MmdPolishOptions | None = None,
) -> SmplxActor:
    """Retarget onto an MMD rig and write one quaternion per driven channel.

    The MMD rig carries cancel bones, 捩 twist bones and D-bones that copy the
    FK chain, so it gets its own plan and solver rather than the Mixamo
    rest-delta path.  Blender only receives channels here.
    """
    motion = boot.lookat_motion
    if motion is None:
        raise RuntimeError("MMD retarget requires joints22 in the motion NPZ")
    options = polish or MmdPolishOptions()

    plan = build_mmd_plan(
        boot.fbx_armature,
        smplx_map=mapping.smplx_to_fbx,
        transfer_modes=mapping.transfer_modes,
        twist_pairs=mapping.twist_pairs or boot.twist_pairs,
        source_rest=motion.rest_by_name(),
        source_names=motion.names,
    )
    frames = polish_source_frames(
        np.asarray(motion.posed_frames[: boot.num_frames], dtype=np.float64),
        motion.names,
        motion.rest_frames,
        options,
    )
    roots = _mmd_root_locations(motion, boot, plan.root_translation_scale, vertical_offset_z)
    result = solve_mmd_retarget(plan, frames, roots)
    _write_mmd_channels(bpy, boot, plan, result)

    hands: dict[str, Any] = {"amount": 0.0}
    if options.enabled and options.hand_relax > 0.0:
        hands = relax_hands(boot.fbx_armature, amount=options.hand_relax)

    boot.fbx_armature["motionviewer_mmd_transfer"] = json.dumps(
        {
            "channels": audit_plan(plan),
            "root_translation_scale": plan.root_translation_scale,
            "polish": {
                "twist_smoothing_window": options.twist_window if options.enabled else 0,
                "hand_relax": hands.get("amount", 0.0),
                "collar_damping": options.collar_damping if options.enabled else 1.0,
                "collar_limit_degrees": options.collar_limit_degrees if options.enabled else None,
                "arm_abduction_degrees": options.arm_abduction_degrees if options.enabled else 0.0,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    bpy.context.view_layer.update()
    return SmplxActor(label=boot.label, armature=boot.fbx_armature, mesh_objects=boot.fbx_meshes)


def _mmd_root_locations(
    motion: Any, boot: BootstrapOutput, scale: float, vertical_offset_z: float
) -> np.ndarray:
    """Height-scaled root path, re-based so frame 0 stands on the floor.

    The MMD armature origin is the ground while SMPL-X ``transl`` is the pelvis,
    so only the *delta* transfers.  Vertical delta is kept, otherwise crouches
    and jumps would flatten out.
    """
    roots = np.asarray(motion.root_locations[: boot.num_frames], dtype=np.float64).copy()
    roots = (roots - roots[0]) * float(scale)
    roots += np.asarray(boot.offset, dtype=np.float64)
    roots[:, 2] += float(vertical_offset_z)
    return roots


def _write_mmd_channels(
    bpy: Any, boot: BootstrapOutput, plan: MmdRetargetPlan, result: MmdSolveResult
) -> None:
    from mathutils import Matrix  # type: ignore

    armature = boot.fbx_armature
    # The plan treats an undriven channel as contributing only its rest offset,
    # so any leftover basis on one would silently shift everything below it.
    for channel in plan.channels:
        if channel.mode != "passthrough":
            continue
        pose_bone = armature.pose.bones.get(channel.name)
        if pose_bone is not None:
            pose_bone.matrix_basis = Matrix.Identity(4)

    driven = [
        (index, channel.name)
        for index, channel in enumerate(plan.channels)
        if channel.mode != "passthrough"
    ]
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"

    for index in range(boot.num_frames):
        frame = boot.frame_start + index
        bpy.context.scene.frame_set(frame)
        armature.location = tuple(float(value) for value in result.root_locations[index])
        armature.keyframe_insert(data_path="location", frame=frame)
        for channel_index, name in driven:
            pose_bone = armature.pose.bones[name]
            pose_bone.rotation_quaternion = tuple(
                float(value) for value in result.local_quaternions_wxyz[index, channel_index]
            )
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=name)
    _linearize(armature)
