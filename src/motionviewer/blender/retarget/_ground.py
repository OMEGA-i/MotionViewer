"""Ground-contact plan for FBX retargeting.

Computes the vertical offset needed so FBX character feet touch Z=0 ground,
and (in Phase 2+) holds per-frame foot contact intervals and locked positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import MixamoNameAdapter


@dataclass(frozen=True)
class GroundPlan:
    """Precomputed ground-contact data for one retarget job.

    Attributes:
        vertical_offset_z:
            Static Z correction (leg-length difference + mesh hang below foot
            bones). Positive = push FBX up so feet touch ground.
        foot_contact_frames:
            Per-foot contact intervals ``{foot_name: [(start, end), ...]}``.
        contact_positions:
            Per-interval target world positions for locked feet.
    """

    vertical_offset_z: float
    foot_contact_frames: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    contact_positions: dict[str, list[np.ndarray]] = field(default_factory=dict)


def _mesh_lowest_z(mesh_obj: Any) -> float:
    """Return the lowest Z coordinate across all vertices of *mesh_obj*.

    Used to measure how far mesh geometry protrudes below the lowest foot
    bone (soles, heels, toes that extend past bone endpoints).
    """
    import bpy  # type: ignore

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    try:
        if not eval_mesh.vertices:
            return 0.0
        mat = mesh_obj.matrix_world
        min_z = float("inf")
        for vertex in eval_mesh.vertices:
            world_z = (mat @ vertex.co).z
            if world_z < min_z:
                min_z = world_z
        return float(min_z) if min_z != float("inf") else 0.0
    finally:
        eval_obj.to_mesh_clear()


def compute_vertical_offset(
    smplx_armature: Any,
    fbx_armature: Any,
    fbx_meshes: list[Any],
) -> float:
    """Compute the Z offset so FBX feet rest on Z=0 ground.

    Formula: ``smplx_leg - (fbx_leg + mesh_hang)``.

    *fbx_leg* is the Z of the lowest FBX foot-bone head.
    *smplx_leg* is the Z of the lowest SMPL-X foot-bone head.
    *mesh_hang* is how far FBX mesh geometry extends below the lowest foot
    bone (accounting for soles, heels, or toes that protrude past bone
    endpoints).
    """
    import bpy  # type: ignore

    bpy.context.view_layer.update()

    # SMPL-X's ankle is the target-side Mixamo ``*Foot`` counterpart, while
    # SMPL-X ``*_foot`` maps to Mixamo ``*ToeBase``. Include both support
    # joints so a lower ankle cannot leave the shoe sole below ground.
    foot_bones = ("left_ankle", "right_ankle", "left_foot", "right_foot", "left_toe", "right_toe")
    smplx_z = float("inf")
    fbx_z = float("inf")

    for bone_name in foot_bones:
        pb = smplx_armature.pose.bones.get(bone_name)
        if pb is not None:
            z = (smplx_armature.matrix_world @ pb.head).z
            if z < smplx_z:
                smplx_z = z

    adapter = MixamoNameAdapter.detect({bone.name for bone in fbx_armature.data.bones})
    if adapter is None:
        return 0.0
    for canonical_name in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"):
        fbx_pb = fbx_armature.pose.bones.get(adapter.target_name(canonical_name))
        if fbx_pb is not None:
            z = (fbx_armature.matrix_world @ fbx_pb.head).z
            if z < fbx_z:
                fbx_z = z

    if smplx_z == float("inf") or fbx_z == float("inf"):
        return 0.0

    mesh_hang = 0.0
    for mesh_obj in fbx_meshes:
        lowest = _mesh_lowest_z(mesh_obj)
        if lowest < fbx_z:
            mesh_hang = min(mesh_hang, lowest - fbx_z)

    return vertical_offset_from_heights(smplx_z, fbx_z, mesh_hang)


def vertical_offset_from_heights(smplx_z: float, fbx_z: float, mesh_hang: float) -> float:
    """Return the FBX root Z correction from measured rest-pose heights.

    ``mesh_hang`` is non-positive when soles extend below the lowest target
    foot bone. Including it inside the parenthesized target height raises the
    character in that case, instead of pushing it farther through the ground.
    """
    return float(smplx_z - (fbx_z + mesh_hang))


def compute_ground_plan(
    smplx_armature: Any,
    fbx_armature: Any,
    fbx_meshes: list[Any],
    *,
    num_frames: int = 0,
    frame_start: int = 1,
    global_orient: np.ndarray | None = None,
    body_pose: np.ndarray | None = None,
    transl: np.ndarray | None = None,
    unit_scale: float = 1.0,
    root_translation_scale: float = 1.0,
    bpy: Any = None,
) -> GroundPlan:
    """Compute vertical alignment and source-space foot contact intervals.

    Args:
        smplx_armature: SMPL-X driver armature.
        fbx_armature: Imported FBX armature.
        fbx_meshes: Imported FBX mesh objects.
        num_frames: Number of motion frames.
        frame_start: First animation frame number.
        global_orient: ``(T, 3)`` SMPL-X global orientation.
        body_pose: ``(T, 21, 3)`` SMPL-X body pose.
        transl: ``(T, 3)`` raw SMPL-X root translation.
        unit_scale: Source-to-Blender root translation scale.
        root_translation_scale: Avatar-height scale applied identically to
            every root translation axis by the animation stage.
        bpy: Blender Python module.
    """
    from ..smplx_mesh import source_transl_to_blender
    from ._quality import detect_foot_contact_frames

    if bpy is None:
        import bpy as _bpy  # type: ignore

        bpy = _bpy

    scene = bpy.context.scene
    original_frame = scene.frame_current
    original_object_basis = smplx_armature.matrix_basis.copy()
    original_pose_bases = {
        pose_bone.name: pose_bone.matrix_basis.copy() for pose_bone in smplx_armature.pose.bones
    }

    try:
        vertical_offset_z = compute_vertical_offset(smplx_armature, fbx_armature, fbx_meshes)
        foot_contact_frames: dict[str, list[tuple[int, int]]] = {}
        contact_positions: dict[str, list[np.ndarray]] = {}

        if num_frames > 0 and global_orient is not None and body_pose is not None:
            from bl_ext.user_default.smplx_blender_addon.utils.pose import (
                set_pose_from_rodrigues,  # type: ignore
            )

            from ...core.smplx_actor import BODY_POSE_BONES

            foot_positions = np.zeros((num_frames, 2, 3), dtype=np.float32)
            lowest_support_z = float("inf")
            for idx in range(num_frames):
                frame = frame_start + idx
                scene.frame_set(frame)
                set_pose_from_rodrigues(smplx_armature, "pelvis", global_orient[idx])
                for bone_idx, bone_name in enumerate(BODY_POSE_BONES):
                    if bone_idx < body_pose.shape[1] and bone_name in smplx_armature.pose.bones:
                        set_pose_from_rodrigues(smplx_armature, bone_name, body_pose[idx, bone_idx])
                bpy.context.view_layer.update()
                root_translation = (
                    source_transl_to_blender(transl[idx], unit_scale)
                    if transl is not None
                    else np.zeros(3, dtype=np.float32)
                )
                root_translation[:2] *= root_translation_scale
                for foot_idx, bone_name in enumerate(("left_foot", "right_foot")):
                    pb = smplx_armature.pose.bones.get(bone_name)
                    if pb is not None:
                        foot_positions[idx, foot_idx] = (
                            np.asarray((smplx_armature.matrix_world @ pb.head).to_tuple(), dtype=np.float32)
                            + root_translation
                        )
                for bone_name in (
                    "left_ankle",
                    "right_ankle",
                    "left_foot",
                    "right_foot",
                    "left_toe",
                    "right_toe",
                ):
                    pb = smplx_armature.pose.bones.get(bone_name)
                    if pb is not None:
                        support_z = float((smplx_armature.matrix_world @ pb.head).z + root_translation[2])
                        lowest_support_z = min(lowest_support_z, support_z)

            # Preserve the complete input root trajectory, but choose its
            # static vertical origin so the lowest source foot reaches ground.
            # Motion files commonly carry a constant pelvis-height bias; this
            # is separate from the FBX leg-length / sole-hang correction.
            if lowest_support_z != float("inf"):
                vertical_offset_z += max(0.0, -lowest_support_z)

            contact_result = detect_foot_contact_frames(foot_positions)
            foot_contact_frames = contact_result.contact_intervals
            contact_positions = contact_result.locked_positions

        return GroundPlan(
            vertical_offset_z=vertical_offset_z,
            foot_contact_frames=foot_contact_frames,
            contact_positions=contact_positions,
        )
    finally:
        scene.frame_set(original_frame)
        smplx_armature.matrix_basis = original_object_basis
        for bone_name, matrix_basis in original_pose_bases.items():
            pose_bone = smplx_armature.pose.bones.get(bone_name)
            if pose_bone is not None:
                pose_bone.matrix_basis = matrix_basis
        bpy.context.view_layer.update()
