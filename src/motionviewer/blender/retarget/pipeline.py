"""Retarget pipeline orchestrator.

Wires the 6 pipeline stages together and exposes the public
``create_fbx_actor_from_npz`` entry point with the same signature
as the original monolith.
"""

from __future__ import annotations

from pathlib import Path

from ...core.smplx_actor import SmplxActor
from ._animate import animate_all_frames, animate_mmd_frames
from ._bootstrap import bootstrap_input
from ._ground import compute_ground_plan
from ._precompute import precompute_retarget_context
from ._resolve import resolve_bone_mapping


def create_fbx_actor_from_npz(
    path: str | Path,
    *,
    label: str,
    fbx_path: str | Path,
    bone_map: str = "auto",
    gender: str = "neutral",
    frame_start: int = 1,
    unit_scale: float = 1.0,
    fbx_scale: float = 1.0,
    retarget_mode: str = "quality",
    layout_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    motion_overrides: dict | None = None,
    mmd_polish: dict | None = None,
    mmd_physics: bool = False,
) -> SmplxActor:
    """Import an FBX character and animate it from SMPL-X motion data.

    Creates a hidden SMPL-X armature as a pose driver, imports the FBX
    character, and copies local bone rotations frame-by-frame from the
    SMPL-X skeleton to the FBX skeleton according to *bone_map*.

    Returns a ``SmplxActor`` whose ``.armature`` is the FBX armature and
    ``.mesh_objects`` are the FBX skinned meshes — ready to be styled,
    ghosted, and rendered just like a native SMPL-X mesh actor.
    """
    import bpy  # type: ignore

    # ---- Stage 1–2: Bootstrap (NPZ load + SMPL-X driver + FBX import) ------
    boot = bootstrap_input(
        bpy,
        path,
        label=label,
        gender=gender,
        fbx_path=fbx_path,
        fbx_scale=fbx_scale,
        motion_overrides=motion_overrides,
        layout_offset=layout_offset,
        unit_scale=unit_scale,
        frame_start=frame_start,
        mmd_physics=mmd_physics,
    )

    smplx_armature = boot.smplx_armature
    fbx_armature = boot.fbx_armature
    fbx_meshes = boot.fbx_meshes

    if retarget_mode not in {"quality", "direct"}:
        raise ValueError("retarget_mode must be 'quality' or 'direct'")

    # ---- Stage 3: Resolve the validated Mixamo mapping ---------------------
    mapping = resolve_bone_mapping(bone_map, fbx_armature=fbx_armature)

    # MMD rigs need their own solve: cancel bones, 捩 twist bones and D-bones
    # that copy the FK chain make the Mixamo rest-delta path the wrong model.
    if mapping.rig_family == "mmd":
        from .mmd import MmdPolishOptions

        return animate_mmd_frames(
            bpy,
            boot,
            mapping,
            vertical_offset_z=_mmd_ground_offset(fbx_meshes),
            polish=MmdPolishOptions.from_mapping(mmd_polish),
        )

    # ---- Stage 4: Precompute rest-delta transfer matrices ------------------
    # This must happen while the hidden source armature is still in rest pose.
    # Ground contact scanning poses every source frame and therefore cannot
    # precede construction of aMatrix.
    context = precompute_retarget_context(
        smplx_armature,
        fbx_armature,
        mapping.smplx_to_fbx,
        bpy=bpy,
        fbx_meshes=fbx_meshes,
    )

    # ---- Stage 5: Compute ground plan --------------------------------------
    ground = compute_ground_plan(
        smplx_armature,
        fbx_armature,
        fbx_meshes,
        num_frames=boot.num_frames,
        frame_start=boot.frame_start,
        global_orient=boot.global_orient,
        body_pose=boot.body_pose,
        transl=boot.transl,
        unit_scale=boot.unit_scale,
        root_translation_scale=context.root_translation_scale,
        bpy=bpy,
    )

    # ---- Stage 6: Animate all frames ---------------------------------------
    return animate_all_frames(bpy, boot, context, ground, retarget_mode=retarget_mode)


def _mmd_ground_offset(meshes: list) -> float:
    """Raise the character so the lowest mesh vertex sits on Z=0."""
    lowest = 0.0
    found = False
    for mesh in meshes:
        matrix = mesh.matrix_world
        for vertex in mesh.data.vertices:
            world_z = (matrix @ vertex.co).z
            if not found or world_z < lowest:
                lowest = float(world_z)
                found = True
    return -lowest if found and lowest < 0.0 else 0.0
