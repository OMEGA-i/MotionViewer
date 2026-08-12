"""Stage 1–2: bootstrap the render scene for FBX retargeting.

Loads SMPL-X motion data, creates a hidden SMPL-X driver armature, imports
the FBX character, and runs import-time calibration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..addon_probe import require_smplx_addon
from ..smplx_mesh import configure_smplx_tool


@dataclass(frozen=True)
class BootstrapOutput:
    """All data produced by the bootstrap stage needed by downstream stages."""

    smplx_armature: Any
    fbx_armature: Any
    fbx_meshes: list[Any]
    num_frames: int
    global_orient: np.ndarray  # (T, 3)
    body_pose: np.ndarray  # (T, 21, 3)
    transl: np.ndarray  # (T, 3)
    offset: np.ndarray  # (3,) layout offset
    calibration: Any  # CalibrationResult
    rig_profile: Any  # MixamoRigProfile
    unit_scale: float
    frame_start: int
    label: str


def bootstrap_input(
    bpy: Any,
    path: str | Path,
    *,
    label: str,
    gender: str,
    fbx_path: str | Path,
    fbx_scale: float,
    motion_overrides: dict | None,
    layout_offset: tuple[float, float, float],
    unit_scale: float = 1.0,
    frame_start: int = 1,
) -> BootstrapOutput:
    """Load SMPL-X data, create driver armature, import FBX, calibrate.

    Returns a :class:`BootstrapOutput` ready for downstream stages
    (mapping, ground, precompute, animate).
    """
    from .calibration import (
        build_mixamo_rig_profile,
        calibrate_imported_fbx,
        normalize_imported_mixamo_units,
    )

    require_smplx_addon()

    # ---- load SMPL-X motion data -------------------------------------------
    data = np.load(path, allow_pickle=False)
    global_orient = np.asarray(data["global_orient"], dtype=np.float32)
    body_pose = np.asarray(data["body_pose"], dtype=np.float32).reshape(global_orient.shape[0], 21, 3)
    transl = np.asarray(data["transl"], dtype=np.float32)
    if motion_overrides and motion_overrides.get("body_pose") is not None:
        body_pose = np.asarray(motion_overrides["body_pose"], dtype=np.float32).reshape(
            global_orient.shape[0], 21, 3
        )
    if motion_overrides and motion_overrides.get("transl") is not None:
        transl = np.asarray(motion_overrides["transl"], dtype=np.float32)

    offset = np.asarray(layout_offset, dtype=np.float32)
    num_frames = global_orient.shape[0]

    # ---- create hidden SMPL-X armature (pose driver) -----------------------
    before_smplx = set(bpy.data.objects)
    configure_smplx_tool(gender)
    result = bpy.ops.scene.smplx_add_gender()
    if "CANCELLED" in set(result):
        raise RuntimeError("SMPL-X addon cancelled model creation; check model assets and addon preferences")

    created_smplx = [obj for obj in bpy.data.objects if obj not in before_smplx]
    smplx_armature = None
    smplx_meshes: list[Any] = []
    for obj in created_smplx:
        if getattr(obj, "type", None) == "ARMATURE":
            smplx_armature = obj
        elif getattr(obj, "type", None) == "MESH":
            smplx_meshes.append(obj)

    if smplx_armature is None:
        raise RuntimeError("SMPL-X addon did not create an armature for FBX pose driver")

    smplx_armature.name = f"{label}_SMPLX_Driver"
    if smplx_armature.animation_data:
        smplx_armature.animation_data_clear()
    # Hide driver meshes from render (armature must stay visible so the
    # dependency graph updates its pose bone matrices).
    smplx_armature.hide_render = True
    for mesh in smplx_meshes:
        mesh.hide_viewport = True
        mesh.hide_render = True

    # ---- import FBX character ----------------------------------------------
    before_fbx = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.fbx(filepath=str(fbx_path), global_scale=fbx_scale, use_anim=False)
    except RuntimeError as exc:
        msg = str(exc)
        if "ASCII FBX" in msg:
            raise RuntimeError(
                f"Blender {bpy.app.version_string} does not support ASCII FBX files. "
                f"Convert {fbx_path} to binary FBX first (use Autodesk FBX Converter, "
                f"an older Blender version, or re-export from Mixamo.com as binary)."
            ) from exc
        raise

    created_fbx = [obj for obj in bpy.data.objects if obj not in before_fbx]
    fbx_armature = None
    fbx_meshes: list[Any] = []
    for obj in created_fbx:
        if getattr(obj, "type", None) == "ARMATURE":
            fbx_armature = obj
        elif getattr(obj, "type", None) == "MESH":
            fbx_meshes.append(obj)

    if fbx_armature is None:
        raise RuntimeError(f"No armature found in FBX file: {fbx_path}")

    # If the FBX also brought extra objects (empties, cameras, etc.), hide them.
    for obj in created_fbx:
        if getattr(obj, "type", None) not in ("ARMATURE", "MESH"):
            obj.hide_viewport = True
            obj.hide_render = True

    fbx_armature.name = f"{label}_FBX_Armature"
    for mesh in fbx_meshes:
        mesh.name = f"{label}_{mesh.name}"
    imported_scale = tuple(float(value) for value in fbx_armature.scale)
    unit_correction = normalize_imported_mixamo_units(fbx_armature, fbx_meshes)
    bpy.context.view_layer.update()
    _calibration = calibrate_imported_fbx(
        fbx_armature,
        unit_correction=unit_correction,
        original_scale=imported_scale,
    )
    rig_profile = build_mixamo_rig_profile(fbx_path, fbx_armature, _calibration, fbx_meshes)
    if not rig_profile.valid:
        raise ValueError("Invalid Mixamo rig: " + "; ".join(rig_profile.validation_errors))
    # Keep the evidence with the Blender object so audit/render tools observe
    # exactly the profile used for this import, not a reconstructed guess.
    fbx_armature["motionviewer_mixamo_profile"] = json.dumps(rig_profile.to_json(), sort_keys=True)

    return BootstrapOutput(
        smplx_armature=smplx_armature,
        fbx_armature=fbx_armature,
        fbx_meshes=fbx_meshes,
        num_frames=num_frames,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        offset=offset,
        calibration=_calibration,
        rig_profile=rig_profile,
        unit_scale=unit_scale,
        frame_start=frame_start,
        label=label,
    )
