from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.smplx_actor import BODY_POSE_BONES, SmplxActor
from .addon_probe import require_smplx_addon


def create_smplx_actor_from_npz(
    path: str | Path,
    *,
    label: str,
    body_model: str = "smplx",
    gender: str = "neutral",
    frame_start: int = 1,
    unit_scale: float = 1.0,
    layout_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    motion_overrides: dict | None = None,
) -> SmplxActor:
    """Create and animate a real SMPL-X mesh actor using the Blender addon.

    The current source files are body-only axis-angle SMPL-X. Hands, face, jaw,
    eyes, and expression are left at the addon's neutral defaults.
    """

    import bpy  # type: ignore

    require_smplx_addon()
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
    betas = np.asarray(data["betas"], dtype=np.float32)

    before = set(bpy.data.objects)
    configure_smplx_tool(gender, body_model=body_model)
    result = bpy.ops.scene.smplx_add_gender()
    if "CANCELLED" in set(result):
        raise RuntimeError("SMPL-X addon cancelled model creation; check model assets and addon preferences")

    created = [obj for obj in bpy.data.objects if obj not in before]
    armature = _find_armature(created) or bpy.context.object
    if armature is None or getattr(armature, "type", None) != "ARMATURE":
        raise RuntimeError("SMPL-X addon did not create an armature")
    armature.name = f"{label}_SMPLX_Armature"
    meshes = [obj for obj in created if getattr(obj, "type", None) == "MESH"]
    for mesh in meshes:
        mesh.name = f"{label}_{mesh.name}"

    _apply_betas(meshes, betas)
    _keyframe_body(armature, global_orient, body_pose, transl, frame_start, unit_scale, layout_offset)
    return SmplxActor(label=label, armature=armature, mesh_objects=meshes)


def configure_smplx_tool(gender: str, *, body_model: str = "smplx") -> None:
    import bpy  # type: ignore

    tool = getattr(bpy.context.window_manager, "smplx_tool", None)
    if tool is None:
        raise RuntimeError("SMPL-X addon window manager properties were not found")
    if hasattr(tool, "body_model"):
        values = _enum_values(tool, "body_model")
        if values and body_model not in values:
            raise ValueError(f"SMPL addon does not support body model {body_model!r}")
        tool.body_model = body_model
    if hasattr(tool, "smplx_gender"):
        values = _enum_values(tool, "smplx_gender")
        candidates = [gender, gender.upper(), gender.lower(), "neutral", "NEUTRAL", "female", "FEMALE"]
        for candidate in candidates:
            if not values or candidate in values:
                tool.smplx_gender = candidate
                break
    if hasattr(tool, "smplx_handpose"):
        tool.smplx_handpose = (
            "flat" if "flat" in _enum_values(tool, "smplx_handpose") else tool.smplx_handpose
        )


def _apply_betas(meshes: list[Any], betas: np.ndarray) -> None:
    for mesh in meshes:
        shape_keys = getattr(getattr(mesh, "data", None), "shape_keys", None)
        key_blocks = getattr(shape_keys, "key_blocks", None)
        if key_blocks is None:
            continue
        for idx, beta in enumerate(betas):
            candidate_names = [f"Shape{idx:03d}", f"shape{idx:03d}", f"beta_{idx}", f"betas_{idx}"]
            for name in candidate_names:
                if name in key_blocks:
                    key_blocks[name].value = float(beta)
                    break


def _keyframe_body(
    armature: Any,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    frame_start: int,
    unit_scale: float,
    layout_offset: tuple[float, float, float],
) -> None:
    import bpy  # type: ignore
    from bl_ext.user_default.smplx_blender_addon.utils.pose import set_pose_from_rodrigues  # type: ignore

    offset = np.asarray(layout_offset, dtype=np.float32)
    for idx in range(global_orient.shape[0]):
        frame = frame_start + idx
        bpy.context.scene.frame_set(frame)
        set_pose_from_rodrigues(armature, "pelvis", global_orient[idx])
        for bone_idx, bone_name in enumerate(BODY_POSE_BONES):
            if bone_idx < body_pose.shape[1] and bone_name in armature.pose.bones:
                set_pose_from_rodrigues(armature, bone_name, body_pose[idx, bone_idx])
        loc = source_transl_to_blender(transl[idx], unit_scale) + offset
        armature.location = tuple(float(v) for v in loc)
        armature.keyframe_insert(data_path="location", frame=frame)
        for bone in armature.pose.bones:
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def source_transl_to_blender(value: np.ndarray, unit_scale: float = 1.0) -> np.ndarray:
    """Convert a y-up source translation vector to Blender z-up space."""
    return np.asarray([value[0], -value[2], value[1]], dtype=np.float32) * unit_scale


def _find_armature(objects: list[Any]) -> Any | None:
    for obj in objects:
        if getattr(obj, "type", None) == "ARMATURE":
            return obj
    return None


def _enum_values(tool: Any, prop_name: str) -> set[str]:
    try:
        prop = tool.bl_rna.properties[prop_name]
        return {item.identifier for item in prop.enum_items}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------

from .backend import MaterialPolicy  # noqa: E402


class SmplxAddonBackend:
    backend_id = "blender_smplx_addon"
    description = "Native SMPL-X mesh via the SMPL-X for Blender addon."
    material_policy = MaterialPolicy.APPLY_MATERIAL

    def create_actor(
        self,
        path,
        *,
        label,
        gender="neutral",
        unit_scale=1.0,
        layout_offset=(0, 0, 0),
        body_config=None,
        motion_overrides=None,
    ):
        return create_smplx_actor_from_npz(
            path,
            label=label,
            body_model="smplx",
            gender=gender,
            unit_scale=unit_scale,
            layout_offset=layout_offset,
            motion_overrides=motion_overrides,
        )

    def resolve_paths(self, body_config, base):
        return body_config

    def validate_config(self, body_config):
        return []
