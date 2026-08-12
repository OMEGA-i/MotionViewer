"""FBX skeleton rendering backend for MotionViewer.

Creates a hidden SMPL-X armature as a pose driver, imports an FBX character,
and retargets SMPL-X motion to the FBX skeleton frame-by-frame. Returns a
``SmplxActor`` compatible with all downstream styling, ghosting, ground, and
camera effects.

The pipeline stages are in ``blender/retarget/``. This module is a thin shim
that re-exports the public API and the ``FBXSkeletonBackend``.
"""

from __future__ import annotations

from pathlib import Path

from .backend import MaterialPolicy

# Re-export bone-map symbols for backward-compatible test imports.
from .retarget._resolve import (
    BONE_MAP_PRESETS,
    resolve_bone_map,
)
from .retarget.export import export_fbx_animation, validate_fbx_roundtrip
from .retarget.pipeline import create_fbx_actor_from_npz

__all__ = [
    "BONE_MAP_PRESETS",
    "FBXSkeletonBackend",
    "create_fbx_actor_from_npz",
    "export_fbx_animation",
    "validate_fbx_roundtrip",
    "resolve_bone_map",
]


class FBXSkeletonBackend:
    backend_id = "fbx_skeleton"
    description = "FBX character driven by retargeted SMPL-X motion."
    material_policy = MaterialPolicy.PRESERVE_MATERIAL

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
        cfg = body_config or {}
        return create_fbx_actor_from_npz(
            path,
            label=label,
            fbx_path=cfg.get("fbx_path", ""),
            bone_map=cfg.get("bone_map", "auto"),
            gender=gender,
            unit_scale=unit_scale,
            fbx_scale=float(cfg.get("fbx_scale", 1.0)),
            retarget_mode=cfg.get("retarget_mode", "quality"),
            layout_offset=layout_offset,
            motion_overrides=motion_overrides,
        )

    def resolve_paths(self, body_config, base):
        cfg = dict(body_config)
        fbx = cfg.get("fbx_path")
        if fbx and not Path(fbx).is_absolute():
            cfg["fbx_path"] = str((base / fbx).resolve())
        return cfg

    def validate_config(self, body_config):
        errors = []
        if not body_config.get("fbx_path"):
            errors.append("fbx_path is required for fbx_skeleton backend")
        return errors
