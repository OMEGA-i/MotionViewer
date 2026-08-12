from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ADDON_MODULE = "bl_ext.user_default.smplx_blender_addon"


@dataclass
class SmplxAddonStatus:
    available: bool
    enabled: bool
    module: str | None = None
    version: tuple[int, ...] | None = None
    operators: list[str] = field(default_factory=list)
    wm_properties: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "enabled": self.enabled,
            "module": self.module,
            "version": list(self.version) if self.version else None,
            "operators": self.operators,
            "wm_properties": self.wm_properties,
            "error": self.error,
        }


def probe_smplx_addon() -> SmplxAddonStatus:
    try:
        import addon_utils  # type: ignore
        import bpy  # type: ignore
    except Exception as exc:
        return SmplxAddonStatus(False, False, error=f"Blender Python is not available: {exc}")

    for module in addon_utils.modules():
        bl_info = getattr(module, "bl_info", {})
        text = f"{module.__name__} {bl_info.get('name', '')}".lower()
        if "smpl" not in text:
            continue
        enabled = addon_utils.check(module.__name__)[1]
        operators = _available_operators(bpy)
        wm_properties = _wm_tool_properties(bpy)
        return SmplxAddonStatus(
            available=True,
            enabled=enabled,
            module=module.__name__,
            version=tuple(bl_info.get("version", ())),
            operators=operators,
            wm_properties=wm_properties,
        )
    return SmplxAddonStatus(False, False, error="SMPL-X for Blender addon was not found")


def require_smplx_addon() -> SmplxAddonStatus:
    status = probe_smplx_addon()
    if not status.available or not status.enabled:
        raise RuntimeError(status.error or "SMPL-X for Blender addon is not enabled")
    if "scene.smplx_add_gender" not in status.operators:
        raise RuntimeError("SMPL-X addon is enabled but scene.smplx_add_gender is unavailable")
    return status


def _available_operators(bpy: Any) -> list[str]:
    candidates = [
        ("scene", "smplx_add_gender"),
        ("object", "smplx_add_animation"),
        ("object", "smplx_load_pose"),
        ("object", "smplx_set_poseshapes"),
        ("object", "smplx_update_joint_locations"),
    ]
    found = []
    for namespace, name in candidates:
        if hasattr(getattr(bpy.ops, namespace), name):
            found.append(f"{namespace}.{name}")
    return found


def _wm_tool_properties(bpy: Any) -> dict[str, Any]:
    tool = getattr(bpy.context.window_manager, "smplx_tool", None)
    if tool is None:
        return {}
    result: dict[str, Any] = {}
    for name in [
        "body_model",
        "smplx_gender",
        "smplx_version",
        "smplx_uv",
        "smplx_texture",
        "smplx_handpose",
        "smplx_corrective_poseshapes",
    ]:
        if hasattr(tool, name):
            value = getattr(tool, name)
            if isinstance(value, (str, int, float, bool)):
                result[name] = value
    return result
