"""Import a PMX character through mmd_tools and prepare it for retarget."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_mmd_tools() -> None:
    """Register bundled mmd_tools if the operator is not already available."""
    import site
    import sys

    import bpy  # type: ignore

    root = Path(__file__).resolve().parents[3] / ".local" / "blender_mmd_tools"
    if not root.is_dir():
        raise RuntimeError(
            "mmd_tools is not enabled and .local/blender_mmd_tools is missing. "
            "Clone https://github.com/MMD-Blender/blender_mmd_tools into that path."
        )
    user_site = Path.home() / ".local/lib/python3.13/site-packages"
    for extra in (str(root), str(user_site)):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    site.addsitedir(str(user_site))
    import mmd_tools

    if not _mmd_classes_loaded(bpy):
        mmd_tools.register()
    if not _mmd_classes_loaded(bpy):
        raise RuntimeError("mmd_tools registered but PMX importer classes are unavailable")


def _mmd_classes_loaded(bpy: Any) -> bool:
    import sys

    return getattr(bpy.types, "MMD_TOOLS_OT_import_model", None) is not None and "mmd_tools" in sys.modules


def import_pmx_character(
    bpy: Any,
    path: str | Path,
    *,
    label: str,
    scale: float = 0.08,
) -> tuple[Any, list[Any]]:
    """Import a PMX file and return ``(armature, meshes)``."""
    ensure_mmd_tools()
    from mmd_tools.core.pmx.importer import PMXImporter  # type: ignore

    before = set(bpy.data.objects)
    PMXImporter().execute(
        filepath=str(path),
        types={"MESH", "ARMATURE"},
        scale=float(scale),
        clean_model=True,
        remove_doubles=False,
        fix_bone_order=True,
        fix_ik_links=False,
        apply_bone_fixed_axis=False,
        rename_LR_bones=False,
        use_underscore=False,
        use_mipmap=True,
        translator=None,
    )

    created = [obj for obj in bpy.data.objects if obj not in before]
    armatures = [obj for obj in created if getattr(obj, "type", None) == "ARMATURE"]
    meshes = [obj for obj in created if getattr(obj, "type", None) == "MESH"]
    if not armatures:
        raise RuntimeError(f"No armature found after importing {path}")
    armature = armatures[0]
    armature.name = f"{label}_MMD_Armature"
    for mesh in meshes:
        mesh.name = f"{label}_{mesh.name}"
    for obj in created:
        if getattr(obj, "type", None) not in {"ARMATURE", "MESH"}:
            obj.hide_render = True
    return armature, meshes
