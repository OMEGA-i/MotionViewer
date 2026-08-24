"""Check every downloaded PMX against what the MMD retarget actually needs.

A model is usable here only if its armature carries the Japanese bone names the
mapping is written against (``左手首``, ``右ひざ`` and the rest). Genshin/Star Rail
rips normally do, but weapon models are separate PMX files with a handful of bones and
no body at all, and some character packs ship a variant rig. Deciding that by opening
each one in a GUI does not scale to eighteen packs, so this reports the counts and the
missing roles per file and leaves the judgement to one table.

  blender --background --python scripts/_probe_characters.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/a26044/projects/MotionViewer")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402

from motionviewer.blender.mmd_import import import_pmx_character  # noqa: E402
from motionviewer.blender.retarget.mmd import REQUIRED_MMD_BONES  # noqa: E402
from motionviewer.blender.scene import clear_scene  # noqa: E402

# The retarget already declares what it cannot do without; reuse that list rather
# than writing a second one that can drift away from it.
ESSENTIAL = REQUIRED_MMD_BONES


def probe(path: Path) -> dict:
    clear_scene()
    try:
        result = import_pmx_character(bpy, path, label="probe")
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    armature = result.armature if hasattr(result, "armature") else result[0]
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    names = {bone.name for bone in armature.data.bones}
    missing = [bone for bone in ESSENTIAL if bone not in names]
    zs = [float((armature.matrix_world @ bone.matrix_local).translation.z) for bone in armature.data.bones]
    return {
        "path": str(path),
        "bones": len(names),
        "meshes": len(meshes),
        "vertices": sum(len(obj.data.vertices) for obj in meshes),
        "height_m": round(max(zs) - min(zs), 4) if zs else 0.0,
        "missing": missing,
        "shape_keys": sum(
            len(obj.data.shape_keys.key_blocks) if obj.data.shape_keys else 0 for obj in meshes
        ),
    }


def main() -> None:
    root = ROOT / "assets/fbx/pmx"
    files = sorted(root.rglob("*.pmx"))
    report = []
    for path in files:
        info = probe(path)
        report.append(info)
        rel = path.relative_to(root)
        if "error" in info:
            print(f"PROBE {rel} :: ERROR {info['error']}", flush=True)
            continue
        verdict = "OK" if not info["missing"] else f"missing {len(info['missing'])}"
        print(
            f"PROBE {rel} :: bones {info['bones']:4d} verts {info['vertices']:7d} "
            f"h {info['height_m']:.3f} keys {info['shape_keys']:3d} :: {verdict}",
            flush=True,
        )
    (ROOT / "outputs/character_probe.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/character_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"PROBE wrote outputs/character_probe.json ({len(report)} files)")


if __name__ == "__main__":
    main()
