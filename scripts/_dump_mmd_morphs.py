"""Dump a PMX's facial morphs and their current values. Local helper.

MMD models carry expressions as vertex morphs, which arrive as Blender shape keys.
The import leaves every one at 0, so whatever the face looks like in a render is
the model's *neutral* mesh — not an expression that got switched on by accident.
Deciding what to do about that needs the list of what is available.

  blender --background --python scripts/_dump_mmd_morphs.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx --output outputs/morphs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore

    from motionviewer.blender.mmd_import import import_pmx_character
    from motionviewer.blender.scene import clear_scene

    clear_scene()
    armature, meshes = import_pmx_character(bpy, args.asset, label="morph", scale=0.08)
    bpy.context.view_layer.update()

    report: dict = {"asset": str(args.asset), "meshes": []}
    for mesh in meshes:
        keys = mesh.data.shape_keys
        entry: dict = {"name": mesh.name, "shape_keys": []}
        if keys is not None:
            for block in keys.key_blocks:
                entry["shape_keys"].append(
                    {
                        "name": block.name,
                        "value": round(float(block.value), 4),
                        "min": round(float(block.slider_min), 3),
                        "max": round(float(block.slider_max), 3),
                        "mute": bool(block.mute),
                    }
                )
        report["meshes"].append(entry)

    root = None
    for obj in bpy.data.objects:
        if str(getattr(obj, "mmd_type", "NONE")) == "ROOT":
            root = obj
            break
    if root is not None:
        mmd_root = getattr(root, "mmd_root", None)
        groups: dict = {}
        for attribute in ("vertex_morphs", "bone_morphs", "material_morphs", "uv_morphs", "group_morphs"):
            collection = getattr(mmd_root, attribute, None)
            if collection is None:
                continue
            groups[attribute] = [
                {"name": item.name, "category": str(getattr(item, "category", ""))} for item in collection
            ]
        report["mmd_root_morphs"] = {key: value for key, value in groups.items() if value}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for mesh in report["meshes"]:
        nonzero = [key for key in mesh["shape_keys"] if abs(key["value"]) > 1e-6]
        print(f"{mesh['name']}: {len(mesh['shape_keys'])} shape keys, {len(nonzero)} non-zero")
        for key in nonzero[:10]:
            print(f"   ACTIVE {key['name']} = {key['value']}")
    counts = {key: len(value) for key, value in report.get("mmd_root_morphs", {}).items()}
    print(f"morph groups: {counts}")


if __name__ == "__main__":
    main()
