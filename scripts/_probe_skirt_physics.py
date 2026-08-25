"""Which characters actually have physics on the parts that clip?

Enabling Bullet does nothing for a garment the model's author never gave rigid bodies
to. Measured on 流萤's school uniform: 222 rigid bodies, 212 bones driven — and 0.0% of
the mesh is dominated by any of them, because the bodies are all hair, bangs and collar
while the pleated skirt is rigidly skinned to 下半身. Physics ran correctly and changed
2.3% of the pixels, all of it hair.

So "MMD does not clip" is not a property of MMD; it is a property of the model. This
reports, per character, how much of the lower-body mesh is driven by a dynamic rigid
body, which is what decides whether --bullet can help it.

  blender --background --python scripts/_probe_skirt_physics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/a26044/projects/MotionViewer")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402

from motionviewer.blender.mmd_import import ensure_mmd_tools, import_pmx_character  # noqa: E402
from motionviewer.blender.scene import clear_scene  # noqa: E402

# MMD rigid body modes: 0 static (collision only), 1 dynamic, 2 dynamic tracking a bone.
DYNAMIC_MODES = {1, 2}


def probe(path: Path) -> dict:
    clear_scene()
    armature, meshes = import_pmx_character(bpy, path, label="sp", physics=True)
    ensure_mmd_tools()
    from mmd_tools.core.model import FnModel, Model  # type: ignore

    root = FnModel.find_root_object(armature)
    if root is None:
        return {"name": path.parent.name, "error": "no MMD root"}
    model = Model(root)
    dynamic_bones = {
        rb.mmd_rigid.bone
        for rb in model.rigidBodies()
        if rb.mmd_rigid.bone and int(rb.mmd_rigid.type) in DYNAMIC_MODES
    }
    # Only vertex groups that are actually bones. ``mmd_tools`` adds a group called
    # ``mmd_edge_scale`` carrying the per-vertex outline width at weight 1.0 on every
    # vertex, so "the group with the highest weight" is that group on every vertex of
    # every model — which is why an earlier version of this reported 0.0% for all
    # nineteen characters while they had hundreds of dynamic bones.
    bone_names = {bone.name for bone in armature.data.bones}
    total = dynamic = 0
    lower_total = lower_dynamic = 0
    for mesh in meshes:
        names = {group.index: group.name for group in mesh.vertex_groups}
        for vertex in mesh.data.vertices:
            weighted = [g for g in vertex.groups if names.get(g.group, "") in bone_names]
            if not weighted:
                continue
            best = max(weighted, key=lambda g: g.weight)
            driven = names.get(best.group, "") in dynamic_bones
            total += 1
            dynamic += int(driven)
            # Waist down to mid-thigh in the model's own space: where a skirt lives.
            if -0.5 < vertex.co.z < 0.35:
                lower_total += 1
                lower_dynamic += int(driven)
    return {
        "name": path.parent.name,
        "dynamic_bones": len(dynamic_bones),
        "mesh_share": dynamic / max(total, 1),
        "lower_share": lower_dynamic / max(lower_total, 1),
        "lower_vertices": lower_total,
    }


def main() -> None:
    paths = [
        Path(line.strip()) for line in Path("/tmp/charlist.txt").read_text().splitlines() if line.strip()
    ]
    report = []
    for path in paths:
        try:
            info = probe(path)
        except Exception as exc:  # noqa: BLE001
            info = {"name": path.parent.name, "error": f"{type(exc).__name__}: {exc}"[:100]}
        report.append(info)
        if "error" in info:
            print(f"SKIRT {info['name'][:26]:28s} :: {info['error']}", flush=True)
        else:
            print(
                f"SKIRT {info['name'][:26]:28s} :: dynamic bones {info['dynamic_bones']:3d} "
                f"mesh {info['mesh_share'] * 100:5.1f}% lower-body {info['lower_share'] * 100:5.1f}%",
                flush=True,
            )
    (ROOT / "outputs/skirt_physics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("SKIRT wrote outputs/skirt_physics.json")


if __name__ == "__main__":
    main()
