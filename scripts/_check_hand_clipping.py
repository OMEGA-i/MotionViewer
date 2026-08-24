"""Is the hand inside the clothing when the character just stands there?

The acceptance test for a downloaded character is whether a plain standing pose already
buries the hands in a skirt or a coat. Eyeballing a contact sheet does not settle it —
a hand drawn *over* a coat and a hand *inside* it look nearly the same at thumbnail
size, and the difference is what decides whether the character is usable.

So it is measured. Hand vertices are found by vertex weight to the wrist and finger
bones, and each one is tested for being inside any other mesh by casting a ray and
reading the normal of the first face it hits: a normal pointing the same way as the ray
means the ray left through a backface, so the point started inside. Three directions
are cast and the majority wins, which tolerates the open boundaries these garment
meshes are full of — a single ray through a hole reports "outside" for a point that is
plainly buried.

  blender --background --python scripts/_check_hand_clipping.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/a26044/projects/MotionViewer")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz  # noqa: E402
from motionviewer.blender.scene import clear_scene  # noqa: E402

STANDING = Path("/Users/a26044/Motion data/t2m_demo400/smplx/quiet/017949.smplx.npz")
HAND_BONES = ("手首", "親指", "人指", "中指", "薬指", "小指")
DIRECTIONS = (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0.577, 0.577, 0.577)))
#: Above this share of buried hand vertices the hand reads as inside the cloth.
BURIED_LIMIT = 0.15


def hand_vertices(obj, armature) -> list:
    """World-space positions of vertices mostly weighted to a hand."""
    groups = {group.index for group in obj.vertex_groups if any(token in group.name for token in HAND_BONES)}
    if not groups:
        return []
    world = obj.matrix_world
    out = []
    for vertex in obj.data.vertices:
        weight = sum(g.weight for g in vertex.groups if g.group in groups)
        if weight > 0.6:
            out.append(world @ vertex.co)
    return out


def is_inside(point: Vector, others: list) -> bool:
    votes = 0
    for direction in DIRECTIONS:
        inside = False
        for other in others:
            local = other.matrix_world.inverted() @ point
            hit, _, normal, _ = other.ray_cast(local, direction)
            if hit and normal.dot(direction) > 0.0:
                inside = True
                break
        votes += int(inside)
    return votes >= 2


def main() -> None:
    paths = [Path(line) for line in Path("/tmp/charlist.txt").read_text().split() if line]
    report = []
    for path in paths:
        clear_scene()
        name = path.parent.name
        try:
            actor = create_fbx_actor_from_npz(STANDING, label="hc", fbx_path=path, retarget_mode="direct")
        except Exception as exc:  # noqa: BLE001
            print(f"HAND {name} :: import failed {type(exc).__name__}", flush=True)
            report.append({"name": name, "path": str(path), "error": str(exc)[:120]})
            continue
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        meshes = list(actor.mesh_objects)
        total = buried = 0
        for obj in meshes:
            points = hand_vertices(obj, depsgraph)
            if not points:
                continue
            others = [m.evaluated_get(depsgraph) for m in meshes if m is not obj]
            # Sampling: a full hand is a few thousand vertices and three rays each
            # against a dozen garment meshes is minutes per character for no extra
            # certainty, so every eighth vertex is tested.
            for point in points[::8]:
                total += 1
                buried += int(is_inside(point, others))
        share = (buried / total) if total else 0.0
        verdict = "CLIPS" if share > BURIED_LIMIT else "ok"
        print(
            f"HAND {name} :: {buried}/{total} hand verts inside cloth = {share * 100:.1f}% :: {verdict}",
            flush=True,
        )
        report.append({"name": name, "path": str(path), "tested": total, "buried": buried, "share": share})
    (ROOT / "outputs/hand_clipping.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("HAND wrote outputs/hand_clipping.json")


if __name__ == "__main__":
    main()
