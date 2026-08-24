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
from mathutils.bvhtree import BVHTree  # noqa: E402

from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz  # noqa: E402
from motionviewer.blender.scene import clear_scene  # noqa: E402

STANDING = Path("/Users/a26044/Motion data/t2m_demo400/smplx/quiet/017949.smplx.npz")
#: Fingers only. The wrist is deliberately excluded: a cuff or a glove encloses the
#: wrist by design, so counting wrist vertices as "inside cloth" flagged Yoimiya at
#: 26.8% — a character whose hands have never clipped in any render we have made. A
#: finger inside a skirt is a defect; a wrist inside a sleeve is a costume.
HAND_BONES = ("親指", "人指", "中指", "薬指", "小指")
DIRECTIONS = (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0.577, 0.577, 0.577)))
#: Threshold on the *increase* in buried fingers from the bind pose to the standing
#: pose, not on the absolute figure. Absolute buriedness measures the costume as much
#: as the problem: a glove's vertices split their weight between the finger and the
#: wrist, so the glove lands on the cloth side of the split and its own fingers read as
#: enclosed. Yoimiya scored 30.2% that way — a character whose hands have never clipped
#: in any render made here. Gloves and cuffs enclose the fingers in the bind pose too,
#: so they cancel; a skirt only closes over the hand once the arm comes down.
BURIED_RISE_LIMIT = 0.15


def hand_and_cloth(obj, depsgraph):
    """Hand vertex positions, and a BVH of everything on the body that is not the hand.

    Every one of these models is a *single* mesh object with twenty-odd materials, so
    the obvious test — cast a hand vertex against the other mesh objects — has nothing
    to cast against and reports zero buried vertices for every character no matter how
    buried they are. Both earlier versions of this script did exactly that. The skirt
    and the hand are faces of the same mesh, so the split has to be made by vertex
    weight and the tree built from the faces that are left.
    """
    groups = {group.index for group in obj.vertex_groups if any(token in group.name for token in HAND_BONES)}
    if not groups:
        return [], None
    hand_flags = [
        sum(g.weight for g in vertex.groups if g.group in groups) > 0.6 for vertex in obj.data.vertices
    ]
    if not any(hand_flags):
        return [], None

    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    world = evaluated.matrix_world
    try:
        points = [world @ vertex.co for vertex in mesh.vertices]
        hands = [p for p, is_hand in zip(points, hand_flags) if is_hand]
        # Faces with no hand vertex at all: the body and the clothing.
        polygons = [
            tuple(polygon.vertices)
            for polygon in mesh.polygons
            if not any(hand_flags[i] for i in polygon.vertices if i < len(hand_flags))
        ]
        if not polygons or not hands:
            return hands, None
        tree = BVHTree.FromPolygons(points, polygons, all_triangles=False, epsilon=0.0)
        return hands, tree
    finally:
        evaluated.to_mesh_clear()


def is_inside(point, tree) -> bool:
    """Inside if a majority of rays leave through a backface."""
    votes = 0
    for direction in DIRECTIONS:
        _, normal, _, _ = tree.ray_cast(point, direction)
        votes += int(normal is not None and normal.dot(direction) > 0.0)
    return votes >= 2


def buried_share(actor, depsgraph) -> tuple[int, int]:
    total = buried = 0
    for obj in actor.mesh_objects:
        hands, tree = hand_and_cloth(obj, depsgraph)
        if tree is None:
            continue
        for point in hands[::4]:
            total += 1
            buried += int(is_inside(point, tree))
    return buried, total


def main() -> None:
    # splitlines, not split: one of these models is named "三月七 1.0.pmx" and splitting
    # on whitespace cut its path in two, which surfaced as an import failure for the
    # character and a FileNotFoundError for a phantom entry called "1.0.pmx".
    paths = [
        Path(line.strip()) for line in Path("/tmp/charlist.txt").read_text().splitlines() if line.strip()
    ]
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
        # Bind pose first: whatever the costume encloses on its own is the baseline.
        for bone in actor.armature.pose.bones:
            bone.matrix_basis.identity()
        bpy.context.view_layer.update()
        rest_buried, rest_total = buried_share(actor, bpy.context.evaluated_depsgraph_get())

        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        posed_buried, posed_total = buried_share(actor, bpy.context.evaluated_depsgraph_get())

        rest_share = (rest_buried / rest_total) if rest_total else 0.0
        posed_share = (posed_buried / posed_total) if posed_total else 0.0
        rise = posed_share - rest_share
        verdict = "CLIPS" if rise > BURIED_RISE_LIMIT else "ok"
        print(
            f"HAND {name} :: bind {rest_share * 100:5.1f}% -> standing {posed_share * 100:5.1f}%"
            f" (rise {rise * 100:+5.1f}%) of {posed_total} finger verts :: {verdict}",
            flush=True,
        )
        report.append(
            {
                "name": name,
                "path": str(path),
                "tested": posed_total,
                "bind_share": rest_share,
                "standing_share": posed_share,
                "rise": rise,
            }
        )
    (ROOT / "outputs/hand_clipping.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("HAND wrote outputs/hand_clipping.json")


if __name__ == "__main__":
    main()
