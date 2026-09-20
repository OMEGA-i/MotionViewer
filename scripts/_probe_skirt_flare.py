"""How far the skirt stands out from the legs, measured on the skirt itself.

The obvious version of this measurement is wrong in a way worth recording. Taking "mesh
below the root bone" as the skirt gives a number dominated by the character's stride:
``センター`` sits at mid-thigh on these rigs, not the waist, so the selection is mostly
shins and feet and the widest point it finds is how far apart the legs are. Every physics
setting that shifted the pose timing shifted that number, which is how a sweep came back
reporting that *more* damping made the skirt flare 32% wider — impossible, since damping
only removes energy.

So the skirt is identified by the bones the model provides for it — Yoimiya has 96 named
``裙_*``, Liuying 112 — and the flare is the radial distance from the leg axis to the
vertices those bones drive. A skirt hanging naturally sits close to the legs; one being
flung stands off them.

  blender --background --python scripts/_probe_skirt_flare.py -- --model <pmx> --frames 10,25,40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz  # noqa: E402
from motionviewer.blender.scene import clear_scene  # noqa: E402

#: Substrings that name a skirt bone across these packs.
SKIRT_TOKENS = ("裙", "スカート", "skirt", "Skirt", "前裾", "後裾", "裾")


def skirt_vertices(actor, depsgraph) -> tuple[np.ndarray, np.ndarray]:
    """World positions of skirt-driven vertices, and of the leg bones to compare against."""
    armature = actor.armature
    skirt_bones = {b.name for b in armature.data.bones if any(t in b.name for t in SKIRT_TOKENS)}
    # Select by skirt weight, not by which group wins a dominance contest. Dominance
    # found nothing: these vertices carry a skirt weight of 0.726 at the median but the
    # legs and hips hold shares of their own, so on many of them no single skirt bone is
    # the largest. 722 vertices are skirt-driven on Yoimiya and dominance reported 0.
    points: list[np.ndarray] = []
    for mesh in actor.mesh_objects:
        names = {g.index: g.name for g in mesh.vertex_groups}
        skirt_groups = {i for i, n in names.items() if n in skirt_bones}
        wanted = [
            v.index
            for v in mesh.data.vertices
            if sum(g.weight for g in v.groups if g.group in skirt_groups) > 0.5
        ]
        if not wanted:
            continue
        evaluated = mesh.evaluated_get(depsgraph)
        data = evaluated.to_mesh()
        try:
            matrix = evaluated.matrix_world
            points += [
                np.array(tuple(matrix @ data.vertices[i].co)) for i in wanted if i < len(data.vertices)
            ]
        finally:
            evaluated.to_mesh_clear()

    legs = []
    for name in ("左足", "右足", "左ひざ", "右ひざ", "左足首", "右足首"):
        bone = armature.pose.bones.get(name)
        if bone is not None:
            legs.append(np.array(tuple(armature.matrix_world @ bone.head)))
    return np.array(points), np.array(legs)


def flare_cm(points: np.ndarray, legs: np.ndarray) -> tuple[float, float]:
    """Median and p95 radial distance from the leg axis, in centimetres."""
    if not len(points) or len(legs) < 2:
        return 0.0, 0.0
    centre = legs.mean(axis=0)
    radial = np.linalg.norm(points[:, :2] - centre[:2], axis=1)
    return float(np.median(radial)) * 100.0, float(np.percentile(radial, 95)) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--motion",
        type=Path,
        default=Path("/Users/a26044/Motion data/t2m_demo400/smplx/active/100948.smplx.npz"),
    )
    parser.add_argument("--frames", default="10,25,40")
    parser.add_argument("--bullet", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    clear_scene()
    actor = create_fbx_actor_from_npz(
        args.motion,
        label="sf",
        fbx_path=args.model,
        retarget_mode="direct",
        mmd_physics=args.bullet,
    )
    frames = [int(v) for v in args.frames.split(",") if v]
    if args.bullet:
        from motionviewer.blender.mmd_bullet import BulletSettings, build_and_bake

        info = build_and_bake(
            bpy,
            actor.armature,
            frame_start=1,
            frame_end=max(frames),
            settings=BulletSettings(settle_frames=25),
        )
        print(f"FLARE bullet built={info.get('built')} bodies={info.get('rigid_bodies')}", flush=True)

    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        points, legs = skirt_vertices(actor, bpy.context.evaluated_depsgraph_get())
        median, p95 = flare_cm(points, legs)
        print(
            f"FLARE f{frame:3d}: skirt verts {len(points):5d}  "
            f"radial median {median:5.1f} cm  p95 {p95:5.1f} cm",
            flush=True,
        )


if __name__ == "__main__":
    main()
