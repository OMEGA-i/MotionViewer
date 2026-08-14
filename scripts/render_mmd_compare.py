"""Render the retargeted character beside the SMPL-X source skeleton.

Same scene, same camera, same light, so a pose that looks wrong can be traced to
either the source motion or the retarget rather than argued about.

  blender --background --python scripts/render_mmd_compare.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
    --motion <clip>/smplx_params.npz \
    --output outputs/compare/<name> --views three_quarter,front
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

_BONE_RADIUS_M = 0.022
_JOINT_RADIUS_M = 0.032


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", default="three_quarter")
    parser.add_argument("--frames", type=int, default=0, help="0 means the whole clip")
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument(
        "--panels",
        default="skeleton,character",
        help="Which passes to render. Both share one camera so frames can be stacked.",
    )
    parser.add_argument("--faithful", action="store_true", help="Disable the polish pass")
    parser.add_argument("--caption", default="")
    parser.add_argument("--toon", action="store_true", help="Cel shading, outline, floor shadow")
    parser.add_argument("--no-outline", action="store_true", help="With --toon, skip the outline shell")
    parser.add_argument("--no-ground", action="store_true", help="With --toon, skip the floor")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np
    from mathutils import Vector  # type: ignore

    from motionviewer.blender.camera import add_camera_for_bounds
    from motionviewer.blender.mmd_toon import add_ground, add_outline, add_toon_lighting, apply_toon_shading
    from motionviewer.blender.render import _normalize_engine
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import add_lighting, clear_scene, setup_world
    from motionviewer.core.smplx_fk import SMPLX_BODY22_PARENTS, source_to_blender

    toon_dir = ROOT / ".local/blender_mmd_tools/mmd_tools/externals/MikuMikuDance"
    for name in ("toon01.bmp", "toon05.bmp"):
        source = toon_dir / name
        target = ROOT / name
        if source.is_file() and not target.exists():
            target.symlink_to(source)

    views = [value.strip() for value in args.views.split(",") if value.strip()]
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]

    clear_scene()
    setup_world(transparent=False)


    actor = create_fbx_actor_from_npz(
        args.motion,
        label="yoimiya",
        fbx_path=args.asset,
        bone_map="mmd",
        gender="female",
        fbx_scale=0.08,
        retarget_mode="direct",
        mmd_polish={"enabled": not args.faithful},
    )
    scene = bpy.context.scene
    frame_start = int(scene.frame_start)

    with np.load(args.motion, allow_pickle=False) as payload:
        joints = np.asarray(payload["joints22"], dtype=np.float64)
    total = len(joints) if args.frames <= 0 else min(args.frames, len(joints))
    frame_end = frame_start + total - 1

    transfer = json.loads(actor.armature.get("motionviewer_mmd_transfer", "{}"))

    # ---- source skeleton -----------------------------------------------------
    # The skeleton is built at the character's own position, not beside her:
    # both passes then share one camera, so stacking the two image sequences
    # compares the same pose from the same angle instead of two viewpoints.
    skeleton_objects: list = []
    if "skeleton" in panels:
        points = source_to_blender(joints[:total])
        # Reuse the retarget's own height ratio. A pelvis-height ratio would
        # mis-scale a stylised character, whose legs and head are not human
        # proportions.
        scale = float(transfer.get("root_translation_scale", 1.0))
        points = points * scale
        scene.frame_set(frame_start)
        bpy.context.view_layer.update()
        character_root = np.asarray(actor.armature.matrix_world.translation, dtype=np.float64)
        points[:, :, 2] -= float(points[:, :, 2].min())
        points[:, :, 0] += float(character_root[0]) - float(points[0, 0, 0])
        points[:, :, 1] += float(character_root[1]) - float(points[0, 0, 1])

        material = bpy.data.materials.new("SourceSkeleton")
        material.use_nodes = True
        principled = material.node_tree.nodes["Principled BSDF"]
        principled.inputs["Base Color"].default_value = (0.16, 0.20, 0.30, 1.0)
        principled.inputs["Roughness"].default_value = 0.55

        def add_primitive(kind: str, name: str) -> object:
            if kind == "cylinder":
                bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, vertices=12)
            else:
                bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=12, ring_count=8)
            obj = bpy.context.active_object
            obj.name = name
            obj.data.materials.append(material)
            obj.rotation_mode = "QUATERNION"
            return obj

        bones = [(child, parent) for child, parent in enumerate(SMPLX_BODY22_PARENTS) if parent >= 0]
        segments = [add_primitive("cylinder", f"src_bone_{child}") for child, _ in bones]
        joint_balls = [add_primitive("sphere", f"src_joint_{index}") for index in range(22)]
        skeleton_objects = [*segments, *joint_balls]

        for offset in range(total):
            frame = frame_start + offset
            for segment, (child, parent) in zip(segments, bones, strict=True):
                head = Vector(points[offset, parent])
                tail = Vector(points[offset, child])
                direction = tail - head
                length = max(direction.length, 1e-5)
                segment.location = (head + tail) * 0.5
                segment.rotation_quaternion = direction.to_track_quat("Z", "Y")
                segment.scale = (_BONE_RADIUS_M, _BONE_RADIUS_M, length * 0.5)
                for channel in ("location", "rotation_quaternion", "scale"):
                    segment.keyframe_insert(data_path=channel, frame=frame)
            for index, ball in enumerate(joint_balls):
                ball.location = Vector(points[offset, index])
                ball.scale = (_JOINT_RADIUS_M,) * 3
                for channel in ("location", "scale"):
                    ball.keyframe_insert(data_path=channel, frame=frame)

        for obj in skeleton_objects:
            action = obj.animation_data.action if obj.animation_data else None
            if action is None:
                continue
            curves = list(getattr(action, "fcurves", None) or [])
            if not curves:
                for layer in getattr(action, "layers", ()):
                    for strip in getattr(layer, "strips", ()):
                        for channelbag in getattr(strip, "channelbags", ()):
                            curves.extend(channelbag.fcurves)
            for curve in curves:
                for keyframe in curve.keyframe_points:
                    keyframe.interpolation = "LINEAR"

    # ---- framing over the whole clip ----------------------------------------
    def scene_bounds() -> tuple[np.ndarray, np.ndarray]:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mins = np.full(3, 1e9)
        maxs = np.full(3, -1e9)
        targets = [*actor.mesh_objects, *skeleton_objects]
        for obj in targets:
            evaluated = obj.evaluated_get(depsgraph)
            matrix = evaluated.matrix_world
            for corner in evaluated.bound_box:
                world = matrix @ Vector(corner)
                mins = np.minimum(mins, (world.x, world.y, world.z))
                maxs = np.maximum(maxs, (world.x, world.y, world.z))
        return mins, maxs

    mins = np.full(3, 1e9)
    maxs = np.full(3, -1e9)
    for frame in sorted({frame_start, (frame_start + frame_end) // 2, frame_end}):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        frame_mins, frame_maxs = scene_bounds()
        mins = np.minimum(mins, frame_mins)
        maxs = np.maximum(maxs, frame_maxs)

    if args.toon:
        report = apply_toon_shading(actor.mesh_objects)
        print(f"toon: {len(report['shaded'])} shaded, {len(report['unlit'])} unlit, {len(report['face'])} face")
        add_toon_lighting(mins.tolist(), maxs.tolist())
        if not args.no_outline:
            add_outline(actor.mesh_objects)
        if not args.no_ground:
            add_ground(mins.tolist(), maxs.tolist())
    else:
        add_lighting(mins.tolist(), maxs.tolist())
    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    args.output.mkdir(parents=True, exist_ok=True)

    for view in views:
        for camera in [obj for obj in bpy.data.objects if obj.type == "CAMERA"]:
            bpy.data.objects.remove(camera, do_unlink=True)
        add_camera_for_bounds(
            mins.tolist(),
            maxs.tolist(),
            preset=view,
            margin=1.1,
            resolution=(args.resolution, args.resolution),
        )
        for panel in panels:
            for obj in actor.mesh_objects:
                obj.hide_render = panel != "character"
            for obj in skeleton_objects:
                obj.hide_render = panel != "skeleton"
            directory = args.output / panel / view
            directory.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(directory / "frame_")
            bpy.ops.render.render(animation=True)
            print(f"rendered {panel}/{view}: {total} frames")

    (args.output / "info.json").write_text(
        json.dumps(
            {
                "motion": str(args.motion),
                "asset": str(args.asset),
                "caption": args.caption,
                "frames": total,
                "views": views,
                "panels": panels,
                "faithful": bool(args.faithful),
                "toon": bool(args.toon),
                "transfer": transfer.get("polish", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
