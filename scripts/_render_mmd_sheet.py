"""Render a view x frame contact sheet of the MMD retarget. Local helper.

One retarget, many cameras, so arm and shoulder quality can be judged from more
than a single silhouette.

  blender --background --python scripts/_render_mmd_sheet.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
    --motion data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz \
    --output outputs/sheet --views front,side,three_quarter --frames 1,12,24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", default="front,side,three_quarter")
    parser.add_argument("--frames", default="1,12,24")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--in-place", action="store_true", help="Zero transl so the camera can stay framed")
    parser.add_argument("--zoom", choices=("full", "upper", "arms"), default="full")
    parser.add_argument("--resolution", type=int, default=700)
    parser.add_argument("--faithful", action="store_true", help="Disable the polish pass")
    parser.add_argument("--abduction", type=float, default=None, help="Override arm abduction degrees")
    parser.add_argument("--toon", action="store_true", help="Cel shading, toon light rig")
    parser.add_argument("--outline", action="store_true", help="Inverted-hull outline")
    parser.add_argument("--ground", action="store_true", help="Floor that receives the shadow")
    parser.add_argument("--shadow-threshold", type=float, default=0.42)
    parser.add_argument("--shadow-depth", type=float, default=0.55)
    parser.add_argument("--rim", type=float, default=0.18)
    parser.add_argument("--sphere", type=float, default=0.55)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np

    from motionviewer.blender.camera import add_camera_for_bounds
    from motionviewer.blender.mmd_toon import (
        ToonStyle,
        add_ground,
        add_outline,
        add_toon_lighting,
        apply_toon_shading,
    )
    from motionviewer.blender.render import _normalize_engine
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import add_lighting, clear_scene, setup_world

    toon_dir = ROOT / ".local/blender_mmd_tools/mmd_tools/externals/MikuMikuDance"
    for name in ("toon01.bmp", "toon05.bmp"):
        source = toon_dir / name
        target = ROOT / name
        if source.is_file() and not target.exists():
            target.symlink_to(source)

    views = [view.strip() for view in args.views.split(",") if view.strip()]
    frames = [int(value) for value in args.frames.split(",") if value.strip()]

    clear_scene()
    setup_world(transparent=not args.ground)

    motion_overrides = None
    if args.identity:
        from motionviewer.core.smplx_fk import recover_rest_offsets, rest_joints_from_offsets

        with np.load(args.motion, allow_pickle=False) as payload:
            joints = np.asarray(payload["joints22"], dtype=np.float64)
            global_orient = np.asarray(payload["global_orient"], dtype=np.float64)
            body_pose = np.asarray(payload["body_pose"], dtype=np.float64)
            offsets = recover_rest_offsets(joints, global_orient, body_pose)
            rest = rest_joints_from_offsets(offsets, root=np.mean(joints[:, 0], axis=0))
            motion_overrides = {
                "body_pose": np.zeros_like(payload["body_pose"]),
                "global_orient": np.zeros_like(payload["global_orient"]),
                "transl": np.zeros_like(payload["transl"]),
                "joints22": np.repeat(rest[None, ...], len(global_orient), axis=0),
            }

    if args.in_place:
        with np.load(args.motion, allow_pickle=False) as payload:
            zeros = np.zeros_like(payload["transl"])
        motion_overrides = {**(motion_overrides or {}), "transl": zeros}

    actor = create_fbx_actor_from_npz(
        args.motion,
        label="yoimiya",
        fbx_path=args.asset,
        bone_map="mmd",
        gender="female",
        fbx_scale=0.08,
        retarget_mode="direct",
        motion_overrides=motion_overrides,
        mmd_polish={
            "enabled": not args.faithful,
            **({} if args.abduction is None else {"arm_abduction_degrees": args.abduction}),
        },
    )
    scene = bpy.context.scene

    def mesh_bounds() -> tuple[np.ndarray, np.ndarray]:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mins = np.full(3, 1e9)
        maxs = np.full(3, -1e9)
        for mesh in actor.mesh_objects:
            evaluated = mesh.evaluated_get(depsgraph)
            evaluated_mesh = evaluated.to_mesh()
            try:
                matrix = evaluated.matrix_world
                for vertex in evaluated_mesh.vertices:
                    world = matrix @ vertex.co
                    mins = np.minimum(mins, (world.x, world.y, world.z))
                    maxs = np.maximum(maxs, (world.x, world.y, world.z))
            finally:
                evaluated.to_mesh_clear()
        return mins, maxs

    mins = np.full(3, 1e9)
    maxs = np.full(3, -1e9)
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        frame_mins, frame_maxs = mesh_bounds()
        mins = np.minimum(mins, frame_mins)
        maxs = np.maximum(maxs, frame_maxs)

    style = ToonStyle(
        shadow_threshold=args.shadow_threshold,
        shadow_depth=args.shadow_depth,
        rim_strength=args.rim,
        sphere_strength=args.sphere,
    )
    if args.toon:
        report = apply_toon_shading(actor.mesh_objects, style=style)
        print(
            f"toon: {len(report['shaded'])} shaded, {len(report['unlit'])} unlit, {len(report['skipped'])} skipped"
        )
        add_toon_lighting(mins.tolist(), maxs.tolist())
    else:
        add_lighting(mins.tolist(), maxs.tolist())
    if args.outline:
        shells = add_outline(actor.mesh_objects, style=style)
        print(f"outline shells: {len(shells)}")
    if args.ground:
        add_ground(mins.tolist(), maxs.tolist())
    if args.zoom != "full":
        height = float(maxs[2] - mins[2])
        # Keep the shoulders and arms; drop the legs and skirt from the frame.
        mins[2] += height * (0.52 if args.zoom == "upper" else 0.62)
        if args.zoom == "arms":
            maxs[2] -= height * 0.10
    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = not args.ground
    scene.render.image_settings.file_format = "PNG"
    args.output.mkdir(parents=True, exist_ok=True)

    for view in views:
        for camera in [obj for obj in bpy.data.objects if obj.type == "CAMERA"]:
            bpy.data.objects.remove(camera, do_unlink=True)
        add_camera_for_bounds(
            mins.tolist(),
            maxs.tolist(),
            preset=view,
            margin=1.06,
            resolution=(args.resolution, args.resolution),
        )
        for frame in frames:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            scene.render.filepath = str(args.output / f"{view}_{frame:04d}.png")
            bpy.ops.render.render(write_still=True)
            print(f"rendered {view} frame {frame}")


if __name__ == "__main__":
    main()
