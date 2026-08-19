"""Publication stills for one clip: a filmstrip and a motion-trail overlay.

The two figures a text-to-motion paper actually uses:

``strip``
    K evenly spaced frames side by side, on transparency, so the reader reads the
    motion left to right. Frames are picked by *motion*, not by clock time —
    evenly spaced samples of a clip that pauses waste half the figure on a static
    pose.

``trail``
    The same frames composited into one image, oldest faintest. Works when the
    root travels; on an in-place clip the poses land on top of each other, so the
    strip is the honest choice there and this script says so.

Rendered at whatever resolution is asked for and with a transparent background,
because a paper figure gets placed on the page's own background.

  blender --background --python scripts/render_paper_figure.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx --motion <clip>/smplx_params.npz \
    --output outputs/figure/<name> --count 6 --resolution 1400
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


def _pick_frames(joints, count: int) -> list[int]:
    """Frames spaced by accumulated joint travel, not by time.

    A clip that holds still for a second would otherwise spend a third of the
    figure on one pose.
    """
    import numpy as np

    points = np.asarray(joints, dtype=np.float64)
    step = np.linalg.norm(np.diff(points, axis=0), axis=-1).sum(axis=1)
    travel = np.concatenate([[0.0], np.cumsum(step)])
    if travel[-1] <= 1e-9:
        return list(np.linspace(0, len(points) - 1, count, dtype=int))
    targets = np.linspace(0.0, travel[-1], count)
    return [int(np.searchsorted(travel, target)) for target in targets]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=1400, help="Height per frame")
    parser.add_argument("--aspect", type=float, default=0.72)
    parser.add_argument("--view", default="three_quarter")
    parser.add_argument("--in-place", action="store_true", help="Zero root translation")
    parser.add_argument("--no-spring", action="store_true")
    parser.add_argument("--ground", action="store_true", help="Opaque floor instead of transparency")
    parser.add_argument("--expression", default="smile", help="Facial expression preset, or 'none'")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np

    from motionviewer.blender.camera import add_camera_for_bounds
    from motionviewer.blender.mmd_expression import apply_expression
    from motionviewer.blender.mmd_spring import apply_secondary_motion
    from motionviewer.blender.mmd_toon import add_ground, add_outline, add_toon_lighting, apply_toon_shading
    from motionviewer.blender.render import _normalize_engine
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import clear_scene, setup_world

    toon_dir = ROOT / ".local/blender_mmd_tools/mmd_tools/externals/MikuMikuDance"
    for name in ("toon01.bmp", "toon05.bmp"):
        source = toon_dir / name
        target = ROOT / name
        if source.is_file() and not target.exists():
            target.symlink_to(source)

    clear_scene()
    setup_world(transparent=not args.ground)

    overrides = None
    if args.in_place:
        with np.load(args.motion, allow_pickle=False) as payload:
            overrides = {"transl": np.zeros_like(payload["transl"])}

    actor = create_fbx_actor_from_npz(
        args.motion,
        label="figure",
        fbx_path=args.asset,
        bone_map="mmd",
        gender="female",
        fbx_scale=0.08,
        retarget_mode="direct",
        motion_overrides=overrides,
        mmd_physics=not args.no_spring,
        mmd_morphs=args.expression != "none",
    )
    scene = bpy.context.scene
    frame_start = int(scene.frame_start)

    with np.load(args.motion, allow_pickle=False) as payload:
        joints = np.asarray(payload["joints22"], dtype=np.float64)
    total = len(joints)
    picks = [frame_start + offset for offset in _pick_frames(joints, args.count)]

    if args.expression != "none":
        print(
            f"expression: {json.dumps(apply_expression(actor.mesh_objects, args.expression), ensure_ascii=False)}"
        )

    if not args.no_spring:
        info = apply_secondary_motion(bpy, actor.armature, frame_start=frame_start, num_frames=total)
        print(f"spring: {json.dumps(info, ensure_ascii=False)}")

    def mesh_bounds(frames: list[int]) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Union of the mesh bounds, plus the lowest vertex height of each frame."""
        from mathutils import Vector  # type: ignore

        mins = np.full(3, 1e9)
        maxs = np.full(3, -1e9)
        lowest: list[float] = []
        for frame in frames:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            frame_low = 1e9
            for mesh in actor.mesh_objects:
                evaluated = mesh.evaluated_get(depsgraph)
                matrix = evaluated.matrix_world
                for corner in evaluated.bound_box:
                    world = matrix @ Vector(corner)
                    mins = np.minimum(mins, (world.x, world.y, world.z))
                    maxs = np.maximum(maxs, (world.x, world.y, world.z))
                    frame_low = min(frame_low, float(world.z))
            lowest.append(frame_low)
        return mins, maxs, lowest

    mins, maxs, sole_heights = mesh_bounds(picks)

    report = apply_toon_shading(actor.mesh_objects)
    print(f"toon: {len(report['shaded'])} shaded, {len(report['unlit'])} unlit, {len(report['face'])} face")
    add_toon_lighting(mins.tolist(), maxs.tolist())
    add_outline(actor.mesh_objects)
    if args.ground:
        # Lower quartile, not the minimum: the retarget grounds the *rest* pose, so a
        # posed character sits several centimetres high and a floor at the minimum
        # leaves every other frame visibly hovering. See add_ground's docstring.
        floor_z = float(np.percentile(sole_heights, 25))
        print(f"floor: {floor_z:.4f} m (frame soles {min(sole_heights):.4f}..{max(sole_heights):.4f})")
        add_ground(mins.tolist(), maxs.tolist(), plane_z=floor_z)

    width = max(int(round(args.resolution * args.aspect)), 16)
    add_camera_for_bounds(
        mins.tolist(), maxs.tolist(), preset=args.view, margin=1.05, resolution=(width, args.resolution)
    )
    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = width
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = not args.ground
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if not args.ground else "RGB"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 128

    args.output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(picks):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        scene.render.filepath = str(args.output / f"pick_{index:02d}_f{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"rendered pick {index} at frame {frame}")

    # Source space is Y-up, so the ground plane is (x, z) — not (x, y).
    from motionviewer.core.smplx_fk import source_to_blender

    ground_track = source_to_blender(joints[:, 0])[:, :2]
    travel = float(np.linalg.norm(ground_track[-1] - ground_track[0]))
    (args.output / "figure.json").write_text(
        json.dumps(
            {
                "motion": str(args.motion),
                "asset": str(args.asset),
                "frames": picks,
                "clip_frames": total,
                "root_travel_m": travel,
                # A trail overlay only reads when the poses do not sit on top of
                # each other, which needs the root to have moved.
                "trail_recommended": bool(travel > 0.6 and not args.in_place),
                "resolution": [width, args.resolution],
                "transparent": not args.ground,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
