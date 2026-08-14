"""Headless Blender entry: retarget SMPL-X onto the local Yoimiya PMX and render.

  blender --background --python scripts/render_yoimiya.py -- \
    --motion data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
    --output outputs/yoimiya_smplx
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


def _bone_debug(armature) -> dict:
    names = ("腰", "左肩", "左腕", "左腕捩", "左ひじ", "左手捩", "左手首", "右腕", "右ひじ")
    world = armature.matrix_world
    bones = {}
    for name in names:
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            continue
        matrix = world @ pose_bone.matrix
        constraints = []
        for constraint in pose_bone.constraints:
            constraints.append(
                {
                    "name": constraint.name,
                    "type": constraint.type,
                    "mute": bool(constraint.mute),
                }
            )
        bones[name] = {
            "head": list(matrix.translation),
            "y": list(matrix.to_3x3().col[1]),
            "quat": list(pose_bone.rotation_quaternion),
            "constraints": constraints,
        }
    return {
        "location": list(armature.location),
        "bones": bones,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0, help="0 means the full clip")
    parser.add_argument("--mode", choices=("quality", "direct"), default="direct")
    parser.add_argument("--still", action="store_true", help="Render one diagnostic frame instead of the clip")
    parser.add_argument("--identity", action="store_true", help="Zero body pose to inspect T-pose transfer")
    parser.add_argument("--camera", default="three_quarter")
    parser.add_argument("--clip-frame", type=int, default=0, help="Source frame index for --still")
    parser.add_argument(
        "--save-blend",
        type=Path,
        default=None,
        help="Also save a .blend with the retargeted action, textures packed",
    )
    parser.add_argument("--no-render", action="store_true", help="Only build the scene and save the .blend")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np

    from motionviewer.blender.camera import add_camera_for_bounds
    from motionviewer.blender.render import _normalize_engine
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import add_lighting, clear_scene, setup_world

    toon_dir = ROOT / ".local/blender_mmd_tools/mmd_tools/externals/MikuMikuDance"
    for name in ("toon01.bmp", "toon05.bmp"):
        source = toon_dir / name
        target = ROOT / name
        if source.is_file() and not target.exists():
            target.symlink_to(source)

    clear_scene()
    setup_world(transparent=True)
    motion_overrides = None
    if args.identity:
        from motionviewer.core.smplx_fk import recover_rest_offsets, rest_joints_from_offsets

        with np.load(args.motion, allow_pickle=False) as payload:
            joints = np.asarray(payload["joints22"], dtype=np.float64)
            global_orient = np.asarray(payload["global_orient"], dtype=np.float64)
            body_pose = np.asarray(payload["body_pose"], dtype=np.float64)
            offsets = recover_rest_offsets(joints, global_orient, body_pose)
            rest = rest_joints_from_offsets(offsets, root=np.mean(joints[:, 0], axis=0))
            rest_seq = np.repeat(rest[None, ...], len(global_orient), axis=0)
            motion_overrides = {
                "body_pose": np.zeros_like(payload["body_pose"]),
                "global_orient": np.zeros_like(payload["global_orient"]),
                "transl": np.zeros_like(payload["transl"]),
                "joints22": rest_seq,
            }
    actor = create_fbx_actor_from_npz(
        args.motion,
        label="yoimiya",
        fbx_path=args.asset,
        bone_map="mmd",
        gender="female",
        fbx_scale=0.08,
        retarget_mode=args.mode,
        motion_overrides=motion_overrides,
    )
    scene = bpy.context.scene
    frame_start = int(scene.frame_start)
    with np.load(args.motion, allow_pickle=False) as payload:
        total = int(len(payload["global_orient"]))
    still_frame = frame_start + max(args.clip_frame, 0)
    frame_end = frame_start + (total - 1 if args.frames <= 0 else min(args.frames, total) - 1)
    if args.still:
        frame_end = still_frame
    scene.frame_set(still_frame if args.still else frame_start)
    bpy.context.view_layer.update()

    def mesh_bounds() -> tuple[np.ndarray, np.ndarray]:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mins = np.array([1e9, 1e9, 1e9], dtype=np.float64)
        maxs = np.array([-1e9, -1e9, -1e9], dtype=np.float64)
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

    if args.still:
        mins, maxs = mesh_bounds()
    else:
        mins = np.array([1e9, 1e9, 1e9], dtype=np.float64)
        maxs = np.array([-1e9, -1e9, -1e9], dtype=np.float64)
        for frame in sorted({frame_start, (frame_start + frame_end) // 2, frame_end}):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            frame_mins, frame_maxs = mesh_bounds()
            mins = np.minimum(mins, frame_mins)
            maxs = np.maximum(maxs, frame_maxs)
    add_camera_for_bounds(mins.tolist(), maxs.tolist(), preset=args.camera, margin=1.35, resolution=(1280, 720))
    add_lighting(mins.tolist(), maxs.tolist())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    if args.save_blend is not None:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        # The still path clamps frame_end to one frame; a delivered .blend must
        # still open with the whole action on its timeline.
        scene.frame_start = frame_start
        scene.frame_end = frame_start + total - 1
        scene.frame_set(frame_start)
        # PMX textures are loaded from the asset directory; pack them so the
        # .blend opens with materials intact on any machine.
        try:
            bpy.ops.file.pack_all()
        except RuntimeError as exc:  # A missing optional texture must not stop delivery.
            print(f"pack_all skipped: {exc}")
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend.resolve()))
        print(f"saved {args.save_blend}")

    if args.no_render:
        return
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    if args.still:
        scene.frame_set(still_frame)
        scene.render.filepath = str(output / "still.png")
        bpy.ops.render.render(write_still=True)
    else:
        scene.render.filepath = str(output / "frame_")
        bpy.ops.render.render(animation=True)
    (output / "retarget_status.json").write_text(
        json.dumps(
            {
                "asset": str(args.asset),
                "motion": str(args.motion),
                "armature": actor.armature.name,
                "meshes": [mesh.name for mesh in actor.mesh_objects],
                "frames": [frame_start, frame_end],
                "identity": bool(args.identity),
                "armature_location": list(actor.armature.location),
                "debug": _bone_debug(actor.armature),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
