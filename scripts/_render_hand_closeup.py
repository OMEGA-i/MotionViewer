"""Close-up of one hand, aimed at the wrist bone. Local helper.

Finger flexion is derived from the rig, so the curl direction has to be checked
visually once: a hand that flexes backwards is anatomically inside out.
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
    parser.add_argument("--bone", default="左手首")
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--distance", type=float, default=0.30)
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--faithful", action="store_true")
    parser.add_argument("--identity", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np
    from mathutils import Vector  # type: ignore

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
    setup_world(transparent=False)

    overrides = None
    if args.identity:
        from motionviewer.core.smplx_fk import recover_rest_offsets, rest_joints_from_offsets

        with np.load(args.motion, allow_pickle=False) as payload:
            joints = np.asarray(payload["joints22"], dtype=np.float64)
            global_orient = np.asarray(payload["global_orient"], dtype=np.float64)
            body_pose = np.asarray(payload["body_pose"], dtype=np.float64)
            offsets = recover_rest_offsets(joints, global_orient, body_pose)
            rest = rest_joints_from_offsets(offsets, root=np.mean(joints[:, 0], axis=0))
            overrides = {
                "body_pose": np.zeros_like(payload["body_pose"]),
                "global_orient": np.zeros_like(payload["global_orient"]),
                "transl": np.zeros_like(payload["transl"]),
                "joints22": np.repeat(rest[None, ...], len(global_orient), axis=0),
            }

    actor = create_fbx_actor_from_npz(
        args.motion,
        label="yoimiya",
        fbx_path=args.asset,
        bone_map="mmd",
        gender="female",
        fbx_scale=0.08,
        retarget_mode="direct",
        motion_overrides=overrides,
        mmd_polish={"enabled": not args.faithful},
    )
    scene = bpy.context.scene
    scene.frame_set(args.frame)
    bpy.context.view_layer.update()

    armature = actor.armature
    pose_bone = armature.pose.bones.get(args.bone)
    if pose_bone is None:
        raise SystemExit(f"no bone {args.bone!r}")
    matrix = armature.matrix_world @ pose_bone.matrix
    # Aim a little past the wrist so the fingers sit in frame, not the forearm.
    target = matrix.translation + matrix.to_3x3().col[1].normalized() * 0.05

    camera_data = bpy.data.cameras.new("HandCam")
    camera_data.lens = 55.0
    camera = bpy.data.objects.new("HandCam", camera_data)
    scene.collection.objects.link(camera)
    # Look at the palm from the front-outside, above the hand.
    direction = Vector((0.45, -0.8, 0.4)).normalized()
    camera.location = target + direction * args.distance
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (target - camera.location).to_track_quat("-Z", "Y")
    scene.camera = camera

    add_lighting([float(v) - 0.3 for v in target], [float(v) + 0.3 for v in target])
    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output)
    bpy.ops.render.render(write_still=True)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
