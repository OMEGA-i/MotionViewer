"""Retarget one SMPL-X clip and export the animated Mixamo FBX.

Example:
  blender --background --python scripts/export_retarget_fbx.py -- \
    --motion data/examples/.../omegamotiongpt.smplx.npz \
    --asset assets/fbx/iron.fbx --output /tmp/iron_retargeted.fbx
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
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("quality", "direct"), default="quality")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore

    from motionviewer.blender.retarget.export import export_fbx_animation, validate_fbx_roundtrip
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import clear_scene

    clear_scene()
    actor = create_fbx_actor_from_npz(
        args.motion,
        label="retarget_export",
        fbx_path=args.asset,
        retarget_mode=args.mode,
    )
    frame_start = bpy.context.scene.frame_start
    with __import__("numpy").load(args.motion, allow_pickle=False) as payload:
        frame_end = frame_start + int(len(payload["global_orient"])) - 1
    output = export_fbx_animation(
        bpy,
        actor.armature,
        actor.mesh_objects,
        args.output,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    bone_names = tuple(pose_bone.name for pose_bone in actor.armature.pose.bones)

    def find_bone(suffix: str) -> str | None:
        exact = actor.armature.pose.bones.get(suffix)
        if exact is not None:
            return exact.name
        matches = [name for name in bone_names if name.endswith(suffix)]
        return matches[0] if len(matches) == 1 else None

    root_bone = find_bone("Hips")
    position_bones = tuple(
        name
        for suffix in ("Hips", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")
        for name in (find_bone(suffix),)
        if name is not None
    )
    foot_bones = tuple(
        name
        for suffix in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")
        for name in (find_bone(suffix),)
        if name is not None
    )
    render = bpy.context.scene.render
    result = validate_fbx_roundtrip(
        bpy,
        actor.armature,
        output,
        bone_names=bone_names,
        frame_start=frame_start,
        frame_end=frame_end,
        root_bone_name=root_bone,
        position_bone_names=position_bones,
        foot_bone_names=foot_bones,
        expected_fps=float(render.fps),
        expected_fps_base=float(getattr(render, "fps_base", 1.0)),
        # Blender's FBX bake quantizes translations by a few millimetres on
        # Mixamo-scale rigs; this remains tighter than the 5 mm sole-contact
        # acceptance gate while avoiding false negatives from the file codec.
        position_tolerance_m=5e-3,
    )
    if not result["passed"]:
        raise RuntimeError(f"FBX round-trip validation failed: {result}")
    print({"output": str(output), "roundtrip": result})


if __name__ == "__main__":
    main()
