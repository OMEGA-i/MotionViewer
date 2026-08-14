"""Per-frame source-vs-character angle traces. Local helper.

Built for two reported symptoms that a still frame cannot settle: a sudden
reverse turn, and a head that jitters. Both need the trace, not the pose.

  blender --background --python scripts/_diagnose_motion.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx --motion <clip>/smplx_params.npz \
    --output outputs/motion_diagnosis.json
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

# Target bone -> source joint, for the traces we care about.
_TRACES: tuple[tuple[str, str], ...] = (
    ("腰", "pelvis"),
    ("上半身2", "spine3"),
    ("首", "neck"),
    ("頭", "head"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--faithful", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np

    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import clear_scene
    from motionviewer.core.smplx_fk import (
        SMPLX_BODY22_NAMES,
        blender_rotation,
        global_rotations,
    )

    clear_scene()
    actor = create_fbx_actor_from_npz(
        args.motion,
        label="diagnose",
        fbx_path=args.asset,
        bone_map="mmd",
        gender="female",
        fbx_scale=0.08,
        retarget_mode="direct",
        mmd_polish={"enabled": not args.faithful},
    )
    armature = actor.armature

    with np.load(args.motion, allow_pickle=False) as payload:
        global_orient = np.asarray(payload["global_orient"], dtype=np.float64)
        body_pose = np.asarray(payload["body_pose"], dtype=np.float64)
    rotations = global_rotations(global_orient, body_pose)
    total = len(global_orient)
    frame_start = int(bpy.context.scene.frame_start)

    def yaw_of(matrix: np.ndarray, axis: np.ndarray) -> float:
        """Heading of a body axis projected on the floor, in degrees."""
        vector = matrix @ axis
        return float(np.degrees(np.arctan2(float(vector[1]), float(vector[0]))))

    traces: dict[str, dict[str, list[float]]] = {
        bone: {"source_yaw": [], "character_yaw": [], "relative_deg": []} for bone, _ in _TRACES
    }
    # SMPL-X rest faces -Y in Blender space; the body's own X axis is the
    # sideways axis, which is the stable one to read a heading from.
    side = np.array((1.0, 0.0, 0.0))

    for offset in range(total):
        bpy.context.scene.frame_set(frame_start + offset)
        bpy.context.view_layer.update()
        evaluated = armature.evaluated_get(bpy.context.evaluated_depsgraph_get())
        world = evaluated.matrix_world
        for bone, source in _TRACES:
            pose_bone = evaluated.pose.bones.get(bone)
            if pose_bone is None:
                continue
            character = np.asarray(
                (world @ pose_bone.matrix).to_quaternion().to_matrix(), dtype=np.float64
            )
            source_rotation = blender_rotation(rotations[offset, SMPLX_BODY22_NAMES.index(source)])
            traces[bone]["character_yaw"].append(yaw_of(character, side))
            traces[bone]["source_yaw"].append(yaw_of(source_rotation, side))

    def unwrapped_steps(values: list[float]) -> dict[str, float]:
        array = np.unwrap(np.radians(np.asarray(values, dtype=np.float64)))
        steps = np.abs(np.diff(array))
        return {
            "max_step_deg": float(np.degrees(steps.max())) if len(steps) else 0.0,
            "mean_step_deg": float(np.degrees(steps.mean())) if len(steps) else 0.0,
            "total_turn_deg": float(np.degrees(array[-1] - array[0])) if len(array) else 0.0,
        }

    report: dict = {"motion": str(args.motion), "frames": total, "bones": {}}
    for bone, _ in _TRACES:
        trace = traces[bone]
        if not trace["character_yaw"]:
            continue
        source = unwrapped_steps(trace["source_yaw"])
        character = unwrapped_steps(trace["character_yaw"])
        # Difference of the unwrapped headings: a constant offset is the rig's
        # own rest orientation, a drift or a jump is a defect.
        source_unwrapped = np.unwrap(np.radians(trace["source_yaw"]))
        character_unwrapped = np.unwrap(np.radians(trace["character_yaw"]))
        difference = np.degrees(character_unwrapped - source_unwrapped)
        report["bones"][bone] = {
            "source": source,
            "character": character,
            "heading_offset_deg": {
                "first": float(difference[0]),
                "last": float(difference[-1]),
                "drift": float(difference[-1] - difference[0]),
                "max_deviation_from_first": float(np.abs(difference - difference[0]).max()),
            },
            "worst_step_frame": int(frame_start + int(np.argmax(np.abs(np.diff(character_unwrapped))))),
        }
    report["traces"] = {bone: traces[bone] for bone, _ in _TRACES if traces[bone]["character_yaw"]}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for bone, data in report["bones"].items():
        print(
            f"{bone}: source turn {data['source']['total_turn_deg']:+.1f} deg, "
            f"character {data['character']['total_turn_deg']:+.1f} deg | "
            f"max step src {data['source']['max_step_deg']:.1f} vs chr "
            f"{data['character']['max_step_deg']:.1f} (frame {data['worst_step_frame']}) | "
            f"offset drift {data['heading_offset_deg']['drift']:+.2f}, "
            f"max dev {data['heading_offset_deg']['max_deviation_from_first']:.2f}"
        )


if __name__ == "__main__":
    main()
