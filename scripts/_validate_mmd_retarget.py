"""Check the MMD retarget against Blender's own evaluation. Local helper.

Four things must agree independently, and a silent disagreement is exactly how a
visually plausible but wrong action escapes:

1. the NumPy solver's world rotations vs Blender's evaluated pose bones, which
   covers the channel write, the real parent chain and every live constraint;
2. the solver vs closed-form theory, which must be exact;
3. every ``absolute`` bone's world Y vs the source aim, the property that keeps
   arms out of the torso;
4. arm drop vs the source, because a character with flared kimono sleeves cannot
   be judged from a silhouette.

Pass ``--motion`` more than once to check several clips in one Blender start.

  blender --background --python scripts/_validate_mmd_retarget.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
    --motion <clip>/smplx_params.npz --output outputs/mmd_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_ARM_SAMPLE_OFFSETS = (0, 13, 27, 39)


def _check(bpy: Any, np: Any, motion_path: Path, asset: Path, frames: int, polish: bool) -> dict:
    from motionviewer.blender.retarget._mmd_plan import build_mmd_plan
    from motionviewer.blender.retarget._resolve import resolve_bone_mapping
    from motionviewer.blender.retarget.mmd import MmdPolishOptions
    from motionviewer.blender.retarget.mmd_solve import polish_source_frames, solve_mmd_retarget
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import clear_scene
    from motionviewer.core.smplx_fk import blender_rotation, build_lookat_motion, global_rotations

    clear_scene()
    actor = create_fbx_actor_from_npz(
        motion_path,
        label="validate",
        fbx_path=asset,
        bone_map="mmd",
        gender="female",
        fbx_scale=0.08,
        retarget_mode="direct",
        mmd_polish={"enabled": polish},
    )
    armature = actor.armature

    with np.load(motion_path, allow_pickle=False) as payload:
        joints = np.asarray(payload["joints22"], dtype=np.float64)
        global_orient = np.asarray(payload["global_orient"], dtype=np.float64)
        body_pose = np.asarray(payload["body_pose"], dtype=np.float64)
        transl = np.asarray(payload["transl"], dtype=np.float64)
    motion = build_lookat_motion(joints, global_orient, body_pose, transl)
    rotations = global_rotations(global_orient, body_pose)

    mapping = resolve_bone_mapping("mmd", fbx_armature=armature)
    plan = build_mmd_plan(
        armature,
        smplx_map=mapping.smplx_to_fbx,
        transfer_modes=mapping.transfer_modes,
        twist_pairs=mapping.twist_pairs,
        source_rest=motion.rest_by_name(),
        source_names=motion.names,
    )
    total = min(frames, len(global_orient)) if frames > 0 else len(global_orient)
    # Feed the solver exactly what the pipeline fed its own, or the comparison
    # measures the polish pass and reports it as transfer error.
    transfer = json.loads(armature.get("motionviewer_mmd_transfer", "{}"))
    source_frames = polish_source_frames(
        motion.posed_frames,
        motion.names,
        motion.rest_frames,
        MmdPolishOptions.from_mapping({"enabled": polish, **transfer.get("polish", {})}),
    )[:total]
    result = solve_mmd_retarget(plan, source_frames, np.zeros((total, 3)))

    source_index = {name: index for index, name in enumerate(motion.names)}
    frame_start = int(bpy.context.scene.frame_start)
    worst: dict[str, dict[str, Any]] = {}

    def track(key: str, name: str, value: float, frame: int) -> None:
        entry = worst.get(key)
        if entry is None or value > float(entry["value"]):
            worst[key] = {"value": value, "bone": name, "frame": frame}

    def geodesic(left: Any, right: Any) -> float:
        cosine = (float(np.trace(left.T @ right)) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    per_bone: dict[str, dict[str, Any]] = {}
    for offset in range(total):
        frame = frame_start + offset
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        # Animation and constraints are evaluated on the depsgraph copy.
        evaluated_armature = armature.evaluated_get(bpy.context.evaluated_depsgraph_get())
        world = evaluated_armature.matrix_world
        for index, channel in enumerate(plan.channels):
            pose_bone = evaluated_armature.pose.bones.get(channel.name)
            if pose_bone is None:
                continue
            evaluated = np.asarray(
                (world @ pose_bone.matrix).to_quaternion().to_matrix(), dtype=np.float64
            )
            solved = result.world_rotations[offset, index]
            record = per_bone.setdefault(channel.name, {"mode": channel.mode})
            gap = geodesic(evaluated, solved)
            track("blender_vs_solver_deg", channel.name, gap, frame)
            record["blender_vs_solver_deg"] = max(record.get("blender_vs_solver_deg", 0.0), gap)

            if not channel.driven:
                continue
            if channel.mode == "absolute":
                source_y = source_frames[offset, source_index[channel.source], :3, 1]
                aim = float(
                    np.degrees(np.arccos(np.clip(float(np.dot(evaluated[:, 1], source_y)), -1.0, 1.0)))
                )
                track("absolute_aim_deg", channel.name, aim, frame)
                record["aim_deg"] = max(record.get("aim_deg", 0.0), aim)
            else:
                # Expected is derived from what the solver was actually given,
                # so the polish pass is not reported as transfer error. How far
                # the polish itself departs from the raw source is tracked
                # separately below.
                joint = source_index[channel.source]
                carried = (
                    source_frames[offset, joint, :3, :3] @ motion.rest_frames[joint, :3, :3].T
                )
                expected = carried @ plan.target_rest_global[index]
                gap = geodesic(evaluated, expected)
                track("relative_transfer_deg", channel.name, gap, frame)
                record["transfer_deg"] = max(record.get("transfer_deg", 0.0), gap)
                solver_gap = geodesic(solved, expected)
                track("solver_vs_theory_deg", channel.name, solver_gap, frame)
                record["solver_vs_theory_deg"] = max(record.get("solver_vs_theory_deg", 0.0), solver_gap)

            raw = blender_rotation(rotations[offset, source_index[channel.source]])
            joint = source_index[channel.source]
            polished = source_frames[offset, joint, :3, :3] @ motion.rest_frames[joint, :3, :3].T
            deviation = geodesic(raw, polished)
            track("polish_deviation_deg", channel.name, deviation, frame)
            record["polish_deviation_deg"] = max(record.get("polish_deviation_deg", 0.0), deviation)

    arm_angles: list[dict[str, Any]] = []
    for offset in _ARM_SAMPLE_OFFSETS:
        if offset >= total:
            continue
        bpy.context.scene.frame_set(frame_start + offset)
        bpy.context.view_layer.update()
        world = armature.matrix_world
        for side, upper, fore, source in (
            ("L", "左腕", "左ひじ", "left_shoulder"),
            ("R", "右腕", "右ひじ", "right_shoulder"),
        ):
            upper_bone = armature.pose.bones.get(upper)
            fore_bone = armature.pose.bones.get(fore)
            if upper_bone is None or fore_bone is None:
                continue
            upper_axis = np.asarray((world @ upper_bone.matrix).to_3x3().col[1], dtype=np.float64)
            fore_axis = np.asarray((world @ fore_bone.matrix).to_3x3().col[1], dtype=np.float64)
            upper_axis /= np.linalg.norm(upper_axis)
            fore_axis /= np.linalg.norm(fore_axis)
            source_upper = source_frames[offset, source_index[source], :3, 1]
            arm_angles.append(
                {
                    "frame": frame_start + offset,
                    "side": side,
                    "upper_drop_deg": float(np.degrees(np.arcsin(-upper_axis[2]))),
                    "source_upper_drop_deg": float(np.degrees(np.arcsin(-source_upper[2]))),
                    "elbow_bend_deg": float(
                        np.degrees(np.arccos(np.clip(float(np.dot(upper_axis, fore_axis)), -1.0, 1.0)))
                    ),
                }
            )

    return {
        "motion": str(motion_path),
        "frames_checked": total,
        "channels": len(plan.channels),
        "driven": sum(1 for channel in plan.channels if channel.driven),
        "root_translation_scale": plan.root_translation_scale,
        "polish": transfer.get("polish", {}),
        "worst": worst,
        "arm_angles": arm_angles,
        "per_bone": per_bone,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12, help="0 means the whole clip")
    parser.add_argument("--faithful", action="store_true", help="Disable the polish pass")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np

    reports = [
        _check(bpy, np, motion, args.asset, args.frames, not args.faithful)
        for motion in args.motion
    ]
    payload: dict[str, Any] = reports[0] if len(reports) == 1 else {"clips": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports:
        summary = {key: round(float(value["value"]), 5) for key, value in report["worst"].items()}
        print(json.dumps({"motion": Path(report["motion"]).parent.name, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
