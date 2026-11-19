"""Audit native SOMA mapped rotations and root deltas in Blender, all frames.

Without --blend, import and animate each --motion using the default faithful path.
With --blend, audit one already baked scene, including effects of prop/cloth work.
This validates the mapping's implementation, not anatomical contact equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def rotation_error(a, b):
    # Project Blender float32 rotations onto SO(3) before measuring small angles.
    u, _, vt = np.linalg.svd(a)
    a = u @ vt
    u, _, vt = np.linalg.svd(b)
    b = u @ vt
    delta = a.T @ b
    sine = np.linalg.norm([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0],
                           delta[1, 0] - delta[0, 1]]) / 2
    return float(np.degrees(np.arctan2(sine, (np.trace(delta) - 1) / 2)))


def audit(bpy, path, asset, blend):
    from motionviewer.blender.retarget._mmd_plan import build_mmd_plan
    from motionviewer.blender.retarget.mmd_solve import solve_mmd_retarget
    from motionviewer.blender.retarget.soma_mmd import build_soma_mmd_mapping
    from motionviewer.blender.retarget.soma_pipeline import create_soma_actor_from_npz, load_soma
    from motionviewer.blender.scene import clear_scene
    from motionviewer.core.soma_fk import SOMA77_PRIMARY_CHILD, build_soma_motion

    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    motion = build_soma_motion(load_soma(path))
    if blend:
        bpy.ops.wm.open_mainfile(filepath=str(blend.resolve()))
        arms = [a for a in bpy.data.objects if a.type == "ARMATURE" and "motionviewer_soma_map" in a]
        if len(arms) != 1:
            raise ValueError(f"Expected one SOMA actor, got {len(arms)}")
        armature = arms[0]
    else:
        clear_scene()
        actor = create_soma_actor_from_npz(path, label="audit", pmx_path=asset)
        armature = actor.armature
    mapping = build_soma_mmd_mapping(armature)
    stored = json.loads(armature["motionviewer_soma_map"])
    mapping_matches = stored == mapping.smplx_to_fbx
    if not mapping_matches:
        raise ValueError("Cached scene has an obsolete SOMA mapping; regenerate it")
    hand_aim = {name: SOMA77_PRIMARY_CHILD[name] for name in ("LeftHand", "RightHand")}
    if json.loads(armature.get("motionviewer_soma_hand_aim", "{}")) != hand_aim:
        raise ValueError("Cached scene has no current anatomical hand-axis landmarks; regenerate it")
    plan = build_mmd_plan(
        armature, smplx_map=mapping.smplx_to_fbx, transfer_modes=mapping.transfer_modes,
        twist_pairs=mapping.twist_pairs, source_rest=motion.rest_by_name(), source_names=motion.names,
    )
    solved = solve_mmd_retarget(plan, motion.posed_frames, np.zeros((len(motion.posed_frames), 3)))
    first = int(bpy.context.scene.frame_start)
    total = len(motion.posed_frames)
    roots = []
    worst = {"body_deg": 0.0, "fingers_deg": 0.0, "all_channels_deg": 0.0}
    locations = {}
    for offset in range(total):
        frame = first + offset
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        arm = armature.evaluated_get(bpy.context.evaluated_depsgraph_get())
        roots.append(np.asarray(arm.matrix_world.translation, dtype=float))
        for i, ch in enumerate(plan.channels):
            if not ch.driven:
                continue
            actual = np.asarray((arm.matrix_world @ arm.pose.bones[ch.name].matrix).to_quaternion().to_matrix())
            error = rotation_error(actual, solved.world_rotations[offset, i])
            group = "fingers_deg" if "Hand" in (ch.source or "") and ch.source not in ("LeftHand", "RightHand") else "body_deg"
            for key in (group, "all_channels_deg"):
                if error > worst[key]:
                    worst[key] = error
                    locations[key] = {"frame": frame, "bone": ch.name, "source": ch.source}
    roots = np.asarray(roots)
    root_error = np.linalg.norm((roots - roots[0]) - (motion.root_locations - motion.root_locations[0]), axis=1)
    after_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "motion": str(path), "source_sha256": before_hash, "source_unchanged": before_hash == after_hash,
        "blend": str(blend) if blend else None, "frames_checked": total,
        "driven_joints": len(mapping.smplx_to_fbx), "driven_fingers": mapping.driven_fingers,
        "mapping_current": mapping_matches, "worst_rotation_error_deg": worst, "worst_at": locations,
        "hand_axis_landmarks": hand_aim,
        "max_root_delta_error_mm": float(root_error.max() * 1000),
        "transfer": json.loads(armature["motionviewer_mmd_transfer"]),
        "limits": "Checks mapped orientations and root deltas, not identical world joint positions on different proportions or physical plausibility.",
    }
    report["passed"] = bool(before_hash == after_hash and worst["all_channels_deg"] < 0.01 and root_error.max() < 1e-5)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--asset", type=Path, default=ROOT / "assets/fbx/pmx/yoimiya/宵宫.pmx")
    parser.add_argument("--blend", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.blend and len(args.motion) != 1:
        parser.error("--blend audits exactly one motion")
    import bpy

    reports = [audit(bpy, p, args.asset, args.blend) for p in args.motion]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"clips": reports, "passed": all(r["passed"] for r in reports)}, ensure_ascii=False, indent=2))
    for report in reports:
        print(json.dumps({key: report[key] for key in ("motion", "passed", "frames_checked", "worst_rotation_error_deg", "max_root_delta_error_mm")}, ensure_ascii=False))
    if not all(r["passed"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
