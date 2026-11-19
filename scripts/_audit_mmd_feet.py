"""Audit evaluated shoe support on real MMD meshes without rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correct", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    import bpy
    import numpy as np

    from motionviewer.blender.mmd_contact import apply_foot_contact, measure_feet, source_feet
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import clear_scene

    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.motion:
        clear_scene()
        with np.load(path, allow_pickle=False) as payload:
            is_soma = "joints77" in payload
        if is_soma:
            from motionviewer.blender.retarget.soma_pipeline import create_soma_actor_from_npz

            actor = create_soma_actor_from_npz(path, label="foot_audit", pmx_path=args.asset,
                                              pmx_scale=0.08)
        else:
            actor = create_fbx_actor_from_npz(path, label="foot_audit", fbx_path=args.asset,
                                              retarget_mode="direct", fbx_scale=0.08)
        source_ankles, source_toes, fps = source_feet(path)
        sample = measure_feet(bpy, actor, frame_start=1, frames=len(source_ankles))
        data = {key: value for key, value in vars(sample).items()}
        data.update(source_ankles=source_ankles, source_toes=source_toes, fps=fps)
        np.savez_compressed(args.output / f"{path.stem}.feet.npz", **data)
        transfer = json.loads(actor.armature.get("motionviewer_mmd_transfer", "{}"))
        report = {"motion": str(path), "frames": len(source_ankles), "shoe_vertices": sample.vertex_counts,
                  "sole_z_cm_percentiles": np.percentile(sample.sole_z, [0, 10, 25, 50, 90, 100], axis=0).tolist(),
                  "root_translation_scale": transfer.get("root_translation_scale")}
        report["sole_z_cm_percentiles"] = (np.asarray(report["sole_z_cm_percentiles"]) * 100).round(3).tolist()
        (args.output / f"{path.stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if args.correct:
            correction = apply_foot_contact(bpy, actor, path, frame_start=1, frames=len(source_ankles))
            (args.output / f"{path.stem}.contact.json").write_text(json.dumps(correction, ensure_ascii=False, indent=2))
            after = measure_feet(bpy, actor, frame_start=1, frames=len(source_ankles))
            np.savez_compressed(args.output / f"{path.stem}.after.npz", **vars(after))
            print("CORRECTION " + json.dumps(correction, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
