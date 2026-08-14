"""Dump the imported MMD armature and its SMPL-X rest alignment. Local helper.

  blender --background --python scripts/_dump_mmd_rig.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
    --motion data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz \
    --output outputs/rig_dump.json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np

    from motionviewer.blender.mmd_import import import_pmx_character
    from motionviewer.blender.retarget.mmd import inspect_mmd_rig
    from motionviewer.blender.scene import clear_scene
    from motionviewer.core.smplx_fk import build_lookat_motion

    clear_scene()
    armature, meshes = import_pmx_character(bpy, args.asset, label="dump", scale=0.08)
    bpy.context.view_layer.update()

    world = armature.matrix_world
    bones: dict[str, dict] = {}
    for bone in armature.data.bones:
        matrix = world @ bone.matrix_local
        rotation = matrix.to_3x3()
        pose_bone = armature.pose.bones.get(bone.name)
        mmd_bone = getattr(pose_bone, "mmd_bone", None) if pose_bone else None
        entry = {
            "parent": bone.parent.name if bone.parent else None,
            "children": [child.name for child in bone.children],
            "length": float(bone.length) * float(sum(world.to_scale()) / 3.0),
            "head": [float(value) for value in matrix.translation],
            "x": [float(value) for value in rotation.col[0]],
            "y": [float(value) for value in rotation.col[1]],
            "z": [float(value) for value in rotation.col[2]],
            "use_deform": bool(bone.use_deform),
            "constraints": [
                {
                    "name": constraint.name,
                    "type": str(constraint.type),
                    "mute": bool(constraint.mute),
                    "influence": float(getattr(constraint, "influence", 1.0)),
                }
                for constraint in (pose_bone.constraints if pose_bone else [])
            ],
        }
        if mmd_bone is not None:
            entry["mmd"] = {
                "additional_transform_bone": str(getattr(mmd_bone, "additional_transform_bone", "") or ""),
                "additional_transform_influence": float(
                    getattr(mmd_bone, "additional_transform_influence", 0.0) or 0.0
                ),
                "has_additional_rotation": bool(getattr(mmd_bone, "has_additional_rotation", False)),
                "has_additional_location": bool(getattr(mmd_bone, "has_additional_location", False)),
                "is_tip": bool(getattr(mmd_bone, "is_tip", False)),
                "enabled_fixed_axis": bool(getattr(mmd_bone, "enabled_fixed_axis", False)),
                "fixed_axis": [float(value) for value in getattr(mmd_bone, "fixed_axis", (0.0, 0.0, 0.0))],
                "enabled_local_axes": bool(getattr(mmd_bone, "enabled_local_axes", False)),
                "local_axis_x": [float(v) for v in getattr(mmd_bone, "local_axis_x", (0.0, 0.0, 0.0))],
                "local_axis_z": [float(v) for v in getattr(mmd_bone, "local_axis_z", (0.0, 0.0, 0.0))],
                "transform_order": int(getattr(mmd_bone, "transform_order", 0) or 0),
                "transform_after_dynamics": bool(getattr(mmd_bone, "transform_after_dynamics", False)),
            }
        bones[bone.name] = entry

    inspection = inspect_mmd_rig(armature)

    # Vertex-group influence tells us which bones actually deform the mesh.
    group_weight: dict[str, float] = {}
    for mesh in meshes:
        names = [group.name for group in mesh.vertex_groups]
        for vertex in mesh.data.vertices:
            for element in vertex.groups:
                if element.group < len(names):
                    name = names[element.group]
                    group_weight[name] = group_weight.get(name, 0.0) + float(element.weight)

    with np.load(args.motion, allow_pickle=False) as payload:
        motion = build_lookat_motion(
            np.asarray(payload["joints22"], dtype=np.float64),
            np.asarray(payload["global_orient"], dtype=np.float64),
            np.asarray(payload["body_pose"], dtype=np.float64),
            np.asarray(payload["transl"], dtype=np.float64),
        )
    source_rest = motion.rest_by_name()

    alignment = {}
    for source_name, target_name in inspection.smplx_map.items():
        target = bones.get(target_name)
        if target is None:
            continue
        source_y = np.asarray(source_rest[source_name], dtype=np.float64)[:3, 1]
        target_y = np.asarray(target["y"], dtype=np.float64)
        cosine = float(np.clip(np.dot(source_y, target_y), -1.0, 1.0))
        alignment[source_name] = {
            "target": target_name,
            "source_rest_y": [float(value) for value in source_y],
            "target_rest_y": [float(value) for value in target_y],
            "rest_angle_deg": float(np.degrees(np.arccos(cosine))),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "armature": armature.name,
                "matrix_world_scale": [float(value) for value in world.to_scale()],
                "bone_count": len(bones),
                "bones": bones,
                "smplx_map": inspection.smplx_map,
                "canonical_map": inspection.canonical_map,
                "twist_pairs": [
                    {"swing": pair.swing_bone, "twist": pair.twist_bone, "axis": list(pair.axis_local)}
                    for pair in inspection.twist_pairs
                ],
                "errors": list(inspection.errors),
                "vertex_group_weight": group_weight,
                "rest_alignment": alignment,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
