"""Render a complete native character review and persist reproducible audits.

Pass the ordinary render_mmd_compare arguments after Blender's ``--``. This local
review wrapper never changes the animation; it adds a saved scene and all-frame
mapping/foot measurements to the renderer's output directory.
"""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import bpy
    import numpy as np

    argv = sys.argv[sys.argv.index("--") + 1:]
    destination = Path(argv[argv.index("--output") + 1]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "render_arguments.json").write_text(json.dumps(argv, ensure_ascii=False, indent=2))
    # Fit one constant screen-space box over every frame. This adjusts only the
    # presentation camera; exact comparison cameras bypass the adjustment.
    from motionviewer.blender import review_camera

    camera_reports = {}
    original_capture = review_camera.capture_camera_track

    def capture_fitted_camera(bpy, camera, *, frame_start, frames):
        if "--match-camera" not in argv:
            armatures = {obj for obj in bpy.data.objects if obj.type == "ARMATURE" and
                         "motionviewer_soma_map" in obj}
            meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and
                      str(getattr(obj, "mmd_type", "NONE")) == "NONE" and
                      (any(mod.type == "ARMATURE" and mod.object in armatures for mod in obj.modifiers)
                       or obj.name.startswith("weapon_"))]
            camera_reports[camera.name] = review_camera.fit_orthographic_geometry(
                bpy, camera, meshes, frame_start=frame_start, frames=frames, padding=1.12,
            )
        return original_capture(bpy, camera, frame_start=frame_start, frames=frames)

    review_camera.capture_camera_track = capture_fitted_camera
    runpy.run_path(str(ROOT / "scripts/render_mmd_compare.py"), run_name="__main__")
    review_camera.capture_camera_track = original_capture
    info = json.loads((destination / "info.json").read_text())
    info["review_camera_geometry_fit"] = camera_reports
    (destination / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    if not info["motion_fidelity"]["mapping_only"]:
        raise ValueError("This review requires unmodified native body and finger channels")
    if info["physics_requested"] == "bullet" and not info["physics"].get("baked"):
        raise ValueError("Requested physics was not actually baked")
    blend = destination / "review.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))

    spec = importlib.util.spec_from_file_location("soma_review_audit", ROOT / "scripts/_audit_soma_display.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit = module.audit(bpy, Path(info["motion"]), Path(info["asset"]), blend)
    (destination / "source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise ValueError("All-frame source mapping audit failed; review source_audit.json")

    from motionviewer.blender.mmd_contact import _support_metrics, measure_feet, source_feet
    from motionviewer.core.foot_contact import FootContactSettings, plan_foot_contacts

    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE" and "motionviewer_soma_map" in obj)
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and
              any(mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers) and
              str(getattr(obj, "mmd_type", "NONE")) == "NONE"]
    actor = SimpleNamespace(armature=armature, mesh_objects=meshes)
    feet = measure_feet(bpy, actor, frame_start=1, frames=info["frames"])
    gap = feet.sole_z - info["floor_z"]
    source_ankles, source_toes, fps = source_feet(info["motion"], info["frames"])
    support = plan_foot_contacts(source_ankles, source_toes, feet.ankles, feet.toes, fps=fps,
                                 settings=FootContactSettings(max_correction_m=0.0))
    sole_report = {
        "frames": info["frames"], "floor_z_m": info["floor_z"],
        "shoe_vertex_counts": list(feet.vertex_counts),
        "min_sole_to_floor_mm": float(gap.min() * 1000),
        "frames_any_sole_below_floor_2mm": (np.flatnonzero((gap < -.002).any(axis=1)) + 1).tolist(),
        "per_frame_sole_to_floor_mm": (gap * 1000).tolist(),
        "world_sole_min_m": float(feet.sole_z.min()),
        "per_frame_lower_shoe_percentiles_m": {
            str(p): float(np.percentile(feet.sole_z.min(axis=1), p)) for p in (0, 5, 25, 50, 95, 100)
        },
        "source_inferred_support_measurement_only": {
            "classification": support.summary(),
            "measured_target": _support_metrics(feet, support.weights, support.support_modes, info["floor_z"], fps),
            "pose_correction_applied": False,
        },
        "limits": "Sole heights are measured on the selected shoe skin. The separate source-support classification is heuristic, not ground truth or a promise of no foot sliding. No foot IK is applied.",
    }
    (destination / "sole_audit.json").write_text(json.dumps(sole_report, ensure_ascii=False, indent=2))
    images = [{"name": image.name, "path": image.filepath} for image in bpy.data.images
              if image.source == "FILE" and image.filepath and not Path(bpy.path.abspath(image.filepath)).is_file()]
    (destination / "asset_audit.json").write_text(json.dumps({
        "bones": len(armature.data.bones), "visible_character_meshes": len(meshes),
        "vertices": sum(len(obj.data.vertices) for obj in meshes), "missing_image_files": images,
        "source_fingers": audit["driven_fingers"], "mapped_joints": audit["driven_joints"],
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"review": str(destination), "source_audit_passed": audit["passed"],
                      "min_sole_mm": sole_report["min_sole_to_floor_mm"], "missing_images": len(images)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
