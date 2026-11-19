"""Measure carry-box and weapon fit, with optional paired review stills.

Run with Blender, for example::

    blender --background --python scripts/_probe_prop_fit.py -- \
        --data "/path/to/t2m_demo400/smplx" --output outputs/prop_fit/probe.json

Bone/shaft distances are diagnostic proxies, not mesh collision acceptance.
Original motion and character files are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {name: round(float(value), 4) for name, value in zip(
        ("min", "p05", "median", "p95", "max"), np.percentile(array, (0, 5, 50, 95, 100)), strict=True
    )}


def render_carry_review(bpy, actor, legacy, fitted, frames: list[int], directory: Path) -> dict:
    from mathutils import Vector

    from motionviewer.blender.mmd_toon import add_toon_lighting, apply_toon_shading
    from motionviewer.blender.render import _normalize_engine
    from motionviewer.blender.scene import setup_world

    setup_world()
    apply_toon_shading(list(actor.mesh_objects) + [legacy, fitted])
    add_toon_lighting([-1, -1, 0], [1, 1, 2])
    scene = bpy.context.scene
    data = bpy.data.cameras.new("carry_review")
    data.type = "ORTHO"
    camera = bpy.data.objects.new("carry_review", data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    offset = Vector((2, -4, 1.4)).normalized() * 4
    camera.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()
    axes = camera.rotation_euler.to_matrix()
    right, up = axes.col[0], axes.col[1]
    targets = {}
    extent = 0.0
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        bones = actor.armature.pose.bones
        world = actor.armature.matrix_world
        wrists = (world @ bones["左手首"].head + world @ bones["右手首"].head) / 2
        head = world @ bones["頭"].head
        target = (wrists + head) / 2 + Vector((0, 0, 0.04))
        targets[frame] = target
        points = [world @ bones[name].head for name in ("左手首", "右手首", "左ひじ", "右ひじ", "頭")]
        points += [head + Vector((0, 0, 0.22))]
        for obj in (legacy, fitted):
            points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
        extent = max(extent, *(max(abs((point - target).dot(right)), abs((point - target).dot(up)))
                               for point in points))
    data.ortho_scale = max(0.8, extent * 2.2)
    directory.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        camera.location = targets[frame] + offset
        for name, visible, hidden in (("legacy", legacy, fitted), ("two_hand", fitted, legacy)):
            visible.hide_render = False
            hidden.hide_render = True
            destination = directory / f"frame_{frame:03d}_{name}.png"
            scene.render.filepath = str(destination)
            bpy.ops.render.render(write_still=True)
            images.append({"frame": frame, "mode": name, "path": str(destination)})
    legacy.hide_render = fitted.hide_render = False
    return {"images": images, "ortho_scale": data.ortho_scale,
            "camera": "Same paired camera, rotation and scale; target follows source wrist/head midpoint",
            "camera_targets": {str(frame): list(target) for frame, target in targets.items()}}


def carry_probe(bpy, data: Path, *, two_hand: bool = False, box_size=None, render_dir: Path | None = None) -> dict:
    from motionviewer.blender.mmd_props import PropStyle, attach_to_bone, build_carry_box_scene
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import clear_scene

    path = data / "active/016783.smplx.npz"
    with np.load(path, allow_pickle=False) as source:
        frames = len(source["transl"])
    clear_scene()
    actor = create_fbx_actor_from_npz(
        path, label="carry_probe", fbx_path=ROOT / "assets/fbx/pmx/yoimiya/宵宫.pmx",
        retarget_mode="direct",
    )
    armature = actor.armature
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    build = build_carry_box_scene(bpy, PropStyle(), hand_bone="左手首")
    prop = build.props[0]
    attach_to_bone(bpy, prop, armature)
    bpy.context.view_layer.update()

    def point(name):
        return np.asarray(armature.matrix_world @ armature.pose.bones[name].head)

    initial_midpoint = (point("左手首") + point("右手首")) / 2
    initial_offset = np.asarray(prop.obj.matrix_world.translation) - initial_midpoint
    rows = []
    for frame in range(1, frames + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        left, right, head = point("左手首"), point("右手首"), point("頭")
        midpoint = (left + right) / 2
        # Palm centre estimate is halfway from the wrist to the middle knuckle.
        palms = [(point(side + "手首") + point(side + "中指１")) / 2 for side in ("左", "右")]
        palm_midpoint = (palms[0] + palms[1]) / 2
        centre = np.asarray(prop.obj.matrix_world.translation)
        rows.append({
            "frame": frame,
            "wrist_gap_m": float(np.linalg.norm(right - left)),
            "palm_gap_m": float(np.linalg.norm(palms[1] - palms[0])),
            "wrist_midpoint_z_m": float(midpoint[2]),
            "head_base_z_m": float(head[2]),
            "height_budget_m": float(head[2] - palm_midpoint[2] - 0.04),
            "legacy_midpoint_drift_m": float(np.linalg.norm(centre - midpoint - initial_offset)),
            "wrist_midpoint": midpoint.tolist(),
            "palm_midpoint": palm_midpoint.tolist(),
            "head_base": head.tolist(),
        })
    report = {
        "motion": str(path), "asset": "yoimiya", "frames": frames,
        "height_budget_definition": "head bone base minus estimated palm centre height minus 40 mm; not camera occlusion",
        "legacy_midpoint_drift_definition": "distance from frame-1 wrist-midpoint offset; includes inherited hand rotation",
        "stats": {key: summary([row[key] for row in rows]) for key in (
            "wrist_gap_m", "palm_gap_m", "wrist_midpoint_z_m", "head_base_z_m",
            "height_budget_m", "legacy_midpoint_drift_m",
        )},
        "samples": rows,
    }
    if two_hand:
        from motionviewer.blender.mmd_props import build_two_hand_carry_box_scene

        fitted = build_two_hand_carry_box_scene(
            bpy, PropStyle(), armature=armature, frames=frames, grip_mode="side",
            **(dict(zip(("side", "depth", "height"), box_size, strict=True)) if box_size else {}),
        )
        report["two_hand_fit"] = fitted.report
        checks = []
        obj = fitted.props[0].obj
        for frame in (1, frames // 2, frames):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            coordinates = np.asarray([tuple(vertex.co) for vertex in obj.data.vertices])
            checks.append({"frame": frame, "scale": list(obj.scale),
                           "local_dimensions": np.ptp(coordinates, axis=0).tolist(),
                           "world_centre": list(obj.matrix_world.translation)})
        report["two_hand_object_checks"] = checks
        if render_dir is not None:
            gaps = [row["palm_gap_m"] for row in rows]
            selected = sorted({1, 30, 60, 100, frames, int(np.argmin(gaps)) + 1, int(np.argmax(gaps)) + 1})
            selected = [frame for frame in selected if frame <= frames]
            report["render_review"] = render_carry_review(
                bpy, actor, prop.obj, obj, selected, render_dir
            )
    return report


def weapon_probe(bpy, data: Path, *, mode: str = "legacy", render_dir: Path | None = None) -> dict:
    from mathutils import Vector

    from motionviewer.blender.mmd_props import (
        _shaft_axis,
        close_hand,
        fit_weapon_to_hand,
        load_weapon,
    )
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import clear_scene

    path = data / "active/128701.smplx.npz"
    model_dir = ROOT / "assets/fbx/pmx/星穹铁道—飞霄无外套"
    clear_scene()
    actor = create_fbx_actor_from_npz(
        path, label="weapon_probe", fbx_path=model_dir / "星穹铁道—飞霄（无外套）.pmx",
        retarget_mode="direct",
    )
    armature = actor.armature
    frame_before = bpy.context.scene.frame_current
    weapon = load_weapon(bpy, model_dir / "星穹铁道—飞霄武器.pmx", label="probe")
    fit_weapon_to_hand(bpy, weapon, armature, "右手首", grip="high", frame=1, mode=mode)
    frame_after = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    first_origin, _ = _shaft_axis(weapon, "high")
    # Explicit caller-side frame selection is the control for the frame argument.
    fit = fit_weapon_to_hand(bpy, weapon, armature, "右手首", grip="high", frame=1, mode=mode)
    bpy.context.view_layer.update()
    correct_origin, _ = _shaft_axis(weapon, "high")
    frame_difference = (first_origin - correct_origin).length
    before = {
        bone.name: [float(v) for v in bone.rotation_quaternion.to_euler("XYZ")]
        for bone in armature.pose.bones if bone.name.startswith("右")
        and any(f in bone.name for f in ("人指", "中指", "薬指", "小指", "親指"))
    }
    result = close_hand(
        bpy, armature, "右手首", frames=1, shaft_m=fit["shaft_mm"] / 1000,
        weapon=weapon, grip="high", palm_normal=Vector(fit["palm_normal"]),
        mode=mode,
    )
    bpy.context.view_layer.update()
    origin, axis = _shaft_axis(weapon, "high")
    radius = fit["shaft_mm"] / 2000
    fingers = {}
    for finger, segments in (("人指", "１２３"), ("中指", "１２３"), ("薬指", "１２３"),
                             ("小指", "１２３"), ("親指", "０１２")):
        chain = [armature.pose.bones.get("右" + finger + segment) for segment in segments]
        bones = [bone for bone in chain if bone is not None]
        if not bones:
            continue
        tip = armature.matrix_world @ bones[-1].tail
        delta = tip - origin
        radial = (delta - axis * delta.dot(axis)).length
        fingers[finger] = {
            "tip_to_cylinder_surface_mm": (radial - radius) * 1000,
            "tip_signed_palm_mm": delta.dot(Vector(fit["palm_normal"])) * 1000,
            "local_euler_before_deg": [np.degrees(before[bone.name]).tolist() for bone in bones],
            "local_euler_after_deg": [np.degrees(tuple(bone.rotation_quaternion.to_euler("XYZ"))).tolist()
                                      for bone in bones],
            "bone_lengths_mm": [bone.length * 1000 for bone in bones],
        }
        if len(bones) == 3 and bones[0].parent is not None:
            fingers[finger]["fk_input"] = {
                "parent_world": np.asarray(armature.matrix_world @ bones[0].parent.matrix).tolist(),
                "offsets": [np.asarray(bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local).tolist()
                            for bone in bones],
                "lengths": [bone.bone.length for bone in bones],
                "reference_eulers": [before[bone.name] for bone in bones],
                "origin": list(origin), "direction": list(axis),
            }
    rendered = []
    if render_dir is not None:
        from motionviewer.blender.mmd_toon import add_toon_lighting, apply_toon_shading
        from motionviewer.blender.render import _normalize_engine
        from motionviewer.blender.scene import setup_world

        setup_world()
        apply_toon_shading(list(actor.mesh_objects) + [weapon])
        add_toon_lighting([-1, -1, 0], [1, 1, 2])
        points = [armature.matrix_world @ armature.pose.bones[f"右{finger}{segment}"].head
                  for finger in ("人指", "中指", "薬指", "小指") for segment in ("１", "３")]
        target = sum(points, Vector()) / len(points)
        camera_data = bpy.data.cameras.new("prop_fit_grip")
        camera_data.lens = 85
        camera = bpy.data.objects.new("prop_fit_grip", camera_data)
        bpy.context.collection.objects.link(camera)
        scene = bpy.context.scene
        scene.camera = camera
        scene.render.engine = _normalize_engine("BLENDER_EEVEE")
        scene.render.resolution_x = scene.render.resolution_y = 640
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        render_dir.mkdir(parents=True, exist_ok=True)
        for tag, offset in (("A", (0.22, -0.22, 0.10)), ("B", (-0.22, -0.22, 0.10))):
            camera.location = target + Vector(offset)
            camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
            destination = render_dir / f"feixiao_{mode}_{tag}.png"
            scene.render.filepath = str(destination)
            bpy.ops.render.render(write_still=True)
            rendered.append(str(destination))
    return {
        "motion": str(path), "asset": "feixiao", "frame_before_fit": frame_before,
        "frame_after_requested_frame_1": frame_after,
        "grip_origin_difference_from_explicit_frame_1_mm": frame_difference * 1000,
        "fit": fit, "curl": result, "fingers": fingers,
        "rendered_views": rendered,
        "limitations": "Bone centres versus circular shaft proxy; no mesh collision or visual acceptance claimed.",
    }


def main() -> None:
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weapon-mode", choices=("legacy", "bounded"), default="legacy")
    parser.add_argument("--two-hand-box", action="store_true")
    parser.add_argument("--weapon-only", action="store_true")
    parser.add_argument("--carry-only", action="store_true")
    parser.add_argument("--box-size", type=float, nargs=3, default=None, metavar=("WIDTH", "DEPTH", "HEIGHT"))
    parser.add_argument("--box-review-dir", type=Path, default=None)
    parser.add_argument("--render-grip-dir", type=Path, default=None,
                        help="Optionally render two orthogonal 640px inspection stills")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.output.exists():
        raise SystemExit(f"Refusing to replace an existing measurement: {args.output}")
    if args.render_grip_dir is not None and args.render_grip_dir.exists() and any(args.render_grip_dir.iterdir()):
        raise SystemExit(f"Refusing to reuse a nonempty render directory: {args.render_grip_dir}")
    if args.box_review_dir is not None and args.box_review_dir.exists() and any(args.box_review_dir.iterdir()):
        raise SystemExit(f"Refusing to reuse a nonempty render directory: {args.box_review_dir}")
    if args.weapon_only and args.carry_only:
        raise SystemExit("Choose at most one of --weapon-only and --carry-only")
    report = {"carry": ({} if args.weapon_only else carry_probe(
        bpy, args.data, two_hand=args.two_hand_box or args.box_review_dir is not None,
        box_size=args.box_size, render_dir=args.box_review_dir)),
              "weapon": ({} if args.carry_only else weapon_probe(
                  bpy, args.data, mode=args.weapon_mode, render_dir=args.render_grip_dir))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PROP_FIT " + json.dumps({"carry": report["carry"].get("two_hand_fit", report["carry"].get("stats")),
                                   "weapon_mode": report["weapon"].get("fit", {}).get("fit_mode"),
                                   "curl": report["weapon"].get("curl")},
                                   ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
