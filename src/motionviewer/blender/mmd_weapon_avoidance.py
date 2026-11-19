"""Bounded non-gripping-arm clearance for the reviewed Feixiao axe profile.

This is a checked artistic adaptation, not a general collision solver. The stored
rotation seed is accepted only if evaluated arm triangles clear the weapon over
its whole transition; an infeasible seed leaves the animation untouched.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def clearance_weights(frames: np.ndarray, centre: float, half_width: float) -> np.ndarray:
    """Compact cosine bump with zero velocity at both ends, peak one at centre."""
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("clearance half width must be positive and finite")
    distance = np.minimum(np.abs(np.asarray(frames, dtype=float) - centre) / half_width, 1.)
    return .5 + .5 * np.cos(np.pi * distance)


def _crossings(weapon_points: list[Any], weapon_faces: list[tuple], weapon_tree: Any,
               points: list[Any], faces: list[tuple]) -> int:
    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import intersect_ray_tri

    tree = BVHTree.FromPolygons(points, faces, all_triangles=True)
    total = 0
    for wi, bi in weapon_tree.overlap(tree):
        first = [weapon_points[i] for i in weapon_faces[wi]]
        second = [points[i] for i in faces[bi]]
        found = False
        for triangle, other in ((first, second), (second, first)):
            for index in range(3):
                start, end = triangle[index], triangle[(index + 1) % 3]
                delta = end - start
                if delta.length_squared < 1e-16:
                    continue
                hit = intersect_ray_tri(*other, delta, start, True)
                if hit is not None and 1e-6 < (hit - start).dot(delta) / delta.length_squared < 1 - 1e-6:
                    found = True
                    break
            if found:
                break
        total += found
    return total


def _apply_weapon_left_avoidance(
    bpy: Any, armature: Any, weapon: Any, *, frame_start: int, frames: int, fps: float,
) -> dict:
    """Keep the elbow bend and wrist world orientation while turning the upper arm.

    Only the exact surface-grip geometry profile is supported. This changes the
    rendered left arm, never source NPZ/PMX files, torso, legs or gripping right arm.
    """
    from mathutils import Quaternion, Vector
    from mathutils.bvhtree import BVHTree

    from .mmd_contact import _set_world_rotation
    from .mmd_weapon_contact import _meshes, _profile

    profile = _profile(bpy, armature, weapon, "右手首", "high")
    config = profile.get("left_arm_clearance")
    if config is None:
        return {"applied": False, "reason": "No reviewed arm-clearance seed for this profile"}
    if frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("arm clearance requires positive frames and fps")
    scene, world = bpy.context.scene, armature.matrix_world
    upper, lower, hand = [armature.pose.bones["左" + name] for name in ("腕", "ひじ", "手首")]
    protected_names = [name for name in (
        "センター", "グルーブ", "下半身", "上半身", "上半身2", "上半身２", "首", "頭",
        "左肩", "右肩", "右腕", "右ひじ", "右手首", "左足", "右足", "左ひざ", "右ひざ", "左足首", "右足首",
    ) if name in armature.pose.bones]
    selections = []
    for obj in _meshes(bpy, armature):
        groups = {g.index: g.name for g in obj.vertex_groups}
        selected = {
            v.index for v in obj.data.vertices
            if sum(g.weight for g in v.groups if groups[g.group].startswith("左") and
                   any(term in groups[g.group] for term in ("指", "手首", "手捩", "ひじ", "腕"))) > .2
        }
        faces = [tuple(p.vertices) for p in obj.data.polygons if any(i in selected for i in p.vertices)]
        indices = sorted({i for face in faces for i in face})
        mapping = {index: i for i, index in enumerate(indices)}
        if faces:
            selections.append((obj, indices, [tuple(mapping[i] for i in face) for face in faces]))
    if not selections:
        raise ValueError("Cannot identify evaluated left-arm skin for clearance validation")
    wp = [v.co.copy() for v in weapon.data.vertices]
    wf = [tuple(p.vertices) for p in weapon.data.polygons]
    weapon_tree = BVHTree.FromPolygons(wp, wf, all_triangles=True)

    def count() -> int:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        inverse = weapon.matrix_world.inverted()
        total = 0
        for obj, indices, faces in selections:
            evaluated = obj.evaluated_get(depsgraph)
            matrix = inverse @ obj.matrix_world
            points = [matrix @ evaluated.data.vertices[i].co for i in indices]
            total += _crossings(wp, wf, weapon_tree, points, faces)
        return total

    originals, initial = {}, []
    for frame in range(frame_start, frame_start + frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        originals[frame] = {
            "armature_world": np.asarray(armature.matrix_world).copy(),
            "basis": [b.matrix_basis.copy() for b in (upper, hand)],
            "rotations": [np.asarray((world @ b.matrix).to_quaternion().to_matrix()).copy() for b in (upper, hand)],
            "wrist": np.asarray(world @ hand.head).copy(),
            "elbow_local": lower.matrix_basis.copy(),
            "protected": {name: np.asarray(world @ armature.pose.bones[name].matrix).copy() for name in protected_names},
            "weapon_rotation": weapon.matrix_world.to_quaternion().to_matrix().copy(),
        }
        crossings = count()
        if crossings:
            initial.append({"frame": frame, "triangle_crossings": crossings})
    def protected_error(frame: int) -> float:
        errors = [float(np.max(np.abs(np.asarray(armature.matrix_world) - originals[frame]["armature_world"]))),
                  float(np.max(np.abs(np.asarray(lower.matrix_basis) - np.asarray(originals[frame]["elbow_local"]))))]
        errors.extend(float(np.max(np.abs(np.asarray(world @ armature.pose.bones[name].matrix) - matrix)))
                      for name, matrix in originals[frame]["protected"].items())
        return max(errors)

    report = {
        "applied": False, "profile": profile["id"], "initial_crossings": initial,
        "source_motion_files_modified": False, "elbow_local_pose_preserved": True,
        "wrist_world_orientation_preserved": True,
        "protected_bones_measured": protected_names,
        "validation_phase": "preview",
        "limits": "Evaluated triangle boundary crossings at integer frames; not continuous collision certification. Original left-arm motion is deliberately adapted.",
    }
    if not initial:
        scene.frame_set(frame_start)
        report["reason"] = "No left-arm/weapon boundary crossings detected"
        return report
    radius = max(1, round(fps * config["transition_seconds"]))
    clusters = [[initial[0]["frame"]]]
    for row in initial[1:]:
        if row["frame"] - clusters[-1][-1] > 2 * radius:
            clusters.append([])
        clusters[-1].append(row["frame"])
    poses, rows, transitions = {}, [], []
    valid = True
    for cluster in clusters:
        centre = (cluster[0] + cluster[-1]) / 2
        half_width = radius + max(0., (cluster[-1] - cluster[0]) / 2 - 1)
        axis = originals[round(centre)]["weapon_rotation"] @ Vector(config["axis_in_weapon_frame"])
        axis.normalize()
        transitions.append({"centre_frame": centre, "half_width_frames": half_width,
                            "peak_shoulder_rotation_degrees": config["degrees"], "axis_world": list(axis)})
        affected = [frame for frame in originals if abs(frame - centre) < half_width]
        weights = clearance_weights(np.asarray(affected), centre, half_width)
        for frame, weight in zip(affected, weights, strict=True):
            scene.frame_set(frame)
            for bone, basis in zip((upper, hand), originals[frame]["basis"], strict=True):
                bone.matrix_basis = basis
            bpy.context.view_layer.update()
            rotation = Quaternion(axis, math.radians(config["degrees"]) * float(weight))
            _set_world_rotation(bpy, armature, upper, np.asarray(rotation.to_matrix()) @ originals[frame]["rotations"][0])
            _set_world_rotation(bpy, armature, hand, originals[frame]["rotations"][1])
            displacement = float(np.linalg.norm(np.asarray(world @ hand.head) - originals[frame]["wrist"]))
            angle = max(math.degrees(basis.to_quaternion().rotation_difference(bone.matrix_basis.to_quaternion()).angle)
                        for basis, bone in zip(originals[frame]["basis"], (upper, hand), strict=True))
            crossings = count()
            protection_error = protected_error(frame)
            valid &= crossings == 0 and displacement <= config["max_offset_m"] + 1e-5 and angle <= config["max_joint_degrees"] + 1e-3 and protection_error < 1e-5
            rows.append({"frame": frame, "wrist_offset_mm": displacement * 1000,
                         "max_local_rotation_change_degrees": angle, "triangle_crossings": crossings,
                         "protected_transform_error": protection_error})
            poses[frame] = [bone.matrix_basis.copy() for bone in (upper, hand)]
            for bone, basis in zip((upper, hand), originals[frame]["basis"], strict=True):
                bone.matrix_basis = basis
    report.update(transitions=transitions, validation=rows,
                  maximum_wrist_offset_mm=max(row["wrist_offset_mm"] for row in rows),
                  maximum_local_rotation_change_degrees=max(row["max_local_rotation_change_degrees"] for row in rows))
    if valid:
        for frame, matrices in poses.items():
            scene.frame_set(frame)
            for bone, basis in zip((upper, hand), matrices, strict=True):
                bone.rotation_mode = "QUATERNION"
                bone.matrix_basis = basis
                bone.keyframe_insert("rotation_quaternion", frame=frame)
        committed = []
        for frame in originals:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            expected = poses.get(frame, originals[frame]["basis"])
            key_error = max(float(np.max(np.abs(np.asarray(bone.matrix_basis) - np.asarray(matrix))))
                            for bone, matrix in zip((upper, hand), expected, strict=True))
            committed.append({"frame": frame, "triangle_crossings": count(),
                              "protected_transform_error": protected_error(frame),
                              "preview_to_committed_pose_error": key_error})
        committed_ok = all(row["triangle_crossings"] == 0 and row["protected_transform_error"] < 1e-5
                           and row["preview_to_committed_pose_error"] < 1e-5 for row in committed)
        report["committed_validation"] = committed
        report["validation_phase"] = "committed"
        report["applied"] = committed_ok
        if not committed_ok:
            report["reason"] = "Committed animation did not reproduce the checked pose; original action retained"
    else:
        report["reason"] = "Reviewed seed is not feasible for this motion; original arm animation retained"
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    return report


def apply_weapon_left_avoidance(
    bpy: Any, armature: Any, weapon: Any, *, frame_start: int, frames: int, fps: float,
) -> dict:
    """Apply to a copied Blender action; restore the original on rejection or error."""
    animation = armature.animation_data
    if animation is None or animation.action is None:
        raise ValueError("Arm clearance requires an existing retargeted animation action")
    original = animation.action
    working = original.copy()
    working.name = original.name + "_weapon_clearance"
    keep = False
    source_frame = bpy.context.scene.frame_current
    source_basis = {name: armature.pose.bones[name].matrix_basis.copy()
                    for name in ("左腕", "左手首") if name in armature.pose.bones}
    try:
        animation.action = working
        report = _apply_weapon_left_avoidance(
            bpy, armature, weapon, frame_start=frame_start, frames=frames, fps=fps,
        )
        keep = bool(report["applied"])
        report["render_action_copy"] = keep
        return report
    finally:
        if not keep:
            animation.action = original
            for name, basis in source_basis.items():
                armature.pose.bones[name].matrix_basis = basis
            bpy.context.scene.frame_set(source_frame)
            bpy.context.view_layer.update()
            if working.users == 0:
                bpy.data.actions.remove(working)
