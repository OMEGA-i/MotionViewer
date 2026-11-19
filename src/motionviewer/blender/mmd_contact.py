"""Measure and optionally correct MMD shoe support after retargeting, before physics.

The input animation stays intact on disk. All measurements use evaluated foot skin,
never the full-character bounding box, skirt, or a rest-pose ankle height.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FootSamples:
    ankles: np.ndarray
    toes: np.ndarray
    heels: np.ndarray
    sole_z: np.ndarray
    hips: np.ndarray
    knees: np.ndarray
    roots: np.ndarray
    vertex_counts: tuple[int, int]


def _foot_indices(actor: Any) -> list[tuple[Any, list[np.ndarray], list[np.ndarray], list[np.ndarray]]]:
    """Select weighted shoe skin below the bind ankle, with forefoot/heel patches."""
    armature = actor.armature
    selections = []
    for mesh in actor.mesh_objects:
        groups = {group.index: group.name for group in mesh.vertex_groups}
        coords = np.empty((len(mesh.data.vertices), 3))
        mesh.data.vertices.foreach_get("co", coords.ravel())
        world = np.asarray(mesh.matrix_world)
        coords = coords @ world[:3, :3].T + world[:3, 3]
        feet, toes, heels = [], [], []
        for side in ("左", "右"):
            names = {f"{side}足首", f"{side}足首D", f"{side}足先EX", f"{side}つま先"}
            indices = {i for i, name in groups.items() if name in names}
            ankle = np.asarray(armature.matrix_world @ armature.data.bones[f"{side}足首"].head_local)
            selected = np.asarray([
                v.index for v in mesh.data.vertices
                if sum(g.weight for g in v.groups if g.group in indices) >= 0.5
                and coords[v.index, 2] <= ankle[2] + 0.005
            ], dtype=int)
            feet.append(selected)
            if not len(selected):
                toes.append(selected)
                heels.append(selected)
                continue
            tip = armature.data.bones.get(f"{side}つま先")
            forward = np.asarray(armature.matrix_world @ tip.head_local) - ankle if tip else np.array([0., -1., 0.])
            forward[2] = 0
            forward /= max(float(np.linalg.norm(forward)), 1e-9)
            depth = (coords[selected] - ankle) @ forward
            # These are material points on the actual shoe, not the undriven MMD toe IK aim.
            toes.append(selected[depth >= np.percentile(depth, 75)])
            heels.append(selected[depth <= np.percentile(depth, 25)])
        selections.append((mesh, feet, toes, heels))
    return selections


def measure_feet(bpy: Any, actor: Any, *, frame_start: int, frames: int) -> FootSamples:
    """Sample the final skinned shoe geometry and FK chain for every requested frame."""
    selections = _foot_indices(actor)
    counts = tuple(sum(len(feet[side]) for _, feet, _, _ in selections) for side in range(2))
    if min(counts) < 8:
        raise ValueError(f"Cannot identify reliable left/right shoe skin: vertex counts {counts}")
    ankles = np.zeros((frames, 2, 3))
    hips, knees, toes, heels = (np.zeros_like(ankles) for _ in range(4))
    roots = np.zeros((frames, 3))
    sole_z = np.full((frames, 2), np.inf)
    armature = actor.armature
    for offset in range(frames):
        bpy.context.scene.frame_set(frame_start + offset)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = armature.evaluated_get(depsgraph)
        matrix = evaluated.matrix_world
        roots[offset] = matrix.translation
        for side, prefix in enumerate(("左", "右")):
            for output, name in ((hips, "足"), (knees, "ひざ"), (ankles, "足首")):
                bone = evaluated.pose.bones.get(prefix + name)
                if bone is None and name == "ひざ":
                    bone = evaluated.pose.bones.get(prefix + "膝")
                output[offset, side] = matrix @ bone.head
        toe_points: list[list[np.ndarray]] = [[], []]
        heel_points: list[list[np.ndarray]] = [[], []]
        for mesh, foot_ids, toe_ids, heel_ids in selections:
            obj = mesh.evaluated_get(depsgraph)
            data = obj.to_mesh()
            try:
                coords = np.empty((len(data.vertices), 3))
                data.vertices.foreach_get("co", coords.ravel())
                matrix_np = np.asarray(obj.matrix_world)
                coords = coords @ matrix_np[:3, :3].T + matrix_np[:3, 3]
                for side in range(2):
                    if len(foot_ids[side]):
                        sole_z[offset, side] = min(sole_z[offset, side], float(coords[foot_ids[side], 2].min()))
                        toe_points[side].append(coords[toe_ids[side]])
                        heel_points[side].append(coords[heel_ids[side]])
            finally:
                obj.to_mesh_clear()
        for side in range(2):
            toes[offset, side] = np.concatenate(toe_points[side]).mean(axis=0)
            heels[offset, side] = np.concatenate(heel_points[side]).mean(axis=0)
    return FootSamples(ankles, toes, heels, sole_z, hips, knees, roots, counts)


def estimate_display_floor(
    bpy: Any, actor: Any, *, frame_start: int, frames: int, fallback_floor: float,
) -> dict:
    """Place a fixed display floor below actual shoes without editing the motion.

    Clothing and hair must not lower the floor. The full-clip sole minimum avoids
    hiding shoe geometry in the floor; residual support gaps remain measurable.
    """
    scene = bpy.context.scene
    original_frame = scene.frame_current
    try:
        sample = measure_feet(bpy, actor, frame_start=frame_start, frames=frames)
        if not np.isfinite(sample.sole_z).all():
            raise ValueError("Shoe heights contain non-finite values")
        return {
            "method": "full_clip_shoe_minimum",
            "floor_z": float(sample.sole_z.min()),
            "frames": frames,
            "shoe_vertex_counts": list(sample.vertex_counts),
            "lowest_shoe_range_m": [float(value) for value in (
                sample.sole_z.min(axis=1).min(), sample.sole_z.min(axis=1).max(),
            )],
            "changes_motion": False,
        }
    except (KeyError, ValueError) as exc:
        return {
            "method": "mesh_bounds_fallback",
            "floor_z": float(fallback_floor),
            "reason": str(exc),
            "changes_motion": False,
        }
    finally:
        scene.frame_set(original_frame)
        bpy.context.view_layer.update()


def source_feet(path: Any, frames: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Read source ankle/toe markers without modifying the source archive."""
    from ..core.smplx_fk import SMPLX_BODY22_NAMES, source_to_blender

    with np.load(path, allow_pickle=False) as data:
        fps = float(data["fps"]) if "fps" in data else 30.0
        if "joints22" in data:
            joints = source_to_blender(data["joints22"][:frames])
            names = list(SMPLX_BODY22_NAMES)
            ankles = [names.index("left_ankle"), names.index("right_ankle")]
            toes = [names.index("left_foot"), names.index("right_foot")]
        else:
            joints = source_to_blender(data["joints77"][:frames])
            names = [str(value) for value in data["joint_names"]]
            ankles = [names.index("LeftFoot"), names.index("RightFoot")]
            toes = [names.index("LeftToeBase"), names.index("RightToeBase")]
    return joints[:, ankles], joints[:, toes], fps


def _set_world_rotation(bpy: Any, armature: Any, bone: Any, rotation: np.ndarray) -> None:
    from mathutils import Matrix

    world = armature.matrix_world @ bone.matrix
    desired = Matrix(np.asarray(rotation).tolist()).to_4x4()
    desired.translation = world.translation
    bone.matrix = armature.matrix_world.inverted() @ desired
    bpy.context.view_layer.update()


def _support_metrics(sample: FootSamples, weights: np.ndarray, modes: np.ndarray,
                     floor_z: float, fps: float) -> dict:
    interior = weights >= 0.99
    heights = sample.sole_z - floor_z
    mask = interior[:-1] & interior[1:]
    speeds = np.linalg.norm(np.diff(sample.ankles[..., :2], axis=0), axis=-1) * fps
    # Evaluate the material support point, and never compare different support
    # definitions across a heel/toe transition as if they were one fixed point.
    support = (sample.ankles + sample.toes) * 0.5
    support = np.where((modes == "toe")[..., None], sample.toes, support)
    support = np.where((modes == "ankle")[..., None], sample.ankles, support)
    support_speed = np.linalg.norm(np.diff(support[..., :2], axis=0), axis=-1) * fps
    support_mask = mask & (modes[:-1] == modes[1:]) & (modes[:-1] != "none")
    return {
        "contact_samples": int(interior.sum()),
        "contact_sole_abs_cm_median": float(np.median(np.abs(heights[interior])) * 100) if interior.any() else None,
        "contact_sole_abs_cm_p95": float(np.percentile(np.abs(heights[interior]), 95) * 100) if interior.any() else None,
        "ankle_contact_speed_cm_s_p90": float(np.percentile(speeds[mask], 90) * 100) if mask.any() else None,
        "support_speed_pairs": int(support_mask.sum()),
        "support_speed_cm_s_mean": float(support_speed[support_mask].mean() * 100) if support_mask.any() else None,
        "support_speed_cm_s_p90": float(np.percentile(support_speed[support_mask], 90) * 100) if support_mask.any() else None,
        "support_speed_cm_s_max": float(support_speed[support_mask].max() * 100) if support_mask.any() else None,
        "all_frames_max_penetration_cm": float(max(0., -heights.min()) * 100),
    }


def apply_foot_contact(
    bpy: Any, actor: Any, source_path: Any, *, frame_start: int, frames: int,
    floor_z: float | None = None, max_xy_m: float = 0.03, max_vertical_m: float = 0.06,
    max_pelvis_drop_m: float = 0.04,
) -> dict:
    """Bake bounded leg-only support correction; keep XY root trajectory and foot rotations.

    Contact comes from source markers, while actual shoe skin determines height.
    No constraint or scale change remains in the rig. Cloth is baked afterwards.
    """
    from ..core.foot_contact import FootContactSettings, plan_foot_contacts
    from ..core.limb_ik import limit_knee_change, solve_two_bone_positions, support_height_adjustment
    from .retarget._animate import _linearize
    from .retarget.mmd_solve import rotation_between

    source_ankles, source_toes, fps = source_feet(source_path, frames)
    before = measure_feet(bpy, actor, frame_start=frame_start, frames=frames)
    plan = plan_foot_contacts(source_ankles, source_toes, before.ankles, before.toes, fps=fps,
                             settings=FootContactSettings(max_correction_m=max_xy_m))
    if floor_z is None:
        indices = np.unique(np.rint(np.linspace(0, frames - 1, max(8, min(24, frames)))).astype(int))
        floor_z = float(np.percentile(before.sole_z[indices].min(axis=1), 25))
    weights = np.asarray(plan.weights)
    goals = np.asarray(plan.ankle_targets).copy()
    # Downward adjustments require support; upward adjustments also prevent a shoe
    # from crossing the chosen ground on a transition frame.
    goals[:, :, 2] += support_height_adjustment(before.sole_z, weights, floor_z=floor_z, limit_m=max_vertical_m)
    penetration_lift = np.maximum(0., floor_z - before.sole_z.min(axis=1))
    root_z = penetration_lift.copy()
    for frame in range(frames):
        requests, confidence = [], []
        for side in range(2):
            if weights[frame, side] <= 0:
                continue
            hip, knee, ankle = before.hips[frame, side], before.knees[frame, side], before.ankles[frame, side]
            # Preserve the original knee bend instead of spending all remaining
            # leg extension on a hovering shoe (which caused visible knee snaps).
            length = np.linalg.norm(ankle - hip)
            horizontal = np.linalg.norm(goals[frame, side, :2] - hip[:2])
            if horizontal < length and hip[2] > goals[frame, side, 2]:
                required = goals[frame, side, 2] + np.sqrt(length * length - horizontal * horizontal) - hip[2]
                requests.append(float(required))
                confidence.append(weights[frame, side])
        if requests:
            root_z[frame] = float(np.average(requests, weights=confidence))
    root_z = np.clip(root_z, -max_pelvis_drop_m, max_vertical_m)
    if frames >= 5:
        root_z = np.convolve(np.pad(root_z, (2, 2), mode="edge"), np.array([1, 2, 3, 2, 1]) / 9., mode="valid")
    # Negative corrections require support. A positive correction may also clear
    # an unclassified landing that is already below the physical floor.
    root_z = np.where(root_z < 0, root_z * weights.max(axis=1), root_z)
    # The unsupported leg keeps its original pose under the small pelvis shift.
    goals[:, :, 2] += root_z[:, None] * (1. - weights)
    goals[:, :, 2] += np.maximum(0., floor_z - (before.sole_z + goals[:, :, 2] - before.ankles[:, :, 2]))
    goals[:, :, 2] = before.ankles[:, :, 2] + np.clip(
        goals[:, :, 2] - before.ankles[:, :, 2], -max_vertical_m, max_vertical_m)
    shifted_hips, shifted_knees, shifted_ankles = (value.copy() for value in (before.hips, before.knees, before.ankles))
    for value in (shifted_hips, shifted_knees, shifted_ankles):
        value[:, :, 2] += root_z[:, None]
    goals, knee_delta = limit_knee_change(shifted_hips, shifted_knees, shifted_ankles, goals,
                                         max_step_degrees=75. / fps)
    armature = actor.armature
    max_reach_error = 0.
    for offset in range(frames):
        frame = frame_start + offset
        bpy.context.scene.frame_set(frame)
        location = before.roots[offset].copy()
        location[2] += root_z[offset]
        armature.location = location
        armature.keyframe_insert(data_path="location", frame=frame)
        bpy.context.view_layer.update()
        for side, prefix in enumerate(("左", "右")):
            upper = armature.pose.bones[prefix + "足"]
            lower = armature.pose.bones.get(prefix + "ひざ") or armature.pose.bones[prefix + "膝"]
            foot = armature.pose.bones[prefix + "足首"]
            world = armature.matrix_world
            hip, knee, ankle = (np.asarray(world @ b.head).copy() for b in (upper, lower, foot))
            if np.linalg.norm(ankle - goals[offset, side]) < 1e-7:
                continue
            foot_rotation = np.asarray((world @ foot.matrix).to_quaternion().to_matrix()).copy()
            upper_rotation = np.asarray((world @ upper.matrix).to_quaternion().to_matrix()).copy()
            middle, reached, error = solve_two_bone_positions(
                hip, knee, ankle, goals[offset, side], pole_hint=upper_rotation[:, 2])
            max_reach_error = max(max_reach_error, error)
            _set_world_rotation(bpy, armature, upper, rotation_between(knee - hip, middle - hip) @ upper_rotation)
            world = armature.matrix_world
            knee, ankle = (np.asarray(world @ b.head).copy() for b in (lower, foot))
            lower_rotation = np.asarray((world @ lower.matrix).to_quaternion().to_matrix()).copy()
            _set_world_rotation(bpy, armature, lower,
                                rotation_between(ankle - knee, reached - knee) @ lower_rotation)
            _set_world_rotation(bpy, armature, foot, foot_rotation)
            for bone in (upper, lower, foot):
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone.name)
    _linearize(armature)
    after = measure_feet(bpy, actor, frame_start=frame_start, frames=frames)
    return {
        "enabled": True, "floor_z": floor_z, "detection": plan.summary(),
        "shoe_vertices": list(before.vertex_counts), "max_xy_m": max_xy_m,
        "max_vertical_m": max_vertical_m, "max_pelvis_drop_m": float(max(0., -root_z.min())),
        "max_pelvis_lift_m": float(max(0., root_z.max())),
        "max_knee_adjustment_degrees": float(np.abs(knee_delta).max()),
        "max_knee_adjustment_speed_deg_s": float(np.max(np.abs(np.diff(knee_delta, axis=0))) * fps) if frames > 1 else 0.,
        "root_xy_max_change_m": float(np.max(np.abs(after.roots[:, :2] - before.roots[:, :2]))),
        "max_unreachable_error_m": max_reach_error,
        "before": _support_metrics(before, weights, plan.support_modes, floor_z, fps),
        "after": _support_metrics(after, weights, plan.support_modes, floor_z, fps),
        "unclassified_frames": np.flatnonzero(weights.max(axis=1) == 0).tolist(),
    }
