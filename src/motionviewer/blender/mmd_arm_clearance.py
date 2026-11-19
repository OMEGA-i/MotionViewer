"""Optional bounded arm clearance on a derived character, before garment physics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.arm_clearance import outward_rotation


def apply_arm_clearance(
    bpy: Any, armature: Any, *, frame_start: int, frame_end: int, degrees: float = 3.0
) -> dict:
    """Rotate upper arms slightly outward; preserve elbow bend and hand world orientation.

    The correction is an explicit character adaptation. It edits only the derived
    upper-arm and hand quaternion keys, not the source motion or root/leg channels.
    """
    from mathutils import Matrix

    from .retarget._animate import _linearize

    if not math.isfinite(degrees) or not 0.0 <= degrees <= 20.0:
        raise ValueError("arm clearance must be finite and between 0 and 20 degrees")
    report = dict(enabled=degrees > 0, max_degrees=degrees, frames=frame_end - frame_start + 1)
    if degrees == 0:
        return report
    bones = armature.pose.bones
    for prefix in ("左", "右"):
        for suffix in ("腕", "手首"):
            if prefix + suffix not in bones:
                raise ValueError(f"arm clearance requires {prefix + suffix}")
    protected = ["下半身", "左足", "右足", "左ひざ", "右ひざ", "左足首", "右足首"]
    root_before, protected_before, wrists_before, wrists_after = [], [], [], []
    max_rotation, max_shoulder, max_hand_rotation, max_protected = 0.0, 0.0, 0.0, 0.0
    originals = []
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        world = armature.matrix_world
        upper = [bones[p + "腕"] for p in ("左", "右")]
        wrists = [bones[p + "手首"] for p in ("左", "右")]
        originals.append(
            (
                [np.asarray(world @ b.matrix).copy() for b in upper],
                [np.asarray(world @ b.matrix).copy() for b in wrists],
            )
        )
        root_before.append(np.asarray(world).copy())
        protected_before.append([np.asarray(world @ bones[n].matrix).copy() for n in protected if n in bones])
    for index, frame in enumerate(range(frame_start, frame_end + 1)):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        upper_matrices, hand_matrices = originals[index]
        lateral = upper_matrices[0][:3, 3] - upper_matrices[1][:3, 3]
        before, after = [], []
        for side, prefix in enumerate(("左", "右")):
            upper, hand = bones[prefix + "腕"], bones[prefix + "手首"]
            original, wrist = upper_matrices[side], hand_matrices[side]
            rotation = outward_rotation(
                wrist[:3, 3] - original[:3, 3], lateral * (1 if side == 0 else -1), degrees
            )
            target = original.copy()
            target[:3, :3] = rotation @ original[:3, :3]
            upper.matrix = armature.matrix_world.inverted() @ Matrix(target.tolist())
            bpy.context.view_layer.update()
            current_hand = armature.matrix_world @ hand.matrix
            desired_hand = Matrix(wrist.tolist())
            desired_hand.translation = current_hand.translation
            hand.matrix = armature.matrix_world.inverted() @ desired_hand
            bpy.context.view_layer.update()
            actual_hand = np.asarray(armature.matrix_world @ hand.matrix)
            actual_upper = np.asarray(armature.matrix_world @ upper.matrix)
            max_rotation = max(
                max_rotation, float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
            )
            max_shoulder = max(max_shoulder, float(np.linalg.norm(actual_upper[:3, 3] - original[:3, 3])))
            a = (armature.matrix_world @ hand.matrix).to_quaternion()
            b = Matrix(wrist.tolist()).to_quaternion()
            max_hand_rotation = max(max_hand_rotation, math.degrees(a.rotation_difference(b).angle))
            before.append(wrist[:3, 3])
            after.append(actual_hand[:3, 3])
            for bone in (upper, hand):
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone.name)
        wrists_before.append(before)
        wrists_after.append(after)
        actual_protected = np.asarray(
            [np.asarray(armature.matrix_world @ bones[n].matrix) for n in protected if n in bones]
        )
        max_protected = max(
            max_protected,
            float(np.abs(actual_protected - np.asarray(protected_before[index])).max()),
            float(np.abs(np.asarray(armature.matrix_world) - root_before[index]).max()),
        )
    _linearize(armature)
    displacement = np.linalg.norm(np.asarray(wrists_after) - np.asarray(wrists_before), axis=-1)
    report.update(
        actual_max_upper_rotation_degrees=max_rotation,
        max_wrist_displacement_m=float(displacement.max()),
        mean_wrist_displacement_m=float(displacement.mean()),
        max_shoulder_displacement_m=max_shoulder,
        max_hand_world_rotation_change_degrees=max_hand_rotation,
        max_root_or_leg_matrix_change=max_protected,
    )
    return report
