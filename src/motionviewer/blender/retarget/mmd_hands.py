"""Relaxed hand pose for an MMD rig.

Body-22 SMPL-X carries no hand pose, so fingers otherwise stay on the model's
flat bind pose with every joint dead straight, which reads as a mannequin no
matter how good the arms are.  This applies a static, mild flexion.

The flexion axis is derived from the rig instead of hard-coded: fingers flex
toward the side the palm faces, and that side is identified by the thumb, which
is the one digit that lies on the palmar aspect.  A model whose hands bind at a
different angle therefore still gets a correct curl.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .mmd_solve import _axis_angle_matrix
from .twist import matrix_to_quaternion

# Base -> tip. MMD numbers the thumb from 0 and the other digits from 1.
_FINGERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("親指", ("親指０", "親指１", "親指２")),
    ("人指", ("人指１", "人指２", "人指３")),
    ("中指", ("中指１", "中指２", "中指３")),
    ("薬指", ("薬指１", "薬指２", "薬指３")),
    ("小指", ("小指１", "小指２", "小指３")),
)

# Relaxed flexion per joint, base -> tip. A resting hand curls more at the
# middle joint than at the knuckle, and the little finger more than the index.
_FLEXION_DEGREES: dict[str, tuple[float, float, float]] = {
    "親指": (5.0, 11.0, 9.0),
    "人指": (13.0, 23.0, 16.0),
    "中指": (14.0, 25.0, 17.0),
    "薬指": (15.0, 26.0, 18.0),
    "小指": (16.0, 27.0, 19.0),
}

_SIDES: tuple[tuple[str, str], ...] = (("L", "左"), ("R", "右"))


def _rest_rotation(world: Any, bone: Any) -> np.ndarray:
    return np.asarray((world @ bone.matrix_local).to_quaternion().to_matrix(), dtype=np.float64)


def _rest_head(world: Any, bone: Any) -> np.ndarray:
    return np.asarray((world @ bone.matrix_local).translation, dtype=np.float64)


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-12 else vector


def _flexion_axis(world: Any, bones: Any, prefix: str) -> np.ndarray | None:
    """World axis whose positive rotation curls the fingers into the palm."""
    required = (f"{prefix}中指１", f"{prefix}人指１", f"{prefix}小指１", f"{prefix}親指０")
    if any(name not in bones for name in required):
        return None
    middle, index, little, thumb = (bones[name] for name in required)

    finger_direction = _normalize(_rest_rotation(world, middle)[:, 1])
    across_palm = _rest_head(world, little) - _rest_head(world, index)
    palm_normal = _normalize(np.cross(finger_direction, across_palm))
    if float(np.linalg.norm(palm_normal)) <= 1e-9:
        return None
    # The thumb sits on the palmar side, which fixes the sign of the normal.
    if float(np.dot(palm_normal, _rest_rotation(world, thumb)[:, 1])) < 0.0:
        palm_normal = -palm_normal

    axis = _normalize(np.cross(finger_direction, palm_normal))
    if float(np.linalg.norm(axis)) <= 1e-9:
        return None
    # Positive rotation about the axis must carry the fingertip palmward. Mirror
    # rigs flip the handedness of the cross products, so check it rather than
    # assume it.
    if float(np.dot(np.cross(axis, finger_direction), palm_normal)) < 0.0:
        axis = -axis
    return axis


def relax_hands(armature: Any, *, amount: float = 1.0) -> dict[str, Any]:
    """Pose both hands into a mild resting curl. Returns an audit record."""
    world = armature.matrix_world
    bones = armature.data.bones
    report: dict[str, Any] = {"amount": float(amount), "hands": {}, "posed_bones": []}
    if amount <= 0.0:
        return report

    for side, prefix in _SIDES:
        axis = _flexion_axis(world, bones, prefix)
        if axis is None:
            report["hands"][side] = "skipped: incomplete finger chain"
            continue
        report["hands"][side] = {"flexion_axis": [float(value) for value in axis]}
        for finger, chain in _FINGERS:
            degrees = _FLEXION_DEGREES[finger]
            for bone_name, angle in zip(chain, degrees, strict=False):
                full_name = f"{prefix}{bone_name}"
                bone = bones.get(full_name)
                pose_bone = armature.pose.bones.get(full_name)
                if bone is None or pose_bone is None:
                    continue
                # A pose channel acts in the bone's rest-local frame, and each
                # joint's parent has already curled about the same world axis,
                # so the local axis is the world axis in that rest frame.
                local_axis = _rest_rotation(world, bone).T @ axis
                rotation = _axis_angle_matrix(
                    _normalize(local_axis), math.radians(float(angle) * float(amount))
                )
                pose_bone.rotation_mode = "QUATERNION"
                pose_bone.rotation_quaternion = tuple(
                    float(value) for value in matrix_to_quaternion(rotation)
                )
                report["posed_bones"].append(full_name)
    return report
