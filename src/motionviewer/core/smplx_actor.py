"""Shared actor types for SMPL-X backends — no Blender dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

BODY_POSE_BONES = [
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

FOOT_POSE_MODES = frozenset({"source", "ankle_neutral", "neutral_feet"})


def override_foot_pose(body_pose: np.ndarray, mode: str) -> np.ndarray:
    """Return body pose with selected local foot channels reset to rest pose."""
    if mode not in FOOT_POSE_MODES:
        raise ValueError(f"foot pose mode must be one of {sorted(FOOT_POSE_MODES)}")
    result = np.asarray(body_pose, dtype=np.float32).reshape(-1, len(BODY_POSE_BONES), 3).copy()
    if mode == "source":
        return result
    names = ["left_ankle", "right_ankle"]
    if mode == "neutral_feet":
        names.extend(("left_foot", "right_foot"))
    result[:, [BODY_POSE_BONES.index(name) for name in names], :] = 0.0
    return result


@dataclass
class SmplxActor:
    label: str
    armature: Any
    mesh_objects: list[Any]
