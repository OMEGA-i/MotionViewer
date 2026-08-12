"""Ground-contact detection primitives — pure NumPy, no Blender dependency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactPatch:
    center: tuple[float, float, float]
    radius: float
    frames: tuple[int, int]


def detect_contact_patches(
    joints_blender: np.ndarray,
    *,
    foot_joint_ids: list[int],
    height_threshold: float,
    velocity_threshold: float,
    patch_radius: float,
) -> list[ContactPatch]:
    if joints_blender.ndim != 3:
        raise ValueError("joints_blender must have shape (T, J, 3)")
    patches: list[ContactPatch] = []
    floor_z = float(np.min(joints_blender[:, foot_joint_ids, 2]))
    for joint_id in foot_joint_ids:
        foot = joints_blender[:, joint_id, :]
        velocity = np.linalg.norm(np.diff(foot[:, :2], axis=0, prepend=foot[:1, :2]), axis=1)
        contact = (foot[:, 2] - floor_z <= height_threshold) & (velocity <= velocity_threshold)
        for start, end in _runs(contact):
            center = foot[start:end].mean(axis=0)
            center[2] = floor_z
            distances = np.linalg.norm(foot[start:end, :2] - center[:2], axis=1)
            radius = max(patch_radius, float(distances.max()) if distances.size else 0.0)
            patches.append(ContactPatch(tuple(float(v) for v in center), radius, (start, end)))
    return patches


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return [(s, e) for s, e in runs if e - s >= 2]
