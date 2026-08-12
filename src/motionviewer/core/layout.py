"""Pure layout math shared by host-side preparation and Blender adapters."""

from __future__ import annotations

import numpy as np


def root_aligned_joints(joints: np.ndarray) -> np.ndarray:
    aligned = np.asarray(joints, dtype=np.float32).copy()
    aligned -= aligned[0:1, 0:1, :]
    return aligned


def trajectory_footprint(joints: np.ndarray, padding: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    root = np.asarray(joints[:, 0, :], dtype=np.float32)
    xy = root[:, :2]
    mins = xy.min(axis=0) - padding
    maxs = xy.max(axis=0) + padding
    return mins, maxs


def start_root_offsets(
    root_positions: list[np.ndarray],
    layout_offsets: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    return [
        tuple((np.asarray(offset, dtype=np.float32) - np.asarray(root, dtype=np.float32)).tolist())
        for root, offset in zip(root_positions, layout_offsets, strict=True)
    ]


def transform_bounds(
    bounds: tuple[list[float], list[float]],
    offset: tuple[float, float, float],
) -> tuple[list[float], list[float]]:
    mins = np.asarray(bounds[0], dtype=np.float32) + np.asarray(offset, dtype=np.float32)
    maxs = np.asarray(bounds[1], dtype=np.float32) + np.asarray(offset, dtype=np.float32)
    return mins.tolist(), maxs.tolist()


def merge_json_bounds(bounds: list[tuple[list[float], list[float]]]) -> tuple[list[float], list[float]]:
    if not bounds:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    mins = np.stack([np.asarray(item[0], dtype=np.float32) for item in bounds], axis=0)
    maxs = np.stack([np.asarray(item[1], dtype=np.float32) for item in bounds], axis=0)
    return mins.min(axis=0).tolist(), maxs.max(axis=0).tolist()
