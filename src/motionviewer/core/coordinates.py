from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .schema import CoordinateSystem


def source_to_blender_points(
    points: NDArray[np.floating],
    coordinate_system: CoordinateSystem,
    *,
    unit_scale: float = 1.0,
) -> NDArray[np.floating]:
    """Convert source points to Blender's z-up convention.

    The observed `smplx_body22_fitted_aa` files are y-up with travel mostly
    along source z. We map source `(x, y, z)` to Blender `(x, -z, y)` so forward
    travel appears along Blender negative Y while height becomes Blender Z.
    """

    arr = np.asarray(points, dtype=np.float32)
    if coordinate_system.vertical_axis != 1:
        return _generic_vertical_to_z(arr, coordinate_system.vertical_axis) * unit_scale
    converted = np.empty_like(arr, dtype=np.float32)
    converted[..., 0] = arr[..., 0]
    converted[..., 1] = -arr[..., 2]
    converted[..., 2] = arr[..., 1]
    return converted * unit_scale


def blender_to_source_points(
    points: NDArray[np.floating],
    coordinate_system: CoordinateSystem,
    *,
    unit_scale: float = 1.0,
) -> NDArray[np.floating]:
    arr = np.asarray(points, dtype=np.float32) / unit_scale
    if coordinate_system.vertical_axis != 1:
        raise NotImplementedError("Only y-up reverse conversion is implemented")
    converted = np.empty_like(arr, dtype=np.float32)
    converted[..., 0] = arr[..., 0]
    converted[..., 1] = arr[..., 2]
    converted[..., 2] = -arr[..., 1]
    return converted


def align_floor(points: NDArray[np.floating], floor_z: float | None = None) -> NDArray[np.floating]:
    arr = np.asarray(points, dtype=np.float32).copy()
    z_floor = float(np.min(arr[..., 2])) if floor_z is None else floor_z
    arr[..., 2] -= z_floor
    return arr


def bounding_box(points: NDArray[np.floating]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return arr.min(axis=0), arr.max(axis=0)


def merge_bounds(
    bounds: list[tuple[NDArray[np.floating], NDArray[np.floating]]],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if not bounds:
        zero = np.zeros(3, dtype=np.float32)
        return zero, zero
    mins = np.stack([np.asarray(b[0], dtype=np.float32) for b in bounds], axis=0)
    maxs = np.stack([np.asarray(b[1], dtype=np.float32) for b in bounds], axis=0)
    return mins.min(axis=0), maxs.max(axis=0)


def _generic_vertical_to_z(arr: NDArray[np.floating], vertical_axis: int) -> NDArray[np.float32]:
    if vertical_axis not in (0, 1, 2):
        raise ValueError(f"vertical_axis must be 0, 1, or 2; got {vertical_axis}")
    axes = [0, 1, 2]
    axes.remove(vertical_axis)
    converted = np.empty_like(arr, dtype=np.float32)
    converted[..., 0] = arr[..., axes[0]]
    converted[..., 1] = arr[..., axes[1]]
    converted[..., 2] = arr[..., vertical_axis]
    return converted
