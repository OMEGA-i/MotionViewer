"""Small convex cells following a rigid prop's actual surface (metres)."""

from __future__ import annotations

import itertools

import numpy as np


def _clip(polygon: np.ndarray, axis: int, bound: float, keep_above: bool) -> np.ndarray:
    result = []
    previous = polygon[-1]
    previous_inside = (previous[axis] >= bound) if keep_above else (previous[axis] <= bound)
    for current in polygon:
        inside = (current[axis] >= bound) if keep_above else (current[axis] <= bound)
        if inside != previous_inside:
            t = (bound - previous[axis]) / (current[axis] - previous[axis])
            result.append(previous + t * (current - previous))
        if inside:
            result.append(current)
        previous, previous_inside = current, inside
    return np.asarray(result, dtype=float).reshape(-1, 3)


def surface_cells(vertices: np.ndarray, triangles: np.ndarray, cell_size: float = 0.12) -> list[np.ndarray]:
    """Clip triangles into Y/Z cells before making their convex collision hulls.

    The thin weapon's X thickness is retained. Unlike grouping only original
    vertices, clipping also covers long faces crossing otherwise empty cells.
    Convexification can fill a concavity only within an individual cell, never
    bridge a large opening between distinct parts of an axe head.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite Nx3 coordinates")
    if (triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)):
        raise ValueError("triangles must be integer Nx3 indices")
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell size must be finite and positive")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("triangle index out of range")
    cells: dict[tuple[int, int], list[np.ndarray]] = {}
    for face in triangles:
        polygon = vertices[face]
        low = np.floor(polygon[:, 1:].min(axis=0) / cell_size).astype(int)
        high = np.floor(polygon[:, 1:].max(axis=0) / cell_size).astype(int)
        for key in itertools.product(range(low[0], high[0] + 1), range(low[1], high[1] + 1)):
            clipped = polygon.copy()
            for axis, index in zip((1, 2), key, strict=True):
                for bound, above in ((index * cell_size, True), ((index + 1) * cell_size, False)):
                    if not len(clipped):
                        break
                    clipped = _clip(clipped, axis, bound, above)
            if len(clipped) >= 3:
                cells.setdefault(key, []).extend(clipped)
    surfaces = []
    for _, points in sorted(cells.items()):
        points = np.unique(np.round(points, 8), axis=0)
        # Clipping at a cell boundary can leave a line with three or more points.
        # Adding one normal thickness to that line would still produce a planar
        # (zero-volume) Bullet hull. Such zero-area remnants cover no surface.
        if len(points) >= 3 and np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-7) >= 2:
            surfaces.append(points)
    return surfaces
