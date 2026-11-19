"""Geometry for optional prop reconstructions; never changes the input motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoHandBoxFit:
    size: tuple[float, float, float]
    centres: np.ndarray
    rotations: np.ndarray
    report: dict


def box_signed_distance(points: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Exact point-to-solid-box distance in the box frame; negative means inside."""
    points, size = np.asarray(points, dtype=float), np.asarray(size, dtype=float)
    if points.shape[-1:] != (3,) or size.shape != (3,) or np.any(size <= 0):
        raise ValueError("Expected (..., 3) points and three positive box dimensions")
    delta = np.abs(points) - size / 2
    return np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(np.max(delta, axis=-1), 0)


def box_triangle_intersections(triangles: np.ndarray, size: np.ndarray, *, inset: float = 0.) -> np.ndarray:
    """Triangle/solid-box intersection by separating axes, in the box frame.

    A positive inset excludes mere surface contact from the reported overlaps.
    """
    triangles = np.asarray(triangles, dtype=float)
    half = np.asarray(size, dtype=float) / 2 - inset
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or np.any(half <= 0):
        raise ValueError("Expected (N, 3, 3) triangles and positive inset box dimensions")
    edges = np.roll(triangles, -1, axis=1) - triangles
    axes = [np.broadcast_to(axis, (len(triangles), 3)) for axis in np.eye(3)]
    axes.append(np.cross(edges[:, 0], edges[:, 1]))
    axes.extend(np.cross(edges[:, edge], axis) for edge in range(3) for axis in np.eye(3))
    overlaps = np.ones(len(triangles), dtype=bool)
    for axis in axes:
        projected = np.einsum("nvc,nc->nv", triangles, axis)
        radius = np.abs(axis) @ half
        overlaps &= (projected.min(axis=1) <= radius + 1e-12) & (projected.max(axis=1) >= -radius - 1e-12)
    return overlaps


def plan_box_elbows(shoulders: np.ndarray, elbows: np.ndarray, wrists: np.ndarray,
                    goals: np.ndarray, centres: np.ndarray, rotations: np.ndarray,
                    size: np.ndarray, *, radius_m: float = .025,
                    max_degrees: float = 30., max_step_degrees: float = 4.) -> tuple[np.ndarray, np.ndarray]:
    """Choose continuous bend-plane rotations while keeping wrist goals and limb lengths.

    The capsule is a planning proxy; evaluated skin/triangle checks remain required.
    A dynamic program bounds frame-to-frame pole changes and avoids greedy flips.
    """
    from .limb_ik import solve_two_bone_positions

    states = np.linspace(-max_degrees, max_degrees, 31)
    count = len(shoulders)
    candidates = np.empty((count, len(states), 3))
    costs = np.empty((count, len(states)))
    fractions = np.linspace(.1, .95, 14)[:, None]
    for frame in range(count):
        shoulder, elbow, wrist, goal = (values[frame] for values in (shoulders, elbows, wrists, goals))
        middle, reached, _ = solve_two_bone_positions(shoulder, elbow, wrist, goal)
        axis = reached - shoulder
        axis /= np.linalg.norm(axis)
        vector = middle - shoulder
        for index, degrees in enumerate(states):
            angle = np.radians(degrees)
            rotated = (vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)
                       + axis * np.dot(axis, vector) * (1 - np.cos(angle)))
            point = shoulder + rotated
            candidates[frame, index] = point
            samples = point * (1 - fractions) + reached * fractions
            local = (samples - centres[frame]) @ rotations[frame]
            penetration = np.maximum(radius_m - box_signed_distance(local, size), 0)
            costs[frame, index] = 100 * float(np.mean(penetration ** 2) + 5 * penetration.max() ** 2)
            costs[frame, index] += .0005 * (degrees / max_degrees) ** 2
    transitions = np.abs(states[:, None] - states[None, :])
    transition_cost = np.where(transitions <= max_step_degrees + 1e-9,
                               .00001 * (transitions / max_step_degrees) ** 2, np.inf)
    cumulative = costs[0].copy()
    parents = np.zeros((count, len(states)), int)
    for frame in range(1, count):
        options = cumulative[:, None] + transition_cost
        parents[frame] = np.argmin(options, axis=0)
        cumulative = costs[frame] + np.min(options, axis=0)
    chosen = np.zeros(count, int)
    chosen[-1] = int(np.argmin(cumulative))
    for frame in range(count - 1, 0, -1):
        chosen[frame - 1] = parents[frame, chosen[frame]]
    return candidates[np.arange(count), chosen], states[chosen]


def _stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    return {name: round(float(value * scale), 3) for name, value in zip(
        ("min", "median", "p95", "max"), np.percentile(values, (0, 50, 95, 100)), strict=True
    )}


def fit_two_hand_box(
    left: np.ndarray,
    right: np.ndarray,
    heads: np.ndarray,
    torsos: np.ndarray,
    *,
    grip_mode: str = "side",
    fit_range: tuple[int, int] | None = None,
    side: float | None = None,
    depth: float | None = None,
    height: float | None = None,
) -> TwoHandBoxFit:
    """Fit one rigid box to animated palms, with explicit contact residuals.

    ``fit_range`` uses zero-based half-open indices. Otherwise choose a contiguous
    20%-of-clip window with the least changing hand separation and hand/head height.
    Size stays fixed even when the source hands separate; the error is reported,
    never hidden by stretching the box or editing the arms.

    ``side`` grip places palms near the rear upper corners. ``bottom`` places the
    base over them. Head clearance is point-to-box geometry, not mesh collision or
    camera visibility. All coordinates and dimensions are metres.
    """
    arrays = [np.asarray(value, dtype=float) for value in (left, right, heads, torsos)]
    if any(a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Expected finite (frames, 3) palm, head and torso positions")
    if len({a.shape for a in arrays}) != 1 or len(arrays[0]) == 0:
        raise ValueError("Position arrays must have the same nonzero number of frames")
    if grip_mode not in {"side", "bottom"}:
        raise ValueError("grip_mode must be 'side' or 'bottom'")
    left, right, heads, torsos = arrays
    count = len(left)
    gaps = np.linalg.norm(right - left, axis=1)
    if np.any(gaps < 0.03):
        raise ValueError("Hands less than 30 mm apart do not define a carry-box frame")
    midpoint = (left + right) / 2
    head_budget = heads[:, 2] - midpoint[:, 2] - 0.04
    if fit_range is None:
        window = min(count, max(8, round(count * 0.2)))
        start = min(range(count - window + 1), key=lambda i: (
            np.ptp(gaps[i:i + window]) + 0.25 * np.ptp(head_budget[i:i + window])
        ))
        end = start + window
    else:
        start, end = fit_range
        if not (0 <= start < end <= count):
            raise ValueError("fit_range must select existing frames with start < end")
    width = float(side) if side is not None else float(np.median(gaps[start:end]))
    box_depth = float(depth) if depth is not None else float(np.clip(width * 0.75, 0.12, 0.32))
    # The head budget only limits upward height; a side grip sits 75% up the box.
    # A negative budget is reported as infeasible, not used to shrink to zero.
    available = float(np.percentile(head_budget[start:end], 10))
    upward_share = 0.25 if grip_mode == "side" else 1.0
    height_cap = available / upward_share if available > 0 else 0.08
    box_height = float(height) if height is not None else float(np.clip(min(width * 0.6, height_cap), 0.08, 0.24))
    size = np.asarray((width, box_depth, box_height), dtype=float)
    if not np.isfinite(size).all() or np.any(size < 0.04):
        raise ValueError("Each box dimension must be finite and at least 40 mm")

    rotations = np.empty((count, 3, 3))
    centres = np.empty((count, 3))
    clearances = np.empty(count)
    contact_error = np.empty(count)
    up = np.array((0.0, 0.0, 1.0))
    # Resolve the front/back ambiguity once, using the first reliable torso offset.
    # Repeating a sign test each frame can flip the box 180 degrees when the torso
    # crosses the hand plane by a millimetre, even though neither hand moved.
    orientation_sign = 1.0
    for index in range(count):
        x = (right[index] - left[index]) / gaps[index]
        z = up - x * np.dot(up, x)
        if np.linalg.norm(z) < 0.05:
            continue
        y = np.cross(z / np.linalg.norm(z), x)
        projection = float(np.dot(y, midpoint[index] - torsos[index]))
        if abs(projection) > 0.02:
            orientation_sign = 1.0 if projection > 0 else -1.0
            break
    for index in range(count):
        x = (right[index] - left[index]) / gaps[index] * orientation_sign
        z = up - x * np.dot(up, x)
        if np.linalg.norm(z) < 0.05:
            raise ValueError(f"Hands almost vertically aligned at index {index}; carry orientation is ambiguous")
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        rotation = np.column_stack((x, y, z))
        if grip_mode == "side":
            offset = y * (box_depth / 2 - 0.01) - z * (box_height * 0.25)
        else:
            offset = z * (box_height / 2 + 0.005)
        centre = midpoint[index] + offset
        rotations[index], centres[index] = rotation, centre
        local_head = rotation.T @ (heads[index] - centre)
        outside = np.abs(local_head) - size / 2
        clearances[index] = (np.linalg.norm(np.maximum(outside, 0)) if np.any(outside > 0)
                             else float(np.max(outside)))
        if grip_mode == "side":
            contact_error[index] = abs(gaps[index] - width) / 2
        else:
            # Palms may support any point inside the bottom rectangle; a wide box
            # does not require the hands to touch its outer side walls.
            palms = np.asarray((left[index], right[index]))
            local_palms = (palms - centre) @ rotation
            horizontal = np.maximum(np.abs(local_palms[:, :2]) - size[:2] / 2, 0)
            vertical = local_palms[:, 2] + box_height / 2
            contact_error[index] = float(np.max(np.sqrt(np.sum(horizontal ** 2, axis=1) + vertical ** 2)))
    report = {
        "mode": "two_hand_rigid_box",
        "grip_mode": grip_mode,
        "size_m": {key: round(float(value), 4) for key, value in zip(("side", "depth", "height"), size, strict=True)},
        "fit_range_0based_half_open": [int(start), int(end)],
        "fit_selection": "explicit" if fit_range is not None else "least_varying_contiguous_window",
        "hand_gap_mm": _stats(gaps, 1000),
        "contact_error_per_hand_mm": _stats(contact_error, 1000),
        "contact_error_definition": "distance to side wall" if grip_mode == "side" else "distance to bottom rectangle",
        "frames_over_30mm_contact_error": int(np.count_nonzero(contact_error > 0.03)),
        "head_point_box_distance_mm": _stats(clearances, 1000),
        "frames_head_point_within_80mm": int(np.count_nonzero(clearances < 0.08)),
        "fit_window_upward_height_budget_m": round(available, 4),
        "source_motion_modified": False,
        "limitations": "Palms/head are bone-derived proxies; no mesh collision or face visibility guarantee. "
                       "Large contact residuals require review, not automatic arm or box scaling.",
    }
    report[f"{grip_mode}_contact_error_per_hand_mm"] = _stats(contact_error, 1000)
    return TwoHandBoxFit(tuple(float(v) for v in size), centres, rotations, report)


def finger_chain_points(
    parent_world: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    eulers: np.ndarray,
) -> np.ndarray:
    """Head plus three tails for an ordinary three-bone local-XYZ finger chain."""
    transform = np.asarray(parent_world, dtype=float).copy()
    points = []
    for index, (offset, length, angles) in enumerate(zip(offsets, lengths, eulers, strict=True)):
        x, y, z = angles
        cx, cy, cz = np.cos((x, y, z))
        sx, sy, sz = np.sin((x, y, z))
        rotation = np.array(((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
                             (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
                             (-sy, cy * sx, cy * cx)))
        basis = np.eye(4)
        basis[:3, :3] = rotation
        transform = transform @ offset @ basis
        if index == 0:
            points.append(transform[:3, 3].copy())
        points.append((transform @ np.array((0, length, 0, 1)))[:3])
    return np.asarray(points)


def fit_finger_cylinder(
    parent_world: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    reference_eulers: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float,
    *,
    limits_degrees: tuple[float, float, float] = (90, 110, 90),
) -> dict:
    """Fit independent *total* local-X flexions, preserving local-Y/Z spread.

    This is a bounded bone/cylinder approximation, not a hand-mesh collision solve.
    Fingers may remain short of contact when the rig cannot reach the shaft.
    """
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and positive")
    direction = np.asarray(direction, dtype=float)
    if np.linalg.norm(direction) < 1e-9:
        raise ValueError("direction must be nonzero")
    direction = direction / np.linalg.norm(direction)
    origin = np.asarray(origin, dtype=float)
    limits = np.asarray(limits_degrees, dtype=float)
    if limits.shape != (3,) or np.any(limits <= 0):
        raise ValueError("Three positive flexion limits are required")
    reference = np.asarray(reference_eulers, dtype=float)
    if reference.shape != (3, 3) or not np.isfinite(reference).all():
        raise ValueError("reference_eulers must be a finite (3, 3) array")

    def evaluate(angles):
        eulers = reference.copy()
        eulers[:, 0] = np.radians(angles)
        points = finger_chain_points(parent_world, offsets, lengths, eulers)
        samples = np.concatenate([points[i:i + 1] * (1 - t) + points[i + 1:i + 2] * t
                                  for i in range(3) for t in (0.25, 0.5, 0.75, 1)])
        delta = samples - origin
        radial = delta - (delta @ direction)[:, None] * direction
        distances = np.linalg.norm(radial, axis=1)
        penetration = np.maximum(radius - distances, 0)
        tip_error = abs(distances[-1] - radius)
        # Penalize the entire phalanx chain entering the shaft, not only its tip.
        error = tip_error ** 2 + 12 * float(np.mean(penetration ** 2))
        error += radius ** 2 * 0.001 * float(np.mean(((angles - (45, 70, 40)) / limits) ** 2))
        # Artistic coordination prior: avoid a straight finger ending in a 90-degree
        # hooked tip. These are pose-shape constraints, not anatomical limit claims.
        if angles[2] > angles[1] + 5 or angles[1] < 0.35 * angles[0]:
            error = float("inf")
        return error, tip_error, float(penetration.max()), points

    best = None
    for initial in ((20, 40, 20), (45, 70, 40), (75, 95, 60)):
        angles = np.minimum(np.asarray(initial, dtype=float), limits)
        angles[0] = min(angles[0], angles[1] / 0.35)
        angles[2] = min(angles[2], angles[1] + 5)
        for step in (20, 10, 5, 2):
            for _ in range(2):
                for joint in range(3):
                    choices = (np.arange(0, limits[joint] + 1e-6, step) if step == 20
                               else np.clip(angles[joint] + np.arange(-2, 3) * step, 0, limits[joint]))
                    choices = np.unique(np.r_[choices, angles[joint], limits[joint]])
                    candidates = [angles.copy() for _ in choices]
                    for candidate, value in zip(candidates, choices, strict=True):
                        candidate[joint] = value
                    angles = min(candidates, key=lambda candidate: evaluate(candidate)[0])
        result = evaluate(angles)
        if best is None or result[0] < best[0]:
            best = (*result, angles.copy())
    error, tip_error, penetration, points, angles = best
    return {
        "degrees": angles.tolist(), "score": float(error),
        "tip_surface_error_mm": float(tip_error * 1000),
        "max_bone_sample_penetration_mm": float(penetration * 1000),
        "coordination_prior": "DIP <= PIP + 5 deg; PIP >= 0.35 * MCP (artistic, not anatomical)",
        "points": points,
    }
