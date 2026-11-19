"""Analytic two-segment positional IK with the original bend plane retained."""

from __future__ import annotations

import numpy as np


def support_height_adjustment(
    sole_z: np.ndarray, weights: np.ndarray, *, floor_z: float, limit_m: float,
) -> np.ndarray:
    """Lower only supported feet; lift penetrating feet with a bounded displacement."""
    delta = np.clip(floor_z - np.asarray(sole_z), -limit_m, limit_m)
    return np.where(delta >= 0., delta, delta * np.asarray(weights))


def limit_knee_change(
    hips: np.ndarray, knees: np.ndarray, ankles: np.ndarray, goals: np.ndarray,
    *, max_degrees: float = 8., max_step_degrees: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound the added bend and its per-frame change without stretching the leg.

    Inputs are (T, feet, 3). The returned goal can yield a small grounding residual;
    avoiding a knee snap takes priority over a misleading zero-error contact metric.
    """
    upper = np.linalg.norm(knees - hips, axis=-1)
    lower = np.linalg.norm(ankles - knees, axis=-1)
    def angle(distance: np.ndarray) -> np.ndarray:
        cosine = (upper ** 2 + lower ** 2 - distance ** 2) / (2 * upper * lower)
        return np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    original = angle(np.linalg.norm(ankles - hips, axis=-1))
    delta = np.clip(angle(np.linalg.norm(goals - hips, axis=-1)) - original, -max_degrees, max_degrees)
    # Two directions avoid an arbitrary delay at either end of a contact interval.
    for _ in range(2):
        for frame in range(1, len(delta)):
            delta[frame] = np.clip(delta[frame], delta[frame - 1] - max_step_degrees,
                                   delta[frame - 1] + max_step_degrees)
        for frame in range(len(delta) - 2, -1, -1):
            delta[frame] = np.clip(delta[frame], delta[frame + 1] - max_step_degrees,
                                   delta[frame + 1] + max_step_degrees)
    chosen = np.radians(np.clip(original + delta, 0.01, 179.99))
    distance = np.sqrt(np.maximum(0., upper ** 2 + lower ** 2 - 2 * upper * lower * np.cos(chosen)))
    direction = goals - hips
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    return hips + direction * distance[..., None], delta


def solve_two_bone_positions(
    hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, goal: np.ndarray,
    *, pole_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return knee, reachable ankle and reach error; never stretch either segment."""
    hip, knee, ankle, goal = (np.asarray(v, dtype=float) for v in (hip, knee, ankle, goal))
    upper, lower = float(np.linalg.norm(knee - hip)), float(np.linalg.norm(ankle - knee))
    if min(upper, lower) <= 1e-8:
        raise ValueError("two-bone IK requires nonzero limb lengths")
    direction = goal - hip
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        direction = ankle - hip
        length = float(np.linalg.norm(direction))
    direction /= max(length, 1e-8)
    distance = float(np.clip(length, abs(upper - lower) + 1e-6, upper + lower - 1e-6))
    reached = hip + direction * distance
    pole = knee - hip
    pole -= direction * np.dot(pole, direction)
    if np.linalg.norm(pole) < 1e-5 and pole_hint is not None:
        pole = np.asarray(pole_hint, dtype=float).copy()
        pole -= direction * np.dot(pole, direction)
    if np.linalg.norm(pole) < 1e-8:
        basis = np.eye(3)[np.argmin(np.abs(direction))]
        pole = basis - direction * np.dot(basis, direction)
    pole /= np.linalg.norm(pole)
    along = (upper * upper - lower * lower + distance * distance) / (2 * distance)
    height = np.sqrt(max(0., upper * upper - along * along))
    return hip + direction * along + pole * height, reached, float(np.linalg.norm(reached - goal))
