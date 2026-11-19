"""Small, bounded character-arm adaptations; source motion stays unchanged."""

from __future__ import annotations

import math

import numpy as np


def outward_rotation(shoulder_to_wrist: np.ndarray, outward: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate toward a lateral direction, fading as the arm extends sideways.

    The result is a rigid rotation no larger than ``degrees``. A caller applies
    it about the shoulder, and can compensate the wrist to retain hand facing.
    """
    if not math.isfinite(degrees) or not 0.0 <= degrees <= 20.0:
        raise ValueError("arm clearance must be finite and between 0 and 20 degrees")
    vector = np.asarray(shoulder_to_wrist, dtype=float)
    direction = np.asarray(outward, dtype=float)
    if vector.shape != (3,) or direction.shape != (3,):
        raise ValueError("arm clearance directions must have shape (3,)")
    if not np.isfinite(vector).all() or not np.isfinite(direction).all():
        raise ValueError("arm clearance directions must be finite")
    v_length, d_length = np.linalg.norm(vector), np.linalg.norm(direction)
    if min(v_length, d_length) < 1e-8:
        return np.eye(3)
    vector, direction = vector / v_length, direction / d_length
    cross = np.cross(vector, direction)
    strength = float(np.linalg.norm(cross))
    if strength < 1e-8:
        return np.eye(3)
    axis = cross / strength
    angle = math.radians(degrees) * strength
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
