"""Swing-twist split for MMD 捩 bones that sit on the FK chain.

Genshin/MMD arms are ``肩 -> 腕 -> 腕捩 -> ひじ -> 手捩 -> 手首``.  Putting the
full SMPL-X shoulder rotation on ``腕`` candy-wraps the mesh because the
deform weights live on the twist bone.  Extracting twist onto ``腕捩`` keeps
swing on ``腕`` while the elbow still inherits the twist through the chain.

Invariant used when writing Blender channels: if ``Q_arm = swing * twist``
and ``腕捩`` is the mapped parent of ``ひじ``, then the solver local for
``ひじ`` (parented to ``腕`` in the collapsed map) equals the Blender local
for ``ひじ`` (parented to ``腕捩``) after this split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwistPair:
    """One swing bone plus the axis-limited twist bone that follows it."""

    swing_bone: str
    twist_bone: str
    axis_local: tuple[float, float, float]


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return vector
    return vector / length


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = (float(value) for value in left)
    w2, x2, y2, z2 = (float(value) for value in right)
    return np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array(
        (float(quaternion[0]), -float(quaternion[1]), -float(quaternion[2]), -float(quaternion[3])),
        dtype=np.float64,
    )


def quaternion_normalize(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    return values / max(float(np.linalg.norm(values)), 1e-12)


def swing_twist_decompose(
    quaternion_wxyz: np.ndarray,
    twist_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``q = swing * twist`` with ``twist`` around ``twist_axis``.

    ``twist_axis`` is in the swing bone's local pose space.  MMD 腕捩 axes
    are stored as world-ish bone directions; callers should convert them
    into this local space before calling.
    """
    quaternion = quaternion_normalize(quaternion_wxyz)
    axis = _normalize(np.asarray(twist_axis, dtype=np.float64).reshape(3))
    imag = quaternion[1:]
    projection = axis * float(np.dot(imag, axis))
    twist = quaternion_normalize(np.array((float(quaternion[0]), *projection), dtype=np.float64))
    if float(np.dot(axis, twist[1:])) < 0.0:
        twist = -twist
    swing = quaternion_normalize(quaternion_multiply(quaternion, quaternion_conjugate(twist)))
    return swing, twist


def axis_angle_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(np.asarray(axis, dtype=np.float64).reshape(3))
    half = 0.5 * float(angle)
    sine = float(np.sin(half))
    return np.array((float(np.cos(half)), *(axis * sine)), dtype=np.float64)


def quaternion_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion_normalize(quaternion_wxyz))
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to ``(w, x, y, z)`` with a positive scalar.

    Uses the largest-diagonal branch so no square root loses precision near a
    180 degree rotation, which arm twist reaches on wide arm swings.
    """
    values = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(values[0, 0] + values[1, 1] + values[2, 2])
    if trace > 0.0:
        scale = float(np.sqrt(trace + 1.0)) * 2.0
        quaternion = np.array(
            (
                0.25 * scale,
                (values[2, 1] - values[1, 2]) / scale,
                (values[0, 2] - values[2, 0]) / scale,
                (values[1, 0] - values[0, 1]) / scale,
            ),
            dtype=np.float64,
        )
    elif values[0, 0] > values[1, 1] and values[0, 0] > values[2, 2]:
        scale = float(np.sqrt(1.0 + values[0, 0] - values[1, 1] - values[2, 2])) * 2.0
        quaternion = np.array(
            (
                (values[2, 1] - values[1, 2]) / scale,
                0.25 * scale,
                (values[0, 1] + values[1, 0]) / scale,
                (values[0, 2] + values[2, 0]) / scale,
            ),
            dtype=np.float64,
        )
    elif values[1, 1] > values[2, 2]:
        scale = float(np.sqrt(1.0 + values[1, 1] - values[0, 0] - values[2, 2])) * 2.0
        quaternion = np.array(
            (
                (values[0, 2] - values[2, 0]) / scale,
                (values[0, 1] + values[1, 0]) / scale,
                0.25 * scale,
                (values[1, 2] + values[2, 1]) / scale,
            ),
            dtype=np.float64,
        )
    else:
        scale = float(np.sqrt(1.0 + values[2, 2] - values[0, 0] - values[1, 1])) * 2.0
        quaternion = np.array(
            (
                (values[1, 0] - values[0, 1]) / scale,
                (values[0, 2] + values[2, 0]) / scale,
                (values[1, 2] + values[2, 1]) / scale,
                0.25 * scale,
            ),
            dtype=np.float64,
        )
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion_normalize(quaternion)
