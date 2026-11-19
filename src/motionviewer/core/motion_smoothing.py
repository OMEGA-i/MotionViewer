"""Optional, bounded upper-body smoothing and signed rotation measurements.

The filter operates on SO(3), not on axis-angle coordinates. It leaves the root
and every lower-body rotation unchanged, so it cannot alter the foot trajectories
of the source skeleton. A small local rotation correction can accumulate along
the spine; the report therefore measures both local and world orientation changes.
It is an editing tool, not an automatic test for whether a fast gesture is noise.
"""

from __future__ import annotations

import numpy as np

from .smplx_fk import (
    SMPLX_BODY22_NAMES,
    SMPLX_BODY22_PARENTS,
    axis_angle_from_rotations,
    global_rotations,
    rodrigues,
)

UPPER_BODY_JOINTS = (
    "spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar",
)


def repose_joints(
    global_orient: np.ndarray,
    before_pose: np.ndarray,
    after_pose: np.ndarray,
    joints22: np.ndarray,
) -> np.ndarray:
    """Carry observed joint positions with a change of local body rotations.

    Each original parent-to-child vector is rotated by the change in its parent's
    world orientation, then accumulated down the tree. This preserves the source's
    per-frame bone lengths and fitting residuals instead of replacing them with an
    estimated constant skeleton. The root stays at its original position. Branches
    whose rotations did not change are copied exactly, including both legs when
    used after ``smooth_upper_body``. No root translation is added a second time.
    """
    roots = np.asarray(global_orient)
    before = np.asarray(before_pose)
    after = np.asarray(after_pose)
    original = np.asarray(joints22)
    total = len(roots)
    if roots.shape != (total, 3) or original.shape != (total, 22, 3):
        raise ValueError("expected global_orient (T, 3) and joints22 (T, 22, 3)")
    if any(pose.shape not in ((total, 63), (total, 21, 3)) for pose in (before, after)):
        raise ValueError("body poses must have shape (T, 63) or (T, 21, 3)")
    if not all(np.isfinite(array).all() for array in (roots, before, after, original)):
        raise ValueError("motion arrays must be finite")
    dtype = original.dtype if np.issubdtype(original.dtype, np.floating) else np.float64
    result = np.array(original, dtype=dtype, copy=True)
    old_world = global_rotations(roots, before)
    new_world = global_rotations(roots, after)
    for joint, parent in enumerate(SMPLX_BODY22_PARENTS):
        if parent < 0:
            continue
        changed = np.any(new_world[:, parent] != old_world[:, parent], axis=(1, 2))
        changed |= np.any(result[:, parent] != original[:, parent], axis=-1)
        if not changed.any():
            continue
        rotation_delta = new_world[changed, parent] @ np.swapaxes(old_world[changed, parent], -1, -2)
        offset = original[changed, joint] - original[changed, parent]
        result[changed, joint] = result[changed, parent] + np.einsum("tij,tj->ti", rotation_delta, offset)
    return result


def angular_velocity_vectors(rotations: np.ndarray, fps: float) -> np.ndarray:
    """Signed angular velocity, radians/second, in a common reference frame.

    Input shape is ``(T, ..., 3, 3)``; output is ``(T-1, ..., 3)``. For local
    joint matrices the reference is that joint's parent coordinate system. Using
    ``R[t+1] @ R[t].T`` keeps successive velocity axes in that same reference,
    rather than comparing components expressed in different moving joint frames.
    Steps use the shortest rotation, so turns of >=180 degrees per frame are
    inherently ambiguous without a higher sampling rate.
    """
    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.ndim < 3 or matrices.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape (T, ..., 3, 3)")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(matrices).all():
        raise ValueError("rotations must be finite")
    relative = matrices[1:] @ np.swapaxes(matrices[:-1], -1, -2)
    return axis_angle_from_rotations(relative) * fps


def _statistics(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if not len(flat):
        return {key: 0.0 for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": float(flat.mean()),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
    }


def _motion_measurements(velocity: np.ndarray, fps: float) -> dict:
    speeds = np.degrees(np.linalg.norm(velocity, axis=-1))
    changes = np.degrees(np.linalg.norm(np.diff(velocity, axis=0), axis=-1))
    return {
        "angular_speed_deg_s": _statistics(speeds),
        "angular_acceleration_deg_s2": _statistics(changes * fps),
        # Retain the per-frame second-difference scale used by the score/page,
        # while measuring vector changes instead of unsigned speed changes.
        "jitter_deg": _statistics(changes / fps),
    }


def smooth_upper_body(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    *,
    fps: float,
    window: int = 3,
    strength: float = 1.0,
    max_deviation_deg: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return copied root, pose and translation arrays plus a JSON-safe report.

    Only ``UPPER_BODY_JOINTS`` are filtered. The root, hip/knee/ankle/foot chains,
    and shoulder/elbow/wrist local rotations retain their exact input values.
    First and last poses also remain unchanged. The output pose keeps the input
    layout, either ``(T, 63)`` or ``(T, 21, 3)``. Inputs are never mutated.

    At each frame, neighbouring orientations are mapped into its tangent space.
    A symmetric triangular average supplies a correction, scaled by ``strength``
    and capped at ``max_deviation_deg`` per joint. A constant angular-velocity
    rotation has equal/opposite neighbours and is preserved while the neighbours
    differ by less than a half turn. Real acceleration
    can also be attenuated: inspect the report and rendered A/B, especially for
    intentional sharp dance beats. This function has no default-on caller.

    Window is an odd frame count (3 is a 60 ms neighbourhood at 50 fps). Use 1,
    strength 0 or a zero cap for an exact no-op copy. Larger windows broaden the
    time interval edited; no claim is made that all high frequencies are noise.
    """
    roots = np.asarray(global_orient)
    original = np.asarray(body_pose)
    translation = np.asarray(transl)
    total = len(roots)
    if roots.shape != (total, 3) or translation.shape != (total, 3):
        raise ValueError("global_orient and transl must have shape (T, 3)")
    if original.shape not in ((total, 63), (total, 21, 3)):
        raise ValueError("body_pose must have shape (T, 63) or (T, 21, 3)")
    if not all(np.isfinite(array).all() for array in (roots, original, translation)):
        raise ValueError("motion arrays must be finite")
    if not isinstance(window, (int, np.integer)) or window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd frame count")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    if not np.isfinite(max_deviation_deg) or not 0 <= max_deviation_deg < 180:
        raise ValueError("max_deviation_deg must be finite and in [0, 180)")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")

    dtype = original.dtype if np.issubdtype(original.dtype, np.floating) else np.float64
    pose = np.array(original, dtype=dtype, copy=True).reshape(total, 21, 3)
    columns = [SMPLX_BODY22_NAMES.index(name) - 1 for name in UPPER_BODY_JOINTS]
    before = rodrigues(pose[:, columns])
    filtered = before.copy()
    cap = np.radians(max_deviation_deg)
    limited_samples = 0
    active = window > 1 and strength > 0 and cap > 0 and total > 2
    if active:
        for frame in range(1, total - 1):
            # Shrink near the ends instead of padding held frames: symmetric
            # support also preserves constant velocity at those boundaries.
            radius = min(window // 2, frame, total - frame - 1)
            offsets = np.arange(-radius, radius + 1)
            weights = radius + 1 - np.abs(offsets)
            relative = np.swapaxes(before[frame], -1, -2) @ before[frame + offsets]
            logs = axis_angle_from_rotations(relative)
            correction = np.average(logs, axis=0, weights=weights) * strength
            norms = np.linalg.norm(correction, axis=-1)
            limited_samples += int(np.count_nonzero(norms > cap))
            correction *= np.minimum(1.0, cap / np.maximum(norms, 1e-15))[:, None]
            filtered[frame] = before[frame] @ rodrigues(correction)
        # Avoid round-tripping unchanged samples: their original axis-angle
        # branch, dtype and values are part of the source record.
        delta = axis_angle_from_rotations(filtered @ np.swapaxes(before, -1, -2))
        changed = np.linalg.norm(delta, axis=-1) > 1e-12
        converted = axis_angle_from_rotations(filtered)
        for index, column in enumerate(columns):
            pose[changed[:, index], column] = converted[changed[:, index], index]

    after = rodrigues(pose[:, columns])
    velocity_before = angular_velocity_vectors(before, fps)
    velocity_after = angular_velocity_vectors(after, fps)
    deviation = np.degrees(np.linalg.norm(
        axis_angle_from_rotations(after @ np.swapaxes(before, -1, -2)), axis=-1
    ))
    world_before = global_rotations(roots, original)
    world_after = global_rotations(roots, pose)
    world_deviation = np.degrees(np.linalg.norm(
        axis_angle_from_rotations(world_after @ np.swapaxes(world_before, -1, -2)), axis=-1
    ))
    worst_joint = (
        SMPLX_BODY22_NAMES[int(np.unravel_index(np.argmax(world_deviation), world_deviation.shape)[1])]
        if world_deviation.size and world_deviation.max() > 1e-10 else ""
    )
    report = {
        "algorithm": "so3_tangent_triangular_v1",
        "enabled": bool(active),
        "frames": total,
        "fps": float(fps),
        "window_frames": int(window),
        "strength": float(strength),
        "max_local_deviation_deg": float(max_deviation_deg),
        "joints": list(UPPER_BODY_JOINTS),
        "root_pose_unchanged": True,
        "root_translation_unchanged": True,
        "lower_body_rotations_unchanged": True,
        "before": _motion_measurements(velocity_before, fps),
        "after": _motion_measurements(velocity_after, fps),
        "local_deviation_deg": _statistics(deviation),
        "world_deviation_deg": _statistics(world_deviation),
        "worst_world_deviation_joint": worst_joint,
        "angular_velocity_change_deg_s": _statistics(np.degrees(np.linalg.norm(
            velocity_after - velocity_before, axis=-1
        ))),
        "limited_joint_frames": limited_samples,
        "modified_frames": int(np.count_nonzero(np.any(deviation > 1e-8, axis=1))),
    }
    return roots.copy(), pose.reshape(original.shape), translation.copy(), report
