"""Pure NumPy quality primitives shared by the Mixamo retarget pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FootContactResult:
    """Half-open support intervals and their source-space lock positions."""

    contact_intervals: dict[str, list[tuple[int, int]]]
    locked_positions: dict[str, list[np.ndarray]]


def morphology_aware_joint_targets(
    source_positions: np.ndarray,
    source_rest_positions: np.ndarray,
    target_rest_positions: np.ndarray,
    parent_indices: np.ndarray,
    *,
    direction_fit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build targets matching rest-delta trunks and direction-fit limbs."""
    source = np.asarray(source_positions, dtype=np.float64)
    source_rest = np.asarray(source_rest_positions, dtype=np.float64)
    target_rest = np.asarray(target_rest_positions, dtype=np.float64)
    parents = np.asarray(parent_indices, dtype=np.int32)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_positions must have shape (joints, 3)")
    if source_rest.shape != source.shape or target_rest.shape != source.shape:
        raise ValueError("rest positions must match source_positions")
    if parents.shape != (len(source),):
        raise ValueError("parent_indices must have shape (joints,)")
    direction_fit = (
        np.zeros(len(source), dtype=bool)
        if direction_fit_mask is None
        else np.asarray(direction_fit_mask, dtype=bool)
    )
    if direction_fit.shape != (len(source),):
        raise ValueError("direction_fit_mask must have shape (joints,)")

    result = target_rest.copy()
    for index, parent in enumerate(parents):
        if parent < 0:
            continue
        if parent >= index:
            raise ValueError("joints must be ordered with parents before children")
        source_pose_edge = source[index] - source[parent]
        source_rest_edge = source_rest[index] - source_rest[parent]
        target_rest_edge = target_rest[index] - target_rest[parent]
        if direction_fit[index]:
            source_length = float(np.linalg.norm(source_pose_edge))
            target_length = float(np.linalg.norm(target_rest_edge))
            posed_edge = (
                target_rest_edge
                if source_length <= 1e-12
                else source_pose_edge * (target_length / source_length)
            )
        else:
            posed_edge = _rotation_between(source_rest_edge, source_pose_edge) @ target_rest_edge
        result[index] = result[parent] + posed_edge
    return result


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-12 or target_norm <= 1e-12:
        return np.eye(3)
    left = source / source_norm
    right = target / target_norm
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3)
    if dot < -1.0 + 1e-10:
        axis = np.cross(left, np.array((1.0, 0.0, 0.0)))
        if float(np.linalg.norm(axis)) <= 1e-8:
            axis = np.cross(left, np.array((0.0, 1.0, 0.0)))
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(left, right)
    skew = np.array(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        )
    )
    return np.eye(3) + skew + skew @ skew / (1.0 + dot)


def detect_foot_contact_frames(
    foot_positions: np.ndarray,
    *,
    height_threshold: float = 0.08,
    velocity_threshold: float = 0.04,
    maximum_lock_drift: float = 0.005,
    min_contact_frames: int = 3,
) -> FootContactResult:
    """Find low, stationary foot intervals from ``(frames, 2, xyz)`` samples.

    A low per-frame velocity alone is insufficient: a slowly sliding foot can
    satisfy it indefinitely while accumulating tens of centimetres of drift.
    Each returned interval is therefore also bounded by its displacement from
    the lock frame, so a contact lock always has a well-defined static target.
    """
    if foot_positions.ndim != 3 or foot_positions.shape[1:] != (2, 3):
        raise ValueError("foot_positions must have shape (frames, 2, 3)")
    intervals: dict[str, list[tuple[int, int]]] = {}
    locked_positions: dict[str, list[np.ndarray]] = {}
    for foot_index, foot_name in enumerate(("left_foot", "right_foot")):
        positions = foot_positions[:, foot_index]
        velocity = np.zeros(len(positions), dtype=np.float64)
        if len(positions) > 1:
            velocity[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        mask = (positions[:, 2] <= height_threshold) & (velocity <= velocity_threshold)
        starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
        ends = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
        kept: list[tuple[int, int]] = []
        for start, end in zip(starts, ends):
            lock_start = int(start)
            for frame in range(int(start) + 1, int(end) + 1):
                exceeds_drift = (
                    frame == int(end)
                    or float(np.linalg.norm(positions[frame, :2] - positions[lock_start, :2]))
                    > maximum_lock_drift
                )
                if not exceeds_drift:
                    continue
                if frame - lock_start >= min_contact_frames:
                    kept.append((lock_start, frame))
                # A foot that has begun to slide has not established a new
                # support merely because its instantaneous velocity remains
                # low.  Require a real mask break (lift or velocity change)
                # before another lock may begin.
                break
        intervals[foot_name] = kept
        locked_positions[foot_name] = [positions[start].copy() for start, _ in kept]
    return FootContactResult(intervals, locked_positions)


def is_foot_in_contact(
    frame: int, foot_name: str, contact_result: FootContactResult
) -> tuple[bool, np.ndarray | None]:
    for index, (start, end) in enumerate(contact_result.contact_intervals.get(foot_name, [])):
        if start <= frame < end:
            return True, contact_result.locked_positions[foot_name][index]
    return False, None


def smooth_root_trajectory(transl: np.ndarray, *, window: int = 5, max_delta: float = 0.05) -> np.ndarray:
    """Smooth a root path while retaining an explicit maximum frame displacement."""
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError("transl must have shape (frames, 3)")
    if len(transl) < window or window < 2:
        return transl.copy()
    half = window // 2
    result = np.empty_like(transl)
    for axis in range(3):
        padded = np.pad(transl[:, axis], (half, half), mode="edge")
        result[:, axis] = np.convolve(padded, np.ones(window) / window, mode="valid")[: len(transl)]
    for index in range(1, len(result)):
        delta = result[index] - result[index - 1]
        magnitude = float(np.linalg.norm(delta))
        if magnitude > max_delta:
            result[index] = result[index - 1] + delta * (max_delta / magnitude)
    return result


def quaternion_continuity(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Choose the equivalent quaternion sign closest to the preceding frame."""
    if quaternions_wxyz.ndim != 2 or quaternions_wxyz.shape[1] != 4:
        raise ValueError("quaternions_wxyz must have shape (frames, 4)")
    result = quaternions_wxyz.astype(np.float64, copy=True)
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result
