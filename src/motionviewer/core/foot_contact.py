"""Conservative support detection and small foot-trajectory corrections.

All inputs are world-space metres with Z up, shaped ``(frames, 2, 3)`` with
left then right foot. Source ankles/toes determine contact; target ankles/toes
determine the proposed correction. Neither the source arrays nor the root path
are edited. This is a contact proposal for a downstream leg IK pass, not IK.

The targets translate an ankle and its toe equally in XY. Their Z coordinates
and relative vector remain unchanged, preserving foot rotation, toe raises and
toe pivots. Shoe-sole grounding and reachability belong to the caller, which has
the target geometry. A joint centre must never be mistaken for the shoe sole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FootContactSettings:
    """Conservative thresholds; distances are metres and durations seconds."""

    enter_height_m: float = 0.035
    exit_height_m: float = 0.055
    enter_speed_m_s: float = 0.18
    exit_speed_m_s: float = 0.32
    enter_vertical_speed_m_s: float = 0.12
    exit_vertical_speed_m_s: float = 0.25
    min_contact_s: float = 0.10
    blend_s: float = 0.08
    max_correction_m: float = 0.03
    max_flat_yaw_speed_deg_s: float = 45.0
    pivot_speed_m_s: float = 0.08
    pivot_speed_advantage_m_s: float = 0.04
    floor_percentile: float = 5.0


@dataclass(frozen=True)
class FootContactInterval:
    """One unchanged support mode; ``end`` is exclusive, side 0 is left."""

    side: int
    start: int
    end: int
    mode: str  # flat, toe, ankle
    anchor_xy: tuple[float, float]


@dataclass(frozen=True)
class FootContactPlan:
    """Arrays are read-only. Targets already include the contact-weight blend.

    ``weights`` describe temporal confidence; do not multiply them into the XY
    target correction a second time. They can separately weight sole-grounding.
    ``floor_heights`` are source ankle/toe *joint-centre* contact heights.
    ``floor_confidence`` is a label, not a calibrated probability: ``provided``
    means externally supplied, ``estimated`` means inferred, and ``unsupported``
    means no sustained contact was found. Even a motionless, airborne clip can
    look like standing without an externally known floor; inferred ground is
    never evidence that physical ground exists there.
    """

    ankle_targets: np.ndarray
    toe_targets: np.ndarray
    weights: np.ndarray
    support_modes: np.ndarray
    intervals: tuple[FootContactInterval, ...]
    floor_heights: tuple[float, float]
    floor_confidence: str
    reason: str
    limited_frames: int
    max_adjustment_m: float
    interior_speed_before_m_s: float | None
    interior_speed_after_m_s: float | None

    def summary(self) -> dict:
        """Compact, JSON-compatible provenance for the optional correction."""
        return {
            "floor_confidence": self.floor_confidence,
            "source_joint_floor_heights": list(self.floor_heights),
            "reason": self.reason,
            "contact_frames": np.count_nonzero(self.support_modes != "none", axis=0).tolist(),
            "intervals": len(self.intervals),
            "limited_frames": self.limited_frames,
            "max_adjustment_m": self.max_adjustment_m,
            "interior_support_speed_before_m_s": self.interior_speed_before_m_s,
            "interior_support_speed_after_m_s": self.interior_speed_after_m_s,
        }


def _points(value: np.ndarray, name: str, shape: tuple | None = None) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (2, 3) or not len(points):
        raise ValueError(f"{name} must have shape (nonempty frames, 2, 3)")
    if shape is not None and points.shape != shape:
        raise ValueError("all foot arrays must have the same shape")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return points


def _validate_settings(settings: FootContactSettings, fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    for name, value in vars(settings).items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    for enter, leave in (
        (settings.enter_height_m, settings.exit_height_m),
        (settings.enter_speed_m_s, settings.exit_speed_m_s),
        (settings.enter_vertical_speed_m_s, settings.exit_vertical_speed_m_s),
    ):
        if leave < enter:
            raise ValueError("exit thresholds must be at least their enter thresholds")
    if settings.floor_percentile > 50:
        raise ValueError("floor_percentile must lie between 0 and 50")
    if settings.min_contact_s <= 0:
        raise ValueError("min_contact_s must be positive")


def _adjacent_max(values: np.ndarray) -> np.ndarray:
    """Put edge magnitudes on both endpoints; opposing steps cannot cancel."""
    if not len(values):
        return np.zeros((1, *values.shape[1:]), dtype=np.float64)
    return np.maximum(
        np.concatenate((values[:1], values), axis=0),
        np.concatenate((values, values[-1:]), axis=0),
    )


def _speeds(points: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    changes = np.diff(points, axis=0) * fps
    return _adjacent_max(np.linalg.norm(changes[..., :2], axis=-1)), _adjacent_max(abs(changes[..., 2]))


def _support_mask(points: np.ndarray, floor: float, fps: float, settings: FootContactSettings) -> np.ndarray:
    horizontal, vertical = _speeds(points, fps)
    height = abs(points[..., 2] - floor)
    enter = (
        (height <= settings.enter_height_m)
        & (horizontal <= settings.enter_speed_m_s)
        & (vertical <= settings.enter_vertical_speed_m_s)
    )
    stay = (
        (height <= settings.exit_height_m)
        & (horizontal <= settings.exit_speed_m_s)
        & (vertical <= settings.exit_vertical_speed_m_s)
    )
    result = np.zeros(enter.shape, dtype=bool)
    active = np.zeros(2, dtype=bool)
    for frame in range(len(points)):
        active = np.where(active, stay[frame], enter[frame])
        result[frame] = active
    return result


def _yaw_speed(ankles: np.ndarray, toes: np.ndarray, fps: float) -> np.ndarray:
    direction = toes[..., :2] - ankles[..., :2]
    lengths = np.linalg.norm(direction, axis=-1)
    a, b = direction[:-1], direction[1:]
    angles = np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(axis=-1))
    speed = _adjacent_max(abs(np.degrees(angles)) * fps)
    # An almost vertical/degenerate foot has no reliable planar heading.
    speed[lengths < 0.025] = np.inf
    return speed


def _runs(modes: np.ndarray):
    start = 0
    for end in range(1, len(modes) + 1):
        if end == len(modes) or modes[end] != modes[start]:
            if modes[start] != "none":
                yield start, end, str(modes[start])
            start = end


def _blend(start: int, end: int, total: int, frames: int) -> np.ndarray:
    weights = np.ones(end - start)
    if frames > 0:
        ramp = np.arange(end - start, dtype=np.float64) / frames
        if start > 0:
            weights = np.minimum(weights, ramp)
        if end < total:
            weights = np.minimum(weights, ramp[::-1])
    weights = np.clip(weights, 0, 1)
    return weights * weights * (3.0 - 2.0 * weights)


def _support_points(ankles: np.ndarray, toes: np.ndarray, mode: str) -> np.ndarray:
    if mode == "toe":
        return toes
    if mode == "ankle":
        return ankles
    return (ankles + toes) * 0.5


def _smooth_support_changes(offsets: np.ndarray, modes: np.ndarray, radius: int) -> np.ndarray:
    """Blend XY corrections at a heel/toe change without dropping contact weight."""
    result = offsets.copy()
    changes = np.flatnonzero(modes[1:] != modes[:-1]) + 1
    if not len(changes) or radius <= 0:
        return result
    padded = np.pad(offsets, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    smoothed = np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)], axis=-1)
    for change in changes:
        start, end = max(0, change - radius), min(len(result), change + radius)
        result[start:end] = smoothed[start:end]
    return result


def plan_foot_contacts(
    source_ankles: np.ndarray,
    source_toes: np.ndarray,
    target_ankles: np.ndarray,
    target_toes: np.ndarray,
    *,
    fps: float,
    source_floor_heights: tuple[float, float] | None = None,
    settings: FootContactSettings | None = None,
) -> FootContactPlan:
    """Propose bounded XY support anchoring without changing Z or foot rotation.

    Contact needs low marker height AND low horizontal/vertical speed, with
    separate entry/release thresholds and a minimum continuous duration. Rapid
    yaw cannot produce a flat-foot lock: a distinctly slower marker may act as
    a pivot, otherwise the foot is left alone. No contact fills are added across
    jumps, foot lifts or short ambiguous intervals.

    A contact episode starts at the median target support position. When its
    heel/toe support mode changes, the new pivot inherits the previous frame's
    correction instead of snapping toward a newly centred anchor. Corrections
    stay capped at ``max_correction_m`` and ease at episode boundaries. The
    caller must still limit IK reach and compare the resulting rendered motion.
    No estimate from joints alone establishes the target shoe's contact height.
    """
    settings = settings or FootContactSettings()
    _validate_settings(settings, fps)
    source_ankles = _points(source_ankles, "source_ankles")
    shape = source_ankles.shape
    source_toes = _points(source_toes, "source_toes", shape)
    target_ankles = _points(target_ankles, "target_ankles", shape)
    target_toes = _points(target_toes, "target_toes", shape)
    total = len(source_ankles)
    estimated = source_floor_heights is None
    if estimated:
        floors = tuple(
            float(np.percentile(points[..., 2], settings.floor_percentile))
            for points in (source_ankles, source_toes)
        )
    else:
        heights = np.asarray(source_floor_heights, dtype=np.float64)
        if heights.shape != (2,) or not np.isfinite(heights).all():
            raise ValueError("source_floor_heights must be two finite ankle/toe joint heights")
        floors = tuple(float(value) for value in heights)

    ankle_support = _support_mask(source_ankles, floors[0], fps, settings)
    toe_support = _support_mask(source_toes, floors[1], fps, settings)
    modes = np.full((total, 2), "none", dtype="U5")
    modes[ankle_support & ~toe_support] = "ankle"
    modes[toe_support & ~ankle_support] = "toe"
    both = ankle_support & toe_support
    slow_yaw = _yaw_speed(source_ankles, source_toes, fps) <= settings.max_flat_yaw_speed_deg_s
    modes[both & slow_yaw] = "flat"
    ankle_speed, _ = _speeds(source_ankles, fps)
    toe_speed, _ = _speeds(source_toes, fps)
    for mode, pivot_speed, other_speed in (
        ("ankle", ankle_speed, toe_speed), ("toe", toe_speed, ankle_speed),
    ):
        pivot = (pivot_speed <= settings.pivot_speed_m_s) & (
            other_speed - pivot_speed >= settings.pivot_speed_advantage_m_s
        )
        modes[both & ~slow_yaw & pivot] = mode

    ankle_targets = target_ankles.copy()
    toe_targets = target_toes.copy()
    weights = np.zeros((total, 2))
    offsets = np.zeros((total, 2, 2))
    intervals = []
    limited_frames = 0
    before_speeds, after_speeds = [], []
    minimum = max(2, int(math.ceil(settings.min_contact_s * fps)))
    blend_frames = int(math.ceil(settings.blend_s * fps))
    for side in range(2):
        # Foot rolls can change their pivot while maintaining continuous support.
        # Duration and boundary weights apply to the full contact episode, not
        # each pivot mode; dropping Z-grounding weight at every toe roll causes hops.
        episodes = tuple(_runs(np.where(modes[:, side] != "none", "contact", "none")))
        for start, end, _ in episodes:
            if end - start < minimum:
                modes[start:end, side] = "none"
            else:
                weights[start:end, side] = _blend(start, end, total, blend_frames)
        previous_end = None
        for start, end, mode in tuple(_runs(modes[:, side])):
            support = _support_points(target_ankles[start:end, side], target_toes[start:end, side], mode)
            if previous_end == start:
                # A foot rolling onto its toe is still in the same stance.
                # Recentring every mode's anchor added 30-40 cm/s of slide at
                # otherwise planted mode boundaries. Transfer the existing
                # displacement to the new material pivot instead.
                anchor = support[0, :2] + offsets[start - 1, side]
            else:
                anchor = np.median(support[:, :2], axis=0)
            delta = anchor - support[:, :2]
            magnitude = np.linalg.norm(delta, axis=-1)
            limited_frames += int(np.count_nonzero(magnitude > settings.max_correction_m))
            delta *= np.minimum(1.0, settings.max_correction_m / np.maximum(magnitude, 1e-12))[:, None]
            offsets[start:end, side] = delta
            intervals.append(FootContactInterval(side, start, end, mode, tuple(float(v) for v in anchor)))
            previous_end = end
        for start, end, _ in episodes:
            if end - start >= minimum:
                offsets[start:end, side] = _smooth_support_changes(
                    offsets[start:end, side], modes[start:end, side], int(math.ceil(blend_frames / 2)),
                )
    offsets *= weights[..., None]
    ankle_targets[..., :2] += offsets
    toe_targets[..., :2] += offsets
    # Report speeds only inside fully weighted support, never claim that
    # blending boundaries or the full motion have been frozen.
    for interval in intervals:
        start, end, side = interval.start, interval.end, interval.side
        weight = weights[start:end, side]
        interior = (weight[:-1] >= 1.0) & (weight[1:] >= 1.0)
        if interior.any():
            support = _support_points(target_ankles[start:end, side], target_toes[start:end, side], interval.mode)
            before_speeds.extend(np.linalg.norm(np.diff(support[:, :2], axis=0), axis=-1)[interior] * fps)
            after = support[:, :2] + offsets[start:end, side]
            after_speeds.extend(np.linalg.norm(np.diff(after, axis=0), axis=-1)[interior] * fps)

    confidence = "estimated" if estimated else "provided"
    reason = (
        "Floor inferred from low source joint positions; a wholly airborne clip needs an external floor."
        if estimated else "Source joint contact heights supplied by caller."
    )
    if not intervals:
        confidence = "unsupported"
        reason = "No sustained low, slow, unambiguous source-foot support was found; targets are unchanged."
    adjustment = np.linalg.norm(ankle_targets - target_ankles, axis=-1)
    for array in (ankle_targets, toe_targets, weights, modes):
        array.setflags(write=False)
    return FootContactPlan(
        ankle_targets=ankle_targets,
        toe_targets=toe_targets,
        weights=weights,
        support_modes=modes,
        intervals=tuple(intervals),
        floor_heights=floors,
        floor_confidence=confidence,
        reason=reason,
        limited_frames=limited_frames,
        max_adjustment_m=float(adjustment.max()),
        interior_speed_before_m_s=float(np.mean(before_speeds)) if before_speeds else None,
        interior_speed_after_m_s=float(np.mean(after_speeds)) if after_speeds else None,
    )
