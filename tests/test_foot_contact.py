"""Support proposals preserve airborne motion, foot orientation and source data."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from motionviewer.core.foot_contact import FootContactSettings, plan_foot_contacts

FLOORS = (0.10, 0.02)


def feet(frames=60):
    ankle = np.zeros((frames, 2, 3))
    ankle[:, :, 0] = (-0.10, 0.10)
    ankle[:, :, 2] = FLOORS[0]
    toe = ankle.copy()
    toe[:, :, 1] += 0.15
    toe[:, :, 2] = FLOORS[1]
    return ankle, toe


def plan(source_ankle, source_toe, target_ankle=None, target_toe=None, **kwargs):
    return plan_foot_contacts(
        source_ankle, source_toe,
        source_ankle if target_ankle is None else target_ankle,
        source_toe if target_toe is None else target_toe,
        fps=kwargs.pop("fps", 50), source_floor_heights=kwargs.pop("source_floor_heights", FLOORS),
        **kwargs,
    )


def test_small_target_support_drift_is_removed_without_editing_source_or_foot_rotation():
    source_ankle, source_toe = feet()
    target_ankle, target_toe = feet()
    drift = np.linspace(0, 0.04, len(target_ankle))
    target_ankle[..., 0] += drift[:, None]
    target_toe[..., 0] += drift[:, None]
    originals = [a.copy() for a in (source_ankle, source_toe, target_ankle, target_toe)]
    result = plan(source_ankle, source_toe, target_ankle, target_toe)

    assert result.floor_confidence == "provided"
    assert np.all(result.support_modes == "flat")
    assert np.all(result.weights == 1)
    assert np.allclose(np.diff(result.ankle_targets[..., :2], axis=0), 0)
    assert np.array_equal(result.ankle_targets[..., 2], target_ankle[..., 2])
    assert np.array_equal(result.toe_targets[..., 2], target_toe[..., 2])
    assert np.allclose(result.toe_targets - result.ankle_targets, target_toe - target_ankle)
    assert result.interior_speed_after_m_s < result.interior_speed_before_m_s * 0.01
    assert result.max_adjustment_m == pytest.approx(0.02)
    for actual, original in zip((source_ankle, source_toe, target_ankle, target_toe), originals, strict=True):
        assert np.array_equal(actual, original)
    assert not result.ankle_targets.flags.writeable
    json.dumps(result.summary())


def test_airborne_frames_keep_their_full_target_trajectory_and_height():
    ankle, toe = feet()
    ankle[20:40, :, 2] += 0.25
    toe[20:40, :, 2] += 0.25
    ankle[20:40, :, 0] += np.linspace(0, 0.4, 20)[:, None]
    toe[20:40, :, 0] += np.linspace(0, 0.4, 20)[:, None]
    target_ankle, target_toe = ankle.copy(), toe.copy()
    target_ankle[..., 0] += np.linspace(0, 0.02, 60)[:, None]
    target_toe[..., 0] += np.linspace(0, 0.02, 60)[:, None]
    result = plan(ankle, toe, target_ankle, target_toe)

    assert np.all(result.weights[20:40] == 0)
    assert np.array_equal(result.ankle_targets[20:40], target_ankle[20:40])
    assert np.array_equal(result.toe_targets[20:40], target_toe[20:40])
    assert np.array_equal(result.ankle_targets[..., 2], target_ankle[..., 2])


def test_a_known_floor_does_not_lock_a_motionless_airborne_foot():
    ankle, toe = feet()
    ankle[..., 2] += 0.5
    toe[..., 2] += 0.5
    result = plan(ankle, toe)
    assert result.floor_confidence == "unsupported"
    assert not result.intervals
    assert result.reason
    assert np.array_equal(result.ankle_targets, ankle)


def test_automatic_floor_is_reported_as_an_estimate_not_confirmed_ground():
    ankle, toe = feet()
    result = plan(ankle, toe, source_floor_heights=None)
    assert result.floor_heights == pytest.approx(FLOORS)
    assert result.floor_confidence == "estimated"
    assert "airborne" in result.reason


@pytest.mark.parametrize("fps, frames", [(50, 3), (100, 6)])
def test_short_low_speed_intervals_are_rejected_in_seconds_not_fixed_frame_count(fps, frames):
    ankle, toe = feet(frames)
    result = plan(ankle, toe, fps=fps)
    assert not result.intervals
    assert np.all(result.weights == 0)
    assert result.floor_confidence == "unsupported"


def test_fast_source_sliding_near_the_floor_is_not_mistaken_for_support():
    ankle, toe = feet()
    for points in (ankle, toe):
        points[..., 0] += np.arange(60)[:, None] * 0.02  # 1 metre/second.
    result = plan(ankle, toe)
    assert not result.intervals
    assert np.array_equal(result.ankle_targets, ankle)


def test_opposing_steps_do_not_cancel_into_a_false_low_velocity():
    ankle, toe = feet()
    for points in (ankle, toe):
        points[..., 0] += np.where(np.arange(60) % 2, 0.003, -0.003)[:, None]
    assert not plan(ankle, toe).intervals


def test_raised_heel_can_pivot_on_toe_without_flattening_the_foot():
    ankle, toe = feet()
    angles = np.linspace(0, np.pi / 2, 60)
    ankle[..., 0] = toe[..., 0] + (0.15 * np.sin(angles))[:, None]
    ankle[..., 1] = toe[..., 1] - (0.15 * np.cos(angles))[:, None]
    ankle[..., 2] = 0.22  # Both heels are raised; toes remain on their floor.
    target_ankle, target_toe = ankle.copy(), toe.copy()
    for points in (target_ankle, target_toe):
        points[..., 0] += np.linspace(0, 0.03, 60)[:, None]
    result = plan(ankle, toe, target_ankle, target_toe)

    assert np.all(result.support_modes == "toe")
    assert np.allclose(np.diff(result.toe_targets[..., :2], axis=0), 0)
    assert np.array_equal(result.ankle_targets[..., 2], target_ankle[..., 2])
    assert np.allclose(result.toe_targets - result.ankle_targets, target_toe - target_ankle)
    assert np.ptp(result.ankle_targets[:, 0, 0]) > 0.1  # The heel still turns around the toe.


def test_fast_turn_about_foot_midpoint_is_not_a_flat_double_foot_lock():
    ankle, toe = feet()
    centres = ankle[..., :2].copy()
    angles = np.arange(60) / 50 * np.pi / 2
    offset = np.stack((0.075 * np.sin(angles), 0.075 * np.cos(angles)), axis=-1)
    ankle[..., :2] = centres - offset[:, None, :]
    toe[..., :2] = centres + offset[:, None, :]
    result = plan(ankle, toe)

    assert not result.intervals
    assert np.all(result.support_modes == "none")
    assert np.array_equal(result.ankle_targets, ankle)


def test_turn_with_a_distinct_stationary_ankle_keeps_the_ankle_pivot():
    ankle, toe = feet()
    angles = np.arange(60) / 50 * np.pi / 2
    offset = np.stack((0.06 * np.sin(angles), 0.06 * np.cos(angles)), axis=-1)
    toe[..., :2] = ankle[..., :2] + offset[:, None, :]
    result = plan(ankle, toe)
    assert np.all(result.support_modes == "ankle")
    assert np.allclose(result.toe_targets - result.ankle_targets, toe - ankle)


def test_large_target_drift_is_bounded_by_the_explicit_correction_limit():
    ankle, toe = feet()
    target_ankle, target_toe = feet()
    for points in (target_ankle, target_toe):
        points[..., :2] += np.linspace(0, 0.2, 60)[:, None, None]
    settings = replace(FootContactSettings(), max_correction_m=0.012)
    result = plan(ankle, toe, target_ankle, target_toe, settings=settings)
    assert result.max_adjustment_m <= 0.012 + 1e-12
    assert result.limited_frames > 0
    assert np.linalg.norm(result.ankle_targets[-1, 0] - result.ankle_targets[0, 0]) > 0.2


def test_contact_blends_do_not_reach_into_swing_frames():
    ankle, toe = feet()
    for points in (ankle, toe):
        points[:10, 0, 2] += 0.2
        points[50:, 0, 2] += 0.2
    target_ankle, target_toe = ankle.copy(), toe.copy()
    for points in (target_ankle, target_toe):
        points[..., 0] += np.linspace(0, 0.02, 60)[:, None]
    result = plan(ankle, toe, target_ankle, target_toe)
    interval = next(i for i in result.intervals if i.side == 0)
    assert result.weights[interval.start, 0] == 0
    assert result.weights[interval.end - 1, 0] == 0
    assert np.any((result.weights[:, 0] > 0) & (result.weights[:, 0] < 1))
    assert np.array_equal(result.ankle_targets[:10, 0], target_ankle[:10, 0])
    assert np.array_equal(result.ankle_targets[50:, 0], target_ankle[50:, 0])


def test_height_hysteresis_keeps_contact_through_small_threshold_fluctuations():
    ankle, toe = feet(20)
    for points in (ankle, toe):
        points[..., 2] += np.where(np.arange(20) % 2, 0.04, 0.03)[:, None]
    result = plan(ankle, toe, fps=10)
    assert np.all(result.support_modes == "flat")


def test_heel_to_toe_support_change_does_not_release_grounding_weight():
    ankle, toe = feet(100)
    # A slow heel raise, with the toe remaining planted throughout.
    ankle[20:60, :, 2] += np.linspace(0, 0.09, 40)[:, None]
    ankle[60:, :, 2] += 0.09
    target_ankle, target_toe = ankle.copy(), toe.copy()
    for points in (target_ankle, target_toe):
        points[..., 0] += np.linspace(0, 0.02, 100)[:, None]
    result = plan(ankle, toe, target_ankle, target_toe)
    assert np.any(result.support_modes == "flat")
    assert np.any(result.support_modes == "toe")
    assert np.all(result.weights == 1)
    assert np.allclose(result.toe_targets - result.ankle_targets, target_toe - target_ankle)
    # A support-mode switch must not invent a burst of sideways motion by
    # recentering a new anchor. The drifting foot rolls in place in the source.
    original_steps = np.linalg.norm(np.diff(target_toe[..., :2], axis=0), axis=-1)
    corrected_steps = np.linalg.norm(np.diff(result.toe_targets[..., :2], axis=0), axis=-1)
    assert corrected_steps.max() <= original_steps.max() * 1.05


def test_a_new_stance_does_not_inherit_the_previous_stances_anchor():
    ankle, toe = feet(100)
    for points in (ankle, toe):
        points[30:60, :, 2] += 0.2
        points[60:, :, 0] += 0.5
    target_ankle, target_toe = ankle.copy(), toe.copy()
    for points in (target_ankle, target_toe):
        points[:30, :, 0] += np.linspace(0, 0.02, 30)[:, None]
        points[60:, :, 0] += np.linspace(0, 0.02, 40)[:, None]
    result = plan(ankle, toe, target_ankle, target_toe)
    # The second stance stabilizes around its new position, 50 cm away, rather
    # than pulling backwards toward a point belonging to the previous step.
    assert np.allclose(np.diff(result.toe_targets[70:90, :, :2], axis=0), 0)
    assert np.all(result.toe_targets[70:90, :, 0] - toe[70:90, :, 0] > 0)


def test_single_frame_has_no_supported_interval():
    ankle, toe = feet(1)
    result = plan(ankle, toe)
    assert not result.intervals
    assert np.array_equal(result.ankle_targets, ankle)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_frame_rate_is_rejected(fps):
    ankle, toe = feet()
    with pytest.raises(ValueError, match="fps"):
        plan(ankle, toe, fps=fps)


def test_invalid_coordinates_and_hysteresis_are_rejected():
    ankle, toe = feet()
    broken = ankle.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plan(broken, toe)
    with pytest.raises(ValueError, match="same shape"):
        plan(ankle, toe[:1])
    with pytest.raises(ValueError, match="exit thresholds"):
        plan(ankle, toe, settings=replace(FootContactSettings(), exit_speed_m_s=0.01))
    with pytest.raises(ValueError, match="two finite"):
        plan(ankle, toe, source_floor_heights=(0.0, float("nan")))
