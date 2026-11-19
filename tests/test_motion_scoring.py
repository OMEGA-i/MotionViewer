"""Motion-quality metrics: the two that were silently wrong.

Both faults here were found by measuring real T2M exports rather than by reading
the code, so each test pins the specific failure mode:

- ``ground_penetration_m`` was subtracted the wrong way round. ``floor`` is the
  minimum over *every* frame, so ``floor - frame0`` can never be positive and the
  term was dead — the penalty it fed never fired once.
- ``foot_skate_m_s`` took the maximum over contact frames. Contact is decided by
  height alone, so the lift-off frame sits at the threshold while already moving
  fast; a clean walk measured 5.42 m/s there against 0.31 m/s across its other 28
  contact frames.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import score_motion_quality  # noqa: E402
from score_motion_quality import score_clip  # noqa: E402

from motionviewer.core.smplx_fk import SMPLX_BODY22_NAMES, axis_angle_from_rotations, rodrigues  # noqa: E402

INDEX = {name: position for position, name in enumerate(SMPLX_BODY22_NAMES)}
FPS = 30.0


def _clip(frames: int, *, height: float = 1.0) -> dict:
    """A still figure in source space, where Y is up and metres are metres."""
    joints = np.zeros((frames, 22, 3), dtype=np.float64)
    joints[:, :, 1] = height
    for foot in ("left_foot", "right_foot"):
        joints[:, INDEX[foot], 1] = 0.0
    return {
        "global_orient": np.zeros((frames, 3), dtype=np.float64),
        "body_pose": np.zeros((frames, 63), dtype=np.float64),
        "joints22": joints,
    }


# ---------------------------------------------------------------------------
# floor penetration
# ---------------------------------------------------------------------------


def test_a_clip_that_never_dips_has_no_penetration():
    assert score_clip(_clip(40), fps=FPS)["ground_penetration_m"] == pytest.approx(0.0)


def test_dipping_below_the_starting_level_is_reported():
    """The regression: this returned 0.0 for every clip ever measured."""
    payload = _clip(40)
    payload["joints22"][20:25, :, 1] -= 0.10
    penetration = score_clip(payload, fps=FPS)["ground_penetration_m"]
    assert penetration == pytest.approx(0.10, abs=1e-6)


def test_penetration_is_relative_to_frame_zero_not_to_absolute_zero():
    """A clip standing a constant centimetre low is absorbed by the static ground
    offset and must not be penalised."""
    payload = _clip(40)
    payload["joints22"][:, :, 1] -= 0.05  # the whole clip, frame 0 included
    assert score_clip(payload, fps=FPS)["ground_penetration_m"] == pytest.approx(0.0)


def test_the_penalty_ignores_a_dip_too_small_to_see():
    """1.4 cm is the measured median across the T2M sets; it must cost nothing."""
    payload = _clip(40)
    payload["joints22"][20, :, 1] -= 0.014
    assert score_clip(payload, fps=FPS)["penalty"] == pytest.approx(0.0, abs=1e-9)


def test_a_ten_centimetre_dip_costs_about_one_penalty_unit():
    payload = _clip(40)
    payload["joints22"][20, :, 1] -= 0.10
    assert score_clip(payload, fps=FPS)["penalty"] == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# foot skate
# ---------------------------------------------------------------------------


def test_a_planted_foot_that_never_moves_does_not_skate():
    assert score_clip(_clip(40), fps=FPS)["foot_skate_m_s"] == pytest.approx(0.0)


def test_steady_sliding_is_measured_at_its_real_speed():
    payload = _clip(40)
    # 0.01 m per frame at 30 fps is 0.3 m/s of sustained slide.
    payload["joints22"][:, INDEX["left_foot"], 0] = np.arange(40) * 0.01
    assert score_clip(payload, fps=FPS)["foot_skate_m_s"] == pytest.approx(0.30, abs=0.02)


def test_one_fast_frame_at_the_contact_boundary_does_not_dominate():
    """The regression: a max over contact frames turned this into 6 m/s."""
    payload = _clip(40)
    track = np.arange(40, dtype=np.float64) * 0.01
    track[20:] += 0.20  # a single 0.2 m jump, as a lift-off frame produces
    payload["joints22"][:, INDEX["left_foot"], 0] = track
    skate = score_clip(payload, fps=FPS)["foot_skate_m_s"]
    assert skate < 1.0
    assert skate == pytest.approx(0.30, abs=0.05)


def test_sustained_sliding_still_registers_over_the_threshold():
    """Robustness must not become blindness: real skating has to cost something."""
    payload = _clip(40)
    payload["joints22"][:, INDEX["left_foot"], 0] = np.arange(40) * 0.05  # 1.5 m/s
    score = score_clip(payload, fps=FPS)
    assert score["foot_skate_m_s"] == pytest.approx(1.50, abs=0.05)
    assert score["penalty"] > 3.0


def test_either_foot_can_be_the_one_skating():
    left = _clip(40)
    left["joints22"][:, INDEX["left_foot"], 0] = np.arange(40) * 0.02
    right = _clip(40)
    right["joints22"][:, INDEX["right_foot"], 0] = np.arange(40) * 0.02
    assert score_clip(left, fps=FPS)["foot_skate_m_s"] == pytest.approx(
        score_clip(right, fps=FPS)["foot_skate_m_s"], abs=1e-9
    )


@pytest.mark.parametrize("joint", ["spine1", "head"])
def test_equal_speed_alternating_rotation_is_not_mistaken_for_no_jitter(joint):
    payload = _clip(40)
    pose = payload["body_pose"].reshape(40, 21, 3)
    pose[:, INDEX[joint] - 1, 0] = np.radians(np.tile([1.0, -1.0], 20))
    score = score_clip(payload, fps=FPS)
    # Steps alternate -2 and +2 degrees; the signed difference is 4, not 0.
    assert score["jitter_deg"] == pytest.approx(4.0)
    assert score["jitter_joint"] == joint
    assert score["max_angular_velocity_deg_s"] == pytest.approx(60.0)
    assert score["jitter_acceleration_deg_s2"] == pytest.approx(3600.0)
    assert score["metric_version"] == 2


def test_changes_of_rotation_axis_are_visible_even_at_constant_speed():
    payload = _clip(30)
    matrices = [np.eye(3)]
    for index in range(29):
        step = np.zeros(3)
        step[index % 2] = np.radians(2.0)
        matrices.append(rodrigues(step) @ matrices[-1])
    pose = payload["body_pose"].reshape(30, 21, 3)
    pose[:, INDEX["neck"] - 1] = axis_angle_from_rotations(np.array(matrices))
    score = score_clip(payload)
    assert score["jitter_deg"] == pytest.approx(np.sqrt(8.0), abs=1e-10)
    assert score["jitter_joint"] == "neck"


def test_constant_angular_velocity_crossing_the_pi_branch_has_no_jitter():
    payload = _clip(40)
    pose = payload["body_pose"].reshape(40, 21, 3)
    pose[:, INDEX["spine2"] - 1, 2] = np.radians(np.arange(40) * 2.0 + 170)
    # Store shortest axis-angle coordinates, including the +pi -> -pi wrap.
    pose[:] = axis_angle_from_rotations(rodrigues(pose))
    score = score_clip(payload)
    assert score["jitter_deg"] < 1e-10
    assert score["max_angular_velocity_deg_s"] == pytest.approx(60.0)


def test_per_second_measurements_follow_the_actual_frame_rate():
    payload = _clip(40)
    pose = payload["body_pose"].reshape(40, 21, 3)
    pose[:, INDEX["spine1"] - 1, 0] = np.radians(np.tile([1.0, -1.0], 20))
    slow = score_clip(payload, fps=30)
    fast = score_clip(payload, fps=60)
    assert fast["jitter_deg"] == pytest.approx(slow["jitter_deg"])
    assert fast["jitter_acceleration_deg_s2"] == pytest.approx(4 * slow["jitter_acceleration_deg_s2"])
    assert fast["max_angular_velocity_deg_s"] == pytest.approx(2 * slow["max_angular_velocity_deg_s"])


def test_scoring_command_reads_fps_from_the_motion_file(tmp_path, monkeypatch):
    motion_dir = tmp_path / "clips/test_rec_1/gt"
    motion_dir.mkdir(parents=True)
    payload = _clip(40)
    payload["global_orient"][:, 2] = np.radians(np.arange(40) * 2.0)
    np.savez(motion_dir / "smplx_params.npz", **payload, fps=50.0)
    output = tmp_path / "scores.json"
    monkeypatch.setattr(sys, "argv", [
        "score_motion_quality.py", "--clips", str(tmp_path / "clips"),
        "--output", str(output), "--min-frames", "1",
    ])
    score_motion_quality.main()
    measured = json.loads(output.read_text())["clips"][0]
    assert measured["fps"] == 50.0
    assert measured["max_angular_velocity_deg_s"] == pytest.approx(100.0)
