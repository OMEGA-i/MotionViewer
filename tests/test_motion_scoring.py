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

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from score_motion_quality import score_clip  # noqa: E402

from motionviewer.core.smplx_fk import SMPLX_BODY22_NAMES  # noqa: E402

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
