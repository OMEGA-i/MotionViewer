from __future__ import annotations

import math

import numpy as np
import pytest

from motionviewer.blender.retarget.export import compare_roundtrip_samples


def _z_quaternion(angle_degrees: float) -> np.ndarray:
    half = math.radians(angle_degrees) * 0.5
    return np.array((math.cos(half), 0.0, 0.0, math.sin(half)), dtype=np.float64)


def test_roundtrip_compares_absolute_positions_not_only_frame_steps() -> None:
    reference_positions = {"Hips": np.zeros((3, 3)), "LeftFoot": np.zeros((3, 3))}
    actual_positions = {name: values.copy() for name, values in reference_positions.items()}
    actual_positions["LeftFoot"][:, 2] = 0.01
    rotations = {"LeftFoot": np.tile(_z_quaternion(0.0), (3, 1))}

    result = compare_roundtrip_samples(
        reference_positions,
        actual_positions,
        rotations,
        rotations,
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        foot_bone_names=("LeftFoot",),
    )

    assert result["passed"] is False
    assert result["maximum_absolute_position_error_m"] == pytest.approx(0.01)
    assert result["maximum_foot_position_error_m"]["LeftFoot"] == pytest.approx(0.01)


def test_roundtrip_quaternion_error_is_sign_invariant_but_detects_foot_twist() -> None:
    positions = {"LeftFoot": np.zeros((2, 3))}
    reference = np.tile(_z_quaternion(0.0), (2, 1))
    sign_flipped = -reference
    result = compare_roundtrip_samples(
        positions,
        positions,
        {"LeftFoot": reference},
        {"LeftFoot": sign_flipped},
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        foot_bone_names=("LeftFoot",),
    )
    assert result["passed"] is True
    assert result["maximum_foot_quaternion_error_degrees"] == pytest.approx(0.0)

    twisted = np.tile(_z_quaternion(20.0), (2, 1))
    result = compare_roundtrip_samples(
        positions,
        positions,
        {"LeftFoot": reference},
        {"LeftFoot": twisted},
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        foot_bone_names=("LeftFoot",),
    )
    assert result["passed"] is False
    assert result["maximum_foot_quaternion_error_degrees"] == pytest.approx(20.0)


def test_roundtrip_rejects_missing_or_mismatched_bones() -> None:
    positions = {"LeftFoot": np.zeros((2, 3))}
    rotations = {"LeftFoot": np.tile(_z_quaternion(0.0), (2, 1))}
    with pytest.raises(ValueError, match="missing position bone"):
        compare_roundtrip_samples(
            positions,
            {},
            rotations,
            rotations,
            np.zeros((2, 3)),
            np.zeros((2, 3)),
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_roundtrip_samples(
            positions,
            {"LeftFoot": np.zeros((1, 3))},
            rotations,
            rotations,
            np.zeros((2, 3)),
            np.zeros((2, 3)),
        )
