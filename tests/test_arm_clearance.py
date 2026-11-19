"""The optional arm adjustment stays small and does not scale the limb."""

import numpy as np
import pytest

from motionviewer.core.arm_clearance import outward_rotation


def test_hanging_arm_moves_outward_with_rigid_bounded_rotation():
    direction = np.array([0.0, 0.0, -0.5])
    outward = np.array([1.0, 0.0, 0.0])
    saved = direction.copy()
    rotation = outward_rotation(direction, outward, 3.0)
    result = rotation @ direction
    np.testing.assert_allclose(result, [0.5 * np.sin(np.radians(3)), 0.0, -0.5 * np.cos(np.radians(3))])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.linalg.norm(result) == pytest.approx(0.5)
    np.testing.assert_array_equal(direction, saved)


def test_arm_already_outstretched_is_unchanged_and_opposite_side_mirrors():
    np.testing.assert_array_equal(
        outward_rotation(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 4.0), np.eye(3)
    )
    left = outward_rotation(np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), 4.0) @ [0.0, 0.0, -1.0]
    right = outward_rotation(np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), 4.0) @ [0.0, 0.0, -1.0]
    np.testing.assert_allclose(left * [-1.0, 1.0, 1.0], right)


def test_random_directions_never_exceed_requested_angle_and_degenerate_is_noop():
    rng = np.random.default_rng(420)
    for _ in range(50):
        rotation = outward_rotation(rng.normal(size=3), rng.normal(size=3), 4.0)
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
        assert angle <= 4.0 + 1e-10
    np.testing.assert_array_equal(outward_rotation(np.zeros(3), np.ones(3), 4.0), np.eye(3))


@pytest.mark.parametrize("degrees", [-1, 20.01, np.nan, np.inf])
def test_unbounded_requested_adjustment_is_rejected(degrees):
    with pytest.raises(ValueError, match="between 0 and 20"):
        outward_rotation(np.ones(3), np.ones(3), degrees)


def test_invalid_directions_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        outward_rotation(np.ones(2), np.ones(3), 3.0)
    with pytest.raises(ValueError, match="finite"):
        outward_rotation(np.array([np.nan, 0, 1]), np.ones(3), 3.0)
