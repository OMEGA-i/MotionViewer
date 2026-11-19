import numpy as np

from motionviewer.core.limb_ik import limit_knee_change, solve_two_bone_positions, support_height_adjustment


def test_knee_guard_bounds_added_bend_and_transition_speed():
    hips = np.tile([0., 0., 1.], (30, 2, 1))
    knees = np.tile([0., -0.2, 0.6], (30, 2, 1))
    ankles = np.tile([0., 0., 0.2], (30, 2, 1))
    goals = ankles.copy()
    goals[10:20, :, 2] = 0.05
    limited, delta = limit_knee_change(hips, knees, ankles, goals)
    assert abs(delta).max() <= 8.
    assert abs(np.diff(delta, axis=0)).max() <= 1.5 + 1e-10
    assert np.isfinite(limited).all()


def test_support_lowers_hovering_sole_without_pulling_airborne_foot_down():
    result = support_height_adjustment(np.array([0.02, 0.10, 0.02]), np.array([1., 0., 0.5]),
                                       floor_z=0., limit_m=0.06)
    np.testing.assert_allclose(result, [-0.02, 0., -0.01])


def test_penetration_lift_is_bounded_even_without_classified_contact():
    result = support_height_adjustment(np.array([-0.02, -0.5]), np.zeros(2), floor_z=0., limit_m=0.06)
    np.testing.assert_allclose(result, [0.02, 0.06])


def test_reaches_goal_preserving_lengths_and_bend_side():
    hip, knee, ankle = np.array([0., 0., 1.]), np.array([0., -0.2, 0.6]), np.array([0., 0., 0.2])
    goal = np.array([0.05, 0.02, 0.23])
    middle, reached, error = solve_two_bone_positions(hip, knee, ankle, goal)
    np.testing.assert_allclose(reached, goal, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(middle - hip), np.linalg.norm(knee - hip))
    np.testing.assert_allclose(np.linalg.norm(reached - middle), np.linalg.norm(ankle - knee))
    assert middle[1] < 0
    assert error < 1e-10


def test_unreachable_goal_does_not_stretch_or_nan():
    hip, knee, ankle = np.array([0., 0., 1.]), np.array([0., 0., 0.6]), np.array([0., 0., 0.2])
    middle, reached, error = solve_two_bone_positions(hip, knee, ankle, np.array([0., 0., -0.2]),
                                                    pole_hint=np.array([0., -1., 0.]))
    np.testing.assert_allclose(np.linalg.norm(middle - hip), 0.4)
    np.testing.assert_allclose(np.linalg.norm(reached - middle), 0.4)
    assert error > 0.39
    assert np.isfinite(middle).all()
