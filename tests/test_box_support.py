import numpy as np
import pytest

from motionviewer.core.box_support import fit_supported_box


def inputs():
    palms = np.tile([[[-.1, 0., 1.], [.1, 0., 1.]]], (8, 1, 1))
    translation = np.column_stack((np.arange(8) * .03, np.arange(8) * .1, np.zeros(8)))
    palms += translation[:, None]
    normals = np.tile([[[0., 0., 1.], [0., 0., 1.]]], (8, 1, 1))
    patch = np.array([[-.02, -.03, 0.], [.02, -.03, 0.], [.02, .03, 0.], [-.02, .03, 0.]])
    vertices = tuple(palms[:, side, None, :] + patch for side in range(2))
    torso = translation + [0., -.3, .9]
    return palms, normals, vertices, torso, translation


def test_fixed_box_follows_translating_hands_from_above_without_mutating_inputs():
    palms, normals, vertices, torso, translation = inputs()
    originals = [a.copy() for a in (palms, normals, *vertices, torso)]
    fit = fit_supported_box(palms, normals, vertices, torso)
    np.testing.assert_allclose(fit.centres - translation, np.tile([0., .02, 1.076], (8, 1)), atol=1e-12)
    np.testing.assert_allclose(fit.rotations, np.tile(np.eye(3), (8, 1, 1)), atol=1e-12)
    assert fit.size == (.4, .22, .15)
    assert fit.report["support_gap_max_mm"] == pytest.approx(1.)
    for before, after in zip(originals, (palms, normals, *vertices, torso), strict=True):
        np.testing.assert_array_equal(before, after)


def test_opposed_palms_are_reported_instead_of_rotating_the_character():
    palms, normals, vertices, torso, _ = inputs()
    normals[:, 1] *= -1
    before = normals.copy()
    fit = fit_supported_box(palms, normals, vertices, torso)
    assert fit.report["actor_pose_modified"] is False
    assert fit.report["frame_offsets_palm_misalignment_over_45deg"] == list(range(8))
    np.testing.assert_array_equal(normals, before)


def test_fixed_footprint_is_not_scaled_to_hide_unreachable_hands():
    palms, normals, vertices, torso, _ = inputs()
    with pytest.raises(ValueError, match="footprint"):
        fit_supported_box(palms, normals, vertices, torso, size=(.04, .22, .15))


def test_smoothed_orientation_recomputes_actual_skin_clearance_and_records_missing_support():
    palms, normals, vertices, torso, _ = inputs()
    vertices[1][:, :, 2] += np.array([0., .02, .04, .06, .08, .06, .04, .02])[:, None]
    fit = fit_supported_box(palms, normals, vertices, torso)
    assert fit.report["max_tilt_about_hand_span_degrees"] <= 15.
    assert fit.report["frame_offsets_support_gap_over_10mm"]
    for frame, row in enumerate(fit.report["frames"]):
        for hand in range(2):
            local = (vertices[hand][frame] - fit.centres[frame]) @ fit.rotations[frame]
            inside = np.all(np.abs(local[:, :2]) <= np.asarray(fit.size[:2]) / 2, axis=1)
            measured_gap = -fit.size[2] / 2 - local[inside, 2].max()
            assert measured_gap >= .001 - 1e-12
            assert row["support_gap_mm"][hand] == pytest.approx(measured_gap * 1000)
