import numpy as np

from motionviewer.core.cloth_despike import DespikeSettings, accelerations, despike, relative_tracks

FPS = 50.0


def _swing(frames=120, bodies=3):
    """Pelvis-frame tracks of bodies swinging smoothly, identity rotation."""
    t = np.arange(frames) / FPS
    tracks = np.tile(np.eye(4), (frames, bodies, 1, 1))
    for b in range(bodies):
        tracks[:, b, 0, 3] = 0.05 * np.sin(2 * np.pi * 1.5 * t + b)
        tracks[:, b, 2, 3] = -0.3 - 0.02 * np.cos(2 * np.pi * 1.5 * t + b)
    return tracks


def test_smooth_swing_is_returned_unchanged():
    tracks = _swing()
    cleaned, report = despike(tracks, FPS)
    assert np.array_equal(cleaned, tracks)
    assert report["share_of_cells_changed"] == 0.0


def test_a_one_frame_snap_is_removed_and_the_rest_is_untouched():
    tracks = _swing()
    snapped = tracks.copy()
    snapped[60, 1, 2, 3] -= 0.12  # body 1 drops 12 cm for one frame
    cleaned, report = despike(snapped, FPS)
    assert report["snap_cells_after"] == 0
    assert accelerations(cleaned[:, :, :3, 3], FPS)[:, 1].max() < 0.25 * accelerations(snapped[:, :, :3, 3], FPS)[:, 1].max()
    # Other bodies, and frames far from the snap, are bit-for-bit the input.
    assert np.array_equal(cleaned[:, [0, 2]], snapped[:, [0, 2]])
    assert np.array_equal(cleaned[:50, 1], snapped[:50, 1])
    assert np.array_equal(cleaned[70:, 1], snapped[70:, 1])


def test_a_step_change_arrives_at_its_destination_without_jumping():
    # A panel that genuinely moves to a new rest place must still get there; only the
    # one-frame jump is replaced by a bridge.
    tracks = np.tile(np.eye(4), (60, 1, 1, 1))
    tracks[:, 0, 2, 3] = -0.3
    tracks[30:, 0, 2, 3] = -0.5
    cleaned, _ = despike(tracks, FPS, DespikeSettings(floor=1.0))
    assert np.isclose(cleaned[-1, 0, 2, 3], -0.5)
    assert np.isclose(cleaned[0, 0, 2, 3], -0.3)
    steps = np.abs(np.diff(cleaned[:, 0, 2, 3]))
    assert steps.max() < 0.2 * 0.2


def test_rotations_in_a_refit_window_stay_orthonormal():
    tracks = _swing(frames=80, bodies=1)
    angle = np.radians(70)
    flip = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    tracks[40, 0, :3, :3] = flip
    tracks[40, 0, 2, 3] -= 0.1
    cleaned, _ = despike(tracks, FPS)
    rot = cleaned[:, 0, :3, :3]
    assert np.abs(np.einsum("tji,tjk->tik", rot, rot) - np.eye(3)).max() < 1e-6


def test_relative_tracks_remove_the_walk():
    frames = 50
    pelvis = np.tile(np.eye(4), (frames, 1, 1))
    pelvis[:, 0, 3] = np.linspace(0.0, 1.5, frames)  # walking forward
    bodies = np.tile(np.eye(4), (frames, 1, 1, 1))
    bodies[:, 0, 0, 3] = pelvis[:, 0, 3] + 0.1
    rel = relative_tracks(pelvis, bodies)
    assert np.allclose(rel[:, 0, 0, 3], 0.1)


def test_smoothing_removes_flicker_and_keeps_the_swing():
    from motionviewer.core.cloth_despike import smooth_tracks

    tracks = _swing(frames=150, bodies=2)
    rng = np.random.default_rng(0)
    noisy = tracks.copy()
    noisy[:, :, :3, 3] += rng.normal(0.0, 0.003, size=noisy[:, :, :3, 3].shape)
    smooth = smooth_tracks(noisy, 1.5)
    error_before = np.abs(noisy[:, :, :3, 3] - tracks[:, :, :3, 3]).mean()
    error_after = np.abs(smooth[:, :, :3, 3] - tracks[:, :, :3, 3]).mean()
    assert error_after < 0.6 * error_before  # sigma 1.5 on white noise: ~2x
    swing = np.ptp(tracks[:, :, 0, 3], axis=0)
    assert np.all(np.ptp(smooth[:, :, 0, 3], axis=0) > 0.9 * swing)
    rot = smooth[..., :3, :3]
    assert np.abs(np.einsum("...ji,...jk->...ik", rot, rot) - np.eye(3)).max() < 1e-6


def test_smoothing_off_is_identity():
    from motionviewer.core.cloth_despike import smooth_tracks

    tracks = _swing()
    assert np.array_equal(smooth_tracks(tracks, 0.0), tracks)
