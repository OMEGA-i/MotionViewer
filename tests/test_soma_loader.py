"""Reject incompatible SOMA encodings before they can silently change the motion."""

from __future__ import annotations

import numpy as np
import pytest

from motionviewer.blender.retarget.soma_pipeline import SOMA_POSE_CONVENTION, load_soma
from motionviewer.core.soma_fk import SOMA77_NAMES


def _payload():
    return {
        "joints77": np.zeros((2, 77, 3)),
        "rotmat77": np.broadcast_to(np.eye(3), (2, 77, 3, 3)).copy(),
        "transl": np.array([[1., 2., 3.], [1.2, 2.1, 3.]]),
        "bone_lengths": np.full(77, .03),
        "joint_names": np.asarray(SOMA77_NAMES),
        "joint_parents": np.arange(-1, 76),
        "fps": np.float64(29.97),
        "caption": np.str_("original generated motion"),
    }


@pytest.mark.parametrize("with_convention", [False, True])
def test_loader_keeps_arrays_and_accepts_legacy_files_without_convention(tmp_path, with_convention):
    payload = _payload()
    if with_convention:
        payload["pose_convention"] = np.str_(SOMA_POSE_CONVENTION)
    source = tmp_path / "source.soma.npz"
    np.savez(source, **payload)
    before = source.read_bytes()
    loaded = load_soma(source)
    assert source.read_bytes() == before
    for key, value in payload.items():
        assert np.array_equal(loaded[key], value)
    assert loaded["fps"] == 29.97


@pytest.mark.parametrize("convention", ["world", "absolute", "", [SOMA_POSE_CONVENTION]])
def test_loader_rejects_an_unknown_or_nonscalar_rotation_convention(tmp_path, convention):
    payload = _payload()
    payload["pose_convention"] = np.asarray(convention)
    source = tmp_path / "source.soma.npz"
    np.savez(source, **payload)
    with pytest.raises(ValueError, match="pose_convention"):
        load_soma(source)


@pytest.mark.parametrize(("key", "value"), [
    ("joints77", np.zeros((0, 77, 3))),
    ("joints77", np.zeros((2, 76, 3))),
    ("rotmat77", np.zeros((3, 77, 3, 3))),
    ("rotmat77", np.zeros((2, 77, 3))),
    ("transl", np.zeros((1, 3))),
    ("bone_lengths", np.zeros(76)),
    ("joint_names", np.asarray(SOMA77_NAMES[:-1])),
    ("joint_parents", np.arange(-1, 75)),
    ("joint_parents", np.array([-1, 77] + list(range(1, 76)))),
    ("joint_parents", np.array([-1., .5] + list(range(1, 76)))),
    ("rotmat77", np.full((2, 77, 3, 3), np.nan)),
    ("transl", np.full((2, 3), np.inf)),
    ("bone_lengths", np.full(77, -.01)),
    ("fps", np.float64(0)),
    ("fps", np.float64(np.nan)),
])
def test_loader_rejects_mismatched_frames_and_invalid_joint_data(tmp_path, key, value):
    payload = _payload()
    payload[key] = value
    source = tmp_path / "source.soma.npz"
    np.savez(source, **payload)
    expected = "parent" if key == "joint_parents" else key
    with pytest.raises(ValueError, match=expected):
        load_soma(source)
