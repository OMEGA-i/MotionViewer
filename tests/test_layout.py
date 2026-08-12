from __future__ import annotations

import numpy as np

from motionviewer.core.layout import start_root_offsets, trajectory_footprint


def test_trajectory_footprint_has_padding() -> None:
    joints = np.zeros((10, 22, 3), dtype=np.float32)
    joints[:, 0, 0] = np.linspace(0.0, 1.0, 10)
    mins, maxs = trajectory_footprint(joints, padding=0.3)
    assert mins[0] < 0.0
    assert maxs[0] > 1.0


def test_start_root_offsets_center_first_root() -> None:
    roots = [np.array([1.0, 2.0, 0.0], dtype=np.float32)]
    offsets = start_root_offsets(roots, [(0.0, 0.0, 0.0)])
    assert offsets[0] == (-1.0, -2.0, 0.0)
