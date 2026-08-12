from __future__ import annotations

import numpy as np

from motionviewer.blender.render import frames_output_dir
from motionviewer.core.inplace import (
    freeze_horizontal_root_joints_blender,
    freeze_horizontal_root_transl_source,
)
from motionviewer.video.spec import (
    CameraSpec,
    CameraViewSpec,
    group_views_by_staging,
    resolve_camera_views,
)


def test_resolve_camera_views_defaults_front_inplace() -> None:
    camera = CameraSpec(
        views=[
            CameraViewSpec(preset="three_quarter"),
            CameraViewSpec(preset="front"),
            CameraViewSpec(preset="side"),
        ]
    )
    views = resolve_camera_views(
        camera,
        style_ghost={"mode": "trail", "alpha": 0.2},
        ground={"mode": "trajectory_carpet", "opacity": 0.1},
    )
    by_preset = {view.preset: view for view in views}
    assert by_preset["three_quarter"].staging == "world"
    assert by_preset["three_quarter"].ghost["mode"] == "trail"
    assert by_preset["front"].staging == "inplace"
    assert by_preset["front"].ghost["mode"] == "none"
    assert by_preset["front"].ground["mode"] == "none"
    assert by_preset["side"].staging == "inplace"
    groups = group_views_by_staging(views)
    assert set(groups) == {"world", "inplace"}
    assert [v.preset for v in groups["inplace"]] == ["front", "side"]


def test_resolve_camera_views_falls_back_to_preset() -> None:
    views = resolve_camera_views(CameraSpec(preset="top"))
    assert len(views) == 1
    assert views[0].preset == "top"
    assert views[0].staging == "world"


def test_freeze_horizontal_root_transl_source() -> None:
    transl = np.zeros((5, 3), dtype=np.float32)
    transl[:, 0] = np.arange(5)
    transl[:, 1] = np.linspace(0.0, 0.4, 5)
    transl[:, 2] = np.arange(5) * 0.5
    out = freeze_horizontal_root_transl_source(transl)
    assert np.allclose(out[:, 0], 0.0)
    assert np.allclose(out[:, 2], 0.0)
    assert np.allclose(out[:, 1], transl[:, 1])


def test_freeze_horizontal_root_joints_blender() -> None:
    joints = np.zeros((4, 22, 3), dtype=np.float32)
    joints[:, 0, 0] = np.arange(4)
    joints[:, 0, 1] = np.arange(4) * 2
    joints[:, 0, 2] = 0.1
    joints[:, 1, 0] = joints[:, 0, 0] + 0.2
    out = freeze_horizontal_root_joints_blender(joints)
    assert np.allclose(out[:, 0, 0], 0.0)
    assert np.allclose(out[:, 0, 1], 0.0)
    assert np.allclose(out[:, 0, 2], 0.1)
    assert np.allclose(out[:, 1, 0], 0.2)


def test_frames_output_dir_helper() -> None:
    assert frames_output_dir("/tmp/out").as_posix().endswith("frames")
    assert frames_output_dir("/tmp/out", "front").as_posix().endswith("frames/front")
