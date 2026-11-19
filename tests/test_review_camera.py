"""Camera records must reproduce a review shot or reject a mismatched job."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from motionviewer.blender.review_camera import capture_camera_track, load_camera_tracks


def _track():
    data = SimpleNamespace(type="ORTHO", sensor_fit="AUTO", sensor_width=36., sensor_height=24.,
                           ortho_scale=2., lens=50., shift_x=0., shift_y=0., clip_start=.1, clip_end=100.)
    camera = SimpleNamespace(data=data, matrix_world=np.eye(4))
    scene = SimpleNamespace(frame_current=7, render=SimpleNamespace(
        fps=50, fps_base=1., resolution_x=640, resolution_y=480, resolution_percentage=100,
        pixel_aspect_x=1., pixel_aspect_y=1.,
    ))

    def frame_set(frame):
        scene.frame_current = frame
        camera.matrix_world = np.eye(4)
        camera.matrix_world[0, 3] = frame * .25
        camera.data.ortho_scale = 2. + frame * .01

    scene.frame_set = frame_set
    bpy = SimpleNamespace(context=SimpleNamespace(scene=scene, view_layer=SimpleNamespace(update=lambda: None)))
    record = capture_camera_track(bpy, camera, frame_start=1, frames=3)
    assert scene.frame_current == 7
    return record


def test_camera_records_every_frame_and_validates_a_matching_job(tmp_path):
    record = _track()
    assert [sample["matrix_world"][0][3] for sample in record["samples"]] == [.25, .5, .75]
    assert [sample["ortho_scale"] for sample in record["samples"]] == [2.01, 2.02, 2.03]
    path = tmp_path / "info.json"
    path.write_text(json.dumps({"camera": "follow", "camera_tracks": {"front": record}}))
    tracks, mode = load_camera_tracks(path, views=["front"], frame_start=1, frames=3, resolution=(640, 480), fps=50)
    assert tracks["front"] == record
    assert mode == "follow"


@pytest.mark.parametrize("change", [
    {"resolution": (1280, 960)}, {"frames": 4}, {"frame_start": 0},
    {"fps": 30}, {"views": ["side"]},
])
def test_camera_reuse_rejects_a_different_shot(tmp_path, change):
    path = tmp_path / "info.json"
    path.write_text(json.dumps({"camera_tracks": {"front": _track()}}))
    options = dict(views=["front"], frame_start=1, frames=3, resolution=(640, 480), fps=50)
    options.update(change)
    with pytest.raises(ValueError):
        load_camera_tracks(path, **options)


def test_missing_camera_sample_cannot_be_silently_interpolated(tmp_path):
    record = copy.deepcopy(_track())
    record["samples"].pop(1)
    path = tmp_path / "info.json"
    path.write_text(json.dumps({"camera_tracks": {"front": record}}))
    with pytest.raises(ValueError, match="missing frame samples"):
        load_camera_tracks(path, views=["front"], frame_start=1, frames=3, resolution=(640, 480), fps=50)
