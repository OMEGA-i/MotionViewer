"""Record and replay exact per-frame cameras for before/after reviews."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_STATIC_FIELDS = ("type", "sensor_fit", "sensor_width", "sensor_height")
_ANIMATED_FIELDS = ("ortho_scale", "lens", "shift_x", "shift_y", "clip_start", "clip_end")


def load_camera_tracks(
    path: Path, *, views: list[str], frame_start: int, frames: int,
    resolution: tuple[int, int], fps: float,
) -> tuple[dict, str]:
    """Validate a renderer info.json before any replacement camera is created."""
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    tracks = info.get("camera_tracks", {})
    selected = {}
    for view in views:
        if view not in tracks:
            raise ValueError(f"camera reference has no recorded view {view!r}")
        track = tracks[view]
        if track.get("frame_start") != frame_start or track.get("frame_count") != frames:
            raise ValueError(f"camera reference {view!r} must have the same start frame and frame count")
        if track.get("resolution") != list(resolution) or track.get("resolution_percentage") != 100:
            raise ValueError(f"camera reference {view!r} must have the same output resolution at 100%")
        if not np.isclose(track.get("fps", 0), fps, rtol=0, atol=1e-6):
            raise ValueError(f"camera reference {view!r} must have the same frame rate")
        if track.get("camera", {}).get("type") not in {"ORTHO", "PERSP"}:
            raise ValueError(f"unsupported reference camera type for {view!r}")
        samples = track.get("samples", [])
        if len(samples) != frames:
            raise ValueError(f"camera reference {view!r} is missing frame samples")
        for offset, sample in enumerate(samples):
            matrix = np.asarray(sample.get("matrix_world"), dtype=float)
            if sample.get("frame") != frame_start + offset or matrix.shape != (4, 4):
                raise ValueError(f"invalid camera sample {offset} for {view!r}")
            if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise ValueError(f"invalid camera transform at sample {offset} for {view!r}")
            if any(not np.isfinite(sample.get(field, np.nan)) for field in _ANIMATED_FIELDS):
                raise ValueError(f"invalid camera settings at sample {offset} for {view!r}")
            if sample["ortho_scale"] <= 0 or sample["lens"] <= 0:
                raise ValueError(f"invalid camera scale/lens at sample {offset} for {view!r}")
        selected[view] = track
    return selected, info.get("camera", "static")


def capture_camera_track(bpy: Any, camera: Any, *, frame_start: int, frames: int) -> dict:
    scene = bpy.context.scene
    previous = scene.frame_current
    samples = []
    for frame in range(frame_start, frame_start + frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        samples.append({
            "frame": frame,
            "matrix_world": [list(row) for row in camera.matrix_world],
            **{field: float(getattr(camera.data, field)) for field in _ANIMATED_FIELDS},
        })
    scene.frame_set(previous)
    bpy.context.view_layer.update()
    return {
        "frame_start": frame_start,
        "frame_count": frames,
        "fps": scene.render.fps / scene.render.fps_base,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "resolution_percentage": scene.render.resolution_percentage,
        "pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
        "camera": {field: getattr(camera.data, field) for field in _STATIC_FIELDS},
        "samples": samples,
    }


def apply_camera_track(bpy: Any, track: dict, *, name: str) -> Any:
    from mathutils import Matrix

    from .retarget._animate import _linearize

    data = bpy.data.cameras.new(name)
    camera = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(camera)
    camera.rotation_mode = "QUATERNION"
    for field in _STATIC_FIELDS:
        setattr(data, field, track["camera"][field])
    previous_rotation = None
    for sample in track["samples"]:
        location, rotation, scale = Matrix(sample["matrix_world"]).decompose()
        if previous_rotation is not None and rotation.dot(previous_rotation) < 0:
            rotation.negate()
        previous_rotation = rotation.copy()
        camera.location, camera.rotation_quaternion, camera.scale = location, rotation, scale
        for field in ("location", "rotation_quaternion", "scale"):
            camera.keyframe_insert(data_path=field, frame=sample["frame"])
        for field in _ANIMATED_FIELDS:
            setattr(data, field, sample[field])
            data.keyframe_insert(data_path=field, frame=sample["frame"])
    _linearize(camera)
    _linearize(data)
    scene = bpy.context.scene
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = track.get("pixel_aspect", [1.0, 1.0])
    scene.camera = camera
    scene.frame_set(track["frame_start"])
    bpy.context.view_layer.update()
    return camera


def fit_orthographic_geometry(
    bpy: Any, camera: Any, objects: list[Any], *, frame_start: int, frames: int,
    padding: float = 1.1,
) -> dict:
    """Fit actual projected geometry over the full clip, retaining camera travel.

    An axis-aligned world box around a swinging weapon substantially overstates
    the screen-space extent. Sample evaluated vertices in each original camera
    frame instead. Use one shared scale and offset, so the shot does not zoom or
    chase individual limbs. Matched review cameras must bypass this adjustment.
    """
    from mathutils import Vector

    from .retarget._animate import _linearize

    if camera.data.type != "ORTHO":
        raise ValueError("projected geometry fitting requires an orthographic camera")
    if not objects or frames <= 0 or not np.isfinite(padding) or padding < 1.0:
        raise ValueError("geometry fitting requires objects, positive frames and padding >= 1")
    scene = bpy.context.scene
    previous = scene.frame_current
    minimum, maximum = np.full(2, np.inf), np.full(2, -np.inf)
    matrices = []
    for frame in range(frame_start, frame_start + frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        matrices.append(camera.matrix_world.copy())
        inverse = np.asarray(camera.matrix_world.inverted())
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            evaluated = obj.evaluated_get(depsgraph)
            vertices = evaluated.data.vertices
            if not len(vertices):
                continue
            points = np.empty(len(vertices) * 3, dtype=np.float32)
            vertices.foreach_get("co", points)
            transform = inverse @ np.asarray(evaluated.matrix_world)
            projected = points.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
            minimum = np.minimum(minimum, projected[:, :2].min(axis=0))
            maximum = np.maximum(maximum, projected[:, :2].max(axis=0))
    if not np.isfinite(minimum).all() or not np.all(maximum > minimum):
        raise ValueError("no non-degenerate visible geometry to frame")
    extent = maximum - minimum
    # Blender's sensor-fit choice also affects portrait framing. Obtain the
    # actual view plane rather than assuming scale always denotes frame height.
    view = np.asarray([tuple(v) for v in camera.data.view_frame(scene=scene)])
    unit_span = np.ptp(view[:, :2], axis=0) / camera.data.ortho_scale
    scale = float(np.max(extent / unit_span) * padding)
    offset = Vector([*(0.5 * (minimum + maximum)), 0.0])
    for frame, matrix in zip(range(frame_start, frame_start + frames), matrices, strict=True):
        scene.frame_set(frame)
        camera.location = matrix.translation + matrix.to_3x3() @ offset
        camera.keyframe_insert("location", frame=frame)
        camera.data.ortho_scale = scale
        camera.data.keyframe_insert("ortho_scale", frame=frame)
    _linearize(camera)
    _linearize(camera.data)
    scene.frame_set(previous)
    bpy.context.view_layer.update()
    return {"method": "all-frame evaluated geometry in camera plane", "frames": frames,
            "padding": padding, "projected_bounds_m": [minimum.tolist(), maximum.tolist()],
            "ortho_scale": scale, "constant_scale_and_offset": True}
