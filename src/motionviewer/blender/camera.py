from __future__ import annotations


def add_camera_for_bounds(
    bounds_min: list[float],
    bounds_max: list[float],
    *,
    preset: str = "three_quarter",
    margin: float = 1.15,
    orthographic: bool = True,
    resolution: tuple[int, int] = (1920, 1080),
    name: str | None = None,
    set_active: bool = True,
):
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    mins = Vector(bounds_min)
    maxs = Vector(bounds_max)
    center = (mins + maxs) * 0.5
    span = maxs - mins
    distance_factor = 0.83 if preset == "perspective_front" else 2.2
    distance = max(float(max(span.x, span.y, span.z)) * distance_factor * margin, 3.0)
    direction = _preset_direction(preset)
    location = center + direction * distance
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = name or f"MotionViewer_Camera_{preset}"
    _look_at(camera, center)
    if orthographic:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = _fit_ortho_scale(camera, mins, maxs, resolution, margin)
    else:
        camera.data.type = "PERSP"
        camera.data.lens = 48.0 if preset == "perspective_front" else 55.0
        camera.data.sensor_fit = "HORIZONTAL"
    if set_active:
        bpy.context.scene.camera = camera
    return camera


def _fit_ortho_scale(
    camera,
    mins,
    maxs,
    resolution: tuple[int, int],
    margin: float,
) -> float:
    """Size the orthographic view to actually contain the scene bounds as seen by this camera.

    A view direction that isn't axis-aligned (e.g. the default 3/4 preset) can need a much
    wider view than any single world-axis span (e.g. long trajectories);
    sizing purely off world-space spans (ignoring the viewing angle and render aspect ratio)
    under-fits and clips actors out of frame.
    """
    from mathutils import Vector  # type: ignore

    rot_matrix = camera.rotation_euler.to_matrix()
    right = rot_matrix.col[0]
    up = rot_matrix.col[1]
    center = (mins + maxs) * 0.5
    corners = [
        Vector((x, y, z)) - center
        for x in (mins.x, maxs.x)
        for y in (mins.y, maxs.y)
        for z in (mins.z, maxs.z)
    ]
    half_width = max(abs(corner.dot(right)) for corner in corners)
    half_height = max(abs(corner.dot(up)) for corner in corners)

    width_px, height_px = max(1, resolution[0]), max(1, resolution[1])
    if width_px >= height_px:
        aspect = width_px / height_px
        view_width = max(half_width * 2.0, half_height * 2.0 * aspect)
        ortho_scale = view_width
    else:
        aspect = height_px / width_px
        view_height = max(half_height * 2.0, half_width * 2.0 * aspect)
        ortho_scale = view_height
    return max(ortho_scale, 1.0) * margin


def _preset_direction(preset: str):
    from mathutils import Vector  # type: ignore

    if preset == "front":
        return Vector((0.0, -1.0, 0.34)).normalized()
    if preset == "perspective_front":
        return Vector((0.0, -1.0, 0.25)).normalized()
    if preset in {"upper_left", "upper_right"}:
        x = -0.70 if preset == "upper_left" else 0.70
        return Vector((x, -1.0, 0.34)).normalized()
    if preset == "side":
        return Vector((1.0, 0.0, 0.14)).normalized()
    if preset == "top":
        return Vector((0.0, 0.0, 1.0)).normalized()
    if preset == "arc":
        # Higher oblique view makes ground-plane arc layouts legible in a square frame.
        return Vector((0.70, -1.00, 0.75)).normalized()
    if preset in {"three_quarter", "showcase"}:
        # Low three-quarter view keeps chest/limb silhouettes readable while
        # leaving enough floor visible for a pose grid.
        return Vector((0.70, -1.00, 0.34)).normalized()
    raise ValueError(f"Unknown camera preset {preset!r}")


def _look_at(obj, target) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
