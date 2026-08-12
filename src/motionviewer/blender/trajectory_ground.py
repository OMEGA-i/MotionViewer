from __future__ import annotations

import numpy as np

from ..core.ground import ContactPatch
from ..core.palette import Color
from .style import make_material


def add_contact_patch_geometry(
    patches: list[ContactPatch],
    *,
    color: Color,
    opacity: float,
    name: str,
) -> None:
    import bpy  # type: ignore

    mat = make_material(f"{name}_Ground_Material", Color(color.r, color.g, color.b, opacity), roughness=0.8)
    for idx, patch in enumerate(patches):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48, radius=patch.radius, depth=0.006, location=patch.center
        )
        obj = bpy.context.object
        obj.name = f"{name}_ContactPatch_{idx:03d}"
        obj.data.materials.append(mat)


def add_trajectory_ribbon(
    points: np.ndarray,
    *,
    color: Color,
    opacity: float,
    name: str,
    width: float = 0.05,
) -> None:
    import bpy  # type: ignore

    curve = bpy.data.curves.new(f"{name}_TrajectoryRibbon", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = width
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, value in zip(spline.points, points):
        point.co = (float(value[0]), float(value[1]), float(value[2]), 1.0)
    obj = bpy.data.objects.new(curve.name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(
        make_material(f"{name}_Ribbon_Material", Color(color.r, color.g, color.b, opacity))
    )


def add_trajectory_carpet(
    root_path: np.ndarray,
    *,
    floor_z: float,
    color: Color,
    opacity: float,
    name: str,
    padding: float = 0.35,
    width_override: float = 0.0,
) -> None:
    """Add a soft axis-aligned carpet under the walked trajectory."""
    import bpy  # type: ignore

    xy = np.asarray(root_path[:, :2], dtype=np.float32)
    mins = xy.min(axis=0) - padding
    maxs = xy.max(axis=0) + padding
    center_xy = (mins + maxs) * 0.5
    size_x = float(max(maxs[0] - mins[0], 0.4))
    size_y = float(max(maxs[1] - mins[1], width_override or 0.55))
    if width_override > 0:
        size_y = max(size_y, width_override)
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(float(center_xy[0]), float(center_xy[1]), floor_z + 0.001),
    )
    obj = bpy.context.object
    obj.name = f"{name}_TrajectoryCarpet"
    obj.scale = (size_x * 0.5, size_y * 0.5, 0.003)
    mat = make_material(f"{name}_Carpet_Material", Color(color.r, color.g, color.b, opacity), roughness=0.95)
    obj.data.materials.append(mat)


def add_trajectory_rectangle_ground(
    root_path: np.ndarray,
    *,
    floor_z: float,
    color: Color,
    opacity: float,
    name: str,
    padding: float = 0.35,
    min_width: float = 0.55,
) -> None:
    """Add one clean rectangle covering the whole clip trajectory footprint."""
    import bpy  # type: ignore

    if len(root_path) == 0:
        return
    xy = np.asarray(root_path[:, :2], dtype=np.float32)
    mins = xy.min(axis=0) - padding
    maxs = xy.max(axis=0) + padding
    center_xy = (mins + maxs) * 0.5
    size_x = float(max(maxs[0] - mins[0], min_width))
    size_y = float(max(maxs[1] - mins[1], min_width))
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(float(center_xy[0]), float(center_xy[1]), floor_z + 0.001),
    )
    obj = bpy.context.object
    obj.name = f"{name}_TrajectoryRectangleGround"
    obj.scale = (size_x * 0.5, size_y * 0.5, 0.003)
    mat = make_material(
        f"{name}_RectangleGround_Material", Color(color.r, color.g, color.b, opacity), roughness=0.95
    )
    obj.data.materials.append(mat)


def add_segment_overlay_carpet(
    root_path: np.ndarray,
    *,
    floor_z: float,
    color: Color,
    opacity: float,
    name: str,
    padding: float = 0.25,
    width_override: float = 0.0,
) -> None:
    if len(root_path) < 2:
        return
    add_trajectory_carpet(
        root_path,
        floor_z=floor_z + 0.003,
        color=color,
        opacity=opacity,
        name=name,
        padding=padding,
        width_override=width_override,
    )


def add_segmented_trajectory_carpet(
    root_path: np.ndarray,
    *,
    floor_z: float,
    prefix_t: int,
    prefix_color: Color,
    generated_color: Color,
    opacity: float,
    name: str,
    width: float = 0.55,
) -> None:
    """Add one continuous ribbon mesh with prefix/generated material segments."""

    if len(root_path) < 2:
        return
    import bpy  # type: ignore

    points = np.asarray(root_path[:, :2], dtype=np.float32)
    half_width = max(width, 0.08) * 0.5
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for idx, point in enumerate(points):
        if idx == 0:
            tangent = points[1] - point
        elif idx == len(points) - 1:
            tangent = point - points[idx - 1]
        else:
            tangent = points[idx + 1] - points[idx - 1]
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
        else:
            tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        left = point + normal * half_width
        right = point - normal * half_width
        z = floor_z + 0.001
        vertices.append((float(left[0]), float(left[1]), z))
        vertices.append((float(right[0]), float(right[1]), z))
    for idx in range(len(points) - 1):
        faces.append((idx * 2, idx * 2 + 1, idx * 2 + 3, idx * 2 + 2))

    mesh = bpy.data.meshes.new(f"{name}_SegmentedCarpetMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{name}_SegmentedCarpet", mesh)
    bpy.context.collection.objects.link(obj)
    prefix_mat = make_material(
        f"{name}_CarpetPrefix_Material",
        Color(prefix_color.r, prefix_color.g, prefix_color.b, opacity * 0.75),
        roughness=0.95,
    )
    generated_mat = make_material(
        f"{name}_CarpetGenerated_Material",
        Color(generated_color.r, generated_color.g, generated_color.b, opacity),
        roughness=0.95,
    )
    mesh.materials.append(prefix_mat)
    mesh.materials.append(generated_mat)
    for idx, poly in enumerate(mesh.polygons):
        poly.material_index = 0 if prefix_t > 0 and idx < prefix_t - 1 else 1


def add_prefix_transition_marker(
    position: tuple[float, float, float],
    *,
    color: Color,
    name: str,
) -> None:
    import bpy  # type: ignore

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.06, location=position)
    obj = bpy.context.object
    obj.name = f"{name}_PrefixMarker"
    mat = make_material(f"{name}_PrefixMarker_Mat", Color(color.r, color.g, color.b, 0.85), roughness=0.4)
    obj.data.materials.append(mat)
