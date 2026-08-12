from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from motionviewer.blender.camera import add_camera_for_bounds
from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
from motionviewer.blender.scene import add_lighting, clear_scene, setup_world
from motionviewer.blender.smplx_mesh import create_smplx_actor_from_npz
from motionviewer.blender.style import make_material
from motionviewer.core.palette import Color
from motionviewer.core.smplx_actor import override_foot_pose


@dataclass
class _SourceState:
    source_id: str
    actor: Any
    snapshots: list[Any]
    final_frame: int
    output_path: Path
    frame_indices: list[int]

    @property
    def render_objects(self) -> list[Any]:
        return self.snapshots


def render_qualitative_bundle(bundle_path: str | Path) -> dict[str, Any]:
    import bpy  # type: ignore

    path = Path(bundle_path).resolve()
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("schema") != "motionviewer.qualitative.v1":
        raise ValueError(f"Unsupported qualitative bundle schema in {path}")

    clear_scene()
    setup_world(transparent=True)
    _configure_scene(
        bpy,
        resolution=tuple(int(value) for value in bundle["resolution"]),
        samples=int(bundle["samples"]),
        snapshot_alpha=float(bundle.get("snapshot_alpha", 1.0)),
    )

    states: list[_SourceState] = []
    for source in bundle["sources"]:
        states.append(_create_source_state(bpy, bundle, source))

    bounds_min, bounds_max = _shared_bounds(bpy, states)
    add_lighting(bounds_min.tolist(), bounds_max.tolist())
    camera_spec = dict(bundle["camera"])
    if bundle.get("snapshot_layout") == "arc":
        camera_spec["preset"] = "arc"
    camera = add_camera_for_bounds(
        bounds_min.tolist(),
        bounds_max.tolist(),
        preset=camera_spec["preset"],
        margin=float(camera_spec["margin"]),
        orthographic=True,
        resolution=tuple(int(value) for value in bundle["resolution"]),
        name="Qualitative_Shared_Camera",
        set_active=True,
    )
    _fit_camera_to_objects(
        bpy,
        camera,
        [obj for state in states for obj in state.render_objects],
        resolution=tuple(int(value) for value in bundle["resolution"]),
        margin=float(camera_spec["margin"]),
    )

    snapshot_alpha = float(bundle.get("snapshot_alpha", 1.0))
    for state in states:
        _render_source(bpy, states, state, snapshot_alpha=snapshot_alpha)

    manifest = {
        "schema": bundle["schema"],
        "clip_id": bundle["clip_id"],
        "provenance": bundle["provenance"],
        "caption": bundle.get("caption"),
        "fbx": bundle["fbx"],
        "snapshot_count": int(bundle["snapshots"]),
        "snapshot_layout": bundle.get("snapshot_layout", "trajectory"),
        "snapshot_spacing": float(bundle.get("snapshot_spacing", 1.0)),
        "material_mode": bundle.get("material_mode", "palette"),
        "body_mode": bundle.get("body_mode", "fbx"),
        "foot_pose": bundle.get("foot_pose", "source"),
        "sources": [
            {
                "source_id": state.source_id,
                "frames": state.final_frame,
                "frame_indices": state.frame_indices,
                "output_path": str(state.output_path),
            }
            for state in states
        ],
        "camera": {
            "preset": camera_spec["preset"],
            "orthographic": True,
            "margin": float(camera_spec["margin"]),
            "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
            "location": [float(value) for value in camera.location],
            "rotation_euler": [float(value) for value in camera.rotation_euler],
            "ortho_scale": float(camera.data.ortho_scale),
            "fit": "projected_mesh_bounds",
        },
        "render": {
            "resolution": list(bundle["resolution"]),
            "samples": int(bundle["samples"]),
            "transparent_background": True,
            "labels": False,
            "ground": False,
            "snapshot_layout": bundle.get("snapshot_layout", "trajectory"),
            "snapshot_alpha": snapshot_alpha,
            "material_mode": bundle.get("material_mode", "palette"),
            "body_mode": bundle.get("body_mode", "fbx"),
            "foot_pose": bundle.get("foot_pose", "source"),
        },
    }
    manifest_path = Path(bundle["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    status = {
        "status": "rendered",
        "clip_id": bundle["clip_id"],
        "outputs": [str(state.output_path) for state in states],
        "manifest": str(manifest_path),
    }
    _write_status(bundle, status)
    return status


def _create_source_state(bpy: Any, bundle: dict[str, Any], source: dict[str, Any]) -> _SourceState:
    source_id = str(source["source_id"])
    frame_indices = [int(value) for value in source["frame_indices"]]
    if len(frame_indices) != int(bundle["snapshots"]):
        raise ValueError(f"{source_id} frame_indices do not match snapshot count")
    offset = _start_root_offset(Path(source["motion_path"]))
    body_mode = str(bundle.get("body_mode", "fbx"))
    motion_overrides = _foot_pose_overrides(
        Path(source["motion_path"]), str(bundle.get("foot_pose", "source"))
    )
    if body_mode == "fbx":
        actor = create_fbx_actor_from_npz(
            source["motion_path"],
            label=f"Qualitative_{source_id}",
            fbx_path=bundle["fbx"]["path"],
            bone_map=bundle["fbx"].get("bone_map", "auto"),
            retarget_mode=bundle["fbx"].get("retarget_mode", "quality"),
            layout_offset=offset,
            motion_overrides=motion_overrides,
        )
    elif body_mode in {"smplh", "smplx"}:
        actor = create_smplx_actor_from_npz(
            source["motion_path"],
            label=f"Qualitative_{source_id}",
            body_model=body_mode,
            layout_offset=offset,
            motion_overrides=motion_overrides,
        )
    else:
        raise ValueError(f"Unknown body_mode {body_mode!r}")
    snapshot_layout = str(bundle.get("snapshot_layout", "trajectory"))
    snapshot_spacing = float(bundle.get("snapshot_spacing", 1.0))
    arc_direction = str(bundle.get("arc_direction", "up"))
    material_mode = str(bundle.get("material_mode", "palette"))
    palette_start_rgb = tuple(int(value) for value in bundle.get("palette_start_rgb", (26, 128, 184)))
    palette_end_rgb = tuple(int(value) for value in bundle.get("palette_end_rgb", (122, 26, 158)))
    palette_color_rgb = bundle.get("palette_color_rgb")
    if palette_color_rgb is not None:
        palette_start_rgb = palette_end_rgb = tuple(int(value) for value in palette_color_rgb)
    if snapshot_layout not in {"trajectory", "root_aligned", "arc"}:
        raise ValueError(f"Unknown snapshot_layout {snapshot_layout!r}")
    if material_mode not in {"palette", "preserve"}:
        raise ValueError(f"Unknown material_mode {material_mode!r}")

    snapshots: list[Any] = []
    for snapshot_index, frame_index in enumerate(frame_indices):
        frame = int(frame_index) + 1
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        root_location = _evaluated_root_location(bpy, actor)
        translation = snapshot_layout_translation(
            root_location,
            snapshot_index=snapshot_index,
            snapshot_count=len(frame_indices),
            layout=snapshot_layout,
            spacing=snapshot_spacing,
            camera_preset=("arc" if snapshot_layout == "arc" else bundle["camera"]["preset"]),
            arc_direction=arc_direction,
        )
        snapshots.extend(
            _freeze_snapshot_meshes(
                bpy,
                actor,
                frame=frame,
                source_id=source_id,
                snapshot_index=snapshot_index,
                snapshot_count=len(frame_indices),
                translation=translation,
                material_mode=material_mode,
                snapshot_layout=snapshot_layout,
                palette_start_rgb=palette_start_rgb,
                palette_end_rgb=palette_end_rgb,
            )
        )
    actor.armature.hide_render = True
    actor.armature.hide_viewport = True
    for mesh in actor.mesh_objects:
        mesh.hide_render = True
        mesh.hide_viewport = True
    return _SourceState(
        source_id=source_id,
        actor=actor,
        snapshots=snapshots,
        final_frame=int(frame_indices[-1]) + 1,
        output_path=Path(source["output_path"]),
        frame_indices=frame_indices,
    )


def _freeze_snapshot_meshes(
    bpy: Any,
    actor: Any,
    *,
    frame: int,
    source_id: str,
    snapshot_index: int,
    snapshot_count: int,
    translation: np.ndarray,
    material_mode: str,
    snapshot_layout: str,
    palette_start_rgb: tuple[int, int, int],
    palette_end_rgb: tuple[int, int, int],
) -> list[Any]:
    from mathutils import Vector  # type: ignore

    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    fraction = snapshot_index / max(1, snapshot_count - 1)
    material = None
    if material_mode == "palette":
        if snapshot_layout == "root_aligned":
            alpha = 1.0
        elif snapshot_layout == "arc":
            alpha = 1.0
        else:
            alpha = 0.12 + 0.25 * fraction
        color = Color(*(_srgb_to_linear(value / 255.0) for value in palette_start_rgb), alpha).mix(
            Color(*(_srgb_to_linear(value / 255.0) for value in palette_end_rgb), alpha),
            fraction,
        )
        material = make_material(
            f"{source_id}_Snapshot_{snapshot_index:02d}_Material",
            color,
            roughness=0.82,
        )
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result: list[Any] = []
    for mesh_index, mesh in enumerate(actor.mesh_objects):
        original_materials = list(mesh.data.materials) if material is None else []
        evaluated = mesh.evaluated_get(depsgraph)
        mesh_data = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
        snapshot = bpy.data.objects.new(
            f"{source_id}_Snapshot_{snapshot_index:02d}_{mesh_index:02d}",
            mesh_data,
        )
        matrix_world = evaluated.matrix_world.copy()
        matrix_world.translation += Vector(tuple(float(value) for value in translation))
        snapshot.matrix_world = matrix_world
        bpy.context.collection.objects.link(snapshot)
        if material is not None:
            snapshot.data.materials.clear()
            snapshot.data.materials.append(material)
            for polygon in snapshot.data.polygons:
                polygon.material_index = 0
        else:
            snapshot.data.materials.clear()
            for original in original_materials:
                snapshot.data.materials.append(original)
        result.append(snapshot)
    return result


def _evaluated_root_location(bpy: Any, actor: Any) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    root = actor.armature.evaluated_get(depsgraph).matrix_world.translation
    return np.asarray((root.x, root.y, root.z), dtype=np.float64)


def snapshot_layout_translation(
    root_location: np.ndarray | tuple[float, float, float],
    *,
    snapshot_index: int,
    snapshot_count: int,
    layout: str,
    spacing: float,
    camera_preset: str = "three_quarter",
    arc_direction: str = "up",
) -> np.ndarray:
    """Return the world-space shift applied to one frozen snapshot.

    Root-aligned layout removes only horizontal locomotion, preserving vertical
    motion such as jumps, then places snapshots along the camera's screen-right
    ground axis. Arc layout uses the same ground plane and bends the sequence
    around a 120-degree arc, keeping all feet at the same height.
    """
    if layout == "trajectory":
        return np.zeros(3, dtype=np.float64)
    if layout not in {"root_aligned", "arc"}:
        raise ValueError(f"Unknown snapshot layout {layout!r}")
    if snapshot_count < 1:
        raise ValueError("snapshot_count must be positive")
    if not 0 <= snapshot_index < snapshot_count:
        raise ValueError("snapshot_index is outside snapshot_count")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if arc_direction not in {"up", "down"}:
        raise ValueError("arc_direction must be 'up' or 'down'")

    root = np.asarray(root_location, dtype=np.float64)
    if root.shape != (3,):
        raise ValueError("root_location must contain three values")
    axis = _camera_ground_right(camera_preset)
    if layout == "root_aligned":
        centered_index = snapshot_index - (snapshot_count - 1) * 0.5
        shift = axis * (centered_index * spacing)
    else:
        # A screen-plane arc fills a square showcase frame and bends upward
        # without changing the pose orientation or the camera-facing scale.
        camera_forward = np.asarray((0.70, -1.0, 0.75), dtype=np.float64)
        camera_forward /= np.linalg.norm(camera_forward)
        screen_up = np.cross(camera_forward, axis)
        screen_up /= np.linalg.norm(screen_up)
        arc_degrees = 120.0
        step = np.deg2rad(arc_degrees / max(1, snapshot_count - 1))
        radius = spacing / max(2.0 * np.sin(step * 0.5), 1e-6)
        angle = np.deg2rad(-arc_degrees * 0.5) + snapshot_index * step
        bend = 1.0 if arc_direction == "up" else -1.0
        shift = radius * (axis * np.sin(angle) + bend * screen_up * np.cos(angle))
        shift -= root
        return shift
    shift[:2] -= root[:2]
    return shift


def _camera_ground_right(camera_preset: str) -> np.ndarray:
    if camera_preset not in {"three_quarter", "arc"}:
        raise ValueError(f"Unsupported qualitative camera preset {camera_preset!r}")
    direction_xy = np.asarray((0.70, -1.0), dtype=np.float64)
    right = np.asarray((-direction_xy[1], direction_xy[0], 0.0), dtype=np.float64)
    return right / np.linalg.norm(right)


def _shared_bounds(bpy: Any, states: list[_SourceState]) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    for state in states:
        bpy.context.scene.frame_set(state.final_frame)
        bpy.context.view_layer.update()
        current_min, current_max = _mesh_bounds(bpy, state.render_objects)
        minimum = np.minimum(minimum, current_min)
        maximum = np.maximum(maximum, current_max)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise RuntimeError("Qualitative scene contains no finite renderable geometry")
    return minimum, maximum


def _mesh_bounds(bpy: Any, objects: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    for mesh in objects:
        evaluated = mesh.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            for vertex in evaluated_mesh.vertices:
                point = evaluated.matrix_world @ vertex.co
                coordinate = np.asarray((point.x, point.y, point.z), dtype=np.float64)
                minimum = np.minimum(minimum, coordinate)
                maximum = np.maximum(maximum, coordinate)
        finally:
            evaluated.to_mesh_clear()
    return minimum, maximum


def _fit_camera_to_objects(
    bpy: Any,
    camera: Any,
    objects: list[Any],
    *,
    resolution: tuple[int, int],
    margin: float,
) -> None:
    """Fit an orthographic camera to actual projected vertices, not AABB corners."""
    rotation = camera.rotation_euler.to_matrix()
    right = rotation.col[0]
    up = rotation.col[1]
    right_min = up_min = np.inf
    right_max = up_max = -np.inf
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for mesh in objects:
        evaluated = mesh.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            for vertex in evaluated_mesh.vertices:
                point = evaluated.matrix_world @ vertex.co
                right_value = _dot_vector3(point, right)
                up_value = _dot_vector3(point, up)
                right_min = min(right_min, right_value)
                right_max = max(right_max, right_value)
                up_min = min(up_min, up_value)
                up_max = max(up_max, up_value)
        finally:
            evaluated.to_mesh_clear()
    if not np.isfinite((right_min, right_max, up_min, up_max)).all():
        raise RuntimeError("Cannot fit qualitative camera without finite mesh vertices")

    right_center = (right_min + right_max) * 0.5
    up_center = (up_min + up_max) * 0.5
    location = camera.location.copy()
    location += right * (right_center - _dot_vector3(location, right))
    location += up * (up_center - _dot_vector3(location, up))
    camera.location = location

    projected_width = right_max - right_min
    projected_height = up_max - up_min
    width_px, height_px = max(1, resolution[0]), max(1, resolution[1])
    if width_px >= height_px:
        fitted_span = max(projected_width, projected_height * width_px / height_px)
    else:
        fitted_span = max(projected_height, projected_width * height_px / width_px)
    camera.data.ortho_scale = max(float(fitted_span) * margin, 1.0)


def _dot_vector3(point: Any, axis: Any) -> float:
    """Dot only the spatial components; Blender evaluated vertices may be 4D."""
    return float(point.x * axis.x + point.y * axis.y + point.z * axis.z)


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _render_source(
    bpy: Any,
    states: list[_SourceState],
    selected: _SourceState,
    *,
    snapshot_alpha: float,
) -> None:
    for state in states:
        visible = state is selected
        for obj in state.render_objects:
            obj.hide_render = not visible
            obj.hide_viewport = not visible
    bpy.context.scene.frame_set(selected.final_frame)
    bpy.context.view_layer.update()
    selected.output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(selected.output_path)
    bpy.ops.render.render(write_still=True)
    if snapshot_alpha < 1.0:
        _apply_snapshot_alpha(bpy, selected.output_path, snapshot_alpha)


def _apply_snapshot_alpha(bpy: Any, path: Path, snapshot_alpha: float) -> None:
    if not 0.0 < snapshot_alpha <= 1.0:
        raise ValueError("snapshot_alpha must be in (0, 1]")
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        pixels[3::4] *= snapshot_alpha
        image.pixels.foreach_set(pixels)
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def _start_root_offset(path: Path) -> tuple[float, float, float]:
    with np.load(path, allow_pickle=False) as payload:
        root = np.asarray(payload["joints22"], dtype=np.float32)[0, 0]
    blender_root = np.asarray((root[0], -root[2], root[1]), dtype=np.float32)
    return tuple(float(value) for value in -blender_root)


def _foot_pose_overrides(path: Path, mode: str) -> dict[str, np.ndarray] | None:
    if mode == "source":
        return None
    with np.load(path, allow_pickle=False) as payload:
        body_pose = override_foot_pose(np.asarray(payload["body_pose"]), mode)
    return {"body_pose": body_pose}


def _configure_scene(
    bpy: Any,
    *,
    resolution: tuple[int, int],
    samples: int,
    snapshot_alpha: float,
) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.engine = _render_engine(bpy)
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples
    if hasattr(scene, "cycles"):
        scene.cycles.samples = samples
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    if not 0.0 < snapshot_alpha <= 1.0:
        raise ValueError("snapshot_alpha must be in (0, 1]")


def _render_engine(bpy: Any) -> str:
    valid = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    if "BLENDER_EEVEE" in valid:
        return "BLENDER_EEVEE"
    if "BLENDER_EEVEE_NEXT" in valid:
        return "BLENDER_EEVEE_NEXT"
    if "CYCLES" in valid:
        return "CYCLES"
    return bpy.context.scene.render.engine


def write_failed_status(bundle_path: str | Path, error: str) -> None:
    try:
        bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    except Exception:
        return
    _write_status(
        bundle,
        {
            "status": "failed",
            "clip_id": bundle.get("clip_id"),
            "error": error,
        },
    )


def _write_status(bundle: dict[str, Any], payload: dict[str, Any]) -> None:
    status_path = Path(bundle["status_path"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
