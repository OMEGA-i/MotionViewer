from __future__ import annotations

import json
import sys
from pathlib import Path


def _bootstrap_package() -> None:
    src = Path(__file__).resolve().parents[2]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_package()

import numpy as np

from motionviewer.blender.addon_probe import probe_smplx_addon
from motionviewer.blender.camera import add_camera_for_bounds
from motionviewer.blender.render import configure_render, render_animation
from motionviewer.blender.scene import add_lighting, add_world_label, clear_scene, setup_world
from motionviewer.blender.smplx_mesh import create_smplx_actor_from_npz
from motionviewer.blender.style import (
    add_freeze_fade,
    add_ghost_snapshots,
    add_prefix_ghost_snapshots,
    apply_actor_material,
    set_actor_visibility,
)
from motionviewer.blender.trajectory_ground import (
    add_contact_patch_geometry,
    add_prefix_transition_marker,
    add_segmented_trajectory_carpet,
    add_trajectory_rectangle_ground,
    add_trajectory_ribbon,
)
from motionviewer.core.coordinates import source_to_blender_points
from motionviewer.core.ground import detect_contact_patches
from motionviewer.core.inplace import (
    freeze_horizontal_root_joints_blender,
    freeze_horizontal_root_transl_source,
)
from motionviewer.core.layout import (
    merge_json_bounds,
    root_aligned_joints,
    start_root_offsets,
)
from motionviewer.core.palette import palette_color, prefix_color_from_spec
from motionviewer.core.schema import CoordinateSystem
from motionviewer.core.smplx_actor import SmplxActor
from motionviewer.video.spec import (
    CameraSpec,
    CameraViewSpec,
    ResolvedCameraView,
    group_views_by_staging,
    resolve_camera_views,
)


def main() -> None:
    bundle_path, staging_filter = _parse_args()
    with bundle_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)

    job = bundle["job"]
    camera_cfg = job.get("camera", {})
    style = job.get("style", {})
    output_dir = Path(job["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    status = probe_smplx_addon()
    camera_spec = _camera_spec_from_job(camera_cfg)
    views = resolve_camera_views(
        camera_spec,
        style_ghost=dict(style.get("ghost", {})),
        ground=dict(job.get("ground", {})),
    )
    if staging_filter is not None:
        views = [view for view in views if view.staging == staging_filter]
        if not views:
            raise SystemExit(f"No camera views match staging={staging_filter!r}")
        _write_json(output_dir / f"blender_status_{staging_filter}.json", status.to_json())
    else:
        _write_json(output_dir / "blender_status.json", status.to_json())

    multi_view = len(views) > 1 or (len(views) == 1 and bool(camera_cfg.get("views")))
    groups = group_views_by_staging(views)
    for staging, views_in_group in groups.items():
        _render_staging_group(
            bundle,
            staging=staging,
            views=views_in_group,
            multi_view=multi_view,
        )


def _render_staging_group(
    bundle: dict,
    *,
    staging: str,
    views: list[ResolvedCameraView],
    multi_view: bool,
) -> None:
    job = bundle["job"]
    task = job.get("task", {})
    task_mode = task.get("mode", "continuation")
    style = job.get("style", {})
    prefix_style = style.get("prefix", {})
    labels_style = style.get("labels", {})
    layout = job.get("layout", {})
    render_cfg = job.get("render", {})
    output_dir = Path(job["output"]["directory"])
    resolution = tuple(render_cfg.get("resolution", [1920, 1080]))

    # Same-staging views share normalized ghost/ground (validated upstream).
    ghost_style = dict(views[0].ghost)
    ground = dict(views[0].ground)

    transparent = bool(render_cfg.get("transparent_background", True))
    clear_scene()
    setup_world(transparent=transparent)
    import bpy  # type: ignore

    requested_frame_count = int(bundle["frames"])
    prepared_inputs = [
        _prepare_input(item, inplace=(staging == "inplace"), frame_count=requested_frame_count)
        for item in bundle["inputs"]
    ]
    offsets, frame_count = _compute_layout(layout, prepared_inputs, requested_frame_count)
    prefix_modes = [item.get("prefix_t") for item in prepared_inputs]
    shared_prefix_t = _shared_prefix_t(prefix_modes)
    prefix_mode = prefix_style.get("mode", "attached")
    world_label_entries: list[tuple[str, tuple[float, float, float]]] = []
    transformed_bounds = []

    for idx, item in enumerate(prepared_inputs):
        input_spec = item["input_spec"]
        seq = item["sequence"]
        label = seq["source"]
        model_color = palette_color(idx, style.get("palette", "paper"))
        prefix_color = prefix_color_from_spec(prefix_style)
        offset = offsets[idx]
        prefix_t = int(item.get("prefix_t") or 0) if task_mode != "text_to_motion" else 0
        transition_gap = int(prefix_style.get("transition_gap_frames", 1)) if prefix_t > 0 else 0
        generated_visible_from = min(prefix_t + 1 + transition_gap, frame_count)
        own_frames = min(int(seq.get("frames", frame_count)), frame_count)
        motion_overrides = item.get("motion_overrides")

        input_body = item.get("input_body")
        backend_id = (input_body or job.get("body", {})).get("backend", "blender_smplx_addon")
        from motionviewer.blender.backend import MaterialPolicy
        from motionviewer.blender.backend_registry import default_backend_registry

        _reg = default_backend_registry()
        apply_mat = _reg.material_policy_for(backend_id) == MaterialPolicy.APPLY_MATERIAL

        if prefix_t > 0 and prefix_mode in {"attached", "marker"}:
            prefix_actor = _create_actor(
                input_spec["path"],
                f"{label}_prefix",
                job,
                offset,
                input_body=input_body,
                motion_overrides=motion_overrides,
            )
            if apply_mat:
                apply_actor_material(
                    prefix_actor, prefix_color, roughness=float(style.get("material_roughness", 0.55))
                )
            set_actor_visibility(
                prefix_actor, visible_from=1, visible_until=prefix_t, frame_start=1, frame_end=frame_count
            )

            actor = _create_actor(
                input_spec["path"],
                label,
                job,
                offset,
                input_body=input_body,
                motion_overrides=motion_overrides,
            )
            if apply_mat:
                apply_actor_material(
                    actor, model_color, roughness=float(style.get("material_roughness", 0.55))
                )
            set_actor_visibility(
                actor,
                visible_from=generated_visible_from,
                visible_until=frame_count,
                frame_start=1,
                frame_end=frame_count,
            )
        else:
            prefix_actor = None
            actor = _create_actor(
                input_spec["path"],
                label,
                job,
                offset,
                input_body=input_body,
                motion_overrides=motion_overrides,
            )
            if apply_mat:
                apply_actor_material(
                    actor, model_color, roughness=float(style.get("material_roughness", 0.55))
                )

        if own_frames < frame_count:
            add_freeze_fade(
                actor,
                freeze_frame=own_frames,
                frame_end=frame_count,
                fade_frames=int(style.get("freeze_fade_frames", 10)),
                fade_alpha=float(style.get("freeze_fade_alpha", 0.25)),
            )

        ghost_count = int(style.get("ghost_snapshots", 0))
        continuation_start = generated_visible_from if prefix_t > 0 else 1
        ghost_start = (
            continuation_start + int(ghost_style.get("warmup_frames", 8))
            if prefix_t > 0
            else continuation_start
        )
        ghost_end = min(frame_count, own_frames)
        if ghost_count > 0 and ghost_style.get("mode", "trail") != "none" and continuation_start < ghost_end:
            add_ghost_snapshots(
                actor,
                frame_start=min(ghost_start, ghost_end),
                frame_end=ghost_end,
                count=ghost_count,
                base_color=model_color,
                ramp=style.get("temporal_ramp", "light_to_dark"),
                ghost_spec=ghost_style,
            )

        if prefix_mode == "attached" and prefix_t > 0 and not ghost_style.get("include_prefix", False):
            if ghost_style.get("mode", "trail") != "none":
                prefix_ghost_count = int(prefix_style.get("ghost_count", 2))
                add_prefix_ghost_snapshots(
                    prefix_actor or actor,
                    frame_start=1,
                    prefix_end_frame=min(prefix_t, frame_count),
                    count=prefix_ghost_count,
                    prefix_color=prefix_color,
                )

        if prefix_mode == "marker" and prefix_style.get("show_marker", True) and prefix_t > 0:
            marker_frame = min(prefix_t, frame_count - 1)
            root = item["aligned_joints"][marker_frame, 0, :] + np.asarray(offset, dtype=np.float32)
            add_prefix_transition_marker(
                tuple(float(v) for v in root),
                color=model_color,
                name=label,
            )

        if labels_style.get("mode", "legend") == "world":
            label_base = item["joints"][0, 0, :] + np.asarray(offset, dtype=np.float32)
            world_label_entries.append((label, (float(label_base[0]), float(label_base[1] - 0.65), 1.95)))

        _add_ground(
            ground,
            item,
            offset,
            model_color,
            prefix_t=prefix_t,
            prefix_style=prefix_style,
            task_mode=task_mode,
            frame_count=frame_count,
        )
        transformed_bounds.append(_actor_motion_bounds(bpy, actor, frame_count))

    mode = layout.get("mode", "single")
    if prefix_mode == "shared" and mode == "overlay" and shared_prefix_t:
        _add_shared_prefix_ghost(bundle["inputs"][0], shared_prefix_t, prefix_color_from_spec(prefix_style))

    scene_min, scene_max = merge_json_bounds(transformed_bounds)
    add_lighting(scene_min, scene_max)

    for view in views:
        camera = add_camera_for_bounds(
            scene_min,
            scene_max,
            preset=view.preset,
            margin=view.margin,
            orthographic=view.orthographic,
            resolution=resolution,
            name=f"MotionViewer_Camera_{view.preset}",
            set_active=True,
        )
        if labels_style.get("mode") == "world":
            # Remove previous world labels before recreating for this camera.
            for obj in list(bpy.data.objects):
                if obj.name.startswith("MotionViewer_Label_"):
                    bpy.data.objects.remove(obj, do_unlink=True)
            for label, location in world_label_entries:
                add_world_label(label, location, camera=camera)

        frames_subdir = view.preset if multi_view else None
        configure_render(
            output_dir=output_dir,
            frames=frame_count,
            fps=float(bundle["fps"]),
            resolution=resolution,
            engine=render_cfg.get("engine", "BLENDER_EEVEE"),
            samples=int(render_cfg.get("samples", 64)),
            frame_format=render_cfg.get("frame_format", "PNG"),
            frames_subdir=frames_subdir,
        )
        render_animation()


def _compute_layout(
    layout: dict,
    prepared_inputs: list[dict],
    frame_count: int,
) -> tuple[list[tuple[float, float, float]], int]:
    mode = layout.get("mode", "single")
    offsets = [(0.0, 0.0, 0.0) for _ in prepared_inputs]
    if mode not in {"single", "overlay"}:
        raise ValueError(f"Unsupported layout.mode {mode!r}; expected 'single' or 'overlay'")
    if layout.get("alignment", "start_root") == "start_root":
        roots = [item["joints"][0, 0, :] for item in prepared_inputs]
        offsets = start_root_offsets(roots, offsets)
    return offsets, frame_count


def _create_actor(
    input_path: str,
    label: str,
    job: dict,
    offset: tuple[float, float, float],
    *,
    input_body: dict | None = None,
    motion_overrides: dict | None = None,
) -> SmplxActor:
    from motionviewer.blender.backend_registry import default_backend_registry

    body = input_body if input_body else job.get("body", {})
    backend_id = body.get("backend", "blender_smplx_addon")
    gender = body.get("gender", "neutral")
    unit_scale = float(job.get("layout", {}).get("unit_scale", 1.0))

    reg = default_backend_registry()
    return reg.create_actor(
        backend_id,
        input_path,
        label=label,
        gender=gender,
        unit_scale=unit_scale,
        layout_offset=offset,
        body_config=body,
        motion_overrides=motion_overrides,
    )


def _actor_motion_bounds(bpy, actor: SmplxActor, frame_count: int) -> tuple[list[float], list[float]]:
    """Measure evaluated actor meshes across the rendered frame range.

    Source joint bounds are only a camera proxy. Imported FBX meshes can have
    a different scale or local origin, so use the geometry that will actually
    be rendered when framing the camera.
    """
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    original_frame = bpy.context.scene.frame_current
    try:
        for frame in range(1, frame_count + 1):
            bpy.context.scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for mesh in actor.mesh_objects:
                evaluated = mesh.evaluated_get(depsgraph)
                evaluated_mesh = evaluated.to_mesh()
                try:
                    for vertex in evaluated_mesh.vertices:
                        world = evaluated.matrix_world @ vertex.co
                        minimum = np.minimum(minimum, (world.x, world.y, world.z))
                        maximum = np.maximum(maximum, (world.x, world.y, world.z))
                finally:
                    evaluated.to_mesh_clear()
    finally:
        bpy.context.scene.frame_set(original_frame)
        bpy.context.view_layer.update()
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise RuntimeError(f"Actor {actor.label!r} has no renderable mesh bounds")
    return minimum.tolist(), maximum.tolist()


def _prepare_input(item: dict, *, inplace: bool, frame_count: int | None = None) -> dict:
    joints = _joints_blender(item)
    if frame_count is not None:
        joints = joints[:frame_count]
    motion_overrides = None
    if inplace:
        joints = freeze_horizontal_root_joints_blender(joints)
        path = Path(item["input"]["path"])
        with np.load(path, allow_pickle=False) as data:
            if "transl" in data.files:
                transl = freeze_horizontal_root_transl_source(np.asarray(data["transl"], dtype=np.float32))
                motion_overrides = {"transl": transl}
    aligned = root_aligned_joints(joints)
    sequence = item["sequence"]
    prefix_t = None
    for segment in sequence.get("segments", []):
        if segment.get("name") == "prefix":
            prefix_t = int(segment.get("end", 0)) - int(segment.get("start", 0))
            break
    if prefix_t is None and "extras" in sequence:
        prefix_t = sequence.get("extras", {}).get("prefix_T")
    blender_bounds = item.get("blender_bounds")
    if inplace or frame_count is not None:
        flat = joints.reshape(-1, 3)
        blender_bounds = {"min": flat.min(axis=0).tolist(), "max": flat.max(axis=0).tolist()}
    return {
        **item,
        "joints": joints,
        "aligned_joints": aligned,
        "input_spec": item["input"],
        "input_body": item["input"].get("body"),
        "prefix_t": prefix_t,
        "motion_overrides": motion_overrides,
        "blender_bounds": blender_bounds or item.get("blender_bounds"),
    }


def _add_ground(
    ground: dict,
    item: dict,
    offset: tuple[float, float, float],
    color,
    *,
    prefix_t: int,
    prefix_style: dict,
    task_mode: str,
    frame_count: int,
) -> None:
    mode = ground.get("mode", "trajectory_carpet")
    if mode == "none":
        return
    joints = item["joints"][:frame_count] + np.asarray(offset, dtype=np.float32)
    floor = float(joints[:, :, 2].min())
    name = item["sequence"]["source"]
    opacity = float(ground.get("opacity", 0.22))

    if mode == "trajectory_rectangle":
        footprint = joints.reshape(-1, 3).copy()
        footprint[:, 2] = floor
        add_trajectory_rectangle_ground(
            footprint,
            floor_z=floor,
            color=color,
            opacity=opacity,
            name=name,
            padding=float(ground.get("carpet_padding", 0.35)),
            min_width=float(ground.get("carpet_width", 0.0)) or 0.55,
        )
    elif mode == "trajectory_carpet":
        root = joints[:, 0, :].copy()
        root[:, 2] = floor
        carpet_width = float(ground.get("carpet_width", 0.0)) or (
            float(ground.get("carpet_padding", 0.35)) * 1.7
        )
        add_segmented_trajectory_carpet(
            root,
            floor_z=floor,
            prefix_t=prefix_t if task_mode != "text_to_motion" else 0,
            prefix_color=prefix_color_from_spec(prefix_style),
            generated_color=color,
            opacity=opacity,
            name=name,
            width=carpet_width,
        )
        if (
            prefix_t > 0
            and task_mode != "text_to_motion"
            and prefix_style.get("show_marker", True)
            and prefix_t < len(joints)
        ):
            add_prefix_transition_marker(
                tuple(float(v) for v in joints[prefix_t, 0, :]),
                color=prefix_color_from_spec(prefix_style),
                name=name,
            )
    elif mode in {"contact_patches", "footprint_trail", "coverage_hull"}:
        patches = detect_contact_patches(
            joints,
            foot_joint_ids=list(ground.get("foot_joint_ids", [10, 11])),
            height_threshold=float(ground.get("height_threshold", 0.045)),
            velocity_threshold=float(ground.get("velocity_threshold", 0.08)),
            patch_radius=float(ground.get("patch_radius", 0.14)),
        )
        add_contact_patch_geometry(patches, color=color, opacity=opacity, name=name)
    elif mode == "trajectory_ribbon":
        root = joints[:, 0, :].copy()
        root[:, 2] = floor
        add_trajectory_ribbon(root, color=color, opacity=opacity, name=name)


def _add_shared_prefix_ghost(item: dict, prefix_t: int, prefix_color) -> None:
    path = Path(item["input"]["path"])
    actor = create_smplx_actor_from_npz(path, label="shared_prefix", gender="neutral")
    add_prefix_ghost_snapshots(
        actor,
        frame_start=1,
        prefix_end_frame=prefix_t,
        count=2,
        prefix_color=prefix_color,
    )


def _joints_blender(item: dict) -> np.ndarray:
    path = Path(item["input"]["path"])
    with np.load(path, allow_pickle=False) as data:
        joints = np.asarray(data["joints22" if "joints22" in data.files else "joints"], dtype=np.float32)
    coord = item["sequence"].get("coordinate_system", {})
    system = CoordinateSystem(
        vertical_axis=int(coord.get("vertical_axis", 1)),
        forward_axis=coord.get("forward_axis", 2),
        units=coord.get("units", "meters"),
    )
    return source_to_blender_points(joints, system)


def _shared_prefix_t(prefix_modes: list[int | None]) -> int:
    values = [int(v) for v in prefix_modes if v]
    return values[0] if values and all(v == values[0] for v in values) else 0


def _camera_spec_from_job(camera_cfg: dict) -> CameraSpec:
    views_raw = camera_cfg.get("views") or []
    views = [CameraViewSpec.from_dict(item) for item in views_raw]
    preset = str(camera_cfg.get("preset", "three_quarter"))
    return CameraSpec(
        preset=preset,  # type: ignore[arg-type]
        orthographic=bool(camera_cfg.get("orthographic", True)),
        margin=float(camera_cfg.get("margin", 1.15)),
        follow_root=bool(camera_cfg.get("follow_root", False)),
        views=views,
    )


def _parse_args() -> tuple[Path, str | None]:
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        args = sys.argv[1:]
    if not args:
        raise SystemExit(
            "Usage: blender --background --python runner.py -- job_bundle.json [--staging world|inplace]"
        )
    bundle = Path(args[0])
    staging = None
    if "--staging" in args:
        idx = args.index("--staging")
        if idx + 1 >= len(args):
            raise SystemExit("--staging requires world|inplace")
        staging = args[idx + 1]
        if staging not in {"world", "inplace"}:
            raise SystemExit(f"Unknown staging {staging!r}")
    return bundle, staging


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


if __name__ == "__main__":
    main()
