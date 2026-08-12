"""Render a filtered grid of retargeted FBX poses from a v2 motion package."""

from __future__ import annotations

import json
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from ..assets.fbx_catalog import FbxModel, is_binary_fbx, list_fbx_models, valid_fbx_models
from ..core.canonical_skeleton import SMPLX_TO_CANONICAL
from ..packages.index import ClipIndex, PackageIndex, build_package_index
from ..packages.store import PackageStore, open_package_store

FrameMode = Literal["first", "random", "index"]
QualityFilter = Literal["retarget", "off"]
FbxMode = Literal["fixed", "random-grid", "random-cell"]
FbxPool = Literal["approved", "binary"]
GroundStyle = Literal["grid", "solid"]
MaterialMode = Literal["preserve", "clay"]
LayoutStyle = Literal["grid", "scatter", "scatter-shallow"]

DEFAULT_BACKGROUND_RGB = (188, 170, 221)
DEFAULT_VIEWS = ("upper_left", "front", "upper_right")
VALID_VIEWS = frozenset({"upper_left", "front", "upper_right", "perspective_front", "showcase", "side"})


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


_LIMB_SEGMENTS = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_upper_leg": ("left_hip", "left_knee"),
    "left_lower_leg": ("left_knee", "left_ankle"),
    "right_upper_leg": ("right_hip", "right_knee"),
    "right_lower_leg": ("right_knee", "right_ankle"),
}


class PoseGridError(RuntimeError):
    """Base error for pose-grid preparation and rendering."""


class InsufficientPoseGridSamples(PoseGridError):
    """Raised after all eligible samples cannot fill the requested grid."""


@dataclass(frozen=True)
class PoseGridRequest:
    package: Path
    fbx: Path | None = None
    output: Path | None = None
    output_dir: Path | None = None
    source_id: str = "gt"
    task: str | None = None
    provenances: tuple[str, ...] = ()
    caption_regex: str | None = None
    min_frames: int | None = None
    max_frames: int | None = None
    count: int = 20
    frame_mode: FrameMode = "random"
    frame_index: int | None = None
    max_attempts: int = 0
    bone_map: str = "auto"
    quality_filter: QualityFilter = "retarget"
    max_penetration_cm: float = 1.0
    max_limb_error_deg: float = 35.0
    columns: int = 0
    spacing: float = 1.5
    seed: int = 0
    resolution: tuple[int, int] = (1600, 1600)
    views: tuple[str, ...] = DEFAULT_VIEWS
    fbx_mode: FbxMode = "fixed"
    fbx_root: Path = Path("assets/fbx")
    fbx_pool: FbxPool = "approved"
    fbx_model_ids: tuple[str, ...] = ()
    batch_count: int = 1
    background_rgb: tuple[int, int, int] = DEFAULT_BACKGROUND_RGB
    background_rgbs: tuple[tuple[int, int, int], ...] = ()
    ground_style: GroundStyle = "grid"
    material_mode: MaterialMode = "preserve"
    layout_style: LayoutStyle = "grid"

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be > 0")
        if self.batch_count <= 0:
            raise ValueError("batch_count must be > 0")
        if self.fbx_mode not in ("fixed", "random-grid", "random-cell"):
            raise ValueError(f"Unknown fbx_mode {self.fbx_mode!r}")
        if self.fbx_pool not in ("approved", "binary"):
            raise ValueError(f"Unknown fbx_pool {self.fbx_pool!r}")
        if len(set(self.fbx_model_ids)) != len(self.fbx_model_ids):
            raise ValueError("fbx_model_ids must not contain duplicates")
        if self.fbx_mode == "fixed" and self.fbx is None:
            raise ValueError("fbx is required when fbx_mode='fixed'")
        if self.fbx_mode != "fixed" and self.fbx is not None:
            raise ValueError("fbx must be omitted when using a random fbx_mode")
        if self.frame_mode not in ("first", "random", "index"):
            raise ValueError(f"Unknown frame_mode {self.frame_mode!r}")
        if self.frame_mode == "index" and self.frame_index is None:
            raise ValueError("frame_index is required when frame_mode='index'")
        if self.frame_mode != "index" and self.frame_index is not None:
            raise ValueError("frame_index is only valid when frame_mode='index'")
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")
        if self.min_frames is not None and self.min_frames <= 0:
            raise ValueError("min_frames must be > 0")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be > 0")
        if self.min_frames is not None and self.max_frames is not None and self.min_frames > self.max_frames:
            raise ValueError("min_frames cannot exceed max_frames")
        if self.columns < 0:
            raise ValueError("columns must be >= 0")
        if self.spacing <= 0:
            raise ValueError("spacing must be > 0")
        if self.max_penetration_cm < 0 or self.max_limb_error_deg <= 0:
            raise ValueError("quality thresholds must be positive")
        if min(self.resolution) <= 0:
            raise ValueError("resolution values must be > 0")
        if len(self.background_rgb) != 3 or any(not 0 <= int(value) <= 255 for value in self.background_rgb):
            raise ValueError("background_rgb must contain three values in [0, 255]")
        if any(
            len(color) != 3 or any(not 0 <= int(value) <= 255 for value in color)
            for color in self.background_rgbs
        ):
            raise ValueError("background_rgbs must contain RGB triples with values in [0, 255]")
        if len(set(self.background_rgbs)) != len(self.background_rgbs):
            raise ValueError("background_rgbs must not contain duplicates")
        if not self.views or any(view not in VALID_VIEWS for view in self.views):
            raise ValueError(f"views must be a non-empty subset of {sorted(VALID_VIEWS)}")
        if len(set(self.views)) != len(self.views):
            raise ValueError("views must not contain duplicates")
        if self.ground_style not in ("grid", "solid"):
            raise ValueError("ground_style must be 'grid' or 'solid'")
        if self.material_mode not in ("preserve", "clay"):
            raise ValueError("material_mode must be 'preserve' or 'clay'")
        if self.layout_style not in ("grid", "scatter", "scatter-shallow"):
            raise ValueError("layout_style must be 'grid', 'scatter', or 'scatter-shallow'")
        if self.caption_regex is not None:
            re.compile(self.caption_regex)
        if self.output is None and self.output_dir is None:
            raise ValueError("output_dir is required")
        if self.output_dir is None and self.output is not None:
            object.__setattr__(self, "output_dir", self.output.parent / self.output.stem)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["package"] = str(self.package)
        payload["fbx"] = None if self.fbx is None else str(self.fbx)
        payload["fbx_root"] = str(self.fbx_root)
        payload["output"] = None if self.output is None else str(self.output)
        payload["output_dir"] = str(self.output_dir)
        payload["provenances"] = list(self.provenances)
        payload["resolution"] = list(self.resolution)
        payload["views"] = list(self.views)
        payload["fbx_model_ids"] = list(self.fbx_model_ids)
        payload["background_rgb"] = list(self.background_rgb)
        payload["background_rgbs"] = [list(color) for color in self.background_rgbs]
        return payload

    @property
    def render_background_rgbs(self) -> tuple[tuple[int, int, int], ...]:
        """Return the explicit palette, or the legacy single background color."""
        return self.background_rgbs or (self.background_rgb,)


def parse_background_color(value: str | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` or ``R,G,B`` into an immutable RGB tuple."""
    if isinstance(value, str):
        raw = value.strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}", raw):
            return tuple(int(raw[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]
        parts = raw.split(",")
    else:
        parts = list(value)
    try:
        result = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError("background color must be #RRGGBB or R,G,B") from exc
    if len(result) != 3 or any(value < 0 or value > 255 for value in result):
        raise ValueError("background color must contain three values in [0, 255]")
    return result  # type: ignore[return-value]


def choose_grid_fbx_models(
    request: PoseGridRequest,
    *,
    rng: random.Random,
    eligible: list[FbxModel] | None = None,
    grid_model: FbxModel | None = None,
) -> list[FbxModel | None]:
    """Choose deterministic model assignments for one grid."""
    if request.fbx_mode == "fixed":
        return [None] * request.count
    models = list(eligible) if eligible is not None else valid_fbx_models(request.fbx_root)
    if not models:
        raise PoseGridError(f"No approved random-eligible FBX models under {request.fbx_root}")
    shuffled = list(models)
    rng.shuffle(shuffled)
    if request.fbx_mode == "random-grid":
        return [grid_model or shuffled[0]] * request.count
    return [shuffled[index % len(shuffled)] for index in range(request.count)]


@dataclass(frozen=True)
class PoseGridCandidate:
    clip: ClipIndex
    asset_relpath: PurePosixPath


@dataclass
class PoseGridSample:
    clip_id: str
    dir_name: str
    task: str
    source_id: str
    provenance: str | None
    caption: str | None
    source_frames: int
    fbx_model_id: str | None = None
    frame_index: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PoseGridReport:
    request: PoseGridRequest
    candidate_count: int
    attempted_count: int = 0
    accepted: list[PoseGridSample] = field(default_factory=list)
    rejected: list[PoseGridSample] = field(default_factory=list)
    output: Path | None = None
    view_outputs: dict[str, Path] = field(default_factory=dict)
    fbx_model_id: str | None = None
    batch_index: int = 0
    selection_path: Path | None = None
    fbx_import_count: int = 0
    elapsed_seconds: float = 0.0
    status: str = "pending"

    def to_json(self) -> dict[str, Any]:
        return {
            "request": self.request.to_json(),
            "candidate_count": self.candidate_count,
            "attempted_count": self.attempted_count,
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "accepted": [sample.to_json() for sample in self.accepted],
            "rejected": [sample.to_json() for sample in self.rejected],
            "output": str(self.output) if self.output is not None else None,
            "view_outputs": {key: str(value) for key, value in self.view_outputs.items()},
            "fbx_model_id": self.fbx_model_id,
            "batch_index": self.batch_index,
            "selection_path": str(self.selection_path) if self.selection_path else None,
            "fbx_import_count": self.fbx_import_count,
            "elapsed_seconds": self.elapsed_seconds,
            "status": self.status,
        }


@dataclass
class PoseGridBatchReport:
    request: PoseGridRequest
    grids: list[PoseGridReport] = field(default_factory=list)
    status: str = "pending"
    manifest: Path | None = None

    @property
    def accepted(self) -> list[PoseGridSample]:
        return [sample for grid in self.grids for sample in grid.accepted]

    def to_json(self) -> dict[str, Any]:
        return {
            "request": self.request.to_json(),
            "status": self.status,
            "manifest": str(self.manifest) if self.manifest else None,
            "grids": [grid.to_json() for grid in self.grids],
        }


def select_pose_grid_candidates(index: PackageIndex, request: PoseGridRequest) -> list[PoseGridCandidate]:
    """Return metadata-eligible clips with the requested SMPL-X source."""
    if request.task is not None and request.task not in index.tasks:
        available = ", ".join(sorted(index.tasks))
        raise PoseGridError(f"Unknown task {request.task!r}. Available: {available}")

    caption_pattern = (
        re.compile(request.caption_regex, flags=re.IGNORECASE) if request.caption_regex else None
    )
    candidates: list[PoseGridCandidate] = []
    for clip in (
        index.clips_for_task(request.task) if request.task else (index.clips[key] for key in index.clip_order)
    ):
        if request.provenances and clip.provenance not in request.provenances:
            continue
        if request.min_frames is not None and clip.frames < request.min_frames:
            continue
        if request.max_frames is not None and clip.frames > request.max_frames:
            continue
        if caption_pattern and not caption_pattern.search(clip.caption or ""):
            continue
        source = clip.gt if request.source_id == "gt" else clip.sources.get(request.source_id)
        if source is None or "smplx_params" not in source.variants:
            continue
        candidates.append(PoseGridCandidate(clip=clip, asset_relpath=source.variants["smplx_params"].relpath))
    if not candidates:
        raise PoseGridError("No package clips matched the requested source and filters")
    return candidates


def shuffled_candidates(
    candidates: list[PoseGridCandidate],
    request: PoseGridRequest,
    *,
    rng: random.Random | None = None,
) -> list[PoseGridCandidate]:
    """Return a deterministic random ordering without duplicate clips."""
    result = list(candidates)
    (rng or random.Random(request.seed)).shuffle(result)
    return result[: request.max_attempts] if request.max_attempts else result


def load_single_frame_payload(
    raw: bytes,
    request: PoseGridRequest,
    rng: random.Random,
) -> tuple[dict[str, np.ndarray], int, int]:
    """Read a package NPZ and return a one-frame retarget payload."""
    with np.load(BytesIO(raw), allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    if "global_orient" not in payload:
        raise PoseGridError("SMPL-X payload is missing global_orient")
    frames = int(payload["global_orient"].shape[0])
    if frames <= 0:
        raise PoseGridError("SMPL-X payload has no frames")
    if request.frame_mode == "first":
        frame_index = 0
    elif request.frame_mode == "index":
        assert request.frame_index is not None
        if request.frame_index >= frames:
            raise PoseGridError(
                f"Requested frame {request.frame_index} is outside payload with {frames} frames"
            )
        frame_index = request.frame_index
    else:
        frame_index = rng.randrange(frames)

    result: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key != "betas" and value.ndim >= 1 and value.shape[0] == frames:
            result[key] = value[frame_index : frame_index + 1]
        else:
            result[key] = value
    return result, frame_index, frames


def write_pose_grid_report(report: PoseGridReport) -> Path:
    if report.selection_path is not None:
        path = report.selection_path.resolve()
    elif report.request.output is not None:
        path = report.request.output.resolve().with_suffix(".selection.json")
    else:
        path = (report.request.output_dir / "selection.json").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    report.selection_path = path
    return path


def render_pose_grid(request: PoseGridRequest) -> PoseGridBatchReport:
    """Render one or more deterministic pose grids through one Blender process."""
    import bpy  # type: ignore

    output_dir = Path(request.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    store = open_package_store(request.package)
    batch = PoseGridBatchReport(request=request)
    try:
        index = build_package_index(store)
        all_candidates = select_pose_grid_candidates(index, request)
        eligible = _eligible_fbx_models(request) if request.fbx_mode != "fixed" else []
        manifest = output_dir / "manifest.json"
        batch.manifest = manifest
        for batch_index in range(request.batch_count):
            grid_seed = request.seed + batch_index * 1_000_003
            rng = random.Random(grid_seed)
            grid_model = rng.choice(eligible) if request.fbx_mode == "random-grid" and eligible else None
            model_assignments = choose_grid_fbx_models(
                request, rng=rng, eligible=eligible, grid_model=grid_model
            )
            grid = _render_one_grid(
                bpy,
                store,
                all_candidates,
                request,
                batch_index=batch_index,
                rng=rng,
                model_assignments=model_assignments,
                output_dir=output_dir,
            )
            batch.grids.append(grid)
            batch.status = "rendering"
            manifest.write_text(json.dumps(batch.to_json(), indent=2), encoding="utf-8")
        batch.status = "rendered"
        manifest.write_text(json.dumps(batch.to_json(), indent=2), encoding="utf-8")
        return batch
    finally:
        store.close()


def _eligible_fbx_models(request: PoseGridRequest) -> list[FbxModel]:
    if request.fbx_pool == "approved":
        models = valid_fbx_models(request.fbx_root)
    else:
        models = [model for model in list_fbx_models(request.fbx_root) if is_binary_fbx(model.path)]
    if not request.fbx_model_ids:
        return models
    by_id = {model.model_id: model for model in models}
    missing = [model_id for model_id in request.fbx_model_ids if model_id not in by_id]
    if missing:
        raise PoseGridError(
            f"Requested FBX model ids are unavailable in the {request.fbx_pool} pool: {', '.join(missing)}"
        )
    return [by_id[model_id] for model_id in request.fbx_model_ids]


def _render_one_grid(
    bpy: Any,
    store: PackageStore,
    all_candidates: list[PoseGridCandidate],
    request: PoseGridRequest,
    *,
    batch_index: int,
    rng: random.Random,
    model_assignments: list[FbxModel | None],
    output_dir: Path,
) -> PoseGridReport:
    from .camera import add_camera_for_bounds
    from .retarget._resolve import resolve_bone_mapping
    from .retarget.pipeline import create_fbx_actor_from_npz
    from .scene import add_lighting, clear_scene, setup_world

    started = time.perf_counter()
    model_label = "mixed" if request.fbx_mode == "random-cell" else _model_label(request, model_assignments)
    grid_dir = output_dir / f"grid_{batch_index:03d}_{model_label}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    report = PoseGridReport(
        request=request,
        candidate_count=len(all_candidates),
        batch_index=batch_index,
        fbx_model_id=None if model_label == "mixed" else model_label,
    )
    clear_scene()
    setup_world(transparent=False, background_rgb=request.render_background_rgbs[0])
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = request.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    columns = request.columns or int(
        np.ceil(np.sqrt(request.count * (0.72 if request.layout_style == "scatter" else 1.0)))
    )
    candidates = shuffled_candidates(all_candidates, request, rng=rng)
    mesh_objects: list[Any] = []
    actors: list[Any] = []
    with tempfile.TemporaryDirectory(prefix=".pose-grid-", dir=grid_dir) as temp_dir:
        temp_root = Path(temp_dir)
        # Gather one-frame payloads first, then retarget each distinct FBX once.
        # This keeps Blender's expensive FBX import and SMPL-X construction out
        # of the per-cell loop while preserving deterministic sampling.
        pending: list[tuple[PoseGridSample, dict[str, np.ndarray], FbxModel | None]] = []
        for candidate in candidates:
            if len(pending) == request.count:
                break
            report.attempted_count += 1
            sample = _sample_metadata(candidate, request.source_id)
            model = model_assignments[len(pending)]
            sample.fbx_model_id = (
                model.model_id if model is not None else _model_label(request, model_assignments)
            )
            try:
                payload, frame_index, source_frames = load_single_frame_payload(
                    store.read_bytes(candidate.asset_relpath), request, rng
                )
                sample.frame_index = frame_index
                sample.source_frames = source_frames
                pending.append((sample, payload, model))
            except Exception as exc:
                sample.reason = f"payload_error: {exc}"
                report.rejected.append(sample)

        groups: dict[str, list[tuple[int, PoseGridSample, dict[str, np.ndarray], FbxModel | None]]] = {}
        for index, (sample, payload, model) in enumerate(pending):
            key = str(request.fbx if model is None else model.path)
            groups.setdefault(key, []).append((index, sample, payload, model))
        for group_index, (fbx_key, group) in enumerate(groups.items()):
            fbx_path = Path(fbx_key)
            label = f"PoseGrid_{batch_index:03d}_{group_index:02d}"
            payload_path = temp_root / f"group_{group_index:02d}.npz"
            _save_combined_payload(payload_path, [item[2] for item in group])
            before = set(bpy.data.objects)
            try:
                actor = create_fbx_actor_from_npz(
                    payload_path,
                    label=label,
                    fbx_path=fbx_path,
                    bone_map=request.bone_map,
                    motion_overrides={"transl": np.zeros((len(group), 3), dtype=np.float32)},
                )
                report.fbx_import_count += 1
                source = bpy.data.objects.get(f"{label}_SMPLX_Driver")
                if source is None:
                    raise PoseGridError("Retarget source driver was not created")
                mapping = resolve_bone_mapping(request.bone_map, fbx_armature=actor.armature).smplx_to_fbx
                if request.material_mode == "clay":
                    _apply_clay_material(bpy, actor)
                for frame_index, (_pending_index, sample, _payload, _model) in enumerate(group):
                    scene.frame_set(frame_index + 1)
                    bpy.context.view_layer.update()
                    if request.quality_filter == "retarget":
                        metrics, reason = _retarget_quality(
                            bpy,
                            actor,
                            source,
                            mapping,
                            max_penetration_cm=request.max_penetration_cm,
                            max_limb_error_deg=request.max_limb_error_deg,
                        )
                    else:
                        metrics, reason = _mesh_metrics(bpy, actor.mesh_objects), None
                    sample.metrics = metrics
                    if reason is not None:
                        sample.reason = reason
                        report.rejected.append(sample)
                        continue
                    static_actor = _freeze_actor_frame(bpy, actor, label=f"{label}_Cell_{frame_index:03d}")
                    actors.append(static_actor)
                    mesh_objects.extend(static_actor.mesh_objects)
                    report.accepted.append(sample)
                # The animated source actor and driver are no longer needed once
                # every accepted frame has become an ordinary static mesh.
                preserved = {actor_obj.armature for actor_obj in actors}
                preserved.update(mesh for item in actors for mesh in item.mesh_objects)
                _remove_created_objects_except(bpy, before, preserved)
            except Exception as exc:
                _remove_created_objects(bpy, before)
                for _local_index, sample, _payload, _model in group:
                    sample.reason = f"retarget_error: {exc}"
                    report.rejected.append(sample)
                if _is_fbx_configuration_error(exc):
                    report.status = "configuration_error"
                    write_pose_grid_report(report)
                    raise PoseGridError(f"FBX retarget configuration failed: {exc}") from exc
    if len(report.accepted) != request.count:
        report.status = "insufficient_samples"
        report.selection_path = grid_dir / "selection.json"
        write_pose_grid_report(report)
        raise InsufficientPoseGridSamples(
            f"Only {len(report.accepted)} of {request.count} samples passed; report: {report.selection_path}"
        )

    if request.layout_style in {"scatter", "scatter-shallow"}:
        shallow = request.layout_style == "scatter-shallow"
        offsets = _scatter_layout_offsets(
            bpy,
            actors,
            request.spacing,
            rng,
            columns=columns,
            width_scale=1.45 if shallow else 1.25,
            depth_scale=1.45 if shallow else 2.25,
            near_width_fraction=0.55 if shallow else 0.35,
            depth_exponent=1.0 if shallow else 1.12,
        )
        for actor in actors:
            actor.armature.rotation_euler[2] += rng.uniform(-np.pi, np.pi)
    else:
        offsets = _auto_layout_offsets(bpy, actors, columns, request.spacing)
    for actor, offset in zip(actors, offsets):
        actor.armature.location.x += float(offset[0])
        actor.armature.location.y += float(offset[1])
    bpy.context.view_layer.update()
    minimum, maximum = _mesh_bounds(bpy, mesh_objects)
    _add_ground(
        bpy,
        minimum,
        maximum,
        request.render_background_rgbs[0],
        request.spacing * 0.7,
        style=request.ground_style,
    )
    add_lighting(minimum.tolist(), maximum.tolist())
    _tune_pose_grid_lighting(bpy)
    cameras = {
        view: add_camera_for_bounds(
            minimum.tolist(),
            maximum.tolist(),
            preset=view,
            margin=1.12,
            orthographic=view != "perspective_front",
            resolution=request.resolution,
            name=f"PoseGrid_Camera_{view}",
        )
        for view in request.views
    }
    multiple_backgrounds = len(request.render_background_rgbs) > 1
    for color in request.render_background_rgbs:
        color_id = _color_id(color)
        _set_pose_grid_background(bpy, color)
        for view, camera in cameras.items():
            if multiple_backgrounds:
                color_dir = output_dir / "colors" / color_id
                color_dir.mkdir(parents=True, exist_ok=True)
                output = color_dir / f"grid_{batch_index:03d}_{model_label}_{view}.png"
                output_key = f"{color_id}/{view}"
            else:
                output = grid_dir / f"{view}.png"
                output_key = view
            scene.camera = camera
            scene.render.filepath = str(output)
            bpy.ops.render.render(write_still=True)
            report.view_outputs[output_key] = output
    report.output = next(iter(report.view_outputs.values()))
    report.selection_path = grid_dir / "selection.json"
    report.elapsed_seconds = time.perf_counter() - started
    report.status = "rendered"
    write_pose_grid_report(report)
    return report


def _save_combined_payload(path: Path, payloads: list[dict[str, np.ndarray]]) -> None:
    """Concatenate one-frame SMPL-X payloads into one retarget job."""
    if not payloads:
        raise PoseGridError("Cannot build an empty pose-grid payload")
    keys = set().union(*(payload.keys() for payload in payloads))
    combined: dict[str, np.ndarray] = {}
    for key in keys:
        values = [payload[key] for payload in payloads if key in payload]
        first = values[0]
        if first.ndim >= 1 and first.shape[0] == 1 and all(value.shape == first.shape for value in values):
            combined[key] = np.concatenate(values, axis=0)
        else:
            combined[key] = first
    np.savez(path, **combined)


def _freeze_actor_frame(bpy: Any, actor: Any, *, label: str) -> Any:
    """Copy evaluated meshes at the current frame and parent them to an Empty."""
    from types import SimpleNamespace

    empty = bpy.data.objects.new(label, None)
    bpy.context.collection.objects.link(empty)
    static_meshes: list[Any] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for index, mesh in enumerate(actor.mesh_objects):
        evaluated = mesh.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            mesh_data = evaluated_mesh.copy()
            static = bpy.data.objects.new(f"{label}_{index:02d}", mesh_data)
            bpy.context.collection.objects.link(static)
            for material in mesh.data.materials:
                static.data.materials.append(material)
            world_matrix = evaluated.matrix_world.copy()
            static.parent = empty
            static.matrix_world = world_matrix
            static_meshes.append(static)
        finally:
            evaluated.to_mesh_clear()
    return SimpleNamespace(label=label, armature=empty, mesh_objects=static_meshes)


def _sample_metadata(candidate: PoseGridCandidate, source_id: str) -> PoseGridSample:
    clip = candidate.clip
    return PoseGridSample(
        clip_id=clip.clip_id,
        dir_name=clip.dir_name,
        task=clip.task,
        source_id=source_id,
        provenance=clip.provenance,
        caption=clip.caption,
        source_frames=clip.frames,
    )


def _model_label(request: PoseGridRequest, assignments: list[FbxModel | None]) -> str:
    if request.fbx_mode == "fixed":
        return request.fbx.stem.lower().replace(" ", "_") if request.fbx else "fixed"
    model = next((item for item in assignments if item is not None), None)
    return model.model_id if model is not None else "unknown"


def _auto_layout_offsets(
    bpy: Any,
    actors: list[Any],
    columns: int,
    gap: float,
) -> list[tuple[float, float, float]]:
    """Lay out actors from measured bounds instead of assuming one avatar size."""
    bounds = []
    for actor in actors:
        minimum, maximum = _mesh_bounds(bpy, list(actor.mesh_objects))
        bounds.append((minimum, maximum))
    if not bounds:
        return []
    widths = [float(maximum[0] - minimum[0]) for minimum, maximum in bounds]
    depths = [float(maximum[1] - minimum[1]) for minimum, maximum in bounds]
    rows = int(np.ceil(len(bounds) / columns))
    column_widths = [
        max(widths[index] for index in range(column, len(bounds), columns)) for column in range(columns)
    ]
    row_depths = [
        max(depths[index] for index in range(row * columns, min(len(bounds), (row + 1) * columns)))
        for row in range(rows)
    ]
    x_centers = []
    cursor = -sum(column_widths) / 2.0 - gap * (columns - 1) / 2.0
    for width in column_widths:
        x_centers.append(cursor + width / 2.0)
        cursor += width + gap
    y_centers = []
    cursor = sum(row_depths) / 2.0 + gap * (rows - 1) / 2.0
    for depth in row_depths:
        y_centers.append(cursor - depth / 2.0)
        cursor -= depth + gap
    # Stagger alternating rows so oblique cameras do not collapse the scene
    # into rigid columns. This keeps deterministic AABB spacing while reading
    # more like the naturally scattered groups used in paper figures.
    typical_pitch = float(np.median(column_widths)) + gap
    row_stagger = typical_pitch * 0.28
    offsets = []
    for index in range(len(bounds)):
        row = index // columns
        stagger = (-row_stagger if row % 2 == 0 else row_stagger) * 0.5
        offsets.append((x_centers[index % columns] + stagger, y_centers[row], 0.0))
    return offsets


def _scatter_layout_offsets(
    bpy: Any,
    actors: list[Any],
    gap: float,
    rng: random.Random,
    *,
    columns: int,
    width_scale: float = 1.5,
    depth_scale: float = 1.85,
    near_width_fraction: float = 0.25,
    depth_exponent: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Place actors randomly on the ground with rotation-safe clearance."""
    if not actors:
        return []
    bounds = [_mesh_bounds(bpy, list(actor.mesh_objects)) for actor in actors]
    centers = [(minimum + maximum) * 0.5 for minimum, maximum in bounds]
    radii = [
        max(0.2, 0.5 * float(np.hypot(maximum[0] - minimum[0], maximum[1] - minimum[1])))
        for minimum, maximum in bounds
    ]
    clearances = [gap * rng.uniform(0.72, 1.28) for _actor in actors]
    rows = int(np.ceil(len(actors) / columns))
    pitch = 2.0 * float(np.median(radii)) + gap
    field_width = max(columns * pitch, 2.0 * max(radii) + gap)
    field_width *= width_scale
    # A sparse, elongated depth axis creates readable near/far scale changes
    # even though the layout uses fewer nominal rows.
    field_depth = max(rows * pitch * depth_scale, 2.0 * max(radii) + gap)
    order = sorted(range(len(actors)), key=lambda index: radii[index], reverse=True)

    placements: dict[int, tuple[float, float]] = {}
    for _expansion in range(10):
        placements.clear()
        for actor_index in order:
            radius = radii[actor_index]
            half_depth = max(radius, field_depth * 0.5 - radius)
            for _attempt in range(2500):
                # The depth exponent controls row density independently from
                # the wedge width. Values above one move some actors toward
                # the front without shortening the overall footprint.
                raw_depth = rng.random()
                depth_fraction = raw_depth**depth_exponent
                y = -half_depth + depth_fraction * (2.0 * half_depth)
                near_width = field_width * near_width_fraction
                width_at_depth = near_width + (field_width - near_width) * depth_fraction
                half_width = max(radius, width_at_depth * 0.5 - radius)
                x = rng.uniform(-half_width, half_width)
                if all(
                    np.hypot(x - other_x, y - other_y)
                    >= radius + radii[other_index] + 0.5 * (clearances[actor_index] + clearances[other_index])
                    for other_index, (other_x, other_y) in placements.items()
                ):
                    placements[actor_index] = (x, y)
                    break
            else:
                break
        if len(placements) == len(actors):
            break
        field_width *= 1.14
        field_depth *= 1.14
    if len(placements) != len(actors):
        raise PoseGridError("Could not place all actors without overlap")

    left = min(placements[index][0] - radii[index] for index in placements)
    right = max(placements[index][0] + radii[index] for index in placements)
    back = min(placements[index][1] - radii[index] for index in placements)
    front = max(placements[index][1] + radii[index] for index in placements)
    center_x = (left + right) * 0.5
    center_y = (back + front) * 0.5
    return [
        (
            placements[index][0] - center_x - float(centers[index][0]),
            placements[index][1] - center_y - float(centers[index][1]),
            0.0,
        )
        for index in range(len(actors))
    ]


def _apply_clay_material(bpy: Any, actor: Any) -> None:
    material = bpy.data.materials.get("PoseGrid_Clay") or bpy.data.materials.new("PoseGrid_Clay")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.78, 0.76, 0.84, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.72
    for mesh in actor.mesh_objects:
        mesh.data.materials.clear()
        mesh.data.materials.append(material)


def _tune_pose_grid_lighting(bpy: Any) -> None:
    """Use a directional white key so actors cast readable soft shadows."""
    settings = {
        "Key_Light": (0.60, 0.30),
        "Fill_Light": (0.10, 0.85),
        "Rim_Light": (0.12, 0.65),
    }
    for obj in bpy.data.objects:
        if getattr(obj, "type", None) == "LIGHT":
            energy_scale, size_scale = settings.get(obj.name, (0.20, 0.75))
            obj.data.energy *= energy_scale
            obj.data.size *= size_scale
            obj.data.color = (1.0, 1.0, 1.0)
            obj.data.use_shadow = True


def _grid_offset(index: int, columns: int, count: int, spacing: float) -> tuple[float, float, float]:
    rows = int(np.ceil(count / columns))
    row, column = divmod(index, columns)
    return (
        (column - (columns - 1) / 2.0) * spacing,
        ((rows - 1) / 2.0 - row) * spacing,
        0.0,
    )


def _remove_created_objects(bpy: Any, before: set[Any]) -> None:
    for obj in [item for item in bpy.data.objects if item not in before]:
        bpy.data.objects.remove(obj, do_unlink=True)


def _remove_created_objects_except(bpy: Any, before: set[Any], preserved: set[Any]) -> None:
    for obj in [item for item in bpy.data.objects if item not in before and item not in preserved]:
        bpy.data.objects.remove(obj, do_unlink=True)


def _remove_non_actor_objects(bpy: Any, before: set[Any], actor: Any) -> None:
    retained = {actor.armature, *actor.mesh_objects}
    for obj in [item for item in bpy.data.objects if item not in before and item not in retained]:
        bpy.data.objects.remove(obj, do_unlink=True)


def _is_fbx_configuration_error(exc: Exception) -> bool:
    message = str(exc)
    return any(
        marker in message
        for marker in (
            "ASCII FBX",
            "No armature found in FBX",
            "Invalid Mixamo body22 mapping",
            "Unknown bone map",
            "does not exist",
        )
    )


def _mesh_metrics(bpy: Any, mesh_objects: list[Any]) -> dict[str, Any]:
    minimum, maximum = _mesh_bounds(bpy, mesh_objects)
    return {
        "minimum_mesh_z_m": float(minimum[2]),
        "mesh_bounds": {"min": minimum.tolist(), "max": maximum.tolist()},
        "finite_mesh": True,
    }


def _mesh_bounds(bpy: Any, mesh_objects: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    for mesh in mesh_objects:
        evaluated = mesh.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            for vertex in evaluated_mesh.vertices:
                point = evaluated.matrix_world @ vertex.co
                coordinates = np.asarray((point.x, point.y, point.z), dtype=np.float64)
                if not np.isfinite(coordinates).all():
                    raise PoseGridError("Evaluated FBX mesh contains non-finite coordinates")
                minimum = np.minimum(minimum, coordinates)
                maximum = np.maximum(maximum, coordinates)
        finally:
            evaluated.to_mesh_clear()
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise PoseGridError("Retargeted actor has no renderable mesh geometry")
    return minimum, maximum


def _retarget_quality(
    bpy: Any,
    actor: Any,
    source: Any,
    mapping: dict[str, str],
    *,
    max_penetration_cm: float,
    max_limb_error_deg: float,
) -> tuple[dict[str, Any], str | None]:
    required = tuple(SMPLX_TO_CANONICAL)
    missing = [name for name in required if name not in mapping or not mapping[name]]
    targets = [mapping[name] for name in required if name in mapping]
    duplicates = sorted({name for name in targets if targets.count(name) > 1})
    absent = [
        f"{source_name}->{target_name}"
        for source_name, target_name in mapping.items()
        if target_name not in actor.armature.pose.bones
    ]
    if missing or duplicates or absent:
        details = []
        if missing:
            details.append(f"missing mappings: {', '.join(missing)}")
        if duplicates:
            details.append(f"duplicate targets: {', '.join(duplicates)}")
        if absent:
            details.append(f"missing target bones: {', '.join(absent)}")
        return {
            "bone_coverage": {"mapped": len(mapping), "expected": len(required), "complete": False}
        }, "; ".join(details)

    metrics = _mesh_metrics(bpy, actor.mesh_objects)
    penetration_m = max(0.0, -float(metrics["minimum_mesh_z_m"]))
    metrics["penetration_cm"] = penetration_m * 100.0
    if penetration_m > max_penetration_cm / 100.0:
        return metrics, f"ground_penetration_cm>{max_penetration_cm:g}"

    errors: dict[str, float] = {}
    for name, (start_name, end_name) in _LIMB_SEGMENTS.items():
        source_start = _world_head(source, start_name)
        source_end = _world_head(source, end_name)
        target_start = _world_head(actor.armature, mapping[start_name])
        target_end = _world_head(actor.armature, mapping[end_name])
        error = _angle_degrees(source_start, source_end, target_start, target_end)
        if not np.isfinite(error):
            return metrics, f"non_finite_segment_error:{name}"
        errors[name] = error
    metrics["bone_coverage"] = {"mapped": len(mapping), "expected": len(required), "complete": True}
    metrics["segment_error_degrees"] = errors
    metrics["maximum_limb_error_degrees"] = max(errors.values())
    if metrics["maximum_limb_error_degrees"] > max_limb_error_deg:
        return metrics, f"limb_error_deg>{max_limb_error_deg:g}"
    return metrics, None


def _world_head(armature: Any, bone_name: str) -> np.ndarray:
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        raise PoseGridError(f"Missing pose bone {bone_name!r}")
    return np.asarray((armature.matrix_world @ bone.head).to_tuple(), dtype=np.float64)


def _angle_degrees(
    start: np.ndarray, end: np.ndarray, target_start: np.ndarray, target_end: np.ndarray
) -> float:
    first = end - start
    second = target_end - target_start
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        return float("nan")
    cosine = float(np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _add_ground(
    bpy: Any,
    minimum: np.ndarray,
    maximum: np.ndarray,
    color: tuple[int, int, int],
    margin: float,
    *,
    style: GroundStyle = "grid",
) -> None:
    center = (minimum + maximum) / 2.0
    size = float(np.max(maximum[:2] - minimum[:2]) + margin * 5.0)
    rgb = tuple(_srgb_to_linear(value / 255.0) for value in color)
    material = bpy.data.materials.new("PoseGrid_Ground")
    material.use_nodes = True
    _configure_ground_material(material, rgb)
    # The plane is deliberately much larger than the actor bounds. A finite
    # plane edge is especially visible in low-elevation orthographic views.
    plane_size = max(size, 100.0)
    # Pose-grid ground is a scene-level reference plane, fixed at world Z=0.
    # Retarget quality metrics still report any mesh penetration below it.
    plane_z = 0.0
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=(center[0], center[1], plane_z))
    bpy.context.object.data.materials.append(material)
    if style == "grid":
        # Keep the grid coplanar with the floor. The tiny epsilon avoids
        # depth-fighting without creating a visibly floating wire surface.
        _add_ground_grid(bpy, center, size, plane_z + 0.0001, rgb)


def _configure_ground_material(material: Any, rgb: tuple[float, float, float]) -> None:
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*rgb, 1.0)
    principled.inputs["Roughness"].default_value = 0.88
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = 0.18
    links.new(principled.outputs[0], output.inputs[0])


def _color_id(color: tuple[int, int, int]) -> str:
    return "".join(f"{value:02x}" for value in color)


def _set_pose_grid_background(bpy: Any, color: tuple[int, int, int]) -> None:
    from .scene import setup_world

    setup_world(transparent=False, background_rgb=color)
    background = bpy.context.scene.world.node_tree.nodes.get("Background")
    if background is not None:
        # The oversized ground plane fills the camera. A neutral world light
        # keeps its requested hue intact instead of multiplying the tint by
        # itself, while still lifting the unlit side of each shadow.
        background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        background.inputs["Strength"].default_value = 0.45
    rgb = tuple(_srgb_to_linear(value / 255.0) for value in color)
    ground = bpy.data.materials.get("PoseGrid_Ground")
    if ground is not None:
        principled = ground.node_tree.nodes.get("Principled BSDF")
        if principled is not None:
            principled.inputs["Base Color"].default_value = (*rgb, 1.0)
    grid = bpy.data.materials.get("PoseGrid_Grid")
    if grid is not None:
        emission = grid.node_tree.nodes.get("Emission")
        if emission is not None:
            emission.inputs["Color"].default_value = tuple(max(0.0, channel * 0.82) for channel in rgb) + (
                1.0,
            )


def _add_ground_grid(
    bpy: Any, center: np.ndarray, size: float, z: float, rgb: tuple[float, float, float]
) -> None:
    material = bpy.data.materials.new("PoseGrid_Grid")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = tuple(max(0.0, channel * 0.82) for channel in rgb) + (1.0,)
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs[0], output.inputs[0])
    half = size / 2.0
    step = 0.5
    lines = int(size / step)
    vertices = []
    faces = []
    thickness = 0.006
    for index in range(-lines, lines + 1):
        value = index * step
        start = len(vertices)
        vertices.extend(
            (
                (center[0] + value - thickness, center[1] - half, z),
                (center[0] + value + thickness, center[1] - half, z),
                (center[0] + value + thickness, center[1] + half, z),
                (center[0] + value - thickness, center[1] + half, z),
            )
        )
        faces.append((start, start + 1, start + 2, start + 3))
        start = len(vertices)
        vertices.extend(
            (
                (center[0] - half, center[1] + value - thickness, z),
                (center[0] + half, center[1] + value - thickness, z),
                (center[0] + half, center[1] + value + thickness, z),
                (center[0] - half, center[1] + value + thickness, z),
            )
        )
        faces.append((start, start + 1, start + 2, start + 3))
    mesh = bpy.data.meshes.new("PoseGrid_GridMesh")
    mesh.from_pydata(vertices, [], faces)
    grid = bpy.data.objects.new("PoseGrid_Grid", mesh)
    bpy.context.collection.objects.link(grid)
    grid.data.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = False
