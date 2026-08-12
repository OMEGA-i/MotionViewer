from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

FramesMode = Literal["max", "min"]
LayoutMode = Literal["single", "overlay"]
AlignmentMode = Literal["start_root", "trajectory_center", "world", "overlay"]
GroundMode = Literal[
    "none",
    "contact_patches",
    "trajectory_ribbon",
    "trajectory_carpet",
    "trajectory_rectangle",
    "footprint_trail",
    "coverage_hull",
]
CameraPreset = Literal["three_quarter", "front", "side", "top"]
StagingMode = Literal["world", "inplace"]
TaskMode = Literal["continuation", "text_to_motion", "comparison"]

_REMOVED_LAYOUT_MODES = frozenset({"side_by_side", "grid"})
_CAMERA_PRESETS = frozenset({"three_quarter", "front", "side", "top"})
_DEFAULT_INPLACE_GHOST = {"mode": "none"}
_DEFAULT_INPLACE_GROUND = {"mode": "none"}


@dataclass
class InputSpec:
    path: Path
    label: str | None = None
    format: str | None = None
    loader_options: dict[str, Any] = field(default_factory=dict)
    # Per-input body override — when set, takes precedence over the global BodySpec.
    body: BodySpec | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputSpec:
        body = None
        if data.get("body"):
            body = BodySpec(**dict(data["body"]))
        return cls(
            path=Path(data["path"]),
            label=data.get("label"),
            format=data.get("format"),
            loader_options=dict(data.get("loader_options", {})),
            body=body,
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": str(self.path),
            "label": self.label,
            "format": self.format,
            "loader_options": self.loader_options,
        }
        if self.body is not None:
            result["body"] = vars(self.body)
        return result


@dataclass
class TimelineSpec:
    fps: float | None = None
    start_frame: int = 0
    end_frame: int | None = None
    speed: float = 1.0
    show_prefix: bool = True
    # "max": render the full length of the longest input, freezing+fading shorter ones once
    # their own data ends. "min": trim every input to the shortest one (old default).
    frames_mode: FramesMode = "max"


@dataclass
class TaskSpec:
    mode: TaskMode = "continuation"
    instruction: str | None = None


@dataclass
class LayoutSpec:
    mode: LayoutMode = "single"
    alignment: AlignmentMode = "start_root"
    spacing: float = 1.25
    actor_scale: float = 1.0
    unit_scale: float = 1.0
    normalize_height: bool = False
    columns: int = 0
    cell_padding: float = 0.45
    reserve_trajectory: bool = True


@dataclass
class BodySpec:
    backend: str = "blender_smplx_addon"
    gender: str = "neutral"
    neutral_hands_face: bool = True
    allow_skeleton_fallback: bool = False
    # FBX skeleton backend fields (only used when backend == "fbx_skeleton")
    fbx_path: str | None = None
    fbx_scale: float = 1.0
    bone_map: str = "auto"
    retarget_mode: str = "quality"


@dataclass
class PrefixStyleSpec:
    mode: str = "attached"
    color: tuple[float, float, float] = (0.72, 0.75, 0.78)
    ghost_count: int = 2
    show_marker: bool = True
    transition_gap_frames: int = 0


@dataclass
class GhostStyleSpec:
    mode: str = "trail"
    include_prefix: bool = False
    warmup_frames: int = 3
    start_lightness: float = 0.55
    end_lightness: float = 0.0
    alpha: float = 0.18


@dataclass
class LabelsStyleSpec:
    mode: str = "legend"
    show_instruction: bool = True


@dataclass
class StyleSpec:
    palette: str = "paper"
    temporal_ramp: str = "lavender_to_purple"
    material_roughness: float = 0.55
    ghost_snapshots: int = 8
    trail_density: int = 10
    freeze_fade_frames: int = 10
    freeze_fade_alpha: float = 0.25
    prefix: PrefixStyleSpec = field(default_factory=PrefixStyleSpec)
    ghost: GhostStyleSpec = field(default_factory=GhostStyleSpec)
    labels: LabelsStyleSpec = field(default_factory=LabelsStyleSpec)


@dataclass
class GroundSpec:
    mode: GroundMode = "trajectory_rectangle"
    foot_joint_ids: list[int] = field(default_factory=lambda: [10, 11])
    height_threshold: float = 0.045
    velocity_threshold: float = 0.08
    patch_radius: float = 0.14
    opacity: float = 0.22
    carpet_padding: float = 0.35
    carpet_width: float = 0.0
    corner_radius: float = 0.12


@dataclass
class CameraViewSpec:
    preset: CameraPreset
    staging: StagingMode | None = None
    margin: float | None = None
    orthographic: bool | None = None
    ghost: dict[str, Any] | None = None
    ground: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraViewSpec:
        preset = str(data.get("preset", "three_quarter"))
        if preset not in _CAMERA_PRESETS:
            raise ValueError(f"Unknown camera preset {preset!r}")
        staging = data.get("staging")
        if staging is not None and staging not in ("world", "inplace"):
            raise ValueError(f"Unknown camera staging {staging!r}")
        return cls(
            preset=preset,  # type: ignore[arg-type]
            staging=staging,  # type: ignore[arg-type]
            margin=None if data.get("margin") is None else float(data["margin"]),
            orthographic=None if data.get("orthographic") is None else bool(data["orthographic"]),
            ghost=None if data.get("ghost") is None else dict(data["ghost"]),
            ground=None if data.get("ground") is None else dict(data["ground"]),
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"preset": self.preset}
        if self.staging is not None:
            payload["staging"] = self.staging
        if self.margin is not None:
            payload["margin"] = self.margin
        if self.orthographic is not None:
            payload["orthographic"] = self.orthographic
        if self.ghost is not None:
            payload["ghost"] = self.ghost
        if self.ground is not None:
            payload["ground"] = self.ground
        return payload


@dataclass(frozen=True)
class ResolvedCameraView:
    preset: CameraPreset
    staging: StagingMode
    margin: float
    orthographic: bool
    ghost: dict[str, Any]
    ground: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "staging": self.staging,
            "margin": self.margin,
            "orthographic": self.orthographic,
            "ghost": self.ghost,
            "ground": self.ground,
        }


@dataclass
class CameraSpec:
    preset: CameraPreset = "three_quarter"
    orthographic: bool = True
    margin: float = 1.15
    follow_root: bool = False
    views: list[CameraViewSpec] = field(default_factory=list)


@dataclass
class RenderSpec:
    engine: str = "BLENDER_EEVEE"
    resolution: tuple[int, int] = (1920, 1080)
    samples: int = 64
    transparent_background: bool = True
    frame_format: str = "PNG"


@dataclass
class OutputSpec:
    directory: Path = Path("outputs")
    mp4_name: str = "motionviewer.mp4"
    keep_frames: bool = True
    manifest_name: str = "manifest.json"
    overwrite: bool = False


@dataclass
class RenderJob:
    inputs: list[InputSpec]
    task: TaskSpec = field(default_factory=TaskSpec)
    timeline: TimelineSpec = field(default_factory=TimelineSpec)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    body: BodySpec = field(default_factory=BodySpec)
    style: StyleSpec = field(default_factory=StyleSpec)
    ground: GroundSpec = field(default_factory=GroundSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    render: RenderSpec = field(default_factory=RenderSpec)
    output: OutputSpec = field(default_factory=OutputSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderJob:
        layout_data = dict(data.get("layout", {}))
        mode = layout_data.get("mode", "single")
        if mode in _REMOVED_LAYOUT_MODES:
            raise ValueError(
                f"layout.mode {mode!r} was removed; use mode='single' (one actor per job) "
                "and compose comparisons in PPT, or mode='overlay' for intentional overlap"
            )
        if mode not in ("single", "overlay"):
            raise ValueError(f"Unknown layout.mode {mode!r}; expected 'single' or 'overlay'")
        camera = _camera_spec_from_dict(dict(data.get("camera", {})))
        return cls(
            inputs=[InputSpec.from_dict(item) for item in data.get("inputs", [])],
            task=TaskSpec(**dict(data.get("task", {}))),
            timeline=TimelineSpec(**dict(data.get("timeline", {}))),
            layout=LayoutSpec(**layout_data),
            body=BodySpec(**dict(data.get("body", {}))),
            style=_style_spec_from_dict(dict(data.get("style", {}))),
            ground=GroundSpec(**dict(data.get("ground", {}))),
            camera=camera,
            render=_render_spec_from_dict(dict(data.get("render", {}))),
            output=_output_spec_from_dict(dict(data.get("output", {}))),
        )

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        if not self.inputs:
            errors.append("At least one input is required")
        if len(self.inputs) > 1 and self.layout.mode == "single":
            errors.append(
                "layout.mode='single' allows only one input; render each model in its own job "
                "and compose in PPT, or set layout.mode='overlay'"
            )
        if self.body.gender not in ("neutral", "male", "female"):
            errors.append(f"Invalid gender {self.body.gender!r}")
        if self.layout.spacing <= 0:
            errors.append("layout.spacing must be > 0")
        if self.layout.cell_padding < 0:
            errors.append("layout.cell_padding must be >= 0")
        if self.layout.actor_scale <= 0:
            errors.append("layout.actor_scale must be > 0")
        if self.layout.unit_scale <= 0:
            errors.append("layout.unit_scale must be > 0")
        if self.camera.margin <= 0:
            errors.append("camera.margin must be > 0")
        if self.camera.follow_root:
            errors.append("camera.follow_root is not supported")
        if self.camera.preset not in _CAMERA_PRESETS:
            errors.append(f"Unknown camera.preset {self.camera.preset!r}")
        view_errors = _validate_camera_views(self.camera)
        errors.extend(view_errors)
        if self.render.samples <= 0:
            errors.append("render.samples must be > 0")
        if self.render.resolution[0] <= 0 or self.render.resolution[1] <= 0:
            errors.append("render.resolution dimensions must be > 0")
        if self.timeline.start_frame < 0:
            errors.append("timeline.start_frame must be >= 0")
        if self.timeline.speed <= 0:
            errors.append("timeline.speed must be > 0")
        if self.timeline.end_frame is not None and self.timeline.end_frame < 0:
            errors.append("timeline.end_frame must be >= 0 when set")
        if self.style.freeze_fade_frames < 0:
            errors.append("style.freeze_fade_frames must be >= 0")
        if self.ground.opacity < 0 or self.ground.opacity > 1:
            errors.append("ground.opacity must be in [0, 1]")
        if self.body.backend == "fbx_skeleton" and not self.body.fbx_path:
            errors.append("body.fbx_path is required when backend is 'fbx_skeleton'")
        return errors

    @classmethod
    def template(
        cls,
        items: list[dict[str, Any]],
        *,
        task_mode: str = "continuation",
        instruction: str | None = None,
        output_directory: str = "outputs/generated",
        mp4_name: str = "comparison.mp4",
    ) -> RenderJob:
        """Build a RenderJob with opinionated rendering defaults."""
        show_prefix = task_mode == "continuation"
        return cls.from_dict(
            {
                "inputs": items,
                "task": {"mode": task_mode, "instruction": instruction},
                "timeline": {"show_prefix": show_prefix},
                "layout": {"mode": "single", "alignment": "start_root", "spacing": 1.6, "cell_padding": 0.5},
                "body": {"backend": "blender_smplx_addon", "gender": "neutral", "neutral_hands_face": True},
                "style": {
                    "palette": "soft_paper",
                    "temporal_ramp": "lavender_to_purple",
                    "ghost_snapshots": 4,
                    "prefix": {
                        "mode": "attached",
                        "color": [0.74, 0.76, 0.76],
                        "ghost_count": 2,
                        "transition_gap_frames": 0,
                    },
                    "ghost": {"mode": "trail", "include_prefix": False, "warmup_frames": 3, "alpha": 0.16},
                    "labels": {"mode": "legend", "show_instruction": bool(instruction)},
                },
                "ground": {"mode": "trajectory_rectangle", "opacity": 0.16, "carpet_padding": 0.4},
                "camera": {"preset": "three_quarter", "orthographic": True, "margin": 1.2},
                "render": {
                    "engine": "BLENDER_EEVEE",
                    "resolution": [1920, 1080],
                    "samples": 64,
                    "transparent_background": True,
                },
                "output": {"directory": output_directory, "mp4_name": mp4_name, "keep_frames": True},
            }
        )

    def to_json(self) -> dict[str, Any]:
        from dataclasses import asdict

        d = asdict(self)
        d["inputs"] = [item.to_json() for item in self.inputs]
        d["output"]["directory"] = str(self.output.directory)
        d["render"]["resolution"] = list(self.render.resolution)
        d["camera"]["views"] = [view.to_json() for view in self.camera.views]
        return d


def resolve_camera_views(
    camera: CameraSpec,
    *,
    style_ghost: dict[str, Any] | None = None,
    ground: dict[str, Any] | None = None,
) -> list[ResolvedCameraView]:
    """Normalize camera.views (or fall back to camera.preset) with staging defaults."""
    base_ghost = dict(style_ghost or {})
    base_ground = dict(ground or {})
    raw_views = list(camera.views) if camera.views else [CameraViewSpec(preset=camera.preset)]
    resolved: list[ResolvedCameraView] = []
    for view in raw_views:
        staging = view.staging or _default_staging(view.preset)
        ghost = dict(base_ghost)
        ground_cfg = dict(base_ground)
        if staging == "inplace":
            ghost = {**ghost, **_DEFAULT_INPLACE_GHOST}
            ground_cfg = {**ground_cfg, **_DEFAULT_INPLACE_GROUND}
        if view.ghost:
            ghost.update(view.ghost)
        if view.ground:
            ground_cfg.update(view.ground)
        resolved.append(
            ResolvedCameraView(
                preset=view.preset,
                staging=staging,
                margin=float(camera.margin if view.margin is None else view.margin),
                orthographic=bool(camera.orthographic if view.orthographic is None else view.orthographic),
                ghost=ghost,
                ground=ground_cfg,
            )
        )
    return resolved


def group_views_by_staging(views: list[ResolvedCameraView]) -> dict[StagingMode, list[ResolvedCameraView]]:
    grouped: dict[StagingMode, list[ResolvedCameraView]] = {}
    for view in views:
        grouped.setdefault(view.staging, []).append(view)
    return grouped


def _default_staging(preset: CameraPreset) -> StagingMode:
    if preset in ("front", "side"):
        return "inplace"
    return "world"


def _validate_camera_views(camera: CameraSpec) -> list[str]:
    errors: list[str] = []
    try:
        views = resolve_camera_views(camera)
    except ValueError as exc:
        return [str(exc)]
    presets = [view.preset for view in views]
    if len(presets) != len(set(presets)):
        errors.append("camera.views presets must be unique")
    for view in views:
        if view.margin <= 0:
            errors.append(f"camera view {view.preset!r} margin must be > 0")
    # Same staging must share normalized ghost/ground overrides.
    by_staging = group_views_by_staging(views)
    for staging, items in by_staging.items():
        ghost_keys = {tuple(sorted(item.ghost.items())) for item in items}
        ground_keys = {tuple(sorted((k, str(v)) for k, v in item.ground.items())) for item in items}
        if len(ghost_keys) > 1 or len(ground_keys) > 1:
            errors.append(f"camera.views with staging={staging!r} must share the same ghost/ground overrides")
    return errors


def _camera_spec_from_dict(data: dict[str, Any]) -> CameraSpec:
    views_raw = data.pop("views", []) or []
    views = [CameraViewSpec.from_dict(item) for item in views_raw]
    preset = str(data.get("preset", "three_quarter"))
    if preset == "follow_root":
        raise ValueError("camera.preset 'follow_root' is not supported; use static presets or camera.views")
    if preset not in _CAMERA_PRESETS:
        raise ValueError(f"Unknown camera.preset {preset!r}")
    return CameraSpec(
        preset=preset,  # type: ignore[arg-type]
        orthographic=bool(data.get("orthographic", True)),
        margin=float(data.get("margin", 1.15)),
        follow_root=bool(data.get("follow_root", False)),
        views=views,
    )


def _style_spec_from_dict(data: dict[str, Any]) -> StyleSpec:
    prefix = PrefixStyleSpec(**dict(data.pop("prefix", {})))
    ghost = GhostStyleSpec(**dict(data.pop("ghost", {})))
    labels = LabelsStyleSpec(**dict(data.pop("labels", {})))
    if "prefix_style" in data:
        data.pop("prefix_style", None)
    return StyleSpec(prefix=prefix, ghost=ghost, labels=labels, **data)


def _render_spec_from_dict(data: dict[str, Any]) -> RenderSpec:
    if "resolution" in data:
        data["resolution"] = tuple(data["resolution"])
    return RenderSpec(**data)


def _output_spec_from_dict(data: dict[str, Any]) -> OutputSpec:
    if "directory" in data:
        data["directory"] = Path(data["directory"])
    return OutputSpec(**data)
