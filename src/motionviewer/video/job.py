from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motionviewer.core.coordinates import bounding_box, merge_bounds, source_to_blender_points
from motionviewer.core.schema import MotionSequence
from motionviewer.loaders import MotionFormatRegistry, default_registry

from .spec import InputSpec, RenderJob


@dataclass
class ResolvedInput:
    spec: InputSpec
    sequence: MotionSequence
    source_hash: str
    blender_bounds: tuple[list[float], list[float]]

    def to_json(self) -> dict[str, Any]:
        return {
            "input": self.spec.to_json(),
            "source_hash": self.source_hash,
            "sequence": self.sequence.to_json_summary(),
            "blender_bounds": {"min": self.blender_bounds[0], "max": self.blender_bounds[1]},
        }


@dataclass
class PreparedRenderJob:
    job: RenderJob
    inputs: list[ResolvedInput]
    scene_bounds: tuple[list[float], list[float]]
    frames: int
    fps: float

    def to_bundle(self) -> dict[str, Any]:
        return {
            "job": self.job.to_json(),
            "inputs": [item.to_json() for item in self.inputs],
            "scene_bounds": {"min": self.scene_bounds[0], "max": self.scene_bounds[1]},
            "frames": self.frames,
            "fps": self.fps,
        }


def load_render_job(path: str | Path) -> RenderJob:
    # Imported lazily: only the CLI (project venv) parses YAML configs. The
    # Blender-side runner consumes the pre-computed JSON bundle and must import
    # this module under Blender's bundled Python, which has no pyyaml.
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    base = config_path.parent
    job = RenderJob.from_dict(data)
    for input_spec in job.inputs:
        if not input_spec.path.is_absolute():
            input_spec.path = (base / input_spec.path).resolve()
    # Resolve backend-specific paths via the backend registry
    from motionviewer.blender.backend_registry import default_backend_registry

    _reg = default_backend_registry()
    resolved_body = _reg.resolve_body(vars(job.body), base)
    job.body.fbx_path = resolved_body.get("fbx_path")
    job.body.bone_map = resolved_body.get("bone_map", job.body.bone_map)
    for inp in job.inputs:
        if inp.body is not None:
            resolved = _reg.resolve_body(vars(inp.body), base)
            inp.body.fbx_path = resolved.get("fbx_path")
            inp.body.bone_map = resolved.get("bone_map", inp.body.bone_map)
    if not job.output.directory.is_absolute():
        job.output.directory = (base / job.output.directory).resolve()
    return job


def prepare_render_job(
    job: RenderJob,
    *,
    registry: MotionFormatRegistry | None = None,
) -> PreparedRenderJob:
    errors = job.validate()
    errors.extend(_validate_body_configs(job))
    if errors:
        raise ValueError("; ".join(errors))
    if not job.inputs:
        raise ValueError("RenderJob requires at least one input")
    reg = registry or default_registry()
    resolved = [_resolve_input(item, reg, job.body.gender) for item in job.inputs]
    fps = _resolve_fps(job, [item.sequence for item in resolved])
    frames = _resolve_frames(job, [item.sequence for item in resolved])
    scene_bounds = _scene_bounds(resolved, job.layout.unit_scale)
    return PreparedRenderJob(job=job, inputs=resolved, scene_bounds=scene_bounds, frames=frames, fps=fps)


def _validate_body_configs(job: RenderJob) -> list[str]:
    from motionviewer.blender.backend_registry import default_backend_registry

    reg = default_backend_registry()
    errors = reg.validate_body(vars(job.body))
    for idx, input_spec in enumerate(job.inputs):
        if input_spec.body is None:
            continue
        for message in reg.validate_body(vars(input_spec.body)):
            errors.append(f"inputs[{idx}].body: {message}")
    return errors


def write_bundle(prepared: PreparedRenderJob, path: str | Path) -> Path:
    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with bundle_path.open("w", encoding="utf-8") as handle:
        json.dump(prepared.to_bundle(), handle, indent=2)
    return bundle_path


def write_manifest(prepared: PreparedRenderJob, path: str | Path) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "motionviewer_version": _version(),
        "prepared_job": prepared.to_bundle(),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def quick_job(
    inputs: list[Path],
    *,
    labels: list[str] | None = None,
    output_dir: Path = Path("outputs/quick"),
) -> RenderJob:
    specs = []
    for idx, path in enumerate(inputs):
        label = labels[idx] if labels and idx < len(labels) else None
        specs.append(InputSpec(path=path, label=label))
    job = RenderJob(inputs=specs)
    job.output.directory = output_dir
    return job


def _resolve_input(input_spec: InputSpec, registry: MotionFormatRegistry, gender: str) -> ResolvedInput:
    options = dict(input_spec.loader_options)
    if input_spec.label:
        options["label"] = input_spec.label
    options.setdefault("gender", gender)
    sequence = registry.load(input_spec.path, format_id=input_spec.format, options=options)
    joints = sequence.require_joints()
    blender_joints = source_to_blender_points(joints, sequence.coordinate_system)
    mins, maxs = bounding_box(blender_joints)
    return ResolvedInput(
        spec=input_spec,
        sequence=sequence,
        source_hash=sha256_file(input_spec.path),
        blender_bounds=(mins.tolist(), maxs.tolist()),
    )


def _resolve_fps(job: RenderJob, sequences: list[MotionSequence]) -> float:
    if job.timeline.fps is not None:
        return float(job.timeline.fps)
    first = sequences[0].fps
    if any(abs(seq.fps - first) > 1e-5 for seq in sequences):
        raise ValueError("Input FPS values differ; set timeline.fps explicitly")
    return first


def _resolve_frames(job: RenderJob, sequences: list[MotionSequence]) -> int:
    if job.timeline.frames_mode == "min":
        natural = min(seq.frames for seq in sequences)
    else:
        natural = max(seq.frames for seq in sequences)
    end = job.timeline.end_frame if job.timeline.end_frame is not None else natural
    return max(0, min(end, natural) - job.timeline.start_frame)


def _scene_bounds(resolved: list[ResolvedInput], unit_scale: float) -> tuple[list[float], list[float]]:
    bounds = []
    for item in resolved:
        sequence = item.sequence
        joints = source_to_blender_points(
            sequence.require_joints(), sequence.coordinate_system, unit_scale=unit_scale
        )
        bounds.append(bounding_box(joints))
    mins, maxs = merge_bounds(bounds)
    return mins.tolist(), maxs.tolist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version() -> str:
    try:
        from motionviewer import __version__

        return __version__
    except Exception:
        return "unknown"
