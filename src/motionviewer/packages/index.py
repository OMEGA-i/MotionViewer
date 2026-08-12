from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .errors import PackageFormatError
from .store import PackageStore
from .types import (
    PackageAsset,
    PackageClipSummary,
    PackageSourceSummary,
    PackageTask,
    PackageTaskSummary,
    SourceAssets,
    SourceKind,
)

SUPPORTED_PROTOCOL_VERSION = "2.0"
ASSET_FORMATS = {
    "smplx_params": "smplx_body22_fitted_aa",
    "joints22": "joints_npy",
    "joints77": "joints_npy",
}


@dataclass
class TaskSection:
    task: PackageTask
    num_clips: int
    provenances: tuple[str, ...]
    sources: tuple[PackageSourceSummary, ...]
    reconstruction_levels: tuple[int, ...] | None


@dataclass
class ClipIndex:
    task: PackageTask
    dir_name: str
    clip_id: str
    provenance: str | None
    rec_id: str | None
    caption: str | None
    fps: float
    frames: int
    prefix_T: int
    predicted_T: int | None
    meta_relpath: PurePosixPath
    gt: SourceAssets
    sources: dict[str, SourceAssets]


@dataclass
class PackageIndex:
    protocol_version: str
    track: str
    split: str
    fps: float
    num_clips: int
    tasks: dict[str, TaskSection] = field(default_factory=dict)
    clips: dict[tuple[str, str], ClipIndex] = field(default_factory=dict)
    clip_order: list[tuple[str, str]] = field(default_factory=list)

    def task_summaries(self) -> dict[str, PackageTaskSummary]:
        return {
            name: PackageTaskSummary(
                task=section.task,
                num_clips=section.num_clips,
                provenances=section.provenances,
                sources=section.sources,
                reconstruction_levels=section.reconstruction_levels,
            )
            for name, section in self.tasks.items()
        }

    def clip_summaries(self) -> tuple[PackageClipSummary, ...]:
        summaries = []
        for key in self.clip_order:
            clip = self.clips[key]
            source_ids = ("gt",) + tuple(clip.sources.keys())
            summaries.append(
                PackageClipSummary(
                    clip_id=clip.clip_id,
                    dir_name=clip.dir_name,
                    task=clip.task,
                    provenance=clip.provenance,
                    rec_id=clip.rec_id,
                    caption=clip.caption,
                    fps=clip.fps,
                    frames=clip.frames,
                    prefix_T=clip.prefix_T,
                    predicted_T=clip.predicted_T,
                    sources=source_ids,
                )
            )
        return tuple(summaries)

    def clips_for_task(self, task: str) -> list[ClipIndex]:
        return [self.clips[key] for key in self.clip_order if key[0] == task]


def build_package_index(store: PackageStore) -> PackageIndex:
    if not store.exists("manifest.json"):
        raise PackageFormatError("Package is missing manifest.json")
    manifest = store.read_json("manifest.json")
    protocol_version = str(manifest.get("protocol_version", ""))
    if protocol_version != SUPPORTED_PROTOCOL_VERSION:
        raise PackageFormatError(
            f"Unsupported protocol_version {protocol_version!r}; expected {SUPPORTED_PROTOCOL_VERSION!r}"
        )

    track = _require_str(manifest, "track")
    split = _require_str(manifest, "split")
    fps = float(manifest.get("fps", 0.0))
    num_clips = int(manifest.get("num_clips", 0))
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise PackageFormatError("manifest.json missing required field 'tasks'")

    tasks: dict[str, TaskSection] = {}
    for task_name, payload in raw_tasks.items():
        if not isinstance(payload, dict):
            raise PackageFormatError(f"manifest.tasks[{task_name!r}] must be an object")
        tasks[str(task_name)] = _parse_task_section(str(task_name), payload)

    index = PackageIndex(
        protocol_version=protocol_version,
        track=track,
        split=split,
        fps=fps,
        num_clips=num_clips,
        tasks=tasks,
    )

    for task_name, dir_name in _discover_clips(store, set(tasks)):
        meta_relpath = PurePosixPath("clips") / task_name / dir_name / "meta.json"
        if not store.exists(meta_relpath):
            raise PackageFormatError(f"Clip {task_name}/{dir_name!r} is missing meta.json")
        meta = store.read_json(meta_relpath)
        section = tasks[task_name]
        clip = _build_clip_index(
            task=section.task,
            dir_name=dir_name,
            meta_relpath=meta_relpath,
            meta=meta,
            manifest_sources=section.sources,
        )
        key = (task_name, dir_name)
        index.clips[key] = clip
        index.clip_order.append(key)

    if not index.clip_order:
        raise PackageFormatError("Package contains no clips/")
    return index


def _discover_clips(store: PackageStore, known_tasks: set[str]) -> list[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for member in store.list_files("clips"):
        parts = member.relpath.parts
        # clips/{task}/{dir_name}/...
        if len(parts) >= 4 and parts[0] == "clips" and parts[1] in known_tasks:
            found.add((parts[1], parts[2]))
    return sorted(found)


def _parse_task_section(task_name: str, payload: dict[str, Any]) -> TaskSection:
    task = _require_task(payload.get("task") or task_name)
    if task != task_name:
        raise PackageFormatError(f"manifest.tasks key {task_name!r} does not match task field {task!r}")
    provenances_raw = payload.get("provenances") or []
    if not isinstance(provenances_raw, list):
        raise PackageFormatError(f"manifest.tasks[{task_name}].provenances must be a list")
    provenances = tuple(str(item) for item in provenances_raw)
    sources = tuple(_parse_manifest_source(item) for item in payload.get("sources", []))
    reconstruction_levels = _parse_reconstruction_levels(payload.get("reconstruction_levels"))
    return TaskSection(
        task=task,
        num_clips=int(payload.get("num_clips", 0)),
        provenances=provenances,
        sources=sources,
        reconstruction_levels=reconstruction_levels,
    )


def _build_clip_index(
    *,
    task: PackageTask,
    dir_name: str,
    meta_relpath: PurePosixPath,
    meta: dict[str, Any],
    manifest_sources: tuple[PackageSourceSummary, ...],
) -> ClipIndex:
    clip_root = PurePosixPath("clips") / task / dir_name
    frames = int(meta.get("T") or meta.get("frames") or 0)
    fps = float(meta.get("fps") or 0.0)
    prefix_t = int(meta.get("prefix_T") or 0)
    predicted_t = meta.get("predicted_T")
    caption = meta.get("caption")
    if caption is not None:
        caption = str(caption)
    provenance = meta.get("provenance")
    provenance_str = None if provenance is None else str(provenance)

    meta_task = meta.get("task")
    if meta_task is not None and str(meta_task) != task:
        raise PackageFormatError(
            f"Clip {task}/{dir_name} meta.task={meta_task!r} does not match directory task {task!r}"
        )

    gt_refs = dict(meta.get("gt") or {})
    if not gt_refs:
        raise PackageFormatError(f"Clip {task}/{dir_name} meta.json is missing gt references")
    gt = _build_source_assets(
        source_id="gt",
        kind="gt",
        label="gt",
        refs=gt_refs,
        clip_root=clip_root,
        role=None,
        native_rep=None,
        num_codebooks_used=None,
        mse_vs_gt=None,
        fit_mse=None,
    )

    source_map: dict[str, SourceAssets] = {}
    meta_sources = dict(meta.get("sources") or {})
    ordered_ids = [item.source_id for item in manifest_sources]
    for source_id in meta_sources:
        if source_id not in ordered_ids:
            ordered_ids.append(source_id)

    for source_id in ordered_ids:
        if source_id not in meta_sources:
            continue
        payload = dict(meta_sources[source_id])
        kind = str(payload.get("kind") or "model")
        if kind not in ("model", "reconstruction_level"):
            kind = "model"
        source_map[source_id] = _build_source_assets(
            source_id=source_id,
            kind=kind,  # type: ignore[arg-type]
            label=_source_label(source_id, payload),
            refs=payload,
            clip_root=clip_root,
            role=None if payload.get("role") is None else str(payload.get("role")),
            native_rep=None if payload.get("native_rep") is None else str(payload.get("native_rep")),
            num_codebooks_used=_optional_int(payload.get("num_codebooks_used")),
            mse_vs_gt=_optional_float(payload.get("mse_vs_gt")),
            fit_mse=_optional_float(payload.get("fit_mse")),
        )

    return ClipIndex(
        task=task,
        dir_name=dir_name,
        clip_id=str(meta.get("clip_id") or dir_name),
        provenance=provenance_str,
        rec_id=None if meta.get("rec_id") is None else str(meta.get("rec_id")),
        caption=caption,
        fps=fps,
        frames=frames,
        prefix_T=prefix_t,
        predicted_T=None if predicted_t is None else int(predicted_t),
        meta_relpath=meta_relpath,
        gt=gt,
        sources=source_map,
    )


def _build_source_assets(
    *,
    source_id: str,
    kind: SourceKind,
    label: str,
    refs: dict[str, Any],
    clip_root: PurePosixPath,
    role: str | None,
    native_rep: str | None,
    num_codebooks_used: int | None,
    mse_vs_gt: float | None,
    fit_mse: float | None,
) -> SourceAssets:
    variants: dict[str, PackageAsset] = {}
    for logical_name in ("smplx_params", "joints22", "joints77", "native"):
        raw = refs.get(logical_name)
        if not raw:
            continue
        rel = _clip_relative(clip_root, str(raw))
        variants[logical_name] = PackageAsset(
            logical_name=logical_name,
            relpath=rel,
            motion_format=ASSET_FORMATS.get(logical_name),
        )
    return SourceAssets(
        source_id=source_id,
        kind=kind,
        label=label,
        role=role,
        native_rep=native_rep,
        num_codebooks_used=num_codebooks_used,
        mse_vs_gt=mse_vs_gt,
        fit_mse=fit_mse,
        variants=variants,
    )


def _clip_relative(clip_root: PurePosixPath, raw: str) -> PurePosixPath:
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts:
        raise PackageFormatError(f"Unsafe asset path in meta.json: {raw}")
    return clip_root / pure


def _parse_manifest_source(item: dict[str, Any]) -> PackageSourceSummary:
    source_id = str(item.get("source_id") or "")
    if not source_id:
        raise PackageFormatError("manifest.sources entry missing source_id")
    kind = str(item.get("kind") or "model")
    if kind not in ("model", "reconstruction_level"):
        raise PackageFormatError(f"Unknown source kind {kind!r} for {source_id}")
    return PackageSourceSummary(
        source_id=source_id,
        kind=kind,  # type: ignore[arg-type]
        role=None if item.get("role") is None else str(item.get("role")),
        native_rep=None if item.get("native_rep") is None else str(item.get("native_rep")),
        num_codebooks_used=_optional_int(item.get("num_codebooks_used")),
    )


def _parse_reconstruction_levels(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise PackageFormatError("reconstruction_levels must be a list or null")
    return tuple(int(item) for item in value)


def _require_task(value: Any) -> PackageTask:
    task = str(value or "")
    if task not in ("t2m", "pred", "recon"):
        raise PackageFormatError(f"Unsupported package task {task!r}")
    return task  # type: ignore[return-value]


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or str(value) == "":
        raise PackageFormatError(f"manifest.json missing required field {key!r}")
    return str(value)


def _source_label(source_id: str, payload: dict[str, Any]) -> str:
    if payload.get("kind") == "reconstruction_level":
        mse = payload.get("mse_vs_gt")
        n_cb = payload.get("num_codebooks_used")
        if n_cb is not None and mse is not None:
            return f"{source_id} ({int(n_cb)} cb, MSE={float(mse):.4f})"
        if n_cb is not None:
            return f"{source_id} ({int(n_cb)} codebooks)"
    return source_id


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
