from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import unicodedata
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from motionviewer.assets.fbx_catalog import FbxModel, is_binary_fbx, list_fbx_models, valid_fbx_models
from motionviewer.core.smplx_actor import FOOT_POSE_MODES
from motionviewer.packages.index import ClipIndex, PackageIndex, build_package_index
from motionviewer.packages.store import PackageStore, open_package_store

QUALITATIVE_SCHEMA = "motionviewer.qualitative.v1"
DEFAULT_SOURCES = ("omegamotiongpt", "kimodo", "hymotion")
DEFAULT_PROVENANCE_COUNTS = (
    ("HumanML3D", 34),
    ("bones-seed", 33),
    ("mm_MotionGV", 33),
)
REQUIRED_SMPLX_KEYS = frozenset({"joints22", "transl", "global_orient", "body_pose"})
SNAPSHOT_LAYOUTS = frozenset({"trajectory", "root_aligned", "arc"})
ARC_DIRECTIONS = frozenset({"up", "down"})
MATERIAL_MODES = frozenset({"palette", "preserve"})
FBX_POOLS = frozenset({"approved", "all_binary"})
BODY_MODES = frozenset({"fbx", "smplh", "smplx"})


@dataclass(frozen=True)
class QualitativeBatchRequest:
    package: Path
    output_dir: Path
    sources: tuple[str, ...] = DEFAULT_SOURCES
    clip_ids: tuple[str, ...] = ()
    provenance_counts: tuple[tuple[str, int], ...] = DEFAULT_PROVENANCE_COUNTS
    task: str = "t2m"
    seed: int = 20260726
    snapshots: int = 6
    snapshot_layout: str = "root_aligned"
    snapshot_spacing: float = 1.25
    arc_direction: str = "up"
    material_mode: str = "preserve"
    palette_start_rgb: tuple[int, int, int] = (26, 128, 184)
    palette_end_rgb: tuple[int, int, int] = (122, 26, 158)
    palette_color_rgb: tuple[int, int, int] | None = None
    snapshot_alpha: float = 1.0
    body_mode: str = "fbx"
    foot_pose: str = "source"
    fbx_pool: str = "approved"
    exclude_fbx: tuple[str, ...] = ()
    fbx_root: Path = Path("assets/fbx")
    resolution: tuple[int, int] = (2400, 900)
    samples: int = 32
    camera_preset: str = "three_quarter"
    camera_margin: float = 1.12

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("sources must contain at least one source id")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("source ids must be unique")
        if any(not clip_id.strip() for clip_id in self.clip_ids):
            raise ValueError("clip ids must be non-empty")
        if len(set(self.clip_ids)) != len(self.clip_ids):
            raise ValueError("clip ids must be unique")
        if not self.provenance_counts or any(count <= 0 for _, count in self.provenance_counts):
            raise ValueError("provenance counts must be positive")
        names = [name for name, _ in self.provenance_counts]
        if len(set(names)) != len(names):
            raise ValueError("provenance names must be unique")
        if self.snapshots < 1:
            raise ValueError("snapshots must be >= 1")
        if self.snapshot_layout not in SNAPSHOT_LAYOUTS:
            raise ValueError(f"snapshot_layout must be one of {sorted(SNAPSHOT_LAYOUTS)}")
        if self.snapshot_spacing <= 0:
            raise ValueError("snapshot_spacing must be > 0")
        if self.arc_direction not in ARC_DIRECTIONS:
            raise ValueError(f"arc_direction must be one of {sorted(ARC_DIRECTIONS)}")
        if self.material_mode not in MATERIAL_MODES:
            raise ValueError(f"material_mode must be one of {sorted(MATERIAL_MODES)}")
        for name, rgb in (
            ("palette_start_rgb", self.palette_start_rgb),
            ("palette_end_rgb", self.palette_end_rgb),
        ):
            if len(rgb) != 3 or any(not 0 <= int(value) <= 255 for value in rgb):
                raise ValueError(f"{name} must contain three RGB values in [0, 255]")
        if self.palette_color_rgb is not None and (
            len(self.palette_color_rgb) != 3
            or any(not 0 <= int(value) <= 255 for value in self.palette_color_rgb)
        ):
            raise ValueError("palette_color_rgb must contain three RGB values in [0, 255]")
        if not 0.0 < self.snapshot_alpha <= 1.0:
            raise ValueError("snapshot_alpha must be in (0, 1]")
        if self.body_mode not in BODY_MODES:
            raise ValueError(f"body_mode must be one of {sorted(BODY_MODES)}")
        if self.foot_pose not in FOOT_POSE_MODES:
            raise ValueError(f"foot_pose must be one of {sorted(FOOT_POSE_MODES)}")
        if self.fbx_pool not in FBX_POOLS:
            raise ValueError(f"fbx_pool must be one of {sorted(FBX_POOLS)}")
        if any(not model_id.strip() for model_id in self.exclude_fbx):
            raise ValueError("exclude_fbx entries must be non-empty")
        if min(self.resolution) <= 0:
            raise ValueError("resolution values must be positive")
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if self.camera_preset != "three_quarter":
            raise ValueError("qualitative rendering currently supports camera_preset='three_quarter'")
        if self.camera_margin <= 1.0:
            raise ValueError("camera_margin must be > 1")

    @property
    def clip_count(self) -> int:
        return len(self.clip_ids) if self.clip_ids else sum(count for _, count in self.provenance_counts)

    def to_json(self) -> dict[str, Any]:
        return {
            "package": str(self.package.resolve()),
            "output_dir": str(self.output_dir.resolve()),
            "sources": list(self.sources),
            "clip_ids": list(self.clip_ids),
            "provenance_counts": {name: count for name, count in self.provenance_counts},
            "task": self.task,
            "seed": self.seed,
            "snapshots": self.snapshots,
            "snapshot_layout": self.snapshot_layout,
            "snapshot_spacing": self.snapshot_spacing,
            "arc_direction": self.arc_direction,
            "material_mode": self.material_mode,
            "palette_start_rgb": list(self.palette_start_rgb),
            "palette_end_rgb": list(self.palette_end_rgb),
            "palette_color_rgb": None if self.palette_color_rgb is None else list(self.palette_color_rgb),
            "snapshot_alpha": self.snapshot_alpha,
            "body_mode": self.body_mode,
            "foot_pose": self.foot_pose,
            "fbx_pool": self.fbx_pool,
            "exclude_fbx": list(self.exclude_fbx),
            "fbx_root": str(self.fbx_root.resolve()),
            "resolution": list(self.resolution),
            "samples": self.samples,
            "camera_preset": self.camera_preset,
            "camera_margin": self.camera_margin,
        }


@dataclass(frozen=True)
class QualitativeSourceJob:
    source_id: str
    motion_path: Path
    frames: int
    frame_indices: tuple[int, ...]
    output_path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "motion_path": str(self.motion_path.resolve()),
            "frames": self.frames,
            "frame_indices": list(self.frame_indices),
            "output_path": str(self.output_path.resolve()),
        }


@dataclass(frozen=True)
class QualitativeClipJob:
    index: int
    provenance_index: int
    clip_id: str
    dir_name: str
    provenance: str
    caption: str | None
    fps: float
    fbx_model_id: str
    fbx_path: Path
    fbx_profile_id: str | None
    bone_map: str
    retarget_mode: str
    sources: tuple[QualitativeSourceJob, ...]
    output_dir: Path
    bundle_path: Path
    manifest_path: Path
    status_path: Path
    log_path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "provenance_index": self.provenance_index,
            "clip_id": self.clip_id,
            "dir_name": self.dir_name,
            "provenance": self.provenance,
            "caption": self.caption,
            "fps": self.fps,
            "fbx": {
                "model_id": self.fbx_model_id,
                "path": str(self.fbx_path.resolve()),
                "profile_id": self.fbx_profile_id,
                "bone_map": self.bone_map,
                "retarget_mode": self.retarget_mode,
            },
            "sources": [source.to_json() for source in self.sources],
            "output_dir": str(self.output_dir.resolve()),
            "bundle_path": str(self.bundle_path.resolve()),
            "manifest_path": str(self.manifest_path.resolve()),
            "status_path": str(self.status_path.resolve()),
            "log_path": str(self.log_path.resolve()),
        }


@dataclass(frozen=True)
class PreparedQualitativeBatch:
    request: QualitativeBatchRequest
    jobs: tuple[QualitativeClipJob, ...]
    selection_path: Path
    rejections: tuple[dict[str, str], ...] = ()


def parse_provenance_counts(values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"provenance quota must be NAME=COUNT, got {raw!r}")
        name, count_raw = raw.rsplit("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("provenance name cannot be empty")
        try:
            count = int(count_raw)
        except ValueError as exc:
            raise ValueError(f"invalid provenance count in {raw!r}") from exc
        result.append((name, count))
    return tuple(result)


def normalized_frame_indices(total_frames: int, snapshots: int) -> tuple[int, ...]:
    if total_frames <= 0:
        raise ValueError("motion must contain at least one frame")
    if snapshots < 1:
        raise ValueError("snapshots must be >= 1")
    if snapshots == 1:
        return (total_frames // 2,)
    values = np.rint(np.linspace(0, total_frames - 1, snapshots)).astype(np.int64)
    return tuple(int(value) for value in values)


def common_source_candidates(
    index: PackageIndex,
    *,
    task: str,
    sources: tuple[str, ...],
    provenance: str,
) -> list[ClipIndex]:
    result = []
    for clip in index.clips_for_task(task):
        if clip.provenance != provenance:
            continue
        if all(_source_has_smplx_params(clip, source_id) for source_id in sources):
            result.append(clip)
    return sorted(result, key=lambda item: item.clip_id)


def shuffled_candidates(candidates: list[ClipIndex], *, seed: int, provenance: str) -> list[ClipIndex]:
    result = list(candidates)
    digest = hashlib.sha256(f"{seed}:{provenance}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(result)
    return result


def prepare_qualitative_batch(
    request: QualitativeBatchRequest, *, overwrite: bool = False
) -> PreparedQualitativeBatch:
    root = request.output_dir.resolve()
    selection_path = root / "selection.json"
    if selection_path.is_file() and not overwrite:
        return load_prepared_qualitative_batch(selection_path, expected=request)

    eligible_fbx = _qualitative_fbx_models(request)
    if not eligible_fbx:
        raise ValueError(f"No FBX models available for pool {request.fbx_pool!r} under {request.fbx_root}")
    eligible_fbx = _shuffled_fbx(eligible_fbx, request.seed)

    root.mkdir(parents=True, exist_ok=True)
    store = open_package_store(request.package)
    jobs: list[QualitativeClipJob] = []
    rejections: list[dict[str, str]] = []
    try:
        index = build_package_index(store)
        if request.task not in index.tasks:
            raise ValueError(f"Unknown task {request.task!r}; available: {', '.join(sorted(index.tasks))}")
        global_index = 0
        if request.clip_ids:
            task_clips = index.clips_for_task(request.task)
            provenance_indices: dict[str, int] = {}
            for requested_clip_id in request.clip_ids:
                matches = [
                    clip
                    for clip in task_clips
                    if requested_clip_id in (clip.clip_id, clip.dir_name, clip.clip_id.replace(":", "_"))
                ]
                if not matches:
                    raise ValueError(f"Clip {requested_clip_id!r} not found in task {request.task!r}")
                if len(matches) > 1:
                    raise ValueError(
                        f"Clip identifier {requested_clip_id!r} is ambiguous in task {request.task!r}"
                    )
                clip = matches[0]
                missing_sources = [
                    source_id
                    for source_id in request.sources
                    if not _source_has_smplx_params(clip, source_id)
                ]
                if missing_sources:
                    raise ValueError(
                        f"Clip {clip.clip_id!r} is missing SMPL-X parameters for sources: "
                        f"{', '.join(missing_sources)}"
                    )
                provenance = clip.provenance or "unknown"
                provenance_index = provenance_indices.get(provenance, 0) + 1
                provenance_indices[provenance] = provenance_index
                fbx = eligible_fbx[global_index % len(eligible_fbx)]
                jobs.append(
                    _prepare_clip_job(
                        store,
                        request,
                        clip,
                        fbx=fbx,
                        index=global_index + 1,
                        provenance_index=provenance_index,
                    )
                )
                global_index += 1
        else:
            for provenance, count in request.provenance_counts:
                candidates = shuffled_candidates(
                    common_source_candidates(
                        index, task=request.task, sources=request.sources, provenance=provenance
                    ),
                    seed=request.seed,
                    provenance=provenance,
                )
                accepted = 0
                for clip in candidates:
                    if accepted >= count:
                        break
                    fbx = eligible_fbx[global_index % len(eligible_fbx)]
                    try:
                        job = _prepare_clip_job(
                            store,
                            request,
                            clip,
                            fbx=fbx,
                            index=global_index + 1,
                            provenance_index=accepted + 1,
                        )
                    except Exception as exc:  # noqa: BLE001 - candidate rejection is persisted
                        rejections.append(
                            {"clip_id": clip.clip_id, "provenance": provenance, "reason": str(exc)}
                        )
                        continue
                    jobs.append(job)
                    accepted += 1
                    global_index += 1
                if accepted != count:
                    raise ValueError(
                        f"Only {accepted} loadable common-source clips available for provenance "
                        f"{provenance!r}; requested {count}"
                    )
    finally:
        store.close()

    batch = PreparedQualitativeBatch(
        request=request,
        jobs=tuple(jobs),
        selection_path=selection_path,
        rejections=tuple(rejections),
    )
    _write_selection(batch, eligible_fbx)
    return batch


def load_prepared_qualitative_batch(
    selection_path: str | Path,
    *,
    expected: QualitativeBatchRequest | None = None,
) -> PreparedQualitativeBatch:
    path = Path(selection_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != QUALITATIVE_SCHEMA:
        raise ValueError(f"Unsupported qualitative selection schema in {path}")
    raw_request = payload["request"]
    request = QualitativeBatchRequest(
        package=Path(raw_request["package"]),
        output_dir=Path(raw_request["output_dir"]),
        sources=tuple(raw_request["sources"]),
        clip_ids=tuple(raw_request.get("clip_ids", ())),
        provenance_counts=tuple(
            (name, int(count)) for name, count in raw_request["provenance_counts"].items()
        ),
        task=raw_request["task"],
        seed=int(raw_request["seed"]),
        snapshots=int(raw_request["snapshots"]),
        snapshot_layout=raw_request.get("snapshot_layout", "trajectory"),
        snapshot_spacing=float(raw_request.get("snapshot_spacing", 1.0)),
        arc_direction=raw_request.get("arc_direction", "up"),
        material_mode=raw_request.get("material_mode", "palette"),
        palette_start_rgb=tuple(raw_request.get("palette_start_rgb", (26, 128, 184))),
        palette_end_rgb=tuple(raw_request.get("palette_end_rgb", (122, 26, 158))),
        palette_color_rgb=(
            None if raw_request.get("palette_color_rgb") is None else tuple(raw_request["palette_color_rgb"])
        ),
        snapshot_alpha=float(raw_request.get("snapshot_alpha", 1.0)),
        body_mode=raw_request.get("body_mode", "fbx"),
        foot_pose=raw_request.get("foot_pose", "source"),
        fbx_pool=raw_request.get("fbx_pool", "approved"),
        exclude_fbx=tuple(raw_request.get("exclude_fbx", ())),
        fbx_root=Path(raw_request["fbx_root"]),
        resolution=tuple(raw_request["resolution"]),
        samples=int(raw_request["samples"]),
        camera_preset=raw_request["camera_preset"],
        camera_margin=float(raw_request["camera_margin"]),
    )
    if expected is not None and request.to_json() != expected.to_json():
        raise ValueError(f"Existing selection request does not match requested options: {path}")
    jobs = tuple(_job_from_json(item) for item in payload["clips"])
    return PreparedQualitativeBatch(
        request=request,
        jobs=jobs,
        selection_path=path,
        rejections=tuple(payload.get("rejections", ())),
    )


def _source_has_smplx_params(clip: ClipIndex, source_id: str) -> bool:
    source = clip.gt if source_id == "gt" else clip.sources.get(source_id)
    return source is not None and "smplx_params" in source.variants


def run_qualitative_batch(
    batch: PreparedQualitativeBatch,
    *,
    blender: str | Path,
    workers: int = 2,
    resume: bool = True,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    blender_path = Path(blender)
    if not blender_path.is_file():
        raise ValueError(f"Blender executable does not exist: {blender_path}")
    script = Path(__file__).resolve().parents[1] / "blender" / "qualitative_entry.py"
    if not script.is_file():
        raise ValueError(f"Qualitative Blender script does not exist: {script}")

    root = batch.request.output_dir.resolve()
    status_path = root / "render_status.json"
    states: dict[int, dict[str, Any]] = {}
    pending: list[QualitativeClipJob] = []
    for job in batch.jobs:
        if resume and _job_outputs_complete(job):
            states[job.index] = _job_state(job, "skipped_complete", returncode=0)
        else:
            states[job.index] = _job_state(job, "pending")
            pending.append(job)
    _write_batch_status(status_path, states, batch)

    def render(job: QualitativeClipJob) -> tuple[QualitativeClipJob, int]:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        with job.log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    str(blender_path),
                    "--background",
                    "--python",
                    str(script),
                    "--",
                    "--bundle",
                    str(job.bundle_path),
                ],
                cwd=Path(__file__).resolve().parents[3],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return job, int(result.returncode)

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as pool:
        futures = {pool.submit(render, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                _, returncode = future.result()
            except Exception as exc:  # noqa: BLE001 - persisted in status
                states[job.index] = _job_state(job, "failed", error=str(exc))
            else:
                complete = returncode == 0 and _job_outputs_complete(job)
                states[job.index] = _job_state(
                    job,
                    "rendered" if complete else "failed",
                    returncode=returncode,
                    error=None if complete else f"Blender exited {returncode}; see {job.log_path}",
                )
            _write_batch_status(status_path, states, batch)

    summary = _write_batch_status(status_path, states, batch)
    if summary["summary"]["failed"]:
        raise RuntimeError(f"{summary['summary']['failed']} qualitative clip(s) failed; see {status_path}")
    return summary


def _prepare_clip_job(
    store: PackageStore,
    request: QualitativeBatchRequest,
    clip: ClipIndex,
    *,
    fbx: FbxModel,
    index: int,
    provenance_index: int,
) -> QualitativeClipJob:
    safe_provenance = _safe_name(clip.provenance or "unknown")
    stem = f"{provenance_index:03d}_{_safe_name(clip.clip_id)}"
    output_dir = request.output_dir.resolve() / "images" / safe_provenance / stem
    asset_dir = request.output_dir.resolve() / "assets" / safe_provenance / stem
    job_dir = request.output_dir.resolve() / "jobs" / safe_provenance / stem
    sources: list[QualitativeSourceJob] = []
    caption_slug = _caption_slug(clip.caption)
    for source_id in request.sources:
        source = clip.gt if source_id == "gt" else clip.sources[source_id]
        asset = source.variants["smplx_params"]
        raw = store.read_bytes(asset.relpath)
        frames = _validate_smplx_payload(raw, clip.clip_id, source_id)
        motion_path = asset_dir / f"{source_id}.smplx.npz"
        store.materialize(((asset.relpath, motion_path),))
        sources.append(
            QualitativeSourceJob(
                source_id=source_id,
                motion_path=motion_path,
                frames=frames,
                frame_indices=normalized_frame_indices(frames, request.snapshots),
                output_path=output_dir / f"{source_id}__{caption_slug}.png",
            )
        )

    bundle_path = job_dir / "bundle.json"
    job = QualitativeClipJob(
        index=index,
        provenance_index=provenance_index,
        clip_id=clip.clip_id,
        dir_name=clip.dir_name,
        provenance=clip.provenance or "unknown",
        caption=clip.caption,
        fps=clip.fps,
        fbx_model_id=fbx.model_id,
        fbx_path=fbx.path.resolve(),
        fbx_profile_id=fbx.profile_id,
        bone_map=fbx.bone_map,
        retarget_mode=fbx.retarget_mode,
        sources=tuple(sources),
        output_dir=output_dir,
        bundle_path=bundle_path,
        manifest_path=output_dir / "manifest.json",
        status_path=job_dir / "status.json",
        log_path=job_dir / "blender.log",
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(_clip_bundle(job, request), indent=2),
        encoding="utf-8",
    )
    return job


def _clip_bundle(job: QualitativeClipJob, request: QualitativeBatchRequest) -> dict[str, Any]:
    payload = job.to_json()
    payload.update(
        {
            "schema": QUALITATIVE_SCHEMA,
            "snapshots": request.snapshots,
            "snapshot_layout": request.snapshot_layout,
            "snapshot_spacing": request.snapshot_spacing,
            "arc_direction": request.arc_direction,
            "material_mode": request.material_mode,
            "palette_start_rgb": list(request.palette_start_rgb),
            "palette_end_rgb": list(request.palette_end_rgb),
            "palette_color_rgb": None
            if request.palette_color_rgb is None
            else list(request.palette_color_rgb),
            "snapshot_alpha": request.snapshot_alpha,
            "body_mode": request.body_mode,
            "foot_pose": request.foot_pose,
            "fbx_pool": request.fbx_pool,
            "resolution": list(request.resolution),
            "samples": request.samples,
            "camera": {
                "preset": request.camera_preset,
                "margin": request.camera_margin,
                "orthographic": True,
            },
            "transparent_background": True,
            "labels": False,
            "ground": False,
        }
    )
    return payload


def _validate_smplx_payload(raw: bytes, clip_id: str, source_id: str) -> int:
    try:
        with np.load(BytesIO(raw), allow_pickle=False) as data:
            missing = sorted(REQUIRED_SMPLX_KEYS - set(data.files))
            if missing:
                raise ValueError(f"missing keys: {', '.join(missing)}")
            frames = int(np.asarray(data["global_orient"]).shape[0])
            if frames <= 0:
                raise ValueError("empty motion")
            for key in REQUIRED_SMPLX_KEYS:
                value = np.asarray(data[key])
                if value.ndim == 0 or value.shape[0] != frames:
                    raise ValueError(f"{key} frame count does not match global_orient")
                if not np.isfinite(value).all():
                    raise ValueError(f"{key} contains non-finite values")
    except Exception as exc:
        raise ValueError(f"{clip_id}/{source_id} is not a loadable SMPL-X payload: {exc}") from exc
    return frames


def _shuffled_fbx(models: list[FbxModel], seed: int) -> list[FbxModel]:
    result = sorted(models, key=lambda item: item.model_id)
    random.Random(seed).shuffle(result)
    return result


def _qualitative_fbx_models(request: QualitativeBatchRequest) -> list[FbxModel]:
    excluded = {model_id.strip().lower() for model_id in request.exclude_fbx}
    if request.fbx_pool == "approved":
        models = valid_fbx_models(request.fbx_root)
        return [model for model in models if model.model_id.lower() not in excluded]
    # Exploratory mode is explicit: include binary catalog assets that are
    # pending the strict full-motion quality gate, but never ASCII FBX files.
    return [
        model
        for model in list_fbx_models(request.fbx_root)
        if is_binary_fbx(model.path)
        and model.profile_id is not None
        and model.model_id.lower() not in excluded
    ]


def _write_selection(batch: PreparedQualitativeBatch, eligible_fbx: list[FbxModel]) -> None:
    payload = {
        "schema": QUALITATIVE_SCHEMA,
        "request": batch.request.to_json(),
        "checkpoint_aliases": {"omegamotiongpt": "t2m_400m_flan_smt_fixed910k_s4_r260717"},
        "eligible_fbx": [model.to_json() for model in eligible_fbx],
        "clips": [job.to_json() for job in batch.jobs],
        "rejections": list(batch.rejections),
    }
    batch.selection_path.parent.mkdir(parents=True, exist_ok=True)
    batch.selection_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _job_from_json(payload: dict[str, Any]) -> QualitativeClipJob:
    fbx = payload["fbx"]
    sources = tuple(
        QualitativeSourceJob(
            source_id=item["source_id"],
            motion_path=Path(item["motion_path"]),
            frames=int(item["frames"]),
            frame_indices=tuple(int(value) for value in item["frame_indices"]),
            output_path=Path(item["output_path"]),
        )
        for item in payload["sources"]
    )
    return QualitativeClipJob(
        index=int(payload["index"]),
        provenance_index=int(payload["provenance_index"]),
        clip_id=payload["clip_id"],
        dir_name=payload["dir_name"],
        provenance=payload["provenance"],
        caption=payload.get("caption"),
        fps=float(payload["fps"]),
        fbx_model_id=fbx["model_id"],
        fbx_path=Path(fbx["path"]),
        fbx_profile_id=fbx.get("profile_id"),
        bone_map=fbx.get("bone_map", "auto"),
        retarget_mode=fbx.get("retarget_mode", "quality"),
        sources=sources,
        output_dir=Path(payload["output_dir"]),
        bundle_path=Path(payload["bundle_path"]),
        manifest_path=Path(payload["manifest_path"]),
        status_path=Path(payload["status_path"]),
        log_path=Path(payload["log_path"]),
    )


def _job_outputs_complete(job: QualitativeClipJob) -> bool:
    return job.manifest_path.is_file() and all(
        source.output_path.is_file() and source.output_path.stat().st_size > 0 for source in job.sources
    )


def _job_state(
    job: QualitativeClipJob,
    state: str,
    *,
    returncode: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "index": job.index,
        "clip_id": job.clip_id,
        "provenance": job.provenance,
        "state": state,
        "returncode": returncode,
        "error": error,
        "log_path": str(job.log_path),
    }


def _write_batch_status(
    path: Path,
    states: dict[int, dict[str, Any]],
    batch: PreparedQualitativeBatch,
) -> dict[str, Any]:
    counts = {"pending": 0, "rendered": 0, "skipped_complete": 0, "failed": 0}
    for state in states.values():
        counts[state["state"]] = counts.get(state["state"], 0) + 1
    payload = {
        "schema": QUALITATIVE_SCHEMA,
        "selection": str(batch.selection_path),
        "summary": counts,
        "clips": [states[index] for index in sorted(states)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _caption_slug(value: str | None, *, max_length: int = 96) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")
    slug = slug[:max_length].rstrip("_")
    return slug or "untitled"
