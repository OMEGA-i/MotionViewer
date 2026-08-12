from __future__ import annotations

import json
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motionviewer.assets.fbx_catalog import choose_random_fbx, valid_fbx_models
from motionviewer.packages.index import ClipIndex, build_package_index
from motionviewer.packages.store import open_package_store
from motionviewer.video.spec import RenderJob


@dataclass(frozen=True)
class PreviewJobInfo:
    index: int
    clip_id: str
    dir_name: str
    provenance: str | None
    source_id: str
    full_frames: int
    frame_indices: list[int]
    caption: str | None
    body_backend: str
    fbx_model_id: str | None
    fbx_path: str | None
    fbx_profile_id: str | None
    fbx_status: str | None
    fbx_evidence: dict[str, Any] | None
    sampling_mode: str
    asset_path: Path
    config_path: Path
    render_dir: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "clip_id": self.clip_id,
            "dir_name": self.dir_name,
            "provenance": self.provenance,
            "source_id": self.source_id,
            "full_frames": self.full_frames,
            "frame_indices": self.frame_indices,
            "caption": self.caption,
            "body_backend": self.body_backend,
            "fbx_model_id": self.fbx_model_id,
            "fbx_path": self.fbx_path,
            "fbx_profile_id": self.fbx_profile_id,
            "fbx_status": self.fbx_status,
            "fbx_evidence": self.fbx_evidence or {},
            "sampling_mode": self.sampling_mode,
            "asset_path": str(self.asset_path),
            "config_path": str(self.config_path),
            "render_dir": str(self.render_dir),
        }


def generate_package_preview(
    package_path: str | Path,
    *,
    task: str,
    source_id: str,
    clip_count: int,
    frame_count: int,
    output_dir: str | Path,
    body_pool: tuple[str, ...] = ("smplx",),
    fbx_root: str | Path = Path("assets/fbx"),
    seed: int = 0,
    render_samples: int = 24,
    resolution: tuple[int, int] = (640, 640),
) -> list[PreviewJobInfo]:
    if clip_count <= 0:
        raise ValueError("clip_count must be > 0")
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    normalized_pool = _normalize_body_pool(body_pool)
    rng = random.Random(seed)
    if "fbx-random" in normalized_pool and not valid_fbx_models(fbx_root):
        raise ValueError(f"body_pool requests fbx-random but no valid FBX models were found under {fbx_root}")

    root = Path(output_dir)
    assets_dir = root / "assets"
    configs_dir = root / "configs"
    renders_dir = root / "renders"
    assets_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    store = open_package_store(package_path)
    try:
        index = build_package_index(store)
        if task not in index.tasks:
            available = ", ".join(sorted(index.tasks))
            raise ValueError(f"Unknown task {task!r}. Available: {available}")
        candidates = _clips_with_source(index.clips_for_task(task), source_id)
        if not candidates:
            raise ValueError(f"No {task} clips contain source {source_id!r}")
        selected = candidates[:clip_count]
        jobs = []
        for idx, clip in enumerate(selected, start=1):
            jobs.append(
                _write_preview_job(
                    store,
                    clip,
                    source_id=source_id,
                    index=idx,
                    frame_count=frame_count,
                    body_pool=normalized_pool,
                    fbx_root=Path(fbx_root),
                    rng=rng,
                    assets_dir=assets_dir,
                    configs_dir=configs_dir,
                    renders_dir=renders_dir,
                    render_samples=render_samples,
                    resolution=resolution,
                )
            )
    finally:
        store.close()

    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps([job.to_json() for job in jobs], indent=2), encoding="utf-8")
    return jobs


def _clips_with_source(clips: list[ClipIndex], source_id: str) -> list[ClipIndex]:
    if source_id == "gt":
        return [clip for clip in clips if "smplx_params" in clip.gt.variants]
    return [
        clip
        for clip in clips
        if source_id in clip.sources and "smplx_params" in clip.sources[source_id].variants
    ]


def _write_preview_job(
    store,
    clip: ClipIndex,
    *,
    source_id: str,
    index: int,
    frame_count: int,
    body_pool: tuple[str, ...],
    fbx_root: Path,
    rng: random.Random,
    assets_dir: Path,
    configs_dir: Path,
    renders_dir: Path,
    render_samples: int,
    resolution: tuple[int, int],
) -> PreviewJobInfo:
    source = clip.gt if source_id == "gt" else clip.sources[source_id]
    asset = source.variants["smplx_params"]
    raw = store.read_bytes(asset.relpath)
    with np.load(BytesIO(raw), allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    source_frames = int(np.asarray(payload["joints22"]).shape[0])
    frame_indices = _sample_indices(source_frames, frame_count)
    sampled = _subsample_payload(payload, frame_indices)

    stem = f"{index:02d}_{_safe_name(clip.dir_name)}"
    asset_path = assets_dir / f"{stem}.smplx.npz"
    np.savez(asset_path, **sampled)

    body_choice = rng.choice(body_pool)
    body_backend = "blender_smplx_addon"
    fbx_model_id = None
    fbx_path = None
    fbx_profile_id = None
    fbx_status = None
    fbx_evidence = None
    input_body = None
    if body_choice == "fbx-random":
        model = choose_random_fbx(fbx_root, rng=rng)
        body_backend = "fbx_skeleton"
        fbx_model_id = model.model_id
        fbx_path = str(model.path.resolve())
        fbx_profile_id = model.profile_id
        fbx_status = model.status
        fbx_evidence = model.evidence or {}
        input_body = {
            "backend": "fbx_skeleton",
            "fbx_path": fbx_path,
            "bone_map": model.bone_map,
            "retarget_mode": model.retarget_mode,
        }

    label = f"{source_id} #{index}" if fbx_model_id is None else f"{source_id} #{index} ({fbx_model_id})"
    item: dict[str, Any] = {
        "path": str(asset_path.resolve()),
        "label": label,
        "format": "smplx_body22_fitted_aa",
        "loader_options": {"label": label, "fps": clip.fps, "sampling_mode": "keyframes"},
    }
    if input_body is not None:
        item["body"] = input_body

    render_dir = renders_dir / stem
    job = RenderJob.template(
        [item],
        task_mode="text_to_motion" if clip.task == "t2m" else "comparison",
        instruction=clip.caption,
        output_directory=str(render_dir.resolve()),
        mp4_name=f"{stem}.mp4",
    )
    job.render.samples = render_samples
    job.render.resolution = resolution
    job.output.keep_frames = True
    job.style.ghost_snapshots = 0
    job.style.ghost.mode = "none"
    job.style.prefix.ghost_count = 0
    job.timeline.show_prefix = False
    config_path = configs_dir / f"{stem}.yaml"
    config_path.write_text(yaml.safe_dump(job.to_json(), sort_keys=False), encoding="utf-8")

    return PreviewJobInfo(
        index=index,
        clip_id=clip.clip_id,
        dir_name=clip.dir_name,
        provenance=clip.provenance,
        source_id=source_id,
        full_frames=source_frames,
        frame_indices=frame_indices,
        caption=clip.caption,
        body_backend=body_backend,
        fbx_model_id=fbx_model_id,
        fbx_path=fbx_path,
        fbx_profile_id=fbx_profile_id,
        fbx_status=fbx_status,
        fbx_evidence=fbx_evidence,
        sampling_mode="keyframes",
        asset_path=asset_path,
        config_path=config_path,
        render_dir=render_dir,
    )


def _sample_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Cannot sample from an empty motion")
    if count == 1:
        return [0]
    return np.linspace(0, total_frames - 1, count, dtype=int).tolist()


def _subsample_payload(payload: dict[str, np.ndarray], frame_indices: list[int]) -> dict[str, np.ndarray]:
    source_frames = int(np.asarray(payload["joints22"]).shape[0])
    result: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if key != "betas" and arr.ndim >= 1 and arr.shape[0] == source_frames:
            result[key] = arr[frame_indices]
        else:
            result[key] = arr
    return result


def _normalize_body_pool(body_pool: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(item.strip() for item in body_pool if item.strip())
    if not values:
        raise ValueError("body_pool must contain at least one entry")
    supported = {"smplx", "fbx-random"}
    unknown = sorted(set(values) - supported)
    if unknown:
        raise ValueError(f"Unknown body_pool values: {', '.join(unknown)}")
    return values


def _safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace(" ", "_")
