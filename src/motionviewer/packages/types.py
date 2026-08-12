from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

ValidationMode = Literal["none", "structural", "assets", "loadable"]
AssetPreference = Literal["smplx", "joints22"]
PackageTask = Literal["t2m", "pred", "recon"]
SourceKind = Literal["model", "reconstruction_level", "gt"]


@dataclass(frozen=True)
class PackageDiagnostic:
    code: str
    message: str
    clip_id: str | None = None
    source_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "clip_id": self.clip_id,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class PackageAsset:
    logical_name: str
    relpath: PurePosixPath
    motion_format: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "relpath": str(self.relpath),
            "motion_format": self.motion_format,
        }


@dataclass(frozen=True)
class SourceAssets:
    source_id: str
    kind: SourceKind
    label: str
    role: str | None = None
    native_rep: str | None = None
    num_codebooks_used: int | None = None
    mse_vs_gt: float | None = None
    fit_mse: float | None = None
    variants: dict[str, PackageAsset] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "label": self.label,
            "role": self.role,
            "native_rep": self.native_rep,
            "num_codebooks_used": self.num_codebooks_used,
            "mse_vs_gt": self.mse_vs_gt,
            "fit_mse": self.fit_mse,
            "variants": {name: asset.to_json() for name, asset in self.variants.items()},
        }


@dataclass(frozen=True)
class PackageSourceSummary:
    source_id: str
    kind: SourceKind
    role: str | None = None
    native_rep: str | None = None
    num_codebooks_used: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "role": self.role,
            "native_rep": self.native_rep,
            "num_codebooks_used": self.num_codebooks_used,
        }


@dataclass(frozen=True)
class PackageTaskSummary:
    task: PackageTask
    num_clips: int
    provenances: tuple[str, ...]
    sources: tuple[PackageSourceSummary, ...]
    reconstruction_levels: tuple[int, ...] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "num_clips": self.num_clips,
            "provenances": list(self.provenances),
            "sources": [source.to_json() for source in self.sources],
            "reconstruction_levels": list(self.reconstruction_levels)
            if self.reconstruction_levels is not None
            else None,
        }


@dataclass(frozen=True)
class PackageClipSummary:
    clip_id: str
    dir_name: str
    task: PackageTask
    provenance: str | None
    rec_id: str | None
    caption: str | None
    fps: float
    frames: int
    prefix_T: int
    predicted_T: int | None
    sources: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "dir_name": self.dir_name,
            "task": self.task,
            "provenance": self.provenance,
            "rec_id": self.rec_id,
            "caption": self.caption,
            "fps": self.fps,
            "frames": self.frames,
            "prefix_T": self.prefix_T,
            "predicted_T": self.predicted_T,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class PackageInspection:
    path: Path
    protocol_version: str
    track: str
    split: str
    fps: float
    num_clips: int
    tasks: dict[str, PackageTaskSummary]
    clips: tuple[PackageClipSummary, ...]
    diagnostics: tuple[PackageDiagnostic, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "protocol_version": self.protocol_version,
            "track": self.track,
            "split": self.split,
            "fps": self.fps,
            "num_clips": self.num_clips,
            "tasks": {name: section.to_json() for name, section in self.tasks.items()},
            "clips": [clip.to_json() for clip in self.clips],
            "diagnostics": [item.to_json() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class PackageSelection:
    task: str | None = None
    clip_id: str | None = None
    sources: tuple[str, ...] = ()
    include_gt: bool = False
    asset_preference: AssetPreference = "smplx"


@dataclass(frozen=True)
class PackageRenderRequest:
    path: Path
    selection: PackageSelection = field(default_factory=PackageSelection)
    cache_dir: Path = Path("data/local/packages")
    output_dir: Path | None = None
    mp4_name: str | None = None
    columns: int = 0
    validation: ValidationMode = "assets"
