from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

from motionviewer.loaders import default_registry
from motionviewer.video.spec import RenderJob

from .errors import PackageError, PackageFormatError, PackagePayloadError
from .index import SUPPORTED_PROTOCOL_VERSION, build_package_index
from .plan import plan_package_selection
from .render import build_render_job
from .store import open_package_store, probe_package
from .types import (
    PackageDiagnostic,
    PackageInspection,
    PackageRenderRequest,
    PackageSelection,
    ValidationMode,
)
from .validate import collect_diagnostics, ensure_valid


def is_package(path: str | Path) -> bool:
    return probe_package(path)


def inspect_package(path: str | Path, *, validation: ValidationMode = "structural") -> PackageInspection:
    store = open_package_store(path)
    try:
        index = build_package_index(store)
        diagnostics = collect_diagnostics(index, store, mode=validation)
        return PackageInspection(
            path=Path(path),
            protocol_version=index.protocol_version,
            track=index.track,
            split=index.split,
            fps=index.fps,
            num_clips=len(index.clip_order),
            tasks=index.task_summaries(),
            clips=index.clip_summaries(),
            diagnostics=tuple(diagnostics),
        )
    finally:
        store.close()


def validate_package(path: str | Path, *, mode: ValidationMode = "assets") -> list[PackageDiagnostic]:
    store = open_package_store(path)
    try:
        index = build_package_index(store)
        return collect_diagnostics(index, store, mode=mode)
    finally:
        store.close()


def import_package(
    path: str | Path,
    dest_root: str | Path = Path("data/packages"),
    *,
    force: bool = False,
) -> Path:
    """Validate a v2 package and copy it as-is into dest_root/{track}_{split}[.tar.gz]."""
    source = Path(path)
    store = open_package_store(source)
    try:
        if not store.exists("manifest.json"):
            raise PackageFormatError("Package is missing manifest.json")
        manifest = store.read_json("manifest.json")
        protocol_version = str(manifest.get("protocol_version", ""))
        if protocol_version != SUPPORTED_PROTOCOL_VERSION:
            raise PackageFormatError(
                f"Unsupported protocol_version {protocol_version!r}; expected {SUPPORTED_PROTOCOL_VERSION!r}"
            )
        track = str(manifest.get("track") or "")
        split = str(manifest.get("split") or "")
        if not track or not split:
            raise PackageFormatError("manifest.json missing required fields 'track' and 'split'")
        # Ensure the package is structurally openable as a full v2 index.
        build_package_index(store)
        stem = f"{track}_{split}"
        copy_source = Path(getattr(store, "root", source)) if source.is_dir() else source
    finally:
        store.close()

    dest_root_path = Path(dest_root)
    dest_root_path.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        dest = dest_root_path / stem
        if dest.exists() and not force:
            raise PackagePayloadError(f"Destination already exists: {dest} (pass force=True to overwrite)")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(copy_source, dest)
        return dest.resolve()

    # Preserve compression suffix when possible.
    name = source.name.lower()
    if name.endswith(".tar.gz"):
        dest = dest_root_path / f"{stem}.tar.gz"
    elif name.endswith(".tgz"):
        dest = dest_root_path / f"{stem}.tgz"
    elif name.endswith(".tar"):
        dest = dest_root_path / f"{stem}.tar"
    else:
        dest = dest_root_path / f"{stem}.tar.gz"

    if dest.exists() and not force:
        raise PackagePayloadError(f"Destination already exists: {dest} (pass force=True to overwrite)")
    shutil.copy2(source, dest)
    return dest.resolve()


def build_render_job_from_package(request: PackageRenderRequest) -> RenderJob:
    store = open_package_store(request.path)
    try:
        index = build_package_index(store)
        ensure_valid(index, store, mode=request.validation if request.validation != "loadable" else "assets")
        plan = plan_package_selection(index, request.selection)

        package_stem = _package_stem(Path(request.path))
        safe_clip = plan.clip.clip_id.replace(":", "_")
        cache_root = Path(request.cache_dir) / package_stem / plan.clip.task / safe_clip
        materialize_pairs: list[tuple[PurePosixPath, Path]] = []
        materialized: dict[str, Path] = {}
        for planned in plan.inputs:
            dest = cache_root / planned.source_id / Path(planned.asset.relpath.name)
            materialize_pairs.append((planned.asset.relpath, dest))
            materialized[planned.source_id] = dest

        try:
            store.materialize(materialize_pairs)
        except PackagePayloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PackagePayloadError(f"Failed to materialize selected assets: {exc}") from exc

        if request.validation == "loadable":
            registry = default_registry()
            for planned in plan.inputs:
                path = materialized[planned.source_id]
                format_id = planned.asset.motion_format
                options = {"label": planned.label, "fps": plan.clip.fps}
                registry.load(path, format_id=format_id, options=options)

        return build_render_job(index, plan, materialized, request)
    finally:
        store.close()


def write_package_render_config(request: PackageRenderRequest, output: str | Path) -> Path:
    import yaml

    job = build_render_job_from_package(request)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(job.to_json(), handle, sort_keys=False)
    return out_path


def _package_stem(path: Path) -> str:
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem if path.is_file() else path.name


__all__ = [
    "PackageError",
    "PackageFormatError",
    "build_render_job_from_package",
    "import_package",
    "inspect_package",
    "is_package",
    "validate_package",
    "write_package_render_config",
    "PackageRenderRequest",
    "PackageSelection",
]
