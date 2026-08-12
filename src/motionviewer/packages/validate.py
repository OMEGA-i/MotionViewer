from __future__ import annotations

from .errors import PackageValidationError
from .index import PackageIndex
from .store import PackageStore
from .types import PackageDiagnostic, ValidationMode


def collect_diagnostics(
    index: PackageIndex,
    store: PackageStore,
    *,
    mode: ValidationMode,
) -> list[PackageDiagnostic]:
    if mode == "none":
        return []

    diagnostics: list[PackageDiagnostic] = []
    diagnostics.extend(_structural_diagnostics(index))
    if mode in ("assets", "loadable"):
        diagnostics.extend(_asset_diagnostics(index, store))
    return diagnostics


def ensure_valid(
    index: PackageIndex,
    store: PackageStore,
    *,
    mode: ValidationMode,
) -> list[PackageDiagnostic]:
    diagnostics = collect_diagnostics(index, store, mode=mode)
    hard = [item for item in diagnostics if not item.code.startswith("warning")]
    if hard:
        messages = "; ".join(item.message for item in hard[:5])
        raise PackageValidationError(messages)
    return diagnostics


def _structural_diagnostics(index: PackageIndex) -> list[PackageDiagnostic]:
    diagnostics: list[PackageDiagnostic] = []
    if index.fps <= 0:
        diagnostics.append(PackageDiagnostic("invalid_fps", "manifest.fps must be > 0"))
    if index.num_clips and index.num_clips != len(index.clip_order):
        diagnostics.append(
            PackageDiagnostic(
                "clip_count_mismatch",
                f"manifest.num_clips={index.num_clips} but found {len(index.clip_order)} clips",
            )
        )

    for task_name, section in index.tasks.items():
        task_clip_count = sum(1 for key in index.clip_order if key[0] == task_name)
        if section.num_clips and section.num_clips != task_clip_count:
            diagnostics.append(
                PackageDiagnostic(
                    "task_clip_count_mismatch",
                    f"manifest.tasks[{task_name}].num_clips={section.num_clips} but found {task_clip_count} clips",
                )
            )

        manifest_ids = {source.source_id for source in section.sources}
        if section.task == "recon":
            if not section.reconstruction_levels:
                diagnostics.append(
                    PackageDiagnostic(
                        "missing_reconstruction_levels",
                        f"recon task {task_name!r} requires reconstruction_levels",
                    )
                )
            else:
                expected = {f"q{level:02d}" for level in section.reconstruction_levels}
                missing = sorted(expected - manifest_ids)
                extra = sorted(
                    source.source_id
                    for source in section.sources
                    if source.kind == "reconstruction_level" and source.source_id not in expected
                )
                if missing:
                    diagnostics.append(
                        PackageDiagnostic(
                            "recon_level_mismatch",
                            f"task {task_name}: reconstruction_levels missing source ids: {', '.join(missing)}",
                        )
                    )
                if extra:
                    diagnostics.append(
                        PackageDiagnostic(
                            "recon_level_extra",
                            f"task {task_name}: unexpected reconstruction source ids: {', '.join(extra)}",
                        )
                    )

        # Manifest sources are a union: each must appear in ≥1 clip; per-clip subsets are OK.
        present_ids: set[str] = set()
        for clip in index.clips_for_task(task_name):
            clip_id = clip.clip_id
            present_ids.update(clip.sources)
            if clip.frames <= 0:
                diagnostics.append(
                    PackageDiagnostic("invalid_frames", f"clip {clip_id} has non-positive T", clip_id=clip_id)
                )
            if clip.fps <= 0:
                diagnostics.append(
                    PackageDiagnostic(
                        "invalid_clip_fps", f"clip {clip_id} has non-positive fps", clip_id=clip_id
                    )
                )
            if section.task == "t2m" and not clip.caption:
                diagnostics.append(
                    PackageDiagnostic(
                        "missing_caption", f"t2m clip {clip_id} is missing caption", clip_id=clip_id
                    )
                )
            if section.task == "pred" and clip.prefix_T <= 0:
                diagnostics.append(
                    PackageDiagnostic(
                        "missing_prefix", f"pred clip {clip_id} requires prefix_T > 0", clip_id=clip_id
                    )
                )
            if "smplx_params" not in clip.gt.variants and "joints22" not in clip.gt.variants:
                diagnostics.append(
                    PackageDiagnostic(
                        "missing_gt_assets",
                        f"clip {clip_id} gt has no smplx_params/joints22",
                        clip_id=clip_id,
                    )
                )
            for source_id, source in clip.sources.items():
                if source_id not in manifest_ids:
                    diagnostics.append(
                        PackageDiagnostic(
                            "extra_clip_source",
                            f"clip {clip_id} has source {source_id} not declared in manifest.tasks[{task_name}].sources",
                            clip_id=clip_id,
                            source_id=source_id,
                        )
                    )
                if "smplx_params" not in source.variants and "joints22" not in source.variants:
                    diagnostics.append(
                        PackageDiagnostic(
                            "missing_source_assets",
                            f"clip {clip_id} source {source_id} has no smplx_params/joints22",
                            clip_id=clip_id,
                            source_id=source_id,
                        )
                    )

        unused = sorted(manifest_ids - present_ids)
        for source_id in unused:
            diagnostics.append(
                PackageDiagnostic(
                    "unused_manifest_source",
                    f"manifest.tasks[{task_name}] source {source_id} does not appear in any clip",
                    source_id=source_id,
                )
            )
    return diagnostics


def _asset_diagnostics(index: PackageIndex, store: PackageStore) -> list[PackageDiagnostic]:
    diagnostics: list[PackageDiagnostic] = []
    for _key, clip in index.clips.items():
        clip_id = clip.clip_id
        for source in (clip.gt, *clip.sources.values()):
            for logical_name, asset in source.variants.items():
                if logical_name not in ("smplx_params", "joints22", "joints77", "native"):
                    continue
                try:
                    exists = store.exists(asset.relpath)
                except Exception as exc:  # noqa: BLE001 - surface as diagnostic
                    diagnostics.append(
                        PackageDiagnostic(
                            "unsafe_asset_path",
                            f"{clip_id}/{source.source_id}/{logical_name}: {exc}",
                            clip_id=clip_id,
                            source_id=source.source_id,
                        )
                    )
                    continue
                if not exists:
                    diagnostics.append(
                        PackageDiagnostic(
                            "missing_asset_file",
                            f"missing file {asset.relpath}",
                            clip_id=clip_id,
                            source_id=source.source_id,
                        )
                    )
    return diagnostics
