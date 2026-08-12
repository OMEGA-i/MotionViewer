from __future__ import annotations

from dataclasses import dataclass

from .errors import PackageSelectionError
from .index import ClipIndex, PackageIndex, TaskSection
from .types import AssetPreference, PackageAsset, PackageSelection, SourceAssets


@dataclass(frozen=True)
class PlannedInput:
    source_id: str
    label: str
    asset: PackageAsset
    source: SourceAssets


@dataclass(frozen=True)
class PackagePlan:
    clip: ClipIndex
    task_section: TaskSection
    inputs: tuple[PlannedInput, ...]


def plan_package_selection(index: PackageIndex, selection: PackageSelection) -> PackagePlan:
    task_name = _resolve_task(index, selection.task)
    section = index.tasks[task_name]
    clip = _select_clip(index, task_name, selection.clip_id)
    source_ids = _resolve_source_ids(section, clip, selection)
    if not source_ids:
        raise PackageSelectionError("Source selection is empty; pass --sources <one source id>")

    planned: list[PlannedInput] = []
    for source_id in source_ids:
        source = clip.gt if source_id == "gt" else clip.sources.get(source_id)
        if source is None:
            available = ", ".join(("gt", *clip.sources.keys()))
            raise PackageSelectionError(
                f"Unknown source {source_id!r} for clip {clip.clip_id!r}. Available: {available}"
            )
        asset = _prefer_asset(source, selection.asset_preference)
        planned.append(
            PlannedInput(
                source_id=source_id,
                label=source.label,
                asset=asset,
                source=source,
            )
        )
    return PackagePlan(clip=clip, task_section=section, inputs=tuple(planned))


def _resolve_task(index: PackageIndex, task: str | None) -> str:
    if task is not None:
        if task not in index.tasks:
            available = ", ".join(sorted(index.tasks))
            raise PackageSelectionError(f"Unknown task {task!r}. Available: {available}")
        return task
    if len(index.tasks) == 1:
        return next(iter(index.tasks))
    available = ", ".join(sorted(index.tasks))
    raise PackageSelectionError(f"Package has multiple tasks; pass --task. Available: {available}")


def _select_clip(index: PackageIndex, task: str, clip_id: str | None) -> ClipIndex:
    task_clips = index.clips_for_task(task)
    if not task_clips:
        raise PackageSelectionError(f"Task {task!r} has no clips")
    if clip_id is None:
        return task_clips[0]

    matches = [clip for clip in task_clips if _clip_matches(clip, clip_id)]
    if not matches:
        available = ", ".join(clip.clip_id for clip in task_clips)
        raise PackageSelectionError(f"Unknown clip {clip_id!r} in task {task!r}. Available: {available}")
    if len(matches) > 1:
        raise PackageSelectionError(f"Clip id {clip_id!r} is ambiguous in task {task!r}")
    return matches[0]


def _clip_matches(clip: ClipIndex, query: str) -> bool:
    candidates = {
        clip.clip_id,
        clip.dir_name,
        clip.clip_id.replace(":", "_"),
        clip.dir_name.replace("_", ":", 1) if "_" in clip.dir_name else clip.dir_name,
    }
    return query in candidates


def _resolve_source_ids(
    section: TaskSection,
    clip: ClipIndex,
    selection: PackageSelection,
) -> list[str]:
    if selection.sources:
        ordered = list(selection.sources)
        if selection.include_gt and "gt" not in ordered:
            ordered = ["gt", *ordered]
        if not selection.include_gt:
            ordered = [source_id for source_id in ordered if source_id != "gt"]
        return ordered

    # Default: exactly one model/recon source for single-actor jobs.
    if section.task == "recon" and section.reconstruction_levels:
        for level in section.reconstruction_levels:
            sid = f"q{level:02d}"
            if sid in clip.sources:
                return [sid]
    for source in section.sources:
        if source.source_id in clip.sources:
            return [source.source_id]
    if clip.sources:
        return [next(iter(clip.sources))]
    if selection.include_gt:
        return ["gt"]
    return []


def _prefer_asset(source: SourceAssets, preference: AssetPreference) -> PackageAsset:
    if preference == "smplx":
        if "smplx_params" in source.variants:
            return source.variants["smplx_params"]
        raise PackageSelectionError(
            f"Source {source.source_id!r} has no smplx_params.npz; set asset_preference='joints22' to fallback"
        )
    if "joints22" in source.variants:
        return source.variants["joints22"]
    raise PackageSelectionError(f"Source {source.source_id!r} has no joints22.npy")
