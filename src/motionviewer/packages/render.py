from __future__ import annotations

from pathlib import Path

from motionviewer.video.spec import RenderJob

from .errors import PackageValidationError
from .index import PackageIndex
from .plan import PackagePlan
from .types import PackageRenderRequest

TASK_MODE = {
    "t2m": "text_to_motion",
    "pred": "continuation",
    "recon": "comparison",
}


def build_render_job(
    index: PackageIndex,
    plan: PackagePlan,
    materialized: dict[str, Path],
    request: PackageRenderRequest,
) -> RenderJob:
    items = []
    for planned in plan.inputs:
        path = materialized[planned.source_id]
        format_id = planned.asset.motion_format
        if format_id is None:
            raise PackageValidationError(f"Selected asset for {planned.source_id} has no motion format")
        loader_options: dict = {"label": planned.label, "fps": plan.clip.fps}
        items.append(
            {
                "path": str(path.resolve()),
                "label": planned.label,
                "format": format_id,
                "loader_options": loader_options,
            }
        )

    task = plan.clip.task
    task_mode = TASK_MODE[task]
    instruction = plan.clip.caption
    safe_clip = plan.clip.clip_id.replace(":", "_").replace("/", "_")
    provenance = plan.clip.provenance or "unknown"
    clip_base = f"{task}_{index.track}_{provenance}_{safe_clip}"

    output_directory = request.output_dir
    if output_directory is None:
        if len(plan.inputs) == 1:
            source_id = plan.inputs[0].source_id
            output_directory = Path("outputs") / f"{clip_base}_{source_id}"
        else:
            output_directory = Path("outputs") / clip_base

    if request.mp4_name:
        mp4_name = request.mp4_name
    elif len(plan.inputs) == 1:
        mp4_name = f"{plan.inputs[0].source_id}.mp4"
    else:
        mp4_name = f"{safe_clip}.mp4"

    job = RenderJob.template(
        items,
        task_mode=task_mode,
        instruction=instruction,
        output_directory=str(Path(output_directory).resolve()),
        mp4_name=mp4_name,
    )
    if request.columns:
        job.layout.columns = request.columns
    if task == "pred":
        job.timeline.show_prefix = True
    else:
        job.timeline.show_prefix = False

    if task == "recon":
        job.style.ghost_snapshots = 0
        job.style.ghost.mode = "none"
        job.style.prefix.ghost_count = 0

    errors = job.validate()
    if errors:
        raise PackageValidationError("; ".join(errors))
    return job
