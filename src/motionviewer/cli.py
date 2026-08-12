from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
import yaml

from motionviewer.loaders import default_registry
from motionviewer.packages import (
    PackageFormatError,
    PackagePayloadError,
    PackageRenderRequest,
    PackageSelection,
    PackageValidationError,
    build_render_job_from_package,
    import_package,
    inspect_package,
    is_package,
)
from motionviewer.video.job import (
    load_render_job,
    prepare_render_job,
    quick_job,
    write_bundle,
    write_manifest,
)
from motionviewer.video.spec import RenderJob, group_views_by_staging, resolve_camera_views

app = typer.Typer(help="Inspect and render motion sequences with Blender.", no_args_is_help=True)
motion_app = typer.Typer(help="Inspect motion files and create render configs.", no_args_is_help=True)
package_app = typer.Typer(help="Inspect, import, and preview protocol packages.", no_args_is_help=True)
render_app = typer.Typer(help="Prepare and execute render workflows.", no_args_is_help=True)
fbx_app = typer.Typer(help="Validate and audit local FBX character assets.", no_args_is_help=True)
app.add_typer(motion_app, name="motion")
app.add_typer(package_app, name="package")
app.add_typer(render_app, name="render")
app.add_typer(fbx_app, name="fbx")


@package_app.command("inspect")
@motion_app.command("inspect")
def inspect(
    path: Annotated[Path, typer.Argument(help="Motion file or Sample Packing Protocol package.")],
    format: Annotated[
        str | None, typer.Option("--format", help="Explicit format id for motion files.")
    ] = None,
    task: Annotated[str | None, typer.Option("--task", help="Package task filter (t2m/pred/recon).")] = None,
    clip: Annotated[str | None, typer.Option("--clip", help="Package clip id to highlight.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    if is_package(path):
        inspection = inspect_package(path, validation="structural")
        selected_clips = inspection.clips
        if task is not None:
            selected_clips = tuple(item for item in selected_clips if item.task == task)
        if clip is not None:
            selected_clips = tuple(
                item
                for item in selected_clips
                if item.clip_id == clip or item.dir_name == clip or item.clip_id.replace(":", "_") == clip
            )
            if not selected_clips:
                raise typer.BadParameter(f"Clip {clip!r} not found in package")
        if json_output:
            payload = inspection.to_json()
            payload["clips"] = [item.to_json() for item in selected_clips]
            typer.echo(json.dumps(payload, indent=2))
            return
        typer.echo(f"path: {inspection.path}")
        typer.echo(f"protocol: {inspection.protocol_version}")
        typer.echo(f"track: {inspection.track}")
        typer.echo(f"split: {inspection.split}")
        typer.echo(f"fps: {inspection.fps}")
        typer.echo(f"clips: {inspection.num_clips}")
        typer.echo(f"tasks: {', '.join(inspection.tasks)}")
        for task_name, section in inspection.tasks.items():
            source_ids = ", ".join(source.source_id for source in section.sources)
            typer.echo(f"  {task_name}: clips={section.num_clips} sources={source_ids}")
            if section.reconstruction_levels:
                typer.echo(f"    reconstruction_levels: {list(section.reconstruction_levels)}")
        for item in selected_clips[:20]:
            caption = item.caption or ""
            preview = caption if len(caption) <= 96 else caption[:93] + "..."
            provenance = item.provenance or "unknown"
            typer.echo(
                f"- [{item.task}] {item.clip_id}  provenance={provenance}  "
                f"T={item.frames}  prefix_T={item.prefix_T}  sources={','.join(item.sources)}"
            )
            if preview:
                typer.echo(f"    caption: {preview}")
        if len(selected_clips) > 20:
            typer.echo(f"... {len(selected_clips) - 20} more clips")
        if inspection.diagnostics:
            typer.echo(f"diagnostics: {len(inspection.diagnostics)}")
            for diag in inspection.diagnostics[:10]:
                typer.echo(f"  - {diag.code}: {diag.message}")
        return

    registry = default_registry()
    sequence = registry.load(path, format_id=format)
    summary = sequence.to_json_summary()
    if json_output:
        typer.echo(json.dumps(summary, indent=2))
        return
    typer.echo(f"path: {summary['path']}")
    typer.echo(f"format: {summary['format_id']}")
    typer.echo(f"source: {summary['source']}")
    typer.echo(f"frames: {summary['frames']}")
    typer.echo(f"fps: {summary['fps']}")
    typer.echo(f"duration_s: {summary['duration_s']:.3f}")
    typer.echo(f"capabilities: {', '.join(summary['capabilities'])}")
    typer.echo(f"segments: {summary['segments']}")
    typer.echo(f"bounds: {summary['bounds']}")
    typer.echo(f"body_model: {summary['body_model'] is not None}")


@package_app.command("import")
def import_package_cmd(
    path: Annotated[Path, typer.Argument(help="Sample Packing Protocol v2 package (.tar.gz or directory).")],
    dest: Annotated[Path, typer.Option("--dest", help="Destination root for imported packages.")] = Path(
        "data/packages"
    ),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing imported package.")] = False,
) -> None:
    """Validate a v2 package and copy it as-is into data/packages/{track}_{split}."""
    try:
        dest_path = import_package(path, dest_root=dest, force=force)
    except (PackagePayloadError, PackageFormatError, PackageValidationError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Imported package to {dest_path}")


@package_app.command("config")
@motion_app.command("config")
def make_config(
    inputs: Annotated[list[Path], typer.Argument(help="One motion file, or a single package path.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Config YAML path.")] = Path(
        "configs/examples/generated.yaml"
    ),
    task: Annotated[
        str | None, typer.Option("--task", help="Package task (required when multiple tasks).")
    ] = None,
    clip: Annotated[str | None, typer.Option("--clip", help="Package clip id (default: first clip).")] = None,
    sources: Annotated[
        str | None,
        typer.Option("--sources", help="Exactly one package source id (default: first model source)."),
    ] = None,
    include_gt: Annotated[
        bool, typer.Option("--gt/--no-gt", help="Include ground-truth (not for single-actor jobs).")
    ] = False,
    asset: Annotated[
        str, typer.Option("--asset", help="Preferred package asset: smplx or joints22.")
    ] = "smplx",
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Local cache root for materialized package assets.")
    ] = Path("data/local/packages"),
) -> None:
    if len(inputs) == 1 and is_package(inputs[0]):
        if asset not in ("smplx", "joints22"):
            raise typer.BadParameter("--asset must be 'smplx' or 'joints22'")
        source_tuple = tuple(item.strip() for item in sources.split(",") if item.strip()) if sources else ()
        if include_gt:
            raise typer.BadParameter(
                "Single-actor configs cannot include gt with a model; render gt and models in separate jobs"
            )
        if len(source_tuple) > 1:
            raise typer.BadParameter("Pass exactly one --sources id (one actor per job; compose in PPT)")
        selection = PackageSelection(
            task=task,
            clip_id=clip,
            sources=source_tuple,
            include_gt=False,
            asset_preference=asset,  # type: ignore[arg-type]
        )
        request = PackageRenderRequest(
            path=inputs[0],
            selection=selection,
            cache_dir=cache_dir,
            validation="assets",
        )
        job = build_render_job_from_package(request)
        if len(job.inputs) != 1:
            raise typer.BadParameter(
                f"Package selection resolved to {len(job.inputs)} inputs; pass exactly one --sources id"
            )
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(job.to_json(), handle, sort_keys=False)
        typer.echo(f"Task: {job.task.mode}")
        if job.task.instruction:
            preview = (
                job.task.instruction if len(job.task.instruction) <= 96 else job.task.instruction[:93] + "..."
            )
            typer.echo(f"Instruction: {preview}")
        typer.echo(f"Inputs: {', '.join(item.label or item.path.name for item in job.inputs)}")
        typer.echo(f"Wrote {out_path}")
        return

    if len(inputs) != 1:
        raise typer.BadParameter(
            "make-config accepts exactly one motion file (or one package). "
            "Render each model separately and compose comparisons in PPT."
        )
    registry = default_registry()
    path = inputs[0]
    seq = registry.load(path)
    items = [{"path": str(path), "label": seq.source, "format": seq.format_id}]
    data = RenderJob.template(items).to_json()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    typer.echo(f"Wrote {output}")


@fbx_app.command("check")
def check_fbx(
    root: Annotated[Path, typer.Option("--root", help="FBX asset root to scan.")] = Path("assets/fbx"),
    deep: Annotated[
        bool, typer.Option("--deep", help="Include catalog evidence/proposed status report.")
    ] = False,
    motions: Annotated[
        list[Path] | None,
        typer.Option("--motion", help="SMPL-X motion for deep audit; repeat for a regression matrix."),
    ] = None,
    blender: Annotated[
        Path | None,
        typer.Option("--blender", help="Blender executable used by --deep."),
    ] = None,
    write_report: Annotated[
        Path | None,
        typer.Option("--write-report", help="Write fbx_validation_report.json to this directory or file."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Validate registered non-PMX FBX assets for the fbx_skeleton backend."""
    from motionviewer.assets.fbx_catalog import list_fbx_models

    models = list_fbx_models(root)
    quality_matrix = None
    if deep:
        default_root = Path(__file__).resolve().parents[2]
        motion_paths = motions or [
            default_root / "data/examples/smplx_body22_fitted_aa/gt.smplx.npz",
            default_root / "data/examples/smplx_body22_fitted_aa/mdm.smplx.npz",
            default_root / "data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz",
        ]
        candidates = [model.path for model in models if model.status in {"pending", "approved"}]
        from motionviewer.blender.executable import resolve_blender

        quality_matrix = _run_fbx_quality_matrix(resolve_blender(blender), motion_paths, candidates)
    report = _fbx_report(root, models, deep=deep, quality_matrix=quality_matrix)
    if write_report is not None:
        report_path = _write_fbx_report(report, write_report)
        typer.echo(f"Wrote FBX report: {report_path}", err=json_output)
    if json_output:
        typer.echo(json.dumps(report if deep else [model.to_json() for model in models], indent=2))
        return
    typer.echo(f"root: {root}")
    if not models:
        typer.echo("No FBX files found")
        return
    valid = [model for model in models if model.valid]
    invalid = [model for model in models if not model.valid]
    typer.echo(f"valid: {len(valid)}")
    for model in valid:
        typer.echo(f"  OK  {model.model_id}: {model.path}")
    if invalid:
        typer.echo(f"invalid: {len(invalid)}")
        for model in invalid:
            typer.echo(f"  BAD {model.model_id}: {model.reason}")


def _fbx_report(
    root: Path,
    models: list,
    *,
    deep: bool = False,
    quality_matrix: list[dict] | None = None,
) -> dict:
    valid = [model for model in models if model.valid]
    invalid = [model for model in models if not model.valid]
    report = {
        "root": str(root),
        "summary": {
            "total": len(models),
            "valid": len(valid),
            "invalid": len(invalid),
        },
        "models": [model.to_json() for model in models],
    }
    if deep:
        audits_by_model: dict[str, list[dict]] = {}
        for motion_report in quality_matrix or ():
            for asset_report in motion_report.get("assets", ()):
                model_id = Path(asset_report["asset"]).stem.lower().replace(" ", "_")
                audits_by_model.setdefault(model_id, []).append(asset_report)
        report["proposed_statuses"] = {
            model.model_id: (
                (
                    "approved"
                    if all(
                        item.get("quality_gate", {}).get("passed") for item in audits_by_model[model.model_id]
                    )
                    else "pending"
                )
                if audits_by_model.get(model.model_id)
                else model.status
            )
            for model in models
        }
        report["quality_matrix"] = quality_matrix or []
    return report


def _run_fbx_quality_matrix(
    blender: Path,
    motions: list[Path],
    assets: list[Path],
) -> list[dict]:
    """Run independent CPU-only Blender audits, one process per motion."""
    missing = [str(path) for path in (blender, *motions, *assets) if not path.is_file()]
    if missing:
        raise typer.BadParameter("Deep FBX audit inputs do not exist: " + ", ".join(missing))
    if not assets:
        return []
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/retarget_quality_audit.py"
    with tempfile.TemporaryDirectory(prefix="motionviewer-fbx-audit-") as temp_dir:
        outputs = [Path(temp_dir) / f"motion-{index}.json" for index in range(len(motions))]

        def run(index: int) -> dict:
            command = [
                str(blender),
                "--background",
                "--python",
                str(script),
                "--",
                "--motion",
                str(motions[index]),
                "--output",
                str(outputs[index]),
            ]
            for asset in assets:
                command.extend(("--asset", str(asset)))
            subprocess.run(
                command,
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            return json.loads(outputs[index].read_text(encoding="utf-8"))

        with ThreadPoolExecutor(max_workers=min(3, len(motions))) as pool:
            futures = {pool.submit(run, index): index for index in range(len(motions))}
            reports = {futures[future]: future.result() for future in as_completed(futures)}
    return [reports[index] for index in range(len(motions))]


def _write_fbx_report(report: dict, output: Path) -> Path:
    report_path = output if output.suffix == ".json" else output / "fbx_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


@package_app.command("preview")
def preview_package(
    package: Annotated[Path, typer.Argument(help="Sample Packing Protocol v2 package.")],
    task: Annotated[str, typer.Option("--task", help="Package task to preview.")] = "t2m",
    source: Annotated[str, typer.Option("--source", help="Source id to preview.")] = "omegamotiongpt",
    clips: Annotated[int, typer.Option("--clips", help="Number of clips to preview.")] = 10,
    frames: Annotated[
        int, typer.Option("--frames", help="Number of uniformly sampled frames per clip.")
    ] = 10,
    body_pool: Annotated[
        str,
        typer.Option("--body-pool", help="Comma-separated body choices: smplx,fbx-random."),
    ] = "smplx",
    fbx_root: Annotated[Path, typer.Option("--fbx-root", help="FBX asset root for fbx-random.")] = Path(
        "assets/fbx"
    ),
    seed: Annotated[int, typer.Option("--seed", help="Deterministic random seed.")] = 0,
    output: Annotated[Path, typer.Option("--output", "-o", help="Preview output directory.")] = Path(
        "outputs/preview/package"
    ),
    blender: Annotated[
        str | None, typer.Option("--blender", help="Render generated configs with Blender.")
    ] = None,
) -> None:
    """Generate deterministic N-clip × N-frame preview jobs from a package source."""
    from motionviewer.video.preview import generate_package_preview

    pool = tuple(item.strip() for item in body_pool.split(",") if item.strip())
    try:
        jobs = generate_package_preview(
            package,
            task=task,
            source_id=source,
            clip_count=clips,
            frame_count=frames,
            output_dir=output,
            body_pool=pool,
            fbx_root=fbx_root,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 - Typer should display data/setup issues succinctly
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Wrote {len(jobs)} preview configs under {output}")
    typer.echo(f"Selection: {output / 'selection.json'}")
    if blender is None:
        typer.echo("Pass --blender /path/to/blender to render the generated configs.")
        return

    for job in jobs:
        typer.echo(f"Rendering {job.index:02d}: {job.clip_id}")
        result = subprocess.run(
            [
                "uv",
                "run",
                "motionviewer",
                "render",
                "job",
                "--config",
                str(job.config_path),
                "--blender",
                blender,
                "--no-parallel-views",
            ],
            check=False,
        )
        if result.returncode != 0:
            raise typer.Exit(code=result.returncode)


@render_app.command("qualitative")
def render_qualitative(
    package: Annotated[Path, typer.Argument(help="Sample Packing Protocol v2 package.")],
    clip_id: Annotated[
        list[str] | None,
        typer.Option("--clip-id", help="Exact clip id to render; repeat for multiple clips."),
    ] = None,
    sources: Annotated[
        str,
        typer.Option("--sources", help="Comma-separated source ids rendered separately per clip."),
    ] = "omegamotiongpt,kimodo,hymotion",
    provenance: Annotated[
        list[str] | None,
        typer.Option("--provenance", help="Repeatable provenance quota in NAME=COUNT form."),
    ] = None,
    task: Annotated[str, typer.Option("--task", help="Package task to render.")] = "t2m",
    seed: Annotated[int, typer.Option("--seed", help="Deterministic selection and FBX seed.")] = 20260726,
    snapshots: Annotated[int, typer.Option("--snapshots", help="Visible temporal poses per image.")] = 6,
    snapshot_layout: Annotated[
        str,
        typer.Option(
            "--snapshot-layout",
            help="Snapshot placement: root_aligned is a row, arc is a ground-plane semicircle, trajectory preserves travel.",
        ),
    ] = "root_aligned",
    snapshot_spacing: Annotated[
        float,
        typer.Option("--snapshot-spacing", help="Root-aligned snapshot center spacing in meters."),
    ] = 1.25,
    arc_direction: Annotated[
        str,
        typer.Option("--arc-direction", help="Arc bend direction: up (∩) or down (∪)."),
    ] = "up",
    material_mode: Annotated[
        str,
        typer.Option("--material-mode", help="Snapshot materials: palette or preserve."),
    ] = "preserve",
    palette_start: Annotated[
        str,
        typer.Option("--palette-start", help="Palette light RGB as R,G,B in [0,255]."),
    ] = "26,128,184",
    palette_end: Annotated[
        str,
        typer.Option("--palette-end", help="Palette dark RGB as R,G,B in [0,255]."),
    ] = "122,26,158",
    palette_color: Annotated[
        str | None,
        typer.Option("--palette-color", help="Use one RGB color for every snapshot."),
    ] = None,
    snapshot_alpha: Annotated[
        float,
        typer.Option("--snapshot-alpha", help="Opacity for each palette snapshot in (0,1]."),
    ] = 1.0,
    body_mode: Annotated[
        str,
        typer.Option("--body-mode", help="Snapshot body: fbx retarget, smplh, or smplx mesh."),
    ] = "fbx",
    foot_pose: Annotated[
        str,
        typer.Option(
            "--foot-pose",
            help="Foot channels: source, ankle_neutral, or neutral_feet (ankles and toes).",
        ),
    ] = "source",
    fbx_pool: Annotated[
        str,
        typer.Option(
            "--fbx-pool",
            help="FBX pool: approved (paper-safe) or all_binary (exploratory pending assets included).",
        ),
    ] = "approved",
    exclude_fbx: Annotated[
        str,
        typer.Option("--exclude-fbx", help="Comma-separated FBX model ids excluded from sampling."),
    ] = "",
    fbx_root: Annotated[Path, typer.Option("--fbx-root", help="Approved FBX catalog root.")] = Path(
        "assets/fbx"
    ),
    output: Annotated[Path, typer.Option("--output", "-o", help="Qualitative output directory.")] = Path(
        "outputs/qualitative/t2m_100_20260726"
    ),
    resolution: Annotated[
        str, typer.Option("--resolution", help="Output resolution as WIDTHxHEIGHT.")
    ] = "2400x900",
    samples: Annotated[int, typer.Option("--samples", help="Eevee render samples.")] = 32,
    blender: Annotated[
        Path | None,
        typer.Option("--blender", help="Blender executable."),
    ] = None,
    workers: Annotated[int, typer.Option("--workers", help="Parallel Blender clip processes.")] = 2,
    prepare_only: Annotated[
        bool,
        typer.Option("--prepare-only", help="Write selection, assets, and bundles without rendering."),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip clips whose three PNGs already exist."),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Regenerate selection and bundles for this request."),
    ] = False,
) -> None:
    """Render one transparent ghost-snapshot PNG per source and clip."""
    from motionviewer.video.qualitative import (
        DEFAULT_PROVENANCE_COUNTS,
        QualitativeBatchRequest,
        parse_provenance_counts,
        prepare_qualitative_batch,
        run_qualitative_batch,
    )

    source_ids = tuple(item.strip() for item in sources.split(",") if item.strip())
    quotas = DEFAULT_PROVENANCE_COUNTS if provenance is None else parse_provenance_counts(provenance)
    try:
        width_raw, height_raw = resolution.lower().split("x", 1)
        output_resolution = (int(width_raw), int(height_raw))
    except ValueError as exc:
        raise typer.BadParameter("resolution must be WIDTHxHEIGHT") from exc

    def parse_rgb(raw: str, option: str) -> tuple[int, int, int]:
        try:
            values = tuple(int(item.strip()) for item in raw.split(","))
        except ValueError as exc:
            raise typer.BadParameter(f"{option} must be R,G,B with integer values") from exc
        if len(values) != 3 or any(not 0 <= value <= 255 for value in values):
            raise typer.BadParameter(f"{option} must be R,G,B with values in [0,255]")
        return values

    try:
        palette_start_rgb = parse_rgb(palette_start, "--palette-start")
        palette_end_rgb = parse_rgb(palette_end, "--palette-end")
        palette_color_rgb = None if palette_color is None else parse_rgb(palette_color, "--palette-color")
        request = QualitativeBatchRequest(
            package=package,
            output_dir=output,
            sources=source_ids,
            clip_ids=tuple(clip_id or ()),
            provenance_counts=quotas,
            task=task,
            seed=seed,
            snapshots=snapshots,
            snapshot_layout=snapshot_layout,
            snapshot_spacing=snapshot_spacing,
            arc_direction=arc_direction,
            material_mode=material_mode,
            palette_start_rgb=palette_start_rgb,
            palette_end_rgb=palette_end_rgb,
            palette_color_rgb=palette_color_rgb,
            snapshot_alpha=snapshot_alpha,
            body_mode=body_mode,
            foot_pose=foot_pose,
            fbx_pool=fbx_pool,
            exclude_fbx=tuple(item.strip() for item in exclude_fbx.split(",") if item.strip()),
            fbx_root=fbx_root,
            resolution=output_resolution,
            samples=samples,
        )
        batch = prepare_qualitative_batch(request, overwrite=overwrite)
    except Exception as exc:  # noqa: BLE001 - Typer presents preparation errors succinctly
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Selected {len(batch.jobs)} clips; selection: {batch.selection_path}")
    typer.echo(f"Expected PNGs: {len(batch.jobs) * len(request.sources)}")
    if prepare_only:
        typer.echo("Prepared bundles only; Blender rendering skipped.")
        return
    try:
        from motionviewer.blender.executable import resolve_blender

        status = run_qualitative_batch(
            batch,
            blender=resolve_blender(blender),
            workers=workers,
            resume=resume,
        )
    except Exception as exc:  # noqa: BLE001 - status file contains per-clip details
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    summary = status["summary"]
    typer.echo(
        f"Rendered={summary['rendered']} skipped={summary['skipped_complete']} "
        f"failed={summary['failed']}; status: {output / 'render_status.json'}"
    )


@app.command("doctor")
def check(
    blender: Annotated[str | None, typer.Option("--blender", help="Blender executable.")] = None,
) -> None:
    """Verify SMPL-X addon installation and Blender prerequisites."""
    from motionviewer.blender.executable import resolve_blender

    blender_exe = resolve_blender(blender)
    script = Path(__file__).resolve().parent / "blender" / "addon_check_entry.py"
    result = subprocess.run(
        [str(blender_exe), "--background", "--python", str(script)],
        check=False,
    )
    raise typer.Exit(code=result.returncode)


@render_app.command("job")
def render(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Render job YAML.")] = None,
    input: Annotated[list[Path] | None, typer.Option("--input", help="Quick render input.")] = None,
    labels: Annotated[
        str | None, typer.Option("--labels", help="Comma-separated labels for quick render.")
    ] = None,
    blender: Annotated[str | None, typer.Option("--blender", help="Blender executable.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only write bundle and manifest.")] = False,
    parallel_views: Annotated[
        bool, typer.Option("--parallel-views/--no-parallel-views", help="Parallel Blender by staging.")
    ] = True,
) -> None:
    if config is None and not input:
        raise typer.BadParameter("Pass --config or at least one --input")
    if config is not None:
        job = load_render_job(config)
    else:
        label_list = [item.strip() for item in labels.split(",")] if labels else None
        job = quick_job(list(input or []), labels=label_list)

    prepared = prepare_render_job(job)
    output_dir = job.output.directory
    bundle_path = write_bundle(prepared, output_dir / "job_bundle.json")
    manifest_path = write_manifest(prepared, output_dir / job.output.manifest_name)
    typer.echo(f"Wrote bundle: {bundle_path}")
    typer.echo(f"Wrote manifest: {manifest_path}")
    if dry_run:
        return
    from motionviewer.blender.executable import resolve_blender

    try:
        blender_exe = resolve_blender(blender)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    resolved_views = resolve_camera_views(
        job.camera,
        style_ghost={
            "mode": job.style.ghost.mode,
            "include_prefix": job.style.ghost.include_prefix,
            "warmup_frames": job.style.ghost.warmup_frames,
            "start_lightness": job.style.ghost.start_lightness,
            "end_lightness": job.style.ghost.end_lightness,
            "alpha": job.style.ghost.alpha,
        },
        ground={
            "mode": job.ground.mode,
            "opacity": job.ground.opacity,
            "carpet_padding": job.ground.carpet_padding,
        },
    )
    # Explicit views list (even length 1) uses frames/<preset>/; bare preset keeps flat frames/.
    use_view_subdirs = bool(job.camera.views)

    runner = Path(__file__).resolve().parent / "blender" / "runner.py"
    frames_root = output_dir / "frames"
    if frames_root.exists():
        shutil.rmtree(frames_root)
    if use_view_subdirs:
        for view in resolved_views:
            mp4_path = output_dir / f"{view.preset}.mp4"
            if mp4_path.exists() and job.output.overwrite:
                mp4_path.unlink()
    else:
        mp4_path = output_dir / job.output.mp4_name
        if mp4_path.exists() and job.output.overwrite:
            mp4_path.unlink()

    staging_groups = group_views_by_staging(resolved_views)
    if use_view_subdirs and parallel_views and len(staging_groups) > 1:
        typer.echo(f"Running Blender in parallel for stagings: {', '.join(staging_groups)}")
        with ThreadPoolExecutor(max_workers=len(staging_groups)) as pool:
            futures = {
                pool.submit(
                    subprocess.run,
                    [
                        str(blender_exe),
                        "--background",
                        "--python",
                        str(runner),
                        "--",
                        str(bundle_path),
                        "--staging",
                        staging,
                    ],
                    check=False,
                ): staging
                for staging in staging_groups
            }
            for future in as_completed(futures):
                staging = futures[future]
                result = future.result()
                if result.returncode != 0:
                    typer.echo(f"Blender failed for staging={staging}", err=True)
                    raise typer.Exit(code=result.returncode)
    elif use_view_subdirs and len(staging_groups) > 1:
        for staging in staging_groups:
            typer.echo(f"Running Blender for staging={staging}...")
            result = subprocess.run(
                [
                    str(blender_exe),
                    "--background",
                    "--python",
                    str(runner),
                    "--",
                    str(bundle_path),
                    "--staging",
                    staging,
                ],
                check=False,
            )
            if result.returncode != 0:
                raise typer.Exit(code=result.returncode)
    else:
        typer.echo("Running Blender render...")
        cmd = [str(blender_exe), "--background", "--python", str(runner), "--", str(bundle_path)]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise typer.Exit(code=result.returncode)

    from motionviewer.video.encode import encode_mp4

    view_manifest = []
    if use_view_subdirs:
        for view in resolved_views:
            frames_dir = frames_root / view.preset
            mp4_path = output_dir / f"{view.preset}.mp4"
            if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
                typer.echo(f"No frames for view {view.preset}; skipping encode", err=True)
                continue
            _apply_overlays(job, prepared, frames_dir)
            typer.echo(f"Encoding MP4 to {mp4_path}...")
            encode_mp4(
                frames_dir,
                mp4_path,
                fps=prepared.fps,
                pattern="frame_%04d.png",
                overwrite=job.output.overwrite,
            )
            typer.echo(f"Wrote video: {mp4_path}")
            view_manifest.append(
                {
                    "preset": view.preset,
                    "staging": view.staging,
                    "frames_dir": str(frames_dir),
                    "mp4": str(mp4_path),
                }
            )
    elif frames_root.exists() and any(frames_root.glob("*.png")):
        mp4_path = output_dir / job.output.mp4_name
        _apply_overlays(job, prepared, frames_root)
        typer.echo(f"Encoding MP4 to {mp4_path}...")
        encode_mp4(
            frames_root, mp4_path, fps=prepared.fps, pattern="frame_%04d.png", overwrite=job.output.overwrite
        )
        typer.echo(f"Wrote video: {mp4_path}")
    else:
        typer.echo("No rendered frames found; MP4 encoding skipped.")

    if view_manifest:
        manifest_path = output_dir / job.output.manifest_name
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        else:
            manifest = {}
        manifest["views"] = view_manifest
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)


def _apply_overlays(job: RenderJob, prepared, frames_dir: Path) -> None:
    from motionviewer.video.overlay import draw_instruction_banner, draw_legend_on_frames

    if job.style.labels.mode == "legend":
        draw_legend_on_frames(
            frames_dir,
            labels=[item.sequence.source for item in prepared.inputs],
            palette=job.style.palette,
            instruction=job.task.instruction if job.style.labels.show_instruction else None,
        )
    elif job.style.labels.show_instruction and job.task.instruction:
        draw_instruction_banner(frames_dir, instruction=job.task.instruction)
