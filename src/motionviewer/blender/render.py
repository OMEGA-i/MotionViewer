from __future__ import annotations

from pathlib import Path


def frames_output_dir(output_dir: str | Path, frames_subdir: str | None = None) -> Path:
    root = Path(output_dir) / "frames"
    if frames_subdir:
        return root / frames_subdir
    return root


def configure_render(
    *,
    output_dir: str | Path,
    frames: int,
    fps: float,
    resolution: tuple[int, int],
    engine: str,
    samples: int,
    frame_format: str,
    frames_subdir: str | None = None,
) -> Path:
    import bpy  # type: ignore

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(1, frames)
    scene.render.fps = int(round(fps))
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.engine = _normalize_engine(engine)
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(samples)
    if hasattr(scene, "cycles"):
        scene.cycles.samples = int(samples)
    frames_dir = frames_output_dir(output_dir, frames_subdir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frames_dir / "frame_")
    scene.render.image_settings.file_format = frame_format
    return frames_dir


def _normalize_engine(engine: str) -> str:
    import bpy  # type: ignore

    if engine == "BLENDER_EEVEE_NEXT":
        engine = "BLENDER_EEVEE"
    valid = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    if engine in valid:
        return engine
    if "BLENDER_EEVEE" in valid:
        return "BLENDER_EEVEE"
    if "CYCLES" in valid:
        return "CYCLES"
    return bpy.context.scene.render.engine


def render_animation() -> None:
    import bpy  # type: ignore

    bpy.ops.render.render(animation=True)
