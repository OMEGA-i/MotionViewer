from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def encode_mp4(
    frames_dir: str | Path,
    output_path: str | Path,
    *,
    fps: float,
    pattern: str = "%06d.png",
    crf: int = 18,
    codec: str = "libx264",
    overwrite: bool = False,
) -> Path:
    """Encode an image sequence to MP4 using ffmpeg.

    Rendering frames first keeps Blender failures separate from video encoding
    failures and lets users rerun this step without rerendering.
    """

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH; install ffmpeg or keep frames only")
    frames = Path(frames_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; set overwrite=True")
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        str(frames / pattern),
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return output
