"""Render the probe set both ways and stack them side by side. Local helper.

The question this answers is whether reading SOMA-77 directly is at least as good as
the SMPL-X body-22 fit the pipeline has been using. Both sides get identical shading,
camera preset, expression and encoder settings, so the only variables are the source
format and, on the SOMA side, the 30 driven finger bones.

  uv run python scripts/_probe_soma_vs_smplx.py --output converted/soma_probe

Framing is within about 1% between the two: the follow box is a joint hull, and SOMA's
hull includes fingertips that reach a little past the wrist. Headings match exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER = ROOT / ".local/Blender.app/Contents/MacOS/Blender"
PROBE = Path("/Users/a26044/Motion data/soma_retarget_probe")


def render(source_flag: str, source: Path, out: Path, args) -> str:
    if out.exists():
        shutil.rmtree(out)
    command = [
        str(BLENDER), "--background", "--python", str(ROOT / "scripts/render_mmd_compare.py"), "--",
        "--asset", str(ROOT / args.asset),
        source_flag, str(source),
        "--output", str(out),
        "--panels", "character",
        "--views", args.view,
        "--resolution", str(args.resolution),
        "--aspect", str(args.aspect),
        "--toon", "--camera", "follow",
        "--expression", args.expression,
    ]  # fmt: skip
    if args.spring:
        command.append("--spring")
    if args.frames > 0:
        command += ["--frames", str(args.frames)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-6:])
        return f"blender exited {result.returncode}\n{tail}"
    if not (out / "character" / args.view).is_dir():
        return "no frames produced"
    for line in result.stdout.splitlines():
        if line.startswith("soma:"):
            print(f"      {line}", flush=True)
    return ""


def stack(left: Path, right: Path, destination: Path, fps: float) -> str:
    """Left panel labelled SOMA-77, right SMPL-X 22, padded to the longer of the two."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "ffmpeg not on PATH"
    counts = [len(list(panel.glob("frame_*.png"))) for panel in (left, right)]
    if min(counts) == 0:
        return f"missing frames {counts}"
    longest = max(counts)
    command = [ffmpeg, "-y", "-loglevel", "error"]
    for panel in (left, right):
        command += ["-framerate", f"{fps:g}", "-start_number", "1", "-i", str(panel / "frame_%04d.png")]
    steps = []
    for index, count in enumerate(counts):
        current = f"{index}:v"
        if count < longest:
            steps.append(
                f"[{current}]tpad=stop_mode=clone:stop_duration={(longest - count) / fps:.4f}[p{index}]"
            )
            current = f"p{index}"
        steps.append(f"[{current}]null[s{index}]")
    steps.append("[s0][s1]hstack=inputs=2[h]")
    steps.append("[h]scale=trunc(iw/2)*2:trunc(ih/2)*2[out]")
    command += [
        "-filter_complex", ";".join(steps), "-map", "[out]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
        str(destination),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True)
    return "" if result.returncode == 0 else f"ffmpeg: {result.stderr.strip()[-300:]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=PROBE)
    parser.add_argument("--output", type=Path, default=ROOT / "converted/soma_probe")
    parser.add_argument("--asset", default="assets/fbx/pmx/yoimiya/宵宫.pmx")
    parser.add_argument("--view", default="character_front")
    parser.add_argument("--resolution", type=int, default=1200)
    parser.add_argument("--aspect", type=float, default=0.62)
    parser.add_argument("--expression", default="smile")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--spring", action="store_true", default=True)
    parser.add_argument("--no-spring", dest="spring", action="store_false")
    args = parser.parse_args()

    meta = json.loads((args.probe / "meta.json").read_text(encoding="utf-8"))
    by_id = {clip["rec_id"]: clip for clip in meta["clips"]}
    pairs = sorted((args.probe / "soma").glob("*.soma.npz"))
    args.output.mkdir(parents=True, exist_ok=True)
    frames_root = args.output / ".frames"

    failures: list[str] = []
    for position, soma in enumerate(pairs, start=1):
        rec = soma.name[: -len(".soma.npz")]
        smplx = args.probe / "smplx" / f"{rec}.smplx.npz"
        info = by_id.get(rec, {})
        destination = args.output / f"{rec}.mp4"
        print(f"[{position}/{len(pairs)}] {rec}  {info.get('seconds', 0):.1f}s", flush=True)
        if destination.is_file():
            print("      already done", flush=True)
            continue
        if not smplx.is_file():
            failures.append(f"{rec}: no SMPL-X counterpart")
            continue
        error = render("--soma", soma, frames_root / f"{rec}_soma", args)
        if not error:
            error = render("--motion", smplx, frames_root / f"{rec}_smplx", args)
        if not error:
            error = stack(
                frames_root / f"{rec}_soma" / "character" / args.view,
                frames_root / f"{rec}_smplx" / "character" / args.view,
                destination,
                float(info.get("src_fps", 50.0)),
            )
        shutil.rmtree(frames_root, ignore_errors=True)
        if error:
            print(f"      FAILED: {error.splitlines()[0]}", flush=True)
            failures.append(f"{rec}: {error.splitlines()[0]}")
            continue
        (args.output / f"{rec}.txt").write_text(
            f"{info.get('caption', '')}\n"
            f"SMPL-X fit RMSE: {info.get('smplx_fit_rmse_mm', float('nan')):.1f} mm\n"
            f"left panel: SOMA-77 direct (fingers driven)\n"
            f"right panel: SMPL-X body-22 fitted (hands static)\n",
            encoding="utf-8",
        )
        print(f"      -> {destination.name}", flush=True)

    shutil.rmtree(frames_root, ignore_errors=True)
    print(f"\n{len(list(args.output.glob('*.mp4')))} comparison video(s) in {args.output}", flush=True)
    if failures:
        print(f"{len(failures)} failure(s):", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
