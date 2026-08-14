"""Tile one frame from every generated video into sheets, for a fast eyeball pass.

Three hundred clips is too many to watch and too many to trust unwatched. A
failure in this pipeline is almost always visible in a single frame — hair flung
across the shot, a limb through the torso, a grey untextured mesh — so one frame
per clip on a contact sheet finds it in seconds.

The frame sampled is 60% of the way in, not the first: clip openings are often a
neutral standing pose that would look fine no matter what broke.

  uv run python scripts/qc_contact_sheet.py --input converted/showcase --output outputs/qc
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def _grab(video: Path, destination: Path, position: float) -> bool:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    seconds = 1.0
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        seconds = float(stream.get("duration") or 1.0) * position
    except Exception:
        pass
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(video),
            "-vframes",
            "1",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and destination.is_file()


def _pop_ratio(video: Path) -> tuple[float, float, int] | None:
    """Largest frame-to-frame change over the clip's median, and where.

    Cheap proxy for a visual glitch: real motion moves a lot every frame, so a
    single frame that changes several times more than the clip's own median is
    almost always a pop rather than choreography.
    """
    import tempfile

    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vf",
                "scale=160:-1",
                "-vsync",
                "0",
                f"{directory}/f_%04d.png",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        frames = sorted(Path(directory).glob("f_*.png"))
        if len(frames) < 3:
            return None
        grey = [np.asarray(Image.open(path).convert("L"), dtype=np.float32) for path in frames]
    diffs = np.array([np.abs(grey[index] - grey[index - 1]).mean() for index in range(1, len(grey))])
    median = float(np.median(diffs))
    largest = float(diffs.max())
    return largest / max(median, 1e-6), largest, int(np.argmax(diffs)) + 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Showcase directory with videos/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cell", type=int, default=210)
    parser.add_argument("--position", type=float, default=0.6, help="Fraction into the clip")
    parser.add_argument(
        "--pop-check",
        type=int,
        default=0,
        help="Also scan N videos for temporal popping (0 skips it)",
    )
    args = parser.parse_args()

    videos = sorted((args.input / "videos").glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no videos in {args.input / 'videos'}")

    manifest_path = args.input / "manifest.json"
    order: dict[str, int] = {}
    if manifest_path.is_file():
        entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("clips", [])
        order = {Path(entry["video"]).stem: index for index, entry in enumerate(entries)}
    videos.sort(key=lambda path: order.get(path.stem, 10**6))

    args.output.mkdir(parents=True, exist_ok=True)
    grabs = args.output / ".grabs"
    grabs.mkdir(exist_ok=True)

    per_sheet = args.columns * args.rows
    label = 15
    sheets = 0
    failed: list[str] = []
    for start in range(0, len(videos), per_sheet):
        batch = videos[start : start + per_sheet]
        sheet = Image.new("RGB", (args.columns * args.cell, args.rows * (args.cell + label)), (245, 243, 240))
        draw = ImageDraw.Draw(sheet)
        for index, video in enumerate(batch):
            row, column = divmod(index, args.columns)
            frame_path = grabs / f"{video.stem}.png"
            if not frame_path.is_file() and not _grab(video, frame_path, args.position):
                failed.append(video.stem)
                continue
            image = Image.open(frame_path).convert("RGB")
            scale = min(args.cell / image.width, args.cell / image.height)
            image = image.resize(
                (max(int(image.width * scale), 1), max(int(image.height * scale), 1)), Image.LANCZOS
            )
            x = column * args.cell + (args.cell - image.width) // 2
            y = row * (args.cell + label) + label
            sheet.paste(image, (x, y))
            draw.text(
                (column * args.cell + 3, row * (args.cell + label) + 2),
                f"{start + index + 1}. {video.stem[:26]}",
                fill=(40, 40, 40),
            )
        destination = args.output / f"qc_{sheets:02d}.png"
        sheet.save(destination)
        print(f"wrote {destination} ({len(batch)} clips)")
        sheets += 1

    print(f"{len(videos)} clips over {sheets} sheet(s)")
    if failed:
        print(f"could not grab a frame from {len(failed)}: {failed[:10]}")

    if args.pop_check > 0:
        print(f"\nscanning {min(args.pop_check, len(videos))} videos for popping")
        suspects = []
        for video in videos[: args.pop_check]:
            result = _pop_ratio(video)
            if result is None:
                continue
            ratio, largest, frame = result
            # A spring-bone glitch or a keyframe discontinuity shows up as one
            # frame differing far more than the clip's own median motion.
            if ratio > 6.0 and largest > 4.0:
                suspects.append((video.stem, ratio, frame))
        if suspects:
            print(f"  {len(suspects)} possible pops:")
            for name, ratio, frame in suspects[:12]:
                print(f"    {name}  ratio {ratio:.1f} at frame {frame}")
        else:
            print("  none: every clip's largest frame-to-frame change is in line with its own motion")


if __name__ == "__main__":
    main()
