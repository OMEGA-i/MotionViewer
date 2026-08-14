"""Turn comparison renders into MP4s and one HTML page. Local helper.

The reviewer here does not use Blender, so every result has to arrive as a video
in a page that opens in a browser.  Each video stacks the SMPL-X source skeleton
next to the retargeted character under one camera, so "the pose looks wrong" can
be attributed to the motion or to the retarget without opening a 3D app.

  uv run python scripts/build_review_page.py --input outputs/compare --output outputs/review
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

_FPS = 30


def _encode(panels: list[Path], destination: Path, fps: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    command = [ffmpeg, "-y", "-loglevel", "error"]
    for panel in panels:
        command += ["-framerate", str(fps), "-start_number", "1", "-i", str(panel / "frame_%04d.png")]
    if len(panels) > 1:
        inputs = "".join(f"[{index}:v]" for index in range(len(panels)))
        command += ["-filter_complex", f"{inputs}hstack=inputs={len(panels)}"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg failed for {destination.name}: {result.stderr.strip()[:300]}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/compare"))
    parser.add_argument("--output", type=Path, default=Path("outputs/review"))
    parser.add_argument("--fps", type=int, default=_FPS)
    parser.add_argument("--title", default="SMPL-X → 宵宫 retarget review")
    parser.add_argument(
        "--still",
        action="append",
        default=[],
        metavar="PATH|CAPTION",
        help="Also embed a still image with a caption. Repeatable.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for info_path in sorted(args.input.glob("*/info.json")):
        clip_dir = info_path.parent
        info = json.loads(info_path.read_text(encoding="utf-8"))
        panels = [name for name in info.get("panels", ["character"])]
        clip_videos: list[dict] = []
        for view in info.get("views", []):
            available = [clip_dir / panel / view for panel in panels]
            available = [path for path in available if path.is_dir() and any(path.glob("frame_*.png"))]
            if not available:
                continue
            destination = videos_dir / f"{clip_dir.name}_{view}.mp4"
            if _encode(available, destination, args.fps):
                clip_videos.append(
                    {
                        "view": view,
                        "path": f"videos/{destination.name}",
                        "panels": [path.parent.name for path in available],
                    }
                )
                print(f"  encoded {destination.name}")
        if clip_videos:
            entries.append({"name": clip_dir.name, "info": info, "videos": clip_videos})

    still_blocks: list[str] = []
    for item in args.still:
        path_text, _, caption = item.partition("|")
        source = Path(path_text.strip())
        if not source.is_file():
            print(f"  missing still {source}")
            continue
        destination = args.output / "stills" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        still_blocks.append(
            "<figure>"
            f'<img src="stills/{html.escape(source.name)}" alt="{html.escape(caption.strip())}">'
            f"<figcaption>{html.escape(caption.strip())}</figcaption></figure>"
        )
        print(f"  copied still {source.name}")

    rows: list[str] = []
    for entry in entries:
        info = entry["info"]
        caption = html.escape(str(info.get("caption") or ""))
        polish = info.get("transfer", {})
        meta_bits = [
            f"{info.get('frames', '?')} frames",
            "faithful (no polish)" if info.get("faithful") else "polished",
        ]
        if polish.get("twist_smoothing_window"):
            meta_bits.append(f"twist smoothing {polish['twist_smoothing_window']}f")
        if polish.get("hand_relax"):
            meta_bits.append(f"hand relax {polish['hand_relax']}")
        players = "\n".join(
            "<figure>"
            f"<video src=\"{html.escape(video['path'])}\" controls loop muted autoplay playsinline></video>"
            f"<figcaption>{html.escape(video['view'])} &middot; "
            f"{html.escape(' | '.join(video['panels']))}</figcaption>"
            "</figure>"
            for video in entry["videos"]
        )
        rows.append(
            f"<section><h2>{html.escape(entry['name'])}</h2>"
            f"<p class=\"meta\">{html.escape(' &middot; '.join(meta_bits))}</p>"
            + (f'<p class="caption">{caption}</p>' if caption else "")
            + f'<div class="videos">{players}</div></section>'
        )

    page = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(args.title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; padding: 32px clamp(16px, 4vw, 64px) 72px;
         background: #14161a; color: #e8e6e3;
         font: 15px/1.6 -apple-system, "Helvetica Neue", "PingFang SC", sans-serif; }}
  h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 6px; }}
  .lede {{ color: #9aa0a6; max-width: 62ch; margin: 0 0 36px; }}
  section {{ border-top: 1px solid #262a30; padding: 24px 0 8px; }}
  h2 {{ font-size: 16px; font-weight: 600; margin: 0 0 4px; font-variant-numeric: tabular-nums; }}
  .meta {{ color: #7f868d; font-size: 13px; margin: 0 0 8px; }}
  .caption {{ color: #b9bec4; max-width: 78ch; margin: 0 0 14px; }}
  .videos {{ display: flex; flex-wrap: wrap; gap: 18px; }}
  figure {{ margin: 0; }}
  video {{ width: min(760px, 92vw); border-radius: 8px; background: #0d0f12; display: block; }}
  figcaption {{ color: #7f868d; font-size: 12px; margin-top: 6px; max-width: 80ch; }}
  .stills figure {{ margin: 0 0 26px; }}
  .stills img {{ width: min(1100px, 94vw); border-radius: 8px; background: #0d0f12; display: block; }}
</style>
</head>
<body>
<h1>{html.escape(args.title)}</h1>
<p class="lede">Each video is one camera. Where two panels are shown, the left is
the SMPL-X source motion drawn as a skeleton and the right is the retargeted
character — same frame, same camera, same light. If a pose looks wrong on the
right and also looks wrong on the left, it came from the motion.</p>
{"".join(rows)}
{'<section class="stills"><h2>Decisions, as pictures</h2>' + "".join(still_blocks) + "</section>" if still_blocks else ""}
</body>
</html>
"""
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {args.output / 'index.html'} with {len(entries)} clips")


if __name__ == "__main__":
    main()
