"""Render a batch of soma clips as source-vs-character comparisons. Local helper.

uv run python scripts/_batch_review.py --clips .local/soma/.../clips/t2m --limit 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER = ROOT / ".local/Blender.app/Contents/MacOS/Blender"
ASSET = ROOT / "assets/fbx/pmx/yoimiya/宵宫.pmx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True, help="Directory of test_rec_* clip dirs")
    parser.add_argument("--source", default="gt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/compare")
    parser.add_argument("--views", default="three_quarter")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--extra-views", default="", help="Clip name substrings that also get a front view")
    args = parser.parse_args()

    clips = sorted(path for path in args.clips.glob("test_rec_*") if path.is_dir())
    if args.limit > 0:
        clips = clips[: args.limit]

    extra = [value.strip() for value in args.extra_views.split(",") if value.strip()]
    failures: list[str] = []
    for index, clip in enumerate(clips, start=1):
        motion = clip / args.source / "smplx_params.npz"
        meta_path = clip / "meta.json"
        if not motion.is_file():
            print(f"[{index}/{len(clips)}] skip {clip.name}: no {args.source} params")
            continue
        caption = ""
        label = clip.name.replace("test_rec_", "")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            caption = str(meta.get("caption") or "")
        views = args.views
        if any(token in clip.name for token in extra):
            views = f"{views},front"
        destination = args.output / label
        print(f"[{index}/{len(clips)}] {label}  views={views}")
        command = [
            str(BLENDER),
            "--background",
            "--python",
            str(ROOT / "scripts/render_mmd_compare.py"),
            "--",
            "--asset",
            str(ASSET),
            "--motion",
            str(motion),
            "--output",
            str(destination),
            "--views",
            views,
            "--resolution",
            str(args.resolution),
            "--caption",
            caption,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join(result.stdout.strip().splitlines()[-6:])
            print(f"   FAILED\n{tail}\n{result.stderr.strip()[-400:]}")
            failures.append(label)
        else:
            rendered = [line for line in result.stdout.splitlines() if line.startswith("rendered")]
            for line in rendered:
                print(f"   {line}")

    if failures:
        print("failed clips:", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
