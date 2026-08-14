"""Render publication figures for the top-ranked clips. Local helper.

Runs `render_paper_figure.py` then `compose_paper_figure.py` over a shortlist, so
a paper's figure set is one command rather than one command per clip. Resumable,
like the showcase generator: a clip whose filmstrip exists is skipped.

  uv run python scripts/_batch_paper_figures.py --clips <dir> \
      --scores outputs/motion_scores.json --limit 20 --character yoimiya
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER = ROOT / ".local/Blender.app/Contents/MacOS/Blender"

CHARACTERS: dict[str, str] = {
    "yoimiya": "assets/fbx/pmx/yoimiya/宵宫.pmx",
    "furina": "assets/fbx/pmx/furina/【芙宁娜】.pmx",
    "furina_wild": "assets/fbx/pmx/furina/【芙宁娜_荒】.pmx",
    "silverwolf": "assets/fbx/pmx/silverwolf/星穹铁道——银狼/银狼.pmx",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--source", default="gt")
    parser.add_argument("--character", default="yoimiya")
    parser.add_argument("--output", type=Path, default=ROOT / "converted/figures")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--count", type=int, default=6, help="Frames per figure")
    parser.add_argument("--resolution", type=int, default=1200)
    parser.add_argument("--strip-height", type=int, default=900)
    parser.add_argument("--max-penalty", type=float, default=0.25)
    args = parser.parse_args()

    if args.character not in CHARACTERS:
        raise SystemExit(f"unknown character {args.character}; known: {sorted(CHARACTERS)}")

    ranked = json.loads(args.scores.read_text(encoding="utf-8"))["clips"]
    shortlist = [item for item in ranked if item["penalty"] <= args.max_penalty][: args.limit]
    print(f"{len(shortlist)} figures for {args.character}")

    index_rows: list[dict] = []
    for position, item in enumerate(shortlist, start=1):
        rec_id = item["clip"].replace("test_rec_", "")
        motion = args.clips / item["clip"] / args.source / "smplx_params.npz"
        out = args.output / args.character / rec_id
        strip = out / "filmstrip.png"
        if strip.is_file():
            print(f"[{position}/{len(shortlist)}] {rec_id} (already done)")
        else:
            if not motion.is_file():
                print(f"[{position}/{len(shortlist)}] {rec_id}: no motion, skipped")
                continue
            print(f"[{position}/{len(shortlist)}] {rec_id}")
            render = subprocess.run(
                [
                    str(BLENDER),
                    "--background",
                    "--python",
                    str(ROOT / "scripts/render_paper_figure.py"),
                    "--",
                    "--asset",
                    str(ROOT / CHARACTERS[args.character]),
                    "--motion",
                    str(motion),
                    "--output",
                    str(out),
                    "--count",
                    str(args.count),
                    "--resolution",
                    str(args.resolution),
                ],
                capture_output=True,
                text=True,
            )
            if render.returncode != 0:
                tail = "\n".join((render.stdout + render.stderr).strip().splitlines()[-5:])
                print(f"      render failed\n{tail}")
                continue
            compose = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/compose_paper_figure.py"),
                    "--input",
                    str(out),
                    "--height",
                    str(args.strip_height),
                ],
                capture_output=True,
                text=True,
            )
            if compose.returncode != 0:
                print(f"      compose failed: {compose.stderr.strip()[-200:]}")
                continue

        meta = {}
        meta_path = out / "figure.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index_rows.append(
            {
                "rec_id": rec_id,
                "caption": item.get("caption", ""),
                "activity": item.get("activity", 0.0),
                "filmstrip": str(strip.relative_to(args.output)) if strip.is_file() else None,
                "trail": "trail.png" if (out / "trail.png").is_file() else None,
                "frames": meta.get("frames", []),
                "root_travel_m": meta.get("root_travel_m"),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures.json").write_text(
        json.dumps({"character": args.character, "figures": index_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trails = sum(1 for row in index_rows if row["trail"])
    print(f"\n{len(index_rows)} figures in {args.output / args.character}")
    print(f"  {trails} also have a trail overlay; the rest are in-place clips where it would not read")


if __name__ == "__main__":
    main()
