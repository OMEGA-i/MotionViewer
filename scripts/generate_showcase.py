"""Render a whole ranked clip set to MP4s plus a browsable page. Local helper.

Built to be left running unattended for hours, which drives three decisions:

- **resumable**: a clip whose MP4 already exists is skipped, so the run can be
  killed and restarted without losing work.
- **frames are deleted as soon as they are encoded**: 300 clips at 120 frames of
  720p PNG is about 18 GB, and there is no reason to keep any of it.
- **one clip failing does not stop the run**: failures are recorded and reported
  at the end.

  uv run python scripts/generate_showcase.py \
      --clips .local/soma_all/.../clips/t2m --scores outputs/motion_scores.json \
      --character yoimiya --limit 0 --resolution 720
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER = ROOT / ".local/Blender.app/Contents/MacOS/Blender"

CHARACTERS: dict[str, str] = {
    "yoimiya": "assets/fbx/pmx/yoimiya/宵宫.pmx",
    "furina": "assets/fbx/pmx/furina/【芙宁娜】.pmx",
    "furina_wild": "assets/fbx/pmx/furina/【芙宁娜_荒】.pmx",
    "silverwolf": "assets/fbx/pmx/silverwolf/星穹铁道——银狼/银狼.pmx",
}


def _encode(frames_dir: Path, destination: Path, fps: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "1",
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        # Web players refuse odd dimensions.
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"      ffmpeg failed: {result.stderr.strip()[:200]}")
        return False
    return True


def _write_index(output: Path, entries: list[dict], title: str) -> None:
    cards: list[str] = []
    for entry in entries:
        score = entry.get("score", {})
        bits = [
            f"{entry['frames']} frames",
            f"activity {score.get('activity', 0):.2f}",
            f"penalty {score.get('penalty', 0):.2f}",
            f"{score.get('max_angular_velocity_deg_s', 0):.0f} deg/s peak",
        ]
        cards.append(
            f'<article data-text="{html.escape((entry["rec_id"] + " " + entry.get("caption", "") + " " + entry["character"]).lower())}">'
            f'<video src="{html.escape(entry["video"])}" controls loop muted playsinline preload="none"></video>'
            f"<h2>{html.escape(entry['rec_id'])} <span>{html.escape(entry['character'])}</span></h2>"
            f'<p class="meta">{html.escape(" · ".join(bits))}</p>'
            f'<p class="caption">{html.escape(entry.get("caption", ""))}</p>'
            "</article>"
        )

    page = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:28px clamp(14px,3vw,48px) 64px; background:#14161a; color:#e8e6e3;
         font:15px/1.55 -apple-system,"Helvetica Neue","PingFang SC",sans-serif; }}
  h1 {{ font-size:21px; margin:0 0 4px; }}
  .lede {{ color:#9aa0a6; margin:0 0 28px; max-width:70ch; }}
  .grid {{ display:grid; gap:22px; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); }}
  article {{ background:#1a1d22; border-radius:10px; overflow:hidden; }}
  video {{ width:100%; display:block; background:#0d0f12; }}
  h2 {{ font-size:13px; font-weight:600; margin:10px 12px 2px; font-variant-numeric:tabular-nums; }}
  h2 span {{ color:#7f868d; font-weight:400; }}
  .meta {{ color:#7f868d; font-size:12px; margin:0 12px 6px; }}
  .caption {{ color:#b9bec4; font-size:13px; margin:0 12px 12px; }}
  .tools {{ position:sticky; top:0; z-index:2; background:#14161a; padding:8px 0 14px;
            display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  input[type=search] {{ background:#1a1d22; border:1px solid #2b3038; color:#e8e6e3;
            border-radius:7px; padding:8px 11px; font:14px inherit; min-width:min(340px,70vw); }}
  #count {{ color:#7f868d; font-size:13px; font-variant-numeric:tabular-nums; }}
  article[hidden] {{ display:none; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="lede">{len(entries)} clips, ranked cleanest-then-most-expressive by
<code>score_motion_quality.py</code>. Cel shaded with the model's own ramp and
sphere maps, outlined, floor shadow, and baked spring bones on every bone the PMX
marks as dynamic. Videos load on demand — click one to play.</p>
<div class="tools">
  <input type="search" id="q" placeholder="filter by caption, id or character — e.g. dance, kick, sit">
  <span id="count"></span>
</div>
<div class="grid" id="grid">{"".join(cards)}</div>
<script>
// Client-side filter: the whole set is one page, and finding the clip worth
// showing in a paper is the actual task here.
const cards = [...document.querySelectorAll('#grid article')];
const box = document.getElementById('q');
const count = document.getElementById('count');
function apply() {{
  const terms = box.value.toLowerCase().split(/\s+/).filter(Boolean);
  let shown = 0;
  for (const card of cards) {{
    const text = card.dataset.text;
    const hit = terms.every(term => text.includes(term));
    card.hidden = !hit;
    if (hit) shown++;
  }}
  count.textContent = shown + ' / ' + cards.length + ' clips';
}}
box.addEventListener('input', apply);
apply();
</script>
</body></html>
"""
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--source", default="gt")
    parser.add_argument("--character", action="append", default=None, help="Repeatable; default yoimiya")
    parser.add_argument("--output", type=Path, default=ROOT / "converted/showcase")
    parser.add_argument("--view", default="three_quarter")
    parser.add_argument("--resolution", type=int, default=800, help="Frame height in pixels")
    parser.add_argument("--aspect", type=float, default=0.72, help="Width / height")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="0 renders the whole shortlist")
    parser.add_argument("--max-penalty", type=float, default=0.25)
    parser.add_argument("--no-spring", action="store_true")
    parser.add_argument("--title", default="SMPL-X → MMD characters · full showcase")
    args = parser.parse_args()

    characters = args.character or ["yoimiya"]
    unknown = [name for name in characters if name not in CHARACTERS]
    if unknown:
        raise SystemExit(f"unknown character(s): {unknown}; known: {sorted(CHARACTERS)}")

    ranked = json.loads(args.scores.read_text(encoding="utf-8"))["clips"]
    shortlist = [item for item in ranked if item["penalty"] <= args.max_penalty]
    if args.limit > 0:
        shortlist = shortlist[: args.limit]
    print(
        f"{len(shortlist)} clips x {len(characters)} character(s) = {len(shortlist) * len(characters)} videos"
    )

    videos_dir = args.output / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_root = args.output / ".frames"
    manifest_path = args.output / "manifest.json"
    entries: list[dict] = []
    if manifest_path.is_file():
        entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("clips", [])
    done = {(entry["rec_id"], entry["character"]) for entry in entries}

    failures: list[str] = []
    started = time.time()
    total = len(shortlist) * len(characters)
    index = 0
    for item in shortlist:
        clip_dir = args.clips / item["clip"]
        motion = clip_dir / args.source / "smplx_params.npz"
        rec_id = item["clip"].replace("test_rec_", "")
        for character in characters:
            index += 1
            key = (rec_id, character)
            name = f"{rec_id}_{character}"
            destination = videos_dir / f"{name}.mp4"
            if key in done and destination.is_file():
                continue
            if not motion.is_file():
                failures.append(f"{name}: missing motion")
                continue

            elapsed = time.time() - started
            rate = elapsed / max(index - len(done), 1)
            print(
                f"[{index}/{total}] {name}  act={item['activity']:.2f}"
                f"  eta {(total - index) * rate / 3600:.1f}h"
            )
            frames_dir = frames_root / name
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            command = [
                str(BLENDER),
                "--background",
                "--python",
                str(ROOT / "scripts/render_mmd_compare.py"),
                "--",
                "--asset",
                str(ROOT / CHARACTERS[character]),
                "--motion",
                str(motion),
                "--output",
                str(frames_dir),
                "--views",
                args.view,
                "--panels",
                "character",
                "--resolution",
                str(args.resolution),
                "--aspect",
                str(args.aspect),
                "--toon",
                "--caption",
                item.get("caption", ""),
            ]
            if not args.no_spring:
                command.append("--spring")
            result = subprocess.run(command, capture_output=True, text=True)
            rendered = frames_dir / "character" / args.view
            if result.returncode != 0 or not rendered.is_dir():
                tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-5:])
                print(f"      render failed\n{tail}")
                failures.append(f"{name}: render")
                shutil.rmtree(frames_dir, ignore_errors=True)
                continue

            if _encode(rendered, destination, args.fps):
                entries.append(
                    {
                        "rec_id": rec_id,
                        "character": character,
                        "video": f"videos/{destination.name}",
                        "frames": item["frames"],
                        "caption": item.get("caption", ""),
                        "score": {
                            key: item[key]
                            for key in (
                                "activity",
                                "penalty",
                                "max_angular_velocity_deg_s",
                                "jitter_deg",
                                "foot_skate_m_s",
                            )
                            if key in item
                        },
                    }
                )
                done.add(key)
                manifest_path.write_text(
                    json.dumps({"clips": entries}, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _write_index(args.output, entries, args.title)
            else:
                failures.append(f"{name}: encode")
            # 18 GB of PNGs is not worth keeping once the MP4 exists.
            shutil.rmtree(frames_dir, ignore_errors=True)

    shutil.rmtree(frames_root, ignore_errors=True)
    _write_index(args.output, entries, args.title)
    print(f"\n{len(entries)} videos in {args.output}")
    if failures:
        print(f"{len(failures)} failures:")
        for line in failures[:20]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
