"""Visualise a whole SMPL-X dataset on an MMD character. One command, no arguments needed.

Point it at a directory tree of ``.npz`` clips and it discovers the tasks inside,
validates every file against the retarget's input contract, scores the motion,
renders the selection to MP4, and writes a browsable page.

    # what is in there?
    uv run python scripts/visualize_dataset.py --list

    # render one task
    uv run python scripts/visualize_dataset.py --task picks38/gt

    # generated beside ground truth, same camera on both sides
    uv run python scripts/visualize_dataset.py --task picks38/gen --compare picks38/gt

    # everything, every character, unattended
    uv run python scripts/visualize_dataset.py --task '*' --character yoimiya --character furina

Design notes, because this is meant to be left running:

- **Validate before rendering.** A clip that fails the contract is reported with
  the reason and skipped, instead of surfacing 40 minutes later as a Blender
  traceback.
- **Resumable.** A clip whose MP4 exists is skipped, so the run survives Ctrl-C.
- **Frames are deleted once encoded.** 500 clips of 720p PNG is tens of GB and
  none of it is worth keeping.
- **One failure does not stop the run.** Failures are collected and printed at
  the end.
- **Nothing is rewritten to make the motion look better** unless asked. Ground penetration,
  run-away root travel and jitter are measured and shown on the page, not fixed
  silently; these videos are evidence about a model, so a flaw in the motion has
  to stay visible.
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLENDER = ROOT / ".local/Blender.app/Contents/MacOS/Blender"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from score_motion_quality import score_clip  # noqa: E402

CHARACTERS: dict[str, str] = {
    "yoimiya": "assets/fbx/pmx/yoimiya/宵宫.pmx",
    "furina": "assets/fbx/pmx/furina/【芙宁娜】.pmx",
    "furina_wild": "assets/fbx/pmx/furina/【芙宁娜_荒】.pmx",
    "silverwolf": "assets/fbx/pmx/silverwolf/星穹铁道——银狼/银狼.pmx",
}

DEFAULT_DATA = Path("/Users/a26044/Motion data")

# The retarget's input contract. joints22 is not optional: the rest skeleton is
# recovered from it, and the MMD path refuses a clip without it.
REQUIRED = ("global_orient", "body_pose", "transl", "joints22")

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


# ---------------------------------------------------------------------------
# discovery and validation
# ---------------------------------------------------------------------------


@dataclass
class Clip:
    task: str
    clip_id: str
    path: Path
    frames: int
    fps: float
    caption: str
    score: dict = field(default_factory=dict)
    error: str = ""
    trim: int = 0
    trim_reason: str = ""
    # Set when leading frames were dropped: a rewritten npz for Blender to read.
    render_path: Path | None = None

    @property
    def motion_path(self) -> Path:
        return self.render_path or self.path


# Below this, a joint is not moving: measured frozen lead-ins sit at exactly
# 0.0 mm/frame while the quietest real motion is around 2 mm/frame.
_STATIC_STEP_M = 0.0005


def detect_trim(joints: np.ndarray) -> tuple[int, str]:
    """Leading frames to drop, and the reason to show on the page.

    Two artefacts appear in these exports, both at the head of the clip and both
    measured rather than assumed:

    - A **frozen lead-in**: every ``old500`` clip holds one pose for exactly 14
      frames before moving. Half a second of a statue at the head of every video.
    - An **anchor pop**: ``gen`` clips start on the conditioning pose and jump to
      the first generated frame, a 358 mm single-frame step in ``new500`` against
      a 9 mm median afterwards. This one matters beyond looks — the retarget
      re-bases the whole root path and the ground contact on frame 0, so leaving
      it in mis-places the entire clip.

    Only the head is trimmed; no measured clip has a frozen tail.
    """
    step = np.linalg.norm(np.diff(joints, axis=0), axis=-1).max(axis=1)
    if len(step) < 6:
        return 0, ""
    moving = step > _STATIC_STEP_M
    frozen = int(np.argmax(moving)) if moving.any() else 0
    if frozen > 2:
        return frozen, f"dropped {frozen}-frame frozen lead-in"
    later = float(np.median(step[3:]))
    if later > 0.0 and step[0] > 3.0 * later:
        return 1, f"dropped anchor pop at frame 0 ({step[0] * 1000:.0f} mm, {step[0] / later:.0f}x)"
    return 0, ""


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred box filter along axis 0, with the ends held rather than tapered."""
    pad = window // 2
    padded = np.concatenate([np.repeat(values[:1], pad, axis=0), values, np.repeat(values[-1:], pad, axis=0)])
    kernel = np.ones(window) / window
    flat = padded.reshape(len(padded), -1)
    out = np.empty((len(values), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        out[:, column] = np.convolve(flat[:, column], kernel, mode="valid")
    return out.reshape(len(values), *values.shape[1:])


def _project_to_rotations(matrices: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix to each input, via SVD. Averaging leaves SO(3)."""
    u, _, vt = np.linalg.svd(matrices)
    rotations = u @ vt
    flipped = np.linalg.det(rotations) < 0.0
    if flipped.any():
        u = u.copy()
        u[flipped, :, -1] *= -1.0
        rotations[flipped] = u[flipped] @ vt[flipped]
    return rotations


# Joints the filter touches. The shake that shows on a character is in the torso
# chain: it carries the head, and through the collars it carries both arms, so
# smoothing here also settles the limbs without blunting their own motion. A wrist
# during a throw moves fast on purpose, and filtering it clips the throw — measured
# over ``picks38/gt``, extending the filter to every joint bought no extra jitter
# reduction (63% either way) while tripling how far the pose moved (p99 3.54°
# against 1.21°, worst 9.67° against 4.74°).
_SMOOTH_JOINTS = (
    "pelvis",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
    "left_collar",
    "right_collar",
)


def smooth_pose(payload: dict, window: int, joints: tuple[str, ...] = _SMOOTH_JOINTS) -> None:
    """Low-pass the torso rotations and the root path, in place.

    The filter exists because the ground truth in these sets is *jitterier* than
    the generations (0.45° against 0.38° of per-frame shake in the spine and
    collars, against 0.09° on the SOMA baseline). Both tracks pass through the
    same 77-joint → SMPL-X fit, whose 6.5–8.3 mm residual is uncorrelated between
    frames, so the shake belongs to the fit rather than to either motion.

    A 5-frame box at 30 fps removes content above about 7.5 Hz. Voluntary human
    motion does not reach that band, so what comes out is noise: measured over
    ``picks38/gt`` it cuts shake by 63% while moving the pose by under 0.005° at the
    median, 1.21° at p99 and 4.74° at worst.

    That 63% is a source-data figure. Whether it is *visible* is not established —
    see ``docs/dataset-visualisation.md``.

    ``joints22`` is deliberately left alone. The MMD solve reads bone *orientation*
    from the SMPL-X rotations and never reads a frame origin — the root path comes
    from ``transl`` — so positions do not drive the character, and rebuilding them
    by forward kinematics would move a wrist by up to 100 mm to no visible effect.
    """
    if window < 2:
        return
    from motionviewer.core.smplx_fk import (
        SMPLX_BODY22_NAMES,
        axis_angle_from_rotations,
        rodrigues,
    )

    # Column 0 is the root, columns 1..21 are body_pose, matching body-22 order.
    columns = [SMPLX_BODY22_NAMES.index(name) for name in joints if name in SMPLX_BODY22_NAMES]
    locals_ = np.concatenate([payload["global_orient"][:, None, :], payload["body_pose"]], axis=1)
    angles = locals_.copy()
    for column in columns:
        # Only the filtered columns are converted back. Round-tripping the rest
        # through matrices would perturb them by float noise, which costs nothing
        # visually but makes "this joint was untouched" untestable.
        rotations = _project_to_rotations(_moving_average(rodrigues(locals_[:, column]), window))
        angles[:, column] = axis_angle_from_rotations(rotations)
    payload["global_orient"] = angles[:, 0]
    payload["body_pose"] = angles[:, 1:]
    payload["transl"] = _moving_average(payload["transl"], window)


def preprocess(payload: dict, trim: int, window: int) -> None:
    """Head trim then temporal smoothing, in place and in that order.

    Order matters: the frame-0 anchor pop would otherwise be smeared across the
    filter window instead of removed, turning one bad frame into several.
    """
    if trim > 0:
        for key in REQUIRED:
            payload[key] = payload[key][trim:]
    smooth_pose(payload, window)


def caption_slug(caption: str, *, limit: int = 56) -> str:
    """A filename-safe fragment of the caption, cut on a word boundary.

    Captions are the point of a text-to-motion result, so the file it lands in should
    say which one it is without opening anything. ASCII only and no punctuation, since
    these files get dragged between machines and into LaTeX.
    """
    words: list[str] = []
    for raw in caption.lower().split():
        word = "".join(character for character in raw if character.isalnum())
        if not word:
            continue
        if words and len("-".join([*words, word])) > limit:
            break
        words.append(word)
    return "-".join(words)


def video_name(clip: Clip, character: str, *, slug: bool = True) -> str:
    """``<task>_<clip>_<character>__<caption slug>``."""
    stem = f"{clip.task.replace('/', '-')}_{clip.clip_id}_{character}"
    fragment = caption_slug(clip.caption) if slug else ""
    return f"{stem}__{fragment}" if fragment else stem


def _alias(relative: Path) -> str:
    """``picks38_smplx/gt`` -> ``picks38/gt``: shorter to type, still unambiguous."""
    return "/".join(part.removesuffix("_smplx") for part in relative.parts)


def _clip_id(path: Path) -> str:
    """``000003.smplx.npz`` -> ``000003``."""
    return path.name.split(".")[0]


def validate(path: Path) -> tuple[dict | None, str]:
    """Load a clip and check it against the contract. Returns (payload, error)."""
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            missing = [key for key in REQUIRED if key not in keys]
            if missing:
                return None, f"missing {', '.join(missing)}"
            payload = {key: np.asarray(data[key], dtype=np.float64) for key in REQUIRED}
            fps = float(data["fps"]) if "fps" in keys else 30.0
            caption = str(data["caption"]) if "caption" in keys else ""
            extras = {key: float(data[key]) for key in ("fit_mse",) if key in keys and data[key].size == 1}
    except Exception as exc:  # a truncated or non-npz file must not kill the sweep
        return None, f"unreadable: {type(exc).__name__}: {exc}"

    total = len(payload["global_orient"])
    if total < 2:
        return None, f"only {total} frame(s)"
    shapes = {
        "global_orient": (total, 3),
        "transl": (total, 3),
        "joints22": (total, 22, 3),
    }
    for key, want in shapes.items():
        if payload[key].shape != want:
            return None, f"{key} is {payload[key].shape}, expected {want}"
    if payload["body_pose"].size != total * 63:
        return None, f"body_pose has {payload['body_pose'].size} values, expected {total * 63}"
    payload["body_pose"] = payload["body_pose"].reshape(total, 21, 3)
    for key in REQUIRED:
        if not np.isfinite(payload[key]).all():
            return None, f"non-finite values in {key}"
    if not 1.0 <= fps <= 240.0:
        return None, f"implausible fps {fps}"

    payload["_fps"] = fps
    payload["_caption"] = caption
    payload["_extras"] = extras
    return payload, ""


def discover(data_root: Path, patterns: list[str], trim_mode: str, smooth: int = 0) -> dict[str, list[Clip]]:
    """Find every directory of clips under ``data_root`` and score the matching ones.

    Scoring happens *after* trimming, so the numbers on the page describe the
    frames that actually get rendered rather than a lead-in nobody sees.
    """
    directories: dict[str, list[Path]] = {}
    for path in sorted(data_root.rglob("*.npz")):
        relative = path.parent.relative_to(data_root)
        directories.setdefault(_alias(relative), []).append(path)

    selected = {
        task: files
        for task, files in directories.items()
        if any(fnmatch.fnmatch(task, pattern) for pattern in patterns)
    }

    tasks: dict[str, list[Clip]] = {}
    for task, files in sorted(selected.items()):
        clips: list[Clip] = []
        for path in files:
            payload, error = validate(path)
            if payload is None:
                clips.append(Clip(task, _clip_id(path), path, 0, 0.0, "", error=error))
                continue

            if trim_mode == "none":
                trim, reason = 0, ""
            elif trim_mode == "auto":
                trim, reason = detect_trim(payload["joints22"])
            else:
                trim = min(int(trim_mode), len(payload["global_orient"]) - 2)
                reason = f"dropped first {trim} frame(s) as asked" if trim > 0 else ""
            if len(payload["global_orient"]) - trim < 2:
                clips.append(
                    Clip(task, _clip_id(path), path, 0, 0.0, "", error="too few frames left after trim")
                )
                continue
            preprocess(payload, trim, smooth)
            if smooth >= 2:
                reason = f"{reason}; smoothed over {smooth} frames".lstrip("; ")

            total = len(payload["global_orient"])
            metrics = score_clip(payload, fps=payload["_fps"])
            metrics.update(payload["_extras"])
            clips.append(
                Clip(
                    task=task,
                    clip_id=_clip_id(path),
                    path=path,
                    frames=total,
                    fps=payload["_fps"],
                    caption=payload["_caption"],
                    score=metrics,
                    trim=trim,
                    trim_reason=reason,
                )
            )
        tasks[task] = clips
    return tasks


def materialize(clip: Clip, cache: Path, smooth: int) -> str:
    """Write the preprocessed clip for Blender to read. Error string, empty on success.

    The cache name carries the settings, so changing the smoothing window does not
    silently reuse a file produced under the old one.
    """
    if clip.trim <= 0 and smooth < 2:
        return ""
    stem = f"{clip.task.replace('/', '-')}_{clip.clip_id}_t{clip.trim}_s{max(smooth, 0)}"
    destination = cache / f"{stem}.npz"
    if not destination.is_file():
        payload, error = validate(clip.path)
        if payload is None:
            return f"could not reload clip: {error}"
        preprocess(payload, clip.trim, smooth)
        arrays = {key: np.asarray(payload[key], dtype=np.float32) for key in REQUIRED}
        # body_pose is stored flat upstream; keep that shape so the file is a
        # drop-in replacement for the original.
        arrays["body_pose"] = arrays["body_pose"].reshape(len(arrays["global_orient"]), -1)
        arrays["fps"] = np.float32(payload["_fps"])
        arrays["caption"] = np.str_(payload["_caption"])
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez(destination, **arrays)
        except Exception as exc:
            return f"could not write preprocessed clip: {type(exc).__name__}: {exc}"
    clip.render_path = destination
    return ""


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def _label_image(text: str, width: int, path: Path) -> Path | None:
    """Render a caption bar to PNG.

    This ffmpeg build has no ``drawtext`` (no libfreetype), so the text is drawn
    with PIL and composited as an image instead.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    font = None
    size = max(14, int(width * 0.042))
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                font = ImageFont.truetype(candidate, size)
                break
            except OSError:
                continue
    if font is None:
        return None
    height = int(size * 2.0)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, height), fill=(12, 14, 18, 190))
    draw.text((int(size * 0.55), int(size * 0.42)), text, font=font, fill=(238, 236, 232, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _frame_count(directory: Path) -> int:
    return len(list(directory.glob("frame_*.png")))


def encode(panels: list[tuple[Path, str]], destination: Path, fps: float, width: int) -> str:
    """Encode one or more frame directories into a single MP4, side by side.

    Panels of unequal length are padded by holding the last frame rather than
    time-stretched: a generation and its ground truth genuinely differ in
    duration, and rescaling one to match would change its speed and misrepresent
    the model.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "ffmpeg not on PATH"
    counts = [_frame_count(directory) for directory, _ in panels]
    if any(count == 0 for count in counts):
        return "no frames rendered"
    longest = max(counts)

    command = [ffmpeg, "-y", "-loglevel", "error"]
    for directory, _ in panels:
        command += ["-framerate", f"{fps:g}", "-start_number", "1", "-i", str(directory / "frame_%04d.png")]

    # One label image per panel, added as extra inputs after the frame sequences
    # so the filter graph can address them by index.
    label_dir = destination.parent / ".labels"
    labels: list[Path | None] = []
    for index, (_, text) in enumerate(panels):
        image = _label_image(text, width, label_dir / f"{destination.stem}_{index}.png") if text else None
        labels.append(image)
        if image is not None:
            command += ["-loop", "1", "-i", str(image)]

    steps: list[str] = []
    streams: list[str] = []
    next_input = len(panels)
    for index, count in enumerate(counts):
        current = f"{index}:v"
        if count < longest:
            # Hold the last frame rather than time-stretch: a generation and its
            # ground truth genuinely differ in length, and rescaling would change
            # the motion's speed.
            steps.append(
                f"[{current}]tpad=stop_mode=clone:stop_duration={(longest - count) / fps:.4f}[p{index}]"
            )
            current = f"p{index}"
        if labels[index] is not None:
            steps.append(f"[{current}][{next_input}:v]overlay=0:0:shortest=1[l{index}]")
            next_input += 1
            current = f"l{index}"
        streams.append(f"[{current}]")

    if len(streams) > 1:
        steps.append(f"{''.join(streams)}hstack=inputs={len(streams)}[stacked]")
        current = "stacked"
    else:
        current = streams[0].strip("[]")
    # Web players refuse odd dimensions, and hstack can produce them.
    steps.append(f"[{current}]scale=trunc(iw/2)*2:trunc(ih/2)*2[out]")

    command += [
        "-filter_complex",
        ";".join(steps),
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        # 16 rather than 19: 47.0 dB against 45.0 dB PSNR on cel-shaded frames, for
        # 62% more bytes. Flat colour areas are where H.264 shows banding first.
        "16",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    for image in labels:
        if image is not None:
            image.unlink(missing_ok=True)
    shutil.rmtree(label_dir, ignore_errors=True)
    if result.returncode != 0:
        return f"ffmpeg: {result.stderr.strip()[-300:]}"
    return ""


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_clip(clip: Clip, character: str, frames_dir: Path, args, share_with: list[Path]) -> str:
    """Run Blender for one clip. Returns an error string, empty on success."""
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
        str(clip.motion_path),
        "--output",
        str(frames_dir),
        "--views",
        args.view,
        "--panels",
        "skeleton,character" if args.skeleton else "character",
        "--resolution",
        str(args.resolution),
        "--aspect",
        str(args.aspect),
        "--toon",
        "--expression",
        args.expression,
        "--caption",
        clip.caption,
        "--camera",
        args.camera,
        "--follow-smooth",
        str(args.follow_smooth),
    ]
    if not args.no_spring:
        command.append("--spring")
    for path in share_with:
        command += ["--frame-motion", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-6:])
        return f"blender exited {result.returncode}\n{tail}"
    if not (frames_dir / "character" / args.view).is_dir():
        return "blender produced no character frames"
    return ""


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:26px clamp(14px,3vw,48px) 64px; background:#14161a; color:#e8e6e3;
         font:15px/1.55 -apple-system,"Helvetica Neue","PingFang SC",sans-serif; }}
  h1 {{ font-size:21px; margin:0 0 6px; }}
  .lede {{ color:#9aa0a6; margin:0 0 20px; max-width:78ch; }}
  .lede code {{ color:#c8cdd3; }}
  .grid {{ display:grid; gap:20px; grid-template-columns:repeat(auto-fill,minmax(var(--w),1fr)); }}
  article {{ background:#1a1d22; border-radius:10px; overflow:hidden; }}
  video {{ width:100%; display:block; background:#0d0f12; }}
  h2 {{ font-size:13px; font-weight:600; margin:10px 12px 2px; font-variant-numeric:tabular-nums; }}
  h2 span {{ color:#7f868d; font-weight:400; }}
  .meta {{ color:#7f868d; font-size:12px; margin:0 12px 5px; font-variant-numeric:tabular-nums; }}
  .caption {{ color:#b9bec4; font-size:13px; margin:0 12px 10px; }}
  .flags {{ margin:0 12px 12px; display:flex; gap:6px; flex-wrap:wrap; }}
  .flag {{ font-size:11px; padding:2px 7px; border-radius:20px; background:#3a2a1c; color:#e8b57a; }}
  .tools {{ position:sticky; top:0; z-index:2; background:#14161a; padding:8px 0 14px;
            display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  input[type=search] {{ background:#1a1d22; border:1px solid #2b3038; color:#e8e6e3;
            border-radius:7px; padding:8px 11px; font:14px inherit; min-width:min(340px,70vw); }}
  select {{ background:#1a1d22; border:1px solid #2b3038; color:#e8e6e3; border-radius:7px;
            padding:8px 9px; font:14px inherit; }}
  #count {{ color:#7f868d; font-size:13px; font-variant-numeric:tabular-nums; }}
  article[hidden] {{ display:none; }}
  details {{ margin:0 0 20px; color:#9aa0a6; }}
  summary {{ cursor:pointer; color:#c8cdd3; }}
  table {{ border-collapse:collapse; font-size:12.5px; margin-top:8px; }}
  td, th {{ text-align:left; padding:3px 14px 3px 0; font-variant-numeric:tabular-nums; }}
</style></head><body>
<h1>{title}</h1>
<p class="lede">{lede}</p>
{report}
<div class="tools">
  <input type="search" id="q" placeholder="filter by caption, clip id, task or character">
  <select id="sort">
    <option value="0">order: as rendered</option>
    <option value="activity">most active first</option>
    <option value="penalty">cleanest first</option>
    <option value="frames">longest first</option>
  </select>
  <span id="count"></span>
</div>
<div class="grid" id="grid" style="--w:{card}px">{cards}</div>
<script>
const grid = document.getElementById('grid');
const cards = [...grid.querySelectorAll('article')];
const box = document.getElementById('q');
const sort = document.getElementById('sort');
const count = document.getElementById('count');
function apply() {{
  const terms = box.value.toLowerCase().split(/\\s+/).filter(Boolean);
  let shown = 0;
  for (const card of cards) {{
    const hit = terms.every(t => card.dataset.text.includes(t));
    card.hidden = !hit;
    if (hit) shown++;
  }}
  const key = sort.value;
  if (key !== '0') {{
    const dir = key === 'penalty' ? 1 : -1;
    [...cards].sort((a, b) => dir * (parseFloat(a.dataset[key]) - parseFloat(b.dataset[key])))
      .forEach(c => grid.appendChild(c));
  }}
  count.textContent = shown + ' / ' + cards.length;
}}
box.addEventListener('input', apply);
sort.addEventListener('change', apply);
apply();
</script>
</body></html>
"""


def write_captions(output: Path, entries: list[dict]) -> None:
    """A tab-separated index of video to caption, openable in Excel or Numbers."""
    lines = ["video\ttask\tclip_id\tcharacter\tframes\tcaption"]
    for entry in sorted(entries, key=lambda item: (item["task"], item["clip_id"], item["character"])):
        caption = str(entry.get("caption", "")).replace("\t", " ").replace("\n", " ")
        lines.append(
            "\t".join(
                [
                    Path(entry["video"]).name,
                    entry["task"],
                    entry["clip_id"],
                    entry["character"],
                    str(entry["frames"]),
                    caption,
                ]
            )
        )
    (output / "captions.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_page(output: Path, entries: list[dict], skipped: list[Clip], title: str, card_width: int) -> None:
    cards: list[str] = []
    for entry in entries:
        score = entry.get("score", {})
        bits = [
            f"{entry['frames']} frames @ {entry['fps']:g}fps",
            f"travel {score.get('root_travel_m', 0):.1f} m",
            f"activity {score.get('activity', 0):.2f}",
            f"penalty {score.get('penalty', 0):.2f}",
        ]
        flags: list[str] = []
        if entry.get("trim_reason"):
            flags.append(entry["trim_reason"])
        if score.get("penetration_cm", 0) > 5.0:
            flags.append(f"sinks {score['penetration_cm']:.0f} cm into the floor")
        if score.get("max_angular_velocity_deg_s", 0) > 900:
            flags.append(
                f"{score['max_angular_velocity_deg_s']:.0f}°/s at {score.get('max_angular_velocity_joint', '?')}"
            )
        if score.get("jitter_deg", 0) > 0.5:
            flags.append(f"jitter {score['jitter_deg']:.2f}° at {score.get('jitter_joint', '?')}")
        if score.get("foot_skate_m_s", 0) > 0.35:
            flags.append(f"foot skate {score['foot_skate_m_s']:.2f} m/s")
        searchable = " ".join(
            [entry["clip_id"], entry.get("caption", ""), entry["task"], entry["character"]]
        ).lower()
        cards.append(
            f'<article data-text="{html.escape(searchable)}"'
            f' data-activity="{score.get("activity", 0):.4f}"'
            f' data-penalty="{score.get("penalty", 0):.4f}"'
            f' data-frames="{entry["frames"]}">'
            f'<video src="{html.escape(entry["video"])}" controls loop muted playsinline preload="none"></video>'
            f"<h2>{html.escape(entry['clip_id'])} "
            f"<span>{html.escape(entry['label'])} · {html.escape(entry['character'])}</span></h2>"
            f'<p class="meta">{html.escape(" · ".join(bits))}</p>'
            f'<p class="caption">{html.escape(entry.get("caption", ""))}</p>'
            + (
                '<p class="flags">'
                + "".join(f'<span class="flag">{html.escape(f)}</span>' for f in flags)
                + "</p>"
                if flags
                else ""
            )
            + "</article>"
        )

    report = ""
    if skipped:
        rows = "".join(
            f"<tr><td>{html.escape(c.task)}</td><td>{html.escape(c.clip_id)}</td>"
            f"<td>{html.escape(c.error)}</td></tr>"
            for c in skipped[:200]
        )
        report = (
            f"<details><summary>{len(skipped)} clip(s) skipped — click for the reason</summary>"
            f"<table><tr><th>task</th><th>clip</th><th>reason</th></tr>{rows}</table></details>"
        )

    lede = (
        f"{len(entries)} video(s). Cel shaded with each model's own toon ramp and sphere maps, "
        "outlined, floor shadow, and spring bones baked on every bone the PMX marks as dynamic. "
        "Videos load on demand — click one to play. Orange badges mark problems measured in the "
        "<em>source motion</em>, not introduced by the retarget: they are left in the render on purpose."
    )
    (output / "index.html").write_text(
        _PAGE.format(
            title=html.escape(title),
            lede=lede,
            report=report,
            cards="".join(cards),
            card=card_width,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, score, render and publish a SMPL-X dataset on an MMD character.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Root of the clip tree")
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task to render, e.g. picks38/gt. Repeatable, glob allowed. Default: all",
    )
    parser.add_argument("--compare", default="", help="Render this task beside --task, same camera")
    parser.add_argument("--list", action="store_true", help="Show what is in --data and exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate and score, render nothing")
    parser.add_argument("--character", action="append", default=None, help="Repeatable")
    parser.add_argument("--clips", default="", help="Comma-separated clip ids, overrides sorting")
    parser.add_argument("--limit", type=int, default=0, help="0 renders everything selected")
    parser.add_argument(
        "--sort",
        default="activity",
        choices=("activity", "penalty", "frames", "travel", "name"),
        help="activity puts the most expressive clips first",
    )
    parser.add_argument(
        "--trim",
        default="auto",
        help=(
            "auto detects and drops a frozen lead-in or a frame-0 anchor pop; "
            "'none' renders the clip untouched; an integer drops that many frames"
        ),
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help=(
            "Box-filter width in frames for the source rotations and root path. "
            "0 or 1 leaves the motion untouched; 5 cuts the fit's jitter by 62%% "
            "and moves no joint by more than 1.8 deg. Off by default: it edits the "
            "motion, and that is the user's call on a figure meant as evidence"
        ),
    )
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-penalty", type=float, default=0.0, help="0 disables the quality gate")
    parser.add_argument("--output", type=Path, default=ROOT / "converted/dataset_videos")
    parser.add_argument("--view", default="three_quarter")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1440,
        help=(
            "Frame height in pixels, per panel. 800 was measurably soft: against it, "
            "1440 carries 56%% more edge detail and 1920 only 72%%, so this is where the "
            "curve starts to flatten against render time"
        ),
    )
    parser.add_argument("--aspect", type=float, default=0.72, help="Width / height")
    parser.add_argument(
        "--camera",
        default="follow",
        choices=("auto", "static", "follow"),
        help=(
            "follow sizes the figure from its poses instead of its trajectory, which "
            "a static camera cannot: framed to its whole path, a 6 m walk leaves a "
            "100 px character in an 800 px frame. Under follow the floor gets a "
            "faint grid, without which a tracking shot turns a walk into a treadmill"
        ),
    )
    parser.add_argument("--follow-smooth", type=int, default=11)
    parser.add_argument(
        "--no-caption-in-name",
        action="store_true",
        help="Name videos <task>_<clip>_<character> only, without the caption fragment",
    )
    parser.add_argument("--no-spring", action="store_true")
    parser.add_argument(
        "--expression",
        default="soft_smile",
        help=(
            "soft_smile rather than smile: at the size a face occupies in a full-body "
            "frame (~100 px at 1440p) the stronger preset narrows the eyes into lines "
            "and the irises disappear, which reads as squinting rather than smiling"
        ),
    )
    parser.add_argument("--skeleton", action="store_true", help="Also render the source skeleton panel")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"--data {args.data} is not a directory")

    if args.trim not in {"auto", "none"}:
        try:
            int(args.trim)
        except ValueError:
            raise SystemExit("--trim must be 'auto', 'none' or an integer") from None

    patterns = args.task or ["*"]
    if args.compare:
        patterns = list(patterns) + [args.compare]
    tasks = discover(args.data, patterns, args.trim, args.smooth)
    if not tasks:
        available = discover(args.data, ["*"], "none", 0)
        raise SystemExit(f"no task matched {patterns}. Available: {sorted(available) or '(none)'}")

    if args.list or args.dry_run:
        for task, clips in tasks.items():
            good = [c for c in clips if not c.error]
            bad = [c for c in clips if c.error]
            print(f"\n{task}: {len(clips)} clips, {len(good)} valid, {len(bad)} unusable")
            if good:
                frames = np.array([c.frames for c in good])
                pen = np.array([c.score["penalty"] for c in good])
                sink = np.array([c.score["ground_penetration_m"] for c in good]) * 100
                print(f"   frames  {frames.min()} .. {frames.max()} (median {int(np.median(frames))})")
                print(f"   fps     {sorted({c.fps for c in good})}")
                print(f"   penalty median {np.median(pen):.2f}, p90 {np.percentile(pen, 90):.2f}")
                print(f"   floor sink median {np.median(sink):.1f} cm, worst {sink.max():.1f} cm")
                trimmed = [c for c in good if c.trim]
                if trimmed:
                    reasons: dict[str, int] = {}
                    for clip in trimmed:
                        reasons[clip.trim_reason.split(" (")[0]] = (
                            reasons.get(clip.trim_reason.split(" (")[0], 0) + 1
                        )
                    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
                        print(f"   trim    {count}/{len(good)} clips: {reason}")
                for clip in sorted(good, key=lambda c: -c.score["activity"])[:3]:
                    print(f"   top activity  {clip.clip_id}  {clip.caption[:64]}")
            for clip in bad[:5]:
                print(f"   UNUSABLE {clip.clip_id}: {clip.error}")
        if args.list:
            return

    characters = args.character or ["yoimiya"]
    unknown = [name for name in characters if name not in CHARACTERS]
    if unknown:
        raise SystemExit(f"unknown character(s) {unknown}; known: {sorted(CHARACTERS)}")
    for name in characters:
        if not (ROOT / CHARACTERS[name]).is_file():
            raise SystemExit(f"character asset missing: {CHARACTERS[name]}")

    # ---- selection ---------------------------------------------------------
    primary = [task for task in tasks if task != args.compare] if args.compare else list(tasks)
    wanted_ids = [value.strip() for value in args.clips.split(",") if value.strip()]
    compare_by_id = {c.clip_id: c for c in tasks.get(args.compare, []) if not c.error}

    selection: list[Clip] = []
    skipped: list[Clip] = []
    for task in primary:
        clips = [c for c in tasks[task] if not c.error]
        skipped += [c for c in tasks[task] if c.error]
        if wanted_ids:
            clips = [c for c in clips if c.clip_id in wanted_ids]
        else:
            clips = [c for c in clips if c.frames >= args.min_frames]
            if args.max_penalty > 0:
                clips = [c for c in clips if c.score["penalty"] <= args.max_penalty]
        keys = {
            "activity": lambda c: -c.score["activity"],
            "penalty": lambda c: c.score["penalty"],
            "frames": lambda c: -c.frames,
            "travel": lambda c: -c.score["root_travel_m"],
            "name": lambda c: c.clip_id,
        }
        clips.sort(key=keys[args.sort])
        if args.limit > 0:
            clips = clips[: args.limit]
        if args.compare:
            paired = [c for c in clips if c.clip_id in compare_by_id]
            # A clip with no counterpart cannot be put side by side, and dropping it
            # in silence looks like the selection simply came up short — most likely
            # when --task matches a set the compare task knows nothing about.
            if len(paired) < len(clips):
                print(
                    f"{task}: {len(clips) - len(paired)} of {len(clips)} clip(s) have no "
                    f"counterpart in {args.compare} and cannot be paired"
                )
            clips = paired
        selection += clips

    total = len(selection) * len(characters)
    width = max(int(round(args.resolution * args.aspect)), 16)
    print(
        f"\n{len(selection)} clip(s) x {len(characters)} character(s) = {total} video(s)"
        f"{' (side-by-side)' if args.compare else ''}, {len(skipped)} unusable clip(s) skipped"
    )
    if args.dry_run:
        for clip in selection[:40]:
            print(f"   {clip.task}/{clip.clip_id}  {clip.frames}f  act {clip.score['activity']:.2f}")
        return
    if not selection:
        raise SystemExit("nothing selected; loosen --limit / --min-frames / --max-penalty")

    # ---- render ------------------------------------------------------------
    videos_dir = args.output / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_root = args.output / ".frames"
    trim_cache = args.output / "trimmed"
    manifest_path = args.output / "manifest.json"
    entries: list[dict] = []
    if manifest_path.is_file():
        entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("videos", [])
    done = {(e["task"], e["clip_id"], e["character"]) for e in entries}

    # Refresh the metrics on videos rendered by an earlier run. Scores come from
    # the clip, not the render, so when a metric is corrected the page can be
    # brought up to date without re-rendering hours of video.
    known = {(clip.task, clip.clip_id): clip for clips in tasks.values() for clip in clips if not clip.error}
    refreshed = 0
    for entry in entries:
        clip = known.get((entry["task"], entry["clip_id"]))
        if clip is None:
            continue
        score = dict(clip.score)
        score["penetration_cm"] = score.get("ground_penetration_m", 0.0) * 100
        # Everything here is derived from the clip rather than the render, caption
        # included, so all of it can go stale when the source or a metric changes.
        fresh = {
            "score": score,
            "frames": clip.frames,
            "fps": clip.fps,
            "caption": clip.caption,
            "trim": clip.trim,
            "trim_reason": clip.trim_reason,
        }
        if all(entry.get(key) == value for key, value in fresh.items()):
            continue
        entry.update(fresh)
        refreshed += 1
    if refreshed:
        print(f"refreshed metrics on {refreshed} previously rendered video(s)")

    title = (
        args.title
        or (f"{args.compare} vs {primary[0]}" if args.compare else " · ".join(primary)) + " — SMPL-X → MMD"
    )
    failures: list[str] = []
    started = time.time()
    index = 0
    # Rate is measured over work done *in this run*, and in frames rather than
    # clips: these sets mix 29-frame and 720-frame clips, so a per-clip average
    # would put the estimate out by an order of magnitude either way.
    frames_done = 0
    frames_left = sum(
        (clip.frames + (compare_by_id[clip.clip_id].frames if args.compare else 0)) * len(characters)
        for clip in selection
        if not args.compare or clip.clip_id in compare_by_id
    )
    for clip in selection:
        other = compare_by_id.get(clip.clip_id) if args.compare else None
        for character in characters:
            index += 1
            key = (clip.task, clip.clip_id, character)
            name = video_name(clip, character, slug=not args.no_caption_in_name)
            destination = videos_dir / f"{name}.mp4"
            work = clip.frames + (other.frames if other else 0)
            # Trust the path recorded in the manifest rather than recomputing it, so a
            # change to the naming scheme does not silently re-render finished work.
            recorded = next(
                (
                    args.output / entry["video"]
                    for entry in entries
                    if (entry["task"], entry["clip_id"], entry["character"]) == key
                ),
                destination,
            )
            if key in done and recorded.is_file():
                frames_left -= work
                continue
            eta = ""
            if frames_done:
                seconds = (time.time() - started) / frames_done * frames_left
                eta = f"  eta {seconds / 60:.0f} min"
            print(
                f"[{index}/{total}] {name}  {clip.frames}f  act {clip.score['activity']:.2f}"
                f"  {frames_left} frames left{eta}"
            )

            panels: list[tuple[Path, str]] = []
            targets = [(other, args.compare), (clip, clip.task)] if other else [(clip, clip.task)]
            failed = next(
                (error for error in (materialize(t, trim_cache, args.smooth) for t, _ in targets) if error),
                "",
            )
            # Both sides of a pair are framed from the union of the two, so a
            # run-away generation and its ground truth stay the same size on screen.
            share = [t.motion_path for t, _ in targets] if other else []
            for target, label in targets:
                if failed:
                    break
                frames_dir = frames_root / f"{name}_{label.replace('/', '-')}"
                failed = render_clip(
                    target, character, frames_dir, args, [p for p in share if p != target.motion_path]
                )
                if failed:
                    break
                panels.append((frames_dir / "character" / args.view, f"{label}  {target.frames}f"))
            if failed:
                print(f"      render failed: {failed.splitlines()[0]}")
                failures.append(f"{name}: {failed.splitlines()[0]}")
                shutil.rmtree(frames_root, ignore_errors=True)
                frames_left -= work
                continue

            error = encode(panels, destination, clip.fps, width)
            shutil.rmtree(frames_root, ignore_errors=True)
            frames_done += work
            frames_left -= work
            if error:
                print(f"      encode failed: {error}")
                failures.append(f"{name}: {error}")
                continue

            # A sidecar next to the video, so the caption travels with the file even
            # when it is copied out of here on its own.
            destination.with_suffix(".txt").write_text(f"{clip.caption}\n", encoding="utf-8")

            score = dict(clip.score)
            score["penetration_cm"] = score.get("ground_penetration_m", 0.0) * 100
            entries.append(
                {
                    "task": clip.task,
                    "label": f"{args.compare} | {clip.task}" if other else clip.task,
                    "clip_id": clip.clip_id,
                    "character": character,
                    "video": f"videos/{destination.name}",
                    "frames": clip.frames,
                    "fps": clip.fps,
                    "caption": clip.caption,
                    "source": str(clip.path),
                    "compare_source": str(other.path) if other else None,
                    "trim": clip.trim,
                    "trim_reason": clip.trim_reason,
                    "compare_trim": other.trim if other else None,
                    "score": score,
                }
            )
            done.add(key)
            manifest_path.write_text(
                json.dumps({"videos": entries}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_captions(args.output, entries)
            write_page(args.output, entries, skipped, title, width * (2 if other else 1) // 2 + 130)

    shutil.rmtree(frames_root, ignore_errors=True)
    # Written unconditionally so a run that only refreshed metrics still persists
    # them, rather than leaving the page and the manifest disagreeing.
    manifest_path.write_text(json.dumps({"videos": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_captions(args.output, entries)
    write_page(args.output, entries, skipped, title, width * (2 if args.compare else 1) // 2 + 130)
    print(f"\n{len(entries)} video(s) in {args.output}")
    print(f"captions: {args.output / 'captions.tsv'} (and one .txt beside each video)")
    print(f"open {args.output / 'index.html'}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for line in failures[:20]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
