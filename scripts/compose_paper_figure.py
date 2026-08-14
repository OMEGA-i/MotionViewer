"""Compose rendered stills into a filmstrip and a motion-trail overlay.

Both keep the alpha channel, so the figure drops onto a paper's own background
without a white box around it.

  uv run python scripts/compose_paper_figure.py --input outputs/figure/<name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def _load(input_dir: Path) -> list[Image.Image]:
    return [Image.open(path).convert("RGBA") for path in sorted(input_dir.glob("pick_*.png"))]


def _trim_union(images: list[Image.Image]) -> tuple[int, int, int, int]:
    """One crop box for every frame, so the figure does not jitter."""
    boxes = [image.split()[-1].getbbox() for image in images]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return (0, 0, images[0].width, images[0].height)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def filmstrip(images: list[Image.Image], *, gap: int, height: int) -> Image.Image:
    """Each panel is cropped to its own pose.

    A union crop is wrong here: the camera frames the whole trajectory, so on a
    clip that walks 7 m every panel would be mostly empty and the figure would
    drift across it. The strip's job is the pose, not the ground it covers — that
    is what the trail is for.
    """
    cropped = []
    for image in images:
        box = image.split()[-1].getbbox()
        cropped.append(image.crop(box) if box is not None else image)
    scale = height / cropped[0].height
    scaled = [image.resize((max(int(image.width * scale), 1), height), Image.LANCZOS) for image in cropped]
    width = sum(image.width for image in scaled) + gap * (len(scaled) - 1)
    strip = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    x = 0
    for image in scaled:
        strip.alpha_composite(image, (x, 0))
        x += image.width + gap
    return strip


def trail(images: list[Image.Image], *, min_alpha: float, height: int) -> Image.Image:
    """Oldest faintest, newest solid, all in one frame."""
    box = _trim_union(images)
    cropped = [image.crop(box) for image in images]
    scale = height / cropped[0].height
    scaled = [image.resize((max(int(image.width * scale), 1), height), Image.LANCZOS) for image in cropped]
    canvas = Image.new("RGBA", scaled[0].size, (0, 0, 0, 0))
    count = len(scaled)
    for index, image in enumerate(scaled):
        weight = min_alpha + (1.0 - min_alpha) * (index / max(count - 1, 1))
        faded = image.copy()
        alpha = faded.split()[-1].point(lambda value, w=weight: int(value * w))
        faded.putalpha(alpha)
        canvas.alpha_composite(faded)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--min-alpha", type=float, default=0.22)
    args = parser.parse_args()

    images = _load(args.input)
    if not images:
        raise SystemExit(f"no pick_*.png in {args.input}")

    strip = filmstrip(images, gap=args.gap, height=args.height)
    strip_path = args.input / "filmstrip.png"
    strip.save(strip_path)
    print(f"wrote {strip_path} ({strip.width}x{strip.height}, {len(images)} frames)")

    meta_path = args.input / "figure.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if meta.get("trail_recommended", True):
        overlay = trail(images, min_alpha=args.min_alpha, height=args.height)
        overlay_path = args.input / "trail.png"
        overlay.save(overlay_path)
        print(f"wrote {overlay_path} ({overlay.width}x{overlay.height})")
    else:
        print(
            "skipped trail: the root barely moves in this clip, so the poses would "
            "stack on one spot. Use the filmstrip."
        )


if __name__ == "__main__":
    main()
