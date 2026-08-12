from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from motionviewer.core.palette import Color, palette_color


def draw_legend_on_frames(
    frames_dir: Path,
    *,
    labels: list[str],
    palette: str,
    instruction: str | None = None,
) -> None:
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths or not labels:
        return

    colors = [palette_color(idx, palette) for idx, _ in enumerate(labels)]
    font = ImageFont.load_default(size=18)
    small_font = ImageFont.load_default(size=14)

    for path in frame_paths:
        image = Image.open(path).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x = image.width - 220
        y = 28
        line_h = 24
        panel_h = line_h * len(labels) + (34 if instruction else 14)
        draw.rounded_rectangle(
            (x - 16, y - 12, image.width - 24, y + panel_h),
            radius=10,
            fill=(0, 0, 0, 70),
        )
        for idx, (label, color) in enumerate(zip(labels, colors)):
            yy = y + idx * line_h
            draw.rounded_rectangle(
                (x, yy + 4, x + 14, yy + 18),
                radius=3,
                fill=_rgba255(color, 230),
            )
            draw.text((x + 22, yy), label, fill=(238, 238, 232, 235), font=font)
        if instruction:
            text = instruction if len(instruction) <= 42 else instruction[:39] + "..."
            draw.text((x, y + len(labels) * line_h + 6), text, fill=(210, 210, 205, 220), font=small_font)
        composed = Image.alpha_composite(image, overlay)
        composed.save(path)


def draw_instruction_banner(
    frames_dir: Path,
    *,
    instruction: str,
    max_chars_per_line: int = 62,
    max_lines: int = 3,
) -> None:
    """Draw the text-to-motion prompt as a top-centered banner, independent of the labels mode.

    `world`-mode labels only annotate each actor with its model name; without a prefix clip,
    the instruction text is the only context a viewer has for judging a text-to-motion result,
    so it always gets its own readable, non-truncated (wrapped) banner.
    """
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths or not instruction.strip():
        return

    font = ImageFont.load_default(size=20)
    lines = textwrap.wrap(instruction.strip(), width=max_chars_per_line) or [instruction.strip()]
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1].rstrip() + "..."]
    line_h = 26
    pad_x, pad_y = 20, 14

    for path in frame_paths:
        image = Image.open(path).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        text_w = max(draw.textlength(line, font=font) for line in lines)
        panel_w = min(image.width - 64, text_w + pad_x * 2)
        panel_h = len(lines) * line_h + pad_y * 2 - 4
        x0 = (image.width - panel_w) / 2
        y0 = 24.0
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=12, fill=(0, 0, 0, 92))
        for idx, line in enumerate(lines):
            line_w = draw.textlength(line, font=font)
            tx = x0 + (panel_w - line_w) / 2
            ty = y0 + pad_y + idx * line_h
            draw.text((tx, ty), line, fill=(240, 240, 235, 235), font=font)
        composed = Image.alpha_composite(image, overlay)
        composed.save(path)


def _rgba255(color: Color, alpha: int) -> tuple[int, int, int, int]:
    return (
        int(max(0, min(255, round(color.r * 255)))),
        int(max(0, min(255, round(color.g * 255)))),
        int(max(0, min(255, round(color.b * 255)))),
        alpha,
    )
