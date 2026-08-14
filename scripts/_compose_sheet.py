"""Compose rendered view x frame PNGs into one contact sheet. Local helper."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", default="front,side,three_quarter")
    parser.add_argument("--frames", default="1,12,24")
    parser.add_argument("--cell", type=int, default=420)
    args = parser.parse_args()

    views = [value.strip() for value in args.views.split(",") if value.strip()]
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    cell = args.cell
    label = 22

    sheet = Image.new("RGB", (cell * len(frames), (cell + label) * len(views)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, view in enumerate(views):
        for column, frame in enumerate(frames):
            path = args.input / f"{view}_{frame:04d}.png"
            top = row * (cell + label)
            draw.text((column * cell + 6, top + 5), f"{view}  f{frame}", fill="black")
            if not path.is_file():
                continue
            image = Image.open(path).convert("RGBA")
            flat = Image.new("RGBA", image.size, (255, 255, 255, 255))
            flat.alpha_composite(image)
            sheet.paste(flat.convert("RGB").resize((cell, cell), Image.LANCZOS), (column * cell, top + label))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"wrote {args.output} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
