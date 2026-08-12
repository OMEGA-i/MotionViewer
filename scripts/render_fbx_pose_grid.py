"""Blender adapter for the generic retargeted FBX pose-grid renderer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _resolution(value: str) -> tuple[int, int]:
    try:
        width, height = (int(item) for item in value.lower().split("x", maxsplit=1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--fbx", type=Path)
    parser.add_argument("--fbx-mode", choices=("fixed", "random-grid", "random-cell"), default="fixed")
    parser.add_argument("--fbx-root", type=Path, default=Path("assets/fbx"))
    parser.add_argument("--fbx-pool", choices=("approved", "binary"), default="approved")
    parser.add_argument(
        "--fbx-model",
        dest="fbx_model_ids",
        action="append",
        default=[],
        help="Limit a random FBX pool to this model id; repeat for multiple models",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source", dest="source_id", default="gt")
    parser.add_argument("--task")
    parser.add_argument("--provenance", dest="provenances", action="append", default=[])
    parser.add_argument("--caption-regex")
    parser.add_argument("--min-frames", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--frame-mode", choices=("first", "random", "index"), default="random")
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--bone-map", default="auto")
    parser.add_argument("--quality-filter", choices=("retarget", "off"), default="retarget")
    parser.add_argument("--max-penetration-cm", type=float, default=1.0)
    parser.add_argument("--max-limb-error-deg", type=float, default=35.0)
    parser.add_argument("--columns", type=int, default=0)
    parser.add_argument("--spacing", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=_resolution, default=(1600, 1600))
    parser.add_argument("--views", default="upper_left,front,upper_right")
    parser.add_argument("--background-color", default="188,170,221")
    parser.add_argument(
        "--background-colors",
        help="Semicolon-separated #RRGGBB or R,G,B colors rendered from each pose grid",
    )
    parser.add_argument("--ground-style", choices=("grid", "solid"), default="grid")
    parser.add_argument("--material-mode", choices=("preserve", "clay"), default="preserve")
    parser.add_argument("--layout-style", choices=("grid", "scatter", "scatter-shallow"), default="grid")
    parser.add_argument("--batch-count", type=int, default=1)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def main() -> int:
    from motionviewer.blender.pose_grid import (
        PoseGridRequest,
        parse_background_color,
        render_pose_grid,
    )

    args = _args()
    try:
        options = vars(args)
        options["provenances"] = tuple(options["provenances"])
        options["fbx_model_ids"] = tuple(options["fbx_model_ids"])
        options["views"] = tuple(item.strip() for item in options["views"].split(",") if item.strip())
        options["background_rgb"] = parse_background_color(options.pop("background_color"))
        background_colors = options.pop("background_colors")
        if background_colors:
            options["background_rgbs"] = tuple(
                parse_background_color(item) for item in background_colors.split(";") if item.strip()
            )
        if options.get("output_dir") is None and options.get("output") is not None:
            options["output_dir"] = options["output"].parent / options["output"].stem
        request = PoseGridRequest(**options)
        report = render_pose_grid(request)
    except (ValueError, RuntimeError) as exc:
        print(f"pose-grid error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Rendered {len(report.accepted)} poses across {len(report.grids)} grid(s); "
        f"manifest: {report.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
