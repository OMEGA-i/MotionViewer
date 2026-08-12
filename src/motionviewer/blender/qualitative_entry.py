"""Blender entry point for one shared-camera qualitative comparison clip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def main() -> int:
    from motionviewer.blender.qualitative import render_qualitative_bundle, write_failed_status

    args = _args()
    try:
        status = render_qualitative_bundle(args.bundle)
    except Exception as exc:  # Blender adapter must persist failure status.
        write_failed_status(args.bundle, str(exc))
        print(f"qualitative render error: {exc}", file=sys.stderr)
        return 2
    print(f"Rendered {status['clip_id']}: {', '.join(status['outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
