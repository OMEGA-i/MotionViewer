#!/usr/bin/env python3
"""Compatibility wrapper for the packaged qualitative Blender entry point."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from motionviewer.blender.qualitative_entry import main

if __name__ == "__main__":
    raise SystemExit(main())
