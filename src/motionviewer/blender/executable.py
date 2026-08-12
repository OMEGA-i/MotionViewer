"""Locate the Blender executable without coupling callers to one platform."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

BLENDER_ENV = "MOTIONVIEWER_BLENDER"


def resolve_blender(explicit: str | Path | None = None) -> Path:
    """Return an executable Blender path or raise a setup-oriented error."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    configured = os.environ.get(BLENDER_ENV)
    if configured:
        candidates.append(Path(configured).expanduser())
    discovered = shutil.which("blender")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(
        (
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path.home() / ".local/bin/blender",
            Path("/usr/bin/blender"),
        )
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    requested = f" {explicit!s}" if explicit else ""
    raise FileNotFoundError(
        f"Blender executable{requested} was not found. Pass --blender PATH or set {BLENDER_ENV}."
    )
