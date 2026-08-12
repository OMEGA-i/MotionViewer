"""Opt-in Blender coverage for the generic FBX pose-grid renderer."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
PACKAGE = ROOT / "data/local/packages/soma_tmr_test_1638"
CH43 = ROOT / "assets/fbx/Ch43_nonPBR.fbx"


@pytest.mark.skipif(
    os.environ.get("MOTIONVIEWER_BLENDER_TESTS") != "1" or not PACKAGE.is_dir() or not CH43.is_file(),
    reason="set MOTIONVIEWER_BLENDER_TESTS=1 with the local SOMA package and Ch43 asset",
)
def test_ch43_pose_grid_has_three_opaque_views(tmp_path: Path) -> None:
    output_dir = tmp_path / "ch43_grid"
    subprocess.run(
        [
            BLENDER,
            "--background",
            "--python",
            str(ROOT / "scripts/render_fbx_pose_grid.py"),
            "--",
            "--package",
            str(PACKAGE),
            "--fbx",
            str(CH43),
            "--count",
            "2",
            "--frame-mode",
            "random",
            "--seed",
            "7",
            "--quality-filter",
            "off",
            "--output-dir",
            str(output_dir),
            "--views",
            "upper_left,front,upper_right",
        ],
        check=True,
        cwd=ROOT,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    grid = manifest["grids"][0]
    images = [
        Image.open(output_dir / "grid_000_ch43_nonpbr" / f"{view}.png")
        for view in ("upper_left", "front", "upper_right")
    ]

    assert all(image.mode in {"RGB", "RGBA"} for image in images)
    for image in images:
        rgba = image.convert("RGBA")
        corners = (
            rgba.getpixel((0, 0)),
            rgba.getpixel((rgba.width - 1, 0)),
            rgba.getpixel((0, rgba.height - 1)),
            rgba.getpixel((rgba.width - 1, rgba.height - 1)),
        )
        assert all(alpha == 255 for *_, alpha in corners)
        assert all(blue > red > green for red, green, blue, _ in corners)
    assert len({image.tobytes() for image in images}) == 3
    report = json.loads((output_dir / "grid_000_ch43_nonpbr" / "selection.json").read_text(encoding="utf-8"))
    assert report["status"] == "rendered"
    assert report["accepted_count"] == 2
    assert report["attempted_count"] >= 2
    assert grid["view_outputs"].keys() == {"upper_left", "front", "upper_right"}
    assert not list(output_dir.glob(".pose-grid-*"))
    for sample in report["accepted"]:
        assert 0 <= sample["frame_index"] < sample["source_frames"]
        assert sample["metrics"]["finite_mesh"] is True
