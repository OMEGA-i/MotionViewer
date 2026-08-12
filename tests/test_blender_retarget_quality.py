"""Opt-in integration coverage for the full Mixamo retarget pipeline."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
MOTION = ROOT / "data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz"
IRON = ROOT / "assets/fbx/iron.fbx"


@pytest.mark.skipif(
    os.environ.get("MOTIONVIEWER_BLENDER_TESTS") != "1",
    reason="set MOTIONVIEWER_BLENDER_TESTS=1 to run Blender integration tests",
)
def test_iron_full_motion_retarget_quality(tmp_path: Path) -> None:
    report_path = tmp_path / "iron_quality.json"
    subprocess.run(
        [
            BLENDER,
            "--background",
            "--python",
            str(ROOT / "scripts/retarget_quality_audit.py"),
            "--",
            "--motion",
            str(MOTION),
            "--asset",
            str(IRON),
            "--output",
            str(report_path),
        ],
        check=True,
        cwd=ROOT,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))["assets"][0]
    assert report["bone_coverage"] == {"mapped": 22, "expected": 22, "complete": True}
    assert report["rig_preflight"] == {"valid": True, "prefix": "", "errors": []}
    assert report["rig_profile"]["schema"] == "motionviewer.mixamo_rig.v2"
    assert report["rig_profile"]["calibration"]["transform_baked"] is False
    assert report["maximum_quaternion_norm_error"] < 1e-4
    assert report["maximum_bone_length_drift_relative"] < 1e-4
    assert report["minimum_profiled_sole_z_m"] >= -0.005
    assert max(report["maximum_contact_drift_m"].values()) <= max(0.01, 0.0075 * report["target_height_m"])
    assert report["mean_joint_error_m"] <= 0.04 * report["target_height_m"]
    assert report["p95_joint_error_m"] <= 0.08 * report["target_height_m"]
    assert max(report["endpoint_error_m"][name] for name in ("left_wrist", "right_wrist")) <= (
        0.08 * report["target_height_m"]
    )
    assert report["representative_frames"]
