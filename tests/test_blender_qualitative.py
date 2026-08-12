from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from motionviewer.video.qualitative import QualitativeBatchRequest, prepare_qualitative_batch

ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
PACKAGE = ROOT / "data/local/packages/soma_tmr_test_1638"
FBX_ROOT = ROOT / "assets/fbx"


@pytest.mark.skipif(
    os.environ.get("MOTIONVIEWER_BLENDER_TESTS") != "1",
    reason="set MOTIONVIEWER_BLENDER_TESTS=1 to run Blender integration tests",
)
def test_one_clip_renders_three_shared_camera_transparent_pngs(tmp_path: Path) -> None:
    request = QualitativeBatchRequest(
        package=PACKAGE,
        output_dir=tmp_path / "qualitative",
        provenance_counts=(("HumanML3D", 1),),
        fbx_root=FBX_ROOT,
        resolution=(480, 480),
        samples=8,
    )
    batch = prepare_qualitative_batch(request)
    job = batch.jobs[0]
    subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--python",
            str(ROOT / "scripts/render_qualitative_clip.py"),
            "--",
            "--bundle",
            str(job.bundle_path),
        ],
        cwd=ROOT,
        check=True,
    )

    assert len(job.sources) == 3
    for source in job.sources:
        image = Image.open(source.output_path).convert("RGBA")
        assert image.size == (480, 480)
        alpha = image.getchannel("A")
        assert alpha.getextrema() == (0, 255)
        bounds = alpha.getbbox()
        assert bounds is not None
        assert min(bounds[0], bounds[1], 480 - bounds[2], 480 - bounds[3]) >= 4
    manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
    assert manifest["render"]["transparent_background"] is True
    assert manifest["render"]["labels"] is False
    assert manifest["render"]["ground"] is False
    assert manifest["fbx"]["model_id"] == job.fbx_model_id
    assert manifest["snapshot_layout"] == "root_aligned"
    assert manifest["material_mode"] == "preserve"
    assert len({tuple(manifest["camera"]["bounds"][side]) for side in ("min", "max")}) == 2
