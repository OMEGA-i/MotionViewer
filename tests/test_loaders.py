from pathlib import Path

import numpy as np

from motionviewer.core.schema import RenderCapability
from motionviewer.loaders import default_registry

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "examples" / "smplx_body22_fitted_aa"


def test_smplx_body22_loader_reads_example() -> None:
    sequence = default_registry().load(EXAMPLES / "omegamotiongpt.smplx.npz")

    assert sequence.format_id == "smplx_body22_fitted_aa"
    assert sequence.source == "ours_150m"
    assert sequence.frames == 100
    assert sequence.fps == 20.0
    assert sequence.joints is not None
    assert sequence.joints.shape == (100, 22, 3)
    assert sequence.body_model is not None
    assert sequence.body_model.body_pose.shape == (100, 63)
    assert RenderCapability.SMPLX_MESH in sequence.capabilities
    assert sequence.prefix_t == 20


def test_registry_probe_prefers_internal_format() -> None:
    matches = default_registry().probe(EXAMPLES / "gt.smplx.npz")

    assert matches
    assert matches[0].format_id == "smplx_body22_fitted_aa"
    assert matches[0].confidence == 1.0


def test_smplx_body22_loader_accepts_native_internal_format(tmp_path: Path) -> None:
    path = tmp_path / "native.smplx.npz"
    frames = 4
    np.savez(
        path,
        joints22=np.zeros((frames, 22, 3), dtype=np.float32),
        transl=np.zeros((frames, 3), dtype=np.float32),
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        body_pose=np.zeros((frames, 63), dtype=np.float32),
        betas=np.zeros((16,), dtype=np.float32),
        prefix_T=np.array(0),
        fps=np.array(30.0),
        source=np.array("native"),
        format=np.array("smplx_body22_native_aa"),
    )

    sequence = default_registry().load(path, format_id="smplx_body22_fitted_aa")

    assert sequence.format_id == "smplx_body22_fitted_aa"
    assert sequence.frames == frames
