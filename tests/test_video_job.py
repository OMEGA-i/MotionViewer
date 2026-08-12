from pathlib import Path

import numpy as np
import pytest

from motionviewer.video.job import load_render_job, prepare_render_job
from motionviewer.video.spec import BodySpec, InputSpec, RenderJob, TimelineSpec

ROOT = Path(__file__).resolve().parents[1]


def _write_smplx_npz(path: Path, *, frames: int, fps: float = 20.0, source: str = "sample") -> None:
    np.savez(
        path,
        joints22=np.zeros((frames, 22, 3), dtype=np.float32),
        transl=np.zeros((frames, 3), dtype=np.float32),
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        body_pose=np.zeros((frames, 63), dtype=np.float32),
        betas=np.zeros((16,), dtype=np.float32),
        prefix_T=np.array(0),
        fps=np.array(fps),
        source=np.array(source),
        format=np.array("smplx_body22_fitted_aa"),
    )


def test_example_render_job_prepares_bundle() -> None:
    job = load_render_job(ROOT / "configs" / "examples" / "multiview_single_actor.yaml")
    prepared = prepare_render_job(job)
    bundle = prepared.to_bundle()

    assert prepared.frames == 100
    assert prepared.fps == 20.0
    assert len(prepared.inputs) == 1
    assert len(job.camera.views) == 3
    assert bundle["inputs"][0]["sequence"]["format_id"] == "smplx_body22_fitted_aa"
    assert bundle["job"]["body"]["backend"] == "blender_smplx_addon"
    assert "scene_bounds" in bundle


def test_text_to_motion_dummy_job_has_instruction_and_no_prefix_mode() -> None:
    job = load_render_job(ROOT / "configs" / "examples" / "text_to_motion_dummy.yaml")
    prepared = prepare_render_job(job)
    bundle = prepared.to_bundle()

    assert bundle["job"]["task"]["mode"] == "text_to_motion"
    assert "walks forward" in bundle["job"]["task"]["instruction"]
    assert bundle["job"]["timeline"]["show_prefix"] is False
    assert len(prepared.inputs) == 1


def test_fbx_body_resolves_relative_bone_map(tmp_path: Path) -> None:
    config = tmp_path / "jobs" / "fbx.yaml"
    config.parent.mkdir()
    config.write_text(
        """
inputs:
  - path: ../motion/sample.smplx.npz
    body:
      backend: fbx_skeleton
      fbx_path: ../assets/character.fbx
      bone_map: mixamo
output:
  directory: ../out
""",
        encoding="utf-8",
    )

    job = load_render_job(config)

    assert job.inputs[0].body is not None
    assert job.inputs[0].body.fbx_path == str((config.parent / "../assets/character.fbx").resolve())
    assert job.inputs[0].body.bone_map == "mixamo"


def test_prepare_validates_per_input_fbx_body(tmp_path: Path) -> None:
    motion = tmp_path / "motion.smplx.npz"
    ascii_fbx = tmp_path / "ascii.fbx"
    _write_smplx_npz(motion, frames=5)
    ascii_fbx.write_text("; FBX ascii", encoding="utf-8")

    job = RenderJob(
        inputs=[
            InputSpec(
                path=motion,
                body=BodySpec(backend="fbx_skeleton", fbx_path=str(ascii_fbx)),
            )
        ]
    )

    with pytest.raises(ValueError, match="binary FBX"):
        prepare_render_job(job)


def test_prepare_accepts_binary_per_input_fbx_body(tmp_path: Path) -> None:
    motion = tmp_path / "motion.smplx.npz"
    binary_fbx = tmp_path / "binary.fbx"
    _write_smplx_npz(motion, frames=5)
    binary_fbx.write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00" + b"\x00" * 32)

    job = RenderJob(
        inputs=[
            InputSpec(
                path=motion,
                body=BodySpec(backend="fbx_skeleton", fbx_path=str(binary_fbx)),
            )
        ]
    )

    assert prepare_render_job(job).frames == 5


def test_mismatched_input_lengths_default_to_longest_frame_count(tmp_path: Path) -> None:
    from motionviewer.video.spec import LayoutSpec

    long_path = tmp_path / "long.smplx.npz"
    short_path = tmp_path / "short.smplx.npz"
    _write_smplx_npz(long_path, frames=76, source="gt")
    _write_smplx_npz(short_path, frames=46, source="hymotion")

    job = RenderJob(
        inputs=[InputSpec(path=long_path), InputSpec(path=short_path)],
        layout=LayoutSpec(mode="overlay"),
    )
    prepared = prepare_render_job(job)

    assert prepared.frames == 76


def test_frames_mode_min_trims_to_shortest_input(tmp_path: Path) -> None:
    from motionviewer.video.spec import LayoutSpec

    long_path = tmp_path / "long.smplx.npz"
    short_path = tmp_path / "short.smplx.npz"
    _write_smplx_npz(long_path, frames=76, source="gt")
    _write_smplx_npz(short_path, frames=46, source="hymotion")

    job = RenderJob(
        inputs=[InputSpec(path=long_path), InputSpec(path=short_path)],
        timeline=TimelineSpec(frames_mode="min"),
        layout=LayoutSpec(mode="overlay"),
    )
    prepared = prepare_render_job(job)

    assert prepared.frames == 46


def test_single_layout_rejects_multiple_inputs(tmp_path: Path) -> None:
    a = tmp_path / "a.smplx.npz"
    b = tmp_path / "b.smplx.npz"
    _write_smplx_npz(a, frames=10, source="a")
    _write_smplx_npz(b, frames=10, source="b")
    job = RenderJob(inputs=[InputSpec(path=a), InputSpec(path=b)])
    errors = job.validate()
    assert any("one input" in err for err in errors)


def test_removed_side_by_side_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="side_by_side"):
        RenderJob.from_dict(
            {
                "inputs": [{"path": "x.smplx.npz"}],
                "layout": {"mode": "side_by_side"},
            }
        )
