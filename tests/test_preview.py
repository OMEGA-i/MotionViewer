from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motionviewer.video.job import load_render_job, prepare_render_job
from motionviewer.video.preview import generate_package_preview


def _write_smplx(path: Path, *, frames: int, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        joints22=np.zeros((frames, 22, 3), dtype=np.float32),
        transl=np.zeros((frames, 3), dtype=np.float32),
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        body_pose=np.zeros((frames, 63), dtype=np.float32),
        betas=np.zeros((16,), dtype=np.float32),
        prefix_T=np.array(0, dtype=np.int32),
        fps=np.array(30.0, dtype=np.float32),
        source=np.array(source),
        format=np.array("smplx_body22_fitted_aa"),
    )


def _write_binary_fbx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00" + b"\x00" * 32)


def _write_fbx_catalog(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "mixamo": {
                        "profile_id": "mixamo",
                        "rig_family": "mixamo",
                        "bone_map": "auto",
                    }
                },
                "assets": [
                    {
                        "model_id": "iron",
                        "path": "iron.fbx",
                        "profile_id": "mixamo",
                        "status": "approved",
                        "random_eligible": True,
                        "evidence": {"smoke_mp4": "smoke.mp4"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _build_package(root: Path) -> Path:
    package = root / "soma_tmr_test"
    clip_dir = package / "clips" / "t2m" / "test_clip"
    _write_smplx(clip_dir / "gt" / "smplx_params.npz", frames=12, source="gt")
    _write_smplx(clip_dir / "omegamotiongpt" / "smplx_params.npz", frames=12, source="omegamotiongpt")
    np.save(clip_dir / "gt" / "joints22.npy", np.zeros((12, 22, 3), dtype=np.float32))
    np.save(clip_dir / "omegamotiongpt" / "joints22.npy", np.zeros((12, 22, 3), dtype=np.float32))
    (package / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "2.0",
                "track": "soma_tmr",
                "split": "test",
                "fps": 30.0,
                "num_clips": 1,
                "tasks": {
                    "t2m": {
                        "task": "t2m",
                        "num_clips": 1,
                        "provenances": ["HumanML3D"],
                        "sources": [{"source_id": "omegamotiongpt", "kind": "model"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (clip_dir / "meta.json").write_text(
        json.dumps(
            {
                "clip_id": "test:clip",
                "rec_id": "clip",
                "provenance": "HumanML3D",
                "task": "t2m",
                "caption": "A person walks forward.",
                "T": 12,
                "fps": 30.0,
                "prefix_T": 0,
                "gt": {"joints22": "gt/joints22.npy", "smplx_params": "gt/smplx_params.npz"},
                "sources": {
                    "omegamotiongpt": {
                        "kind": "model",
                        "joints22": "omegamotiongpt/joints22.npy",
                        "smplx_params": "omegamotiongpt/smplx_params.npz",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return package


def test_generate_package_preview_with_random_fbx(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    fbx_root = tmp_path / "fbx"
    _write_fbx_catalog(fbx_root)
    _write_binary_fbx(fbx_root / "iron.fbx")

    jobs = generate_package_preview(
        package,
        task="t2m",
        source_id="omegamotiongpt",
        clip_count=1,
        frame_count=5,
        output_dir=tmp_path / "preview",
        body_pool=("fbx-random",),
        fbx_root=fbx_root,
        seed=123,
    )

    assert len(jobs) == 1
    assert jobs[0].frame_indices == [0, 2, 5, 8, 11]
    assert jobs[0].body_backend == "fbx_skeleton"
    assert jobs[0].fbx_model_id == "iron"
    assert jobs[0].fbx_profile_id == "mixamo"
    assert jobs[0].fbx_status == "approved"
    assert jobs[0].fbx_evidence == {"smoke_mp4": "smoke.mp4"}
    assert jobs[0].sampling_mode == "keyframes"
    assert (tmp_path / "preview" / "selection.json").is_file()
    selection = json.loads((tmp_path / "preview" / "selection.json").read_text(encoding="utf-8"))
    assert selection[0]["sampling_mode"] == "keyframes"

    job = load_render_job(jobs[0].config_path)
    prepared = prepare_render_job(job)
    assert prepared.frames == 5
    assert job.inputs[0].body is not None
    assert job.inputs[0].body.backend == "fbx_skeleton"
    assert job.inputs[0].loader_options["sampling_mode"] == "keyframes"
