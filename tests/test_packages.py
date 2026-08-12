from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from motionviewer.packages import (
    PackageFormatError,
    PackagePayloadError,
    PackageRenderRequest,
    PackageSelection,
    PackageSelectionError,
    PackageValidationError,
    build_render_job_from_package,
    import_package,
    inspect_package,
    is_package,
    validate_package,
)
from motionviewer.video.job import prepare_render_job


def _write_smplx(
    path: Path, *, frames: int = 8, fps: float = 30.0, source: str = "gt", prefix_t: int = 0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        joints22=np.zeros((frames, 22, 3), dtype=np.float32),
        transl=np.zeros((frames, 3), dtype=np.float32),
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        body_pose=np.zeros((frames, 63), dtype=np.float32),
        betas=np.zeros((10,), dtype=np.float32),
        prefix_T=np.array(prefix_t, dtype=np.int32),
        fps=np.array(fps, dtype=np.float32),
        source=np.array(source),
        format=np.array("smplx_body22_fitted_aa"),
        fit_mse=np.array(0.01, dtype=np.float32),
    )


def _write_joints(path: Path, *, frames: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((frames, 22, 3), dtype=np.float32))


def _write_clip_assets(clip_dir: Path, source_ids: tuple[str, ...], *, prefix_t: int = 0) -> None:
    for name in ("gt", *source_ids):
        (clip_dir / name).mkdir(parents=True, exist_ok=True)
        _write_smplx(clip_dir / name / "smplx_params.npz", source=name, prefix_t=prefix_t)
        _write_joints(clip_dir / name / "joints22.npy")


def _build_v2_package(root: Path, *, include_recon: bool = True, include_pred: bool = False) -> Path:
    package = root / "soma_tmr_test"
    package.mkdir(parents=True)

    t2m_clip_id = "test:abc123"
    t2m_dir = "test_abc123"
    t2m_clip = package / "clips" / "t2m" / t2m_dir
    _write_clip_assets(t2m_clip, ("omegamotiongpt", "kimodo"))

    tasks: dict = {
        "t2m": {
            "task": "t2m",
            "num_clips": 1,
            "provenances": ["mm_MotionGV"],
            "sources": [
                {
                    "source_id": "omegamotiongpt",
                    "kind": "model",
                    "role": "ours",
                    "native_rep": "motion_codes",
                },
                {"source_id": "kimodo", "kind": "model", "role": "baseline", "native_rep": "soma77"},
            ],
            "tokenizer": {"num_codebooks": 16, "codebook_size": 1024, "temporal_stride": 2},
            "clip_stats": {"min_T": 8, "max_T": 8, "median_T": 8},
        }
    }

    (t2m_clip / "meta.json").write_text(
        json.dumps(
            {
                "clip_id": t2m_clip_id,
                "rec_id": "abc123",
                "provenance": "mm_MotionGV",
                "task": "t2m",
                "caption": "A person walks forward, then turns left.",
                "T": 8,
                "fps": 30.0,
                "prefix_T": 0,
                "gt": {"joints22": "gt/joints22.npy", "smplx_params": "gt/smplx_params.npz"},
                "sources": {
                    "omegamotiongpt": {
                        "kind": "model",
                        "role": "ours",
                        "native_rep": "motion_codes",
                        "joints22": "omegamotiongpt/joints22.npy",
                        "smplx_params": "omegamotiongpt/smplx_params.npz",
                        "fit_mse": 0.023,
                    },
                    "kimodo": {
                        "kind": "model",
                        "role": "baseline",
                        "native_rep": "soma77",
                        "joints22": "kimodo/joints22.npy",
                        "smplx_params": "kimodo/smplx_params.npz",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    num_clips = 1

    if include_pred:
        pred_clip_id = "test:pred1"
        pred_dir = "test_pred1"
        pred_clip = package / "clips" / "pred" / pred_dir
        _write_clip_assets(pred_clip, ("omegamotiongpt", "kimodo"), prefix_t=4)
        tasks["pred"] = {
            "task": "pred",
            "num_clips": 1,
            "provenances": ["HumanML3D"],
            "sources": [
                {
                    "source_id": "omegamotiongpt",
                    "kind": "model",
                    "role": "ours",
                    "native_rep": "motion_codes",
                },
                {"source_id": "kimodo", "kind": "model", "role": "baseline", "native_rep": "soma77"},
            ],
        }
        (pred_clip / "meta.json").write_text(
            json.dumps(
                {
                    "clip_id": pred_clip_id,
                    "rec_id": "pred1",
                    "provenance": "HumanML3D",
                    "task": "pred",
                    "T": 8,
                    "fps": 30.0,
                    "prefix_T": 4,
                    "predicted_T": 4,
                    "gt": {"joints22": "gt/joints22.npy", "smplx_params": "gt/smplx_params.npz"},
                    "sources": {
                        "omegamotiongpt": {
                            "kind": "model",
                            "role": "ours",
                            "joints22": "omegamotiongpt/joints22.npy",
                            "smplx_params": "omegamotiongpt/smplx_params.npz",
                        },
                        "kimodo": {
                            "kind": "model",
                            "role": "baseline",
                            "joints22": "kimodo/joints22.npy",
                            "smplx_params": "kimodo/smplx_params.npz",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        num_clips += 1

    if include_recon:
        recon_clip_id = "test:recon1"
        recon_dir = "test_recon1"
        recon_clip = package / "clips" / "recon" / recon_dir
        _write_clip_assets(recon_clip, ("q04", "q10", "q16"))
        tasks["recon"] = {
            "task": "recon",
            "num_clips": 1,
            "provenances": ["HumanML3D"],
            "sources": [
                {"source_id": "q04", "kind": "reconstruction_level", "num_codebooks_used": 4},
                {"source_id": "q10", "kind": "reconstruction_level", "num_codebooks_used": 10},
                {"source_id": "q16", "kind": "reconstruction_level", "num_codebooks_used": 16},
            ],
            "reconstruction_levels": [4, 10, 16],
            "source_clip_id": recon_clip_id,
        }
        (recon_clip / "meta.json").write_text(
            json.dumps(
                {
                    "clip_id": recon_clip_id,
                    "rec_id": "recon1",
                    "provenance": "HumanML3D",
                    "task": "recon",
                    "T": 8,
                    "fps": 30.0,
                    "prefix_T": 0,
                    "gt": {"joints22": "gt/joints22.npy", "smplx_params": "gt/smplx_params.npz"},
                    "sources": {
                        "q04": {
                            "kind": "reconstruction_level",
                            "num_codebooks_used": 4,
                            "total_codebooks": 16,
                            "joints22": "q04/joints22.npy",
                            "smplx_params": "q04/smplx_params.npz",
                            "mse_vs_gt": 0.0085,
                        },
                        "q10": {
                            "kind": "reconstruction_level",
                            "num_codebooks_used": 10,
                            "total_codebooks": 16,
                            "joints22": "q10/joints22.npy",
                            "smplx_params": "q10/smplx_params.npz",
                            "mse_vs_gt": 0.0042,
                        },
                        "q16": {
                            "kind": "reconstruction_level",
                            "num_codebooks_used": 16,
                            "total_codebooks": 16,
                            "joints22": "q16/joints22.npy",
                            "smplx_params": "q16/smplx_params.npz",
                            "mse_vs_gt": 0.0011,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        num_clips += 1

    manifest = {
        "protocol_version": "2.0",
        "track": "soma_tmr",
        "split": "test",
        "fps": 30.0,
        "num_clips": num_clips,
        "tasks": tasks,
        "clip_stats": {"min_T": 8, "max_T": 8, "median_T": 8},
        "created_at": "2026-07-26T12:00:00Z",
        "generator": "test/2.0",
        "notes": [],
    }
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package


def _tar_directory(src: Path, dest: Path) -> Path:
    with tarfile.open(dest, "w:gz") as archive:
        archive.add(src, arcname=src.name)
    return dest


def test_directory_and_tar_inspection_parity(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path)
    tar_path = _tar_directory(package, tmp_path / "bundle.tar.gz")

    assert is_package(package)
    assert is_package(tar_path)

    dir_inspection = inspect_package(package)
    tar_inspection = inspect_package(tar_path)

    assert dir_inspection.protocol_version == "2.0"
    assert set(dir_inspection.tasks) == {"t2m", "recon"}
    assert [clip.clip_id for clip in dir_inspection.clips] == [clip.clip_id for clip in tar_inspection.clips]
    assert not dir_inspection.diagnostics


def test_default_t2m_render_job_is_loadable(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    job = build_render_job_from_package(
        PackageRenderRequest(path=package, cache_dir=tmp_path / "cache", validation="loadable")
    )
    assert job.task.mode == "text_to_motion"
    assert job.task.instruction and "walks forward" in job.task.instruction
    assert [item.label for item in job.inputs] == ["omegamotiongpt"]
    assert job.timeline.show_prefix is False
    prepared = prepare_render_job(job)
    assert prepared.frames == 8
    assert prepared.fps == 30.0


def test_multi_task_requires_task_selection(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path)
    with pytest.raises(PackageSelectionError, match="multiple tasks"):
        build_render_job_from_package(
            PackageRenderRequest(path=package, cache_dir=tmp_path / "cache", validation="assets")
        )


def test_clip_resolve_by_meta_id_and_dir_name(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    for clip_query in ("test:abc123", "test_abc123"):
        job = build_render_job_from_package(
            PackageRenderRequest(
                path=package,
                selection=PackageSelection(task="t2m", clip_id=clip_query, sources=("omegamotiongpt",)),
                cache_dir=tmp_path / "cache" / clip_query,
                validation="assets",
            )
        )
        assert job.task.mode == "text_to_motion"
        assert "omegamotiongpt" in job.output.directory.name


def test_pred_maps_to_continuation(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False, include_pred=True)
    job = build_render_job_from_package(
        PackageRenderRequest(
            path=package,
            selection=PackageSelection(task="pred"),
            cache_dir=tmp_path / "cache",
            validation="assets",
        )
    )
    assert job.task.mode == "continuation"
    assert job.timeline.show_prefix is True


def test_recon_orders_levels_and_labels(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=True)
    job = build_render_job_from_package(
        PackageRenderRequest(
            path=package,
            selection=PackageSelection(task="recon"),
            cache_dir=tmp_path / "cache",
            validation="assets",
        )
    )
    assert job.task.mode == "comparison"
    assert [item.label for item in job.inputs] == ["q04 (4 cb, MSE=0.0085)"]
    assert job.style.ghost_snapshots == 0
    assert job.style.ghost.mode == "none"
    assert job.style.prefix.ghost_count == 0
    assert job.output.directory.name.endswith("_q04")
    assert job.output.mp4_name == "q04.mp4"


def test_single_source_output_paths_include_source_id(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    job = build_render_job_from_package(
        PackageRenderRequest(
            path=package,
            selection=PackageSelection(sources=("omegamotiongpt",)),
            cache_dir=tmp_path / "cache",
            validation="assets",
        )
    )
    assert job.output.directory.name.endswith("_omegamotiongpt")
    assert job.output.mp4_name == "omegamotiongpt.mp4"


def test_selection_unknown_source_raises(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    with pytest.raises(PackageSelectionError):
        build_render_job_from_package(
            PackageRenderRequest(
                path=package,
                selection=PackageSelection(sources=("missing_model",)),
                cache_dir=tmp_path / "cache",
            )
        )


def test_joints22_asset_preference(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    job = build_render_job_from_package(
        PackageRenderRequest(
            path=package,
            selection=PackageSelection(
                asset_preference="joints22", sources=("omegamotiongpt",), include_gt=False
            ),
            cache_dir=tmp_path / "cache",
            validation="loadable",
        )
    )
    assert len(job.inputs) == 1
    assert all(item.format == "joints_npy" for item in job.inputs)
    prepare_render_job(job)


def test_validate_reports_missing_asset(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    clip_dir = package / "clips" / "t2m" / "test_abc123"
    (clip_dir / "kimodo" / "smplx_params.npz").unlink()
    diagnostics = validate_package(package, mode="assets")
    assert any(item.code == "missing_asset_file" for item in diagnostics)
    with pytest.raises(PackageValidationError):
        build_render_job_from_package(
            PackageRenderRequest(path=package, cache_dir=tmp_path / "cache", validation="assets")
        )


def test_clip_may_omit_manifest_source(tmp_path: Path) -> None:
    """Manifest sources are a union; individual clips may lack some models."""
    package = _build_v2_package(tmp_path, include_recon=False)
    clip_dir = package / "clips" / "t2m" / "test_abc123"
    meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
    del meta["sources"]["omegamotiongpt"]
    (clip_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # Keep files on disk but remove from meta — index only sees meta sources.
    for path in (clip_dir / "omegamotiongpt").iterdir():
        path.unlink()
    (clip_dir / "omegamotiongpt").rmdir()

    # Still valid at package level because kimodo covers the other manifest source,
    # and omegamotiongpt is gone from every clip → unused_manifest_source error.
    # Add a second clip that still has omegamotiongpt so the union rule holds.
    second = package / "clips" / "t2m" / "test_other"
    _write_clip_assets(second, ("omegamotiongpt", "kimodo"))
    (second / "meta.json").write_text(
        json.dumps(
            {
                "clip_id": "test:other",
                "rec_id": "other",
                "provenance": "HumanML3D",
                "task": "t2m",
                "caption": "Another clip.",
                "T": 8,
                "fps": 30.0,
                "prefix_T": 0,
                "gt": {"joints22": "gt/joints22.npy", "smplx_params": "gt/smplx_params.npz"},
                "sources": {
                    "omegamotiongpt": {
                        "kind": "model",
                        "role": "ours",
                        "joints22": "omegamotiongpt/joints22.npy",
                        "smplx_params": "omegamotiongpt/smplx_params.npz",
                    },
                    "kimodo": {
                        "kind": "model",
                        "role": "baseline",
                        "joints22": "kimodo/joints22.npy",
                        "smplx_params": "kimodo/smplx_params.npz",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["num_clips"] = 2
    manifest["tasks"]["t2m"]["num_clips"] = 2
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    diagnostics = validate_package(package, mode="structural")
    assert not any(item.code == "missing_clip_source" for item in diagnostics)
    assert not any(item.code == "unused_manifest_source" for item in diagnostics)
    hard = [item for item in diagnostics if not item.code.startswith("warning")]
    assert hard == []

    job = build_render_job_from_package(
        PackageRenderRequest(
            path=package,
            selection=PackageSelection(clip_id="test:abc123", sources=("kimodo",)),
            cache_dir=tmp_path / "cache",
            validation="assets",
        )
    )
    assert [item.label for item in job.inputs] == ["kimodo"]


def test_extra_clip_source_rejected(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    clip_dir = package / "clips" / "t2m" / "test_abc123"
    _write_clip_assets(clip_dir, ("rogue",))
    meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
    meta["sources"]["rogue"] = {
        "kind": "model",
        "joints22": "rogue/joints22.npy",
        "smplx_params": "rogue/smplx_params.npz",
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    diagnostics = validate_package(package, mode="structural")
    assert any(item.code == "extra_clip_source" and item.source_id == "rogue" for item in diagnostics)
    with pytest.raises(PackageValidationError, match="rogue"):
        build_render_job_from_package(
            PackageRenderRequest(path=package, cache_dir=tmp_path / "cache", validation="structural")
        )


def test_unused_manifest_source_rejected(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    clip_dir = package / "clips" / "t2m" / "test_abc123"
    meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
    del meta["sources"]["omegamotiongpt"]
    (clip_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for path in (clip_dir / "omegamotiongpt").iterdir():
        path.unlink()
    (clip_dir / "omegamotiongpt").rmdir()

    diagnostics = validate_package(package, mode="structural")
    assert any(
        item.code == "unused_manifest_source" and item.source_id == "omegamotiongpt" for item in diagnostics
    )


def test_import_package_copies_tar_as_is(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    tar_path = _tar_directory(package, tmp_path / "incoming.tar.gz")
    dest_root = tmp_path / "packages"
    imported = import_package(tar_path, dest_root=dest_root)
    assert imported == (dest_root / "soma_tmr_test.tar.gz").resolve()
    assert imported.is_file()
    assert is_package(imported)

    with pytest.raises(PackagePayloadError, match="already exists"):
        import_package(tar_path, dest_root=dest_root)

    imported_again = import_package(tar_path, dest_root=dest_root, force=True)
    assert imported_again == imported


def test_import_package_rejects_v1(tmp_path: Path) -> None:
    package = tmp_path / "old"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "1.0",
                "task": "t2m",
                "track": "soma_tmr",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PackageFormatError, match="protocol_version"):
        import_package(package, dest_root=tmp_path / "packages")


def test_tar_path_traversal_rejected(tmp_path: Path) -> None:
    package = _build_v2_package(tmp_path, include_recon=False)
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"x")
    with tarfile.open(evil, "w:gz") as archive:
        archive.add(package / "manifest.json", arcname="bundle/manifest.json")
        archive.add(package / "clips", arcname="bundle/clips")
        archive.add(payload, arcname="bundle/clips/t2m/test_abc123/gt/../../../evil.bin")

    from pathlib import PurePosixPath

    from motionviewer.packages.store import TarPackageStore

    store = TarPackageStore(evil)
    try:
        with pytest.raises(PackagePayloadError):
            store.materialize(
                [(PurePosixPath("clips/t2m/test_abc123/gt/../../../evil.bin"), tmp_path / "out.bin")]
            )
    finally:
        store.close()
