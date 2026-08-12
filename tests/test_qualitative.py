from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motionviewer.blender.qualitative import _dot_vector3, snapshot_layout_translation
from motionviewer.core.smplx_actor import BODY_POSE_BONES, override_foot_pose
from motionviewer.packages.index import build_package_index
from motionviewer.packages.store import open_package_store
from motionviewer.video.qualitative import (
    QualitativeBatchRequest,
    _qualitative_fbx_models,
    common_source_candidates,
    normalized_frame_indices,
    parse_provenance_counts,
    prepare_qualitative_batch,
    shuffled_candidates,
)

SOURCES = ("omegamotiongpt", "kimodo", "hymotion")


def _write_motion(path: Path, frames: int, *, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joints22": np.zeros((frames, 22, 3), dtype=np.float32),
        "transl": np.zeros((frames, 3), dtype=np.float32),
        "global_orient": np.zeros((frames, 3), dtype=np.float32),
        "body_pose": np.zeros((frames, 63), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
    }
    if not valid:
        payload.pop("body_pose")
    np.savez(path, **payload)


def _write_package(root: Path) -> Path:
    package = root / "package"
    provenances = ("HumanML3D", "bones-seed", "mm_MotionGV")
    clip_total = 0
    for provenance in provenances:
        for index in range(4):
            clip_total += 1
            dir_name = f"{provenance}_{index}"
            clip_root = package / "clips" / "t2m" / dir_name
            sources = {}
            for source_index, source_id in enumerate(SOURCES):
                frames = 9 + index + source_index
                _write_motion(
                    clip_root / source_id / "smplx_params.npz",
                    frames,
                    valid=not (provenance == "HumanML3D" and index == 0 and source_id == "kimodo"),
                )
                sources[source_id] = {
                    "kind": "model",
                    "smplx_params": f"{source_id}/smplx_params.npz",
                    "T": frames,
                }
            _write_motion(clip_root / "gt" / "smplx_params.npz", 12)
            (clip_root / "meta.json").write_text(
                json.dumps(
                    {
                        "clip_id": f"test:{dir_name}",
                        "rec_id": dir_name,
                        "task": "t2m",
                        "provenance": provenance,
                        "caption": f"motion {dir_name}",
                        "T": 12,
                        "fps": 30.0,
                        "prefix_T": 0,
                        "gt": {"smplx_params": "gt/smplx_params.npz"},
                        "sources": sources,
                    }
                ),
                encoding="utf-8",
            )
    package.mkdir(parents=True, exist_ok=True)
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "2.0",
                "track": "test",
                "split": "test",
                "fps": 30.0,
                "num_clips": clip_total,
                "tasks": {
                    "t2m": {
                        "task": "t2m",
                        "num_clips": clip_total,
                        "provenances": list(provenances),
                        "sources": [{"source_id": source_id, "kind": "model"} for source_id in SOURCES],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return package


def _write_fbx(root: Path, count: int = 2) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    assets = []
    for index in range(count):
        name = f"actor{index}"
        (root / f"{name}.fbx").write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00" + b"\x00" * 32)
        assets.append(
            {
                "model_id": name,
                "path": f"{name}.fbx",
                "profile_id": "mixamo",
                "status": "approved",
                "random_eligible": True,
            }
        )
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "mixamo": {
                        "profile_id": "mixamo",
                        "rig_family": "mixamo",
                        "bone_map": "auto",
                        "retarget_mode": "quality",
                    }
                },
                "assets": assets,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_normalized_frame_indices_include_both_endpoints() -> None:
    assert normalized_frame_indices(15, 8) == (0, 2, 4, 6, 8, 10, 12, 14)


def test_normalized_frame_indices_use_deterministic_midpoint_for_one_snapshot() -> None:
    assert normalized_frame_indices(15, 1) == (7,)


def test_parse_provenance_counts_preserves_requested_order() -> None:
    assert parse_provenance_counts(("HumanML3D=34", "bones-seed=33")) == (
        ("HumanML3D", 34),
        ("bones-seed", 33),
    )


def test_qualitative_request_accepts_one_source(tmp_path: Path) -> None:
    request = QualitativeBatchRequest(
        package=tmp_path / "package",
        output_dir=tmp_path / "output",
        sources=("omegamotiongpt",),
        provenance_counts=(("HumanML3D", 1),),
    )

    assert request.sources == ("omegamotiongpt",)


def test_qualitative_request_serializes_smplx_body_mode(tmp_path: Path) -> None:
    request = QualitativeBatchRequest(
        package=tmp_path / "package",
        output_dir=tmp_path / "output",
        sources=("omegamotiongpt",),
        provenance_counts=(("HumanML3D", 1),),
        body_mode="smplx",
    )

    assert request.to_json()["body_mode"] == "smplx"


def test_qualitative_request_accepts_smplh_body_mode(tmp_path: Path) -> None:
    request = QualitativeBatchRequest(
        package=tmp_path / "package",
        output_dir=tmp_path / "output",
        sources=("omegamotiongpt",),
        provenance_counts=(("HumanML3D", 1),),
        body_mode="smplh",
    )

    assert request.to_json()["body_mode"] == "smplh"


def test_qualitative_request_serializes_custom_palette(tmp_path: Path) -> None:
    request = QualitativeBatchRequest(
        package=tmp_path / "package",
        output_dir=tmp_path / "output",
        sources=("omegamotiongpt",),
        provenance_counts=(("HumanML3D", 1),),
        material_mode="palette",
        palette_start_rgb=(222, 235, 247),
        palette_end_rgb=(46, 117, 182),
        snapshot_alpha=0.88,
    )

    assert request.to_json()["palette_start_rgb"] == [222, 235, 247]
    assert request.to_json()["palette_end_rgb"] == [46, 117, 182]
    assert request.to_json()["snapshot_alpha"] == 0.88


@pytest.mark.parametrize("snapshot_alpha", (0.0, -0.1, 1.01))
def test_qualitative_request_rejects_invalid_snapshot_alpha(tmp_path: Path, snapshot_alpha: float) -> None:
    with pytest.raises(ValueError, match="snapshot_alpha"):
        QualitativeBatchRequest(
            package=tmp_path / "package",
            output_dir=tmp_path / "output",
            sources=("omegamotiongpt",),
            provenance_counts=(("HumanML3D", 1),),
            snapshot_alpha=snapshot_alpha,
        )


def test_foot_pose_overrides_are_local_and_do_not_modify_source() -> None:
    source = np.ones((2, 21, 3), dtype=np.float32)
    ankle_only = override_foot_pose(source, "ankle_neutral")
    neutral_feet = override_foot_pose(source, "neutral_feet")
    ankle_indices = [BODY_POSE_BONES.index(name) for name in ("left_ankle", "right_ankle")]
    toe_indices = [BODY_POSE_BONES.index(name) for name in ("left_foot", "right_foot")]

    assert np.all(source == 1.0)
    assert np.all(ankle_only[:, ankle_indices] == 0.0)
    assert np.all(ankle_only[:, toe_indices] == 1.0)
    assert np.all(neutral_feet[:, ankle_indices + toe_indices] == 0.0)


def test_root_aligned_snapshot_layout_removes_only_horizontal_root_motion() -> None:
    first = snapshot_layout_translation(
        np.asarray((3.0, -4.0, 1.5)),
        snapshot_index=0,
        snapshot_count=3,
        layout="root_aligned",
        spacing=1.0,
    )
    last = snapshot_layout_translation(
        np.asarray((-2.0, 7.0, 0.25)),
        snapshot_index=2,
        snapshot_count=3,
        layout="root_aligned",
        spacing=1.0,
    )

    assert first[2] == 0.0
    assert last[2] == 0.0
    assert np.allclose(first[:2] + np.asarray((3.0, -4.0)), (-0.81923192, -0.57346234))
    assert np.allclose(last[:2] + np.asarray((-2.0, 7.0)), (0.81923192, 0.57346234))


def test_trajectory_snapshot_layout_does_not_move_snapshots() -> None:
    shift = snapshot_layout_translation(
        np.asarray((3.0, -4.0, 1.5)),
        snapshot_index=1,
        snapshot_count=3,
        layout="trajectory",
        spacing=1.0,
    )
    assert np.array_equal(shift, np.zeros(3))


def test_arc_snapshot_layout_keeps_snapshots_grounded_and_curved() -> None:
    shifts = [
        snapshot_layout_translation(
            np.zeros(3),
            snapshot_index=index,
            snapshot_count=6,
            layout="arc",
            spacing=1.25,
        )
        for index in range(6)
    ]

    assert any(abs(shift[2]) > 0.0 for shift in shifts)
    assert np.linalg.norm(shifts[0][:2] - shifts[-1][:2]) > 1.25
    assert np.linalg.norm(shifts[2][:2]) > 0.0
    assert np.linalg.norm(shifts[0] - shifts[1]) == pytest.approx(1.25)


def test_arc_direction_flips_the_bend() -> None:
    up = snapshot_layout_translation(
        np.zeros(3), snapshot_index=2, snapshot_count=6, layout="arc", spacing=1.25, arc_direction="up"
    )
    down = snapshot_layout_translation(
        np.zeros(3), snapshot_index=2, snapshot_count=6, layout="arc", spacing=1.25, arc_direction="down"
    )
    assert np.linalg.norm(up - down) > 0.1
    assert np.allclose(up[2], -down[2])


def test_projected_camera_dot_ignores_evaluated_vertex_w_component() -> None:
    point = SimpleNamespace(x=1.0, y=2.0, z=3.0, w=1.0)
    axis = SimpleNamespace(x=0.0, y=1.0, z=0.0, w=0.0)
    assert _dot_vector3(point, axis) == 2.0


def test_all_binary_fbx_pool_exposes_pending_binary_assets(tmp_path: Path) -> None:
    fbx_root = _write_fbx(tmp_path / "fbx")
    catalog_path = fbx_root / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["assets"][0]["status"] = "pending"
    catalog["assets"][0]["random_eligible"] = False
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    approved_request = QualitativeBatchRequest(
        package=tmp_path / "package",
        output_dir=tmp_path / "approved",
        provenance_counts=(("HumanML3D", 1),),
        fbx_root=fbx_root,
        fbx_pool="approved",
    )
    exploratory_request = QualitativeBatchRequest(
        package=tmp_path / "package",
        output_dir=tmp_path / "all_binary",
        provenance_counts=(("HumanML3D", 1),),
        fbx_root=fbx_root,
        fbx_pool="all_binary",
    )
    assert [model.model_id for model in _qualitative_fbx_models(approved_request)] == ["actor1"]
    assert [model.model_id for model in _qualitative_fbx_models(exploratory_request)] == ["actor0", "actor1"]

    excluded_request = QualitativeBatchRequest(
        package=tmp_path / "package",
        output_dir=tmp_path / "excluded",
        provenance_counts=(("HumanML3D", 1),),
        fbx_root=fbx_root,
        fbx_pool="all_binary",
        exclude_fbx=("actor0",),
    )
    assert [model.model_id for model in _qualitative_fbx_models(excluded_request)] == ["actor1"]


def test_common_source_selection_and_shuffle_are_deterministic(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    store = open_package_store(package)
    try:
        index = build_package_index(store)
    finally:
        store.close()
    candidates = common_source_candidates(
        index,
        task="t2m",
        sources=SOURCES,
        provenance="HumanML3D",
    )
    first = [clip.clip_id for clip in shuffled_candidates(candidates, seed=20260726, provenance="HumanML3D")]
    second = [clip.clip_id for clip in shuffled_candidates(candidates, seed=20260726, provenance="HumanML3D")]
    assert len(first) == 4
    assert first == second


def test_common_source_selection_supports_ground_truth(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    store = open_package_store(package)
    try:
        index = build_package_index(store)
    finally:
        store.close()

    candidates = common_source_candidates(
        index,
        task="t2m",
        sources=("gt",),
        provenance="HumanML3D",
    )

    assert len(candidates) == 4


def test_prepare_batch_meets_quotas_backfills_invalid_and_groups_three_images(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    fbx_root = _write_fbx(tmp_path / "fbx")
    output = tmp_path / "out"
    request = QualitativeBatchRequest(
        package=package,
        output_dir=output,
        sources=SOURCES,
        provenance_counts=(("HumanML3D", 2), ("bones-seed", 2), ("mm_MotionGV", 2)),
        fbx_root=fbx_root,
        snapshots=8,
        resolution=(320, 320),
        samples=4,
    )

    first = prepare_qualitative_batch(request)
    second = prepare_qualitative_batch(request)

    assert len(first.jobs) == 6
    assert [job.clip_id for job in first.jobs] == [job.clip_id for job in second.jobs]
    assert sum(job.provenance == "HumanML3D" for job in first.jobs) == 2
    assert sum(job.provenance == "bones-seed" for job in first.jobs) == 2
    assert sum(job.provenance == "mm_MotionGV" for job in first.jobs) == 2
    assert all(len(job.sources) == 3 for job in first.jobs)
    assert all(source.output_path.parent == job.output_dir for job in first.jobs for source in job.sources)
    assert all(
        source.output_path.name.startswith(f"{source.source_id}__motion_")
        and source.output_path.suffix == ".png"
        for job in first.jobs
        for source in job.sources
    )
    assert all(len(source.frame_indices) == 8 for job in first.jobs for source in job.sources)
    assert any(item["clip_id"] == "test:HumanML3D_0" for item in first.rejections)
    assert {job.fbx_model_id for job in first.jobs} == {"actor0", "actor1"}
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert selection["checkpoint_aliases"]["omegamotiongpt"].startswith("t2m_400m_flan")
    assert selection["request"]["snapshot_layout"] == "root_aligned"
    assert selection["request"]["snapshot_spacing"] == 1.25
    assert selection["request"]["material_mode"] == "preserve"
    assert selection["request"]["body_mode"] == "fbx"
    assert selection["request"]["foot_pose"] == "source"
    assert selection["request"]["fbx_pool"] == "approved"
    bundle = json.loads(first.jobs[0].bundle_path.read_text(encoding="utf-8"))
    assert bundle["snapshot_layout"] == "root_aligned"
    assert bundle["material_mode"] == "preserve"
    assert bundle["body_mode"] == "fbx"
    assert bundle["foot_pose"] == "source"


def test_prepare_batch_can_select_an_exact_clip_id(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    fbx_root = _write_fbx(tmp_path / "fbx")
    request = QualitativeBatchRequest(
        package=package,
        output_dir=tmp_path / "out",
        sources=SOURCES,
        clip_ids=("test:bones-seed_2",),
        fbx_root=fbx_root,
    )

    batch = prepare_qualitative_batch(request)

    assert [job.clip_id for job in batch.jobs] == ["test:bones-seed_2"]
    assert batch.request.clip_count == 1
    selection = json.loads(batch.selection_path.read_text(encoding="utf-8"))
    assert selection["request"]["clip_ids"] == ["test:bones-seed_2"]
