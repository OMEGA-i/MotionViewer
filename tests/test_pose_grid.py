"""Pure selection coverage for the FBX pose-grid renderer."""

from __future__ import annotations

import json
import random
import tarfile
from pathlib import Path

import numpy as np
import pytest

from motionviewer.blender.pose_grid import (
    PoseGridError,
    PoseGridReport,
    PoseGridRequest,
    choose_grid_fbx_models,
    load_single_frame_payload,
    parse_background_color,
    select_pose_grid_candidates,
    shuffled_candidates,
    write_pose_grid_report,
)
from motionviewer.packages.index import build_package_index
from motionviewer.packages.store import open_package_store


def _write_motion(path: Path, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        body_pose=np.zeros((frames, 63), dtype=np.float32),
        transl=np.zeros((frames, 3), dtype=np.float32),
        joints22=np.zeros((frames, 22, 3), dtype=np.float32),
        betas=np.zeros((10,), dtype=np.float32),
    )


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "package"
    clips = []
    for number, (caption, provenance, frames) in enumerate(
        (("walks forward", "HumanML3D", 8), ("raises an arm", "KIT", 12), ("walks backward", "HumanML3D", 16))
    ):
        name = f"clip_{number}"
        clip_root = root / "clips" / "t2m" / name
        _write_motion(clip_root / "gt" / "smplx_params.npz", frames)
        (clip_root / "meta.json").write_text(
            json.dumps(
                {
                    "clip_id": f"test:{name}",
                    "task": "t2m",
                    "provenance": provenance,
                    "caption": caption,
                    "T": frames,
                    "fps": 30.0,
                    "prefix_T": 0,
                    "gt": {"smplx_params": "gt/smplx_params.npz"},
                    "sources": {},
                }
            ),
            encoding="utf-8",
        )
        clips.append(name)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "2.0",
                "track": "test",
                "split": "test",
                "fps": 30.0,
                "num_clips": len(clips),
                "tasks": {
                    "t2m": {
                        "task": "t2m",
                        "num_clips": len(clips),
                        "provenances": ["HumanML3D", "KIT"],
                        "sources": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def _request(package: Path, **overrides) -> PoseGridRequest:
    return PoseGridRequest(package=package, fbx=Path("actor.fbx"), output=Path("grid.png"), **overrides)


def _write_binary_fbx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00" + b"\x00" * 32)


def _write_fbx_catalog(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "profiles": {"mixamo": {"profile_id": "mixamo", "rig_family": "mixamo", "bone_map": "auto"}},
                "assets": [
                    {
                        "model_id": "iron",
                        "path": "iron.fbx",
                        "profile_id": "mixamo",
                        "status": "approved",
                        "random_eligible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _index(package: Path):
    store = open_package_store(package)
    try:
        return build_package_index(store)
    finally:
        store.close()


def _tar_package(package: Path, destination: Path) -> Path:
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(package, arcname=package.name)
    return destination


def test_candidates_filter_metadata_and_keep_one_entry_per_clip(tmp_path: Path) -> None:
    package = _package(tmp_path)
    request = _request(
        package,
        task="t2m",
        provenances=("HumanML3D",),
        caption_regex="walk",
        min_frames=8,
        max_frames=12,
    )

    selected = select_pose_grid_candidates(_index(package), request)

    assert [item.clip.clip_id for item in selected] == ["test:clip_0"]
    assert len({item.clip.dir_name for item in selected}) == len(selected)


def test_shuffled_candidates_are_reproducible(tmp_path: Path) -> None:
    package = _package(tmp_path)
    request = _request(package, seed=7)
    candidates = select_pose_grid_candidates(_index(package), request)

    first = [item.clip.clip_id for item in shuffled_candidates(candidates, request)]
    second = [item.clip.clip_id for item in shuffled_candidates(candidates, request)]

    assert first == second
    assert set(first) == {"test:clip_0", "test:clip_1", "test:clip_2"}


def test_selection_supports_tar_packages(tmp_path: Path) -> None:
    package = _package(tmp_path)
    archive = _tar_package(package, tmp_path / "package.tar.gz")
    request = _request(archive, task="t2m", source_id="gt")

    selected = select_pose_grid_candidates(_index(archive), request)

    assert len(selected) == 3


def test_max_attempts_limits_the_shuffled_candidate_pool(tmp_path: Path) -> None:
    package = _package(tmp_path)
    request = _request(package, seed=7, max_attempts=2)

    selected = shuffled_candidates(select_pose_grid_candidates(_index(package), request), request)

    assert len(selected) == 2


def test_random_frame_is_reproducible_and_payload_is_single_frame(tmp_path: Path) -> None:
    package = _package(tmp_path)
    request = _request(package, frame_mode="random", seed=13)
    raw = (package / "clips" / "t2m" / "clip_2" / "gt" / "smplx_params.npz").read_bytes()

    first, first_index, first_frames = load_single_frame_payload(raw, request, random.Random(0))
    second, second_index, second_frames = load_single_frame_payload(raw, request, random.Random(0))

    assert first_frames == second_frames == 16
    assert first_index == second_index
    assert 0 <= first_index < first_frames
    assert first["global_orient"].shape == (1, 3)
    assert second["body_pose"].shape == (1, 63)


def test_fixed_frame_outside_payload_is_rejected(tmp_path: Path) -> None:
    package = _package(tmp_path)
    request = _request(package, frame_mode="index", frame_index=20)
    raw = (package / "clips" / "t2m" / "clip_0" / "gt" / "smplx_params.npz").read_bytes()

    with pytest.raises(PoseGridError, match="outside payload"):
        load_single_frame_payload(raw, request, random.Random(0))


def test_negative_fixed_frame_is_rejected_at_request_construction(tmp_path: Path) -> None:
    package = _package(tmp_path)

    with pytest.raises(ValueError, match="frame_index must be >= 0"):
        _request(package, frame_mode="index", frame_index=-1)


def test_no_matching_candidates_has_specific_error(tmp_path: Path) -> None:
    package = _package(tmp_path)

    with pytest.raises(PoseGridError, match="No package clips"):
        select_pose_grid_candidates(_index(package), _request(package, caption_regex="dance"))


def test_report_persists_request_and_status(tmp_path: Path) -> None:
    output = tmp_path / "output" / "grid.png"
    request = PoseGridRequest(package=tmp_path, fbx=Path("actor.fbx"), output=output)
    report = PoseGridReport(
        request=request, candidate_count=3, attempted_count=2, status="insufficient_samples"
    )

    path = write_pose_grid_report(report)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == output.with_suffix(".selection.json")
    assert payload["candidate_count"] == 3
    assert payload["attempted_count"] == 2
    assert payload["status"] == "insufficient_samples"


def test_pose_grid_defaults_match_paper_background_and_views(tmp_path: Path) -> None:
    request = PoseGridRequest(package=tmp_path, fbx=Path("actor.fbx"), output_dir=tmp_path / "out")

    assert request.background_rgb == (188, 170, 221)
    assert request.views == ("upper_left", "front", "upper_right")
    assert parse_background_color("#BCAADD") == (188, 170, 221)
    assert parse_background_color("188,170,221") == (188, 170, 221)


def test_pose_grid_accepts_multiple_background_colors(tmp_path: Path) -> None:
    colors = ((190, 224, 231), (232, 198, 226), (165, 200, 231), (188, 170, 221))
    request = PoseGridRequest(
        package=tmp_path,
        fbx=Path("actor.fbx"),
        output_dir=tmp_path / "out",
        background_rgbs=colors,
    )

    assert request.render_background_rgbs == colors
    assert request.to_json()["background_rgbs"] == [list(color) for color in colors]


def test_pose_grid_batch_rng_can_vary_candidate_order(tmp_path: Path) -> None:
    package = _package(tmp_path)
    request = _request(package)
    candidates = select_pose_grid_candidates(_index(package), request)

    first = shuffled_candidates(candidates, request, rng=random.Random(10))
    second = shuffled_candidates(candidates, request, rng=random.Random(11))

    assert [item.clip.clip_id for item in first] != [item.clip.clip_id for item in second]


def test_pose_grid_accepts_shallow_scatter_layout(tmp_path: Path) -> None:
    request = PoseGridRequest(
        package=tmp_path,
        fbx=Path("actor.fbx"),
        output_dir=tmp_path / "out",
        layout_style="scatter-shallow",
    )

    assert request.layout_style == "scatter-shallow"


def test_pose_grid_accepts_explicit_binary_fbx_pool(tmp_path: Path) -> None:
    request = PoseGridRequest(
        package=tmp_path,
        output_dir=tmp_path / "out",
        fbx_mode="random-grid",
        fbx_pool="binary",
    )

    assert request.fbx_pool == "binary"


def test_pose_grid_accepts_explicit_random_fbx_model_ids(tmp_path: Path) -> None:
    request = PoseGridRequest(
        package=tmp_path,
        output_dir=tmp_path / "out",
        fbx_mode="random-grid",
        fbx_pool="binary",
        fbx_model_ids=("erika_archer", "ch43_nonpbr"),
    )

    assert request.fbx_model_ids == ("erika_archer", "ch43_nonpbr")


def test_pose_grid_rejects_duplicate_random_fbx_model_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        PoseGridRequest(
            package=tmp_path,
            output_dir=tmp_path / "out",
            fbx_mode="random-grid",
            fbx_model_ids=("ch43_nonpbr", "ch43_nonpbr"),
        )


def test_random_fbx_modes_are_explicit_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "fbx"
    _write_fbx_catalog(root)
    _write_binary_fbx(root / "iron.fbx")
    _write_binary_fbx(root / "second.fbx")
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    catalog["assets"].append(
        {
            "model_id": "second",
            "path": "second.fbx",
            "profile_id": "mixamo",
            "status": "approved",
            "random_eligible": True,
        }
    )
    (root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    request = PoseGridRequest(
        package=tmp_path,
        output_dir=tmp_path / "out",
        fbx_mode="random-cell",
        fbx_root=root,
        count=4,
    )
    first = choose_grid_fbx_models(request, rng=random.Random(9))
    second = choose_grid_fbx_models(request, rng=random.Random(9))

    assert [model.model_id for model in first] == [model.model_id for model in second]
    assert len({model.model_id for model in first}) == 2
    with pytest.raises(ValueError, match="fbx must be omitted"):
        PoseGridRequest(
            package=tmp_path, fbx=Path("actor.fbx"), output_dir=tmp_path / "out", fbx_mode="random-grid"
        )
