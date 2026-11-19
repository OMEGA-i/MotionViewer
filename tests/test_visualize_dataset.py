"""Contract validation and lead-in trimming for the dataset visualiser.

The trimming rules were derived from measurements on the T2M exports, so the
numbers in these tests are the measured ones: a 14-frame frozen lead-in on every
``old500`` clip, and a ~300 mm single-frame jump out of the conditioning pose at
the head of every ``gen`` clip.  Getting the frame-0 case wrong is not cosmetic —
the retarget re-bases the whole root path on frame 0, so a bad first frame
mis-places the entire clip.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import visualize_dataset as visualizer  # noqa: E402
from visualize_dataset import (  # noqa: E402
    REQUIRED,
    Clip,
    _alias,
    _clip_id,
    caption_slug,
    detect_trim,
    materialize,
    parse_args,
    preprocess,
    render_clip,
    smooth_pose,
    validate,
    video_name,
    write_captions,
)


def _walk(frames: int = 60, *, step: float = 0.006) -> np.ndarray:
    """A smooth synthetic clip: every joint drifting at a constant rate."""
    base = np.linspace(0.0, 1.0, 22)[None, :, None] * np.ones((1, 1, 3))
    drift = (np.arange(frames, dtype=np.float64) * step)[:, None, None]
    return base + drift


def _write(path: Path, joints: np.ndarray, **overrides) -> Path:
    frames = len(joints)
    payload = {
        "global_orient": np.zeros((frames, 3), dtype=np.float32),
        "body_pose": np.zeros((frames, 63), dtype=np.float32),
        "transl": joints[:, 0].astype(np.float32),
        "joints22": joints.astype(np.float32),
        "fps": np.float32(30.0),
        "caption": np.str_("a person walks"),
    }
    payload.update(overrides)
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------


def test_alias_drops_the_smplx_suffix():
    assert _alias(Path("picks38_smplx/gt")) == "picks38/gt"
    assert _alias(Path("new500_smplx/gen")) == "new500/gen"


def test_alias_leaves_other_names_alone():
    assert _alias(Path("clips/t2m")) == "clips/t2m"


def test_clip_id_strips_the_double_extension():
    assert _clip_id(Path("000003.smplx.npz")) == "000003"
    assert _clip_id(Path("smplx_params.npz")) == "smplx_params"


# ---------------------------------------------------------------------------
# trimming
# ---------------------------------------------------------------------------


def test_smooth_clip_is_not_trimmed():
    assert detect_trim(_walk()) == (0, "")


def test_frozen_lead_in_is_measured_and_dropped():
    """Every old500 clip holds one pose for 14 frames before it moves."""
    joints = _walk(60)
    joints[:15] = joints[14]
    trim, reason = detect_trim(joints)
    assert trim == 14
    assert "14-frame frozen lead-in" in reason


def test_a_two_frame_hold_is_left_alone():
    """Short holds are how real motion starts; only a run of >2 counts."""
    joints = _walk(60)
    joints[:3] = joints[2]
    assert detect_trim(joints)[0] == 0


def test_anchor_pop_at_frame_zero_drops_exactly_one_frame():
    """gen clips start on the conditioning pose, then jump ~300 mm in one frame."""
    joints = _walk(60)
    joints[0, :, 0] += 0.30  # one axis, so the reported magnitude is predictable
    trim, reason = detect_trim(joints)
    assert trim == 1
    assert "anchor pop" in reason
    millimetres = float(re.search(r"\((\d+) mm", reason).group(1))
    assert millimetres == pytest.approx(294, abs=2)


def test_a_pop_later_in_the_clip_is_not_trimmed():
    """Only the head is trimmed: a mid-clip pop is the model's, and must stay visible."""
    joints = _walk(60)
    joints[30:] += 0.30
    assert detect_trim(joints) == (0, "")


def test_frozen_lead_in_wins_over_the_pop_test():
    """A clip that is frozen *and* pops must lose the whole frozen run, not one frame."""
    joints = _walk(60)
    joints[:15] = joints[14]
    joints[0] += 0.30  # inside the frozen run, so the run is what matters
    assert detect_trim(joints)[0] >= 1


def test_short_clips_are_never_trimmed():
    """Under six frames there is no reliable "later median" to compare against."""
    joints = _walk(4)
    joints[0] += 0.30
    assert detect_trim(joints) == (0, "")


def test_a_completely_static_clip_is_not_trimmed_to_nothing():
    joints = np.repeat(_walk(1), 40, axis=0)
    trim, _ = detect_trim(joints)
    assert trim < len(joints)


# ---------------------------------------------------------------------------
# the input contract
# ---------------------------------------------------------------------------


def test_a_well_formed_clip_validates(tmp_path):
    payload, error = validate(_write(tmp_path / "ok.npz", _walk()))
    assert error == ""
    assert payload is not None
    assert payload["body_pose"].shape == (60, 21, 3)
    assert payload["_fps"] == 30.0
    assert payload["_caption"] == "a person walks"


def test_body_pose_is_accepted_already_split(tmp_path):
    """(T, 21, 3) and (T, 63) are the same data; both must load."""
    path = _write(tmp_path / "split.npz", _walk(), body_pose=np.zeros((60, 21, 3), dtype=np.float32))
    payload, error = validate(path)
    assert error == ""
    assert payload["body_pose"].shape == (60, 21, 3)


@pytest.mark.parametrize("missing", REQUIRED)
def test_each_required_key_is_reported_by_name(tmp_path, missing):
    joints = _walk()
    full = {
        "global_orient": np.zeros((60, 3), dtype=np.float32),
        "body_pose": np.zeros((60, 63), dtype=np.float32),
        "transl": np.zeros((60, 3), dtype=np.float32),
        "joints22": joints.astype(np.float32),
    }
    del full[missing]
    path = tmp_path / f"no_{missing}.npz"
    np.savez(path, **full)
    payload, error = validate(path)
    assert payload is None
    assert missing in error


def test_wrong_joint_count_is_rejected(tmp_path):
    """A 77-joint export is a real thing that arrived; it must fail loudly."""
    path = _write(tmp_path / "j77.npz", _walk(), joints22=np.zeros((60, 77, 3), dtype=np.float32))
    payload, error = validate(path)
    assert payload is None
    assert "joints22" in error and "77" in error


def test_frame_count_mismatch_is_rejected(tmp_path):
    path = _write(tmp_path / "short.npz", _walk(), transl=np.zeros((59, 3), dtype=np.float32))
    payload, error = validate(path)
    assert payload is None
    assert "transl" in error


def test_non_finite_values_are_rejected(tmp_path):
    joints = _walk()
    joints[5, 3, 1] = np.nan
    payload, error = validate(_write(tmp_path / "nan.npz", joints))
    assert payload is None
    assert "non-finite" in error and "joints22" in error


def test_implausible_fps_is_rejected(tmp_path):
    path = _write(tmp_path / "fps.npz", _walk(), fps=np.float32(0.0))
    payload, error = validate(path)
    assert payload is None
    assert "fps" in error


def test_missing_fps_defaults_to_thirty(tmp_path):
    """The older exports carried no fps; 30 is the documented assumption."""
    frames = 60
    joints = _walk(frames)
    path = tmp_path / "nofps.npz"
    np.savez(
        path,
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        body_pose=np.zeros((frames, 63), dtype=np.float32),
        transl=np.zeros((frames, 3), dtype=np.float32),
        joints22=joints.astype(np.float32),
    )
    payload, error = validate(path)
    assert error == ""
    assert payload["_fps"] == 30.0


def test_a_corrupt_file_is_reported_not_raised(tmp_path):
    path = tmp_path / "junk.npz"
    path.write_bytes(b"this is not an npz")
    payload, error = validate(path)
    assert payload is None
    assert "unreadable" in error


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------


def _posed(frames: int = 60, *, noise: float = 0.0, seed: int = 0) -> dict:
    """A slow torso rotation, optionally with per-frame shake added."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.0, 0.6, frames)
    global_orient = np.stack([ramp * 0.0, ramp, ramp * 0.0], axis=1)
    body_pose = np.zeros((frames, 21, 3))
    body_pose[:, 2, 0] = ramp  # spine1 in body_pose order
    if noise:
        global_orient = global_orient + rng.normal(scale=noise, size=global_orient.shape)
        body_pose = body_pose + rng.normal(scale=noise, size=body_pose.shape)
    return {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": np.zeros((frames, 3)),
        "joints22": _walk(frames),
    }


def _shake(payload: dict) -> float:
    """Mean absolute second difference of the root's per-frame rotation step."""
    from motionviewer.core.smplx_fk import rodrigues

    matrices = rodrigues(payload["global_orient"])
    relative = np.einsum("tji,tjk->tik", matrices[:-1], matrices[1:])
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.abs(np.diff(np.degrees(np.arccos(trace)))).mean())


def test_a_window_under_two_is_a_no_op():
    payload = _posed(noise=0.01, seed=1)
    reference = {key: value.copy() for key, value in payload.items()}
    for window in (0, 1):
        smooth_pose(payload, window)
        for key, value in reference.items():
            assert np.array_equal(payload[key], value)


def test_smoothing_reduces_shake():
    payload = _posed(noise=0.01, seed=2)
    before = _shake(payload)
    smooth_pose(payload, 5)
    assert _shake(payload) < before * 0.6


def test_smoothing_leaves_joint_positions_untouched():
    """Positions do not drive the MMD solve, so the filter must not disturb them."""
    payload = _posed(noise=0.01, seed=3)
    reference = payload["joints22"].copy()
    smooth_pose(payload, 5)
    assert np.array_equal(payload["joints22"], reference)


def test_smoothing_only_touches_the_listed_joints():
    payload = _posed(noise=0.01, seed=4)
    reference = payload["body_pose"].copy()
    smooth_pose(payload, 5, joints=("pelvis",))
    assert np.array_equal(payload["body_pose"], reference)
    assert not np.array_equal(payload["global_orient"], _posed(noise=0.01, seed=4)["global_orient"])


def test_smoothing_preserves_the_overall_motion():
    """A clean ramp must survive the filter: this is a low-pass, not a flattener."""
    payload = _posed(noise=0.0, seed=5)
    reference = payload["global_orient"].copy()
    smooth_pose(payload, 5)
    assert np.abs(payload["global_orient"] - reference).max() < np.radians(1.0)


def test_smoothing_output_is_a_valid_rotation_sequence():
    """Averaging matrices leaves SO(3); the result has to be projected back."""
    from motionviewer.core.smplx_fk import rodrigues

    payload = _posed(noise=0.05, seed=6)
    smooth_pose(payload, 7)
    matrices = rodrigues(
        np.concatenate([payload["global_orient"][:, None, :], payload["body_pose"]], axis=1)
    ).reshape(-1, 3, 3)
    assert np.allclose(np.linalg.det(matrices), 1.0, atol=1e-9)
    identity = np.einsum("tji,tjk->tik", matrices, matrices)
    assert np.allclose(identity, np.eye(3), atol=1e-9)


def test_preprocess_trims_before_it_smooths():
    """Smoothing first would smear the anchor pop across the window."""
    payload = _posed(noise=0.0, seed=7)
    payload["global_orient"][0] += 0.5
    payload["joints22"][0, :, 0] += 0.30
    preprocess(payload, 1, 5)
    assert len(payload["global_orient"]) == 59
    # The 0.5 rad spike is gone rather than spread over the first frames.
    assert np.abs(payload["global_orient"][:4]).max() < 0.2


# ---------------------------------------------------------------------------
# caption in the filename
# ---------------------------------------------------------------------------


def test_caption_slug_is_filename_safe():
    slug = caption_slug("A person effortlessly throws a ball! (left arm)")
    assert slug == "a-person-effortlessly-throws-a-ball-left-arm"
    assert all(character.isalnum() or character == "-" for character in slug)


def test_caption_slug_cuts_on_a_word_boundary():
    """Truncating mid-word makes a filename that reads like a typo."""
    slug = caption_slug("The dancer performs a simple dance routine, swaying their arms")
    assert len(slug) <= 56
    assert not slug.endswith("-")
    assert slug.split("-")[-1] in {"a", "simple", "dance", "routine", "swaying", "their", "performs"}


def test_caption_slug_keeps_the_first_word_even_when_it_is_long():
    assert caption_slug("Antidisestablishmentarianismisaverylongsinglewordindeedyes") != ""


def test_caption_slug_of_an_empty_caption_is_empty():
    assert caption_slug("") == ""
    assert caption_slug("!!! ???") == ""


def test_video_name_carries_the_caption():
    clip = Clip(
        task="picks38/gen",
        clip_id="000015",
        path=Path("x.npz"),
        frames=40,
        fps=30.0,
        caption="A person is walking forward at a steady pace.",
    )
    name = video_name(clip, "yoimiya")
    assert name.startswith("picks38-gen_000015_yoimiya__")
    assert "walking-forward" in name


def test_video_name_without_a_caption_has_no_trailing_separator():
    clip = Clip(task="t/gen", clip_id="1", path=Path("x.npz"), frames=2, fps=30.0, caption="")
    assert video_name(clip, "yoimiya") == "t-gen_1_yoimiya"


def test_video_name_can_omit_the_caption():
    clip = Clip(task="t/gen", clip_id="1", path=Path("x.npz"), frames=2, fps=30.0, caption="a walk")
    assert video_name(clip, "yoimiya", slug=False) == "t-gen_1_yoimiya"


def test_captions_tsv_has_a_header_and_one_row_per_video(tmp_path):
    entries = [
        {
            "video": "videos/b.mp4",
            "task": "t/gen",
            "clip_id": "2",
            "character": "yoimiya",
            "frames": 5,
            "caption": "second\tclip\nwrapped",
        },
        {
            "video": "videos/a.mp4",
            "task": "t/gen",
            "clip_id": "1",
            "character": "yoimiya",
            "frames": 4,
            "caption": "first clip",
        },
    ]
    write_captions(tmp_path, entries)
    lines = (tmp_path / "captions.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == ["video", "task", "clip_id", "character", "frames", "caption"]
    assert len(lines) == 3
    # sorted by clip id, and tabs and newlines inside a caption cannot break a row
    assert lines[1].startswith("a.mp4\t")
    assert all(len(line.split("\t")) == 6 for line in lines[1:])


@pytest.mark.parametrize(
    ("flags", "mode"),
    [([], "bullet"), (["--bullet"], "bullet"), (["--spring"], "spring"),
     (["--physics", "none"], "none"), (["--no-spring"], "none")],
)
def test_physics_mode_reaches_the_renderer(tmp_path, monkeypatch, flags, mode):
    args = parse_args(flags)
    calls = []
    frames = tmp_path / "frames"

    def run(command, **kwargs):
        calls.append(command)
        (frames / "character" / args.view).mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(visualizer.subprocess, "run", run)
    clip = Clip("active", "1", tmp_path / "motion.npz", 20, 30, "walk")
    assert render_clip(clip, "yoimiya", frames, args, []) == ""
    command = calls[0]
    assert ("--bullet" in command) == (mode == "bullet")
    assert ("--spring" in command) == (mode == "spring")
    assert command[command.index("--foot-contact") + 1] == "off"
    assert command[command.index("--upper-smooth") + 1] == "0"
    assert command[command.index("--arm-clearance-degrees") + 1] == "0.0"
    assert "--faithful" in command
    assert "--polish" not in command
    assert command[command.index("--root-motion") + 1] == "source"
    if mode == "bullet":
        assert command[command.index("--bullet-damping") + 1] == "legacy-inverted"


@pytest.mark.parametrize("flags", [
    ["--bullet", "--spring"], ["--physics", "bullet", "--no-spring"],
    ["--bullet-substeps", "0"], ["--bullet-settle", "-1"], ["--smooth", "4"],
    ["--upper-smooth", "4"], ["--upper-smooth", "3", "--smooth", "3"],
    ["--floor-z", "nan"],
    ["--bullet-cloth-linear-slack", "nan"], ["--bullet-cloth-linear-slack", "inf"],
    ["--bullet-cloth-linear-slack", "-0.0001"], ["--bullet-cloth-linear-slack", "0.01001"],
    ["--arm-clearance-degrees", "nan"], ["--arm-clearance-degrees", "inf"],
    ["--arm-clearance-degrees", "-0.1"], ["--arm-clearance-degrees", "4.01"],
])
def test_invalid_physics_or_smoothing_settings_fail_before_rendering(flags):
    with pytest.raises(SystemExit):
        parse_args(flags)


def test_preprocessed_cache_tracks_source_changes_at_the_same_path(tmp_path, monkeypatch):
    monkeypatch.setattr(visualizer, "source_fingerprint", lambda: "test-pipeline")
    path = _write(tmp_path / "clip.npz", _walk())
    clip = Clip("active", "1", path, 60, 30, "walk", trim=1)
    cache = tmp_path / "cache"
    assert materialize(clip, cache, 3) == ""
    first_path = clip.motion_path
    first_bytes = first_path.read_bytes()
    _write(path, _walk() + 1.0)
    assert materialize(clip, cache, 3) == ""
    assert clip.motion_path != first_path
    assert first_path.read_bytes() == first_bytes
    with np.load(first_path) as old, np.load(clip.motion_path) as new:
        assert np.allclose(new["transl"] - old["transl"], 1.0)


def test_batch_resume_checks_inputs_and_settings_and_keeps_previous_results(tmp_path, monkeypatch):
    """Run orchestration without Blender, including changes and a failed rerender."""
    source = tmp_path / "motion.npz"
    source.write_bytes(b"first motion")
    asset = tmp_path / "character/model.pmx"
    asset.parent.mkdir()
    asset.write_bytes(b"model")
    blender = tmp_path / "blender"
    blender.write_bytes(b"fake executable")
    output = tmp_path / "output"
    clip = Clip("active", "1", source, 30, 30, "walk", score={"activity": 1, "penalty": 0})
    monkeypatch.setattr(visualizer, "CHARACTERS", {"yoimiya": str(asset)})
    monkeypatch.setattr(visualizer, "BLENDER", blender)
    monkeypatch.setattr(visualizer, "source_fingerprint", lambda: "test-pipeline")
    monkeypatch.setattr(visualizer, "discover", lambda *args: {"active": [clip]})
    renders = []
    fail = False

    def fake_render(target, character, frames_dir, args, share):
        renders.append(args.physics)
        if fail:
            return "test render failure"
        (frames_dir / "character" / args.view).mkdir(parents=True)
        (frames_dir / "info.json").write_text(json.dumps({
            "physics_mode": args.physics, "physics": {"built": args.physics == "bullet"},
        }))
        return ""

    def fake_encode(panels, destination, fps, width):
        destination.write_bytes(b"encoded video")
        return ""

    monkeypatch.setattr(visualizer, "render_clip", fake_render)
    monkeypatch.setattr(visualizer, "encode", fake_encode)

    def run(*flags):
        monkeypatch.setattr(sys, "argv", ["visualize_dataset.py", "--data", str(tmp_path),
                                         "--output", str(output), "--clips", "1", *flags])
        visualizer.main()
        return json.loads((output / "manifest.json").read_text())

    first = run()
    first_video = output / first["videos"][0]["video"]
    assert first["videos"][0]["render"]["settings"]["physics"] == "bullet"
    assert first["videos"][0]["renderer_reports"][0]["info"]["physics"]["built"] is True
    assert run() == first
    assert renders == ["bullet"]

    second = run("--physics", "spring")
    assert len(second["videos"]) == 1
    assert second["history"] == first["videos"]
    assert first_video.read_bytes() == b"encoded video"
    assert second["videos"][0]["video"] != first["videos"][0]["video"]

    source.write_bytes(b"changed motion at the same path")
    third = run("--physics", "spring")
    assert renders == ["bullet", "spring", "spring"]
    assert len(third["history"]) == 2
    assert third["videos"][0]["render_signature"] != second["videos"][0]["render_signature"]

    # A failed new render must leave both old metadata and the playable file intact.
    fail = True
    clip.caption = "new caption must not be applied to old video"
    source.write_bytes(b"another revision")
    with pytest.raises(SystemExit, match="1"):
        run()
    assert json.loads((output / "manifest.json").read_text()) == third
    assert (output / third["videos"][0]["video"]).read_bytes() == b"encoded video"

    # Different-rate panels must not silently play at the primary clip's speed.
    other = Clip("gt", "1", source, 60, 60, "walk", score=clip.score)
    monkeypatch.setattr(visualizer, "discover", lambda *args: {"active": [clip], "gt": [other]})
    render_count = len(renders)
    with pytest.raises(SystemExit, match="1"):
        run("--compare", "gt")
    assert len(renders) == render_count
    assert json.loads((output / "manifest.json").read_text()) == third


def test_legacy_videos_without_provenance_are_not_treated_as_bullet(tmp_path):
    video = tmp_path / "old.mp4"
    video.write_bytes(b"legacy spring video")
    assert not visualizer.reusable_video({"video": video.name}, tmp_path, "new-bullet-signature")


def test_character_texture_changes_invalidate_the_asset_fingerprint(tmp_path):
    asset = tmp_path / "model.pmx"
    texture = tmp_path / "face.png"
    asset.write_bytes(b"unchanged model")
    texture.write_bytes(b"first texture")
    before = visualizer.asset_fingerprint(asset)
    texture.write_bytes(b"fixed texture")
    assert visualizer.asset_fingerprint(asset) != before


def test_contact_floor_and_upper_smoothing_reach_the_renderer(tmp_path, monkeypatch):
    args = parse_args(["--foot-contact", "on", "--floor-z", "0.04", "--upper-smooth", "3",
                       "--arm-clearance-degrees", "4"])
    calls = []
    frames = tmp_path / "frames"

    def run(command, **kwargs):
        calls.append(command)
        (frames / "character" / args.view).mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(visualizer.subprocess, "run", run)
    clip = Clip("active", "1", tmp_path / "motion.npz", 20, 30, "walk")
    assert render_clip(clip, "yoimiya", frames, args, []) == ""
    command = calls[0]
    for flag, value in (("--foot-contact", "on"), ("--floor-z", "0.04"), ("--upper-smooth", "3"),
                        ("--arm-clearance-degrees", "4.0")):
        assert command[command.index(flag) + 1] == value


def test_explicit_legacy_pose_settings_reach_the_renderer(tmp_path, monkeypatch):
    args = parse_args(["--polish", "--root-motion", "legacy-scaled"])
    calls = []
    frames = tmp_path / "frames"

    def run(command, **kwargs):
        calls.append(command)
        (frames / "character" / args.view).mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(visualizer.subprocess, "run", run)
    clip = Clip("active", "1", tmp_path / "motion.npz", 20, 30, "walk")
    assert render_clip(clip, "yoimiya", frames, args, []) == ""
    command = calls[0]
    assert "--polish" in command
    assert "--faithful" not in command
    assert command[command.index("--root-motion") + 1] == "legacy-scaled"


def test_batch_linear_slack_round_trips_through_the_actual_renderer_parser(tmp_path, monkeypatch):
    from render_mmd_compare import parse_args as parse_render_args

    args = parse_args(["--bullet", "--bullet-cloth-linear-slack", ".002"])
    parsed = []
    frames = tmp_path / "frames"

    def run(command, **kwargs):
        parsed.append(parse_render_args(command[command.index("--") + 1:]))
        (frames / "character" / args.view).mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(visualizer.subprocess, "run", run)
    clip = Clip("active", "1", tmp_path / "motion.npz", 20, 30, "walk")
    assert render_clip(clip, "yoimiya", frames, args, []) == ""
    assert len(parsed) == 1
    assert parsed[0].bullet is True
    assert parsed[0].bullet_cloth_linear_slack == .002


@pytest.mark.parametrize("flags", [
    ["--foot-contact", "on"], ["--floor-z", "0.04"], ["--upper-smooth", "3"],
    ["--bullet-cloth-slack", "10"], ["--bullet-hand-colliders"], ["--bullet-damping", "pmx"],
    ["--arm-clearance-degrees", "4"],
    ["--polish"], ["--root-motion", "legacy-scaled"],
    ["--bullet-cloth-linear-slack", ".002"],
])
def test_contact_and_smoothing_changes_invalidate_video_resume(tmp_path, flags):
    source = _write(tmp_path / "motion.npz", _walk())
    clip = Clip("active", "1", source, 60, 30, "walk")
    old = visualizer.render_provenance(clip, None, "yoimiya", parse_args([]), {})
    new = visualizer.render_provenance(clip, None, "yoimiya", parse_args(flags), {})
    assert visualizer.signature(old) != visualizer.signature(new)


def test_profile_contents_invalidate_resume_even_at_the_same_path(tmp_path, monkeypatch):
    import hashlib

    asset = tmp_path / "character.pmx"
    asset.write_bytes(b"local test asset")
    monkeypatch.setattr(visualizer, "CHARACTERS", {"test": str(asset)})
    source = _write(tmp_path / "motion.npz", _walk())
    clip = Clip("active", "1", source, 60, 30, "walk")
    profile = tmp_path / "physics.json"
    payload = dict(schema_version=1, profile_id="test", asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                   import_scale=0.08, bullet={"substeps": 12}, visual_verified=[])
    profile.write_text(json.dumps(payload))
    args = parse_args(["--physics-profile", str(profile)])
    before = visualizer.render_provenance(clip, None, "test", args, {})
    payload["bullet"]["substeps"] = 24
    profile.write_text(json.dumps(payload))
    after = visualizer.render_provenance(clip, None, "test", args, {})
    assert visualizer.signature(before) != visualizer.signature(after)
    with pytest.raises(SystemExit):
        parse_args(["--physics-profile", str(profile), "--physics", "none"])
