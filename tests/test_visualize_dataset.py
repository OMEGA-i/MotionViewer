"""Contract validation and lead-in trimming for the dataset visualiser.

The trimming rules were derived from measurements on the T2M exports, so the
numbers in these tests are the measured ones: a 14-frame frozen lead-in on every
``old500`` clip, and a ~300 mm single-frame jump out of the conditioning pose at
the head of every ``gen`` clip.  Getting the frame-0 case wrong is not cosmetic —
the retarget re-bases the whole root path on frame 0, so a bad first frame
mis-places the entire clip.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from visualize_dataset import (  # noqa: E402
    REQUIRED,
    Clip,
    _alias,
    _clip_id,
    caption_slug,
    detect_trim,
    preprocess,
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
