"""Single-clip preprocessing is optional and preserves original motion files."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from render_mmd_compare import parse_args, prepare_upper_smoothed_motion  # noqa: E402


def _source(path):
    frames = 12
    pose = np.zeros((frames, 21, 3), dtype=np.float32)
    pose[:, 2, 0] = np.radians(np.tile([1., -1.], frames // 2))
    np.savez(path, global_orient=np.zeros((frames, 3), dtype=np.float32), body_pose=pose,
             transl=np.zeros((frames, 3), dtype=np.float32), joints22=np.ones((frames, 22, 3), dtype=np.float32),
             fps=np.float32(50), caption=np.str_("原始动作"), extra_array=np.array([1, 2, 3]))
    return path


def test_processed_motion_is_a_complete_copy_with_provenance_and_original_untouched(tmp_path):
    source = _source(tmp_path / "source.npz")
    original_bytes = source.read_bytes()
    output = tmp_path / "output"
    processed, report = prepare_upper_smoothed_motion(source, output, 3)
    assert source.read_bytes() == original_bytes
    assert processed == output / "processed-motion.npz"
    assert report["source_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert report["max_local_deviation_deg"] == 2.0
    with np.load(source) as original, np.load(processed) as updated:
        for key in ("global_orient", "transl", "fps", "caption", "extra_array"):
            assert np.array_equal(updated[key], original[key])
        assert not np.array_equal(updated["body_pose"], original["body_pose"])
        assert str(updated["motionviewer_source_sha256"]) == report["source_sha256"]
        assert json.loads(str(updated["motionviewer_upper_smoothing_report"])) == report


def test_disabled_processing_does_not_create_a_derived_file(tmp_path):
    source = _source(tmp_path / "source.npz")
    returned, report = prepare_upper_smoothed_motion(source, tmp_path / "output", 0)
    assert returned == source
    assert report["enabled"] is False
    assert not (tmp_path / "output").exists()


def test_processing_refuses_to_overwrite_a_source_named_like_the_output(tmp_path):
    source = _source(tmp_path / "processed-motion.npz")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite the source"):
        prepare_upper_smoothed_motion(source, tmp_path, 3)
    assert source.read_bytes() == before


def test_an_input_named_like_the_old_temporary_file_is_preserved(tmp_path):
    source = _source(tmp_path / "processed-motion.tmp.npz")
    before = source.read_bytes()
    processed, _ = prepare_upper_smoothed_motion(source, tmp_path, 3)
    assert source.read_bytes() == before
    assert processed.is_file()
    assert not list(tmp_path.glob(".processed-motion-*.npz"))


def test_review_edits_are_opt_in():
    args = parse_args(["--asset", "model.pmx", "--motion", "motion.npz", "--output", "output"])
    assert args.foot_contact == "off"
    assert args.upper_smooth == 0
    assert args.arm_clearance_degrees == 0
    assert args.floor_z is None
    assert args.match_camera is None
    assert args.weapon_fit_mode == "legacy"
    assert args.weapon_grip_section is None
    assert args.bullet_cloth_slack == 0
    assert args.bullet_cloth_linear_slack == 0
    assert args.bullet_hand_colliders is False
    assert args.scene_contact is False
    assert args.faithful is True
    assert args.root_motion == "source"
    assert args.weapon_arm_clearance is False
    assert args.override_source_fingers is False
    assert args.physics_profile is None


def test_physics_profile_cannot_silently_be_ignored():
    base = ["--asset", "model.pmx", "--soma", "motion.npz", "--output", "output", "--physics-profile", "profile.json"]
    with pytest.raises(SystemExit):
        parse_args(base)
    args = parse_args(base + ["--bullet"])
    assert args.physics_profile == Path("profile.json")
    assert args.faithful and args.root_motion == "source"


def test_supported_box_accepts_fixed_dimensions_without_actor_contact_edits():
    args = parse_args(["--asset", "model.pmx", "--soma", "motion.npz", "--output", "output",
                       "--toon", "--scene", "carry_box_supported", "--scene-box-size", ".4", ".22", ".15"])
    assert args.scene_box_size == [.4, .22, .15]
    assert args.scene_contact is False
    assert args.faithful is True


@pytest.mark.parametrize("flags", [
    ["--scene-contact"],
    ["--scene", "carry_box_two_hand", "--scene-contact", "--scene-grip-mode", "bottom"],
    ["--scene", "carry_box_two_hand", "--scene-box-size", "0.3", "nan", "0.15"],
    ["--bullet-cloth-slack", "nan"],
    ["--bullet-cloth-slack", "-1"],
    ["--bullet-cloth-slack", "31"],
    ["--bullet-cloth-linear-slack", "nan"],
    ["--bullet-cloth-linear-slack", "inf"],
    ["--bullet-cloth-linear-slack", "-0.0001"],
    ["--bullet-cloth-linear-slack", "0.01001"],
    ["--weapon-grip", "0.03"],
    ["--arm-clearance-degrees", "nan"],
    ["--arm-clearance-degrees", "inf"],
    ["--arm-clearance-degrees", "-0.1"],
    ["--arm-clearance-degrees", "20.01"],
    ["--weapon-grip-section", "middle"],
    ["--root-motion", "automatic"],
    ["--faithful", "--polish"],
    ["--weapon-arm-clearance"],
    ["--toon", "--weapon", "auto", "--weapon-arm-clearance"],
    ["--override-source-fingers"],
])
def test_reject_unvalidated_or_nonphysical_contact_settings(flags):
    with pytest.raises(SystemExit):
        parse_args(["--asset", "model.pmx", "--motion", "motion.npz", "--output", "output", *flags])


def test_soma_cannot_accidentally_use_the_body22_filter():
    with pytest.raises(SystemExit):
        parse_args(["--asset", "model.pmx", "--soma", "motion.npz", "--output", "output", "--upper-smooth", "3"])


@pytest.mark.parametrize("metres", [0., .002, .01])
def test_cloth_linear_slack_is_explicitly_measured_in_metres(metres):
    args = parse_args(["--asset", "model.pmx", "--soma", "motion.npz", "--output", "output",
                       "--bullet", "--bullet-cloth-linear-slack", str(metres)])
    assert args.bullet_cloth_linear_slack == metres


def test_legacy_pose_changes_and_source_finger_override_require_explicit_flags():
    args = parse_args([
        "--asset", "model.pmx", "--soma", "motion.npz", "--output", "output", "--toon",
        "--polish", "--root-motion", "legacy-scaled", "--weapon", "auto", "--weapon-fit-mode", "surface",
        "--weapon-arm-clearance", "--override-source-fingers",
    ])
    assert args.faithful is False
    assert args.root_motion == "legacy-scaled"
    assert args.weapon_arm_clearance is True
    assert args.override_source_fingers is True


@pytest.mark.parametrize("section", ["low", "high"])
def test_explicit_weapon_path_can_choose_a_calibrated_grip_region(section):
    args = parse_args([
        "--asset", "model.pmx", "--motion", "motion.npz", "--output", "output",
        "--toon", "--weapon", "weapon.pmx", "--weapon-fit-mode", "surface", "--weapon-grip-section", section,
    ])
    assert args.weapon_grip_section == section
    assert args.weapon_grip == 0.0


@pytest.mark.parametrize("flags", [
    ["--scene", "carry_box_two_hand", "--scene-contact"],
    ["--weapon", "auto", "--weapon-fit-mode", "surface"],
    ["--weapon", "weapon.pmx", "--weapon-fit-mode", "surface"],
])
def test_new_prop_fits_cannot_be_silently_skipped_without_toon(flags, capsys):
    base = ["--asset", "model.pmx", "--motion", "motion.npz", "--output", "output"]
    with pytest.raises(SystemExit):
        parse_args([*base, *flags])
    assert "require --toon" in capsys.readouterr().err
    assert parse_args([*base, "--toon", *flags]).toon is True


def test_legacy_weapon_parse_defaults_remain_compatible():
    args = parse_args(["--asset", "model.pmx", "--motion", "motion.npz", "--output", "output",
                       "--weapon", "auto"])
    assert args.weapon_fit_mode == "legacy"
    assert args.toon is False


def test_cloth_audit_forwards_arm_clearance_to_the_renderer_once(monkeypatch):
    import _audit_mmd_cloth
    import render_mmd_compare

    from motionviewer.blender import mmd_bullet

    parsed = []
    monkeypatch.setattr(sys, "argv", [
        "_audit_mmd_cloth.py", "--", "--audit-only", "--asset", "model.pmx",
        "--motion", "motion.npz", "--output", "output", "--arm-clearance-degrees", "4",
    ])
    monkeypatch.setattr(mmd_bullet, "build_and_bake", lambda *args, **kwargs: {})
    monkeypatch.setattr(render_mmd_compare, "main", lambda: parsed.append(parse_args()))
    _audit_mmd_cloth.main()
    assert len(parsed) == 1
    assert parsed[0].arm_clearance_degrees == 4
    assert parsed[0].panels == ""
    assert sys.argv.count("--arm-clearance-degrees") == 1
