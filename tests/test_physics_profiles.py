import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from motionviewer.assets.physics_profiles import (
    load_physics_profile,
    physics_profile_from_json,
    resolve_physics_profile,
)


def profile_data(asset=b"one original PMX"):
    return {
        "schema_version": 1, "profile_id": "costume-review",
        "asset_sha256": hashlib.sha256(asset).hexdigest(), "import_scale": 0.08,
        "bullet": {"damping_mode": "pmx", "cloth_angular_slack_degrees": 20,
                   "cloth_linear_slack_m": 0.002, "cloth_bone_tokens": ["裙", "skirt"]},
        "visual_verified": [{"motion_sha256": "a" * 64, "motion_id": "100948",
                             "frame_start": 1, "frames": 152, "fps": 50,
                             "views": ["three_quarter", "side"],
                             "display_context": "native SOMA, faithful, no foot or arm correction",
                             "evidence": "review/100948/index.html", "notes": "Some rigid-panel appearance remains."}],
    }


def test_exact_asset_and_scale_resolve_only_explicit_physics_values(tmp_path):
    asset = tmp_path / "renamed.pmx"
    asset.write_bytes(b"one original PMX")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile_data()))
    before_asset, before_profile = asset.read_bytes(), path.read_bytes()
    resolved = resolve_physics_profile(path, asset_path=asset, import_scale=0.08)
    assert resolved.bullet_overrides == {
        "fix_damping": False, "cloth_angular_slack_degrees": 20.0,
        "cloth_linear_slack_m": 0.002, "cloth_bone_tokens": ("裙", "skirt"),
    }
    assert "substeps" not in resolved.bullet_overrides
    assert asset.read_bytes() == before_asset and path.read_bytes() == before_profile
    report = resolved.report()
    assert report["asset_match"] is True
    assert len(report["visual_verified"]) == 1
    assert report["visual_verified"][0]["motion_sha256"] == "a" * 64
    assert "Listed renders only" in report["verification_scope"]


def test_unknown_or_replaced_asset_and_different_scale_rejected(tmp_path):
    asset = tmp_path / "same-name.pmx"
    asset.write_bytes(b"different PMX")
    profile = physics_profile_from_json(profile_data())
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_physics_profile(profile, asset_path=asset, import_scale=0.08)
    asset.write_bytes(b"one original PMX")
    with pytest.raises(ValueError, match="import_scale mismatch"):
        resolve_physics_profile(profile, asset_path=asset, import_scale=0.081)


@pytest.mark.parametrize("field", ["polish", "root_motion", "arm_clearance", "finger_pose", "cloth_nodes"])
def test_pose_and_unimplemented_settings_are_not_silently_accepted(field):
    data = profile_data()
    data["bullet"][field] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        physics_profile_from_json(data)
    data = profile_data()
    data[field] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        physics_profile_from_json(data)


@pytest.mark.parametrize(("field", "value"), [
    ("substeps", True), ("iterations", 12.5), ("settle_frames", -1),
    ("cloth_linear_slack_m", 0.011), ("cloth_linear_slack_m", float("nan")),
    ("cloth_angular_slack_degrees", float("inf")), ("collision_margin", -1),
    ("fit_hand_colliders", "false"), ("despike", 1), ("neutral_settle", "yes"), ("damping_mode", "auto"),
    ("cloth_bone_tokens", []), ("cloth_bone_tokens", ["裙", "裙"]),
])
def test_invalid_physics_values_rejected(field, value):
    data = profile_data()
    data["bullet"][field] = value
    with pytest.raises(ValueError):
        physics_profile_from_json(data)


@pytest.mark.parametrize(("field", "value"), [
    ("frames", 0), ("fps", 0), ("fps", float("nan")), ("frame_start", True),
    ("motion_sha256", "name-not-a-digest"), ("views", []), ("display_context", ""),
])
def test_visual_acceptance_requires_identified_sample_context(field, value):
    data = profile_data()
    data["visual_verified"][0][field] = value
    with pytest.raises(ValueError):
        physics_profile_from_json(data)


def test_unreviewed_profile_is_allowed_but_no_global_acceptance_claim():
    data = profile_data()
    data["visual_verified"] = []
    profile = physics_profile_from_json(data)
    assert profile.visual_verified == ()
    assert profile.to_json()["visual_verified"] == []
    data["visual_verified"] = True
    with pytest.raises(ValueError, match="list of reviewed samples"):
        physics_profile_from_json(data)


def test_profile_is_immutable_and_reports_cannot_mutate_it():
    data = profile_data()
    profile = physics_profile_from_json(data)
    expected = copy.deepcopy(profile.to_json())
    data["bullet"]["cloth_bone_tokens"].append("body")
    profile.to_json()["visual_verified"][0]["views"].append("rear")
    profile.bullet_overrides["substeps"] = 100
    assert profile.to_json() == expected
    with pytest.raises(FrozenInstanceError):
        profile.import_scale = 1


def test_fingerprint_changes_with_settings_selection_asset_and_evidence():
    data = profile_data()
    original = physics_profile_from_json(data)
    assert physics_profile_from_json(original.to_json()).fingerprint == original.fingerprint
    for section, key, value in [
        ("bullet", "cloth_linear_slack_m", 0.001),
        ("bullet", "cloth_bone_tokens", ["裾"]),
        (None, "asset_sha256", "b" * 64),
        (None, "import_scale", 0.1),
        (None, "visual_verified", []),
    ]:
        changed = copy.deepcopy(data)
        (changed if section is None else changed[section])[key] = value
        assert physics_profile_from_json(changed).fingerprint != original.fingerprint


def test_duplicate_json_keys_rejected_and_legacy_damping_is_explicit(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_physics_profile(path)
    data = profile_data()
    data["bullet"]["damping_mode"] = "legacy-inverted"
    assert physics_profile_from_json(data).bullet_overrides["fix_damping"] is True


def test_mesh_cloth_settings_validate_freeze_and_participate_in_fingerprint(tmp_path):
    data = profile_data()
    plain = physics_profile_from_json(data)
    assert plain.mesh_cloth_settings is None
    data["mesh_cloth"] = {"bone_prefixes": ["裙_", "衣襬_"], "materials": ["裙", "衣摆"],
                          "outer_surface_materials": ["衣摆"], "bending_stiffness": 0.1}
    profile = physics_profile_from_json(data)
    assert profile.fingerprint != plain.fingerprint
    settings = profile.mesh_cloth_settings
    assert settings["bone_prefixes"] == ("裙_", "衣襬_")
    assert settings["quality"] == 12
    settings["quality"] = 1
    data["mesh_cloth"]["bone_prefixes"].append("body")
    assert profile.mesh_cloth_settings["quality"] == 12
    assert profile.mesh_cloth_settings["bone_prefixes"] == ("裙_", "衣襬_")
    asset = tmp_path / "asset.pmx"
    asset.write_bytes(b"one original PMX")
    resolved = resolve_physics_profile(profile, asset_path=asset, import_scale=0.08)
    assert resolved.mesh_cloth_settings == profile.mesh_cloth_settings
    assert resolved.report()["mesh_cloth"]["materials"] == ("裙", "衣摆")
    assert physics_profile_from_json(json.loads(json.dumps(profile.to_json()))).fingerprint == profile.fingerprint


@pytest.mark.parametrize("extra", [
    {"root_motion": "source"}, {"polish": False}, {"collision_distance_m": 0},
    {"quality": True}, {"outer_surface_materials": ["not selected"]},
    {"pin_full_weight": 0.8, "pin_transition_weight": 0.7},
])
def test_mesh_cloth_rejects_pose_changes_and_invalid_adapter_values(extra):
    data = profile_data()
    data["mesh_cloth"] = {"bone_prefixes": ["裙_"], "materials": ["裙"], **extra}
    with pytest.raises(ValueError):
        physics_profile_from_json(data)


def test_programmatic_profile_cannot_bypass_pose_setting_rejection(tmp_path):
    profile = replace(physics_profile_from_json(profile_data()), bullet=(("polish", True),))
    asset = tmp_path / "asset.pmx"
    asset.write_bytes(b"one original PMX")
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_physics_profile(profile, asset_path=asset, import_scale=0.08)


def test_joint_rules_are_asset_bound_immutable_and_in_fingerprint(tmp_path):
    data = profile_data()
    original = physics_profile_from_json(data)
    data["joint_rules"] = [
        {"bone_prefix": "左D_", "locked_angular_half_range_degrees": 3},
        {"bone_prefix": "右D_", "locked_angular_half_range_degrees": 3},
    ]
    profile = physics_profile_from_json(data)
    asset = tmp_path / "asset.pmx"
    asset.write_bytes(b"one original PMX")
    resolved = resolve_physics_profile(profile, asset_path=asset, import_scale=0.08)
    assert resolved.joint_rules == profile.joint_rules
    assert resolved.joint_rules[0].bone_prefix == "左D_"
    assert profile.fingerprint != original.fingerprint
    assert "joint_rules" not in resolved.bullet_overrides
    assert physics_profile_from_json(profile.to_json()).fingerprint == profile.fingerprint
    data["joint_rules"][0]["locked_angular_half_range_degrees"] = 4
    assert physics_profile_from_json(data).fingerprint != profile.fingerprint
    resolved.report()["joint_rules"][0]["bone_prefix"] = "body"
    assert resolved.joint_rules[0].bone_prefix == "左D_"
    with pytest.raises(FrozenInstanceError):
        resolved.joint_rules[0].bone_prefix = "body"


@pytest.mark.parametrize("rules", [
    True,
    [{"bone_prefix": "裙_", "root_motion": "source"}],
    [{"bone_prefix": "裙_", "locked_angular_half_range_degrees": True}],
    [{"bone_prefix": "裙_", "locked_linear_half_range_m": 0.02}],
    [{"bone_prefix": "裙", "locked_angular_half_range_degrees": 3},
     {"bone_prefix": "裙_", "locked_angular_half_range_degrees": 3}],
])
def test_profile_reuses_joint_rule_safety_validation(rules):
    data = profile_data()
    data["joint_rules"] = rules
    with pytest.raises(ValueError):
        physics_profile_from_json(data)


def test_despike_is_an_explicit_boolean_bullet_override():
    data = profile_data()
    data["bullet"]["despike"] = True
    assert physics_profile_from_json(data).bullet_overrides["despike"] is True
    data["bullet"].pop("despike")
    assert "despike" not in physics_profile_from_json(data).bullet_overrides


def test_side_link_rules_round_trip_and_change_the_fingerprint():
    data = profile_data()
    base = physics_profile_from_json(data)
    data["side_link_rules"] = [{"joint_prefix": "HB1横", "linear_scale": 0.3}]
    profile = physics_profile_from_json(data)
    assert profile.side_link_rules[0].joint_prefix == "HB1横"
    assert profile.to_json()["side_link_rules"] == [{"joint_prefix": "HB1横", "linear_scale": 0.3}]
    assert profile.fingerprint != base.fingerprint
