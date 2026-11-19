from copy import deepcopy
from math import radians
from types import SimpleNamespace as NS

import pytest

from motionviewer.assets.physics_joint_rules import JointRule, joint_rules_from_json, widened_limits
from motionviewer.blender.mmd_joint_rules import apply_joint_rules


def joint(name, a="左D_0", b="左D_1", modes=("1", "1")):
    c = NS(object1=NS(mmd_rigid=NS(bone=a, type=modes[0])),
           object2=NS(mmd_rigid=NS(bone=b, type=modes[1])), disable_collisions=True)
    for kind in ("ang", "lin"):
        for axis in "xyz":
            setattr(c, f"use_limit_{kind}_{axis}", True)
            setattr(c, f"limit_{kind}_{axis}_lower", 0.0)
            setattr(c, f"limit_{kind}_{axis}_upper", 0.0)
    return NS(name=name, rigid_body_constraint=c)


def test_ribbon_rule_only_changes_internal_locked_angular_limits():
    joints = [joint("interior"), joint("waist", a="下半身", modes=("0", "1")),
              joint("other-chain", b="右D_1"), joint("ornament", b="左Dls2"),
              joint("static-in-prefix", modes=("0", "1"))]
    old = deepcopy(joints)
    c = joints[0].rigid_body_constraint
    c.limit_ang_x_lower, c.limit_ang_x_upper = -0.2, 0.2
    c.use_limit_ang_z = False
    result = apply_joint_rules(NS(joints=lambda: joints), [JointRule("左D_", 3)])
    assert result["changed_joints"] == result["matched_internal_dynamic_joints"] == 1
    assert result["changed_axes"] == 1
    assert c.limit_ang_y_lower == pytest.approx(-radians(3))
    assert c.limit_ang_x_lower == -0.2
    assert c.limit_ang_z_lower == c.limit_lin_x_lower == c.limit_lin_y_upper == 0
    assert c.disable_collisions is True
    assert joints[1:] == old[1:]
    assert result["changes"][0]["axes"][0]["before"] == [0, 0]


def test_minimum_range_preserves_centre_and_wider_authored_limits():
    j = joint("coat", a="HB1_1", b="HB1_2")
    c = j.rigid_body_constraint
    c.limit_ang_x_lower, c.limit_ang_x_upper = 0.1, 0.2
    c.limit_ang_y_lower, c.limit_ang_y_upper = -0.5, 0.5
    c.limit_lin_z_lower, c.limit_lin_z_upper = -0.01, 0.01
    rule = JointRule("HB1_", min_angular_half_range_degrees=15, locked_linear_half_range_m=0.001)
    report = apply_joint_rules(NS(joints=lambda: [j]), [rule])
    assert c.limit_ang_x_lower == pytest.approx(0.15 - radians(15))
    assert c.limit_ang_y_upper == 0.5
    assert c.limit_lin_x_lower == -0.001
    assert c.limit_lin_z_lower == -0.01
    assert report["changed_axes"] == 4


@pytest.mark.parametrize("value", [
    None, {}, [{"bone_prefix": ""}], [{"bone_prefix": "D_", "unknown": 1}],
    [{"bone_prefix": "D_", "locked_angular_half_range_degrees": True}],
    [{"bone_prefix": "D_", "locked_angular_half_range_degrees": float("nan")}],
    [{"bone_prefix": "D_", "locked_linear_half_range_m": 0.011}],
    [{"bone_prefix": "D_", "locked_angular_half_range_degrees": 46}],
    [{"bone_prefix": "D_", "locked_angular_half_range_degrees": 3, "min_angular_half_range_degrees": 5}],
    [{"bone_prefix": "D_", "locked_angular_half_range_degrees": 3},
     {"bone_prefix": "D_1", "locked_angular_half_range_degrees": 3}],
])
def test_reject_unsafe_or_ambiguous_rules(value):
    with pytest.raises(ValueError):
        joint_rules_from_json(value)


def test_rule_roundtrip_and_unlimited_reversed_range():
    rules = joint_rules_from_json([{"bone_prefix": "D_", "locked_angular_half_range_degrees": 3}])
    assert joint_rules_from_json([r.to_json() for r in rules]) == rules
    assert widened_limits(0.1, -0.1, 0.2, locked_only=False) == (0.1, -0.1)


def test_empty_rules_do_not_access_model():
    assert apply_joint_rules(object(), ()) == {
        "enabled": False, "rules": [], "matched_internal_dynamic_joints": 0,
        "changed_joints": 0, "changed_axes": 0, "changes": [],
        "root_anchors_preserved": True, "cross_chain_joints_preserved": True,
        "source_files_modified": False, "solver_targets_not_measured_error_bounds": True,
    }


def test_side_link_rule_only_narrows_toward_the_centre():
    from motionviewer.assets.physics_joint_rules import narrowed_limits

    assert narrowed_limits(-0.04, 0.02, 0.3) == pytest.approx((-0.019, -0.001))  # centre -0.01
    assert narrowed_limits(-0.01, 0.01, 0.0) == pytest.approx((0.0, 0.0))
    assert narrowed_limits(0.02, -0.02, 0.3) == (0.02, -0.02)  # reversed: preserved


@pytest.mark.parametrize("item", [
    {"joint_prefix": "", "linear_scale": 0.3},
    {"joint_prefix": "HB1横", "linear_scale": 1.0},
    {"joint_prefix": "HB1横", "linear_scale": -0.1},
    {"joint_prefix": "HB1横", "linear_scale": True},
    {"joint_prefix": "HB1横"},
    {"joint_prefix": "HB1横", "linear_scale": 0.3, "extra": 1},
])
def test_side_link_rule_validation(item):
    from motionviewer.assets.physics_joint_rules import side_link_rules_from_json

    with pytest.raises((ValueError, TypeError)):
        side_link_rules_from_json([item])


def test_side_link_rules_match_decorated_joint_names_and_leave_others():
    from types import SimpleNamespace

    from motionviewer.assets.physics_joint_rules import SideLinkRule
    from motionviewer.blender.mmd_joint_rules import apply_side_link_rules

    def joint(name):
        c = SimpleNamespace(**{f"use_limit_lin_{a}": True for a in "xyz"},
                            **{f"limit_lin_{a}_lower": -0.04 for a in "xyz"},
                            **{f"limit_lin_{a}_upper": 0.04 for a in "xyz"})
        return SimpleNamespace(name=name, rigid_body_constraint=c)

    side, chain = joint("183_J.HB1横_7_1"), joint("063_J.HB1_1_0")
    model = SimpleNamespace(joints=lambda: [side, chain])
    report = apply_side_link_rules(model, (SideLinkRule("HB1横", 0.3),))
    assert report["changed_joints"] == 1
    assert side.rigid_body_constraint.limit_lin_x_upper == pytest.approx(0.012)
    assert chain.rigid_body_constraint.limit_lin_x_upper == 0.04
