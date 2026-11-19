"""Apply asset-bound garment compliance after MMD build, before Bullet bake."""

from __future__ import annotations

import math
from collections.abc import Sequence

from motionviewer.assets.physics_joint_rules import JointRule, widened_limits


def apply_joint_rules(model, rules: Sequence[JointRule]) -> dict:
    """Change only enabled joint limits between two same-chain dynamic bodies.

    Asset SHA/import-scale validation is the profile loader's responsibility.
    This adapter never changes masses, masks, damping, bind frames, body motion,
    or the source asset. The full before/after limits are returned for provenance.
    """
    report = {
        "enabled": bool(rules), "rules": [r.to_json() for r in rules],
        "matched_internal_dynamic_joints": 0, "changed_joints": 0,
        "changed_axes": 0, "changes": [],
        "root_anchors_preserved": True, "cross_chain_joints_preserved": True,
        "source_files_modified": False, "solver_targets_not_measured_error_bounds": True,
    }
    if not rules:
        return report
    changes = []
    matched = 0
    for joint in model.joints():
        constraint = joint.rigid_body_constraint
        if not constraint or not constraint.object1 or not constraint.object2:
            continue
        bodies = (constraint.object1, constraint.object2)
        if not all(str(body.mmd_rigid.type) in ("1", "2") for body in bodies):
            continue
        names = [body.mmd_rigid.bone for body in bodies]
        rule = next((r for r in rules if all(n.startswith(r.bone_prefix) for n in names)), None)
        if rule is None:
            continue
        matched += 1
        axes = []
        angular = rule.locked_angular_half_range_degrees or rule.min_angular_half_range_degrees
        for kind, half_range, locked_only in (
            ("ang", math.radians(angular), not bool(rule.min_angular_half_range_degrees)),
            ("lin", rule.locked_linear_half_range_m, True),
        ):
            for axis in "xyz":
                if not getattr(constraint, f"use_limit_{kind}_{axis}"):
                    continue
                low, high = f"limit_{kind}_{axis}_lower", f"limit_{kind}_{axis}_upper"
                before = (getattr(constraint, low), getattr(constraint, high))
                after = widened_limits(*before, half_range, locked_only=locked_only)
                if after == before:
                    continue
                setattr(constraint, low, after[0])
                setattr(constraint, high, after[1])
                axes.append({"kind": kind, "axis": axis, "before": list(before), "after": list(after)})
        if axes:
            changes.append({"joint": joint.name, "bones": names, "bone_prefix": rule.bone_prefix, "axes": axes})
    report.update({
        "matched_internal_dynamic_joints": matched, "changed_joints": len(changes),
        "changed_axes": sum(len(c["axes"]) for c in changes), "changes": changes,
    })
    return report


def apply_side_link_rules(model, rules) -> dict:
    """Narrow the linear limits of joints named ``<joint_prefix>...``; report each change.

    PMX joint names arrive as ``NNN_J.<name>`` from mmd_tools, so the prefix is matched
    after that decoration.
    """
    from motionviewer.assets.physics_joint_rules import narrowed_limits

    report = {"enabled": bool(rules), "rules": [r.to_json() for r in rules], "changed_joints": 0, "changes": []}
    if not rules:
        return report
    for joint in model.joints():
        constraint = joint.rigid_body_constraint
        if not constraint:
            continue
        name = joint.name.split("_J.", 1)[-1] if "_J." in joint.name else joint.name
        rule = next((r for r in rules if name.startswith(r.joint_prefix)), None)
        if rule is None:
            continue
        axes = []
        for axis in "xyz":
            if not getattr(constraint, f"use_limit_lin_{axis}"):
                continue
            low, high = f"limit_lin_{axis}_lower", f"limit_lin_{axis}_upper"
            before = (getattr(constraint, low), getattr(constraint, high))
            after = narrowed_limits(*before, rule.linear_scale)
            if after == before:
                continue
            setattr(constraint, low, after[0])
            setattr(constraint, high, after[1])
            axes.append({"axis": axis, "before": list(before), "after": list(after)})
        if axes:
            report["changes"].append({"joint": joint.name, "joint_prefix": rule.joint_prefix, "axes": axes})
    report["changed_joints"] = len(report["changes"])
    return report
