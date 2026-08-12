"""Deterministic acceptance gate for Mixamo retarget audit reports."""

from __future__ import annotations

from typing import Any


def evaluate_quality_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return auditable pass/fail evidence for one asset-motion report."""
    height = max(float(report.get("target_height_m", 0.0)), 1e-8)
    thresholds = {
        "maximum_bone_length_drift_relative": 1e-4,
        "maximum_quaternion_norm_error": 1e-4,
        "mean_joint_error_m": 0.04 * height,
        "p95_joint_error_m": 0.08 * height,
        "maximum_contact_drift_m": 0.005,
        "maximum_sole_anchor_drift_m": 0.005,
        "maximum_sole_tilt_degrees": 3.0,
        "maximum_foot_heading_error_degrees": 5.0,
        "maximum_ankle_orientation_error_degrees": 12.0,
        "maximum_segment_error_degrees": 12.0,
        "maximum_ankle_endpoint_error_m": 0.04 * height,
        "minimum_profiled_sole_z_m": -0.005,
    }
    failures: list[dict[str, Any]] = []

    def fail(metric: str, actual: Any, limit: Any) -> None:
        failures.append({"metric": metric, "actual": actual, "limit": limit})

    coverage = report.get("bone_coverage", {})
    if coverage.get("mapped") != 22 or coverage.get("expected") != 22 or not coverage.get("complete"):
        fail("bone_coverage", coverage, {"mapped": 22, "expected": 22, "complete": True})
    preflight = report.get("rig_preflight", {})
    if not preflight.get("valid"):
        fail("rig_preflight", preflight, {"valid": True})

    for metric in (
        "maximum_bone_length_drift_relative",
        "maximum_quaternion_norm_error",
        "mean_joint_error_m",
        "p95_joint_error_m",
        "maximum_contact_drift_m",
        "maximum_sole_anchor_drift_m",
        "maximum_sole_tilt_degrees",
        "maximum_foot_heading_error_degrees",
        "maximum_ankle_orientation_error_degrees",
        "maximum_segment_error_degrees",
        "maximum_ankle_endpoint_error_m",
    ):
        raw = report.get(metric)
        if raw is None:
            continue
        actual = (
            max((float(value) for value in raw.values()), default=0.0)
            if isinstance(raw, dict)
            else float(raw)
        )
        if actual > thresholds[metric]:
            fail(metric, actual, thresholds[metric])

    roundtrip = report.get("fbx_roundtrip_error")
    if isinstance(roundtrip, dict) and roundtrip.get("passed") is False:
        fail("fbx_roundtrip_error", roundtrip, {"passed": True})

    sole_z = float(report.get("minimum_profiled_sole_z_m", float("-inf")))
    if sole_z < thresholds["minimum_profiled_sole_z_m"]:
        fail("minimum_profiled_sole_z_m", sole_z, thresholds["minimum_profiled_sole_z_m"])
    for metric in ("joint_limit_violations", "angular_velocity_violations"):
        violations = list(report.get(metric, ()))
        if violations:
            fail(metric, violations, [])

    return {"passed": not failures, "thresholds": thresholds, "failures": failures}
