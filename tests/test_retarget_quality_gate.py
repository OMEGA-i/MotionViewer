from motionviewer.blender.retarget.quality_gate import evaluate_quality_report


def _passing_report() -> dict:
    return {
        "bone_coverage": {"mapped": 22, "expected": 22, "complete": True},
        "rig_preflight": {"valid": True, "errors": []},
        "target_height_m": 1.8,
        "mean_joint_error_m": 0.05,
        "p95_joint_error_m": 0.10,
        "maximum_bone_length_drift_relative": 5e-6,
        "maximum_quaternion_norm_error": 4e-8,
        "maximum_contact_drift_m": {"left_foot": 0.004, "right_foot": 0.0045},
        "maximum_sole_anchor_drift_m": {"left_foot": 0.004, "right_foot": 0.0045},
        "maximum_sole_tilt_degrees": {"left_foot": 1.0, "right_foot": 1.5},
        "maximum_foot_heading_error_degrees": {"left_foot": 2.0, "right_foot": 2.5},
        "maximum_ankle_orientation_error_degrees": {"left_foot": 4.0, "right_foot": 5.0},
        "maximum_segment_error_degrees": 8.0,
        "maximum_ankle_endpoint_error_m": 0.04,
        "minimum_profiled_sole_z_m": -0.002,
        "joint_limit_violations": [],
        "angular_velocity_violations": [],
    }


def test_quality_gate_approves_a_complete_mixamo_audit() -> None:
    result = evaluate_quality_report(_passing_report())
    assert result["passed"] is True
    assert result["failures"] == []
    assert result["thresholds"]["maximum_contact_drift_m"] == 0.005


def test_quality_gate_reports_each_failed_motion_invariant() -> None:
    report = _passing_report()
    report["p95_joint_error_m"] = 0.2
    report["maximum_contact_drift_m"]["left_foot"] = 0.03
    report["maximum_sole_anchor_drift_m"]["left_foot"] = 0.03
    report["maximum_sole_tilt_degrees"]["left_foot"] = 4.0
    report["maximum_foot_heading_error_degrees"]["left_foot"] = 6.0
    report["maximum_ankle_orientation_error_degrees"]["left_foot"] = 13.0
    report["maximum_segment_error_degrees"] = 13.0
    report["maximum_ankle_endpoint_error_m"] = 0.1
    report["minimum_profiled_sole_z_m"] = -0.01
    report["joint_limit_violations"] = [{"bone": "LeftLeg", "frame": 4}]
    report["angular_velocity_violations"] = [{"bone": "Hips", "frame": 8}]

    result = evaluate_quality_report(report)

    assert result["passed"] is False
    assert {failure["metric"] for failure in result["failures"]} == {
        "p95_joint_error_m",
        "maximum_contact_drift_m",
        "maximum_sole_anchor_drift_m",
        "maximum_sole_tilt_degrees",
        "maximum_foot_heading_error_degrees",
        "maximum_ankle_orientation_error_degrees",
        "maximum_segment_error_degrees",
        "maximum_ankle_endpoint_error_m",
        "minimum_profiled_sole_z_m",
        "joint_limit_violations",
        "angular_velocity_violations",
    }
