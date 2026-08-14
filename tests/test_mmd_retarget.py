from __future__ import annotations

import numpy as np
import pytest

from motionviewer.blender.retarget.mmd import detect_mmd_family, inspect_mmd_rig, mute_mmd_ik
from motionviewer.blender.retarget.twist import swing_twist_decompose
from motionviewer.core.smplx_fk import (
    SMPLX_BODY22_NAMES,
    build_lookat_motion,
    recover_rest_offsets,
    rodrigues,
    source_to_blender,
)


class _Bone:
    def __init__(self, name: str, parent: _Bone | None = None) -> None:
        self.name = name
        self.parent = parent


class _Armature:
    def __init__(self, names: list[str]) -> None:
        self.data = type("ArmatureData", (), {"bones": [_Bone(name) for name in names]})()


YOIMIYA_CORE = [
    "腰",
    "上半身",
    "上半身3",
    "上半身2",
    "首",
    "頭",
    "左肩P",
    "左肩",
    "左肩C",
    "左腕",
    "左腕捩",
    "左腕捩1",
    "左ひじ",
    "左手捩",
    "左手首",
    "右肩P",
    "右肩",
    "右肩C",
    "右腕",
    "右腕捩",
    "右ひじ",
    "右手捩",
    "右手首",
    "左足",
    "左ひざ",
    "左足首",
    "左つま先",
    "右足",
    "右ひざ",
    "右足首",
    "右つま先",
]


def test_mmd_family_detects_yoimiya_names() -> None:
    assert detect_mmd_family(set(YOIMIYA_CORE))
    assert not detect_mmd_family({"Hips", "LeftArm", "LeftForeArm"})


def test_mmd_mapping_covers_body22_and_skips_cancel_bones() -> None:
    inspection = inspect_mmd_rig(_Armature(YOIMIYA_CORE))
    assert inspection.valid, inspection.errors
    # The pelvis carries turning and the collar carries shoulder deformation;
    # both are driven. Toes are not: they hold no mesh weight on this family.
    assert inspection.smplx_map["pelvis"] == "腰"
    assert inspection.smplx_map["left_collar"] == "左肩"
    assert "left_foot" not in inspection.smplx_map
    assert inspection.smplx_map["left_shoulder"] == "左腕"
    assert inspection.smplx_map["left_elbow"] == "左ひじ"
    assert inspection.smplx_map["left_wrist"] == "左手首"
    assert "左肩C" not in inspection.smplx_map.values()
    assert "左肩P" not in inspection.smplx_map.values()
    twist_swings = {pair.swing_bone for pair in inspection.twist_pairs}
    assert twist_swings == {"左腕", "左ひじ", "右腕", "右ひじ"}
    # Arms disagree with SMPL-X rest by 23-51 deg (A-pose bind vs T-pose rest)
    # and must copy the source aim; everything else keeps its own rest.
    assert inspection.transfer_modes["左腕"] == "absolute"
    assert inspection.transfer_modes["左ひじ"] == "absolute"
    assert inspection.transfer_modes["左手首"] == "absolute"
    assert inspection.transfer_modes["腰"] == "relative"
    assert inspection.transfer_modes["左肩"] == "relative"
    assert inspection.transfer_modes["頭"] == "relative"
    assert inspection.transfer_modes["左足首"] == "relative"


def test_swing_twist_roundtrip_preserves_rotation() -> None:
    axis = np.array((0.0, 1.0, 0.0))
    # 40 deg swing around X and 25 deg twist around Y.
    swing_src = np.array((np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.0, 0.0))
    twist_src = np.array((np.cos(np.deg2rad(12.5)), 0.0, np.sin(np.deg2rad(12.5)), 0.0))
    from motionviewer.blender.retarget.twist import quaternion_multiply

    combined = quaternion_multiply(swing_src, twist_src)
    swing, twist = swing_twist_decompose(combined, axis)
    rebuilt = quaternion_multiply(swing, twist)
    if np.dot(rebuilt, combined) < 0:
        rebuilt = -rebuilt
    assert rebuilt == pytest.approx(combined, abs=1e-6)
    assert abs(float(np.dot(twist[1:], axis))) == pytest.approx(float(np.linalg.norm(twist[1:])), abs=1e-6)


def test_lookat_identity_keeps_tpose_arm_direction() -> None:
    rng = np.random.default_rng(0)
    rest = np.zeros((8, 22, 3), dtype=np.float64)
    rest[:, 16] = (0.2, 1.3, 0.0)  # left_shoulder
    rest[:, 18] = (0.45, 1.3, 0.0)  # left_elbow, T-pose +X
    rest[:, 20] = (0.7, 1.3, 0.0)
    rest[:, 0] = (0.0, 1.0, 0.0)
    rest[:, 3] = (0.0, 1.15, 0.0)
    rest[:, 6] = (0.0, 1.25, 0.0)
    rest[:, 9] = (0.0, 1.35, 0.0)
    rest[:, 12] = (0.0, 1.45, 0.0)
    rest[:, 15] = (0.0, 1.6, 0.0)
    noise = rng.normal(0.0, 1e-4, rest.shape)
    motion = build_lookat_motion(rest + noise, np.zeros((8, 3)), np.zeros((8, 63)), np.zeros((8, 3)))
    left_arm = motion.rest_frames[SMPLX_BODY22_NAMES.index("left_shoulder")]
    direction = left_arm[:3, 1]
    blender_dir = source_to_blender(np.array((1.0, 0.0, 0.0)))
    blender_dir = blender_dir / np.linalg.norm(blender_dir)
    assert float(np.dot(direction, blender_dir)) > 0.95


def test_mute_mmd_ik_keeps_append_copies_and_mutes_cancel() -> None:
    class _Constraint:
        def __init__(self, type_: str, name: str) -> None:
            self.type = type_
            self.name = name
            self.mute = False

    class _Mmd:
        def __init__(self, influence: float) -> None:
            self.additional_transform_influence = influence

    class _PoseBone:
        def __init__(self, name: str, constraints: list[_Constraint], influence: float = 0.0) -> None:
            self.name = name
            self.constraints = constraints
            self.mmd_bone = _Mmd(influence)

    class _Armature:
        def __init__(self, bones: list[_PoseBone]) -> None:
            self.pose = type("Pose", (), {"bones": bones})()

    ik = _Constraint("IK", "IK")
    damped = _Constraint("DAMPED_TRACK", "mmd_ik_limit")
    cancel = _Constraint("TRANSFORM", "mmd_additional_rotation")
    append = _Constraint("TRANSFORM", "mmd_additional_rotation")
    copy_rot = _Constraint("COPY_ROTATION", "mmd_copy")
    armature = _Armature(
        [
            _PoseBone("左ひざ", [ik, damped]),
            _PoseBone("左肩C", [cancel], influence=-1.0),
            _PoseBone("左足D", [append], influence=1.0),
            _PoseBone("左腕捩1", [copy_rot], influence=1.0),
        ]
    )
    muted = mute_mmd_ik(armature)
    assert ik.mute and damped.mute and cancel.mute
    assert not append.mute and not copy_rot.mute
    assert any("左肩C" in item for item in muted)
    assert all("左足D" not in item for item in muted)


def _frame_from_y(direction: tuple[float, float, float], roll: tuple[float, float, float]) -> np.ndarray:
    y_axis = np.asarray(direction, dtype=np.float64)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.asarray(roll, dtype=np.float64)
    x_axis = x_axis - y_axis * float(np.dot(x_axis, y_axis))
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=1)


def test_absolute_calibration_copies_source_aim_across_a_pose_gap() -> None:
    from motionviewer.blender.retarget.mmd_solve import absolute_calibration

    source_rest = _frame_from_y((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # SMPL-X T-pose arm
    target_rest = _frame_from_y((0.7071, 0.0, -0.7071), (0.0, 0.0, 1.0))  # MMD A-pose bind
    calibration = absolute_calibration(source_rest, target_rest)

    # Any source pose lands the bone exactly on the source aim, so a 47 deg
    # rest gap is never applied twice.
    for angle in (0.0, 0.4, -1.1):
        rotation = _frame_from_y((np.cos(angle), 0.0, np.sin(angle)), (0.0, 1.0, 0.0))
        source = rotation @ source_rest
        world = source @ calibration
        assert world[:, 1] == pytest.approx(source[:, 1], abs=1e-12)


def test_absolute_calibration_preserves_source_twist() -> None:
    """The regression that made hands look stiff: aim-only copies drop twist."""
    from motionviewer.blender.retarget.mmd_solve import absolute_calibration, roll_matrix

    source_rest = _frame_from_y((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    target_rest = _frame_from_y((0.7071, 0.0, -0.7071), (0.0, 0.0, 1.0))
    calibration = absolute_calibration(source_rest, target_rest)

    neutral = source_rest @ calibration
    for angle in (0.3, -0.9, 2.2):
        # Rotating the source about its own bone axis must rotate the target
        # about its bone axis by the same angle, not leave it unchanged.
        twisted = (source_rest @ roll_matrix(angle)) @ calibration
        relative = neutral.T @ twisted
        assert relative == pytest.approx(roll_matrix(angle), abs=1e-12)


def test_relative_calibration_applies_source_rotation_to_target_rest() -> None:
    from motionviewer.blender.retarget.mmd_solve import relative_calibration

    source_rest = _frame_from_y((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    target_rest = _frame_from_y((0.0, -0.678, 0.735), (1.0, 0.0, 0.0))  # 腰 display axis
    calibration = relative_calibration(source_rest, target_rest)
    rotation = (
        _frame_from_y((0.3, 0.4, 0.87), (1.0, 0.0, 0.0)) @ _frame_from_y((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)).T
    )
    world = (rotation @ source_rest) @ calibration
    # W == R_g @ target_rest: the character keeps its own rest orientation.
    assert world == pytest.approx(rotation @ target_rest, abs=1e-12)


def test_identity_rest_delta_copies_lookat_world() -> None:
    from motionviewer.blender.retarget.solver import RetargetDefinition, solve_retarget

    rest_local = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    rest_delta = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    source = np.zeros((1, 2, 4, 4), dtype=np.float64)
    source[0, 0] = np.eye(4)
    source[0, 1, :3, :3] = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    source[0, 1, 3, 3] = 1.0
    definition = RetargetDefinition(
        source_names=("parent", "child"),
        target_names=("腰", "左腕"),
        parent_indices=np.array((-1, 0), dtype=np.int32),
        rest_delta=rest_delta,
        target_rest_local=rest_local,
    )
    result = solve_retarget(definition, source, np.zeros((1, 3)), mode="direct")
    assert result.target_matrices[0, 1, :3, :3] == pytest.approx(source[0, 1, :3, :3], abs=1e-6)


def test_a_pose_rest_delta_doubles_arm_drop() -> None:
    from motionviewer.blender.retarget.solver import RetargetDefinition, solve_retarget

    def frame_from_y(direction: tuple[float, float, float]) -> np.ndarray:
        y_axis = np.asarray(direction, dtype=np.float64)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.array((0.0, 0.0, 1.0))
        x_axis = x_axis - y_axis * float(np.dot(x_axis, y_axis))
        x_axis = x_axis / np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis)
        matrix = np.eye(4)
        matrix[:3, 0] = x_axis
        matrix[:3, 1] = y_axis
        matrix[:3, 2] = z_axis
        return matrix

    tpose = frame_from_y((1.0, 0.0, 0.0))
    apose = frame_from_y((0.7071, 0.0, -0.7071))
    rest_delta = np.linalg.inv(tpose) @ apose
    definition = RetargetDefinition(
        source_names=("left_shoulder",),
        target_names=("左腕",),
        parent_indices=np.array((-1,), dtype=np.int32),
        rest_delta=rest_delta[None, ...],
        target_rest_local=apose[None, ...],
    )
    result = solve_retarget(definition, apose[None, None, ...], np.zeros((1, 3)), mode="direct")
    world_y = result.target_matrices[0, 0, :3, 1]
    bind_y = apose[:3, 1]
    assert float(np.dot(world_y / np.linalg.norm(world_y), bind_y)) < 0.75


def _humanoid_rest() -> np.ndarray:
    """A plausible body-22 rest skeleton in SMPL-X source space (Y up)."""
    rest = np.zeros((22, 3), dtype=np.float64)
    rest[0] = (0.0, 0.95, 0.0)  # pelvis
    rest[1], rest[2] = (0.08, 0.9, 0.0), (-0.08, 0.9, 0.0)  # hips
    rest[3] = (0.0, 1.05, 0.0)  # spine1
    rest[4], rest[5] = (0.09, 0.52, 0.0), (-0.09, 0.52, 0.0)  # knees
    rest[6] = (0.0, 1.15, 0.0)  # spine2
    rest[7], rest[8] = (0.09, 0.1, 0.0), (-0.09, 0.1, 0.0)  # ankles
    rest[9] = (0.0, 1.25, 0.0)  # spine3
    rest[10], rest[11] = (0.09, 0.05, 0.12), (-0.09, 0.05, 0.12)  # toes
    rest[12] = (0.0, 1.42, 0.0)  # neck
    rest[13], rest[14] = (0.07, 1.34, 0.0), (-0.07, 1.34, 0.0)  # collars
    rest[15] = (0.0, 1.55, 0.02)  # head, slightly forward of the neck
    rest[16], rest[17] = (0.18, 1.32, 0.0), (-0.18, 1.32, 0.0)  # shoulders
    rest[18], rest[19] = (0.45, 1.29, 0.0), (-0.45, 1.29, 0.0)  # elbows
    rest[20], rest[21] = (0.70, 1.27, 0.0), (-0.70, 1.27, 0.0)  # wrists
    return rest


def _forward_kinematics(rest: np.ndarray, global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    from motionviewer.core.smplx_fk import SMPLX_BODY22_PARENTS, global_rotations

    rotations = global_rotations(global_orient, body_pose)
    joints = np.zeros((len(rotations), 22, 3), dtype=np.float64)
    for frame in range(len(rotations)):
        joints[frame, 0] = rest[0]
        for index, parent in enumerate(SMPLX_BODY22_PARENTS):
            if parent < 0:
                continue
            joints[frame, index] = joints[frame, parent] + rotations[frame, parent] @ (
                rest[index] - rest[parent]
            )
    return joints


def test_lookat_frame_is_rest_frame_carried_by_its_own_rotation() -> None:
    """``S(t, b) == R_g[b] @ S_rest(b)`` for every bone, end effectors included.

    Reusing the incoming edge for wrist/head aim ties them to their parent, so
    wrist flexion and head nod silently vanish. This invariant is also what
    lets one constant calibration carry the whole transfer.
    """
    from motionviewer.core.smplx_fk import blender_rotation, global_rotations

    rng = np.random.default_rng(7)
    rest = _humanoid_rest()
    global_orient = rng.normal(0.0, 0.5, (5, 3))
    body_pose = rng.normal(0.0, 0.5, (5, 21, 3))
    joints = _forward_kinematics(rest, global_orient, body_pose)
    motion = build_lookat_motion(joints, global_orient, body_pose, np.zeros((5, 3)))
    rotations = global_rotations(global_orient, body_pose)

    for frame in range(len(joints)):
        for index, name in enumerate(SMPLX_BODY22_NAMES):
            expected = blender_rotation(rotations[frame, index]) @ motion.rest_frames[index, :3, :3]
            assert motion.posed_frames[frame, index, :3, :3] == pytest.approx(expected, abs=1e-9), name


def test_wrist_aim_follows_its_own_rotation_not_the_forearm() -> None:
    from motionviewer.core.smplx_fk import SMPLX_BODY22_NAMES as NAMES

    rest = _humanoid_rest()
    wrist = NAMES.index("left_wrist")
    body_pose = np.zeros((2, 21, 3))
    # Frame 1 bends the left wrist 50 deg; the elbow never moves.
    body_pose[1, NAMES.index("left_wrist") - 1] = (0.0, 0.0, np.deg2rad(50.0))
    joints = _forward_kinematics(rest, np.zeros((2, 3)), body_pose)
    motion = build_lookat_motion(joints, np.zeros((2, 3)), body_pose, np.zeros((2, 3)))

    straight = motion.posed_frames[0, wrist, :3, 1]
    flexed = motion.posed_frames[1, wrist, :3, 1]
    angle = np.degrees(np.arccos(np.clip(float(np.dot(straight, flexed)), -1.0, 1.0)))
    assert angle == pytest.approx(50.0, abs=1e-6)


def test_twist_channel_reproduces_the_full_rotation_below_it() -> None:
    """``rest_local @ B_twist == twist @ rest_local``.

    Handing a 捩 bone the raw twist quaternion ignores its own rest rotation,
    which shifts everything below it. Splitting must leave the chain identical
    to putting the whole rotation on the swing bone.
    """
    from motionviewer.blender.retarget.mmd_solve import (
        MmdChannel,
        MmdRetargetPlan,
        absolute_calibration,
        solve_mmd_retarget,
    )

    arm_rest = _frame_from_y((0.766, 0.019, -0.643), (0.0, 0.0, 1.0))
    twist_rest = _frame_from_y((0.762, 0.019, -0.647), (0.1, 0.2, 1.0))
    elbow_rest = _frame_from_y((0.762, -0.046, -0.646), (0.0, 0.0, 1.0))
    source_arm = _frame_from_y((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    source_elbow = _frame_from_y((0.99, 0.06, 0.11), (0.0, 0.0, 1.0))

    def plan_for(*, split: bool) -> MmdRetargetPlan:
        return MmdRetargetPlan(
            channels=(
                MmdChannel(
                    name="左腕",
                    parent=-1,
                    rest_local=arm_rest,
                    mode="absolute",
                    source="left_shoulder",
                    source_index=0,
                    calibration=absolute_calibration(source_arm, arm_rest),
                    twist_partner=1 if split else -1,
                ),
                MmdChannel(
                    name="左腕捩",
                    parent=0,
                    rest_local=arm_rest.T @ twist_rest,
                    mode="twist" if split else "passthrough",
                    swing_of=0 if split else -1,
                ),
                MmdChannel(
                    name="左ひじ",
                    parent=1,
                    rest_local=twist_rest.T @ elbow_rest,
                    mode="absolute",
                    source="left_elbow",
                    source_index=1,
                    calibration=absolute_calibration(source_elbow, elbow_rest),
                ),
            ),
            source_names=("left_shoulder", "left_elbow"),
            target_rest_global=np.stack((arm_rest, twist_rest, elbow_rest)),
            source_rest_global=np.stack((source_arm, np.eye(3), source_elbow)),
        )

    rng = np.random.default_rng(3)
    frames = np.zeros((6, 2, 4, 4), dtype=np.float64)
    for frame in range(6):
        for source in range(2):
            axis = rng.normal(0.0, 1.0, 3)
            axis = axis / np.linalg.norm(axis)
            rotation = rodrigues(axis * rng.uniform(-2.0, 2.0))
            frames[frame, source, :3, :3] = rotation @ (source_arm if source == 0 else source_elbow)
            frames[frame, source, 3, 3] = 1.0

    split = solve_mmd_retarget(plan_for(split=True), frames, np.zeros((6, 3)))
    whole = solve_mmd_retarget(plan_for(split=False), frames, np.zeros((6, 3)))

    # The twist bone's own output and everything below it are unchanged.
    assert split.world_rotations[:, 1] == pytest.approx(whole.world_rotations[:, 1], abs=1e-9)
    assert split.world_rotations[:, 2] == pytest.approx(whole.world_rotations[:, 2], abs=1e-9)
    # But the swing bone no longer carries the twist, so 腕捩1/2/3 can spread it.
    assert not np.allclose(split.world_rotations[:, 0], whole.world_rotations[:, 0], atol=1e-6)
    for frame in range(6):
        swing_local = split.local_quaternions_wxyz[frame, 0]
        assert abs(float(swing_local[2])) < 1e-9  # no rotation about the bone axis


def test_absolute_arm_never_folds_below_the_a_pose_bind() -> None:
    """The reported symptom: a rest-delta transfer buries a hanging arm in the torso."""
    from motionviewer.blender.retarget.mmd_solve import absolute_calibration, relative_calibration

    drop = np.deg2rad(60.0)
    source_rest = _frame_from_y((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # T-pose
    target_rest = _frame_from_y((0.766, 0.019, -0.643), (0.0, 0.0, 1.0))  # A-pose, 40 deg down
    hanging = _frame_from_y((float(np.cos(drop)), 0.0, -float(np.sin(drop))), (0.0, 0.0, 1.0))
    rotation = hanging @ source_rest.T

    absolute = (rotation @ source_rest) @ absolute_calibration(source_rest, target_rest)
    relative = (rotation @ source_rest) @ relative_calibration(source_rest, target_rest)

    def drop_degrees(matrix: np.ndarray) -> float:
        """Angle of the bone axis below the outward horizontal, over the full turn."""
        return float(np.degrees(np.arctan2(-matrix[2, 1], matrix[0, 1])))

    assert drop_degrees(absolute) == pytest.approx(60.0, abs=1e-6)
    # Relative stacks the 40 deg bind on top of the 60 deg source drop, pushing
    # the arm past vertical so it points back across the midline into the torso.
    assert drop_degrees(relative) > 95.0
    assert absolute[0, 1] > 0.0
    assert relative[0, 1] < 0.0


def test_rest_offset_recovers_constant_parent_child() -> None:
    offsets = recover_rest_offsets(
        np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 20], dtype=np.float64).repeat(
            4, axis=0
        ),
        np.zeros((4, 3)),
        np.zeros((4, 63)),
    )
    assert offsets.shape == (22, 3)
    identity = rodrigues(np.zeros(3))
    assert identity == pytest.approx(np.eye(3))


def test_abduction_rotates_arm_rigidly_and_fades_when_extended() -> None:
    """Clearing the costume must not alter any angle inside the arm."""
    from motionviewer.blender.retarget.mmd_solve import abduct_source_arm

    def frame(direction: tuple[float, float, float]) -> np.ndarray:
        # Pick a roll reference that is not parallel to the bone; that is the
        # same degeneracy the production rest frames are built to avoid.
        axis = np.asarray(direction, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
        matrix = np.eye(4)
        matrix[:3, :3] = _frame_from_y(direction, tuple(float(v) for v in reference))
        return matrix

    # Hanging arm: upper arm down, forearm bent forward, collar pointing out.
    frames = np.zeros((2, 4, 4, 4), dtype=np.float64)
    # Exactly perpendicular to the collar, so the fade weight is 1 and the
    # applied angle is the full setting.
    frames[:, 0] = frame((0.0, 0.0, -1.0))  # shoulder, hanging
    frames[:, 1] = frame((0.0, -0.7, -0.7))  # elbow
    frames[:, 2] = frame((0.0, -0.9, -0.4))  # wrist
    frames[:, 3] = frame((1.0, 0.0, 0.0))  # collar, outward
    # Second frame extends the arm straight out along the collar direction.
    frames[1, 0] = frame((1.0, 0.0, 0.0))

    result = abduct_source_arm(frames, (0, 1, 2), 0, 3, degrees=12.0)

    hanging_before = [frames[0, index, :3, :3] for index in range(3)]
    hanging_after = [result[0, index, :3, :3] for index in range(3)]
    # Every relative angle inside the chain is preserved: one rigid rotation.
    for first, second in ((0, 1), (1, 2), (0, 2)):
        before = hanging_before[first].T @ hanging_before[second]
        after = hanging_after[first].T @ hanging_after[second]
        assert after == pytest.approx(before, abs=1e-12)
    # And the arm actually moved outward.
    moved = float(
        np.degrees(
            np.arccos(np.clip(float(np.dot(hanging_before[0][:, 1], hanging_after[0][:, 1])), -1.0, 1.0))
        )
    )
    assert moved == pytest.approx(12.0, abs=1e-6)
    # An arm already pointing outward is left exactly alone.
    assert result[1, 0, :3, :3] == pytest.approx(frames[1, 0, :3, :3], abs=1e-12)


def test_collar_damping_only_scales_that_joint() -> None:
    from motionviewer.blender.retarget.mmd_solve import damp_source_local_rotation

    rest = np.zeros((2, 4, 4), dtype=np.float64)
    rest[0, :3, :3] = _frame_from_y((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))  # spine3
    rest[1, :3, :3] = _frame_from_y((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # collar

    frames = np.zeros((1, 2, 4, 4), dtype=np.float64)
    frames[0, 0, :3, :3] = rest[0, :3, :3]
    turn = rodrigues(np.array((0.0, 0.0, np.deg2rad(60.0))))
    frames[0, 1, :3, :3] = turn @ rest[1, :3, :3]

    damped = damp_source_local_rotation(frames, rest, 1, 0, factor=0.5, limit_degrees=90.0)
    carried = damped[0, 1, :3, :3] @ rest[1, :3, :3].T
    angle = float(np.degrees(np.arccos(np.clip((np.trace(carried) - 1.0) / 2.0, -1.0, 1.0))))
    assert angle == pytest.approx(30.0, abs=1e-6)
    # The parent is untouched.
    assert damped[0, 0, :3, :3] == pytest.approx(frames[0, 0, :3, :3], abs=1e-12)


def test_twist_smoothing_leaves_the_aim_untouched() -> None:
    from motionviewer.blender.retarget.mmd_solve import roll_matrix, smooth_source_twist

    rng = np.random.default_rng(11)
    base = _frame_from_y((0.6, 0.1, -0.8), (0.0, 0.0, 1.0))
    frames = np.zeros((24, 1, 4, 4), dtype=np.float64)
    for index in range(24):
        drift = rodrigues(np.array((0.0, 0.0, 0.02 * index)))
        jitter = roll_matrix(float(rng.normal(0.0, 0.5)))
        frames[index, 0, :3, :3] = drift @ base @ jitter

    smoothed = smooth_source_twist(frames, (0,), window=5)
    for index in range(24):
        assert smoothed[index, 0, :3, 1] == pytest.approx(frames[index, 0, :3, 1], abs=1e-9)

    def roll_steps(data: np.ndarray) -> float:
        angles = []
        for index in range(1, len(data)):
            previous = data[index - 1, 0, :3, :3]
            current = data[index, 0, :3, :3]
            relative = previous.T @ current
            angles.append(abs(float(np.arctan2(relative[0, 2], relative[0, 0]))))
        return float(np.max(angles))

    assert roll_steps(smoothed) < roll_steps(frames) * 0.6


def test_two_segment_upper_body_collapses_spine2() -> None:
    """Honkai rigs ship 上半身/上半身2 with no 上半身3.

    Transfers are global, so spine3's rotation on 上半身2 already contains
    spine1 and spine2. Refusing the rig would be wrong; silently dropping a
    mapped joint would not be.
    """
    two_segment = [name for name in YOIMIYA_CORE if name != "上半身3"]
    inspection = inspect_mmd_rig(_Armature(two_segment))
    assert inspection.valid, inspection.errors
    assert inspection.smplx_map["spine1"] == "上半身"
    assert inspection.smplx_map["spine3"] == "上半身2"
    assert "spine2" not in inspection.smplx_map
    # Every other joint is still required.
    assert inspection.smplx_map["pelvis"] == "腰"
    assert inspection.smplx_map["left_wrist"] == "左手首"


def test_missing_spine2_still_fails_when_the_rig_has_three_segments() -> None:
    """The excuse is the rig's shape, not a blanket exemption."""
    from motionviewer.blender.retarget.mmd import _CANONICAL_TO_MMD_ALIASES

    # A rig with 上半身3 present must map spine2; drop 上半身2 so `chest` is the
    # one that goes missing and the mapping is genuinely incomplete.
    assert _CANONICAL_TO_MMD_ALIASES["spine-1"] == ("上半身3",)
    broken = [name for name in YOIMIYA_CORE if name != "上半身2"]
    inspection = inspect_mmd_rig(_Armature(broken))
    assert not inspection.valid
    assert any("上半身2" in message or "chest" in message for message in inspection.errors)


def test_face_materials_are_found_by_which_ones_are_unlit() -> None:
    """The face group is read off the model, not guessed from names."""
    from motionviewer.blender.mmd_toon import face_base_images

    # 目/白目/齒 carry no toon ramp and sample the face sheet; 面/口/睫 share it.
    inventory = [
        ("面.png", False),  # eyes: unlit
        ("面.png", True),  # face skin
        ("面.png", True),  # mouth
        ("肌.png", True),  # body skin
        ("衣.png", True),  # clothes
        ("髮.png", True),  # hair
    ]
    assert face_base_images(inventory) == {"面.png"}

    # A rig that shades everything has no unlit hint, so no face override.
    assert face_base_images([("面.png", True), ("衣.png", True)]) == set()
