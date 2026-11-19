"""The SOMA-77 to MMD name mapping, fingers included.

These are tables, so the risk is not arithmetic but silent omission: a finger that maps
to nothing simply does not move, and on a full-body render that is easy to miss. The
tests therefore assert the *counts* as well as the individual entries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from motionviewer.blender.retarget.mmd import _SKIP_DRIVE, _TRANSFER_MODES  # noqa: E402
from motionviewer.blender.retarget.soma_mmd import (  # noqa: E402
    CANONICAL_TO_MMD_ALIASES,
    FINGER_MMD_ALIASES,
    SOMA_BODY_TO_CANONICAL,
    SOMA_FINGER_TO_CANONICAL,
    SOMA_TO_CANONICAL,
    TRANSFER_MODES,
    build_soma_mmd_mapping,
)
from motionviewer.core.soma_fk import SOMA77_NAMES  # noqa: E402


def test_every_mapped_source_is_a_real_soma_joint():
    """A typo here would map a joint that never arrives, and move nothing."""
    unknown = [name for name in SOMA_TO_CANONICAL if name not in SOMA77_NAMES]
    assert unknown == []


def test_the_whole_body_is_mapped():
    """22 body joints, matching what the SMPL-X path drove."""
    assert len(SOMA_BODY_TO_CANONICAL) == 22
    for joint in ("Hips", "Chest", "Head", "LeftHand", "RightHand", "LeftFoot", "RightToeBase"):
        assert joint in SOMA_BODY_TO_CANONICAL


def test_all_ten_fingers_are_mapped_at_three_segments_each():
    assert len(SOMA_FINGER_TO_CANONICAL) == 30
    assert len(FINGER_MMD_ALIASES) == 30
    for side in ("Left", "Right"):
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            for segment in ((1, 2, 3) if finger == "Thumb" else (2, 3, 4)):
                assert f"{side}Hand{finger}{segment}" in SOMA_FINGER_TO_CANONICAL


def test_the_thumb_is_offset_by_one_because_mmd_numbers_from_the_metacarpal():
    assert FINGER_MMD_ALIASES[SOMA_TO_CANONICAL["LeftHandThumb1"]] == ("左親指０",)
    assert FINGER_MMD_ALIASES[SOMA_TO_CANONICAL["LeftHandThumb3"]] == ("左親指２",)
    # MMD's other fingers start at the knuckle, after SOMA's metacarpal.
    assert FINGER_MMD_ALIASES[SOMA_TO_CANONICAL["LeftHandIndex2"]] == ("左人指１",)


def test_non_thumb_metacarpals_are_implicit_and_distal_phalanges_are_mapped():
    for side, prefix in (("Left", "左"), ("Right", "右")):
        for finger, stem in (("Index", "人指"), ("Middle", "中指"), ("Ring", "薬指"), ("Pinky", "小指")):
            assert f"{side}Hand{finger}1" not in SOMA_TO_CANONICAL
            role = SOMA_TO_CANONICAL[f"{side}Hand{finger}4"]
            assert FINGER_MMD_ALIASES[role] == (f"{prefix}{stem}３",)


def test_finger_solve_keeps_metacarpal_motion_and_the_distal_curl():
    """Compare solved bone axes with FK phalanges, including an animated distal joint.

    Copying Index1/2/3 instead would aim at the palm/first/second segments and
    would drop Index4's independent motion. No character assets are needed.
    """
    from motionviewer.blender.retarget.mmd_solve import (
        MmdChannel,
        MmdRetargetPlan,
        absolute_calibration,
        solve_mmd_retarget,
    )
    from motionviewer.core.smplx_fk import rodrigues, source_to_blender
    from motionviewer.core.soma_fk import (
        carry_rest_frames,
        forward_kinematics,
        rest_lookat_frames,
        world_rotations,
    )

    bones = [SimpleNamespace(name=aliases[0]) for aliases in CANONICAL_TO_MMD_ALIASES.values()]
    mapping = build_soma_mmd_mapping(SimpleNamespace(data=SimpleNamespace(bones=bones)))
    assert not mapping.errors
    names = ("LeftHand", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4",
             "LeftHandIndexEnd")
    parents = np.arange(-1, 5)
    # A wrist-to-palm offset, a long metacarpal and three shorter phalanges.
    rest = np.zeros((6, 3))
    rest[:, 0] = np.cumsum([0., .038, .066, .038, .024, .025])
    local_angles = np.zeros((3, 6, 3))
    local_angles[1, :, 2] = np.radians([8., 15., 20., 35., 45., 0.])
    local_angles[2] = local_angles[1]
    local_angles[2, 4, 2] += np.radians(25.)  # only the distal phalanx changes
    world = world_rotations(rodrigues(local_angles), parents)
    joints = source_to_blender(forward_kinematics(rest, world, parents, np.zeros((3, 3))))
    # The generic hand frame would ask for Middle1, absent in this short fixture.
    frame_names = ("Wrist",) + names[1:]
    rest_frames = rest_lookat_frames(source_to_blender(rest), parents, frame_names)
    frames = carry_rest_frames(rest_frames, joints, world)
    targets = ("左人指１", "左人指２", "左人指３")
    sources = {target: source for source, target in mapping.smplx_to_fbx.items()}
    target_rest = np.repeat(rest_frames[2, :3, :3][None], 3, axis=0)
    channels = tuple(
        MmdChannel(
            name=target, parent=index - 1,
            rest_local=target_rest[index] if index == 0 else np.eye(3),
            mode="absolute", source=sources[target], source_index=names.index(sources[target]),
            calibration=absolute_calibration(rest_frames[names.index(sources[target]), :3, :3], target_rest[index]),
        )
        for index, target in enumerate(targets)
    )
    plan = MmdRetargetPlan(channels, names, target_rest, target_rest)
    solved = solve_mmd_retarget(plan, frames, np.zeros((3, 3)))
    phalanges = joints[:, 3:6] - joints[:, 2:5]
    phalanges /= np.linalg.norm(phalanges, axis=-1, keepdims=True)
    assert solved.world_rotations[..., :, 1] == pytest.approx(phalanges, abs=1e-12)
    assert solved.world_rotations[2, :2] == pytest.approx(solved.world_rotations[1, :2], abs=1e-12)
    assert not np.allclose(solved.world_rotations[2, 2], solved.world_rotations[1, 2])


def test_tip_and_face_joints_are_unmapped():
    """``*End``, the eyes and the jaw have no MMD counterpart on these rigs."""
    for name in SOMA77_NAMES:
        if name.endswith("End") or name in ("Jaw", "LeftEye", "RightEye", "Neck2"):
            assert name not in SOMA_TO_CANONICAL


def test_no_two_soma_joints_claim_the_same_mmd_bone():
    roles = list(SOMA_TO_CANONICAL.values())
    assert len(roles) == len(set(roles))
    targets = [CANONICAL_TO_MMD_ALIASES[role][0] for role in roles]
    assert len(targets) == len(set(targets))


def test_every_role_has_a_transfer_mode():
    missing = [
        role
        for role in set(SOMA_TO_CANONICAL.values())
        if role not in TRANSFER_MODES and role not in _SKIP_DRIVE
    ]
    assert missing == []


def test_fingers_use_the_absolute_transfer():
    """MMD binds hands in a curl and SOMA's rest hand is flat, so relative would
    add the source curl on top of the bind and over-close the fist."""
    for role in FINGER_MMD_ALIASES:
        assert TRANSFER_MODES[role] == "absolute"


def test_the_body_transfer_modes_are_inherited_unchanged():
    """The body solve must be identical to the SMPL-X path, or this is not a fair swap."""
    for role, mode in _TRANSFER_MODES.items():
        assert TRANSFER_MODES[role] == mode


def test_extending_the_tables_did_not_disturb_the_existing_ones():
    for role, aliases in CANONICAL_TO_MMD_ALIASES.items():
        if role not in FINGER_MMD_ALIASES:
            assert isinstance(aliases, tuple) and aliases


def test_two_segment_native_torso_retains_intermediate_source_rotation():
    """A Honkai torso omits Spine2's target but its motion still reaches Chest."""
    from motionviewer.blender.retarget.mmd_solve import MmdChannel, MmdRetargetPlan, solve_mmd_retarget
    from motionviewer.core.smplx_fk import rodrigues

    names = [aliases[0] for aliases in CANONICAL_TO_MMD_ALIASES.values() if aliases[0] != "上半身3"]
    mapping = build_soma_mmd_mapping(SimpleNamespace(data=SimpleNamespace(
        bones=[SimpleNamespace(name=name) for name in names],
    )))
    assert not mapping.errors
    assert "Spine2" not in mapping.smplx_to_fbx
    assert mapping.smplx_to_fbx["Spine1"] == "上半身"
    assert mapping.smplx_to_fbx["Chest"] == "上半身2"
    assert mapping.driven_fingers == 30

    # Independent, noncommuting source rotations: omitting the middle rotation
    # changes the downstream chest. The two target bones must retain all three.
    a, b, c = rodrigues(np.radians([[11., 0., 0.], [0., 23., 0.], [0., 0., -17.]]))
    frames = np.repeat(np.eye(4)[None, None], 3, axis=1)
    frames[0, :, :3, :3] = [a, a @ b, a @ b @ c]
    channels = (
        MmdChannel("上半身", -1, np.eye(3), "relative", "Spine1", 0),
        MmdChannel("上半身2", 0, np.eye(3), "relative", "Chest", 2),
    )
    plan = MmdRetargetPlan(channels, ("Spine1", "Spine2", "Chest"),
                           np.repeat(np.eye(3)[None], 2, axis=0),
                           np.repeat(np.eye(3)[None], 2, axis=0))
    solved = solve_mmd_retarget(plan, frames, np.zeros((1, 3)))
    assert solved.world_rotations[0, 1] == pytest.approx(a @ b @ c, abs=1e-12)
    assert not np.allclose(solved.world_rotations[0, 1], a @ c)


def test_missing_native_chest_is_not_excused_by_optional_spine():
    names = [aliases[0] for aliases in CANONICAL_TO_MMD_ALIASES.values() if aliases[0] != "上半身2"]
    mapping = build_soma_mmd_mapping(SimpleNamespace(data=SimpleNamespace(
        bones=[SimpleNamespace(name=name) for name in names],
    )))
    assert any("Chest" in error or "chest" in error for error in mapping.errors)
    assert mapping.smplx_to_fbx["Spine2"] == "上半身3"
