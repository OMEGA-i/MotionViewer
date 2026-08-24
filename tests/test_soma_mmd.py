"""The SOMA-77 to MMD name mapping, fingers included.

These are tables, so the risk is not arithmetic but silent omission: a finger that maps
to nothing simply does not move, and on a full-body render that is easy to miss. The
tests therefore assert the *counts* as well as the individual entries.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
            for segment in (1, 2, 3):
                assert f"{side}Hand{finger}{segment}" in SOMA_FINGER_TO_CANONICAL


def test_the_thumb_is_offset_by_one_because_mmd_numbers_from_the_metacarpal():
    assert FINGER_MMD_ALIASES[SOMA_TO_CANONICAL["LeftHandThumb1"]] == ("左親指０",)
    assert FINGER_MMD_ALIASES[SOMA_TO_CANONICAL["LeftHandThumb3"]] == ("左親指２",)
    # Every other finger starts at 1.
    assert FINGER_MMD_ALIASES[SOMA_TO_CANONICAL["LeftHandIndex1"]] == ("左人指１",)


def test_the_fourth_finger_segment_is_deliberately_unmapped():
    """MMD has three driven finger bones; SOMA has four. The distal one is dropped."""
    for name in ("LeftHandIndex4", "RightHandMiddle4", "LeftHandPinky4"):
        assert name in SOMA77_NAMES
        assert name not in SOMA_TO_CANONICAL


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
