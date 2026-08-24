"""Map SOMA-77 onto an MMD rig, fingers included.

The body mapping is the existing one: SOMA's joints land on the same canonical roles
SMPL-X did, so ``_CANONICAL_TO_MMD_ALIASES`` and ``_TRANSFER_MODES`` are reused rather
than restated. What is new is the hands. SOMA carries 52 finger joints; body-22 carried
none, which is why ``hand_relax`` exists at all — it poses a static curl to stop a flat
bind reading as a mannequin. Yoimiya's rig has 40 finger bones waiting to be driven.

Two mapping details are worth stating because they are choices, not facts:

- **SOMA has one more finger segment than MMD.** SOMA runs
  ``Index1 -> Index2 -> Index3 -> Index4 -> IndexEnd``; MMD runs
  ``人指１ -> 人指２ -> 人指３ -> 人指先``. The first three joints are mapped and
  ``Index4`` is left undriven. Because transfers apply *global* rotations, an undriven
  leaf loses only its own segment — the distal joint stops curling, the rest is exact.
- **The thumb is offset by one.** MMD numbers the thumb from the metacarpal
  (``親指０``), so ``親指０/１/２`` take SOMA's ``Thumb1/2/3``; the other fingers have no
  metacarpal bone and start at ``１``.

Fingers use the ``absolute`` transfer, like the arm chain and for the same reason: an
MMD hand is bound in a relaxed curl while SOMA's rest hand is flat, so ``relative``
would add the source's curl *on top of* the bind and over-close the fist. ``absolute``
aims each MMD finger bone where the SOMA bone points, which is what makes a grip land
where the motion put it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mmd import (
    _CANONICAL_TO_MMD_ALIASES,
    _OPTIONAL_CANONICAL,
    _SKIP_DRIVE,
    _TRANSFER_MODES,
    _TWIST_ALIASES,
    REQUIRED_MMD_BONES,
    _normalize_name,
    _pick,
)
from .twist import TwistPair

# SOMA-77 joint -> canonical role, body only. ``Neck2`` and every ``*End`` joint are
# absent on purpose: they have no MMD counterpart, and a global-rotation transfer keeps
# their motion in the mapped descendant.
SOMA_BODY_TO_CANONICAL: dict[str, str] = {
    "Hips": "hips",
    "Spine1": "spine",
    "Spine2": "spine-1",
    "Chest": "chest",
    "Neck1": "neck",
    "Head": "head",
    "LeftShoulder": "shoulder.L",
    "LeftArm": "upper_arm.L",
    "LeftForeArm": "forearm.L",
    "LeftHand": "hand.L",
    "RightShoulder": "shoulder.R",
    "RightArm": "upper_arm.R",
    "RightForeArm": "forearm.R",
    "RightHand": "hand.R",
    "LeftLeg": "hip.L",
    "LeftShin": "thigh.L",
    "LeftFoot": "foot.L",
    "LeftToeBase": "toe.L",
    "RightLeg": "hip.R",
    "RightShin": "thigh.R",
    "RightFoot": "foot.R",
    "RightToeBase": "toe.R",
}

# (SOMA finger stem, MMD finger stem, MMD segment numbers). MMD's thumb starts at 0.
_FINGERS: tuple[tuple[str, str, tuple[str, str, str]], ...] = (
    ("Thumb", "親指", ("０", "１", "２")),
    ("Index", "人指", ("１", "２", "３")),
    ("Middle", "中指", ("１", "２", "３")),
    ("Ring", "薬指", ("１", "２", "３")),
    ("Pinky", "小指", ("１", "２", "３")),
)
_SIDES: tuple[tuple[str, str, str], ...] = (("Left", "左", "L"), ("Right", "右", "R"))


def _finger_tables() -> tuple[dict[str, str], dict[str, tuple[str, ...]], dict[str, str]]:
    """canonical role per SOMA joint, MMD aliases per role, transfer mode per role."""
    to_canonical: dict[str, str] = {}
    aliases: dict[str, tuple[str, ...]] = {}
    modes: dict[str, str] = {}
    for soma_side, mmd_side, suffix in _SIDES:
        for soma_stem, mmd_stem, numbers in _FINGERS:
            for segment, number in enumerate(numbers, start=1):
                role = f"{soma_stem.lower()}{segment}.{suffix}"
                to_canonical[f"{soma_side}Hand{soma_stem}{segment}"] = role
                aliases[role] = (f"{mmd_side}{mmd_stem}{number}",)
                modes[role] = "absolute"
    return to_canonical, aliases, modes


SOMA_FINGER_TO_CANONICAL, FINGER_MMD_ALIASES, FINGER_TRANSFER_MODES = _finger_tables()

SOMA_TO_CANONICAL: dict[str, str] = {**SOMA_BODY_TO_CANONICAL, **SOMA_FINGER_TO_CANONICAL}

# Fingers may be absent: a rig without them still retargets, it just keeps its bind
# pose hands, which is what the SMPL-X path always did.
OPTIONAL_CANONICAL: frozenset[str] = _OPTIONAL_CANONICAL | frozenset(FINGER_MMD_ALIASES)

CANONICAL_TO_MMD_ALIASES: dict[str, tuple[str, ...]] = {
    **_CANONICAL_TO_MMD_ALIASES,
    **FINGER_MMD_ALIASES,
}
TRANSFER_MODES: dict[str, str] = {**_TRANSFER_MODES, **FINGER_TRANSFER_MODES}


@dataclass(frozen=True)
class SomaRigMapping:
    """The shape ``animate_mmd_frames`` expects from a ``BoneMapping``."""

    smplx_to_fbx: dict[str, str]
    transfer_modes: dict[str, str]
    twist_pairs: tuple[TwistPair, ...]
    rig_family: str = "mmd"
    canonical_map: dict[str, str] | None = None
    driven_fingers: int = 0
    errors: tuple[str, ...] = ()


def build_soma_mmd_mapping(armature: Any, *, drive_fingers: bool = True) -> SomaRigMapping:
    """Resolve SOMA-77 onto this armature's bones."""
    bones = [bone.name for bone in armature.data.bones]
    available = {_normalize_name(name): name for name in bones}
    used: set[str] = set()
    canonical_map: dict[str, str] = {}
    errors: list[str] = []

    for canonical, aliases in CANONICAL_TO_MMD_ALIASES.items():
        if not drive_fingers and canonical in FINGER_MMD_ALIASES:
            continue
        chosen = _pick(aliases, available, used)
        if chosen is None:
            if canonical in OPTIONAL_CANONICAL:
                continue
            errors.append(f"missing MMD bone for {canonical}: {aliases[0]}")
            continue
        canonical_map[canonical] = chosen
        used.add(chosen)

    source_map = {
        source: canonical_map[canonical]
        for source, canonical in SOMA_TO_CANONICAL.items()
        if canonical in canonical_map and canonical not in _SKIP_DRIVE
    }

    # Body joints that must resolve. Fingers are allowed to be missing; the rest of the
    # body is not, and a silent gap there would show as a limb that does not move.
    skip = {source for source, canonical in SOMA_TO_CANONICAL.items() if canonical in _SKIP_DRIVE}
    missing = [source for source in SOMA_BODY_TO_CANONICAL if source not in source_map and source not in skip]
    if missing:
        errors.append("incomplete SOMA body mapping: " + ", ".join(missing))

    for required in REQUIRED_MMD_BONES:
        if _normalize_name(required) not in available:
            errors.append(f"missing required MMD bone: {required}")

    twist_pairs: list[TwistPair] = []
    for source, aliases in _TWIST_ALIASES.items():
        # _TWIST_ALIASES is keyed by SMPL-X names; translate to the SOMA equivalent.
        soma_source = {
            "left_shoulder": "LeftArm",
            "left_elbow": "LeftForeArm",
            "right_shoulder": "RightArm",
            "right_elbow": "RightForeArm",
        }.get(source)
        target = source_map.get(soma_source or "")
        twist_name = _pick(aliases, available, set())
        if target is None or twist_name is None:
            continue
        twist_pairs.append(TwistPair(swing_bone=target, twist_bone=twist_name, axis_local=(0.0, 1.0, 0.0)))

    transfer_modes = {
        canonical_map[canonical]: mode
        for canonical, mode in TRANSFER_MODES.items()
        if canonical in canonical_map and canonical not in _SKIP_DRIVE
    }
    driven_fingers = sum(1 for canonical in canonical_map if canonical in FINGER_MMD_ALIASES)

    return SomaRigMapping(
        smplx_to_fbx=source_map,
        transfer_modes=transfer_modes,
        twist_pairs=tuple(twist_pairs),
        canonical_map=canonical_map,
        driven_fingers=driven_fingers,
        errors=tuple(dict.fromkeys(errors)),
    )
