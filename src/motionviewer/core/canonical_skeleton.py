"""Canonical skeleton definitions shared across all bone-mapping backends.

Adopts the retarget_bvh canonical skeleton naming convention. Both the SMPL-X
addon armature and imported FBX characters are mapped to this intermediate
representation before retargeting, which eliminates parent-chain mismatches
and makes multi-rig support trivial.
"""

from __future__ import annotations

# ---- canonical bone names (retarget_bvh convention) ----------------------

CANONICAL_BODY_BONES: list[str] = [
    "hips",
    "spine",
    "spine-1",
    "chest",
    # "chest-1"  -- only present in 5-segment spines
    "neck",
    "head",
    "shoulder.L",
    "upper_arm.L",
    "forearm.L",
    "hand.L",
    "shoulder.R",
    "upper_arm.R",
    "forearm.R",
    "hand.R",
    "hip.L",
    "thigh.L",
    "shin.L",
    "foot.L",
    "toe.L",
    "hip.R",
    "thigh.R",
    "shin.R",
    "foot.R",
    "toe.R",
]

# Canonical parent hierarchy (bones that are always present).
CANONICAL_PARENT_MAP: dict[str, str | None] = {
    "hips": None,
    "spine": "hips",
    "spine-1": "spine",
    "chest": "spine-1",
    "neck": "chest",
    "head": "neck",
    "shoulder.L": "chest",
    "upper_arm.L": "shoulder.L",
    "forearm.L": "upper_arm.L",
    "hand.L": "forearm.L",
    "shoulder.R": "chest",
    "upper_arm.R": "shoulder.R",
    "forearm.R": "upper_arm.R",
    "hand.R": "forearm.R",
    "hip.L": "hips",
    "thigh.L": "hip.L",
    "shin.L": "thigh.L",
    "foot.L": "shin.L",
    "toe.L": "foot.L",
    "hip.R": "hips",
    "thigh.R": "hip.R",
    "shin.R": "thigh.R",
    "foot.R": "shin.R",
    "toe.R": "foot.R",
}


# ---- SMPL-X addon bone name → canonical ---------------------------------
# SMPL-X body22 bones as created by the `smplx_blender_addon`.
# Note: SMPL-X has `left_hip`/`right_hip` as children of `pelvis`, so we map
# them to canonical `hip.L`/`hip.R`. Similarly, `left_knee` → `thigh.L`
# (upper leg) and `left_ankle` → `foot.L` (foot root, not ankle).

SMPLX_TO_CANONICAL: dict[str, str] = {
    "pelvis": "hips",
    "spine1": "spine",
    "spine2": "spine-1",
    "spine3": "chest",
    "neck": "neck",
    "head": "head",
    "left_collar": "shoulder.L",
    "left_shoulder": "upper_arm.L",
    "left_elbow": "forearm.L",
    "left_wrist": "hand.L",
    "right_collar": "shoulder.R",
    "right_shoulder": "upper_arm.R",
    "right_elbow": "forearm.R",
    "right_wrist": "hand.R",
    "left_hip": "hip.L",
    "left_knee": "thigh.L",
    "left_ankle": "foot.L",
    "left_foot": "toe.L",
    "right_hip": "hip.R",
    "right_knee": "thigh.R",
    "right_ankle": "foot.R",
    "right_foot": "toe.R",
}

# Inverse: canonical name → SMPL-X addon bone name
CANONICAL_TO_SMPLX: dict[str, str] = {v: k for k, v in SMPLX_TO_CANONICAL.items()}
