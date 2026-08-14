"""MMD/PMX rig identification and SMPL-X body-22 mapping.

Genshin models add cancel bones (肩P/肩C) and axis-limited 捩 bones on the
arm FK chain.  The mapper binds SMPL-X joints onto the deform bones and
records the twist pairs so the animation adapter can split swing/twist
without writing to cancel, IK, or append-copy bones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.canonical_skeleton import SMPLX_TO_CANONICAL
from .twist import TwistPair

# Canonical Mixamo-style roles -> preferred MMD names, first match wins.
_CANONICAL_TO_MMD_ALIASES: dict[str, tuple[str, ...]] = {
    "hips": ("腰", "下半身", "センター"),
    "spine": ("上半身",),
    "spine-1": ("上半身3",),
    "chest": ("上半身2",),
    "neck": ("首",),
    "head": ("頭",),
    "shoulder.L": ("左肩",),
    "upper_arm.L": ("左腕",),
    "forearm.L": ("左ひじ", "左肘"),
    "hand.L": ("左手首",),
    "shoulder.R": ("右肩",),
    "upper_arm.R": ("右腕",),
    "forearm.R": ("右ひじ", "右肘"),
    "hand.R": ("右手首",),
    "hip.L": ("左足",),
    "thigh.L": ("左ひざ", "左膝"),
    "foot.L": ("左足首",),
    "toe.L": ("左つま先", "左足先EX"),
    "hip.R": ("右足",),
    "thigh.R": ("右ひざ", "右膝"),
    "foot.R": ("右足首",),
    "toe.R": ("右つま先", "右足先EX"),
}

_TWIST_ALIASES: dict[str, tuple[str, ...]] = {
    "left_shoulder": ("左腕捩",),
    "left_elbow": ("左手捩",),
    "right_shoulder": ("右腕捩",),
    "right_elbow": ("右手捩",),
}

# Toe bones on this rig family carry no mesh weight (``左足先EX`` is empty and
# ``左つま先`` is an IK aim), so driving them moves nothing and only risks
# fighting the foot IK.  Every other joint is driven.
_SKIP_DRIVE: frozenset[str] = frozenset({"toe.L", "toe.R"})

# Canonical roles a rig may legitimately not have.
_OPTIONAL_CANONICAL: frozenset[str] = frozenset({"spine-1"})

# Source joints that may go unmapped without losing their motion, and the
# canonical role whose absence excuses them.
#
# Transfers apply global rotations, so an intermediate source joint that has no
# target is not dropped: its rotation is already contained in the mapped
# descendant.  A two-segment MMD upper body (``上半身``/``上半身2`` with no
# ``上半身3``, as on Honkai rigs) therefore still reproduces
# ``spine1 . spine2 . spine3`` exactly on ``上半身2`` — only the curve between
# them is coarser.  If ``上半身3`` *is* present, a missing ``spine2`` mapping is
# a real fault and still fails.
_COLLAPSIBLE_SOURCES: dict[str, str] = {"spine2": "spine-1"}

# How each joint's rest orientation relates to SMPL-X, and therefore which
# transfer is correct.  See mmd_solve for the derivation.
#
# ``absolute`` is only for the arm chain, where MMD binds A-pose against
# SMPL-X's T-pose rest (23-51 deg apart on a Genshin rig).  Everything else
# agrees within ~6 deg, or differs purely by bone convention (``腰`` displays
# 47 deg off vertical, ``頭`` is vertical while SMPL-X's neck-to-head edge
# leans forward, ``足首`` points further down because of the footwear), which
# ``relative`` preserves by construction.
_TRANSFER_MODES: dict[str, str] = {
    "hips": "relative",
    "spine": "relative",
    "spine-1": "relative",
    "chest": "relative",
    "neck": "relative",
    "head": "relative",
    "shoulder.L": "relative",
    "shoulder.R": "relative",
    "upper_arm.L": "absolute",
    "upper_arm.R": "absolute",
    "forearm.L": "absolute",
    "forearm.R": "absolute",
    "hand.L": "absolute",
    "hand.R": "absolute",
    "hip.L": "relative",
    "hip.R": "relative",
    "thigh.L": "relative",
    "thigh.R": "relative",
    "foot.L": "relative",
    "foot.R": "relative",
}

_SKIP_SUBSTRINGS: tuple[str, ...] = (
    "IK",
    "ＩＫ",
    "肩P",
    "肩C",
    "ダミー",
)

REQUIRED_MMD_BONES: tuple[str, ...] = (
    "上半身",
    "首",
    "頭",
    "左肩",
    "左腕",
    "左ひじ",
    "左手首",
    "右肩",
    "右腕",
    "右ひじ",
    "右手首",
    "左足",
    "左ひざ",
    "左足首",
    "右足",
    "右ひざ",
    "右足首",
)


@dataclass(frozen=True)
class MmdPolishOptions:
    """Departures from strict source fidelity, for looks rather than accuracy.

    All of these change what the character does, so they are grouped and
    recorded on the armature rather than folded into the transfer.  ``enabled``
    off reproduces the source exactly.
    """

    enabled: bool = True
    # Frames of moving average on the arm chain's axial rotation only.
    twist_window: int = 5
    # 0 leaves the model's flat bind hands; 1 is a relaxed curl.
    hand_relax: float = 1.0
    # Clavicle scaling. Fits overdrive the collar; 肩 carries real mesh weight
    # here, so the full value reads as a hunched shoulder.
    collar_damping: float = 0.45
    collar_limit_degrees: float = 22.0
    # Outward arm swing, in degrees, to clear a costume the source body never
    # had. Scaled to zero as the arm rises. 0 reproduces the source angles.
    arm_abduction_degrees: float = 12.0

    @classmethod
    def from_mapping(cls, values: dict | None) -> MmdPolishOptions:
        if not values:
            return cls()
        return cls(
            enabled=bool(values.get("enabled", True)),
            twist_window=int(values.get("twist_window", 5)),
            hand_relax=float(values.get("hand_relax", 1.0)),
            collar_damping=float(values.get("collar_damping", 0.45)),
            collar_limit_degrees=float(values.get("collar_limit_degrees", 22.0)),
            arm_abduction_degrees=float(values.get("arm_abduction_degrees", 12.0)),
        )


@dataclass(frozen=True)
class MmdRigInspection:
    bone_names: tuple[str, ...]
    canonical_map: dict[str, str]
    smplx_map: dict[str, str]
    twist_pairs: tuple[TwistPair, ...]
    errors: tuple[str, ...]
    transfer_modes: dict[str, str] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors


def _normalize_name(name: str) -> str:
    return name.replace(" ", "").replace("_", "")


def detect_mmd_family(bone_names: set[str]) -> bool:
    normalized = {_normalize_name(name) for name in bone_names}
    return "左腕" in normalized and "左ひじ" in normalized and "上半身" in normalized


def _pick(aliases: tuple[str, ...], available: dict[str, str], used: set[str]) -> str | None:
    for alias in aliases:
        actual = available.get(_normalize_name(alias))
        if actual is not None and actual not in used:
            return actual
    return None


def inspect_mmd_rig(armature: Any, twist_axes: dict[str, tuple[float, float, float]] | None = None) -> MmdRigInspection:
    bones = [bone.name for bone in armature.data.bones]
    available = {_normalize_name(name): name for name in bones}
    used: set[str] = set()
    canonical_map: dict[str, str] = {}
    errors: list[str] = []

    for canonical, aliases in _CANONICAL_TO_MMD_ALIASES.items():
        chosen = _pick(aliases, available, used)
        if chosen is None:
            if canonical in _OPTIONAL_CANONICAL:
                continue
            errors.append(f"missing MMD bone for {canonical}: {aliases[0]}")
            continue
        canonical_map[canonical] = chosen
        used.add(chosen)

    smplx_map = {
        smplx_name: canonical_map[canonical]
        for smplx_name, canonical in SMPLX_TO_CANONICAL.items()
        if canonical in canonical_map and canonical not in _SKIP_DRIVE
    }
    skip_sources = {name for name, canonical in SMPLX_TO_CANONICAL.items() if canonical in _SKIP_DRIVE}
    skip_sources |= {
        source for source, canonical in _COLLAPSIBLE_SOURCES.items() if canonical not in canonical_map
    }
    required = tuple(SMPLX_TO_CANONICAL)
    missing = [name for name in required if name not in smplx_map and name not in skip_sources]
    if missing:
        errors.append("incomplete SMPL-X mapping: " + ", ".join(missing))

    twist_pairs: list[TwistPair] = []
    axes = twist_axes or {}
    for source_name, aliases in _TWIST_ALIASES.items():
        target = smplx_map.get(source_name)
        twist_name = _pick(aliases, available, set())
        if target is None or twist_name is None:
            continue
        axis = axes.get(twist_name, (0.0, 1.0, 0.0))
        twist_pairs.append(TwistPair(swing_bone=target, twist_bone=twist_name, axis_local=axis))

    for required_name in REQUIRED_MMD_BONES:
        if _normalize_name(required_name) not in available:
            errors.append(f"missing required MMD bone: {required_name}")

    transfer_modes = {
        canonical_map[canonical]: mode
        for canonical, mode in _TRANSFER_MODES.items()
        if canonical in canonical_map and canonical not in _SKIP_DRIVE
    }

    return MmdRigInspection(
        bone_names=tuple(bones),
        canonical_map=canonical_map,
        smplx_map=smplx_map,
        twist_pairs=tuple(twist_pairs),
        errors=tuple(dict.fromkeys(errors)),
        transfer_modes=transfer_modes,
    )


def is_auxiliary_mmd_bone(name: str) -> bool:
    if name.endswith(("D", "P", "C")) and len(name) > 1:
        if name[-1] in {"P", "C"} and "肩" in name:
            return True
        if name.endswith("D") and any(part in name for part in ("足", "ひざ", "膝", "足首")):
            return True
    return any(token in name for token in _SKIP_SUBSTRINGS) or name[-1:].isdigit() and "捩" in name


def _additional_influence(pose_bone: Any) -> float:
    mmd_bone = getattr(pose_bone, "mmd_bone", None)
    if mmd_bone is None:
        return 0.0
    return float(getattr(mmd_bone, "additional_transform_influence", 0.0) or 0.0)


def mute_mmd_ik(armature: Any) -> list[str]:
    """Disable IK/cancel constraints that would overwrite retargeted FK.

    Append copies stay live: ``足D`` and ``腕捩1/2/3`` use positive-influence
    additional transforms to deform the mesh from the FK bones we keyframe.
    Cancel bones such as ``肩C`` use negative influence and must be muted or
    they fold the shoulder chain.
    """
    muted: list[str] = []
    for pose_bone in armature.pose.bones:
        influence = _additional_influence(pose_bone)
        for constraint in pose_bone.constraints:
            name = str(getattr(constraint, "type", "")).upper()
            label = str(getattr(constraint, "name", ""))
            is_ik = name in {"IK", "DAMPED_TRACK"} or "IK" in label or "ＩＫ" in label
            is_limit = name == "LIMIT_ROTATION"
            is_cancel = "mmd_additional" in label and influence < 0.0
            if not (is_ik or is_limit or is_cancel):
                continue
            constraint.mute = True
            muted.append(f"{pose_bone.name}:{label or name}")
    return muted
