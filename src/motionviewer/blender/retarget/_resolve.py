"""Bone-map resolution for Mixamo and MMD target rigs.

All callers receive SMPL-X names mapped to the actual imported bone names.
The adapter is the only place that knows whether an asset uses a prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.canonical_skeleton import SMPLX_TO_CANONICAL
from .calibration import MIXAMO_BODY_BONES, MixamoNameAdapter, inspect_mixamo_rig
from .mmd import detect_mmd_family, inspect_mmd_rig
from .twist import TwistPair

# Retained as a public discovery constant.
BONE_MAP_PRESETS: dict[str, tuple[str, ...]] = {"mixamo": MIXAMO_BODY_BONES, "mmd": ()}

_CANONICAL_TO_MIXAMO: dict[str, str] = {
    "hips": "Hips",
    "spine": "Spine",
    "spine-1": "Spine1",
    "chest": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "shoulder.L": "LeftShoulder",
    "upper_arm.L": "LeftArm",
    "forearm.L": "LeftForeArm",
    "hand.L": "LeftHand",
    "shoulder.R": "RightShoulder",
    "upper_arm.R": "RightArm",
    "forearm.R": "RightForeArm",
    "hand.R": "RightHand",
    "hip.L": "LeftUpLeg",
    "thigh.L": "LeftLeg",
    "foot.L": "LeftFoot",
    "toe.L": "LeftToeBase",
    "hip.R": "RightUpLeg",
    "thigh.R": "RightLeg",
    "foot.R": "RightFoot",
    "toe.R": "RightToeBase",
}


@dataclass(frozen=True)
class BoneMapping:
    smplx_to_fbx: dict[str, str]
    rig_family: str = "mixamo"
    prefix: str = ""
    twist_pairs: tuple[TwistPair, ...] = field(default_factory=tuple)
    # Target bone name -> transfer mode. MMD only; see mmd_solve.
    transfer_modes: dict[str, str] = field(default_factory=dict)


def _cross_with_smplx(canonical_map: dict[str, str]) -> dict[str, str]:
    return {
        smplx_name: canonical_map[canonical_name]
        for smplx_name, canonical_name in SMPLX_TO_CANONICAL.items()
        if canonical_name in canonical_map
    }


def resolve_bone_mapping(
    bone_map_spec: str = "auto",
    *,
    fbx_armature: Any = None,
) -> BoneMapping:
    """Resolve a Mixamo or MMD mapping, or fail before animation begins."""
    if bone_map_spec not in {"auto", "mixamo", "mmd"}:
        raise ValueError("retarget accepts bone_map 'auto', 'mixamo', or 'mmd'")
    if fbx_armature is None:
        raise ValueError("A target armature is required to resolve its namespace")

    bone_names = {bone.name for bone in fbx_armature.data.bones}
    use_mmd = bone_map_spec == "mmd" or (bone_map_spec == "auto" and detect_mmd_family(bone_names))
    if use_mmd:
        inspection = inspect_mmd_rig(fbx_armature)
        if not inspection.valid:
            raise ValueError("Invalid MMD rig: " + "; ".join(inspection.errors))
        return BoneMapping(
            smplx_to_fbx=inspection.smplx_map,
            rig_family="mmd",
            twist_pairs=inspection.twist_pairs,
            transfer_modes=inspection.transfer_modes,
        )

    inspection = inspect_mixamo_rig(fbx_armature)
    if not inspection.valid or inspection.adapter is None:
        raise ValueError("Invalid Mixamo rig: " + "; ".join(inspection.errors))
    adapter = inspection.adapter
    canonical_map = {
        canonical: adapter.target_name(mixamo_name) for canonical, mixamo_name in _CANONICAL_TO_MIXAMO.items()
    }
    mapping = _cross_with_smplx(canonical_map)
    _validate_mixamo_mapping(mapping, fbx_armature, adapter)
    return BoneMapping(smplx_to_fbx=mapping, prefix=adapter.prefix)


def resolve_bone_map(
    bone_map_spec: str = "auto",
    *,
    fbx_armature: Any = None,
) -> dict[str, str]:
    """Compatibility wrapper returning the SMPL-X-to-target map."""
    return resolve_bone_mapping(bone_map_spec, fbx_armature=fbx_armature).smplx_to_fbx


def _validate_mixamo_mapping(mapping: dict[str, str], fbx_armature: Any, adapter: MixamoNameAdapter) -> None:
    target_bones = {bone.name for bone in fbx_armature.data.bones}
    required = tuple(SMPLX_TO_CANONICAL)
    missing = [source for source in required if source not in mapping]
    absent = [
        f"{source}->{mapping[source]}" for source in required if mapping.get(source) not in target_bones
    ]
    targets = [mapping[source] for source in required if source in mapping]
    duplicates = sorted({target for target in targets if targets.count(target) > 1})
    escaped_namespace = [target for target in targets if not target.startswith(adapter.prefix)]
    if missing or absent or duplicates or escaped_namespace:
        problems: list[str] = []
        if missing:
            problems.append("missing source mappings: " + ", ".join(missing))
        if absent:
            problems.append("target bones not found: " + ", ".join(absent))
        if duplicates:
            problems.append("duplicate target bones: " + ", ".join(duplicates))
        if escaped_namespace:
            problems.append("target bones escape Mixamo namespace: " + ", ".join(escaped_namespace))
        raise ValueError("Invalid Mixamo body22 mapping: " + "; ".join(problems))
