"""Mixamo-only bone-map resolution tests."""

from __future__ import annotations

import pytest

from motionviewer.blender.retarget._resolve import BONE_MAP_PRESETS, resolve_bone_mapping
from motionviewer.core.canonical_skeleton import SMPLX_TO_CANONICAL


class _Bone:
    def __init__(self, name: str, parent: _Bone | None = None) -> None:
        self.name = name
        self.parent = parent
        self.length = 1.0


class _MixamoArmature:
    def __init__(self, prefix: str = "", *, missing: str | None = None) -> None:
        parents = {
            "Hips": None,
            "Spine": "Hips",
            "Spine1": "Spine",
            "Spine2": "Spine1",
            "Neck": "Spine2",
            "Head": "Neck",
            "LeftShoulder": "Spine2",
            "LeftArm": "LeftShoulder",
            "LeftForeArm": "LeftArm",
            "LeftHand": "LeftForeArm",
            "RightShoulder": "Spine2",
            "RightArm": "RightShoulder",
            "RightForeArm": "RightArm",
            "RightHand": "RightForeArm",
            "LeftUpLeg": "Hips",
            "LeftLeg": "LeftUpLeg",
            "LeftFoot": "LeftLeg",
            "LeftToeBase": "LeftFoot",
            "RightUpLeg": "Hips",
            "RightLeg": "RightUpLeg",
            "RightFoot": "RightLeg",
            "RightToeBase": "RightFoot",
        }
        bones: dict[str, _Bone] = {}
        for name, parent_name in parents.items():
            if name != missing:
                bones[name] = _Bone(prefix + name, bones.get(parent_name))
        self.data = type("ArmatureData", (), {"bones": list(bones.values())})()


def test_mixamo_is_the_supported_fbx_preset() -> None:
    assert "mixamo" in BONE_MAP_PRESETS
    assert "mmd" in BONE_MAP_PRESETS


@pytest.mark.parametrize("prefix", ["", "mixamorig:", "mixamorig1:"])
def test_all_mixamo_namespaces_resolve_to_one_family(prefix: str) -> None:
    mapping = resolve_bone_mapping("auto", fbx_armature=_MixamoArmature(prefix))
    assert mapping.rig_family == "mixamo"
    assert mapping.prefix == prefix
    assert set(mapping.smplx_to_fbx) == set(SMPLX_TO_CANONICAL)
    assert mapping.smplx_to_fbx["left_elbow"] == prefix + "LeftForeArm"


def test_incomplete_mixamo_rig_fails_preflight() -> None:
    with pytest.raises(ValueError, match="missing Mixamo bone: LeftForeArm"):
        resolve_bone_mapping("auto", fbx_armature=_MixamoArmature(missing="LeftForeArm"))


def test_mixamo_mapping_has_no_per_asset_override_escape_hatch() -> None:
    with pytest.raises(TypeError):
        resolve_bone_mapping(
            "auto",
            fbx_armature=_MixamoArmature(),
            overrides={"left_elbow": "case_specific_bone"},
        )
