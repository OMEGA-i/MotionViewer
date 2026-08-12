from types import SimpleNamespace

import pytest

from motionviewer.blender.retarget.calibration import (
    MIXAMO_BODY_BONES,
    MixamoNameAdapter,
    calibrate_imported_fbx,
    detect_mixamo_family,
    infer_mixamo_unit_correction,
)


class _Obj:
    def __init__(self, name: str = "Armature", *, bones: list[str] | None = None) -> None:
        self.name = name
        self.rotation_euler = (1.570796, 0.0, 0.0)
        self.scale = (0.01, 0.01, 0.01)
        self.matrix_world = (
            (0.01, 0.0, 0.0, 0.0),
            (0.0, 0.0, -0.01, 0.0),
            (0.0, 0.01, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.data = SimpleNamespace(bones=[SimpleNamespace(name=item) for item in (bones or [])])

    def select_set(self, value: bool) -> None:
        pass


class _Bpy:
    def __init__(self) -> None:
        self.transform_apply_calls = 0
        self.context = SimpleNamespace(view_layer=SimpleNamespace(objects=SimpleNamespace(active=None)))
        self.ops = SimpleNamespace(
            object=SimpleNamespace(select_all=lambda action: None, transform_apply=self._transform_apply)
        )

    def _transform_apply(self, *, location: bool, rotation: bool, scale: bool) -> None:
        self.transform_apply_calls += 1
        assert location is False and rotation is True and scale is True


@pytest.mark.parametrize("prefix", ["", "mixamorig:", "mixamorig1:"])
def test_adapter_unifies_mixamo_namespaces(prefix: str) -> None:
    adapter = MixamoNameAdapter.detect({prefix + name for name in MIXAMO_BODY_BONES})
    assert adapter is not None
    assert adapter.prefix == prefix
    assert adapter.target_name("LeftArm") == prefix + "LeftArm"


def test_calibration_records_basis_without_mutating_imported_objects() -> None:
    bpy = _Bpy()
    armature = _Obj(bones=["mixamorig:" + name for name in MIXAMO_BODY_BONES])
    result = calibrate_imported_fbx(armature)
    assert detect_mixamo_family(armature) == "mixamo"
    assert result.rig_family == "mixamo"
    assert result.prefix == "mixamorig:"
    assert result.transform_baked is False
    assert result.uniform_scale == pytest.approx(0.01)
    assert result.object_basis[1][2] == pytest.approx(-0.01)
    assert bpy.transform_apply_calls == 0


def test_calibration_rejects_non_uniform_object_basis() -> None:
    armature = _Obj(bones=["mixamorig:" + name for name in MIXAMO_BODY_BONES])
    armature.scale = (0.01, 0.02, 0.01)

    with pytest.raises(ValueError, match="non-uniform"):
        calibrate_imported_fbx(armature)


def test_calibration_rejects_unknown_rig() -> None:
    with pytest.raises(ValueError, match="Unsupported FBX rig"):
        calibrate_imported_fbx(_Obj(bones=["root", "spine"]))


@pytest.mark.parametrize(
    ("height_m", "expected"),
    ((1.85, 1.0), (3.5, 1.0), (16.65, 0.1), (0.17, 10.0), (166.5, 0.01)),
)
def test_mixamo_unit_correction_only_repairs_decimal_unit_mismatches(
    height_m: float, expected: float
) -> None:
    assert infer_mixamo_unit_correction(height_m) == pytest.approx(expected)
