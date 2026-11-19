"""Physics reports must describe a finished bake at the source's actual frame rate."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from motionviewer.blender.mmd_bullet import (
    BulletSettings,
    add_fitted_hand_colliders,
    build_and_bake,
    relax_locked_cloth_angles,
    relax_locked_cloth_translations,
)


@pytest.fixture
def physics(monkeypatch):
    cache = SimpleNamespace(frame_start=1, frame_end=250, is_baked=False)
    world = SimpleNamespace(point_cache=cache, enabled=False)
    scene = SimpleNamespace(
        render=SimpleNamespace(fps=24, fps_base=1.0),
        frame_start=1,
        frame_end=250,
        frame_set=Mock(),
        rigidbody_world=world,
    )
    dynamic = SimpleNamespace(kinematic=False, linear_damping=0.8, angular_damping=0.3)
    kinematic = SimpleNamespace(kinematic=True, linear_damping=0.9, angular_damping=0.4)
    model = SimpleNamespace(
        rigidBodies=Mock(return_value=[
            SimpleNamespace(rigid_body=dynamic),
            SimpleNamespace(rigid_body=kinematic),
        ]),
        joints=Mock(return_value=[object()]),
        build=Mock(),
    )
    root_lookup = Mock(return_value=object())
    module = ModuleType("mmd_tools.core.model")
    module.FnModel = SimpleNamespace(find_root_object=root_lookup)
    module.Model = Mock(return_value=model)
    for name in ("mmd_tools", "mmd_tools.core"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "mmd_tools.core.model", module)

    def bake(**kwargs):
        cache.is_baked = True
        return {"FINISHED"}

    bake_operator = Mock(side_effect=bake)
    bpy = SimpleNamespace(
        context=SimpleNamespace(
            scene=scene,
            view_layer=SimpleNamespace(update=Mock()),
            temp_override=lambda **kwargs: nullcontext(),
        ),
        ops=SimpleNamespace(ptcache=SimpleNamespace(bake=bake_operator)),
    )
    return SimpleNamespace(
        bpy=bpy,
        scene=scene,
        cache=cache,
        model=model,
        dynamic=dynamic,
        kinematic=kinematic,
        root_lookup=root_lookup,
        bake=bake_operator,
    )


def test_fractional_source_rate_survives_simulation_and_report(physics):
    fps = 30000 / 1001
    result = build_and_bake(
        physics.bpy, object(), frame_start=1, frame_end=60, fps=fps,
        settings=BulletSettings(fix_damping=False, settle_frames=20),
    )

    assert physics.scene.render.fps / physics.scene.render.fps_base == pytest.approx(fps)
    assert result["fps"] == pytest.approx(fps)
    assert result["scene_fps"] == 30
    assert result["fps_base"] == pytest.approx(1.001)
    assert result["built"] is True and result["baked"] is True
    assert result["settled_from"] == -19
    assert result["baked_to"] == 60
    assert (physics.cache.frame_start, physics.cache.frame_end) == (-19, 60)
    assert (physics.scene.frame_start, physics.scene.frame_end) == (1, 250)


@pytest.mark.parametrize("legacy, expected_linear, expected_angular, mode", [
    (False, 0.8, 0.3, "pmx"),
    (True, 0.2, 0.7, "legacy-inverted"),
])
def test_report_distinguishes_preserved_pmx_damping_from_legacy_adjustment(
    physics, legacy, expected_linear, expected_angular, mode,
):
    result = build_and_bake(
        physics.bpy, object(), frame_start=1, frame_end=60,
        settings=BulletSettings(fix_damping=legacy),
    )

    assert result["damping_mode"] == mode
    assert physics.dynamic.linear_damping == pytest.approx(expected_linear)
    assert physics.dynamic.angular_damping == pytest.approx(expected_angular)
    assert physics.kinematic.linear_damping == 0.9
    assert physics.kinematic.angular_damping == 0.4


def test_legacy_default_remains_compatible_with_existing_renders(physics):
    result = build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60)
    assert result["damping_mode"] == "legacy-inverted"
    assert physics.dynamic.linear_damping == pytest.approx(0.2)


def test_scene_box_colliders_are_added_after_rig_build_and_before_baking(physics, monkeypatch):
    from motionviewer.blender import mmd_prop_physics

    boxes = (object(),)
    report = {"enabled": True, "bodies": [{"name": "carry_box"}]}

    def add(bpy, objects, model):
        assert bpy is physics.bpy and objects is boxes and model is physics.model
        physics.model.build.assert_called_once()
        physics.bake.assert_not_called()
        return report

    helper = Mock(side_effect=add)
    monkeypatch.setattr(mmd_prop_physics, "add_animated_box_colliders", helper)
    result = build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60, scene_box_colliders=boxes)
    helper.assert_called_once()
    physics.bake.assert_called_once()
    assert result["scene_box_colliders"] == report
    assert result["baked"] is True


def test_default_scene_does_not_create_prop_colliders(physics, monkeypatch):
    from motionviewer.blender import mmd_prop_physics

    helper = Mock()
    monkeypatch.setattr(mmd_prop_physics, "add_animated_box_colliders", helper)
    result = build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60)
    helper.assert_not_called()
    assert result["scene_box_colliders"] == {"enabled": False, "bodies": []}


def test_failed_prop_collider_setup_does_not_bake_or_report_success(physics, monkeypatch):
    from motionviewer.blender import mmd_prop_physics

    monkeypatch.setattr(mmd_prop_physics, "add_animated_box_colliders", Mock(side_effect=ValueError("not a box")))
    with pytest.raises(ValueError, match="not a box"):
        build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60, scene_box_colliders=(object(),))
    physics.bake.assert_not_called()


def test_weapon_colliders_use_the_selected_object_after_build_and_before_bake(physics, monkeypatch):
    from motionviewer.blender import mmd_weapon_physics

    weapons = (object(),)
    report = {"enabled": True, "bodies": [{"name": "segmented_axe"}]}

    def add(bpy, objects, model):
        assert bpy is physics.bpy and objects is weapons and model is physics.model
        physics.model.build.assert_called_once()
        physics.bake.assert_not_called()
        return report

    helper = Mock(side_effect=add)
    monkeypatch.setattr(mmd_weapon_physics, "add_animated_weapon_colliders", helper)
    result = build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60,
                            scene_weapon_colliders=weapons)
    helper.assert_called_once()
    physics.bake.assert_called_once()
    assert result["scene_weapon_colliders"] == report


def test_legacy_scene_does_not_create_weapon_colliders(physics, monkeypatch):
    from motionviewer.blender import mmd_weapon_physics

    helper = Mock()
    monkeypatch.setattr(mmd_weapon_physics, "add_animated_weapon_colliders", helper)
    result = build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60)
    helper.assert_not_called()
    assert result["scene_weapon_colliders"] == {"enabled": False, "bodies": []}


def test_invalid_weapon_proxy_fails_before_baking(physics, monkeypatch):
    from motionviewer.blender import mmd_weapon_physics

    monkeypatch.setattr(mmd_weapon_physics, "add_animated_weapon_colliders",
                        Mock(side_effect=ValueError("unapplied scale")))
    with pytest.raises(ValueError, match="unapplied scale"):
        build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60,
                       scene_weapon_colliders=(object(),))
    physics.bake.assert_not_called()


def test_existing_scene_fractional_rate_is_reported_when_source_rate_is_omitted(physics):
    physics.scene.render.fps = 24
    physics.scene.render.fps_base = 1.001
    result = build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60)
    assert result["fps"] == pytest.approx(24 / 1.001)


@pytest.mark.parametrize("failure", ["cancelled", "unbaked", "exception"])
def test_failed_bake_never_reports_success_and_restores_scene_range(physics, failure):
    if failure == "exception":
        physics.bake.side_effect = RuntimeError("simulator failed")
    else:
        physics.bake.side_effect = None
        physics.bake.return_value = {"CANCELLED"} if failure == "cancelled" else {"FINISHED"}
        # Even a stale baked cache must not turn a cancelled operation into success.
        physics.cache.is_baked = failure == "cancelled"

    with pytest.raises(RuntimeError):
        build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60)
    assert (physics.scene.frame_start, physics.scene.frame_end) == (1, 250)


@pytest.mark.parametrize("missing", ["root", "bodies"])
def test_missing_physics_data_is_explicit_and_does_not_attempt_a_bake(physics, missing):
    if missing == "root":
        physics.root_lookup.return_value = None
    else:
        physics.model.rigidBodies.return_value = []

    result = build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60)
    assert result["built"] is False and result["baked"] is False
    assert result["reason"]
    physics.model.build.assert_not_called()
    physics.bake.assert_not_called()


def test_missing_point_cache_cannot_report_a_finished_simulation(physics):
    physics.scene.rigidbody_world.point_cache = None
    with pytest.raises(RuntimeError, match="no point cache"):
        build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60)
    physics.bake.assert_not_called()
    assert (physics.scene.frame_start, physics.scene.frame_end) == (1, 250)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_rate_fails_before_building_the_rig(physics, fps):
    with pytest.raises(ValueError, match="finite and positive"):
        build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60, fps=fps)
    physics.model.build.assert_not_called()
    physics.bake.assert_not_called()


def _joint(name, bone, *, mode="1"):
    body = SimpleNamespace(mmd_rigid=SimpleNamespace(bone=bone, type=mode))
    constraint = SimpleNamespace(object1=body, object2=None)
    for axis in "xyz":
        setattr(constraint, f"use_limit_ang_{axis}", True)
        setattr(constraint, f"limit_ang_{axis}_lower", 0.0)
        setattr(constraint, f"limit_ang_{axis}_upper", 0.0)
        setattr(constraint, f"limit_lin_{axis}_lower", 0.0)
        setattr(constraint, f"limit_lin_{axis}_upper", 0.0)
        setattr(constraint, f"use_limit_lin_{axis}", True)
    return SimpleNamespace(name=name, rigid_body_constraint=constraint)


def test_cloth_adaptation_only_opens_enabled_locked_angles_of_dynamic_garments():
    cloth = _joint("cloth", "Skirt_01")
    c = cloth.rigid_body_constraint
    c.limit_ang_y_lower, c.limit_ang_y_upper = -0.3, 0.2
    c.use_limit_ang_z = False
    hair = _joint("hair", "hair_01")
    static = _joint("static", "裙_00", mode="0")
    model = SimpleNamespace(joints=lambda: [cloth, hair, static])
    report = relax_locked_cloth_angles(model, BulletSettings(cloth_angular_slack_degrees=10))
    assert report["joints"] == 1 and report["axes"] == 1
    assert report["joint_names"] == ["cloth"]
    assert c.limit_ang_x_lower == pytest.approx(-0.1745329252)
    assert c.limit_ang_x_upper == pytest.approx(0.1745329252)
    assert (c.limit_ang_y_lower, c.limit_ang_y_upper) == (-0.3, 0.2)
    assert c.limit_ang_z_lower == c.limit_ang_z_upper == 0
    assert c.limit_lin_x_lower == c.limit_lin_x_upper == 0
    assert hair.rigid_body_constraint.limit_ang_x_upper == 0
    assert static.rigid_body_constraint.limit_ang_x_upper == 0


def test_default_and_explicit_part_selection_preserve_other_joints():
    skirt, ribbon = _joint("skirt", "裙_01"), _joint("ribbon", "ribbon_01")
    model = SimpleNamespace(joints=lambda: [skirt, ribbon])
    report = relax_locked_cloth_angles(model, BulletSettings())
    assert report["joints"] == 0
    assert skirt.rigid_body_constraint.limit_ang_x_upper == 0
    report = relax_locked_cloth_angles(
        model, BulletSettings(cloth_angular_slack_degrees=5, cloth_bone_tokens=("ribbon",)),
    )
    assert report["joint_names"] == ["ribbon"]
    assert skirt.rigid_body_constraint.limit_ang_x_upper == 0


@pytest.mark.parametrize("degrees", [-1, 46, float("nan"), float("inf")])
def test_invalid_cloth_slack_fails_before_mutating_the_scene(physics, degrees):
    with pytest.raises(ValueError, match="cloth angular slack"):
        build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60,
                       settings=BulletSettings(cloth_angular_slack_degrees=degrees))
    physics.model.build.assert_not_called()
    physics.scene.frame_set.assert_not_called()


def test_hand_colliders_are_opt_in_without_touching_scene_or_model():
    assert add_fitted_hand_colliders(None, None, None, BulletSettings()) == {
        "enabled": False, "bodies": [],
    }


def test_hand_colliders_report_absent_garment_without_modifying_body_collisions():
    rb = SimpleNamespace(kinematic=False, collision_collections=[True] + [False] * 19)
    hair = SimpleNamespace(rigid_body=rb, mmd_rigid=SimpleNamespace(bone="hair_01"))
    model = SimpleNamespace(rigidBodies=lambda: [hair])
    report = add_fitted_hand_colliders(None, None, model, BulletSettings(fit_hand_colliders=True))
    assert report["bodies"] == []
    assert report["reason"] == "no selected dynamic garment bodies"
    assert rb.collision_collections == [True] + [False] * 19


def test_hand_colliders_refuse_to_repurpose_an_existing_collision_layer():
    rb = SimpleNamespace(kinematic=False, collision_collections=[True] * 20)
    skirt = SimpleNamespace(rigid_body=rb, mmd_rigid=SimpleNamespace(bone="裙_01"))
    model = SimpleNamespace(rigidBodies=lambda: [skirt])
    bpy = SimpleNamespace(context=SimpleNamespace(scene=SimpleNamespace(objects=[skirt])))
    with pytest.raises(RuntimeError, match="unused Blender collision layer"):
        add_fitted_hand_colliders(bpy, None, model, BulletSettings(fit_hand_colliders=True))
    assert rb.collision_collections == [True] * 20


def test_millimetre_cloth_compliance_preserves_authored_ranges_and_other_parts():
    skirt = _joint("vertical skirt", "裙_1_14")
    hair = _joint("hair", "hair_01")
    static = _joint("static", "裙_00", mode="0")
    c = skirt.rigid_body_constraint
    c.limit_lin_x_lower = c.limit_lin_x_upper = 0.003
    c.limit_lin_y_lower, c.limit_lin_y_upper = -0.065, 0.065
    c.use_limit_lin_z = False
    model = SimpleNamespace(joints=lambda: [skirt, hair, static])
    report = relax_locked_cloth_translations(model, BulletSettings(cloth_linear_slack_m=0.002))
    assert report["joints"] == 1 and report["axes"] == 1
    assert report["changes"] == [{"joint": "vertical skirt", "axes": [
        {"axis": "x", "before_m": [0.003, 0.003], "after_m": [0.001, 0.005]},
    ]}]
    assert (c.limit_lin_x_lower, c.limit_lin_x_upper) == pytest.approx((0.001, 0.005))
    assert (c.limit_lin_y_lower, c.limit_lin_y_upper) == (-0.065, 0.065)
    assert c.limit_lin_z_lower == c.limit_lin_z_upper == 0
    assert c.limit_ang_x_lower == c.limit_ang_x_upper == 0
    assert hair.rigid_body_constraint.limit_lin_x_upper == 0
    assert static.rigid_body_constraint.limit_lin_x_upper == 0
    assert report["solver_target_not_measured_error_bound"] is True


def test_linear_cloth_compliance_is_opt_in_and_reports_no_adaptation_by_default(physics):
    assert relax_locked_cloth_translations(None, BulletSettings())["changes"] == []
    result = build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60)
    assert result["cloth_linear_constraints"]["slack_m"] == 0
    assert result["cloth_linear_constraints"]["joints"] == 0


@pytest.mark.parametrize("slack", [-0.001, 0.011, float("nan"), float("inf")])
def test_invalid_linear_cloth_compliance_fails_before_scene_changes(physics, slack):
    with pytest.raises(ValueError, match="cloth linear slack"):
        build_and_bake(physics.bpy, object(), frame_start=1, frame_end=60,
                       settings=BulletSettings(cloth_linear_slack_m=slack))
    physics.model.build.assert_not_called()
    physics.scene.frame_set.assert_not_called()


@pytest.mark.parametrize("reverse", [False, True])
def test_skirt_compliance_preserves_pins_to_other_dynamic_chains(reverse):
    waist = _joint("waist anchor", "裙_0_1")
    waist.rigid_body_constraint.object2 = SimpleNamespace(
        mmd_rigid=SimpleNamespace(bone="下半身", type="0"),
    )
    tail = _joint("coat tail pin", "裙_2_1")
    c = tail.rigid_body_constraint
    c.object2 = SimpleNamespace(mmd_rigid=SimpleNamespace(bone="衣襬_0_1", type="1"))
    if reverse:
        c.object1, c.object2 = c.object2, c.object1
    model = SimpleNamespace(joints=lambda: [waist, tail])
    settings = BulletSettings(cloth_linear_slack_m=0.002, cloth_angular_slack_degrees=10)
    linear = relax_locked_cloth_translations(model, settings)
    angular = relax_locked_cloth_angles(model, settings)
    assert linear["joints"] == angular["joints"] == 1
    assert linear["changes"][0]["joint"] == "waist anchor"
    assert angular["joint_names"] == ["waist anchor"]
    for axis in "xyz":
        for kind in ("lin", "ang"):
            assert getattr(c, f"limit_{kind}_{axis}_lower") == 0
            assert getattr(c, f"limit_{kind}_{axis}_upper") == 0


def test_neutral_leg_settle_keys_only_driven_legs_and_only_before_the_clip(monkeypatch):
    import sys as _sys

    from motionviewer.blender import mmd_bullet

    class Quat(tuple):
        def __new__(cls, values=(1.0, 0.0, 0.0, 0.0)):
            return super().__new__(cls, values)

        def copy(self):
            return Quat(self)

        def slerp(self, other, w):
            return Quat(tuple(a + (b - a) * w for a, b in zip(self, other)))

    class Vec(tuple):
        def copy(self):
            return Vec(self)

        def __mul__(self, w):
            return Vec(tuple(a * w for a in self))

    mathutils = ModuleType("mathutils")
    mathutils.Quaternion = Quat
    monkeypatch.setitem(_sys.modules, "mathutils", mathutils)

    keys = []

    def bone(name):
        b = SimpleNamespace(name=name, rotation_mode="XYZ", rotation_quaternion=Quat((0.9, 0.1, 0.0, 0.0)),
                            location=Vec((0.0, 0.0, 0.0)))
        b.keyframe_insert = lambda path, frame, group=None: keys.append((name, path, frame))
        return b

    bones = [bone("左足"), bone("右ひざ"), bone("左足D"), bone("上半身")]
    curves = [SimpleNamespace(data_path=f'pose.bones["{n}"].rotation_quaternion') for n in ("左足", "右ひざ", "上半身")]
    action = SimpleNamespace(fcurves=curves)
    armature = SimpleNamespace(pose=SimpleNamespace(bones=bones), animation_data=SimpleNamespace(action=action))
    scene = SimpleNamespace(frame_set=lambda f: None)
    bpy = SimpleNamespace(context=SimpleNamespace(scene=scene))

    report = mmd_bullet.neutral_leg_settle(bpy, armature, frame_start=1, settle_frames=30)

    # The unkeyed D-bone and the upper body are left alone.
    assert report["bones"] == ["左足", "右ひざ"]
    assert {name for name, _, _ in keys} == {"左足", "右ひざ"}
    # Every key is a pre-roll frame; nothing the camera sees is rewritten.
    assert max(frame for _, _, frame in keys) == 0
    assert min(frame for _, _, frame in keys) == -29
