"""Build and cache the collision bodies and joints authored in a PMX model.

Physics can move weighted hair and garment bones around body colliders, but it does
not guarantee an intersection-free mesh. Coverage depends on the asset; Blender's
hard constraints and emulated PMX collision masks also differ from MMD's simulation.
Use matched animation and visible mesh checks when evaluating a setting change.

Build only after retargeting and pose corrections. mmd_tools makes the physics
bodies drive their bones, and building at a different animation frame misaligns the
initial rig. Bake before framing or rendering, since those passes visit frames out
of order. Original PMX and motion files are never edited by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class BulletSettings:
    """Numerical settings and explicit PMX adaptations; visual review is required."""

    #: Bullet substeps per animation frame. Increasing this can reduce solver error;
    #: it cannot repair missing colliders or an over-constrained garment.
    substeps: int = 12
    #: Solver iterations. Long chains of joints — a skirt panel is four or five bodies
    #: deep — need more than the default 10 or the chain stretches visibly.
    iterations: int = 24
    #: Frames simulated before the first rendered frame, so the cloth is hanging under
    #: gravity rather than starting from the bind pose and dropping into place on
    #: camera. MMD users call this the settling period.
    settle_frames: int = 30
    #: Passed straight to ``Model.build``; this is a collision-shape margin in metres,
    #: not continuous collision detection or a guarantee against tunnelling.
    collision_margin: float = 1e-06
    non_collision_distance_scale: float = 1.5
    #: Preserve the legacy 1-damping artistic adjustment for existing renders.
    #: This is NOT a PMX-to-Blender conversion: both pass rigid-body damping to
    #: Bullet directly. False preserves the PMX values imported by mmd_tools.
    fix_damping: bool = True
    #: Optional Blender-specific garment adaptation. Open only angular axes that
    #: the asset locked completely; leave existing ranges and translation locks.
    #: Pins connecting a selected garment to a different dynamic part are preserved.
    #: Zero preserves the asset. Review each character before opting in.
    cloth_angular_slack_degrees: float = 0.0
    #: Optional half-range in metres for enabled, locked garment translation axes.
    #: This is bounded joint compliance, not mesh smoothing or a guarantee that
    #: Bullet will satisfy the limits exactly. Zero preserves authored limits.
    cloth_linear_slack_m: float = 0.0
    cloth_bone_tokens: tuple[str, ...] = ("裙", "裾", "スカート", "skirt")
    #: Optional hand coverage for assets whose wrist capsule stops before the hand.
    #: Uses actual skin vertices and collides only with the selected garment bodies.
    fit_hand_colliders: bool = False
    #: After the bake, replace whole-panel solver snaps with smooth bridges and leave
    #: ordinary swing bit-for-bit. Applied by the caller through
    #: ``mmd_cloth_cleanup.cleanup`` once every bake (including mesh cloth) is done,
    #: because it switches the Bullet world off. See ``core.cloth_despike``.
    despike: bool = False
    #: Settle the cloth with the legs at rest, then bring them into the clip's first
    #: pose before frame 1. A clip that opens mid-stride otherwise settles the skirt
    #: over a forward thigh, where the front panels rest on the leg colliders and stay
    #: propped out horizontally until the legs come together. Only pre-roll frames
    #: change; every rendered frame keeps its source pose. See ``neutral_leg_settle``.
    neutral_settle: bool = False
    #: After the bake, Gaussian-smooth every dynamic body's pelvis-relative track over
    #: this many frames (sigma; 0 = off). For continuous flicker rather than rare
    #: snaps. 1.5 keeps ~94% of a coat's swing. See ``core.cloth_despike.smooth_tracks``.
    smooth: float = 0.0
    #: Scene gravity. MMD's own convention, which ``mmd_tools`` carries as a commented
    #: out ``-9.81 * 10 * scale``; at the 0.08 import scale that is -7.848, against
    #: Blender's default -9.81.
    gravity: float = -7.848


def correct_damping(bpy: Any, model: Any) -> dict:
    """Apply the legacy inverted-damping look; retain the name for compatibility.

    This changes the asset's intended damping, rather than correcting its units.
    PMX players (Saba and three.js) and Blender all pass rigid-body damping directly
    to Bullet. Bullet attenuates velocity by ``(1 - damping) ** timestep``. Blender's
    separate legacy 6DOF spring-constraint damping conversion does not apply here.

    Existing renders used this inversion, so it remains opt-out until a visual
    comparison establishes a replacement. The report identifies it as an artistic
    adjustment, not a faithful PMX simulation.
    """
    changed = 0
    values = []
    for rigid in model.rigidBodies():
        body = rigid.rigid_body
        if body is None or body.kinematic:
            continue
        linear = max(0.0, min(1.0, 1.0 - body.linear_damping))
        angular = max(0.0, min(1.0, 1.0 - body.angular_damping))
        body.linear_damping = linear
        body.angular_damping = angular
        values.append(linear)
        changed += 1
    median = sorted(values)[len(values) // 2] if values else 0.0
    return {"bodies": changed, "median_linear_damping": round(median, 4)}


def _selected_garment_joint(constraint: Any, tokens: tuple[str, ...]) -> bool:
    """Exclude attachments to other dynamic parts, even when a skirt is one end.

    Yoimiya's long coat tail attaches to a short-skirt body. Selecting either
    endpoint loosened the coat's root pin unintentionally. A garment adaptation
    may touch a static waist anchor, but must not change another dynamic chain.
    """
    dynamic = [rigid for rigid in
               (getattr(obj, "mmd_rigid", None)
                for obj in (constraint.object1, constraint.object2))
               if rigid is not None and str(rigid.type) in {"1", "2"}]
    return bool(dynamic) and all(
        any(token in rigid.bone.casefold() for token in tokens) for rigid in dynamic
    )


def relax_locked_cloth_angles(model: Any, settings: BulletSettings) -> dict:
    """Give selected locked garment axes bounded room to respond to collisions.

    Some PMX garments rely on MMD's softer constraints. In Blender, a fully
    locked lattice can fight body collisions and chatter even at high damping.
    This is an explicit artistic adaptation, not a PMX parameter conversion.
    """
    degrees = settings.cloth_angular_slack_degrees
    if not math.isfinite(degrees) or not 0.0 <= degrees <= 45.0:
        raise ValueError("cloth angular slack must be finite and between 0 and 45 degrees")
    report: dict = {"slack_degrees": degrees, "joints": 0, "axes": 0, "joint_names": []}
    if degrees == 0:
        return report
    tokens = tuple(token.casefold() for token in settings.cloth_bone_tokens if token)
    slack = math.radians(degrees)
    for joint in model.joints():
        constraint = joint.rigid_body_constraint
        if constraint is None:
            continue
        if not _selected_garment_joint(constraint, tokens):
            continue
        changed = 0
        for axis in "xyz":
            if not getattr(constraint, f"use_limit_ang_{axis}"):
                continue
            lower = f"limit_ang_{axis}_lower"
            upper = f"limit_ang_{axis}_upper"
            lo, hi = getattr(constraint, lower), getattr(constraint, upper)
            if abs(hi - lo) <= 1e-8:
                centre = (lo + hi) * 0.5
                setattr(constraint, lower, centre - slack)
                setattr(constraint, upper, centre + slack)
                changed += 1
        if changed:
            report["joints"] += 1
            report["axes"] += changed
            report["joint_names"].append(joint.name)
    return report


def relax_locked_cloth_translations(model: Any, settings: BulletSettings) -> dict:
    """Give locked garment translations a small, explicitly requested range.

    Yoimiya's locked vertical skirt lattice can fight the animated leg colliders
    and release in a one-frame fold. A millimetre-scale range reduced those pops
    in matched native-motion reviews. This is a Blender garment adaptation,
    not a faithful conversion of PMX parameters; review each asset and motion.
    Existing nonzero ranges, disabled axes, body animation and source files are
    untouched. The reported range is a solver target, not a measured error bound.
    """
    slack = settings.cloth_linear_slack_m
    if not math.isfinite(slack) or not 0.0 <= slack <= 0.01:
        raise ValueError("cloth linear slack must be finite and between 0 and 0.01 metres")
    report: dict = {
        "slack_m": slack, "joints": 0, "axes": 0, "changes": [],
        "solver_target_not_measured_error_bound": True,
    }
    if slack == 0:
        return report
    tokens = tuple(token.casefold() for token in settings.cloth_bone_tokens if token)
    for joint in model.joints():
        constraint = joint.rigid_body_constraint
        if constraint is None:
            continue
        if not _selected_garment_joint(constraint, tokens):
            continue
        changes = []
        for axis in "xyz":
            if not getattr(constraint, f"use_limit_lin_{axis}"):
                continue
            lower, upper = f"limit_lin_{axis}_lower", f"limit_lin_{axis}_upper"
            lo, hi = getattr(constraint, lower), getattr(constraint, upper)
            if abs(hi - lo) <= 1e-8:
                centre = (lo + hi) * 0.5
                setattr(constraint, lower, centre - slack)
                setattr(constraint, upper, centre + slack)
                changes.append({"axis": axis, "before_m": [lo, hi],
                                "after_m": [centre - slack, centre + slack]})
        if changes:
            report["joints"] += 1
            report["axes"] += len(changes)
            report["changes"].append({"joint": joint.name, "axes": changes})
    return report


def add_fitted_hand_colliders(bpy: Any, armature: Any, model: Any, settings: BulletSettings) -> dict:
    """Fit kinematic hand hulls without enlarging the asset's whole collision rig.

    This opt-in is useful for a hand resting on a garment when the PMX only has a
    wrist capsule. A dedicated unused Blender collision layer restricts the new
    hulls to garment bodies. Each finger segment follows its own animated bone;
    the whole hand is not enclosed in one oversized rigid hull.
    """
    report: dict = {"enabled": settings.fit_hand_colliders, "bodies": []}
    if not settings.fit_hand_colliders:
        return report
    tokens = tuple(t.casefold() for t in settings.cloth_bone_tokens if t)
    rigids = list(model.rigidBodies())
    garment = [obj for obj in rigids if obj.rigid_body is not None
               and not obj.rigid_body.kinematic
               and any(t in obj.mmd_rigid.bone.casefold() for t in tokens)]
    if not garment:
        report["reason"] = "no selected dynamic garment bodies"
        return report
    used = {i for obj in bpy.context.scene.objects if obj.rigid_body is not None
            for i, enabled in enumerate(obj.rigid_body.collision_collections) if enabled}
    available = [i for i in range(20) if i not in used]
    if not available:
        raise RuntimeError("fitted hand colliders need an unused Blender collision layer")
    layer = available[-1]
    hand_roots = {"左手首", "右手首"}
    names = []
    for bone in armature.data.bones:
        if bone.name in hand_roots or any(parent.name in hand_roots for parent in bone.parent_recursive):
            names.append(bone.name)
    for name in names:
        bone = armature.data.bones.get(name)
        pose_bone = armature.pose.bones.get(name)
        if bone is None or pose_bone is None:
            continue
        inverse = (armature.matrix_world @ bone.matrix_local).inverted()
        points = []
        for mesh in model.meshes():
            if any(mod.type == "SOLIDIFY" and mod.name == "Outline" for mod in mesh.modifiers):
                continue
            group = mesh.vertex_groups.get(name)
            if group is None:
                continue
            skin_vertices = {index for poly in mesh.data.polygons
                             if any(t in mesh.data.materials[poly.material_index].name.casefold()
                                    for t in ("肌", "skin")) for index in poly.vertices}
            for index in sorted(skin_vertices):
                vertex = mesh.data.vertices[index]
                if any(g.group == group.index and g.weight > 0.5 for g in vertex.groups):
                    points.append(inverse @ (mesh.matrix_world @ vertex.co))
        if len(points) < 4:
            continue
        data = bpy.data.meshes.new(f"MotionViewerHand_{name}")
        data.from_pydata(points, [], [])
        data.update()
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(obj)
        obj.parent = armature
        obj.parent_type = "BONE"
        obj.parent_bone = name
        obj.matrix_world = armature.matrix_world @ pose_bone.matrix
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
            bpy.ops.rigidbody.object_add()
        rb = obj.rigid_body
        rb.kinematic = True
        rb.collision_shape = "CONVEX_HULL"
        rb.use_margin = True
        rb.collision_margin = 0.001
        rb.collision_collections = [i == layer for i in range(20)]
        obj.hide_render = True
        obj.hide_set(True)
        report["bodies"].append({"name": obj.name, "bone": name, "skin_vertices": len(points)})
    if report["bodies"]:
        # Giving every garment the hand layer must not enable a previously
        # disjoint garment pair. Keep those exclusions with free constraints.
        added_exclusions = 0
        for index, a in enumerate(garment):
            for b in garment[index + 1:]:
                if any(x and y for x, y in zip(a.rigid_body.collision_collections,
                                              b.rigid_body.collision_collections, strict=True)):
                    continue
                obj = bpy.data.objects.new("MotionViewerHandLayerExclusion", None)
                bpy.context.collection.objects.link(obj)
                with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
                    bpy.ops.rigidbody.constraint_add(type="GENERIC")
                constraint = obj.rigid_body_constraint
                constraint.object1, constraint.object2 = a, b
                constraint.disable_collisions = True
                obj.hide_render = True
                obj.hide_set(True)
                added_exclusions += 1
        for obj in garment:
            obj.rigid_body.collision_collections[layer] = True
        report["garment_bodies"] = len(garment)
        report["collision_layer"] = layer
        report["margin_m"] = 0.001
        report["preserved_disjoint_garment_pairs"] = added_exclusions
    else:
        report["reason"] = "no skin vertices with majority hand or finger-bone weight"
    return report


#: Bone-name fragments of the legs on a standard MMD rig, including the D-bones the
#: legs are actually skinned to and the IK targets that drive them.
_LEG_TOKENS = ("足", "ひざ", "膝", "つま先")


def neutral_leg_settle(bpy: Any, armature: Any, *, frame_start: int, settle_frames: int) -> dict:
    """Key the legs from rest into the clip's first pose across the settling frames.

    The first half of the settle holds the legs at the authored rest pose, so the cloth
    hangs straight; the second half eases them into frame ``frame_start``'s pose with a
    smoothstep, so they arrive with no velocity and the skirt is pushed aside by a leg
    moving *under* it rather than seated on top of one. Keys are written only before
    ``frame_start``; the clip's own keys, and every rendered frame, are untouched.
    """
    from mathutils import Quaternion

    scene = bpy.context.scene
    if settle_frames < 4:
        return {"enabled": True, "skipped": "settle too short to blend"}
    from .retarget._animate import _action_fcurves

    action = armature.animation_data.action if armature.animation_data else None
    keyed = {
        curve.data_path.split('"')[1]
        for curve in _action_fcurves(action)
        if curve.data_path.startswith("pose.bones[") and curve.data_path.endswith("rotation_quaternion")
    }
    # Only bones the retarget drives. D-bones and IK targets follow through constraints
    # and carry no keys; keying them here would add channels the clip never had.
    bones = [
        b for b in armature.pose.bones
        if b.name in keyed and b.name[:1] in "左右" and any(t in b.name for t in _LEG_TOKENS)
    ]
    if not bones:
        return {"enabled": True, "skipped": "no leg bones"}
    scene.frame_set(frame_start)
    target = {b.name: (b.rotation_quaternion.copy(), b.location.copy()) for b in bones}
    for bone in bones:
        bone.rotation_mode = "QUATERNION"
    start = frame_start - settle_frames
    hold = settle_frames // 2
    for frame in range(start, frame_start):
        t = (frame - start - hold) / float(settle_frames - hold)
        t = max(0.0, min(1.0, t))
        w = t * t * (3.0 - 2.0 * t)
        for bone in bones:
            quat, loc = target[bone.name]
            bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(quat, w)
            bone.location = loc * w
            bone.keyframe_insert("rotation_quaternion", frame=frame, group=bone.name)
            bone.keyframe_insert("location", frame=frame, group=bone.name)
    scene.frame_set(frame_start)
    return {
        "enabled": True,
        "bones": [b.name for b in bones],
        "rest_hold_frames": hold,
        "blend_frames": settle_frames - hold,
        "first_keyed_frame": start,
        "last_keyed_frame": frame_start - 1,
    }


def build_and_bake(
    bpy: Any,
    armature: Any,
    *,
    frame_start: int,
    frame_end: int,
    fps: float | None = None,
    settings: BulletSettings | None = None,
    scene_box_colliders: tuple[Any, ...] = (),
    scene_weapon_colliders: tuple[Any, ...] = (),
    profile_joint_rules: tuple[Any, ...] = (),
    profile_side_link_rules: tuple[Any, ...] = (),
) -> dict:
    """Build the PMX physics rig on ``armature``'s model and bake the simulation.

    A model without physics data returns ``built=False, baked=False``. A cancelled
    or incomplete bake raises instead of reporting a successful simulation. The
    caller decides whether absent physics is acceptable for its requested output.
    Supplied rectangular props and calibrated weapons get kinematic colliders
    after the model builds, before baking; their geometry and animation are preserved.
    """
    settings = settings or BulletSettings()
    if (not math.isfinite(settings.cloth_angular_slack_degrees)
            or not 0.0 <= settings.cloth_angular_slack_degrees <= 45.0):
        raise ValueError("cloth angular slack must be finite and between 0 and 45 degrees")
    if (not math.isfinite(settings.cloth_linear_slack_m)
            or not 0.0 <= settings.cloth_linear_slack_m <= 0.01):
        raise ValueError("cloth linear slack must be finite and between 0 and 0.01 metres")
    from mmd_tools.core.model import FnModel, Model  # type: ignore

    scene = bpy.context.scene
    # Bullet uses the effective scene rate, including fps_base. Preserve fractional
    # source rates such as 30000/1001 rather than silently simulating them at 30 Hz.
    if fps is not None:
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("physics fps must be finite and positive")
        scene.render.fps = max(1, int(round(fps)))
        scene.render.fps_base = scene.render.fps / float(fps)
    timing = {
        "fps": scene.render.fps / scene.render.fps_base,
        "scene_fps": scene.render.fps,
        "fps_base": scene.render.fps_base,
    }
    damping_mode = "legacy-inverted" if settings.fix_damping else "pmx"

    root = FnModel.find_root_object(armature)
    if root is None:
        return {
            "built": False,
            "baked": False,
            "reason": "no MMD root object; was the PMX imported by mmd_tools?",
            "damping_mode": damping_mode,
            **timing,
        }

    model = Model(root)
    rigid_bodies = list(model.rigidBodies())
    joints = list(model.joints())
    if not rigid_bodies:
        return {
            "built": False,
            "baked": False,
            "reason": "this PMX carries no rigid bodies",
            "damping_mode": damping_mode,
            **timing,
        }
    bake_start = frame_start - settings.settle_frames
    settle_report = (
        neutral_leg_settle(bpy, armature, frame_start=frame_start, settle_frames=settings.settle_frames)
        if settings.neutral_settle
        else {"enabled": False}
    )

    # Stand the character on the frame the simulation will begin from *before* building.
    # ``Model.updateRigid`` seats each body on ``target_bone.matrix``, which is the
    # current pose, so building at whatever frame the previous pass happened to leave
    # behind puts every body where the character was *then*. Measured on a 60-frame
    # clip built at the wrong frame, bodies started 20-40 cm from their bones and the
    # skirt blew outward into a flat sheet on the first simulated step.
    scene.frame_set(bake_start)
    bpy.context.view_layer.update()

    # Build the rig. This disconnects the physics bones, so the retarget must already
    # have been applied — see the module docstring.
    model.build(
        non_collision_distance_scale=settings.non_collision_distance_scale,
        collision_margin=settings.collision_margin,
    )
    cloth_constraints = relax_locked_cloth_angles(model, settings)
    cloth_linear_constraints = relax_locked_cloth_translations(model, settings)
    from .mmd_joint_rules import apply_joint_rules, apply_side_link_rules

    joint_rule_report = apply_joint_rules(model, profile_joint_rules) if profile_joint_rules else {"enabled": False, "rules": []}
    side_link_report = apply_side_link_rules(model, profile_side_link_rules)

    report_damping: dict = {}
    if settings.fix_damping:
        report_damping = correct_damping(bpy, model)
    hand_colliders = add_fitted_hand_colliders(bpy, armature, model, settings)
    box_colliders: dict = {"enabled": False, "bodies": []}
    if scene_box_colliders:
        from .mmd_prop_physics import add_animated_box_colliders

        box_colliders = add_animated_box_colliders(bpy, scene_box_colliders, model)
    weapon_colliders: dict = {"enabled": False, "bodies": []}
    if scene_weapon_colliders:
        from .mmd_weapon_physics import add_animated_weapon_colliders

        weapon_colliders = add_animated_weapon_colliders(bpy, scene_weapon_colliders, model)

    # MMD's gravity convention, not Blender's default.
    scene.gravity = (0.0, 0.0, settings.gravity)

    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    world = scene.rigidbody_world
    world.enabled = True
    world.substeps_per_frame = settings.substeps
    world.solver_iterations = settings.iterations

    # Simulate from before the first rendered frame so the cloth has settled. The scene
    # frame range has to cover the settling period too, or the cache simply has no
    # frames there to bake into.
    cache = world.point_cache
    if cache is None:
        raise RuntimeError("Bullet world has no point cache; physics was not baked")
    original = (scene.frame_start, scene.frame_end)
    try:
        scene.frame_start = bake_start
        scene.frame_end = frame_end
        cache.frame_start = bake_start
        cache.frame_end = frame_end

        scene.frame_set(bake_start)
        bpy.context.view_layer.update()
        with bpy.context.temp_override(scene=scene, point_cache=cache):
            outcome = bpy.ops.ptcache.bake(bake=True)
        if "FINISHED" not in outcome or not cache.is_baked:
            raise RuntimeError("Bullet point-cache bake did not finish; physics is not ready to render")
    finally:
        scene.frame_start, scene.frame_end = original

    return {
        "built": True,
        "baked": True,
        "rigid_bodies": len(rigid_bodies),
        "joints": len(joints),
        "substeps": settings.substeps,
        "iterations": settings.iterations,
        "settled_from": bake_start,
        "baked_to": frame_end,
        **timing,
        "gravity": settings.gravity,
        "damping_mode": damping_mode,
        "damping": report_damping,
        "cloth_constraints": cloth_constraints,
        "cloth_linear_constraints": cloth_linear_constraints,
        "profile_joint_rules": joint_rule_report,
        "profile_side_link_rules": side_link_report,
        "neutral_settle": settle_report,
        "hand_colliders": hand_colliders,
        "scene_box_colliders": box_colliders,
        "scene_weapon_colliders": weapon_colliders,
        "collision_margin_m": settings.collision_margin,
        "non_collision_distance_scale": settings.non_collision_distance_scale,
    }
