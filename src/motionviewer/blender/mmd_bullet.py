"""Run the PMX's own Bullet physics, so a skirt is pushed aside instead of walked through.

A skinned skirt has no idea a leg exists. It follows bone weights, the leg follows the
motion, and where they overlap the leg comes out the other side. MMD does not have this
problem because a PMX ships **rigid bodies and joints** for exactly these parts — the
skirt panels, the hair, the ribbons — and MMD players simulate them with Bullet against
collision bodies on the legs and torso. Official games do the same thing with runtime
cloth. Ours skinned them rigidly, so the clipping was structural rather than a matter of
tuning.

``mmd_tools`` can build that rig, and this module drives it. What it does is the same
sequence MMD does: disconnect the physics bones from their parents so the bodies own
them, build the rigid bodies and the joints, enable the scene's rigid body world, and
bake the point cache once the retarget's bone animation is already in place.

The order is not negotiable. The physics bones stop being parented once the rig is
built, so anything that poses the armature has to happen **first** — build after the
retarget, never before. And the cache has to be baked rather than left to accumulate as
frames are rendered: the framing pass jumps around the timeline, and a rigid body cache
that sees frames out of order produces garbage.

This is why the importer previously read the rigid bodies for their metadata only and
switched the world off. Half of this is worse than none: with the bodies absent but the
bones already disconnected, the physics bones fall back to the armature origin and smear
across the frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BulletSettings:
    """Simulation quality. Defaults chosen for cloth-like chains on a walking figure."""

    #: Bullet substeps per frame. MMD itself runs a fixed 120 Hz internally; at 50 fps
    #: source that is between two and three substeps, and skirts on a fast turn need
    #: more than the Blender default of 10 to stop panels crossing each other.
    substeps: int = 12
    #: Solver iterations. Long chains of joints — a skirt panel is four or five bodies
    #: deep — need more than the default 10 or the chain stretches visibly.
    iterations: int = 24
    #: Frames simulated before the first rendered frame, so the cloth is hanging under
    #: gravity rather than starting from the bind pose and dropping into place on
    #: camera. MMD users call this the settling period.
    settle_frames: int = 30
    #: Passed straight to ``Model.build``. The margin keeps thin panels from tunnelling.
    collision_margin: float = 1e-06
    non_collision_distance_scale: float = 1.5
    #: Translate the PMX's damping instead of copying the number. See
    #: ``correct_damping`` — the two formats mean opposite things by the same value.
    fix_damping: bool = True
    #: Scene gravity. MMD's own convention, which ``mmd_tools`` carries as a commented
    #: out ``-9.81 * 10 * scale``; at the 0.08 import scale that is -7.848, against
    #: Blender's default -9.81.
    gravity: float = -7.848


def correct_damping(bpy: Any, model: Any) -> dict:
    """Convert MMD's velocity attenuation into Blender's damping.

    The two formats mean opposite things by the same number, and ``mmd_tools`` copies it
    across unchanged:

        MMD      v(t+1) = v(t) * attenuation      0.9999 barely damps
        Blender  v(t+1) = v(t) * (1 - damping)    0.9999 removes almost everything

    Yoimiya's skirt panels carry ``velocity_attenuation = 0.9999`` and
    ``rotation_attenuation = 1.0`` — the author asking for cloth that keeps its momentum.
    Imported literally they become bodies with damping 1.0, the maximum Blender allows,
    which pins them in place. A pinned body cannot follow its bone, so the joints drag it
    and the hem whips outward: measured, the skirt stands 25% further off the legs with
    physics on than with it off, and no amount of extra damping helped because damping was
    already saturated.

    Returns what was changed, so a caller can report it.
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


def build_and_bake(
    bpy: Any,
    armature: Any,
    *,
    frame_start: int,
    frame_end: int,
    fps: float | None = None,
    settings: BulletSettings | None = None,
) -> dict:
    """Build the PMX physics rig on ``armature``'s model and bake the simulation.

    Returns a report; raises nothing on a model without physics data, since plenty of
    accessory PMX files have none and that is not an error.
    """
    settings = settings or BulletSettings()
    from mmd_tools.core.model import FnModel, Model  # type: ignore

    root = FnModel.find_root_object(armature)
    if root is None:
        return {"built": False, "reason": "no MMD root object; was the PMX imported by mmd_tools?"}

    model = Model(root)
    rigid_bodies = list(model.rigidBodies())
    joints = list(model.joints())
    if not rigid_bodies:
        return {"built": False, "reason": "this PMX carries no rigid bodies"}

    scene = bpy.context.scene
    # Bullet takes its timestep from the scene frame rate, so a wrong rate is a wrong
    # simulation. It was found sitting at Blender's default 24 against 50 fps motion,
    # which advances the bones a 50 fps step inside a 24 fps tick and tears the cloth
    # apart. The renderer sets it, but anything else that drives a bake would not, so the
    # rate is asserted here rather than assumed.
    if fps is not None:
        scene.render.fps = int(round(fps))
        scene.render.fps_base = 1.0
    bake_start = frame_start - settings.settle_frames

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

    report_damping: dict = {}
    if settings.fix_damping:
        report_damping = correct_damping(bpy, model)

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
    original = (scene.frame_start, scene.frame_end)
    scene.frame_start = bake_start
    scene.frame_end = frame_end
    if world.point_cache is not None:
        world.point_cache.frame_start = bake_start
        world.point_cache.frame_end = frame_end

    scene.frame_set(bake_start)
    bpy.context.view_layer.update()
    with bpy.context.temp_override(scene=scene, point_cache=world.point_cache):
        bpy.ops.ptcache.bake(bake=True)
    scene.frame_start, scene.frame_end = original

    return {
        "built": True,
        "rigid_bodies": len(rigid_bodies),
        "joints": len(joints),
        "substeps": settings.substeps,
        "iterations": settings.iterations,
        "settled_from": bake_start,
        "baked_to": frame_end,
        "fps": scene.render.fps,
        "gravity": settings.gravity,
        "damping": report_damping,
    }
