"""Post-bake cleanup of garment physics: remove solver snaps, keep the swing.

Runs after ``mmd_bullet.build_and_bake`` and before any frame is rendered. The baked
track of every dynamic body is read, taken into the pelvis frame, cleaned by
``core.cloth_despike``, and written back as keyframes on that body with Bullet switched
off. The PMX's track constraints then read the cleaned bodies exactly as they read
simulated ones, so nothing downstream changes and the source motion is untouched.

Why after the bake and not inside the solver: three solver-side changes were measured
and each made the cloth worse — see ``core.cloth_despike``.

**Hand-through-skirt is deliberately not handled here**, and two attempts are recorded
so they are not repeated. Measured on Nilou's walk, frames 6-15:

- *Per-finger hand colliders* (``BulletSettings.fit_hand_colliders``): a kinematic hand
  shoves the panels, whole-panel snaps rise from 9 frames to 27, and triangle crossings
  *rise* (1565 -> 1791 pairs). Combined with this despike it is worse still — 27
  crossing frames, 4512 pairs — because panels pushed aside swing back through the hand
  later. Note the option selects garment bodies by ``cloth_bone_tokens``; Nilou's skirt
  bones are ``Q_*`` and match none of the defaults.
- *Pushing garment vertices out of the skin* barely moves the count (crossing frames 10 -> 8,
  triangle pairs 1383 -> 1278). The crossings are not vertices behind the hand: the
  garment vertices involved sit 13-17 mm *in front of* the hand normal and 18-89 mm
  away. A skirt triangle there spans several centimetres, wider than the hand, so its
  edge slices through the fingers while every corner stays outside. No per-vertex
  correction can remove an edge passing between its vertices. It needs finer garment
  geometry, which is what the mesh-cloth proxy gives Yoimiya.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.cloth_despike import DespikeSettings, despike, relative_tracks, smooth_tracks


def _garment_bodies(model: Any) -> list[Any]:
    return [
        obj
        for obj in model.rigidBodies()
        if obj.rigid_body is not None and not obj.rigid_body.kinematic and obj.mmd_rigid.type in ("1", "2")
    ]


def _pelvis_matrix(armature: Any) -> np.ndarray:
    bones = armature.pose.bones
    name = "下半身" if "下半身" in bones else "センター"
    return np.asarray(armature.matrix_world @ bones[name].matrix)


def despike_garment(
    bpy: Any,
    armature: Any,
    model: Any,
    *,
    frame_start: int,
    frame_end: int,
    fps: float,
    settings: DespikeSettings = DespikeSettings(),
    despike_enabled: bool = True,
    smooth_sigma: float = 0.0,
) -> dict:
    """Replace Bullet's garment tracks with cleaned keyframes. Returns a report.

    ``despike_enabled`` refits rare whole-panel snaps; ``smooth_sigma`` (frames) removes
    continuous flicker. Either alone, or both: despike first, so a snap is not smeared
    into its neighbours by the low-pass.
    """
    from mathutils import Matrix

    scene = bpy.context.scene
    bodies = _garment_bodies(model)
    if not bodies:
        return {"enabled": True, "bodies": 0, "reason": "no dynamic garment bodies"}
    pelvis, world = [], []
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        pelvis.append(_pelvis_matrix(armature))
        world.append([np.asarray(obj.evaluated_get(depsgraph).matrix_world) for obj in bodies])
    pelvis_arr, world_arr = np.array(pelvis), np.array(world)
    relative = relative_tracks(pelvis_arr, world_arr)
    if despike_enabled:
        cleaned, report = despike(relative, fps, settings)
    else:
        cleaned, report = relative, {"despike": False}
    if smooth_sigma > 0.0:
        cleaned = smooth_tracks(cleaned, smooth_sigma)
    report["smooth_sigma_frames"] = float(smooth_sigma)
    cleaned_world = np.einsum("tij,tnjk->tnik", pelvis_arr, cleaned)

    # Hand the bodies to the keyframes. Kinematic + animated is how Blender lets
    # fcurves drive a rigid body; with the world off the point cache is not consulted,
    # so every frame reads the keys and nothing is simulated twice.
    for obj in bodies:
        obj.rigid_body.kinematic = True
        obj.rotation_mode = "QUATERNION"
        if obj.animation_data and obj.animation_data.action:
            obj.animation_data.action = None
    for index, frame in enumerate(range(frame_start, frame_end + 1)):
        for body, obj in enumerate(bodies):
            matrix = Matrix(cleaned_world[index, body].tolist())
            if obj.parent is not None:
                matrix = obj.parent.matrix_world.inverted() @ matrix
            loc, rot, _ = matrix.decompose()
            obj.location = loc
            obj.rotation_quaternion = rot
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_quaternion", frame=frame)
    if scene.rigidbody_world is not None:
        scene.rigidbody_world.enabled = False
    for obj in bodies:
        action = obj.animation_data.action if obj.animation_data else None
        fcurves = _action_fcurves(action)
        for curve in fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
    scene.frame_set(frame_start)
    report.update({"enabled": True, "written_bodies": len(bodies)})
    return report


def _action_fcurves(action: Any) -> list[Any]:
    """F-curves of an action, across the legacy and layered (Blender 4.4+) layouts."""
    if action is None:
        return []
    curves = list(getattr(action, "fcurves", []) or [])
    if curves:
        return curves
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                curves.extend(bag.fcurves)
    return curves


def cleanup(
    bpy: Any,
    armature: Any,
    *,
    frame_start: int,
    frame_end: int,
    fps: float,
    despike_settings: DespikeSettings = DespikeSettings(),
    despike_enabled: bool = True,
    smooth_sigma: float = 0.0,
) -> dict:
    """Run the post-bake cleanup on ``armature``'s model."""
    from mmd_tools.core.model import FnModel, Model  # type: ignore

    root = FnModel.find_root_object(armature)
    if root is None:
        return {"enabled": False, "reason": "no MMD root"}
    model = Model(root)
    return {
        "despike": despike_garment(
            bpy, armature, model, frame_start=frame_start, frame_end=frame_end, fps=fps, settings=despike_settings,
            despike_enabled=despike_enabled, smooth_sigma=smooth_sigma,
        )
    }
