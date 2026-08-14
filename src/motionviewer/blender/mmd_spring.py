"""Secondary motion for hair, skirts and accessories, as baked spring bones.

The PMX ships rigid bodies and joints for exactly this, and Blender can import
them, but the simulation does not survive contact with animation: it is stable at
rest and diverges once the body moves, because the masses and collision sizes are
authored for MMD's own solver and units — a 4 cm capsule carrying 2.5 kg. Retuning
221 bodies per character is not a bounded job, and a rigid-body cache also forces
the whole pipeline to render frames in order.

So the *selection* is taken from the PMX and the *simulation* is not. A bone whose
rigid body is dynamic (mode 1 or 2) is a bone the artist wanted to swing; those
bones get a damped spring instead, solved here and baked to quaternion keys. That
is deterministic, needs no cache, renders in any order, and behaves the same on
every character.

The spring is the standard one: each bone's tip is pulled toward where it would be
if the bone were rigid, carries its own velocity, and is then projected back onto
its own length so nothing stretches. Divergence is impossible by construction —
the tip cannot leave a sphere around its own head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Helper bones mmd_tools creates; they deform nothing and must not be simulated.
_HELPER_PREFIXES = ("_shadow_", "_dummy_")


@dataclass(frozen=True)
class SpringStyle:
    """How much the hair lags. Per frame at 30 fps."""

    # Pull back toward the rigid pose. Higher is stiffer hair.
    stiffness: float = 0.34
    # Velocity retained per frame. Higher swings longer.
    damping: float = 0.76
    # Extra downward acceleration, in metres per frame squared. Zero by default,
    # and that is not an omission: the model's rest pose is already the hanging
    # pose, so the artist has accounted for gravity. Adding it again drags every
    # bone onto the clamp and the ponytail becomes a vertical bar. What is wanted
    # here is lag, not sag.
    gravity: float = 0.0
    # Hard cap on how far a bone may deviate from its rigid direction, which
    # keeps a fast turn from flinging hair through the head.
    max_angle_degrees: float = 38.0
    # Frames simulated before the first rendered frame so hair starts hanging.
    settle_frames: int = 8
    # Chains longer than this are truncated; nothing useful is that deep.
    max_chain_length: int = 24


def dynamic_spring_bones(armature: Any, rigid_bodies: list[Any]) -> set[str]:
    """Bone names whose PMX rigid body is dynamic.

    ``mmd_rigid.type`` is the PMX physics mode: 0 follows the bone, 1 is driven
    by the simulation, 2 is driven but keeps the bone's position.  Only 1 and 2
    are things that should swing.
    """
    names: set[str] = set()
    bones = armature.data.bones
    for body in rigid_bodies:
        mmd_rigid = getattr(body, "mmd_rigid", None)
        if mmd_rigid is None:
            continue
        if str(getattr(mmd_rigid, "type", "0")) not in {"1", "2"}:
            continue
        bone_name = str(getattr(mmd_rigid, "bone", "") or "")
        if bone_name and bone_name in bones and not bone_name.startswith(_HELPER_PREFIXES):
            names.add(bone_name)
    return names


def build_spring_chains(
    armature: Any, spring_bones: set[str], driven: set[str], *, style: SpringStyle | None = None
) -> list[list[str]]:
    """Group spring bones into parent-first chains rooted under a driven bone.

    A chain needs a driven ancestor, because the whole effect is lag relative to
    something that moves.  Bones the retarget drives are never included: they
    carry the motion and must stay exact.
    """
    settings = style or SpringStyle()
    bones = armature.data.bones
    candidates = {name for name in spring_bones if name not in driven}

    roots = [
        name for name in candidates if bones[name].parent is None or bones[name].parent.name not in candidates
    ]
    chains: list[list[str]] = []
    for root in sorted(roots):
        # Only simulate what hangs off animated geometry.
        ancestor = bones[root].parent
        while ancestor is not None and ancestor.name not in driven:
            ancestor = ancestor.parent
        if ancestor is None:
            continue
        chain = [root]
        while len(chain) < settings.max_chain_length:
            children = [child.name for child in bones[chain[-1]].children if child.name in candidates]
            if len(children) != 1:
                break
            chain.append(children[0])
        chains.append(chain)
    return chains


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    from_vector = source / max(float(np.linalg.norm(source)), 1e-12)
    to_vector = target / max(float(np.linalg.norm(target)), 1e-12)
    cosine = float(np.clip(np.dot(from_vector, to_vector), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    axis = np.cross(from_vector, to_vector)
    sine = float(np.linalg.norm(axis))
    if sine <= 1e-12:
        helper = np.array((1.0, 0.0, 0.0)) if abs(from_vector[0]) < 0.9 else np.array((0.0, 0.0, 1.0))
        axis = np.cross(from_vector, helper)
        sine = float(np.linalg.norm(axis))
    axis = axis / sine
    angle = float(np.arctan2(sine, cosine))
    skew = np.array(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = float(np.sqrt(trace + 1.0)) * 2.0
        quaternion = np.array(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = float(np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
        quaternion = np.array(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = float(np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
        quaternion = np.array(
            (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        )
    else:
        scale = float(np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
        quaternion = np.array(
            (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
        )
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)


def simulate_spring_bones(
    bpy: Any,
    armature: Any,
    chains: list[list[str]],
    *,
    frame_start: int,
    num_frames: int,
    style: SpringStyle | None = None,
) -> dict:
    """Solve the springs over the clip and bake quaternion keys.

    One dependency-graph update per frame: the driven pose is read from Blender,
    then each chain is advanced in NumPy from its root's parent downward, because
    a child's head is its parent's tip and therefore depends on the spring result
    above it.
    """
    settings = style or SpringStyle()
    bones = armature.data.bones
    pose_bones = armature.pose.bones

    chains = [chain for chain in chains if chain]
    if not chains:
        return {"chains": 0, "bones": 0}

    # Rest local matrices and lengths, read once.
    rest_local: dict[str, np.ndarray] = {}
    lengths: dict[str, float] = {}
    world_scale = float(sum(abs(value) for value in armature.matrix_world.to_scale())) / 3.0
    for chain in chains:
        for name in chain:
            bone = bones[name]
            parent_matrix = (
                np.asarray(bone.parent.matrix_local, dtype=np.float64)
                if bone.parent is not None
                else np.eye(4)
            )
            rest_local[name] = np.linalg.inv(parent_matrix) @ np.asarray(bone.matrix_local, dtype=np.float64)
            lengths[name] = float(bone.length)
    _ = world_scale

    for chain in chains:
        for name in chain:
            pose_bones[name].rotation_mode = "QUATERNION"

    tips: dict[str, np.ndarray | None] = {name: None for chain in chains for name in chain}
    velocities: dict[str, np.ndarray] = {
        name: np.zeros(3, dtype=np.float64) for chain in chains for name in chain
    }
    limit = float(np.cos(np.radians(settings.max_angle_degrees)))
    written = 0

    first = frame_start - max(settings.settle_frames, 0)
    for frame in range(first, frame_start + num_frames):
        clamped = min(max(frame, frame_start), frame_start + num_frames - 1)
        bpy.context.scene.frame_set(clamped)
        bpy.context.view_layer.update()
        evaluated = armature.evaluated_get(bpy.context.evaluated_depsgraph_get())
        object_matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)

        for chain in chains:
            parent_bone = bones[chain[0]].parent
            if parent_bone is None:
                continue
            parent_pose = evaluated.pose.bones.get(parent_bone.name)
            if parent_pose is None:
                continue
            carrier = object_matrix @ np.asarray(parent_pose.matrix, dtype=np.float64)

            for name in chain:
                base = carrier @ rest_local[name]
                head = base[:3, 3]
                rigid_direction = base[:3, 1] / max(float(np.linalg.norm(base[:3, 1])), 1e-12)
                length = max(lengths[name], 1e-6)
                rigid_tip = head + rigid_direction * length

                tip = tips[name]
                if tip is None:
                    tip = rigid_tip.copy()
                else:
                    velocity = velocities[name] * settings.damping
                    velocity += (rigid_tip - tip) * settings.stiffness
                    velocity[2] -= settings.gravity
                    tip = tip + velocity

                # Keep the bone its own length: the tip may swing but never stretch.
                offset = tip - head
                norm = float(np.linalg.norm(offset))
                direction = offset / norm if norm > 1e-9 else rigid_direction
                cosine = float(np.dot(direction, rigid_direction))
                if cosine < limit:
                    # Too far from the rigid pose; slide back along the arc.
                    blend = _rotation_between(rigid_direction, direction)
                    axis_angle = np.arccos(np.clip(cosine, -1.0, 1.0))
                    scale = float(np.radians(settings.max_angle_degrees) / max(axis_angle, 1e-9))
                    direction = rigid_direction + (blend @ rigid_direction - rigid_direction) * scale
                    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
                tip = head + direction * length
                velocities[name] = tip - (tips[name] if tips[name] is not None else tip)
                tips[name] = tip

                local = _rotation_between(np.array((0.0, 1.0, 0.0)), base[:3, :3].T @ direction)
                if frame >= frame_start:
                    pose_bone = pose_bones[name]
                    pose_bone.rotation_quaternion = tuple(
                        float(value) for value in _matrix_to_quaternion(local)
                    )
                    pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=name)
                    written += 1

                carried = np.eye(4)
                carried[:3, :3] = local
                carrier = base @ carried

    action = getattr(getattr(armature, "animation_data", None), "action", None)
    if action is not None:
        curves = list(getattr(action, "fcurves", None) or [])
        if not curves:
            for layer in getattr(action, "layers", ()):
                for strip in getattr(layer, "strips", ()):
                    for channelbag in getattr(strip, "channelbags", ()):
                        curves.extend(channelbag.fcurves)
        for curve in curves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"

    return {
        "chains": len(chains),
        "bones": sum(len(chain) for chain in chains),
        "keys_written": written,
        "settle_frames": settings.settle_frames,
        "max_angle_degrees": settings.max_angle_degrees,
    }


def remove_rigid_bodies(bpy: Any) -> int:
    """Delete the imported rigid bodies and joints once their roles are known."""
    scene = bpy.context.scene
    world = scene.rigidbody_world
    removed = 0
    if world is None:
        return 0
    for collection in (world.collection, world.constraints):
        if collection is None:
            continue
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    for obj in list(bpy.data.objects):
        if str(getattr(obj, "mmd_type", "NONE")) in {"RIGID_BODY", "JOINT", "RIGID_GRP_OBJ", "JOINT_GRP_OBJ"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    try:
        bpy.ops.rigidbody.world_remove()
    except RuntimeError:
        pass
    return removed


def keyframed_bones(armature: Any) -> set[str]:
    """Bones the retarget wrote channels for; these carry motion and stay exact."""
    action = getattr(getattr(armature, "animation_data", None), "action", None)
    if action is None:
        return set()
    curves = list(getattr(action, "fcurves", None) or [])
    if not curves:
        for layer in getattr(action, "layers", ()):
            for strip in getattr(layer, "strips", ()):
                for channelbag in getattr(strip, "channelbags", ()):
                    curves.extend(channelbag.fcurves)
    names: set[str] = set()
    for curve in curves:
        path = getattr(curve, "data_path", "")
        if 'pose.bones["' in path:
            names.add(path.split('pose.bones["', 1)[1].split('"]', 1)[0])
    return names


def apply_secondary_motion(
    bpy: Any,
    armature: Any,
    *,
    frame_start: int,
    num_frames: int,
    style: SpringStyle | None = None,
) -> dict:
    """Read the PMX's dynamic bodies, drop them, and bake springs in their place.

    Must run after the retarget has written its channels: the springs lag behind
    the driven pose, so that pose has to exist first.
    """
    world = bpy.context.scene.rigidbody_world
    bodies = list(world.collection.objects) if world is not None and world.collection is not None else []
    if not bodies:
        return {"chains": 0, "bones": 0, "reason": "no rigid bodies; import with physics=True"}

    spring_bones = dynamic_spring_bones(armature, bodies)
    driven = keyframed_bones(armature)
    chains = build_spring_chains(armature, spring_bones, driven, style=style)
    removed = remove_rigid_bodies(bpy)
    report = simulate_spring_bones(
        bpy,
        armature,
        chains,
        frame_start=frame_start,
        num_frames=num_frames,
        style=style,
    )
    report["dynamic_bones_declared"] = len(spring_bones)
    report["rigid_objects_removed"] = removed
    return report
