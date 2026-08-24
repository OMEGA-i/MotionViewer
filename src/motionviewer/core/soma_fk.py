"""SOMA-77 kinematics, in the form the MMD solver already consumes.

Why this exists: the render path was fed SMPL-X body-22, which is a *fit* to the
SOMA data. Measured on the probe set, that fit costs **26.1-28.3 mm RMSE** and
discards 55 of the 77 joints — both hands (52 joints) and the toes (4). Reading SOMA
directly removes the fit entirely and brings the fingers with it.

Three things about the format had to be established by measurement, because the
first two are easy to get wrong and one contradicts the data's own README:

1. **``rotmat77`` is a *local* rotation**, relative to each joint's rest orientation
   (the ``relative_to_soma_joint_orient`` convention). Treating it as a world
   rotation reproduces bone directions to only 11-19 deg. Chaining it down the tree
   reproduces them to **0.03 deg**.
2. **The rest direction of each bone is recoverable exactly.** Solving
   ``d_rest = mean_t(W[t, parent]^T @ observed_direction(t))`` and renormalising gives
   a direction that reproduces every observed bone direction to 0.03 deg, and the
   directions agree across the five probe clips to 0.001 deg — they are a property of
   the skeleton, not of the clip.
3. **Rigid forward kinematics closes exactly.** With chained world rotations, those
   rest directions, and the measured ``bone_lengths``, reconstructing positions from
   the root gives **0.00 mm** error against ``joints77``. The README warns that rigid
   FK drifts 14.4 mm; that figure comes from treating the stored rotations as world
   rotations. Chained, the skeleton is rigid to 0.0003 mm per frame.

The rest pose is an exact T-pose: arms horizontal to 0.0 deg, the same convention
SMPL-X uses, so the transfer-mode reasoning carries over unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .smplx_fk import _TIP_LENGTH_M, _rest_frame, blender_rotation, source_to_blender

# The joint order every probe file carries. Kept here so a file with a different
# layout is rejected rather than silently retargeted through the wrong tree.
SOMA77_NAMES: tuple[str, ...] = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4",
    "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
)  # fmt: skip

# Child each bone's Y axis aims at, where the joint has more than one and the choice
# is not obvious. Any consistent choice retargets correctly, because the same frame is
# used on the rest and the posed side; these are picked so a bone points along the
# limb a reader would expect.
SOMA77_PRIMARY_CHILD: dict[str, str] = {
    "Hips": "Spine1",
    "Chest": "Neck1",
    "Head": "HeadEnd",
    "LeftHand": "LeftHandMiddle1",
    "RightHand": "RightHandMiddle1",
}


def joint_index(names: tuple[str, ...] | list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


def check_layout(names: tuple[str, ...] | list[str], parents: np.ndarray) -> None:
    """Reject a file whose joint tree is not the one this module was written for."""
    if tuple(str(name) for name in names) != SOMA77_NAMES:
        raise ValueError("SOMA joint_names do not match the expected 77-joint layout")
    index_of = joint_index(SOMA77_NAMES)
    for child, parent in enumerate(np.asarray(parents, dtype=int)):
        if child == 0:
            if parent != -1:
                raise ValueError("SOMA joint 0 must be the root")
            continue
        if not 0 <= parent < child:
            raise ValueError(f"SOMA parent of {SOMA77_NAMES[child]} is not an earlier joint")
    if index_of["Hips"] != 0:
        raise ValueError("SOMA root must be Hips")


def world_rotations(local_rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Chain per-joint local rotations into world rotations. ``(T, J, 3, 3)``.

    This is the step the format's name (``relative_to_soma_joint_orient``) implies and
    that the numbers confirm: chained, bone directions reproduce to 0.03 deg; unchained
    they are 11-19 deg out.
    """
    local = np.asarray(local_rotations, dtype=np.float64)
    parents = np.asarray(parents, dtype=int)
    world = np.empty_like(local)
    for joint, parent in enumerate(parents):
        if parent < 0:
            world[:, joint] = local[:, joint]
        else:
            world[:, joint] = np.einsum("tij,tjk->tik", world[:, parent], local[:, joint])
    return world


def recover_rest_directions(joints: np.ndarray, world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Unit rest direction of each bone in its parent's rest frame. ``(J, 3)``.

    Each bone's observed direction is ``W[t, parent] @ d_rest``, so ``d_rest`` is
    recovered by pulling the observation back through the parent's rotation and
    averaging over the clip. Averaging then renormalising is what makes this robust:
    the per-frame estimates agree to well under a degree, so the mean is not fighting
    outliers, and the result is identical across clips of the same skeleton.
    """
    points = np.asarray(joints, dtype=np.float64)
    rotations = np.asarray(world, dtype=np.float64)
    parents = np.asarray(parents, dtype=int)
    directions = np.zeros((points.shape[1], 3), dtype=np.float64)
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        delta = points[:, child] - points[:, parent]
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        observed = delta / np.maximum(length, 1e-12)
        local = np.einsum("tji,tj->ti", rotations[:, parent], observed)
        mean = local.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        directions[child] = mean / norm if norm > 1e-12 else np.array([0.0, 1.0, 0.0])
    return directions


def rest_joints_from_directions(
    directions: np.ndarray, bone_lengths: np.ndarray, parents: np.ndarray
) -> np.ndarray:
    """Rest skeleton positions, root at the origin. ``(J, 3)``.

    Lengths come from ``bone_lengths``, measured off ``joints77``, rather than from the
    identity-driven rest the file also carries: that one is 6.6% out (forearm -14.8 mm,
    shin +11.2 mm) because the official skeleton transfer is an RBF deformation, not a
    rigid bind.
    """
    unit = np.asarray(directions, dtype=np.float64)
    lengths = np.asarray(bone_lengths, dtype=np.float64)
    parents = np.asarray(parents, dtype=int)
    rest = np.zeros_like(unit)
    for child, parent in enumerate(parents):
        if parent >= 0:
            rest[child] = rest[parent] + unit[child] * lengths[child]
    return rest


def forward_kinematics(
    rest: np.ndarray, world: np.ndarray, parents: np.ndarray, root: np.ndarray
) -> np.ndarray:
    """Rigid FK from the rest skeleton and world rotations. ``(T, J, 3)``.

    Only used to *verify* the reconstruction against ``joints77``; positions for the
    retarget come from ``joints77`` itself.
    """
    rest = np.asarray(rest, dtype=np.float64)
    rotations = np.asarray(world, dtype=np.float64)
    parents = np.asarray(parents, dtype=int)
    out = np.zeros((len(rotations), rest.shape[0], 3), dtype=np.float64)
    out[:, 0] = np.asarray(root, dtype=np.float64)
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        offset = rest[child] - rest[parent]
        out[:, child] = out[:, parent] + np.einsum("tij,j->ti", rotations[:, parent], offset)
    return out


def _aim_point(rest: np.ndarray, index: int, parents: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Rest point this bone's Y axis looks at."""
    name = names[index]
    preferred = SOMA77_PRIMARY_CHILD.get(name)
    if preferred is not None:
        return rest[names.index(preferred)]
    children = [child for child, parent in enumerate(parents) if parent == index]
    if children:
        return rest[children[0]]
    parent = int(parents[index])
    if parent < 0:
        return rest[index] + np.array((0.0, 0.0, _TIP_LENGTH_M))
    incoming = rest[index] - rest[parent]
    length = float(np.linalg.norm(incoming))
    if length <= 1e-8:
        return rest[index] + np.array((0.0, 0.0, _TIP_LENGTH_M))
    return rest[index] + incoming * (_TIP_LENGTH_M / length)


def rest_lookat_frames(rest_blender: np.ndarray, parents: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Bone frames of the rest skeleton, in Blender space. ``(J, 4, 4)``."""
    rest = np.asarray(rest_blender, dtype=np.float64)
    frames = np.zeros((rest.shape[0], 4, 4), dtype=np.float64)
    for index in range(rest.shape[0]):
        frames[index] = _rest_frame(rest[index], _aim_point(rest, index, parents, names))
    return frames


def carry_rest_frames(rest_frames: np.ndarray, joints_blender: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Posed frames as ``S(t, b) = R_g[b] @ S_rest(b)``. ``(T, J, 4, 4)``.

    The same invariant the SMPL-X path is built on, which is why the MMD plan and
    solver need no changes to accept SOMA: orientation comes from the rotations,
    origins from the posed joints.
    """
    rest = np.asarray(rest_frames, dtype=np.float64)
    points = np.asarray(joints_blender, dtype=np.float64)
    rotations = np.asarray(world, dtype=np.float64)
    frames = np.zeros((len(points), rest.shape[0], 4, 4), dtype=np.float64)
    for frame in range(len(points)):
        for index in range(rest.shape[0]):
            frames[frame, index, :3, :3] = blender_rotation(rotations[frame, index]) @ rest[index, :3, :3]
            frames[frame, index, :3, 3] = points[frame, index]
            frames[frame, index, 3, 3] = 1.0
    return frames


@dataclass(frozen=True)
class SomaLookatMotion:
    """Same surface as ``SmplxLookatMotion``, so the MMD stages accept either."""

    names: tuple[str, ...]
    parents: np.ndarray
    rest_joints_blender: np.ndarray
    posed_joints_blender: np.ndarray
    rest_frames: np.ndarray
    posed_frames: np.ndarray
    root_locations: np.ndarray
    rest_directions: np.ndarray
    bone_lengths: np.ndarray

    # The polish pass addresses joints by their SMPL-X names; this is what lets it find
    # SOMA's. Without it the 12-degree arm abduction never runs and the hands clip into
    # the torso, which is exactly how the first SOMA renders looked.
    polish_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "spine3": "Chest",
            "left_collar": "LeftShoulder",
            "right_collar": "RightShoulder",
            "left_shoulder": "LeftArm",
            "left_elbow": "LeftForeArm",
            "left_wrist": "LeftHand",
            "right_shoulder": "RightArm",
            "right_elbow": "RightForeArm",
            "right_wrist": "RightHand",
        }
    )

    def rest_by_name(self) -> dict[str, np.ndarray]:
        return {name: self.rest_frames[index] for index, name in enumerate(self.names)}

    def posed_matrices(self) -> np.ndarray:
        return self.posed_frames


def build_soma_motion(payload: dict) -> SomaLookatMotion:
    """Everything the MMD retarget needs, from one SOMA clip.

    ``payload`` is an opened ``soma.npz``: ``joints77``, ``rotmat77``, ``transl``,
    ``bone_lengths``, ``joint_names``, ``joint_parents``.
    """
    names = tuple(str(name) for name in payload["joint_names"])
    parents = np.asarray(payload["joint_parents"], dtype=int)
    check_layout(names, parents)

    joints = np.asarray(payload["joints77"], dtype=np.float64)
    world = world_rotations(np.asarray(payload["rotmat77"], dtype=np.float64), parents)
    directions = recover_rest_directions(joints, world, parents)
    lengths = np.asarray(payload["bone_lengths"], dtype=np.float64)
    rest_source = rest_joints_from_directions(directions, lengths, parents)

    rest_blender = source_to_blender(rest_source)
    posed_blender = source_to_blender(joints)
    frames = rest_lookat_frames(rest_blender, parents, names)
    return SomaLookatMotion(
        names=names,
        parents=parents,
        rest_joints_blender=rest_blender,
        posed_joints_blender=posed_blender,
        rest_frames=frames,
        posed_frames=carry_rest_frames(frames, posed_blender, world),
        root_locations=source_to_blender(np.asarray(payload["transl"], dtype=np.float64)),
        rest_directions=directions,
        bone_lengths=lengths,
    )
