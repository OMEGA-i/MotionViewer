"""Body-22 SMPL-X kinematics without Blender or model weights.

Bone frames are built once on the rest skeleton and then carried by each
joint's SMPL-X world rotation, so ``S(t, b) = R_g[b] @ S_rest(b)`` holds by
construction.  Retarget calibrations are solved against ``S_rest``, which is
what lets one constant matrix per bone carry a whole transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SMPLX_BODY22_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

SMPLX_BODY22_PARENTS: tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
)

# Child used to aim the bone Y axis. End effectors reuse the incoming edge.
_LOOKAT_CHILD: dict[str, str | None] = {
    "pelvis": "spine1",
    "left_hip": "left_knee",
    "right_hip": "right_knee",
    "spine1": "spine2",
    "left_knee": "left_ankle",
    "right_knee": "right_ankle",
    "spine2": "spine3",
    "left_ankle": "left_foot",
    "right_ankle": "right_foot",
    "spine3": "neck",
    "left_foot": None,
    "right_foot": None,
    "neck": "head",
    "left_collar": "left_shoulder",
    "right_collar": "right_shoulder",
    "head": None,
    "left_shoulder": "left_elbow",
    "right_shoulder": "right_elbow",
    "left_elbow": "left_wrist",
    "right_elbow": "right_wrist",
    "left_wrist": None,
    "right_wrist": None,
}

_NAME_INDEX = {name: index for index, name in enumerate(SMPLX_BODY22_NAMES)}

# Source y-up (x, y, z) -> Blender z-up (x, -z, y).
_SOURCE_TO_BLENDER = np.array(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors to rotation matrices."""
    vectors = np.asarray(axis_angle, dtype=np.float64)
    if vectors.shape[-1] != 3:
        raise ValueError("axis-angle arrays must end with a size-3 axis")
    leading = vectors.shape[:-1]
    flat = vectors.reshape(-1, 3)
    matrices = np.zeros((len(flat), 3, 3), dtype=np.float64)
    for index, vector in enumerate(flat):
        angle = float(np.linalg.norm(vector))
        if angle < 1e-12:
            matrices[index] = np.eye(3)
            continue
        axis = vector / angle
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        outer = np.outer(axis, axis)
        skew = np.array(
            (
                (0.0, -axis[2], axis[1]),
                (axis[2], 0.0, -axis[0]),
                (-axis[1], axis[0], 0.0),
            )
        )
        matrices[index] = cosine * np.eye(3) + sine * skew + (1.0 - cosine) * outer
    return matrices.reshape(*leading, 3, 3)


def axis_angle_from_rotations(matrices: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rodrigues`. Shape ``(..., 3, 3)`` to ``(..., 3)``.

    Routed through a quaternion rather than ``arccos`` of the trace, which loses
    all precision near identity — exactly where a smoothed pose sequence lives —
    and cannot recover the axis at all near a half turn.  The branch picks
    whichever quaternion component is largest, so no denominator approaches zero.
    """
    array = np.asarray(matrices, dtype=np.float64)
    if array.shape[-2:] != (3, 3):
        raise ValueError("rotation arrays must end with a 3x3 block")
    leading = array.shape[:-2]
    flat = array.reshape(-1, 3, 3)
    result = np.zeros((len(flat), 3), dtype=np.float64)
    for index, matrix in enumerate(flat):
        trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
        if trace > -0.5:
            scale = float(np.sqrt(max(trace + 1.0, 0.0))) * 2.0
            w = 0.25 * scale
            vector = np.array(
                (
                    matrix[2, 1] - matrix[1, 2],
                    matrix[0, 2] - matrix[2, 0],
                    matrix[1, 0] - matrix[0, 1],
                )
            ) / max(scale, 1e-300)
        else:
            # Near a half turn the vector part dominates; take it from the largest
            # diagonal entry.
            axis = int(np.argmax(np.diag(matrix)))
            other = [(axis + 1) % 3, (axis + 2) % 3]
            scale = (
                float(
                    np.sqrt(
                        max(
                            1.0
                            + matrix[axis, axis]
                            - matrix[other[0], other[0]]
                            - matrix[other[1], other[1]],
                            0.0,
                        )
                    )
                )
                * 2.0
            )
            vector = np.zeros(3)
            vector[axis] = 0.25 * scale
            vector[other[0]] = (matrix[other[0], axis] + matrix[axis, other[0]]) / max(scale, 1e-300)
            vector[other[1]] = (matrix[other[1], axis] + matrix[axis, other[1]]) / max(scale, 1e-300)
            w = (matrix[other[1], other[0]] - matrix[other[0], other[1]]) / max(scale, 1e-300)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-15:
            continue
        angle = 2.0 * float(np.arctan2(norm, abs(w)))
        result[index] = vector / norm * (angle if w >= 0.0 else -angle)
    return result.reshape(*leading, 3)


def global_rotations(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    """Compose body-22 world rotations. Shape ``(frames, 22, 3, 3)``."""
    roots = rodrigues(np.asarray(global_orient, dtype=np.float64))
    locals_ = rodrigues(np.asarray(body_pose, dtype=np.float64).reshape(len(roots), 21, 3))
    world = np.zeros((len(roots), 22, 3, 3), dtype=np.float64)
    world[:, 0] = roots
    for index, parent in enumerate(SMPLX_BODY22_PARENTS):
        if parent < 0:
            continue
        world[:, index] = world[:, parent] @ locals_[:, index - 1]
    return world


def recover_rest_offsets(
    joints: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
) -> np.ndarray:
    """Recover rest parent-to-child offsets in source space. Shape ``(22, 3)``."""
    posed = np.asarray(joints, dtype=np.float64)
    rotations = global_rotations(global_orient, body_pose)
    offsets = np.zeros((22, 3), dtype=np.float64)
    counts = np.zeros(22, dtype=np.float64)
    for frame in range(len(posed)):
        for index, parent in enumerate(SMPLX_BODY22_PARENTS):
            if parent < 0:
                continue
            delta = posed[frame, index] - posed[frame, parent]
            offsets[index] += rotations[frame, parent].T @ delta
            counts[index] += 1.0
    counts = np.maximum(counts, 1.0)
    return offsets / counts[:, None]


def rest_joints_from_offsets(offsets: np.ndarray, root: np.ndarray | None = None) -> np.ndarray:
    joints = np.zeros((22, 3), dtype=np.float64)
    if root is not None:
        joints[0] = np.asarray(root, dtype=np.float64)
    for index, parent in enumerate(SMPLX_BODY22_PARENTS):
        if parent < 0:
            continue
        joints[index] = joints[parent] + offsets[index]
    return joints


def source_to_blender(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    return array @ _SOURCE_TO_BLENDER.T


def blender_rotation(source_rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(source_rotation, dtype=np.float64)
    return _SOURCE_TO_BLENDER @ rotation @ _SOURCE_TO_BLENDER.T


_TIP_LENGTH_M = 0.35 * 0.08


def _stable_reference(y_axis: np.ndarray) -> np.ndarray:
    """World axis least parallel to the bone, so the rest roll is conditioned.

    Arm bones lie almost exactly along world X.  Projecting a fixed X reference
    onto their perpendicular plane leaves a residual near zero, and normalising
    it turns rounding noise into roll: on real motion this flipped
    ``left_collar`` by up to 133 degrees between neighbouring frames.
    """
    axes = np.eye(3, dtype=np.float64)
    return axes[int(np.argmin(np.abs(axes @ y_axis)))]


def _rest_frame(origin: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Rest frame with Y along the bone and a well-conditioned roll.

    Which roll it picks does not matter: every transfer solves its calibration
    against this frame, so a constant roll cancels exactly.  Only stability
    matters, hence the reference-axis choice above.
    """
    y_axis = tip - origin
    length = float(np.linalg.norm(y_axis))
    y_axis = y_axis / length if length > 1e-9 else np.array((0.0, 0.0, 1.0))
    reference = _stable_reference(y_axis)
    x_axis = reference - y_axis * float(np.dot(reference, y_axis))
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / float(np.linalg.norm(z_axis))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = np.cross(y_axis, z_axis)
    matrix[:3, 1] = y_axis
    matrix[:3, 2] = z_axis
    matrix[:3, 3] = origin
    return matrix


def _bone_tip(rest_joints: np.ndarray, name: str) -> np.ndarray:
    """Rest aim point for one bone's Y axis."""
    child = _LOOKAT_CHILD[name]
    if child is not None:
        return rest_joints[_NAME_INDEX[child]]
    index = _NAME_INDEX[name]
    parent = SMPLX_BODY22_PARENTS[index]
    if parent < 0:
        return rest_joints[index] + np.array((0.0, 0.0, 0.15))
    incoming = rest_joints[index] - rest_joints[parent]
    length = float(np.linalg.norm(incoming))
    if length <= 1e-8:
        return rest_joints[index] + np.array((0.0, 0.0, _TIP_LENGTH_M))
    return rest_joints[index] + incoming * (_TIP_LENGTH_M / length)


def rest_lookat_frames(rest_joints_blender: np.ndarray) -> np.ndarray:
    """Bone frames of the rest skeleton. Shape ``(22, 4, 4)``."""
    joints = np.asarray(rest_joints_blender, dtype=np.float64)
    frames = np.zeros((22, 4, 4), dtype=np.float64)
    for index, name in enumerate(SMPLX_BODY22_NAMES):
        frames[index] = _rest_frame(joints[index], _bone_tip(joints, name))
    return frames


def carry_rest_frames(
    rest_frames: np.ndarray,
    joints_blender: np.ndarray,
    rotations_source: np.ndarray,
) -> np.ndarray:
    """Posed frames as ``S(t, b) = R_g[b] @ S_rest(b)``. Shape ``(T, 22, 4, 4)``.

    Rotations come from SMPL-X directly rather than from posed joint positions.
    Aiming a bone at its posed child looks equivalent — and is, for the Y axis
    up to shape blend shapes — but it leaves the roll undetermined for bones
    that lie along the reference axis, and the roll is exactly what carries
    twist.  Origins still come from the posed joints, so translation is
    unaffected.
    """
    rest = np.asarray(rest_frames, dtype=np.float64)
    joints = np.asarray(joints_blender, dtype=np.float64)
    rotations = np.asarray(rotations_source, dtype=np.float64)
    frames = np.zeros((len(joints), 22, 4, 4), dtype=np.float64)
    for frame in range(len(joints)):
        for index in range(22):
            frames[frame, index, :3, :3] = blender_rotation(rotations[frame, index]) @ rest[index, :3, :3]
            frames[frame, index, :3, 3] = joints[frame, index]
            frames[frame, index, 3, 3] = 1.0
    return frames


@dataclass(frozen=True)
class SmplxLookatMotion:
    names: tuple[str, ...]
    rest_offsets: np.ndarray
    rest_joints_blender: np.ndarray
    posed_joints_blender: np.ndarray
    rest_frames: np.ndarray
    posed_frames: np.ndarray
    root_locations: np.ndarray

    def rest_by_name(self) -> dict[str, np.ndarray]:
        return {name: self.rest_frames[index] for index, name in enumerate(self.names)}

    def posed_matrices(self) -> np.ndarray:
        return self.posed_frames


def build_lookat_motion(
    joints: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray | None = None,
) -> SmplxLookatMotion:
    """Recover rest offsets, then emit Blender-space look-at frames."""
    posed_joints = np.asarray(joints, dtype=np.float64)
    offsets = recover_rest_offsets(posed_joints, global_orient, body_pose)
    rest_source = rest_joints_from_offsets(offsets, root=np.mean(posed_joints[:, 0], axis=0))
    rotations = global_rotations(global_orient, body_pose)
    rest_blender = source_to_blender(rest_source)
    posed_blender = source_to_blender(posed_joints)
    if transl is not None:
        root = source_to_blender(np.asarray(transl, dtype=np.float64))
    else:
        root = posed_blender[:, 0].copy()
    rest_frames = rest_lookat_frames(rest_blender)
    posed_frames = carry_rest_frames(rest_frames, posed_blender, rotations)
    return SmplxLookatMotion(
        names=SMPLX_BODY22_NAMES,
        rest_offsets=offsets,
        rest_joints_blender=rest_blender,
        posed_joints_blender=posed_blender,
        rest_frames=rest_frames,
        posed_frames=posed_frames,
        root_locations=root,
    )
