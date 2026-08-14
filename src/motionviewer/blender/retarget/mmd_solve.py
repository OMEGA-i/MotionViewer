"""Pure NumPy MMD retarget solve over the real bone chain.

The source look-at frames satisfy ``S(t, b) = R_g[b] @ S_rest(b)``: a bone's
frame is its rest frame carried by that joint's SMPL-X world rotation.  Every
transfer therefore collapses to one constant per-bone calibration ``C`` with
``W(t, b) = S(t, b) @ C_b``, and the only real decision is what ``C_b`` is.

``relative``
    ``C = S_rest^-1 @ target_rest``, i.e. ``W = R_g @ target_rest``.  The
    character keeps its own rest orientation and receives the source's rotation
    about it.  Correct wherever the two rigs already agree on a bone's rest
    direction (torso, legs: 1.5-5.8 deg apart on this rig) and required wherever
    the target's rest axis is a pure convention rather than an anatomical aim
    (``腰`` displays 47 deg off vertical, ``頭`` is a vertical bone while the
    SMPL-X neck-to-head edge leans 17 deg forward).

``absolute``
    ``C = Roll_y(theta)``.  Because a roll about Y cannot move the Y axis,
    ``W`` keeps the source bone direction exactly while the source's rotation
    about the bone axis survives untouched.  ``theta`` is solved once at rest so
    the character's own roll — its elbow hinge plane and 捩 axes — is preserved.
    Required on arms: MMD binds A-pose and SMPL-X rests in T-pose, 23-51 deg
    apart, so a relative transfer would add the droop twice and fold the arms
    into the torso.

``twist``
    A 捩 bone.  It receives its swing bone's rotation about the bone axis so
    ``腕捩1/2/3`` spread the skin twist instead of candy-wrapping one joint.

``passthrough``
    An undriven bone on the path to a driven one (``センター``, ``左肩P``,
    ``左肩C``, ``下半身``).  It contributes only its rest offset.

Local bases are reconstructed against each bone's **actual** parent, so an
inserted 捩 bone is an explicit node rather than an assumption that a mapped
parent telescopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .twist import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    swing_twist_decompose,
)

if TYPE_CHECKING:
    from .mmd import MmdPolishOptions

TransferMode = Literal["relative", "absolute", "twist", "passthrough"]

_BONE_AXIS = np.array((0.0, 1.0, 0.0), dtype=np.float64)


@dataclass(frozen=True)
class MmdChannel:
    """One Blender pose channel in parent-before-child order."""

    name: str
    parent: int
    rest_local: np.ndarray  # (3, 3) rest rotation relative to the actual parent
    mode: TransferMode
    source: str = ""
    source_index: int = -1
    calibration: np.ndarray = field(default_factory=lambda: np.eye(3))
    twist_partner: int = -1  # swing channel -> its 捩 channel
    swing_of: int = -1  # 捩 channel -> its swing channel

    @property
    def driven(self) -> bool:
        return self.mode in {"relative", "absolute"}


@dataclass(frozen=True)
class MmdRetargetPlan:
    channels: tuple[MmdChannel, ...]
    source_names: tuple[str, ...]
    target_rest_global: np.ndarray  # (C, 3, 3) rest world rotations, for audits
    source_rest_global: np.ndarray  # (C, 3, 3) aligned to channels, identity when undriven
    root_translation_scale: float = 1.0

    def index_of(self, name: str) -> int:
        for index, channel in enumerate(self.channels):
            if channel.name == name:
                return index
        return -1


@dataclass(frozen=True)
class MmdSolveResult:
    channel_names: tuple[str, ...]
    local_quaternions_wxyz: np.ndarray  # (T, C, 4)
    world_rotations: np.ndarray  # (T, C, 3, 3)
    root_locations: np.ndarray  # (T, 3)


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest rotation carrying ``source`` onto ``target``."""
    from_vector = np.asarray(source, dtype=np.float64).reshape(3)
    to_vector = np.asarray(target, dtype=np.float64).reshape(3)
    from_norm = float(np.linalg.norm(from_vector))
    to_norm = float(np.linalg.norm(to_vector))
    if from_norm <= 1e-12 or to_norm <= 1e-12:
        return np.eye(3)
    from_vector = from_vector / from_norm
    to_vector = to_vector / to_norm
    cosine = float(np.clip(np.dot(from_vector, to_vector), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    axis = np.cross(from_vector, to_vector)
    sine = float(np.linalg.norm(axis))
    if sine <= 1e-12:
        # Antiparallel: any perpendicular axis gives the same 180 degree turn.
        helper = np.array((1.0, 0.0, 0.0)) if abs(from_vector[0]) < 0.9 else np.array((0.0, 0.0, 1.0))
        axis = np.cross(from_vector, helper)
        axis = axis / float(np.linalg.norm(axis))
        return _axis_angle_matrix(axis, np.pi)
    return _axis_angle_matrix(axis / sine, float(np.arctan2(sine, cosine)))


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    skew = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=np.float64,
    )
    return cosine * np.eye(3) + sine * skew + (1.0 - cosine) * np.outer(axis, axis)


def roll_matrix(angle: float) -> np.ndarray:
    """Rotation about the bone axis (+Y)."""
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.array(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


def relative_calibration(source_rest: np.ndarray, target_rest: np.ndarray) -> np.ndarray:
    """``C`` such that ``S(t) @ C == R_g @ target_rest``."""
    return np.asarray(source_rest, dtype=np.float64).T @ np.asarray(target_rest, dtype=np.float64)


def absolute_calibration(source_rest: np.ndarray, target_rest: np.ndarray) -> np.ndarray:
    """``C = Roll_y(theta)`` keeping the source aim and the target's own roll.

    ``target_rest = P @ source_rest @ Roll_y(theta)`` where ``P`` is the
    smallest rotation between the two rest aims, so ``theta`` is read straight
    off ``source_rest^T P^T target_rest``.
    """
    source = np.asarray(source_rest, dtype=np.float64)
    target = np.asarray(target_rest, dtype=np.float64)
    pole = rotation_between(source[:, 1], target[:, 1])
    residual = source.T @ pole.T @ target
    return roll_matrix(float(np.arctan2(residual[0, 2], residual[0, 0])))


def calibration_for(mode: TransferMode, source_rest: np.ndarray, target_rest: np.ndarray) -> np.ndarray:
    if mode == "absolute":
        return absolute_calibration(source_rest, target_rest)
    if mode == "relative":
        return relative_calibration(source_rest, target_rest)
    return np.eye(3)


def scale_rotation(matrix: np.ndarray, factor: float, limit_radians: float | None = None) -> np.ndarray:
    """Shrink a rotation toward identity along its own axis."""
    quaternion = matrix_to_quaternion(matrix)
    angle = 2.0 * float(np.arccos(np.clip(float(quaternion[0]), -1.0, 1.0)))
    if angle <= 1e-9:
        return np.eye(3)
    axis = np.asarray(quaternion[1:], dtype=np.float64) / float(np.sin(angle * 0.5))
    axis = _normalize_vector(axis)
    scaled = angle * float(factor)
    if limit_radians is not None:
        scaled = float(np.clip(scaled, -abs(limit_radians), abs(limit_radians)))
    return _axis_angle_matrix(axis, scaled)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-12 else np.array((0.0, 1.0, 0.0))


def damp_source_local_rotation(
    source_frames: np.ndarray,
    source_rest: np.ndarray,
    joint: int,
    parent: int,
    *,
    factor: float,
    limit_degrees: float | None = None,
) -> np.ndarray:
    """Shrink one joint's rotation relative to its parent, in source space.

    Monocular fits dump shoulder motion into the clavicle because it is barely
    constrained: on these clips the collar averages 9-28 deg and peaks at 78,
    where a real clavicle manages about 20.  On a stylised rig ``肩`` carries
    real mesh weight, so that reads as a hunched, yanked shoulder.

    Only the named joint is touched.  Arms are transferred in ``absolute`` mode,
    so their world orientation does not depend on the collar and stays exact;
    what changes is the shoulder's deformation and the arm root's position.
    """
    frames = np.asarray(source_frames, dtype=np.float64).copy()
    rest = np.asarray(source_rest, dtype=np.float64)
    limit = None if limit_degrees is None else float(np.radians(limit_degrees))
    joint_rest = rest[joint, :3, :3]
    parent_rest = rest[parent, :3, :3]
    for index in range(len(frames)):
        joint_world = frames[index, joint, :3, :3] @ joint_rest.T
        parent_world = frames[index, parent, :3, :3] @ parent_rest.T
        local = parent_world.T @ joint_world
        frames[index, joint, :3, :3] = parent_world @ scale_rotation(local, factor, limit) @ joint_rest
    return frames


def abduct_source_arm(
    source_frames: np.ndarray,
    chain: tuple[int, ...],
    aim_joint: int,
    outward_joint: int,
    *,
    degrees: float,
) -> np.ndarray:
    """Swing a whole arm away from the torso midline.

    Angle-exact retargeting still buries the arms of a stylised character: on
    this rig the shoulders are 47% narrower than SMPL-X's relative to height and
    the arm is 20% shorter, while the costume is not narrower at all.  An arm
    that hangs at the source's angle therefore starts much closer to the midline
    and ends up inside the kimono.

    The whole chain is rotated rigidly about the shoulder, so elbow bend and
    every relative angle inside the arm are untouched — only where the arm
    hangs changes.  The correction is scaled by how far the arm is from pointing
    outward, so a raised or extended arm, which never clips, is left exactly on
    the source.
    """
    frames = np.asarray(source_frames, dtype=np.float64).copy()
    limit = float(np.radians(degrees))
    if abs(limit) <= 1e-9:
        return frames
    for index in range(len(frames)):
        aim = _normalize_vector(frames[index, aim_joint, :3, 1])
        outward = _normalize_vector(frames[index, outward_joint, :3, 1])
        axis = np.cross(aim, outward)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-6:
            continue
        separation = float(np.arccos(np.clip(float(np.dot(aim, outward)), -1.0, 1.0)))
        weight = float(np.clip(separation / (0.5 * np.pi), 0.0, 1.0))
        rotation = _axis_angle_matrix(axis / norm, limit * weight)
        for joint in chain:
            frames[index, joint, :3, :3] = rotation @ frames[index, joint, :3, :3]
    return frames


def smooth_source_twist(
    source_frames: np.ndarray,
    bone_indices: tuple[int, ...],
    *,
    window: int = 5,
) -> np.ndarray:
    """Smooth each listed bone's roll about its own axis, leaving the aim exact.

    Monocular SMPL-X fits barely constrain axial rotation, because it is nearly
    invisible in 2D.  On the bundled walk clip the upper arm's roll jumps up to
    29 degrees between neighbouring frames at 30 fps — 870 deg/s of pronation,
    which is jitter rather than motion.  Swing is left untouched: bone direction
    is the observable part and the retarget's whole guarantee rests on it.

    The roll is measured against a parallel transport of frame 0, so a bone that
    swings through a large arc does not accumulate phantom twist.
    """
    frames = np.asarray(source_frames, dtype=np.float64).copy()
    half = max(int(window), 1) // 2
    if half == 0:
        return frames
    for bone in bone_indices:
        rotations = frames[:, bone, :3, :3]
        total = len(rotations)
        if total < 3:
            continue
        transported = np.empty_like(rotations)
        transported[0] = rotations[0]
        for index in range(1, total):
            swing = rotation_between(rotations[index - 1][:, 1], rotations[index][:, 1])
            carried = swing @ transported[index - 1]
            transported[index] = quaternion_to_matrix(matrix_to_quaternion(carried))
        rolls = np.zeros(total, dtype=np.float64)
        for index in range(total):
            axis = rotations[index][:, 1]
            reference = transported[index][:, 0]
            current = rotations[index][:, 0]
            rolls[index] = np.arctan2(
                float(np.dot(np.cross(reference, current), axis)),
                float(np.dot(reference, current)),
            )
        rolls = np.unwrap(rolls)
        padded = np.pad(rolls, (half, half), mode="edge")
        kernel = np.ones(2 * half + 1, dtype=np.float64) / float(2 * half + 1)
        smoothed = np.convolve(padded, kernel, mode="valid")
        for index in range(total):
            axis = rotations[index][:, 1]
            frames[index, bone, :3, :3] = (
                _axis_angle_matrix(axis, float(smoothed[index])) @ transported[index]
            )
    return frames


def polish_source_frames(
    source_frames: np.ndarray,
    names: tuple[str, ...],
    rest_frames: np.ndarray,
    options: MmdPolishOptions,
) -> np.ndarray:
    """Apply every look-focused departure from the source, in one place.

    The retarget itself is exact; this is where fidelity is knowingly traded for
    appearance.  Keeping it in one function means the validator can reproduce
    precisely what the pipeline fed its solver, instead of measuring the polish
    and calling it transfer error.
    """
    frames = np.asarray(source_frames, dtype=np.float64)
    if not getattr(options, "enabled", False):
        return frames
    index_of = {name: index for index, name in enumerate(names)}

    if options.collar_damping < 1.0:
        for collar in ("left_collar", "right_collar"):
            if collar in index_of and "spine3" in index_of:
                frames = damp_source_local_rotation(
                    frames,
                    rest_frames,
                    index_of[collar],
                    index_of["spine3"],
                    factor=options.collar_damping,
                    limit_degrees=options.collar_limit_degrees,
                )

    if abs(options.arm_abduction_degrees) > 1e-9:
        for side in ("left", "right"):
            chain = tuple(
                index_of[f"{side}_{part}"]
                for part in ("shoulder", "elbow", "wrist")
                if f"{side}_{part}" in index_of
            )
            collar = f"{side}_collar"
            if len(chain) == 3 and collar in index_of:
                frames = abduct_source_arm(
                    frames, chain, chain[0], index_of[collar], degrees=options.arm_abduction_degrees
                )

    if options.twist_window > 1:
        smoothed = tuple(
            index_of[name]
            for name in (
                "left_shoulder",
                "left_elbow",
                "left_wrist",
                "right_shoulder",
                "right_elbow",
                "right_wrist",
            )
            if name in index_of
        )
        frames = smooth_source_twist(frames, smoothed, window=options.twist_window)
    return frames


def solve_mmd_retarget(
    plan: MmdRetargetPlan,
    source_frames: np.ndarray,
    root_locations: np.ndarray,
) -> MmdSolveResult:
    """Reconstruct one local quaternion per channel per frame.

    ``source_frames`` is ``(T, S, 4, 4)`` indexed by ``plan.source_names``.
    """
    frames = np.asarray(source_frames, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[-2:] != (4, 4):
        raise ValueError("source_frames must have shape (frames, sources, 4, 4)")
    channel_count = len(plan.channels)
    total = len(frames)
    locals_wxyz = np.zeros((total, channel_count, 4), dtype=np.float64)
    locals_wxyz[..., 0] = 1.0
    worlds = np.zeros((total, channel_count, 3, 3), dtype=np.float64)

    for frame in range(total):
        # A 捩 channel is solved when its swing bone is, one node earlier.
        pending_twist: dict[int, np.ndarray] = {}
        for index, channel in enumerate(plan.channels):
            parent_world = np.eye(3) if channel.parent < 0 else worlds[frame, channel.parent]
            base = parent_world @ channel.rest_local
            if channel.mode == "passthrough":
                basis = np.eye(3)
            elif channel.mode == "twist":
                basis = pending_twist.pop(index, np.eye(3))
            else:
                desired = frames[frame, channel.source_index, :3, :3] @ channel.calibration
                basis = base.T @ desired
                if channel.twist_partner >= 0:
                    basis = _split_twist_onto(plan, basis, channel.twist_partner, pending_twist)
            # Blender receives a quaternion, so the forward model has to use the
            # quaternion too.  Reconstructing a local basis as
            # ``base^T @ desired`` only recovers ``desired`` when ``base`` is
            # exactly orthonormal, and Blender's float32 rest matrices are not:
            # left alone, that defect is amplified once per level and reached
            # 0.47 deg at 頭, eleven bones down the chain.
            quaternion = matrix_to_quaternion(basis)
            worlds[frame, index] = base @ quaternion_to_matrix(quaternion)
            locals_wxyz[frame, index] = quaternion

    return MmdSolveResult(
        channel_names=tuple(channel.name for channel in plan.channels),
        local_quaternions_wxyz=locals_wxyz,
        world_rotations=worlds,
        root_locations=np.asarray(root_locations, dtype=np.float64).copy(),
    )


def _split_twist_onto(
    plan: MmdRetargetPlan,
    basis: np.ndarray,
    twist_index: int,
    pending_twist: dict[int, np.ndarray],
) -> np.ndarray:
    """Move ``basis``'s rotation about the bone axis onto a 捩 channel.

    Blender applies a pose channel between the bone's rest offset and its
    children, so handing the raw twist to the 捩 bone would also carry that
    bone's own rest rotation.  Conjugating by its rest local keeps the chain
    below it identical to putting the whole rotation on the swing bone:
    ``rest_local @ B_twist == twist @ rest_local``.
    """
    swing_quaternion, twist_quaternion = swing_twist_decompose(
        matrix_to_quaternion(basis), _BONE_AXIS
    )
    rest_local = plan.channels[twist_index].rest_local
    pending_twist[twist_index] = rest_local.T @ quaternion_to_matrix(twist_quaternion) @ rest_local
    return quaternion_to_matrix(swing_quaternion)


def audit_plan(plan: MmdRetargetPlan) -> dict[str, dict[str, float | str]]:
    """Per-channel rest evidence: what each mode does to the bone at source rest."""
    report: dict[str, dict[str, float | str]] = {}
    for index, channel in enumerate(plan.channels):
        if not channel.driven:
            continue
        source_rest = plan.source_rest_global[index]
        target_rest = plan.target_rest_global[index]
        rest_pose = source_rest @ channel.calibration
        aim_error = float(
            np.degrees(np.arccos(np.clip(np.dot(rest_pose[:, 1], source_rest[:, 1]), -1.0, 1.0)))
        )
        rest_gap = float(
            np.degrees(np.arccos(np.clip(np.dot(source_rest[:, 1], target_rest[:, 1]), -1.0, 1.0)))
        )
        report[channel.name] = {
            "mode": channel.mode,
            "source": channel.source,
            "rest_gap_deg": rest_gap,
            "source_aim_error_deg": aim_error,
        }
    return report
