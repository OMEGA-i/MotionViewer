"""Pure NumPy Mixamo retarget solver.

The public interface deliberately accepts only evaluated skeleton matrices and
returns animation data.  FBX import, Blender pose evaluation, and F-curve
writing are adapters around this module rather than solver concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Leave a 1 mm adapter margin above the 2 mm nominal sole clearance.  Blender
# FBX pose evaluation and the NumPy FK path differ by sub-millimetre rounding
# on some assets; this prevents a geometrically grounded sole from crossing
# the strict -5 mm audit floor after round-trip channel evaluation.
GROUND_CLEARANCE_M = 0.003
# Match the half-width of the long double-support smoothing kernel.  Contact
# acquisition begins inside this window so the foot lock is introduced without
# a one-frame jump.
CONTACT_PREP_FRAMES = 5


@dataclass(frozen=True)
class FootContactProfile:
    """Calibrated sole frame for one target foot bone."""

    bone_index: int
    anchors_local: np.ndarray  # (4, 3), heel/toe/medial/lateral support points
    forward_local: np.ndarray  # (3,)
    normal_local: np.ndarray  # (3,)
    lateral_local: np.ndarray  # (3,)
    planar_residual: float = 0.0


@dataclass(frozen=True)
class RetargetDefinition:
    source_names: tuple[str, ...]
    target_names: tuple[str, ...]
    parent_indices: np.ndarray  # (bones,), -1 for a root
    rest_delta: np.ndarray  # (bones, 4, 4), inverse(source_rest) @ target_rest
    target_rest_local: np.ndarray  # (bones, 4, 4), target FK offsets
    source_rest_global: np.ndarray | None = None  # (bones, 3, 3), optional explicit basis
    target_rest_global: np.ndarray | None = None  # (bones, 3, 3), optional explicit basis
    source_to_target_scale: float = 1.0
    target_height_m: float = 1.0
    joint_limit_degrees: dict[int, float] = field(default_factory=dict)
    foot_profiles: dict[str, FootContactProfile] = field(default_factory=dict)
    # Compatibility input for callers outside the Blender adapter.  New
    # quality solves normalize it to a FootContactProfile once at entry.
    contact_bone_indices: dict[str, int] = field(default_factory=dict)
    contact_points_local: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class RetargetSolveResult:
    local_quaternions_wxyz: np.ndarray  # (frames, bones, 4)
    target_matrices: np.ndarray  # (frames, bones, 4, 4)
    root_locations: np.ndarray  # (frames, 3)


def solve_retarget(
    definition: RetargetDefinition,
    source_matrices: np.ndarray,
    root_locations: np.ndarray,
    *,
    contact_intervals: dict[str, list[tuple[int, int]]] | None = None,
    mode: str = "quality",
) -> RetargetSolveResult:
    """Transfer global rest deltas and solve a contact-consistent root path.

    ``source_matrices`` must be evaluated source pose matrices in the same bone
    order as ``definition.source_names``.  The target hierarchy is rebuilt in
    dependency order; therefore callers never need to reason about Blender
    pose channels, Euler order, or parent-space conversion.
    """
    source = np.asarray(source_matrices, dtype=np.float64)
    roots = np.asarray(root_locations, dtype=np.float64).copy()
    bone_count = len(definition.source_names)
    if source.ndim != 4 or source.shape[1:] != (bone_count, 4, 4):
        raise ValueError("source_matrices must have shape (frames, bones, 4, 4)")
    if roots.shape != (source.shape[0], 3):
        raise ValueError("root_locations must have shape (frames, 3)")
    if mode not in {"quality", "direct"}:
        raise ValueError("mode must be 'quality' or 'direct'")

    frames = source.shape[0]
    target = np.zeros_like(source)
    local_quaternions = np.zeros((frames, bone_count, 4), dtype=np.float64)
    previous_quaternions: np.ndarray | None = None

    for frame in range(frames):
        for index in range(bone_count):
            parent = int(definition.parent_indices[index])
            parent_global = np.eye(3) if parent < 0 else target[frame, parent, :3, :3]
            rest_local = definition.target_rest_local[index]
            desired_global = _desired_target_rotation(definition, source[frame], index)
            local_rotation = np.linalg.inv(parent_global @ rest_local[:3, :3]) @ desired_global
            quaternion = _matrix_to_quaternion_wxyz(local_rotation)
            if previous_quaternions is not None and np.dot(previous_quaternions[index], quaternion) < 0.0:
                quaternion *= -1.0
            local_quaternions[frame, index] = quaternion
            fk_local = rest_local @ _quaternion_matrix(quaternion)
            target[frame, index] = fk_local if parent < 0 else target[frame, parent] @ fk_local
        previous_quaternions = local_quaternions[frame]

    baseline_target = target.copy()
    if mode == "quality" and contact_intervals:
        temporal_baseline = local_quaternions.copy()
        roots = _solve_profiled_contacts(
            roots,
            local_quaternions,
            target,
            baseline_target,
            definition,
            contact_intervals,
        )
        _project_contact_boundaries(
            local_quaternions,
            temporal_baseline,
            definition,
            source,
            contact_intervals,
        )
        for frame in range(frames):
            _forward_kinematics_frame(definition, local_quaternions[frame], target[frame])
        # Boundary projection never edits contact frames, but rerunning the
        # exact lock solve makes that invariant explicit in the final output.
        roots = _solve_profiled_contacts(
            roots,
            local_quaternions,
            target,
            baseline_target,
            definition,
            contact_intervals,
        )
    if mode == "quality" and definition.contact_points_local:
        _prevent_ground_penetration(roots, target, definition)
    if mode == "quality":
        _enforce_joint_limits(definition, local_quaternions)
        for frame in range(frames):
            _forward_kinematics_frame(definition, local_quaternions[frame], target[frame])
        # Rotation-limit projection can lower a sole by a sub-millimetre.
        # Ground clearance is a final world-space constraint and must be
        # evaluated after every pose-changing projection.
        if definition.contact_points_local:
            _prevent_ground_penetration(roots, target, definition)
    _stabilize_quaternions(local_quaternions)
    return RetargetSolveResult(local_quaternions, target, roots)


def _enforce_joint_limits(
    definition: RetargetDefinition,
    local_quaternions: np.ndarray,
) -> None:
    """Project local channels to calibrated Mixamo rotation magnitudes."""
    identity = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    for bone, maximum_degrees in definition.joint_limit_degrees.items():
        maximum = np.deg2rad(float(maximum_degrees))
        for frame in range(len(local_quaternions)):
            quaternion = local_quaternions[frame, bone]
            angle = 2.0 * np.arccos(np.clip(abs(float(quaternion[0])), -1.0, 1.0))
            if angle > maximum:
                local_quaternions[frame, bone] = _slerp_quaternion(identity, quaternion, maximum / angle)


_LIMB_CHAINS: tuple[tuple[str, str, str], ...] = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
)


def _fit_limbs_from_baseline(
    definition: RetargetDefinition,
    baseline: np.ndarray,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
) -> None:
    """Rebuild limb lengths while preserving the transferred target-space pose.

    ``baseline`` is the direct FK transfer.  It is the only valid direction
    reference here: raw source world edges carry a different armature basis on
    many Mixamo FBX files.
    """
    indices = {name: index for index, name in enumerate(definition.source_names)}
    previous_poles: dict[tuple[str, str, str], np.ndarray] = {}
    for frame in range(len(target_matrices)):
        for chain in _LIMB_CHAINS:
            if any(name not in indices for name in chain):
                continue
            start, middle, end = (indices[name] for name in chain)
            start_position = target_matrices[frame, start, :3, 3]
            upper_length = float(np.linalg.norm(target_matrices[frame, middle, :3, 3] - start_position))
            lower_length = float(
                np.linalg.norm(target_matrices[frame, end, :3, 3] - target_matrices[frame, middle, :3, 3])
            )
            upper = baseline[frame, middle, :3, 3] - baseline[frame, start, :3, 3]
            lower = baseline[frame, end, :3, 3] - baseline[frame, middle, :3, 3]
            upper /= max(float(np.linalg.norm(upper)), 1e-12)
            lower /= max(float(np.linalg.norm(lower)), 1e-12)
            preferred_middle = start_position + upper_length * upper
            goal = preferred_middle + lower_length * lower
            previous_poles[chain] = _fit_two_bone_frame(
                definition,
                local_quaternions[frame],
                target_matrices[frame],
                start,
                middle,
                end,
                goal,
                previous_poles.get(chain),
                preferred_middle=preferred_middle,
                end_global_rotation=baseline[frame, end, :3, :3],
            )
            if chain[2].endswith("ankle"):
                _restore_foot_orientation_from_baseline(
                    definition,
                    baseline[frame],
                    local_quaternions[frame],
                    target_matrices[frame],
                    end,
                )


def _restore_foot_orientation_from_baseline(
    definition: RetargetDefinition,
    baseline_frame: np.ndarray,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
    ankle: int,
) -> None:
    _set_bone_global_rotation(
        definition,
        local_quaternions,
        target_matrices,
        ankle,
        baseline_frame[ankle, :3, :3],
    )
    _forward_kinematics_frame(definition, local_quaternions, target_matrices)
    for child, parent in enumerate(definition.parent_indices):
        if int(parent) == ankle and definition.source_names[child].endswith("_foot"):
            _set_bone_global_rotation(
                definition,
                local_quaternions,
                target_matrices,
                child,
                baseline_frame[child, :3, :3],
            )
            _forward_kinematics_frame(definition, local_quaternions, target_matrices)


def _contact_profiles(definition: RetargetDefinition) -> dict[str, FootContactProfile]:
    if definition.foot_profiles:
        return definition.foot_profiles
    profiles: dict[str, FootContactProfile] = {}
    for foot_name, points in definition.contact_points_local.items():
        values = np.asarray(points, dtype=np.float64)
        if len(values) < 3:
            continue
        centered = values - np.mean(values, axis=0)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        forward = values[int(np.argmax(values[:, 1]))] - values[int(np.argmin(values[:, 1]))]
        forward -= normal * float(np.dot(forward, normal))
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        lateral = np.cross(forward, normal)
        lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
        profiles[foot_name] = FootContactProfile(
            bone_index=definition.contact_bone_indices.get(foot_name, 0),
            anchors_local=values,
            forward_local=forward,
            normal_local=normal,
            lateral_local=lateral,
            planar_residual=float(singular_values[-1] / max(singular_values[0], 1e-12)),
        )
    return profiles


def _contact_bone_rotation(profile: FootContactProfile, rotation: np.ndarray) -> np.ndarray:
    """Return a horizontal sole rotation with the current semantic heading."""
    forward = rotation @ profile.forward_local
    forward[2] = 0.0
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    normal = np.array((0.0, 0.0, 1.0))
    lateral = np.cross(forward, normal)
    lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
    world_frame = np.column_stack((lateral, forward, normal))
    local_frame = np.column_stack((profile.lateral_local, profile.forward_local, profile.normal_local))
    return world_frame @ local_frame.T


def _sole_world(profile: FootContactProfile, matrix: np.ndarray) -> np.ndarray:
    return profile.anchors_local @ matrix[:3, :3].T + matrix[:3, 3]


def _solve_profiled_contacts(
    root_locations: np.ndarray,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
    baseline: np.ndarray,
    definition: RetargetDefinition,
    intervals: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    """Lock calibrated soles by root translation without rewriting leg IK.

    The direct transfer is already a valid target-space FK solution with the
    correct target morphology.  Re-solving hip/knee against a contact target
    introduces a second, incompatible limb objective and was the source of
    the visible knee/ankle twists.  A support constraint only needs a root
    translation plus the calibrated foot-bone sole orientation; this preserves
    all transferred leg channels and makes double support an explicit least-
    squares root residual instead of a chain of destructive compensations.
    """
    roots = root_locations.copy()
    profiles = _contact_profiles(definition)
    locks: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for frame in range(len(roots)):
        active: list[tuple[str, np.ndarray, np.ndarray]] = []
        for foot_name, profile in profiles.items():
            for interval_index, (start, end) in enumerate(intervals.get(foot_name, ())):
                if not start <= frame < end:
                    continue
                key = (foot_name, interval_index)
                if key not in locks:
                    bone = profile.bone_index
                    desired_rotation = _contact_bone_rotation(profile, baseline[frame, bone, :3, :3])
                    desired_matrix = target_matrices[frame, bone].copy()
                    desired_matrix[:3, :3] = desired_rotation
                    anchor_lock = _sole_world(profile, desired_matrix) + roots[frame]
                    anchor_lock[:, 2] += GROUND_CLEARANCE_M - float(np.min(anchor_lock[:, 2]))
                    locks[key] = (anchor_lock, desired_rotation)
                active.append((foot_name, *locks[key]))
                break
        if not active:
            continue
        for foot_name, _, desired_rotation in active:
            profile = profiles[foot_name]
            _set_bone_global_rotation(
                definition,
                local_quaternions[frame],
                target_matrices[frame],
                profile.bone_index,
                desired_rotation,
            )
        _forward_kinematics_frame(definition, local_quaternions[frame], target_matrices[frame])

        # With foot rotations fixed, root translation is the exact least-
        # squares solution for all active four-anchor residuals.
        corrections: list[np.ndarray] = []
        for foot_name, anchor_lock, _ in active:
            profile = profiles[foot_name]
            current = _sole_world(profile, target_matrices[frame, profile.bone_index]) + roots[frame]
            corrections.append(np.mean(anchor_lock - current, axis=0))
        roots[frame] += np.mean(corrections, axis=0)

    return roots


def _project_contact_boundaries(
    local_quaternions: np.ndarray,
    temporal_baseline: np.ndarray,
    definition: RetargetDefinition,
    source_matrices: np.ndarray,
    intervals: dict[str, list[tuple[int, int]]],
    *,
    transition_frames: int = 4,
) -> None:
    """Spread support-foot rotation changes into neighbouring swing frames.

    Contact frames are hard constraints and are deliberately never blended.
    The only available freedom is the adjacent swing interval, where a
    geodesic quaternion ramp avoids a channel discontinuity at heel strike or
    toe-off without changing a locked sole anchor.
    """
    profiles = _contact_profiles(definition)
    frame_count = len(local_quaternions)
    for foot_name, profile in profiles.items():
        bone = profile.bone_index
        for start, end in intervals.get(foot_name, ()):
            before = max(0, start - transition_frames)
            if before < start:
                source = temporal_baseline[before, bone]
                target = local_quaternions[start, bone]
                count = start - before
                for frame in range(before, start):
                    local_quaternions[frame, bone] = _slerp_quaternion(
                        source, target, (frame - before + 1) / (count + 1)
                    )
            after = min(frame_count, end + transition_frames)
            if end < after:
                source = local_quaternions[end - 1, bone]
                target = temporal_baseline[after - 1, bone]
                count = after - end
                for frame in range(end, after):
                    local_quaternions[frame, bone] = _slerp_quaternion(
                        source, target, (frame - end + 1) / (count + 1)
                    )
        _limit_swing_frames_against_contact_locks(
            local_quaternions,
            definition,
            source_matrices,
            profile.bone_index,
            intervals.get(foot_name, ()),
        )


def _limit_swing_frames_against_contact_locks(
    local_quaternions: np.ndarray,
    definition: RetargetDefinition,
    source_matrices: np.ndarray,
    bone: int,
    intervals: list[tuple[int, int]],
) -> None:
    """Project only free swing frames while treating support frames as fixed."""
    fixed = np.zeros(len(local_quaternions), dtype=bool)
    for start, end in intervals:
        fixed[start:end] = True
    source_quaternions = np.empty((len(source_matrices), 4), dtype=np.float64)
    parent = int(definition.parent_indices[bone])
    for frame in range(len(source_matrices)):
        parent_rotation = np.eye(3) if parent < 0 else source_matrices[frame, parent, :3, :3]
        source_quaternions[frame] = _matrix_to_quaternion_wxyz(
            np.linalg.inv(parent_rotation) @ source_matrices[frame, bone, :3, :3]
        )
    limits = np.zeros(len(local_quaternions), dtype=np.float64)
    for frame in range(1, len(local_quaternions)):
        source_dot = abs(float(np.dot(source_quaternions[frame - 1], source_quaternions[frame])))
        source_step = 2.0 * np.arccos(np.clip(source_dot, -1.0, 1.0))
        limits[frame] = min(
            np.deg2rad(45.0 - 1e-4),
            1.5 * source_step + np.deg2rad(10.0 - 1e-4),
        )
    # Alternating passes propagate each fixed contact orientation into the
    # surrounding free interval without moving a constrained sample.
    for _ in range(3):
        for frame in range(1, len(local_quaternions)):
            if not fixed[frame]:
                local_quaternions[frame, bone] = _clamp_quaternion_step(
                    local_quaternions[frame - 1, bone],
                    local_quaternions[frame, bone],
                    limits[frame],
                )
        for frame in range(len(local_quaternions) - 2, -1, -1):
            if not fixed[frame]:
                local_quaternions[frame, bone] = _clamp_quaternion_step(
                    local_quaternions[frame + 1, bone],
                    local_quaternions[frame, bone],
                    limits[frame + 1],
                )


def _clamp_quaternion_step(anchor: np.ndarray, candidate: np.ndarray, limit: float) -> np.ndarray:
    """Return the nearest candidate no farther than ``limit`` from anchor."""
    current = candidate.copy()
    dot = float(np.dot(anchor, current))
    if dot < 0.0:
        current *= -1.0
        dot = -dot
    angle = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
    if angle <= limit + 1e-10:
        return current
    half_angle = 0.5 * angle
    sine = float(np.sin(half_angle))
    if sine <= 1e-10:
        return anchor.copy()
    fraction = float(limit / angle)
    result = (
        np.sin((1.0 - fraction) * half_angle) / sine * anchor + np.sin(fraction * half_angle) / sine * current
    )
    return result / max(float(np.linalg.norm(result)), 1e-12)


def _slerp_quaternion(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    """Return the shortest-path unit-quaternion interpolation."""
    start = left / max(float(np.linalg.norm(left)), 1e-12)
    end = right / max(float(np.linalg.norm(right)), 1e-12)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        result = (1.0 - fraction) * start + fraction * end
        return result / max(float(np.linalg.norm(result)), 1e-12)
    angle = float(np.arccos(dot))
    sine = float(np.sin(angle))
    result = np.sin((1.0 - fraction) * angle) / sine * start + np.sin(fraction * angle) / sine * end
    return result / max(float(np.linalg.norm(result)), 1e-12)


def _fit_limbs(
    definition: RetargetDefinition,
    source_matrices: np.ndarray,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
) -> None:
    """Fit wrists and ankles with a closed-form two-bone solve."""
    indices = {name: index for index, name in enumerate(definition.source_names)}
    previous_poles: dict[tuple[str, str, str], np.ndarray] = {}
    for frame in range(len(source_matrices)):
        for chain in _LIMB_CHAINS:
            if any(name not in indices for name in chain):
                continue
            start, middle, end = (indices[name] for name in chain)
            target_start = target_matrices[frame, start, :3, 3]
            target_upper_length = float(np.linalg.norm(target_matrices[frame, middle, :3, 3] - target_start))
            target_lower_length = float(
                np.linalg.norm(target_matrices[frame, end, :3, 3] - target_matrices[frame, middle, :3, 3])
            )
            # Build the pole and goal in target space. Applying raw source
            # world vectors here is wrong whenever the rigs have different
            # rest axes or object bases.
            source_upper_raw = source_matrices[frame, middle, :3, 3] - source_matrices[frame, start, :3, 3]
            source_lower_raw = source_matrices[frame, end, :3, 3] - source_matrices[frame, middle, :3, 3]
            upper_edge = source_upper_raw
            lower_edge = source_lower_raw
            source_upper = upper_edge / max(float(np.linalg.norm(upper_edge)), 1e-12)
            source_lower = lower_edge / max(float(np.linalg.norm(lower_edge)), 1e-12)
            preferred_middle = target_start + target_upper_length * source_upper
            goal = preferred_middle + target_lower_length * source_lower
            pole = _fit_two_bone_frame(
                definition,
                local_quaternions[frame],
                target_matrices[frame],
                start,
                middle,
                end,
                goal,
                previous_poles.get(chain),
                preferred_middle=preferred_middle,
                end_global_rotation=_desired_target_rotation(definition, source_matrices[frame], end),
            )
            previous_poles[chain] = pole
            _restore_foot_orientation(
                definition,
                source_matrices[frame],
                local_quaternions[frame],
                target_matrices[frame],
                end,
            )


def _fit_two_bone_frame(
    definition: RetargetDefinition,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
    start: int,
    middle: int,
    end: int,
    goal: np.ndarray,
    previous_pole: np.ndarray | None,
    *,
    preferred_middle: np.ndarray | None = None,
    end_global_rotation: np.ndarray | None = None,
) -> np.ndarray:
    start_position = target_matrices[start, :3, 3]
    middle_position = target_matrices[middle, :3, 3]
    end_position = target_matrices[end, :3, 3]
    upper_length = float(np.linalg.norm(middle_position - start_position))
    lower_length = float(np.linalg.norm(end_position - middle_position))
    if upper_length <= 1e-8 or lower_length <= 1e-8:
        return np.array((0.0, 0.0, 1.0))

    to_goal = goal - start_position
    raw_distance = float(np.linalg.norm(to_goal))
    if raw_distance <= 1e-8:
        return previous_pole if previous_pole is not None else np.array((0.0, 0.0, 1.0))
    direction = to_goal / raw_distance
    distance = float(
        np.clip(raw_distance, abs(upper_length - lower_length) + 1e-7, upper_length + lower_length - 1e-7)
    )
    reachable_goal = start_position + direction * distance

    pole = (
        preferred_middle - start_position
        if preferred_middle is not None
        else middle_position - start_position
    )
    pole -= direction * float(np.dot(pole, direction))
    if float(np.linalg.norm(pole)) <= 1e-7 and previous_pole is not None:
        pole = previous_pole - direction * float(np.dot(previous_pole, direction))
    if float(np.linalg.norm(pole)) <= 1e-7:
        candidate = target_matrices[start, :3, 0]
        pole = candidate - direction * float(np.dot(candidate, direction))
    if float(np.linalg.norm(pole)) <= 1e-7:
        candidate = np.array((0.0, 0.0, 1.0))
        pole = candidate - direction * float(np.dot(candidate, direction))
    pole /= max(float(np.linalg.norm(pole)), 1e-12)

    along = (upper_length * upper_length - lower_length * lower_length + distance * distance) / (
        2.0 * distance
    )
    height = np.sqrt(max(0.0, upper_length * upper_length - along * along))
    desired_middle = start_position + direction * along + pole * height

    _aim_bone(definition, local_quaternions, target_matrices, start, middle, desired_middle)
    _forward_kinematics_frame(definition, local_quaternions, target_matrices)
    _aim_bone(definition, local_quaternions, target_matrices, middle, end, reachable_goal)
    _forward_kinematics_frame(definition, local_quaternions, target_matrices)
    if end_global_rotation is not None:
        _set_bone_global_rotation(definition, local_quaternions, target_matrices, end, end_global_rotation)
        _forward_kinematics_frame(definition, local_quaternions, target_matrices)
    return pole


def _desired_target_rotation(
    definition: RetargetDefinition,
    source_frame: np.ndarray,
    bone: int,
) -> np.ndarray:
    source_rotation = source_frame[bone, :3, :3]
    # Blender pose-bone matrices carry the imported bone channel basis.  The
    # calibrated rest delta is therefore applied on the right; changing the
    # order to a global left transfer double-applies FBX bone roll.
    return source_rotation @ definition.rest_delta[bone, :3, :3]


def _set_bone_global_rotation(
    definition: RetargetDefinition,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
    bone: int,
    desired_global: np.ndarray,
) -> None:
    parent = int(definition.parent_indices[bone])
    parent_global = np.eye(3) if parent < 0 else target_matrices[parent, :3, :3]
    rest_rotation = definition.target_rest_local[bone, :3, :3]
    local_rotation = np.linalg.inv(parent_global @ rest_rotation) @ desired_global
    local_quaternions[bone] = _matrix_to_quaternion_wxyz(local_rotation)


def _restore_foot_orientation(
    definition: RetargetDefinition,
    source_frame: np.ndarray,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
    ankle: int,
) -> None:
    """Restore ankle and direct toe orientation after parent-chain IK."""
    _set_bone_global_rotation(
        definition,
        local_quaternions,
        target_matrices,
        ankle,
        _desired_target_rotation(definition, source_frame, ankle),
    )
    _forward_kinematics_frame(definition, local_quaternions, target_matrices)
    for child, parent in enumerate(definition.parent_indices):
        if int(parent) != ankle:
            continue
        source_index = definition.source_names[child]
        if source_index.endswith("_foot"):
            _set_bone_global_rotation(
                definition,
                local_quaternions,
                target_matrices,
                child,
                _desired_target_rotation(definition, source_frame, child),
            )
            _forward_kinematics_frame(definition, local_quaternions, target_matrices)


def _restore_current_foot_orientation(
    definition: RetargetDefinition,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
    ankle: int,
) -> None:
    """Re-FK after contact IK and keep the direct toe child coherent."""
    _forward_kinematics_frame(definition, local_quaternions, target_matrices)
    for child, parent in enumerate(definition.parent_indices):
        if int(parent) == ankle and definition.source_names[child].endswith("_foot"):
            # The toe has no positional DOF in the contact solve. Its local
            # rotation remains the transferred source-relative value while
            # the ankle is explicitly restored above.
            _forward_kinematics_frame(definition, local_quaternions, target_matrices)
            break


def _align_foot_to_ground(
    definition: RetargetDefinition,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
    ankle: int,
    foot_name: str,
    reference_points: np.ndarray | None = None,
) -> None:
    """Project a profiled sole's swing so its support plane is horizontal."""
    points = np.asarray(definition.contact_points_local.get(foot_name, ()), dtype=np.float64)
    if len(points) < 3:
        return
    bone_index = definition.contact_bone_indices.get(foot_name, ankle)
    matrix = target_matrices[bone_index]
    world = points @ matrix[:3, :3].T + matrix[:3, 3]
    _, _, vh = np.linalg.svd(world - np.mean(world, axis=0), full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    swing = _rotation_between(normal, np.array((0.0, 0.0, 1.0)))
    current_rotation = target_matrices[ankle, :3, :3]
    desired_global = swing @ current_rotation
    if reference_points is not None and len(reference_points) == len(world):
        swung_world = points @ desired_global.T + matrix[:3, 3]
        current_xy = swung_world[:, :2] - np.mean(swung_world[:, :2], axis=0)
        reference_xy = reference_points[:, :2] - np.mean(reference_points[:, :2], axis=0)
        cosine = float(np.sum(current_xy * reference_xy))
        sine = float(np.sum(current_xy[:, 0] * reference_xy[:, 1] - current_xy[:, 1] * reference_xy[:, 0]))
        if abs(cosine) + abs(sine) > 1e-10:
            yaw = float(np.arctan2(sine, cosine))
            c, s = np.cos(yaw), np.sin(yaw)
            yaw_rotation = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
            desired_global = yaw_rotation @ desired_global
    _set_bone_global_rotation(definition, local_quaternions, target_matrices, ankle, desired_global)
    _forward_kinematics_frame(definition, local_quaternions, target_matrices)


def _aim_bone(
    definition: RetargetDefinition,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
    bone: int,
    child: int,
    desired_child_position: np.ndarray,
) -> None:
    origin = target_matrices[bone, :3, 3]
    current_direction = target_matrices[child, :3, 3] - origin
    desired_direction = desired_child_position - origin
    swing = _rotation_between(current_direction, desired_direction)
    desired_global = swing @ target_matrices[bone, :3, :3]
    parent = int(definition.parent_indices[bone])
    parent_global = np.eye(3) if parent < 0 else target_matrices[parent, :3, :3]
    rest_rotation = definition.target_rest_local[bone, :3, :3]
    local_rotation = np.linalg.inv(parent_global @ rest_rotation) @ desired_global
    local_quaternions[bone] = _matrix_to_quaternion_wxyz(local_rotation)


def _forward_kinematics_frame(
    definition: RetargetDefinition,
    local_quaternions: np.ndarray,
    output: np.ndarray,
) -> None:
    for index, quaternion in enumerate(local_quaternions):
        local = definition.target_rest_local[index] @ _quaternion_matrix(quaternion)
        parent = int(definition.parent_indices[index])
        output[index] = local if parent < 0 else output[parent] @ local


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-12 or target_norm <= 1e-12:
        return np.eye(3)
    left = source / source_norm
    right = target / target_norm
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3)
    if dot < -1.0 + 1e-10:
        axis = np.cross(left, np.array((1.0, 0.0, 0.0)))
        if float(np.linalg.norm(axis)) <= 1e-8:
            axis = np.cross(left, np.array((0.0, 1.0, 0.0)))
        axis /= float(np.linalg.norm(axis))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(left, right)
    skew = np.array(((0.0, -cross[2], cross[1]), (cross[2], 0.0, -cross[0]), (-cross[1], cross[0], 0.0)))
    return np.eye(3) + skew + skew @ skew / (1.0 + dot)


def _solve_contacts(
    root_locations: np.ndarray,
    local_quaternions: np.ndarray,
    target_matrices: np.ndarray,
    definition: RetargetDefinition,
    intervals: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    """Lock simultaneous contacts with root translation and analytic leg IK."""
    result = root_locations.copy()
    indices = {name: index for index, name in enumerate(definition.source_names)}
    # Profiled contacts keep every sole anchor in the lock.  A single
    # centroid/min-Z point cannot constrain a shoe that rotates around it.
    locks: dict[tuple[str, int], np.ndarray] = {}
    previous_poles: dict[str, np.ndarray] = {}
    leg_names = {
        "left_foot": ("left_hip", "left_knee", "left_ankle"),
        "right_foot": ("right_hip", "right_knee", "right_ankle"),
    }
    for frame in range(len(result)):
        active: list[tuple[str, np.ndarray, bool]] = []
        for source_name in ("left_foot", "right_foot"):
            index = indices.get(source_name)
            if index is None:
                continue
            profiled_support = source_name in definition.contact_points_local
            anchor_points = (
                _contact_points_world(definition, target_matrices[frame], source_name, index) + result[frame]
            )
            point = _contact_point(definition, target_matrices[frame], source_name, index) + result[frame]
            for interval_index, (start, end) in enumerate(intervals.get(source_name, [])):
                if start <= frame < end:
                    key = (source_name, interval_index)
                    if key not in locks:
                        if profiled_support:
                            lock_points = anchor_points.copy()
                            lock_points[:, 2] += GROUND_CLEARANCE_M - float(np.min(lock_points[:, 2]))
                            locks[key] = lock_points
                        else:
                            locks[key] = point.copy()
                    active.append((source_name, locks[key], profiled_support))
                    break
        for _ in range(3):
            corrections_xy = []
            for foot_name, lock, profiled in active:
                current = (
                    _contact_points_world(definition, target_matrices[frame], foot_name, indices[foot_name])
                    + result[frame]
                )
                if profiled:
                    corrections_xy.append(np.mean(lock[:, :2] - current[:, :2], axis=0))
                else:
                    corrections_xy.append(lock[:2] - current[0, :2])
            if corrections_xy:
                result[frame, :2] += np.mean(corrections_xy, axis=0)
            vertical_corrections = []
            for foot_name, lock, profiled in active:
                if not profiled:
                    continue
                current = (
                    _contact_points_world(definition, target_matrices[frame], foot_name, indices[foot_name])
                    + result[frame]
                )
                vertical_corrections.append(float(np.mean(lock[:, 2] - current[:, 2])))
            if vertical_corrections:
                result[frame, 2] += float(np.mean(vertical_corrections))
            for foot_name, lock, _ in active:
                chain_names = leg_names[foot_name]
                if any(name not in indices for name in chain_names):
                    continue
                start, middle, ankle = (indices[name] for name in chain_names)
                contact = indices[foot_name]
                lock_point = _contact_lock_point(lock, profiled=foot_name in definition.contact_points_local)
                contact_offset = (
                    _contact_point(definition, target_matrices[frame], foot_name, contact)
                    - target_matrices[frame, ankle, :3, 3]
                )
                ankle_goal = lock_point - result[frame] - contact_offset
                previous_poles[foot_name] = _fit_two_bone_frame(
                    definition,
                    local_quaternions[frame],
                    target_matrices[frame],
                    start,
                    middle,
                    ankle,
                    ankle_goal,
                    previous_poles.get(foot_name),
                    end_global_rotation=target_matrices[frame, ankle, :3, :3].copy(),
                )
                _restore_current_foot_orientation(
                    definition, local_quaternions[frame], target_matrices[frame], ankle
                )
                if foot_name in definition.contact_points_local:
                    _align_foot_to_ground(
                        definition,
                        local_quaternions[frame],
                        target_matrices[frame],
                        ankle,
                        foot_name,
                        reference_points=lock,
                    )
                corrected = (
                    _contact_point(definition, target_matrices[frame], foot_name, contact) + result[frame]
                )
                correction = lock_point - corrected
                result[frame, :2] += correction[:2]
                if foot_name in definition.contact_points_local:
                    result[frame, 2] += correction[2]
    # Profiled sole contacts are solved against actual mesh anchors. Smoothing
    # their root correction here would move a locked foot away from its exact
    # multi-point target after the IK pass, so preserve the solved trajectory.
    if definition.contact_points_local:
        return result

    # Root corrections change discretely when support switches between one
    # and two feet.  Smooth only that correction, then re-solve the same world
    # locks so the transition is carried by a short temporal window instead
    # of a one-frame knee/hip rotation spike.
    correction = result - root_locations
    has_leg_chain = any(all(name in indices for name in chain) for chain in leg_names.values())
    if len(correction) >= 3 and has_leg_chain:
        contact_masks = []
        for foot_name in ("left_foot", "right_foot"):
            mask = np.zeros(len(result), dtype=bool)
            for start_frame, end_frame in intervals.get(foot_name, ()):
                mask[start_frame:end_frame] = True
            contact_masks.append(mask)
        double_support = contact_masks[0] & contact_masks[1]
        longest_double_support = 0
        current_run = 0
        for active in double_support:
            current_run = current_run + 1 if active else 0
            longest_double_support = max(longest_double_support, current_run)
        if longest_double_support >= 10 and len(correction) >= 11:
            kernel = (
                np.array(
                    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0),
                    dtype=np.float64,
                )
                / 36.0
            )
            half = 5
        else:
            kernel = np.array((0.25, 0.5, 0.25), dtype=np.float64)
            half = 1
        prep_frames = CONTACT_PREP_FRAMES if longest_double_support >= 10 else 0
        for axis in range(3):
            padded = np.pad(correction[:, axis], (half, half), mode="edge")
            correction[:, axis] = np.convolve(padded, kernel, mode="valid")
        result = root_locations + correction

        previous_poles.clear()
        for frame in range(len(result)):
            active_locks: list[tuple[str, np.ndarray]] = []
            for foot_name in ("left_foot", "right_foot"):
                for interval_index, (start_frame, end_frame) in enumerate(intervals.get(foot_name, ())):
                    if start_frame <= frame < end_frame:
                        active_locks.append((foot_name, locks[(foot_name, interval_index)]))
                        break
                    prep_start = max(0, start_frame - prep_frames)
                    if prep_start <= frame < start_frame:
                        already_locked = any(
                            other_start <= frame < other_end
                            for other_start, other_end in intervals.get(foot_name, ())
                            if (other_start, other_end) != (start_frame, end_frame)
                        )
                        if not already_locked:
                            contact = indices.get(foot_name)
                            if contact is not None:
                                current = (
                                    _contact_point(
                                        definition,
                                        target_matrices[frame],
                                        foot_name,
                                        contact,
                                    )
                                    + result[frame]
                                )
                                weight = (frame - prep_start + 1) / (start_frame - prep_start + 1)
                                lock = locks[(foot_name, interval_index)]
                                active_locks.append((foot_name, current + weight * (lock - current)))
                        break
            for _ in range(3):
                for foot_name, lock in active_locks:
                    chain_names = leg_names[foot_name]
                    if any(name not in indices for name in chain_names):
                        continue
                    start, middle, ankle = (indices[name] for name in chain_names)
                    contact = indices[foot_name]
                    contact_offset = (
                        _contact_point(definition, target_matrices[frame], foot_name, contact)
                        - target_matrices[frame, ankle, :3, 3]
                    )
                    ankle_goal = lock - result[frame] - contact_offset
                    previous_poles[foot_name] = _fit_two_bone_frame(
                        definition,
                        local_quaternions[frame],
                        target_matrices[frame],
                        start,
                        middle,
                        ankle,
                        ankle_goal,
                        previous_poles.get(foot_name),
                        end_global_rotation=target_matrices[frame, ankle, :3, :3].copy(),
                    )
                    _restore_current_foot_orientation(
                        definition, local_quaternions[frame], target_matrices[frame], ankle
                    )
                    if foot_name in definition.contact_points_local:
                        _align_foot_to_ground(
                            definition,
                            local_quaternions[frame],
                            target_matrices[frame],
                            ankle,
                            foot_name,
                            reference_points=lock,
                        )
                    corrected = (
                        _contact_point(definition, target_matrices[frame], foot_name, contact) + result[frame]
                    )
                    correction = lock - corrected
                    result[frame, :2] += correction[:2]
                    if foot_name in definition.contact_points_local:
                        result[frame, 2] += correction[2]
    return result


def _blend_quaternion_sequences(
    current: np.ndarray,
    target: np.ndarray,
    fraction: float,
) -> np.ndarray:
    aligned = target.copy()
    dots = np.sum(current * aligned, axis=-1)
    aligned[dots < 0.0] *= -1.0
    blended = (1.0 - fraction) * current + fraction * aligned
    blended /= np.maximum(np.linalg.norm(blended, axis=-1, keepdims=True), 1e-12)
    return blended


def _limit_angular_velocity(
    definition: RetargetDefinition,
    local_quaternions: np.ndarray,
    source_matrices: np.ndarray,
) -> None:
    """Clamp quality output to source-relative and absolute angular limits."""
    source_quaternions = np.empty_like(local_quaternions)
    for frame in range(len(source_matrices)):
        for bone in range(source_matrices.shape[1]):
            parent = int(definition.parent_indices[bone])
            parent_rotation = np.eye(3) if parent < 0 else source_matrices[frame, parent, :3, :3]
            local_rotation = np.linalg.inv(parent_rotation) @ source_matrices[frame, bone, :3, :3]
            source_quaternions[frame, bone] = _matrix_to_quaternion_wxyz(local_rotation)
    limits = np.zeros((len(local_quaternions), local_quaternions.shape[1]), dtype=np.float64)
    for frame in range(1, len(local_quaternions)):
        for bone in range(local_quaternions.shape[1]):
            source_dot = float(
                abs(np.dot(source_quaternions[frame - 1, bone], source_quaternions[frame, bone]))
            )
            source_step = 2.0 * np.arccos(np.clip(source_dot, -1.0, 1.0))
            limits[frame, bone] = min(
                np.deg2rad(45.0 - 1e-4),
                1.5 * source_step + np.deg2rad(10.0 - 1e-4),
            )

    def clamp(anchor: np.ndarray, candidate: np.ndarray, limit: float) -> np.ndarray:
        current = candidate.copy()
        dot = float(np.dot(anchor, current))
        if dot < 0.0:
            current *= -1.0
            dot = -dot
        angle = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
        if angle <= limit + 1e-10:
            return current
        half_angle = 0.5 * angle
        fraction = float(limit / angle)
        sine = float(np.sin(half_angle))
        if sine <= 1e-10:
            return anchor.copy()
        blended = (
            np.sin((1.0 - fraction) * half_angle) / sine * anchor
            + np.sin(fraction * half_angle) / sine * current
        )
        return blended / max(float(np.linalg.norm(blended)), 1e-12)

    # A forward/backward/forward projection distributes corrections over a
    # short implicit window and ends with exact per-edge guarantees.
    for frame in range(1, len(local_quaternions)):
        for bone in range(local_quaternions.shape[1]):
            local_quaternions[frame, bone] = clamp(
                local_quaternions[frame - 1, bone],
                local_quaternions[frame, bone],
                limits[frame, bone],
            )
    for frame in range(len(local_quaternions) - 1, 1, -1):
        for bone in range(local_quaternions.shape[1]):
            local_quaternions[frame - 1, bone] = clamp(
                local_quaternions[frame, bone],
                local_quaternions[frame - 1, bone],
                limits[frame, bone],
            )
    for frame in range(1, len(local_quaternions)):
        for bone in range(local_quaternions.shape[1]):
            local_quaternions[frame, bone] = clamp(
                local_quaternions[frame - 1, bone], local_quaternions[frame, bone], limits[frame, bone]
            )


def _lock_contacts_with_root_only(
    root_locations: np.ndarray,
    target_matrices: np.ndarray,
    definition: RetargetDefinition,
    intervals: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    result = root_locations.copy()
    indices = {name: index for index, name in enumerate(definition.source_names)}
    locks: dict[tuple[str, int], np.ndarray] = {}
    for frame in range(len(result)):
        corrections: list[np.ndarray] = []
        for foot_name in ("left_foot", "right_foot"):
            fallback = indices.get(foot_name)
            if fallback is None:
                continue
            for interval_index, (start, end) in enumerate(intervals.get(foot_name, ())):
                if not start <= frame < end:
                    continue
                key = (foot_name, interval_index)
                points = _contact_points_world(definition, target_matrices[frame], foot_name, fallback)
                point = _contact_point(definition, target_matrices[frame], foot_name, fallback)
                if key not in locks:
                    if foot_name in definition.contact_points_local:
                        lock_points = points + result[frame]
                        lock_points[:, 2] += GROUND_CLEARANCE_M - float(np.min(lock_points[:, 2]))
                        locks[key] = lock_points
                    else:
                        locks[key] = point + result[frame]
                if foot_name in definition.contact_points_local:
                    corrections.append(np.mean(locks[key] - (points + result[frame]), axis=0))
                else:
                    corrections.append(locks[key] - (point + result[frame]))
                break
        if corrections:
            result[frame] += np.mean(corrections, axis=0)
    return result


def _maximum_contact_drift(
    root_locations: np.ndarray,
    target_matrices: np.ndarray,
    definition: RetargetDefinition,
    intervals: dict[str, list[tuple[int, int]]],
) -> float:
    indices = {name: index for index, name in enumerate(definition.source_names)}
    maximum = 0.0
    for foot_name, foot_intervals in intervals.items():
        fallback = indices.get(foot_name)
        if fallback is None:
            continue
        points = [
            _contact_points_world(definition, target_matrices[frame], foot_name, fallback)
            + root_locations[frame]
            for frame in range(len(root_locations))
        ]
        for start, end in foot_intervals:
            for frame in range(start, end):
                displacement = points[frame][:, :2] - points[start][:, :2]
                maximum = max(maximum, float(np.linalg.norm(displacement, axis=1).max(initial=0.0)))
    return maximum


def _contact_point(
    definition: RetargetDefinition,
    target_matrices: np.ndarray,
    foot_name: str,
    fallback_index: int,
) -> np.ndarray:
    world = _contact_points_world(definition, target_matrices, foot_name, fallback_index)
    if len(world) == 1:
        return world[0].copy()
    return np.array((float(np.mean(world[:, 0])), float(np.mean(world[:, 1])), float(np.min(world[:, 2]))))


def _contact_lock_point(lock: np.ndarray, *, profiled: bool) -> np.ndarray:
    """Reduce a vector or multi-anchor lock to the point used by leg IK."""
    values = np.asarray(lock, dtype=np.float64)
    if not profiled or values.ndim == 1:
        return values.copy()
    return np.array((float(np.mean(values[:, 0])), float(np.mean(values[:, 1])), float(np.min(values[:, 2]))))


def _contact_points_world(
    definition: RetargetDefinition,
    target_matrices: np.ndarray,
    foot_name: str,
    fallback_index: int,
) -> np.ndarray:
    """Transform the complete sole profile into target/armature space."""
    points = definition.contact_points_local.get(foot_name)
    if points is None or len(points) == 0:
        return target_matrices[fallback_index, :3, 3][None, :].copy()
    bone_index = definition.contact_bone_indices.get(foot_name, fallback_index)
    matrix = target_matrices[bone_index]
    local = np.asarray(points, dtype=np.float64)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def _prevent_ground_penetration(
    root_locations: np.ndarray,
    target_matrices: np.ndarray,
    definition: RetargetDefinition,
) -> None:
    fallback = {name: index for index, name in enumerate(definition.source_names)}
    for frame in range(len(root_locations)):
        support_z = [
            _contact_point(definition, target_matrices[frame], name, fallback.get(name, bone_index))[2]
            + root_locations[frame, 2]
            for name, bone_index in definition.contact_bone_indices.items()
            if name in definition.contact_points_local
        ]
        if support_z:
            root_locations[frame, 2] += max(0.0, GROUND_CLEARANCE_M - min(support_z))


def _stabilize_quaternions(quaternions: np.ndarray) -> None:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    quaternions /= np.maximum(norms, 1e-12)
    for frame in range(1, len(quaternions)):
        dots = np.sum(quaternions[frame - 1] * quaternions[frame], axis=1)
        quaternions[frame, dots < 0.0] *= -1.0


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized WXYZ quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            quat = np.array(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            quat = np.array(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            quat = np.array(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def _quaternion_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    return matrix
