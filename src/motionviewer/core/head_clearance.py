"""Push a hand out of the character's head, by rotating the shoulder and nothing else.

The source is fitted to a bare SMPL-X body whose head is a scalp. The characters this
renders onto have hair, a topknot and a hat, so a gesture the source records as *near*
the head — scratching it, lifting a cup to the mouth, shielding the eyes — lands inside
the rendered silhouette. Measured over this dataset, the wrist enters the source's own
head sphere on 6 of 400 clips, worst case 2.4 cm from the head joint against a 12.9 cm
neck-to-head radius, so the hand is well inside even before the hair is added.

The correction is deliberately the smallest one that can work: a single rotation of the
**shoulder**, about the axis that carries the wrist directly away from the head. Every
joint below the shoulder comes along rigidly, so the elbow keeps its bend and the
gesture still reads as a scratch rather than a stiff-armed reach. Bending the elbow
instead would change the gesture; moving the wrist alone would need full-arm IK and
would leave the forearm pointing somewhere the elbow does not agree with.

Because it rotates a joint, this **edits the motion**, and it is off unless asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .smplx_fk import SMPLX_BODY22_PARENTS, global_rotations, rodrigues

HEAD, NECK = 15, 12
SHOULDERS = {"left": 16, "right": 17}
WRISTS = {"left": 20, "right": 21}
#: Joints that ride along when the shoulder turns: elbow then wrist.
_ARM_BELOW = {"left": (18, 20), "right": (19, 21)}

#: How much bigger than the source's own head to treat the obstacle as. The rendered
#: head is wider than the fitted scalp by roughly the hair, and 1.35 was the smallest
#: multiplier that cleared the visible silhouette on the scratch clip without visibly
#: straightening the arm on the others.
HEAD_MARGIN = 1.35
#: Cap on how far the shoulder may be turned. Past this the gesture stops being the
#: one the caption describes, and a clip that needs more is better left alone and
#: reported than quietly rewritten into a different motion.
MAX_PUSH_DEG = 22.0
_STEP_DEG = 1.0


@dataclass
class ClearanceReport:
    frames_corrected: int
    max_push_deg: float
    closest_before_cm: float
    closest_after_cm: float
    radius_cm: float
    hit_cap: bool

    def summary(self) -> str:
        if not self.frames_corrected:
            return "head clearance: nothing to do"
        capped = ", hit the angle cap" if self.hit_cap else ""
        return (
            f"head clearance: {self.frames_corrected} frames, "
            f"shoulder up to {self.max_push_deg:.1f} deg, "
            f"wrist {self.closest_before_cm:.1f} -> {self.closest_after_cm:.1f} cm "
            f"from head (needs {self.radius_cm:.1f}){capped}"
        )


def clear_head(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    joints: np.ndarray,
    *,
    margin: float = HEAD_MARGIN,
    max_push_deg: float = MAX_PUSH_DEG,
    smooth: int = 5,
) -> tuple[np.ndarray, np.ndarray, ClearanceReport]:
    """Return corrected ``body_pose`` and ``joints``, plus what was done.

    ``body_pose`` is taken and returned as (T, 21, 3) axis-angle, the layout the
    retarget already uses. ``global_orient`` and ``transl`` are read but never changed:
    turning the body or moving the root to solve a hand collision would move the feet.
    """
    global_orient = np.asarray(global_orient, dtype=np.float64)
    pose = np.asarray(body_pose, dtype=np.float64).reshape(len(global_orient), 21, 3).copy()
    joints = np.asarray(joints, dtype=np.float64)

    rotations = global_rotations(global_orient, pose)
    radius = float(np.linalg.norm(joints[:, HEAD] - joints[:, NECK], axis=1).mean()) * margin

    # Per-frame, per-side angle needed, solved by stepping outward until clear. A
    # closed form exists but the stepping version cannot overshoot, and at a degree a
    # step the whole dataset costs under a second.
    angles = {side: np.zeros(len(joints)) for side in SHOULDERS}
    axes = {side: np.zeros((len(joints), 3)) for side in SHOULDERS}
    for side, shoulder in SHOULDERS.items():
        wrist = WRISTS[side]
        for frame in range(len(joints)):
            head = joints[frame, HEAD]
            origin = joints[frame, shoulder]
            tip = joints[frame, wrist]
            if np.linalg.norm(tip - head) >= radius:
                continue
            arm = tip - origin
            away = tip - head
            arm_norm = np.linalg.norm(arm)
            if arm_norm < 1e-6:
                continue
            unit = arm / arm_norm
            # The component of "away from the head" that a rotation about the shoulder
            # can actually deliver: rotation cannot move the wrist along the arm.
            perpendicular = away - unit * float(away @ unit)
            if np.linalg.norm(perpendicular) < 1e-6:
                # Wrist is straight out along the head direction; any perpendicular
                # axis works, so take one that is stable rather than degenerate.
                perpendicular = np.cross(unit, np.array([0.0, 1.0, 0.0]))
                if np.linalg.norm(perpendicular) < 1e-6:
                    perpendicular = np.cross(unit, np.array([1.0, 0.0, 0.0]))
            axis = np.cross(unit, perpendicular / np.linalg.norm(perpendicular))
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                continue
            axis /= axis_norm
            angle = 0.0
            while angle < max_push_deg:
                angle += _STEP_DEG
                turned = rodrigues((axis * np.radians(angle))[None])[0]
                moved = origin + turned @ arm
                if np.linalg.norm(moved - head) >= radius:
                    break
            angles[side][frame] = angle
            axes[side][frame] = axis

    # Smooth the angle so the arm eases in and out instead of snapping on the frame the
    # threshold happens to be crossed. The axis is left per-frame: it is already
    # continuous because the geometry driving it is.
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        for side in angles:
            padded = np.pad(angles[side], smooth, mode="edge")
            angles[side] = np.convolve(padded, kernel, mode="same")[smooth:-smooth]

    corrected = 0
    hit_cap = False
    for side, shoulder in SHOULDERS.items():
        parent = SMPLX_BODY22_PARENTS[shoulder]
        for frame in range(len(joints)):
            angle = angles[side][frame]
            if angle <= 1e-6:
                continue
            corrected += 1
            hit_cap = hit_cap or angle >= max_push_deg - 1e-6
            delta = rodrigues((axes[side][frame] * np.radians(angle))[None])[0]
            # Pre-multiplying the *global* rotation turns the whole arm about the
            # shoulder in world space, which is what "away from the head" means; the
            # local value is then whatever produces that under the unchanged parent.
            target = delta @ rotations[frame, shoulder]
            pose[frame, shoulder - 1] = _axis_angle(rotations[frame, parent].T @ target)

    # Everything below the shoulder turns rigidly with it, so the corrected positions
    # come from rotating those joints about the shoulder directly. Rebuilding the whole
    # skeleton by forward kinematics from an averaged rest pose would be the obvious
    # alternative and is wrong here: the rest offsets are a least-squares fit, so FK
    # reproduces the *unmodified* joints only to 12-57 mm, and that residual would show
    # up as the feet and head having moved when nothing touched them.
    fixed_joints = joints.copy()
    for side, shoulder in SHOULDERS.items():
        for frame in range(len(joints)):
            angle = angles[side][frame]
            if angle <= 1e-6:
                continue
            delta = rodrigues((axes[side][frame] * np.radians(angle))[None])[0]
            origin = joints[frame, shoulder]
            for joint in _ARM_BELOW[side]:
                fixed_joints[frame, joint] = origin + delta @ (joints[frame, joint] - origin)
    before = min(float(np.linalg.norm(joints[:, w] - joints[:, HEAD], axis=1).min()) for w in WRISTS.values())
    after = min(
        float(np.linalg.norm(fixed_joints[:, w] - fixed_joints[:, HEAD], axis=1).min())
        for w in WRISTS.values()
    )
    report = ClearanceReport(
        frames_corrected=corrected,
        max_push_deg=float(max(angles[side].max() for side in angles)),
        closest_before_cm=before * 100.0,
        closest_after_cm=after * 100.0,
        radius_cm=radius * 100.0,
        hit_cap=hit_cap,
    )
    return pose, fixed_joints, report


def _axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Inverse of ``rodrigues`` for one matrix, routed through the module's own."""
    from .smplx_fk import axis_angle_from_rotations

    return axis_angle_from_rotations(matrix[None])[0]
