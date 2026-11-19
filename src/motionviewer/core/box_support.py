"""Place a rigid box above existing hands without altering the motion."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupportedBoxFit:
    size: tuple[float, float, float]
    centres: np.ndarray
    rotations: np.ndarray
    report: dict


def fit_supported_box(palms: np.ndarray, normals: np.ndarray,
                      hand_vertices: tuple[np.ndarray, np.ndarray], torsos: np.ndarray, *,
                      size: tuple[float, float, float] = (.40, .22, .15),
                      gap_m: float = .001) -> SupportedBoxFit:
    """Fit the bottom face to the highest existing hand skin inside its footprint.

    A small orientation search balances the two support heights while keeping
    the box above the hands. Contact can occur on fingertips; palm orientation
    disagreement is reported, never repaired by changing the actor. Sizes remain
    constant and source arrays are read only.
    """
    palms, normals, torsos = (np.asarray(a, dtype=float) for a in (palms, normals, torsos))
    if (palms.ndim != 3 or palms.shape[1:] != (2, 3) or len(palms) == 0
            or normals.shape != palms.shape or torsos.shape != (len(palms), 3)):
        raise ValueError("Expected nonempty palm/normal (frames,2,3) and torso (frames,3) arrays")
    hands = tuple(np.asarray(a, dtype=float) for a in hand_vertices)
    if len(hands) != 2 or any(a.ndim != 3 or a.shape[0] != len(palms) or a.shape[2] != 3
                              or a.shape[1] == 0 for a in hands):
        raise ValueError("Both hands need nonempty evaluated vertices for every frame")
    dimensions = np.asarray(size, dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions < .04):
        raise ValueError("Fixed box dimensions must be finite and at least 40 mm")
    if not all(np.isfinite(a).all() for a in (palms, normals, torsos, *hands)):
        raise ValueError("Support inputs must be finite")
    if not np.isfinite(gap_m) or not 0 <= gap_m <= .01:
        raise ValueError("Support gap must be between zero and 10 mm")
    lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Palm normals must have nonzero length")
    normals = normals / lengths
    midpoint = palms.mean(axis=1)
    spans = palms[:, 1] - palms[:, 0]
    lengths = np.linalg.norm(spans, axis=1)
    if np.any(lengths < .03):
        raise ValueError("Hands closer than 30 mm cannot define a stable box span")
    x_axes = spans / lengths[:, None]
    up = np.array([0., 0., 1.])
    z_axes = up - x_axes * x_axes[:, 2, None]
    z_lengths = np.linalg.norm(z_axes, axis=1)
    if np.any(z_lengths < .2):
        raise ValueError("Nearly vertical hand separation does not define an upward support plane")
    z_axes /= z_lengths[:, None]
    y_axes = np.cross(z_axes, x_axes)
    reliable = np.flatnonzero(np.abs(np.einsum("ij,ij->i", y_axes, midpoint - torsos)) > .02)
    if len(reliable) and np.dot(y_axes[reliable[0]], midpoint[reliable[0]] - torsos[reliable[0]]) < 0:
        x_axes = -x_axes
        y_axes = -y_axes
    # Prefer a nearly level object over forcing a tip-to-tip contact by tipping
    # the box into the face. This bound changes the prop only, never the pose.
    angles = np.arange(-15., 15.1, 3.)

    def candidate(frame, degrees):
        theta = np.radians(degrees)
        z = np.cos(theta) * z_axes[frame] + np.sin(theta) * y_axes[frame]
        y = np.cross(z, x_axes[frame])
        rotation = np.column_stack((x_axes[frame], y, z))
        origin = midpoint[frame] + .02 * y
        heights, support_indices = [], []
        for vertices in hands:
            local = (vertices[frame] - origin) @ rotation
            inside = np.all(np.abs(local[:, :2]) <= dimensions[:2] / 2, axis=1)
            indices = np.flatnonzero(inside)
            if not len(indices):
                return None
            index = int(indices[np.argmax(local[indices, 2])])
            heights.append(float(local[index, 2]))
            support_indices.append(index)
        bottom = max(heights) + gap_m
        gaps = bottom - np.asarray(heights)
        palm_angles = np.arccos(np.clip(normals[frame] @ z, -1., 1.))
        return (origin + z * (bottom + dimensions[2] / 2), rotation, gaps,
                bottom - ((palms[frame] - origin) @ z), palm_angles, support_indices)

    costs = np.full((len(palms), len(angles)), np.inf)
    for frame in range(len(palms)):
        for state, degrees in enumerate(angles):
            value = candidate(frame, degrees)
            if value is not None:
                costs[frame, state] = (((max(value[2]) - gap_m) / .01) ** 2
                                       + .1 * np.mean(value[4] ** 2) + .3 * (degrees / 15) ** 2)
    if np.any(~np.isfinite(costs).any(axis=1)):
        raise ValueError("Fixed box footprint cannot reach both hands in every frame")
    previous = np.zeros_like(costs, dtype=int)
    total = costs[0].copy()
    transitions = np.abs(angles[:, None] - angles[None, :])
    transition_cost = np.where(transitions <= 6., .03 * (transitions / 6) ** 2, np.inf)
    for frame in range(1, len(palms)):
        trial = total[:, None] + transition_cost
        previous[frame] = np.argmin(trial, axis=0)
        total = costs[frame] + trial[previous[frame], np.arange(len(angles))]
    if not np.isfinite(total).any():
        raise ValueError("No temporally bounded box orientation can reach both hands")
    states = [int(np.argmin(total))]
    for frame in range(len(palms) - 1, 0, -1):
        states.append(int(previous[frame, states[-1]]))
    states.reverse()
    selected_angles = angles[states]
    # Smooth the quantized search result, then recompute contact on the exact
    # resulting plane. Smoothing never reuses stale support-gap measurements.
    selected_angles = np.convolve(np.pad(selected_angles, (2, 2), mode="edge"),
                                  np.array([1., 2., 3., 2., 1.]) / 9, mode="valid")
    selected = [candidate(i, angle) for i, angle in enumerate(selected_angles)]
    if any(value is None for value in selected):
        raise ValueError("Smoothed box footprint cannot reach both hands in every frame")
    centres = np.asarray([a[0] for a in selected])
    rotations = np.asarray([a[1] for a in selected])
    rows = [{"frame_offset": i, "support_gap_mm": (a[2] * 1000).tolist(),
             "palm_centre_to_bottom_mm": (a[3] * 1000).tolist(),
             "palm_normal_to_box_up_degrees": np.degrees(a[4]).tolist(),
             "box_up_to_world_up_degrees": float(np.degrees(np.arccos(np.clip(a[1][2, 2], -1., 1.)))),
             "support_vertex_indices": a[5], "tilt_about_hand_span_degrees": float(selected_angles[i])}
            for i, a in enumerate(selected)]
    report = {"mode": "box_above_unchanged_hands", "size_m": dimensions.tolist(),
              "actor_pose_modified": False, "source_motion_modified": False,
              "support_gap_max_mm": max(max(r["support_gap_mm"]) for r in rows),
              "frame_offsets_support_gap_over_10mm": [i for i, r in enumerate(rows) if max(r["support_gap_mm"]) > 10],
              "frame_offsets_palm_misalignment_over_45deg": [i for i, r in enumerate(rows) if max(r["palm_normal_to_box_up_degrees"]) > 45],
              "frame_offsets_palm_back_facing_box": [i for i, r in enumerate(rows) if max(r["palm_normal_to_box_up_degrees"]) > 90],
              "max_tilt_about_hand_span_degrees": float(np.max(np.abs(selected_angles))),
              "max_box_up_world_tilt_degrees": max(r["box_up_to_world_up_degrees"] for r in rows),
              "source_palm_height_difference_max_mm": float(np.max(np.abs(spans[:, 2])) * 1000),
              "max_frame_tilt_change_degrees": float(np.max(np.abs(np.diff(selected_angles)), initial=0)),
              "frames": rows,
              "limitations": "Box-only placement. Support may be on fingertips rather than palms; "
                             "palm orientation and unequal support gaps remain as source-data limitations. "
                             "This is a geometric placement, not a validated load-bearing grasp or dynamic simulation. "
                             "No hand, wrist, arm or finger rotations are changed; whole-body collision needs a separate mesh audit."}
    return SupportedBoxFit(tuple(float(v) for v in dimensions), centres, rotations, report)
