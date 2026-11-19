"""Remove solver snaps from baked garment physics, and leave the swing alone.

The garment on these characters is a rigid-body chain per panel, simulated by Bullet.
Most of the time it swings well. Occasionally a whole panel snaps: on Nilou's walk the
front panels drop 28 cm in 3 frames at frame 93, against the 12 frames free fall would
take over that height. Nothing in the body jerks there — pelvis acceleration is
ordinary — so it is the solver releasing accumulated constraint error, not cloth
responding to motion. Three global changes were measured and each made it worse:

    damping removed          snap frames  9 -> 10,  peak 415 -> 464 m/s^2
    substeps 12 -> 30        snap frames  9 -> 15,  peak 415 -> 1306 m/s^2
    per-finger hand hulls    snap frames  9 -> 27,  peak 415 -> 571 m/s^2

More substeps are worse, not better, because these joints are stiff limits and a
smaller timestep corrects the same violation with a proportionally larger impulse.

MMD animators handle this after the bake rather than inside the solver: bake, then
clean the curves. This is that step. Each body's track is taken relative to the
pelvis, so the walk is not mistaken for cloth motion; a frame is a snap only where the
acceleration is an outlier *for that body*, so a panel that swings hard all clip is not
flattened; and a snap is replaced by a smooth fit through the frames around it, so the
panel arrives where the simulation sent it, only not in one frame.

Pure NumPy over (T, N, 4, 4) matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DespikeSettings:
    #: A frame is a snap when a body's acceleration exceeds this multiple of that body's
    #: own median. Swing well above the median is ordinary; 6x is not.
    ratio: float = 6.0
    #: ...and exceeds this floor, so a body that barely moves cannot trigger on noise.
    floor: float = 20.0
    #: Frames either side of a snap that are refitted with it.
    radius: int = 3
    #: Iterations of detect-and-refit. A refit can expose a smaller snap behind it.
    passes: int = 3


def relative_tracks(pelvis: np.ndarray, bodies: np.ndarray) -> np.ndarray:
    """Each body's matrix in the pelvis frame. ``pelvis`` (T,4,4), ``bodies`` (T,N,4,4)."""
    return np.einsum("tij,tnjk->tnik", np.linalg.inv(pelvis), bodies)


def accelerations(positions: np.ndarray, fps: float) -> np.ndarray:
    """|second difference| per frame and body, (T, N), m/s^2. Ends are zero."""
    out = np.zeros(positions.shape[:2])
    out[1:-1] = np.linalg.norm(positions[2:] - 2 * positions[1:-1] + positions[:-2], axis=-1) * fps * fps
    return out


def snap_mask(acc: np.ndarray, settings: DespikeSettings) -> np.ndarray:
    """(T, N) bool: frames where a body's acceleration is an outlier for that body."""
    median = np.median(acc, axis=0, keepdims=True)
    return (acc > settings.ratio * median) & (acc > settings.floor)


def _windows(mask: np.ndarray, radius: int, frames: int) -> list[tuple[int, int]]:
    """Merge flagged frames, grown by ``radius``, into [start, end] spans."""
    spans: list[tuple[int, int]] = []
    for frame in np.nonzero(mask)[0]:
        start, end = max(0, frame - radius), min(frames - 1, frame + radius)
        if spans and start <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def _hermite(p0, p1, m0, m1, t):
    t = t[:, None]
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """(…,3,3) -> (…,4) w,x,y,z."""
    w = np.sqrt(np.maximum(0.0, 1.0 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2])) / 2
    x = np.sqrt(np.maximum(0.0, 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2
    y = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2
    z = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2
    x = np.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = np.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = np.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    q = np.stack([w, x, y, z], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        out = q0[None] + t[:, None] * (q1 - q0)[None]
        return out / np.linalg.norm(out, axis=-1, keepdims=True)
    theta = np.arccos(dot)
    s = np.sin(theta)
    return (np.sin((1 - t) * theta)[:, None] * q0 + np.sin(t * theta)[:, None] * q1) / s


def _refit(track: np.ndarray, start: int, end: int) -> None:
    """Replace frames strictly between ``start`` and ``end`` with a smooth bridge, in place.

    Position: cubic Hermite with end tangents from the frames just outside, so the bridge
    leaves and joins the simulation at the simulation's own velocity. Rotation: slerp.
    """
    frames = len(track)
    if end - start < 2:
        return
    p0, p1 = track[start, :3, 3], track[end, :3, 3]
    before = track[max(start - 1, 0), :3, 3]
    after = track[min(end + 1, frames - 1), :3, 3]
    span = float(end - start)
    m0 = (p0 - before) * span
    m1 = (after - p1) * span
    t = np.arange(1, end - start) / span
    track[start + 1 : end, :3, 3] = _hermite(p0, p1, m0, m1, t)
    q = _slerp(_quat_from_matrix(track[start, :3, :3]), _quat_from_matrix(track[end, :3, :3]), t)
    track[start + 1 : end, :3, :3] = _matrix_from_quat(q)


def despike(relative: np.ndarray, fps: float, settings: DespikeSettings = DespikeSettings()) -> tuple[np.ndarray, dict]:
    """Clean ``relative`` (T, N, 4, 4) pelvis-frame tracks. Returns (cleaned, report).

    Only flagged windows change; every other frame is returned bit-for-bit.
    """
    out = relative.copy()
    frames, bodies = relative.shape[:2]
    touched = np.zeros((frames, bodies), dtype=bool)
    before = accelerations(relative[:, :, :3, 3], fps)
    flagged_first = snap_mask(before, settings)
    for _ in range(settings.passes):
        mask = snap_mask(accelerations(out[:, :, :3, 3], fps), settings)
        if not mask.any():
            break
        for body in np.nonzero(mask.any(0))[0]:
            for start, end in _windows(mask[:, body], settings.radius, frames):
                # Anchor on frames that are themselves clean, one step outside the window.
                start, end = max(0, start - 1), min(frames - 1, end + 1)
                _refit(out[:, body], start, end)
                touched[start + 1 : end, body] = True
    after = accelerations(out[:, :, :3, 3], fps)
    report = {
        "bodies": int(bodies),
        "frames": int(frames),
        "snap_cells_before": int(flagged_first.sum()),
        "snap_cells_after": int(snap_mask(after, settings).sum()),
        "frames_with_mass_snap_before": int((flagged_first.sum(1) > 40).sum()),
        "frames_with_mass_snap_after": int((snap_mask(after, settings).sum(1) > 40).sum()),
        "peak_accel_before": float(before.max()),
        "peak_accel_after": float(after.max()),
        "share_of_cells_changed": float(touched.mean()),
    }
    return out, report


def smooth_tracks(relative: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass of pelvis-frame tracks over time, (T, N, 4, 4) -> same shape.

    For cloth that does not snap but *flickers*: Furina's rear coat in the walk wobbles
    2.0 mm per frame around its own path and reverses direction on 65% of frames, against
    0.5 mm on her body. Damping (both axes zero: 2.0 -> 2.4 mm), leg contact (panels
    100 mm from any collider flicker as much as touching ones) and joint limits (already
    widened by the profile) were measured and are not the cause.

    sigma is in frames. At 1.5 on that coat, jitter falls to 0.78 mm and reversals to
    10%, while the swing range keeps 94% — a stride is ~25 frames, far longer than the
    window. Positions are filtered directly; rotations are averaged as sign-aligned
    quaternions and renormalised, which is accurate for the small per-frame angles here.
    """
    if sigma <= 0.0:
        return relative.copy()
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    weights /= weights.sum()
    frames = len(relative)
    index = np.clip(np.arange(frames)[:, None] + offsets[None, :], 0, frames - 1)  # edge hold

    out = relative.copy()
    positions = relative[:, :, :3, 3]
    out[:, :, :3, 3] = np.einsum("k,tknd->tnd", weights, positions[index])

    quats = _quat_from_matrix(relative[:, :, :3, :3])  # (T, N, 4)
    # Align each frame to its predecessor so q and -q do not cancel in the average.
    for frame in range(1, frames):
        flip = np.einsum("nd,nd->n", quats[frame], quats[frame - 1]) < 0.0
        quats[frame, flip] *= -1.0
    averaged = np.einsum("k,tknd->tnd", weights, quats[index])
    averaged /= np.linalg.norm(averaged, axis=-1, keepdims=True)
    out[:, :, :3, :3] = _matrix_from_quat(averaged)
    return out
