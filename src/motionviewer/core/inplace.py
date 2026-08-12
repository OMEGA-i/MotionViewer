from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def freeze_horizontal_root_transl_source(transl: FloatArray) -> FloatArray:
    """Freeze source y-up root horizontal motion (X/Z); keep vertical Y."""
    out = np.asarray(transl, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != 3 or out.shape[0] == 0:
        raise ValueError(f"transl must have shape (T, 3); got {out.shape}")
    out[:, 0] = out[0, 0]
    out[:, 2] = out[0, 2]
    return out


def freeze_horizontal_root_joints_blender(joints: FloatArray) -> FloatArray:
    """Freeze Blender z-up root horizontal motion (X/Y) for all joints; keep Z."""
    out = np.asarray(joints, dtype=np.float32).copy()
    if out.ndim != 3 or out.shape[-1] != 3 or out.shape[0] == 0:
        raise ValueError(f"joints must have shape (T, J, 3); got {out.shape}")
    root0 = out[0, 0, :2].copy()
    deltas = out[:, 0, :2] - root0
    out[:, :, 0] -= deltas[:, None, 0]
    out[:, :, 1] -= deltas[:, None, 1]
    return out
