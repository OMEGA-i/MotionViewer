from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from motionviewer.core.schema import CoordinateSystem, MotionSequence, RenderCapability

from .base import MotionFormatError, ProbeResult


class JointsNpzLoader:
    format_id = "joints_npz"
    extensions = (".npz",)
    capabilities = frozenset({RenderCapability.SKELETON, RenderCapability.CONTACTS, RenderCapability.METRICS})
    description = "Generic NPZ containing a joints array with shape (T, J, 3)."

    def probe(self, path: Path) -> ProbeResult:
        if path.suffix.lower() != ".npz":
            return ProbeResult(False, reason="suffix is not .npz")
        try:
            with np.load(path, allow_pickle=False) as data:
                key = _find_joints_key(data.files)
                if key is None:
                    return ProbeResult(False, reason="no joints/joints22 key found")
                shape = data[key].shape
                matched = len(shape) == 3 and shape[-1] == 3
                return ProbeResult(
                    matched, self.format_id if matched else None, 0.55, f"joints key {key}", {"key": key}
                )
        except Exception as exc:
            return ProbeResult(False, reason=f"NPZ probe failed: {exc}")

    def load(self, path: Path, options: dict[str, Any] | None = None) -> MotionSequence:
        opts = options or {}
        with np.load(path, allow_pickle=False) as data:
            key = str(opts.get("joints_key") or _find_joints_key(data.files) or "")
            if not key:
                raise MotionFormatError(f"{path} has no joints key; pass loader option joints_key")
            joints = np.asarray(data[key], dtype=np.float32)
            fps = float(opts.get("fps", _optional_scalar(data, "fps", 20.0)))
            source = str(opts.get("label") or _optional_scalar(data, "source", path.stem))
        sequence = _make_joints_sequence(path, self.format_id, joints, fps, source, opts, {"joints_key": key})
        self.validate(sequence)
        return sequence

    def validate(self, sequence: MotionSequence) -> None:
        _validate_joints(sequence)


class JointsNpyLoader:
    format_id = "joints_npy"
    extensions = (".npy",)
    capabilities = frozenset({RenderCapability.SKELETON, RenderCapability.CONTACTS, RenderCapability.METRICS})
    description = "Raw NPY joints array with shape (T, J, 3)."

    def probe(self, path: Path) -> ProbeResult:
        if path.suffix.lower() != ".npy":
            return ProbeResult(False, reason="suffix is not .npy")
        return ProbeResult(True, self.format_id, 0.4, "raw NPY requires shape validation on load")

    def load(self, path: Path, options: dict[str, Any] | None = None) -> MotionSequence:
        opts = options or {}
        joints = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        fps = float(opts.get("fps", 20.0))
        source = str(opts.get("label") or path.stem)
        sequence = _make_joints_sequence(path, self.format_id, joints, fps, source, opts, {})
        self.validate(sequence)
        return sequence

    def validate(self, sequence: MotionSequence) -> None:
        _validate_joints(sequence)


def _make_joints_sequence(
    path: Path,
    format_id: str,
    joints: np.ndarray,
    fps: float,
    source: str,
    options: dict[str, Any],
    extras: dict[str, Any],
) -> MotionSequence:
    vertical_axis = int(options.get("vertical_axis", 1))
    forward_axis_opt = options.get("forward_axis", 2)
    forward_axis = None if forward_axis_opt is None else int(forward_axis_opt)
    joint_names = list(options.get("joint_names", []))
    return MotionSequence(
        path=path,
        format_id=format_id,
        source=source,
        fps=fps,
        frames=int(joints.shape[0]) if joints.ndim >= 1 else 0,
        joints=joints,
        joint_names=joint_names,
        coordinate_system=CoordinateSystem(vertical_axis=vertical_axis, forward_axis=forward_axis),
        capabilities=set(JointsNpzLoader.capabilities),
        extras=extras,
    )


def _find_joints_key(keys: list[str]) -> str | None:
    for key in ("joints", "joints22", "positions"):
        if key in keys:
            return key
    return None


def _optional_scalar(data: Any, key: str, default: Any) -> Any:
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if getattr(value, "ndim", 0) == 0 else default


def _validate_joints(sequence: MotionSequence) -> None:
    if sequence.joints is None or sequence.joints.ndim != 3 or sequence.joints.shape[-1] != 3:
        raise MotionFormatError(f"{sequence.path} must contain joints with shape (T, J, 3)")
    if sequence.frames <= 0:
        raise MotionFormatError(f"{sequence.path} contains no frames")
    if sequence.fps <= 0:
        raise MotionFormatError("fps must be positive")
