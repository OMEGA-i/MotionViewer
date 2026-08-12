from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from motionviewer.core.schema import (
    BodyModelData,
    CoordinateSystem,
    MotionSequence,
    RenderCapability,
    SequenceSegment,
)

from .base import MotionFormatError, ProbeResult

SMPLX_BODY22_FORMAT = "smplx_body22_fitted_aa"
SMPLX_BODY22_NATIVE_FORMAT = "smplx_body22_native_aa"
SUPPORTED_INTERNAL_FORMATS = frozenset({SMPLX_BODY22_FORMAT, SMPLX_BODY22_NATIVE_FORMAT})


class SmplxBody22NpzLoader:
    format_id = SMPLX_BODY22_FORMAT
    extensions = (".smplx.npz", ".npz")
    capabilities = frozenset(
        {
            RenderCapability.SMPLX_MESH,
            RenderCapability.SKELETON,
            RenderCapability.CONTACTS,
            RenderCapability.PREFIX_MARKERS,
            RenderCapability.METRICS,
        }
    )
    description = "Body-only SMPL-X axis-angle NPZ with joints22 and prefix metadata."

    def probe(self, path: Path) -> ProbeResult:
        if not _has_supported_suffix(path):
            return ProbeResult(False, reason="suffix is not .npz or .smplx.npz")
        try:
            with np.load(path, allow_pickle=False) as data:
                fmt = _scalar_str(data["format"]) if "format" in data.files else None
                required = {"joints22", "transl", "global_orient", "body_pose", "betas"}
                if fmt in SUPPORTED_INTERNAL_FORMATS:
                    return ProbeResult(True, self.format_id, 1.0, "internal format matched", {"format": fmt})
                if required.issubset(set(data.files)):
                    return ProbeResult(True, self.format_id, 0.75, "required SMPL-X keys present")
        except Exception as exc:
            return ProbeResult(False, reason=f"NPZ probe failed: {exc}")
        return ProbeResult(False, reason="required SMPL-X keys not present")

    def load(self, path: Path, options: dict[str, Any] | None = None) -> MotionSequence:
        opts = options or {}
        with np.load(path, allow_pickle=False) as data:
            _require_keys(
                data.files,
                [
                    "joints22",
                    "transl",
                    "prefix_T",
                    "fps",
                    "source",
                    "format",
                    "global_orient",
                    "body_pose",
                    "betas",
                ],
                path,
            )
            fmt = _scalar_str(data["format"])
            if fmt not in SUPPORTED_INTERNAL_FORMATS:
                expected = ", ".join(sorted(SUPPORTED_INTERNAL_FORMATS))
                raise MotionFormatError(f"{path} declares format {fmt!r}, expected one of: {expected}")

            joints = np.asarray(data["joints22"], dtype=np.float32)
            transl = np.asarray(data["transl"], dtype=np.float32)
            global_orient = np.asarray(data["global_orient"], dtype=np.float32)
            body_pose = np.asarray(data["body_pose"], dtype=np.float32)
            betas = np.asarray(data["betas"], dtype=np.float32)
            fps = float(data["fps"].item())
            prefix_t = int(data["prefix_T"].item())
            source = str(opts.get("label") or _scalar_str(data["source"]))
            fit_mse = float(data["fit_mse"].item()) if "fit_mse" in data.files else None

        frames = int(joints.shape[0])
        segments = [SequenceSegment("prefix", 0, prefix_t)]
        if prefix_t < frames:
            segments.append(SequenceSegment("generated", prefix_t, frames))
        sequence = MotionSequence(
            path=path,
            format_id=self.format_id,
            source=source,
            fps=fps,
            frames=frames,
            joints=joints,
            joint_names=BODY22_JOINT_NAMES.copy(),
            body_model=BodyModelData(
                model_type="smplx",
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                betas=betas,
                gender=str(opts.get("gender", "neutral")),
            ),
            segments=segments,
            coordinate_system=CoordinateSystem(vertical_axis=1, forward_axis=2, units="meters"),
            capabilities=set(self.capabilities),
            extras={"fit_mse": fit_mse, "prefix_T": prefix_t, "neutral_hands_face": True},
        )
        self.validate(sequence)
        return sequence

    def validate(self, sequence: MotionSequence) -> None:
        if sequence.joints is None or sequence.joints.ndim != 3 or sequence.joints.shape[1:] != (22, 3):
            raise MotionFormatError("SMPL-X body22 sequences require joints22 shape (T, 22, 3)")
        if sequence.body_model is None:
            raise MotionFormatError("SMPL-X body22 sequences require body_model data")
        body = sequence.body_model
        t = sequence.frames
        checks = {
            "transl": body.transl.shape == (t, 3),
            "global_orient": body.global_orient.shape == (t, 3),
            "body_pose": body.body_pose.shape == (t, 63),
            "betas": body.betas.ndim == 1 and body.betas.shape[0] >= 10,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise MotionFormatError(f"Invalid SMPL-X body22 shapes for {sequence.path}: {', '.join(failed)}")
        if sequence.fps <= 0:
            raise MotionFormatError("fps must be positive")


BODY22_JOINT_NAMES = [
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
]


def _has_supported_suffix(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".smplx.npz") or name.endswith(".npz")


def _require_keys(keys: list[str], required: list[str], path: Path) -> None:
    missing = [key for key in required if key not in keys]
    if missing:
        raise MotionFormatError(f"{path} is missing required keys: {', '.join(missing)}")


def _scalar_str(value: np.ndarray) -> str:
    return str(value.item())
