from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


class RenderCapability(StrEnum):
    """Capabilities a loaded motion can offer to render/analysis code."""

    SMPLX_MESH = "smplx_mesh"
    SKELETON = "skeleton"
    CONTACTS = "contacts"
    PREFIX_MARKERS = "prefix_markers"
    METRICS = "metrics"


@dataclass(frozen=True)
class SequenceSegment:
    name: str
    start: int
    end: int

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class CoordinateSystem:
    vertical_axis: int = 1
    forward_axis: int | None = 2
    units: str = "meters"
    target: str = "blender_z_up"

    def to_json(self) -> dict[str, Any]:
        return {
            "vertical_axis": self.vertical_axis,
            "forward_axis": self.forward_axis,
            "units": self.units,
            "target": self.target,
        }


@dataclass
class BodyModelData:
    model_type: str
    global_orient: FloatArray
    body_pose: FloatArray
    transl: FloatArray
    betas: FloatArray
    left_hand_pose: FloatArray | None = None
    right_hand_pose: FloatArray | None = None
    jaw_pose: FloatArray | None = None
    leye_pose: FloatArray | None = None
    reye_pose: FloatArray | None = None
    expression: FloatArray | None = None
    gender: str = "neutral"

    @property
    def frames(self) -> int:
        return int(self.transl.shape[0])

    def to_json_summary(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "gender": self.gender,
            "frames": self.frames,
            "global_orient_shape": list(self.global_orient.shape),
            "body_pose_shape": list(self.body_pose.shape),
            "transl_shape": list(self.transl.shape),
            "betas_shape": list(self.betas.shape),
            "has_hands": self.left_hand_pose is not None and self.right_hand_pose is not None,
            "has_face": self.expression is not None,
        }


@dataclass
class MotionSequence:
    path: Path
    format_id: str
    source: str
    fps: float
    frames: int
    joints: FloatArray | None = None
    joint_names: list[str] = field(default_factory=list)
    body_model: BodyModelData | None = None
    segments: list[SequenceSegment] = field(default_factory=list)
    coordinate_system: CoordinateSystem = field(default_factory=CoordinateSystem)
    capabilities: set[RenderCapability] = field(default_factory=set)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def prefix_t(self) -> int | None:
        for segment in self.segments:
            if segment.name == "prefix":
                return segment.end - segment.start
        return None

    def require_joints(self) -> FloatArray:
        if self.joints is None:
            raise ValueError(f"{self.path} does not contain normalized joints")
        return self.joints

    def bounds(self) -> tuple[FloatArray, FloatArray]:
        joints = self.require_joints()
        flat = joints.reshape(-1, 3)
        return flat.min(axis=0), flat.max(axis=0)

    def to_json_summary(self) -> dict[str, Any]:
        mins, maxs = self.bounds() if self.joints is not None else (None, None)
        return {
            "path": str(self.path),
            "format_id": self.format_id,
            "source": self.source,
            "fps": self.fps,
            "frames": self.frames,
            "duration_s": self.duration_s,
            "capabilities": sorted(cap.value for cap in self.capabilities),
            "segments": [segment.to_json() for segment in self.segments],
            "coordinate_system": self.coordinate_system.to_json(),
            "joints_shape": list(self.joints.shape) if self.joints is not None else None,
            "bounds": {
                "min": mins.tolist() if mins is not None else None,
                "max": maxs.tolist() if maxs is not None else None,
            },
            "body_model": self.body_model.to_json_summary() if self.body_model else None,
            "extras": _json_safe_extras(self.extras),
        }


def _json_safe_extras(extras: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in extras.items():
        if isinstance(value, np.ndarray):
            safe[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
