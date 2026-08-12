"""Backend protocol and supporting types for pluggable actor creation.

Each backend encapsulates how an actor (armature + mesh objects) is created
from motion data — SMPL-X mesh, FBX character, or future backends.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from motionviewer.core.smplx_actor import SmplxActor


class MaterialPolicy(StrEnum):
    APPLY_MATERIAL = "apply_material"
    PRESERVE_MATERIAL = "preserve_material"


class MotionBackend(Protocol):
    """Protocol that every actor-creation backend must satisfy."""

    backend_id: str
    description: str
    material_policy: MaterialPolicy

    def create_actor(
        self,
        path: str | Path,
        *,
        label: str,
        gender: str = "neutral",
        unit_scale: float = 1.0,
        layout_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        body_config: dict[str, Any] | None = None,
        motion_overrides: dict[str, Any] | None = None,
    ) -> SmplxActor: ...

    def resolve_paths(self, body_config: dict[str, Any], base: Path) -> dict[str, Any]:
        """Resolve relative paths in *body_config* against *base*.

        Called during job loading so bundle JSON contains absolute paths.
        Default: return *body_config* unchanged.
        """
        ...

    def validate_config(self, body_config: dict[str, Any]) -> list[str]:
        """Return a list of configuration errors (empty = valid).

        Default: no validation beyond what the schema enforces.
        """
        ...
