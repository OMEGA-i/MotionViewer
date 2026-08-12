"""Pluggable backend registry for actor creation.

Mirrors the pattern established by ``MotionFormatRegistry`` in loaders/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from motionviewer.core.smplx_actor import SmplxActor

from .backend import MaterialPolicy, MotionBackend


class BackendRegistry:
    def __init__(self, backends: list[MotionBackend] | None = None) -> None:
        self._backends: dict[str, MotionBackend] = {}
        for backend in backends or []:
            self.register(backend)

    def register(self, backend: MotionBackend) -> None:
        if backend.backend_id in self._backends:
            raise ValueError(f"Backend {backend.backend_id!r} is already registered")
        self._backends[backend.backend_id] = backend

    def get(self, backend_id: str) -> MotionBackend:
        try:
            return self._backends[backend_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._backends))
            raise ValueError(f"Unknown backend {backend_id!r}. Available: {available}") from exc

    def create_actor(
        self,
        backend_id: str,
        path: str | Path,
        *,
        label: str,
        gender: str = "neutral",
        unit_scale: float = 1.0,
        layout_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        body_config: dict[str, Any] | None = None,
        motion_overrides: dict[str, Any] | None = None,
    ) -> SmplxActor:
        backend = self.get(backend_id)
        return backend.create_actor(
            path,
            label=label,
            gender=gender,
            unit_scale=unit_scale,
            layout_offset=layout_offset,
            body_config=body_config or {},
            motion_overrides=motion_overrides,
        )

    def material_policy_for(self, backend_id: str) -> MaterialPolicy:
        return self.get(backend_id).material_policy

    def resolve_body(self, body_config: dict[str, Any] | None, base: Path) -> dict[str, Any]:
        """Resolve paths in *body_config* using the appropriate backend."""
        if not body_config:
            return {}
        backend_id = body_config.get("backend", "blender_smplx_addon")
        try:
            backend = self.get(backend_id)
        except ValueError:
            return dict(body_config)
        return backend.resolve_paths(dict(body_config), base)

    def validate_body(self, body_config: dict[str, Any] | None) -> list[str]:
        if not body_config:
            return []
        backend_id = body_config.get("backend", "blender_smplx_addon")
        try:
            backend = self.get(backend_id)
        except ValueError:
            return [f"Unknown backend {backend_id!r}"]
        return backend.validate_config(dict(body_config))

    @property
    def backends(self) -> list[MotionBackend]:
        return list(self._backends.values())


def default_backend_registry() -> BackendRegistry:
    """Return a registry pre-populated with built-in backends.

    Backend classes are imported lazily because they live in modules that
    import ``bpy`` — this factory is safe to call outside Blender for path
    resolution and validation (the backends are only instantiated when
    ``create_actor`` or ``material_policy_for`` are called).
    """

    from .backend import MaterialPolicy

    class _LazySmplxAddonBackend:
        backend_id = "blender_smplx_addon"
        description = "Native SMPL-X mesh via the SMPL-X for Blender addon."
        material_policy = MaterialPolicy.APPLY_MATERIAL

        def create_actor(
            self,
            path,
            *,
            label,
            gender="neutral",
            unit_scale=1.0,
            layout_offset=(0, 0, 0),
            body_config=None,
            motion_overrides=None,
        ):
            from .smplx_mesh import create_smplx_actor_from_npz

            return create_smplx_actor_from_npz(
                path,
                label=label,
                gender=gender,
                unit_scale=unit_scale,
                layout_offset=layout_offset,
                motion_overrides=motion_overrides,
            )

        def resolve_paths(self, body_config, base):
            return body_config

        def validate_config(self, body_config):
            return []

    class _LazyFBXSkeletonBackend:
        backend_id = "fbx_skeleton"
        description = "FBX character driven by retargeted SMPL-X motion."
        material_policy = MaterialPolicy.PRESERVE_MATERIAL

        def create_actor(
            self,
            path,
            *,
            label,
            gender="neutral",
            unit_scale=1.0,
            layout_offset=(0, 0, 0),
            body_config=None,
            motion_overrides=None,
        ):
            from .fbx_skeleton import create_fbx_actor_from_npz

            cfg = body_config or {}
            return create_fbx_actor_from_npz(
                path,
                label=label,
                fbx_path=cfg.get("fbx_path", ""),
                bone_map=cfg.get("bone_map", "auto"),
                gender=gender,
                unit_scale=unit_scale,
                fbx_scale=float(cfg.get("fbx_scale", 1.0)),
                retarget_mode=cfg.get("retarget_mode", "quality"),
                layout_offset=layout_offset,
                motion_overrides=motion_overrides,
            )

        def resolve_paths(self, body_config, base):
            cfg = dict(body_config)
            fbx = cfg.get("fbx_path")
            if fbx and not Path(fbx).is_absolute():
                cfg["fbx_path"] = str((base / fbx).resolve())
            bone_map = cfg.get("bone_map")
            if bone_map and str(bone_map).endswith(".json") and not Path(bone_map).is_absolute():
                cfg["bone_map"] = str((base / bone_map).resolve())
            return cfg

        def validate_config(self, body_config):
            errors = []
            fbx_path = body_config.get("fbx_path")
            if not fbx_path:
                errors.append("fbx_path is required for fbx_skeleton backend")
            else:
                from motionviewer.assets.fbx_catalog import validate_fbx_path

                errors.extend(validate_fbx_path(fbx_path))
            return errors

    return BackendRegistry([_LazySmplxAddonBackend(), _LazyFBXSkeletonBackend()])
