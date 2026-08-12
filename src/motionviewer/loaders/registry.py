from __future__ import annotations

from pathlib import Path
from typing import Any

from motionviewer.core.schema import MotionSequence

from .base import AmbiguousFormatError, MotionFormatLoader, ProbeResult, UnsupportedFormatError
from .joints import JointsNpyLoader, JointsNpzLoader
from .smplx_npz import SmplxBody22NpzLoader


class MotionFormatRegistry:
    def __init__(self, loaders: list[MotionFormatLoader] | None = None) -> None:
        self._loaders: dict[str, MotionFormatLoader] = {}
        for loader in loaders or []:
            self.register(loader)

    def register(self, loader: MotionFormatLoader) -> None:
        if loader.format_id in self._loaders:
            raise ValueError(f"Loader {loader.format_id!r} is already registered")
        self._loaders[loader.format_id] = loader

    @property
    def loaders(self) -> list[MotionFormatLoader]:
        return list(self._loaders.values())

    def get(self, format_id: str) -> MotionFormatLoader:
        try:
            return self._loaders[format_id]
        except KeyError as exc:
            supported = ", ".join(sorted(self._loaders))
            raise UnsupportedFormatError(
                f"Unknown format {format_id!r}. Supported formats: {supported}"
            ) from exc

    def probe(self, path: str | Path) -> list[ProbeResult]:
        p = Path(path)
        results = [loader.probe(p) for loader in self.loaders]
        return sorted((r for r in results if r.matched), key=lambda r: r.confidence, reverse=True)

    def select(self, path: str | Path, format_id: str | None = None) -> MotionFormatLoader:
        p = Path(path)
        if format_id:
            loader = self.get(format_id)
            result = loader.probe(p)
            if not result.matched:
                raise UnsupportedFormatError(
                    f"{p} was explicitly configured as {format_id!r}, but probe failed: {result.reason}"
                )
            return loader

        matches = self.probe(p)
        if not matches:
            supported = ", ".join(sorted(self._loaders))
            raise UnsupportedFormatError(f"No loader recognized {p}. Supported formats: {supported}")
        best_confidence = matches[0].confidence
        best = [m for m in matches if m.confidence == best_confidence]
        if len(best) > 1:
            formats = ", ".join(str(m.format_id) for m in best)
            raise AmbiguousFormatError(f"{p} matched multiple loaders with equal confidence: {formats}")
        return self.get(str(matches[0].format_id))

    def load(
        self,
        path: str | Path,
        *,
        format_id: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> MotionSequence:
        p = Path(path)
        loader = self.select(p, format_id)
        return loader.load(p, options or {})

    def supported_formats(self) -> list[dict[str, Any]]:
        return [
            {
                "format_id": loader.format_id,
                "extensions": list(loader.extensions),
                "capabilities": sorted(cap.value for cap in loader.capabilities),
                "description": loader.description,
            }
            for loader in self.loaders
        ]


def default_registry() -> MotionFormatRegistry:
    return MotionFormatRegistry([SmplxBody22NpzLoader(), JointsNpzLoader(), JointsNpyLoader()])
