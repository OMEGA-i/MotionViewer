from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from motionviewer.core.schema import MotionSequence, RenderCapability


class MotionFormatError(ValueError):
    """Base error for format detection, parsing, and validation."""


class UnsupportedFormatError(MotionFormatError):
    """Raised when no loader can handle an input."""


class AmbiguousFormatError(MotionFormatError):
    """Raised when multiple loaders claim the same input without override."""


@dataclass(frozen=True)
class ProbeResult:
    matched: bool
    format_id: str | None = None
    confidence: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoaderInfo:
    format_id: str
    extensions: tuple[str, ...]
    capabilities: frozenset[RenderCapability]
    description: str


class MotionFormatLoader(Protocol):
    format_id: str
    extensions: tuple[str, ...]
    capabilities: frozenset[RenderCapability]
    description: str

    def probe(self, path: Path) -> ProbeResult: ...

    def load(self, path: Path, options: dict[str, Any] | None = None) -> MotionSequence: ...

    def validate(self, sequence: MotionSequence) -> None: ...

    def info(self) -> LoaderInfo:
        return LoaderInfo(
            format_id=self.format_id,
            extensions=self.extensions,
            capabilities=self.capabilities,
            description=self.description,
        )
