from __future__ import annotations


class PackageError(ValueError):
    """Base error for Sample Packing Protocol package handling."""


class PackageFormatError(PackageError):
    """Raised when a path is not a recognizable protocol package."""


class PackageValidationError(PackageError):
    """Raised when package contents fail structural/asset validation."""


class PackageSelectionError(PackageError):
    """Raised when a clip/source selection cannot be resolved."""


class PackagePayloadError(PackageError):
    """Raised when a selected asset cannot be materialized safely."""
