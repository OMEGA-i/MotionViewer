"""Retarget profile, calibration, pipeline stages, and validation helpers.

Public API:
- ``RetargetProfile`` / ``RetargetAssetEntry`` / ``RetargetCatalog`` — catalog data model
- ``CalibrationResult`` — import-time calibration facts
- ``create_fbx_actor_from_npz`` — pipeline orchestrator (entry point)
- ``resolve_bone_map`` / ``BONE_MAP_PRESETS`` — bone-map resolution
"""

from ._resolve import BONE_MAP_PRESETS, resolve_bone_map
from .calibration import CalibrationResult, MixamoNameAdapter, MixamoRigProfile, inspect_mixamo_rig
from .export import export_fbx_animation, validate_fbx_roundtrip
from .pipeline import create_fbx_actor_from_npz
from .profile import RetargetAssetEntry, RetargetCatalog, RetargetProfile

__all__ = [
    "BONE_MAP_PRESETS",
    "CalibrationResult",
    "MixamoNameAdapter",
    "MixamoRigProfile",
    "RetargetAssetEntry",
    "RetargetCatalog",
    "RetargetProfile",
    "create_fbx_actor_from_npz",
    "export_fbx_animation",
    "inspect_mixamo_rig",
    "resolve_bone_map",
    "validate_fbx_roundtrip",
]
