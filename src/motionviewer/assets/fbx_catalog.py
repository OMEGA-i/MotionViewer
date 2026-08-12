from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from motionviewer.blender.retarget.profile import (
    RetargetAssetEntry,
    RetargetProfile,
    load_retarget_catalog,
    model_id_from_path,
)

FBX_BINARY_MAGIC = b"Kaydara FBX Binary"


@dataclass(frozen=True)
class FbxModel:
    model_id: str
    path: Path
    valid: bool
    reason: str | None = None
    bone_map: str = "auto"
    profile_id: str | None = None
    status: str = "pending"
    random_eligible: bool = False
    evidence: dict | None = None
    retarget_mode: str = "quality"

    def to_json(self) -> dict:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "valid": self.valid,
            "reason": self.reason,
            "bone_map": self.bone_map,
            "profile_id": self.profile_id,
            "status": self.status,
            "random_eligible": self.random_eligible,
            "evidence": self.evidence or {},
            "retarget_mode": self.retarget_mode,
        }


def list_fbx_models(root: str | Path = Path("assets/fbx")) -> list[FbxModel]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    models: list[FbxModel] = []
    for path in sorted(root_path.glob("*.fbx")):
        models.append(_inspect_fbx(path))
    return models


def valid_fbx_models(root: str | Path = Path("assets/fbx")) -> list[FbxModel]:
    return [model for model in list_fbx_models(root) if model.valid]


def choose_random_fbx(
    root: str | Path = Path("assets/fbx"),
    *,
    rng: random.Random | None = None,
) -> FbxModel:
    models = valid_fbx_models(root)
    if not models:
        raise ValueError(f"No valid binary FBX models found under {root}")
    picker = rng or random.Random()
    return picker.choice(models)


def validate_fbx_path(path: str | Path) -> list[str]:
    fbx_path = Path(path)
    if not fbx_path.exists():
        return [f"fbx_path does not exist: {fbx_path}"]
    if not fbx_path.is_file():
        return [f"fbx_path is not a file: {fbx_path}"]
    if not is_binary_fbx(fbx_path):
        return [f"fbx_path is not a Blender-compatible binary FBX: {fbx_path}"]
    return []


def is_binary_fbx(path: str | Path) -> bool:
    try:
        with Path(path).open("rb") as handle:
            header = handle.read(32)
    except OSError:
        return False
    return header.startswith(FBX_BINARY_MAGIC)


def _inspect_fbx(path: Path) -> FbxModel:
    model_id = model_id_from_path(path)
    catalog = load_retarget_catalog(path.parent)
    entry = catalog.asset(model_id)
    errors = validate_fbx_path(path)
    if errors:
        profile = catalog.profile(entry.profile_id) if entry is not None else None
        return _model_from_entry(path, model_id, entry, profile, valid=False, reason="; ".join(errors))
    if entry is None:
        return FbxModel(
            model_id=model_id,
            path=path,
            valid=False,
            reason="binary FBX has not been retarget-approved yet",
            status="pending",
        )
    valid = entry.status == "approved" and entry.random_eligible
    reason = entry.reason if not valid else None
    profile = catalog.profile(entry.profile_id)
    return _model_from_entry(path, model_id, entry, profile, valid=valid, reason=reason)


def _model_from_entry(
    path: Path,
    model_id: str,
    entry: RetargetAssetEntry | None,
    profile: RetargetProfile | None,
    *,
    valid: bool,
    reason: str | None,
) -> FbxModel:
    if entry is None:
        return FbxModel(model_id=model_id, path=path, valid=valid, reason=reason)
    return FbxModel(
        model_id=model_id,
        path=path,
        valid=valid,
        reason=reason,
        bone_map=profile.bone_map if profile is not None else "auto",
        profile_id=entry.profile_id,
        status=entry.status,
        random_eligible=entry.random_eligible,
        evidence=entry.evidence,
        retarget_mode=profile.retarget_mode if profile is not None else "quality",
    )
