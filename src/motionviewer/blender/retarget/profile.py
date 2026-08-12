from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_FBX_ROOT = Path("assets/fbx")


@dataclass(frozen=True)
class RetargetProfile:
    profile_id: str
    rig_family: str = "mixamo"
    bone_map: str = "auto"
    retarget_mode: str = "quality"
    validation_status: str = "active"

    def __post_init__(self) -> None:
        if self.rig_family != "mixamo":
            raise ValueError("Retarget profiles only support the mixamo rig family")
        if self.bone_map not in {"auto", "mixamo"}:
            raise ValueError("Mixamo retarget profiles only support bone_map 'auto' or 'mixamo'")
        if self.retarget_mode not in {"quality", "direct"}:
            raise ValueError("retarget_mode must be 'quality' or 'direct'")

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RetargetProfile:
        return cls(
            profile_id=str(data["profile_id"]),
            rig_family=str(data.get("rig_family", "mixamo")),
            bone_map=str(data.get("bone_map", "auto")),
            retarget_mode=str(data.get("retarget_mode", "quality")),
            validation_status=str(data.get("validation_status", "active")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "rig_family": self.rig_family,
            "bone_map": self.bone_map,
            "retarget_mode": self.retarget_mode,
            "validation_status": self.validation_status,
        }


@dataclass(frozen=True)
class RetargetAssetEntry:
    model_id: str
    path: Path
    profile_id: str
    status: str
    random_eligible: bool
    reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any], *, root: Path) -> RetargetAssetEntry:
        return cls(
            model_id=str(data["model_id"]),
            path=(root / str(data["path"])),
            profile_id=str(data.get("profile_id", "mixamo")),
            status=str(data.get("status", "pending")),
            random_eligible=bool(data.get("random_eligible", False)),
            reason=None if data.get("reason") is None else str(data.get("reason")),
            evidence=dict(data.get("evidence", {})),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "profile_id": self.profile_id,
            "status": self.status,
            "random_eligible": self.random_eligible,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class RetargetCatalog:
    root: Path
    profiles: dict[str, RetargetProfile]
    assets: dict[str, RetargetAssetEntry]

    def asset(self, model_id: str) -> RetargetAssetEntry | None:
        return self.assets.get(model_id)

    def profile(self, profile_id: str) -> RetargetProfile | None:
        return self.profiles.get(profile_id)

    def random_eligible_assets(self) -> list[RetargetAssetEntry]:
        return [
            asset
            for asset in self.assets.values()
            if asset.status == "approved" and asset.random_eligible and asset.path.is_file()
        ]

    def to_json(self) -> dict[str, Any]:
        return {
            "profiles": {key: value.to_json() for key, value in self.profiles.items()},
            "assets": [asset.to_json() for asset in self.assets.values()],
        }


def load_retarget_catalog(
    root: str | Path = DEFAULT_FBX_ROOT,
    *,
    catalog_path: str | Path | None = None,
) -> RetargetCatalog:
    root_path = Path(root)
    path = Path(catalog_path) if catalog_path is not None else root_path / "catalog.json"
    if not path.exists():
        return RetargetCatalog(root=root_path, profiles={}, assets={})
    raw = json.loads(path.read_text(encoding="utf-8"))
    profiles = {
        key: RetargetProfile.from_json({**value, "profile_id": value.get("profile_id", key)})
        for key, value in dict(raw.get("profiles", {})).items()
    }
    assets = {}
    for item in raw.get("assets", []):
        asset = RetargetAssetEntry.from_json(dict(item), root=root_path)
        assets[asset.model_id] = asset
    return RetargetCatalog(root=root_path, profiles=profiles, assets=assets)


def model_id_from_path(path: str | Path) -> str:
    return Path(path).stem.lower().replace(" ", "_")
