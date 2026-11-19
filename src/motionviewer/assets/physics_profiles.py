"""Asset-bound secondary-physics settings and explicit visual-review evidence.

Loading a profile never edits an asset, motion, skeleton, or Blender scene. The
caller applies only ``bullet_overrides`` and records its *final merged* settings.
Review samples describe particular renders; they do not certify a character or
another motion. Optional mesh-cloth settings use the adapter's pure validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from motionviewer.assets.physics_joint_rules import (
    JointRule,
    SideLinkRule,
    joint_rules_from_json,
    side_link_rules_from_json,
)
from motionviewer.blender.mmd_mesh_cloth import MeshClothSettings

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_KEYS = {
    "schema_version", "profile_id", "asset_sha256", "import_scale", "bullet",
    "visual_verified", "notes", "mesh_cloth", "joint_rules", "side_link_rules",
}
_NUMERIC_BULLET = {
    "substeps": (int, 1, 1000),
    "iterations": (int, 1, 1000),
    "settle_frames": (int, 0, 10000),
    "collision_margin": (float, 0, 0.1),
    "non_collision_distance_scale": (float, 0, 10),
    "cloth_angular_slack_degrees": (float, 0, 45),
    "cloth_linear_slack_m": (float, 0, 0.01),
    "gravity": (float, -100, 0),
    "smooth": (float, 0, 5),
}
_BULLET_KEYS = set(_NUMERIC_BULLET) | {"damping_mode", "cloth_bone_tokens", "fit_hand_colliders", "despike", "neutral_settle"}
_SAMPLE_KEYS = {
    "motion_sha256", "frame_start", "frames", "fps", "views", "display_context",
    "evidence", "motion_id", "notes",
}


def _object(value: Any, label: str, allowed: set[str], required: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown or missing:
        raise ValueError(f"{label}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _number(value: Any, label: str, kind: type, lower: float, upper: float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if kind is int and not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{label} must be finite and in [{lower}, {upper}]")
    return kind(value)


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty string list")
    result = tuple(_text(v, label) for v in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


@dataclass(frozen=True)
class VisualVerification:
    motion_sha256: str
    frame_start: int
    frames: int
    fps: float
    views: tuple[str, ...]
    display_context: str
    evidence: str
    motion_id: str | None = None
    notes: str | None = None

    def to_json(self) -> dict:
        return {
            "motion_sha256": self.motion_sha256, "frame_start": self.frame_start,
            "frames": self.frames, "fps": self.fps, "views": list(self.views),
            "display_context": self.display_context, "evidence": self.evidence,
            **({"motion_id": self.motion_id} if self.motion_id is not None else {}),
            **({"notes": self.notes} if self.notes is not None else {}),
        }


@dataclass(frozen=True)
class PhysicsProfile:
    profile_id: str
    asset_sha256: str
    import_scale: float
    bullet: tuple[tuple[str, Any], ...]
    visual_verified: tuple[VisualVerification, ...]
    notes: str | None = None
    mesh_cloth: MeshClothSettings | None = None
    joint_rules: tuple[JointRule, ...] = ()
    side_link_rules: tuple[SideLinkRule, ...] = ()

    @property
    def bullet_overrides(self) -> dict:
        """Only explicitly configured BulletSettings keyword arguments."""
        result = dict(self.bullet)
        if "damping_mode" in result:
            result["fix_damping"] = result.pop("damping_mode") == "legacy-inverted"
        return result

    @property
    def mesh_cloth_settings(self) -> dict | None:
        return asdict(self.mesh_cloth) if self.mesh_cloth is not None else None

    def to_json(self) -> dict:
        bullet = {k: list(v) if isinstance(v, tuple) else v for k, v in self.bullet}
        return {
            "schema_version": 1, "profile_id": self.profile_id,
            "asset_sha256": self.asset_sha256, "import_scale": self.import_scale,
            "bullet": bullet, "visual_verified": [v.to_json() for v in self.visual_verified],
            **({"notes": self.notes} if self.notes is not None else {}),
            **({"mesh_cloth": self.mesh_cloth_settings} if self.mesh_cloth is not None else {}),
            **({"joint_rules": [rule.to_json() for rule in self.joint_rules]} if self.joint_rules else {}),
            **({"side_link_rules": [rule.to_json() for rule in self.side_link_rules]} if self.side_link_rules else {}),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def physics_profile_from_json(value: Any) -> PhysicsProfile:
    data = _object(value, "physics profile", _PROFILE_KEYS,
                   _PROFILE_KEYS - {"notes", "mesh_cloth", "joint_rules", "side_link_rules"})
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("physics profile schema_version must be 1")
    bullet = _object(data["bullet"], "bullet", _BULLET_KEYS, set())
    settings = {}
    for key, raw in bullet.items():
        if key in _NUMERIC_BULLET:
            settings[key] = _number(raw, f"bullet.{key}", *_NUMERIC_BULLET[key])
        elif key == "damping_mode":
            if raw not in ("pmx", "legacy-inverted"):
                raise ValueError("bullet.damping_mode must be pmx or legacy-inverted")
            settings[key] = raw
        elif key == "cloth_bone_tokens":
            settings[key] = _strings(raw, "bullet.cloth_bone_tokens")
        elif key in ("fit_hand_colliders", "despike", "neutral_settle"):
            if type(raw) is not bool:
                raise ValueError(f"bullet.{key} must be boolean")
            settings[key] = raw
    if not isinstance(data["visual_verified"], list):
        raise ValueError("visual_verified must be a list of reviewed samples, possibly empty")
    samples = []
    for raw in data["visual_verified"]:
        sample = _object(raw, "visual_verified sample", _SAMPLE_KEYS, _SAMPLE_KEYS - {"motion_id", "notes"})
        samples.append(VisualVerification(
            motion_sha256=_hash(sample["motion_sha256"], "sample.motion_sha256"),
            frame_start=_number(sample["frame_start"], "sample.frame_start", int, 0, 10**9),
            frames=_number(sample["frames"], "sample.frames", int, 1, 10**9),
            fps=_number(sample["fps"], "sample.fps", float, 1e-9, 100000),
            views=_strings(sample["views"], "sample.views"),
            display_context=_text(sample["display_context"], "sample.display_context"),
            evidence=_text(sample["evidence"], "sample.evidence"),
            motion_id=_text(sample["motion_id"], "sample.motion_id") if "motion_id" in sample else None,
            notes=_text(sample["notes"], "sample.notes") if "notes" in sample else None,
        ))
    mesh_cloth = None
    if "mesh_cloth" in data:
        mesh = _object(data["mesh_cloth"], "mesh_cloth", {f.name for f in fields(MeshClothSettings)},
                       {"bone_prefixes", "materials"})
        mesh_cloth = MeshClothSettings(**mesh)
    return PhysicsProfile(
        profile_id=_text(data["profile_id"], "profile_id"),
        asset_sha256=_hash(data["asset_sha256"], "asset_sha256"),
        import_scale=_number(data["import_scale"], "import_scale", float, 1e-9, 1000),
        bullet=tuple(sorted(settings.items())), visual_verified=tuple(samples),
        notes=_text(data["notes"], "notes") if "notes" in data else None,
        mesh_cloth=mesh_cloth,
        joint_rules=joint_rules_from_json(data.get("joint_rules", [])),
        side_link_rules=side_link_rules_from_json(data.get("side_link_rules", [])),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate physics profile key: {key}")
        result[key] = value
    return result


def load_physics_profile(path: str | Path) -> PhysicsProfile:
    """Load a strict JSON profile. Unknown settings are errors, never ignored."""
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    return physics_profile_from_json(data)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResolvedPhysicsProfile:
    profile: PhysicsProfile
    asset_path: Path

    @property
    def bullet_overrides(self) -> dict:
        return self.profile.bullet_overrides

    @property
    def mesh_cloth_settings(self) -> dict | None:
        return self.profile.mesh_cloth_settings

    @property
    def joint_rules(self) -> tuple[JointRule, ...]:
        return self.profile.joint_rules

    @property
    def side_link_rules(self) -> tuple[SideLinkRule, ...]:
        return self.profile.side_link_rules

    def report(self) -> dict:
        return {
            **self.profile.to_json(), "profile_fingerprint": self.profile.fingerprint,
            "matched_asset": str(self.asset_path), "asset_match": True,
            "verification_scope": "Listed renders only; asset matching does not verify another motion, "
            "pose-processing configuration, camera or unlisted physics settings.",
        }


def resolve_physics_profile(
    profile: PhysicsProfile | str | Path, *, asset_path: str | Path, import_scale: float,
) -> ResolvedPhysicsProfile:
    """Require the exact PMX bytes and import scale before exposing overrides."""
    if not isinstance(profile, PhysicsProfile):
        profile = load_physics_profile(profile)
    else:
        # Public constructors must not bypass the same schema gate as JSON files.
        profile = physics_profile_from_json(profile.to_json())
    actual_scale = _number(import_scale, "actual import_scale", float, 1e-9, 1000)
    if actual_scale != profile.import_scale:
        raise ValueError(f"Physics profile import_scale mismatch: {actual_scale} != {profile.import_scale}")
    asset = Path(asset_path).resolve()
    if file_sha256(asset) != profile.asset_sha256:
        raise ValueError(f"Physics profile asset SHA256 mismatch: {asset}")
    return ResolvedPhysicsProfile(profile=profile, asset_path=asset)
