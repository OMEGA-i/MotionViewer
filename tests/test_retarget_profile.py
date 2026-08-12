"""Unit tests for retarget profile and catalog data model."""

from __future__ import annotations

import json
from pathlib import Path

from motionviewer.blender.retarget.profile import (
    RetargetAssetEntry,
    RetargetCatalog,
    RetargetProfile,
    load_retarget_catalog,
    model_id_from_path,
)


def _write_catalog(root: Path) -> None:
    root.mkdir()
    (root / "eligible.fbx").touch()
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "profiles": {"mixamo": {"rig_family": "mixamo"}},
                "assets": [
                    {"model_id": "pending", "path": "pending.fbx", "status": "pending"},
                    {
                        "model_id": "eligible",
                        "path": "eligible.fbx",
                        "status": "approved",
                        "random_eligible": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


class TestRetargetProfile:
    def test_from_json_roundtrip(self) -> None:
        data = {
            "profile_id": "mixamo_test",
            "rig_family": "mixamo",
            "bone_map": "auto",
            "retarget_mode": "quality",
            "validation_status": "active",
        }
        profile = RetargetProfile.from_json(data)
        assert profile.profile_id == "mixamo_test"
        assert profile.rig_family == "mixamo"
        assert profile.retarget_mode == "quality"

        out = profile.to_json()
        assert out["profile_id"] == "mixamo_test"
        assert out["retarget_mode"] == "quality"

    def test_from_json_defaults(self) -> None:
        profile = RetargetProfile.from_json({"profile_id": "minimal"})
        assert profile.profile_id == "minimal"
        assert profile.rig_family == "mixamo"
        assert profile.bone_map == "auto"
        assert profile.retarget_mode == "quality"
        assert profile.validation_status == "active"


class TestRetargetAssetEntry:
    def test_from_json_roundtrip(self) -> None:
        data = {
            "model_id": "test_model",
            "path": "test.fbx",
            "profile_id": "mixamo",
            "status": "approved",
            "random_eligible": True,
            "reason": None,
            "evidence": {"smoke_mp4": "outputs/test.mp4"},
        }
        entry = RetargetAssetEntry.from_json(data, root=Path("/fake/root"))
        assert entry.model_id == "test_model"
        assert entry.path == Path("/fake/root/test.fbx")
        assert entry.status == "approved"
        assert entry.random_eligible is True
        assert entry.reason is None
        assert entry.evidence == {"smoke_mp4": "outputs/test.mp4"}

        out = entry.to_json()
        assert out["model_id"] == "test_model"
        assert out["status"] == "approved"

    def test_from_json_defaults(self) -> None:
        entry = RetargetAssetEntry.from_json(
            {"model_id": "pending_model", "path": "p.fbx"},
            root=Path("/root"),
        )
        assert entry.status == "pending"
        assert entry.random_eligible is False
        assert entry.reason is None


class TestRetargetCatalog:
    def test_random_eligible_assets(self, tmp_path: Path) -> None:
        a_path = tmp_path / "a.fbx"
        a_path.write_text("")
        q_path = tmp_path / "q.fbx"
        q_path.write_text("")
        n_path = tmp_path / "n.fbx"
        n_path.write_text("")
        approved = RetargetAssetEntry(
            model_id="a",
            path=a_path,
            profile_id="p",
            status="approved",
            random_eligible=True,
        )
        quarantined = RetargetAssetEntry(
            model_id="q",
            path=q_path,
            profile_id="p",
            status="quarantined",
            random_eligible=False,
        )
        not_eligible = RetargetAssetEntry(
            model_id="n",
            path=n_path,
            profile_id="p",
            status="approved",
            random_eligible=False,
        )
        catalog = RetargetCatalog(
            root=tmp_path,
            profiles={},
            assets={"a": approved, "q": quarantined, "n": not_eligible},
        )
        eligible = catalog.random_eligible_assets()
        assert len(eligible) == 1
        assert eligible[0].model_id == "a"

    def test_asset_lookup(self, tmp_path: Path) -> None:
        p = tmp_path / "iron.fbx"
        p.write_text("")
        entry = RetargetAssetEntry(
            model_id="iron",
            path=p,
            profile_id="p",
            status="approved",
            random_eligible=True,
        )
        catalog = RetargetCatalog(root=tmp_path, profiles={}, assets={"iron": entry})
        assert catalog.asset("iron") is entry
        assert catalog.asset("nonexistent") is None

    def test_profile_lookup(self) -> None:
        profile = RetargetProfile(profile_id="mixamo", rig_family="mixamo")
        catalog = RetargetCatalog(
            root=Path("/"),
            profiles={"mixamo": profile},
            assets={},
        )
        assert catalog.profile("mixamo") is profile
        assert catalog.profile("nonexistent") is None


class TestLoadRetargetCatalog:
    def test_loads_catalog(self, tmp_path: Path) -> None:
        root = tmp_path / "fbx"
        _write_catalog(root)

        catalog = load_retarget_catalog(root)

        assert catalog.root == root
        assert set(catalog.profiles) == {"mixamo"}
        assert set(catalog.assets) == {"pending", "eligible"}
        pending = catalog.asset("pending")
        assert pending is not None
        assert pending.status == "pending"
        assert pending.random_eligible is False

    def test_missing_catalog_returns_empty(self) -> None:
        catalog = load_retarget_catalog(Path("/nonexistent/path"))
        assert len(catalog.profiles) == 0
        assert len(catalog.assets) == 0

    def test_custom_catalog_path(self, tmp_path: Path) -> None:
        catalog_json = tmp_path / "custom.json"
        catalog_json.write_text(
            json.dumps(
                {
                    "profiles": {"test_profile": {"rig_family": "mixamo"}},
                    "assets": [
                        {"model_id": "m", "path": "m.fbx", "status": "approved", "random_eligible": True}
                    ],
                }
            )
        )
        catalog = load_retarget_catalog(tmp_path, catalog_path=catalog_json)
        assert "test_profile" in catalog.profiles
        assert catalog.asset("m") is not None

    def test_random_eligible_from_catalog(self, tmp_path: Path) -> None:
        root = tmp_path / "fbx"
        _write_catalog(root)

        catalog = load_retarget_catalog(root)
        eligible = catalog.random_eligible_assets()
        model_ids = {e.model_id for e in eligible}
        assert model_ids == {"eligible"}


class TestModelIdFromPath:
    def test_simple_stem(self) -> None:
        assert model_id_from_path("/path/to/Iron.fbx") == "iron"

    def test_spaces_to_underscores(self) -> None:
        assert model_id_from_path("Erika Archer.fbx") == "erika_archer"

    def test_mixed_case(self) -> None:
        assert model_id_from_path("Paladin WProp J Nordstrom.fbx") == "paladin_wprop_j_nordstrom"
