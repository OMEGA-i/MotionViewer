import json
from pathlib import Path

from motionviewer.assets.fbx_catalog import list_fbx_models, valid_fbx_models, validate_fbx_path
from motionviewer.cli import _fbx_report


def _write_binary_fbx(path: Path) -> None:
    path.write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00" + b"\x00" * 32)


def _write_catalog(root: Path) -> None:
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "mixamo": {
                        "profile_id": "mixamo",
                        "rig_family": "mixamo",
                        "bone_map": "auto",
                    },
                },
                "assets": [
                    {
                        "model_id": "iron",
                        "path": "iron.fbx",
                        "profile_id": "mixamo",
                        "status": "approved",
                        "random_eligible": True,
                    },
                    {
                        "model_id": "erika_archer",
                        "path": "Erika Archer.fbx",
                        "profile_id": "mixamo",
                        "status": "approved",
                        "random_eligible": True,
                    },
                    {
                        "model_id": "ch43_nonpbr",
                        "path": "Ch43_nonPBR.fbx",
                        "profile_id": "mixamo",
                        "status": "quarantined",
                        "random_eligible": False,
                        "reason": "retarget smoke rendered invisible/out-of-camera",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_fbx_catalog_filters_binary_ascii_and_pmx(tmp_path: Path) -> None:
    root = tmp_path / "fbx"
    pmx = root / "pmx"
    pmx.mkdir(parents=True)
    _write_catalog(root)
    _write_binary_fbx(root / "iron.fbx")
    _write_binary_fbx(root / "Erika Archer.fbx")
    _write_binary_fbx(root / "Unapproved Character.fbx")
    (root / "Ascii.fbx").write_text("; FBX 7.1.0 project file\n", encoding="utf-8")
    _write_binary_fbx(pmx / "Ignored PMX.fbx")

    models = list_fbx_models(root)
    assert [model.model_id for model in models] == ["ascii", "erika_archer", "unapproved_character", "iron"]
    assert [model.model_id for model in valid_fbx_models(root)] == ["erika_archer", "iron"]
    invalid = next(model for model in models if model.model_id == "ascii")
    assert invalid.reason and "binary FBX" in invalid.reason
    pending = next(model for model in models if model.model_id == "unapproved_character")
    assert pending.reason and "not been retarget-approved" in pending.reason


def test_quarantined_binary_fbx_is_not_random_eligible(tmp_path: Path) -> None:
    root = tmp_path / "fbx"
    root.mkdir()
    _write_catalog(root)
    _write_binary_fbx(root / "Ch43_nonPBR.fbx")

    model = list_fbx_models(root)[0]

    assert model.model_id == "ch43_nonpbr"
    assert model.valid is False
    assert model.reason and "invisible" in model.reason


def test_validate_fbx_path_reports_missing_and_ascii(tmp_path: Path) -> None:
    assert validate_fbx_path(tmp_path / "missing.fbx")
    ascii_path = tmp_path / "ascii.fbx"
    ascii_path.write_text("; FBX ascii", encoding="utf-8")
    assert "binary FBX" in validate_fbx_path(ascii_path)[0]


def test_deep_report_proposes_demotion_when_an_approved_asset_regresses() -> None:
    class Model:
        model_id = "iron"
        status = "approved"
        valid = True

        def to_json(self) -> dict:
            return {"model_id": self.model_id, "status": self.status}

    report = _fbx_report(
        Path("assets/fbx"),
        [Model()],
        deep=True,
        quality_matrix=[
            {
                "assets": [
                    {
                        "asset": "assets/fbx/iron.fbx",
                        "quality_gate": {"passed": False},
                    }
                ]
            }
        ],
    )

    assert report["proposed_statuses"]["iron"] == "pending"
