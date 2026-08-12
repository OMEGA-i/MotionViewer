from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from motionviewer.blender.executable import BLENDER_ENV, resolve_blender
from motionviewer.cli import app

runner = CliRunner()


def test_cli_exposes_domain_groups() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for name in ("motion", "package", "render", "fbx", "doctor"):
        assert name in result.stdout


def test_motion_inspect_emits_json() -> None:
    sample = Path("data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz")
    result = runner.invoke(app, ["motion", "inspect", str(sample), "--json"])

    assert result.exit_code == 0
    assert '"format_id": "smplx_body22_fitted_aa"' in result.stdout


def test_resolve_blender_prefers_explicit_path(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "blender"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.delenv(BLENDER_ENV, raising=False)

    assert resolve_blender(executable) == executable.resolve()
