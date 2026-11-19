"""Locations for the Unitree G1 closed-loop simulation scripts.

Robot descriptions, SONIC weights, and SOMA clips stay outside the repository.
Generated rollouts and FK bundles go under ``outputs/g1_sim/``, which is gitignored.
Override any location with the environment variables named below.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path | None:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "motionviewer").is_dir():
            return candidate
    return None


_ROOT = _repo_root()
REPO_ROOT = _ROOT if _ROOT is not None else Path(__file__).resolve().parent
_DEFAULT_WORK = (REPO_ROOT / "outputs" / "g1_sim") if _ROOT is not None else (REPO_ROOT / "work")
WORK = Path(os.environ.get("MOTIONVIEWER_G1_WORK", _DEFAULT_WORK))


def _first_existing(candidates: list[Path], fallback: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return fallback


def g1_dir() -> Path:
    """Unitree G1 MuJoCo description (``g1_29dof.xml`` and meshes)."""
    env = os.environ.get("MOTIONVIEWER_G1_DIR")
    if env:
        return Path(env)
    home = Path.home()
    return _first_existing(
        [
            home / "projects/GR00T-WholeBodyControl/gear_sonic_deploy/g1",
            home / "projects/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/g1",
        ],
        home / "projects/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/g1",
    )


def policy_dir() -> Path:
    """GEAR-SONIC ``release`` directory with the encoder and decoder ONNX files."""
    env = os.environ.get("MOTIONVIEWER_SONIC_POLICY")
    if env:
        return Path(env)
    legacy = REPO_ROOT / "outputs/video_demo_studio/g1_sim_work/sonic_deploy/policy/release"
    return _first_existing([WORK / "sonic_deploy/policy/release", legacy], WORK / "sonic_deploy/policy/release")


def soma_active_dir() -> Path:
    env = os.environ.get("MOTIONVIEWER_SOMA_DIR")
    if env:
        return Path(env)
    home = Path.home()
    return _first_existing(
        [
            home / "Motion data/t2m_demo400/soma/active",
            home / "projects/ws-OmegaZero/t2m_demo400/soma/active",
        ],
        home / "projects/ws-OmegaZero/t2m_demo400/soma/active",
    )


def soma_meta_path() -> Path:
    env = os.environ.get("MOTIONVIEWER_SOMA_META")
    if env:
        return Path(env)
    return soma_active_dir().parent.parent / "meta.json"


def soma_orient_path() -> Path:
    env = os.environ.get("MOTIONVIEWER_SOMA_ORIENT")
    if env:
        return Path(env)
    home = Path.home()
    return _first_existing(
        [
            home / "projects/ws-OmegaZero/code/OmegaMotionGPT/runs/_sonic_v1/soma_joint_orient.npz",
            home / "projects/OmegaMotionGPT/runs/_sonic_v1/soma_joint_orient.npz",
        ],
        home / "projects/ws-OmegaZero/code/OmegaMotionGPT/runs/_sonic_v1/soma_joint_orient.npz",
    )


def picks39_csv_dir() -> Path:
    env = os.environ.get("MOTIONVIEWER_PICKS39_CSV")
    if env:
        return Path(env)
    home = Path.home()
    return _first_existing(
        [
            home / "projects/ws-OmegaZero/code/OmegaMotionGPT/runs/_sonic_v1/picks39_g1csv",
            home / "projects/OmegaMotionGPT/runs/_sonic_v1/picks39_g1csv",
        ],
        home / "projects/ws-OmegaZero/code/OmegaMotionGPT/runs/_sonic_v1/picks39_g1csv",
    )


def ensure_work(*parts: str) -> Path:
    path = WORK.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
