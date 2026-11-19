"""Drive an MMD character straight from SOMA-77, skipping the SMPL-X fit.

The existing SMPL-X path is untouched. This is a second entry point that assembles the
same ``BootstrapOutput`` the MMD stages already consume, with a
:class:`SomaLookatMotion` in place of the SMPL-X one — which works because
``build_mmd_plan`` and ``solve_mmd_retarget`` are written against *source names and
frames*, not against SMPL-X specifically.

What changes for the render: nothing about the shading, camera or expression, and on the
body nothing about the solve either. What is added is 30 driven finger joints per clip,
and what is removed is the 26-28 mm RMSE of fitting SMPL-X to the SOMA data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...core.soma_fk import SOMA77_NAMES, SOMA77_PRIMARY_CHILD, build_soma_motion, check_layout
from ._animate import animate_mmd_frames
from ._bootstrap import BootstrapOutput
from .mmd import MmdPolishOptions, mute_mmd_ik
from .pipeline import _mmd_ground_offset
from .soma_mmd import build_soma_mmd_mapping

REQUIRED_SOMA_KEYS: tuple[str, ...] = (
    "joints77",
    "rotmat77",
    "transl",
    "bone_lengths",
    "joint_names",
    "joint_parents",
)
SOMA_POSE_CONVENTION = "relative_to_soma_joint_orient"


def load_soma(path: str | Path) -> dict:
    """Read one ``*.soma.npz`` and check it carries what the retarget needs."""
    with np.load(Path(path), allow_pickle=False) as data:
        missing = [key for key in REQUIRED_SOMA_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"SOMA clip is missing {', '.join(missing)}")
        payload = {key: np.asarray(data[key]) for key in REQUIRED_SOMA_KEYS}
        payload["fps"] = float(data["fps"]) if "fps" in data.files else 30.0
        payload["caption"] = str(data["caption"]) if "caption" in data.files else ""
        if "pose_convention" in data.files:
            convention = np.asarray(data["pose_convention"])
            if convention.ndim != 0 or str(convention.item()) != SOMA_POSE_CONVENTION:
                raise ValueError(f"SOMA pose_convention must be {SOMA_POSE_CONVENTION!r}")
            payload["pose_convention"] = SOMA_POSE_CONVENTION

    joints = payload["joints77"]
    joint_count = len(SOMA77_NAMES)
    if joints.ndim != 3 or joints.shape[1:] != (joint_count, 3) or not len(joints):
        raise ValueError("SOMA joints77 must have shape (frames > 0, 77, 3)")
    shapes = {
        "rotmat77": (len(joints), joint_count, 3, 3),
        "transl": (len(joints), 3),
        "bone_lengths": (joint_count,),
        "joint_names": (joint_count,),
        "joint_parents": (joint_count,),
    }
    for key, shape in shapes.items():
        if payload[key].shape != shape:
            raise ValueError(f"SOMA {key} must have shape {shape}, got {payload[key].shape}")
    for key in ("joints77", "rotmat77", "transl", "bone_lengths", "joint_parents"):
        values = payload[key]
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"SOMA {key} must contain finite numbers")
    parents = payload["joint_parents"]
    if not np.equal(parents, np.floor(parents)).all():
        raise ValueError("SOMA joint_parents must contain integer indices")
    check_layout(tuple(str(name) for name in payload["joint_names"]), parents)
    if (payload["bone_lengths"] < 0).any():
        raise ValueError("SOMA bone_lengths must be non-negative")
    if not np.isfinite(payload["fps"]) or payload["fps"] <= 0:
        raise ValueError("SOMA fps must be finite and positive")
    return payload


def create_soma_actor_from_npz(
    path: str | Path,
    *,
    label: str,
    pmx_path: str | Path,
    pmx_scale: float = 0.08,
    frame_start: int = 1,
    layout_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drive_fingers: bool = True,
    polish: dict | None = None,
    physics: bool = False,
    morphs: bool = False,
) -> Any:
    """Import the character and retarget one SOMA clip onto it."""
    import bpy  # type: ignore

    from ..mmd_import import import_pmx_character

    payload = load_soma(path)
    motion = build_soma_motion(payload)

    scale = float(pmx_scale) if pmx_scale not in {0.0, 1.0} else 0.08
    armature, meshes = import_pmx_character(
        bpy, Path(pmx_path), label=label, scale=scale, physics=physics, morphs=morphs
    )
    mute_mmd_ik(armature)

    mapping = build_soma_mmd_mapping(armature, drive_fingers=drive_fingers)
    if mapping.errors:
        raise ValueError("SOMA/MMD mapping failed: " + "; ".join(mapping.errors))
    armature["motionviewer_soma_map"] = json.dumps(mapping.smplx_to_fbx, ensure_ascii=False, sort_keys=True)
    armature["motionviewer_soma_fingers"] = mapping.driven_fingers
    armature["motionviewer_soma_hand_aim"] = json.dumps(
        {name: SOMA77_PRIMARY_CHILD[name] for name in ("LeftHand", "RightHand")}, sort_keys=True,
    )
    print(
        f"soma: {len(mapping.smplx_to_fbx)} driven joints "
        f"({mapping.driven_fingers} finger bones), {len(motion.posed_frames)} frames"
    )

    frames = len(motion.posed_frames)
    boot = BootstrapOutput(
        smplx_armature=None,
        fbx_armature=armature,
        fbx_meshes=meshes,
        num_frames=frames,
        # The SMPL-X pose channels are not part of this path; the MMD stages read the
        # look-at motion instead. Zeros keep the dataclass honest about that.
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        body_pose=np.zeros((frames, 21, 3), dtype=np.float32),
        transl=np.asarray(payload["transl"], dtype=np.float32),
        offset=np.asarray(layout_offset, dtype=np.float32),
        calibration=None,
        rig_profile=None,
        unit_scale=1.0,
        frame_start=frame_start,
        label=label,
        rig_family="mmd",
        lookat_motion=motion,  # type: ignore[arg-type]
        twist_pairs=mapping.twist_pairs,
    )
    # ``hand_relax`` poses a static resting curl, which exists only because body-22 had
    # no finger data. With the fingers driven it would be applied on top of them and
    # fight the motion, so it is forced off whenever this path is actually driving them.
    options = dict(polish or {})
    if mapping.driven_fingers:
        options["hand_relax"] = 0.0
    return animate_mmd_frames(
        bpy,
        boot,
        mapping,  # type: ignore[arg-type]
        vertical_offset_z=_mmd_ground_offset(meshes),
        polish=MmdPolishOptions.from_mapping(options),
    )
