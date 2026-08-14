"""Build the MMD channel plan from an imported armature.

This is the only MMD stage that reads Blender data.  It records rest rotations
along the real bone chain and hands a pure-NumPy plan to ``mmd_solve``; no
animation decision is made here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .mmd_solve import MmdChannel, MmdRetargetPlan, calibration_for
from .twist import TwistPair, quaternion_to_matrix


def _rest_rotation(world: Any, bone: Any) -> np.ndarray:
    """Orthonormal rest rotation of one bone in Blender world space.

    Blender's matrices are single precision, so the round trip is redone in
    float64: a 1e-7 orthonormality defect here is amplified once per bone when
    local bases are reconstructed down a chain.
    """
    matrix = world @ bone.matrix_local
    quaternion = np.asarray(matrix.to_quaternion(), dtype=np.float64)
    return quaternion_to_matrix(quaternion)


def _root_translation_scale(
    armature: Any,
    bones: Any,
    target_to_source: dict[str, str],
    source_rest: dict[str, np.ndarray],
    head_bone: str,
) -> float:
    """Ratio of skeleton heights.

    Local rotations transfer unchanged between rigs, but root translation is
    proportional to how tall the character is, so a 1.6 m SMPL-X walk cannot
    move a stylised rig by the same metres.
    """
    world = armature.matrix_world
    world_scale = float(sum(abs(value) for value in world.to_scale())) / 3.0

    target_z = [float((world @ bones[name].matrix_local).translation.z) for name in target_to_source]
    source_z = [
        float(np.asarray(source_rest[source], dtype=np.float64)[2, 3]) for source in target_to_source.values()
    ]
    if not target_z or not source_z:
        return 1.0
    if head_bone in bones:
        target_z.append(
            float((world @ bones[head_bone].matrix_local).translation.z)
            + world_scale * float(bones[head_bone].length)
        )
    target_height = max(target_z) - min(target_z)
    source_height = max(source_z) - min(source_z)
    if source_height <= 1e-6 or target_height <= 1e-6:
        return 1.0
    return target_height / source_height


def _depth(bone: Any) -> int:
    depth = 0
    parent = bone.parent
    while parent is not None:
        depth += 1
        parent = parent.parent
    return depth


def build_mmd_plan(
    armature: Any,
    *,
    smplx_map: dict[str, str],
    transfer_modes: dict[str, str],
    twist_pairs: tuple[TwistPair, ...],
    source_rest: dict[str, np.ndarray],
    source_names: tuple[str, ...],
) -> MmdRetargetPlan:
    """Record every channel between the armature root and each driven bone."""
    world = armature.matrix_world
    bones = armature.data.bones
    target_to_source = {target: source for source, target in smplx_map.items()}
    modes = dict(transfer_modes)
    twist_by_swing = {pair.swing_bone: pair.twist_bone for pair in twist_pairs}
    twist_bones = set(twist_by_swing.values())

    missing = [name for name in (*target_to_source, *twist_bones) if name not in bones]
    if missing:
        raise ValueError("MMD plan references absent bones: " + ", ".join(sorted(missing)))
    unknown_source = [name for name in target_to_source.values() if name not in source_rest]
    if unknown_source:
        raise ValueError("No source rest frame for: " + ", ".join(sorted(unknown_source)))

    # Ancestors are included so an undriven bone between two driven ones is an
    # explicit pass-through node instead of an assumed identity.
    channel_names: set[str] = set()
    for name in (*target_to_source, *twist_bones):
        bone = bones[name]
        while bone is not None:
            channel_names.add(bone.name)
            bone = bone.parent

    ordered = sorted(channel_names, key=lambda name: (_depth(bones[name]), name))
    index_of = {name: index for index, name in enumerate(ordered)}
    rest_rotations = {name: _rest_rotation(world, bones[name]) for name in ordered}

    channels: list[MmdChannel] = []
    for name in ordered:
        bone = bones[name]
        parent_name = bone.parent.name if bone.parent is not None else None
        parent_index = index_of[parent_name] if parent_name is not None else -1
        rest_global = rest_rotations[name]
        rest_local = rest_global if parent_name is None else rest_rotations[parent_name].T @ rest_global

        source = target_to_source.get(name, "")
        if name in twist_bones:
            swing_name = next(swing for swing, twist in twist_by_swing.items() if twist == name)
            channels.append(
                MmdChannel(
                    name=name,
                    parent=parent_index,
                    rest_local=rest_local,
                    mode="twist",
                    swing_of=index_of[swing_name],
                )
            )
            continue
        if not source:
            channels.append(
                MmdChannel(name=name, parent=parent_index, rest_local=rest_local, mode="passthrough")
            )
            continue

        mode = modes.get(name, "relative")
        source_frame = np.asarray(source_rest[source], dtype=np.float64)[:3, :3]
        channels.append(
            MmdChannel(
                name=name,
                parent=parent_index,
                rest_local=rest_local,
                mode=mode,  # type: ignore[arg-type]
                source=source,
                source_index=source_names.index(source),
                calibration=calibration_for(mode, source_frame, rest_global),  # type: ignore[arg-type]
                twist_partner=index_of.get(twist_by_swing.get(name, ""), -1),
            )
        )

    target_rest_global = np.stack([rest_rotations[name] for name in ordered])
    source_rest_global = np.stack(
        [
            np.asarray(source_rest[target_to_source[name]], dtype=np.float64)[:3, :3]
            if name in target_to_source
            else np.eye(3)
            for name in ordered
        ]
    )
    return MmdRetargetPlan(
        channels=tuple(channels),
        source_names=source_names,
        target_rest_global=target_rest_global,
        source_rest_global=source_rest_global,
        root_translation_scale=_root_translation_scale(
            armature, bones, target_to_source, source_rest, smplx_map.get("head", "")
        ),
    )
