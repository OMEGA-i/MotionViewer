"""Precompute Mixamo rest-pose delta transfer data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import _JOINT_LIMIT_DEGREES, extract_mixamo_sole_anchors
from .solver import FootContactProfile, RetargetDefinition


class BoneLink:
    """One SMPL-X to Mixamo relation, evaluated in parent dependency order."""

    __slots__ = ("src_name", "trg_name", "a_matrix", "parent")

    def __init__(self) -> None:
        self.src_name: str = ""
        self.trg_name: str = ""
        self.a_matrix: Any = None
        self.parent: BoneLink | None = None


@dataclass(frozen=True)
class RetargetContext:
    ordered_links: list[BoneLink] = field(default_factory=list)
    root_translation_scale: float = 1.0
    solver_definition: RetargetDefinition | None = None


def _find_mapped_parent(
    fbx_armature: Any, fbx_bone: Any, links_by_target: dict[str, BoneLink]
) -> BoneLink | None:
    parent = fbx_bone.parent
    while parent is not None:
        parent_link = links_by_target.get(parent.name)
        if parent_link is not None:
            return parent_link
        parent = parent.parent
    return None


def precompute_retarget_context(
    smplx_armature: Any,
    fbx_armature: Any,
    mapping: dict[str, str],
    *,
    bpy: Any,
    fbx_meshes: list[Any] | None = None,
) -> RetargetContext:
    """Build the global-rest-delta relation for a validated Mixamo rig.

    For every bone this stores ``inverse(source_rest) @ target_rest``.  The
    animation loop applies it to the posed source world rotation, then removes
    the already reconstructed target parent world transform to obtain a local
    quaternion.  This avoids importing any FBX Euler curves or rotation order.
    """
    bpy.context.view_layer.update()
    from mathutils import Matrix  # type: ignore

    source_pose_bases = {
        pose_bone.name: pose_bone.matrix_basis.copy() for pose_bone in smplx_armature.pose.bones
    }
    target_pose_bases = {
        pose_bone.name: pose_bone.matrix_basis.copy() for pose_bone in fbx_armature.pose.bones
    }
    for pose_bone in smplx_armature.pose.bones:
        pose_bone.matrix_basis = Matrix.Identity(4)
    for pose_bone in fbx_armature.pose.bones:
        pose_bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    links: list[BoneLink] = []
    links_by_target: dict[str, BoneLink] = {}
    smplx_world = smplx_armature.matrix_world
    fbx_world = fbx_armature.matrix_world

    def rigid(matrix: Any) -> Any:
        result = matrix.to_quaternion().to_matrix().to_4x4()
        result.translation = matrix.translation
        return result

    source_rest_by_name: dict[str, Any] = {}
    target_rest_by_name: dict[str, Any] = {}
    target_channel_rest_by_name: dict[str, Any] = {}

    for source_name, target_name in mapping.items():
        source_pb = smplx_armature.pose.bones.get(source_name)
        target_pb = fbx_armature.pose.bones.get(target_name)
        if source_pb is None or target_pb is None:
            raise ValueError(f"Cannot precompute Mixamo link {source_name!r} -> {target_name!r}")
        source_rest = rigid(smplx_world @ source_pb.matrix)
        target_rest = rigid(fbx_world @ target_pb.matrix)
        source_rest_by_name[source_name] = source_rest
        target_rest_by_name[target_name] = target_rest
        target_channel_rest_by_name[target_name] = rigid(
            fbx_world @ fbx_armature.data.bones[target_name].matrix_local
        )
        link = BoneLink()
        link.src_name = source_name
        link.trg_name = target_name
        link.a_matrix = source_rest.inverted() @ target_rest
        links.append(link)
        links_by_target[target_name] = link

    for link in links:
        link.parent = _find_mapped_parent(
            fbx_armature, fbx_armature.data.bones[link.trg_name], links_by_target
        )

    ordered: list[BoneLink] = []
    visited: set[str] = set()

    def visit(link: BoneLink) -> None:
        if link.trg_name in visited:
            return
        if link.parent is not None:
            visit(link.parent)
        visited.add(link.trg_name)
        ordered.append(link)

    for link in links:
        visit(link)
    for pose_bone in fbx_armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"

    # Translation is proportional to avatar height, unlike local rotations.
    # The pose matrices are still rest matrices here, before the source motion
    # has been sampled by ground/contact planning.
    def skeleton_height(rest_by_name: dict[str, Any], armature: Any, head_name: str) -> float:
        positions = [float(matrix.translation.z) for matrix in rest_by_name.values()]
        world_scale = sum(float(value) for value in armature.matrix_world.to_scale()) / 3.0
        head_top = float(rest_by_name[head_name].translation.z) + world_scale * float(
            armature.data.bones[head_name].length
        )
        return max((*positions, head_top)) - min(positions)

    root_scale = 1.0
    target_height = 1.0
    source_head_name = "head"
    target_head_name = mapping.get("head", "")
    if source_head_name in source_rest_by_name and target_head_name in target_rest_by_name:
        source_height = skeleton_height(source_rest_by_name, smplx_armature, source_head_name)
        target_height = skeleton_height(target_rest_by_name, fbx_armature, target_head_name)
        if source_height > 1e-6 and target_height > 1e-6:
            root_scale = target_height / source_height
    index_by_target = {link.trg_name: index for index, link in enumerate(ordered)}
    sole_anchors = extract_mixamo_sole_anchors(fbx_armature, fbx_meshes or [])

    def as_array(matrix: Any) -> np.ndarray:
        return np.asarray([[float(value) for value in row] for row in matrix], dtype=np.float64)

    definition = RetargetDefinition(
        source_names=tuple(link.src_name for link in ordered),
        target_names=tuple(link.trg_name for link in ordered),
        parent_indices=np.asarray(
            [-1 if link.parent is None else index_by_target[link.parent.trg_name] for link in ordered],
            dtype=np.int32,
        ),
        rest_delta=np.stack([as_array(link.a_matrix) for link in ordered]),
        target_rest_local=np.stack(
            [
                as_array(target_channel_rest_by_name[link.trg_name])
                if link.parent is None
                else as_array(
                    target_channel_rest_by_name[link.parent.trg_name].inverted()
                    @ target_channel_rest_by_name[link.trg_name]
                )
                for link in ordered
            ]
        ),
        source_rest_global=np.stack(
            [as_array(source_rest_by_name[link.src_name])[:3, :3] for link in ordered]
        ),
        target_rest_global=np.stack(
            [as_array(target_rest_by_name[link.trg_name])[:3, :3] for link in ordered]
        ),
        source_to_target_scale=root_scale,
        target_height_m=target_height,
        joint_limit_degrees={
            index_by_target[link.trg_name]: float(
                max(
                    _JOINT_LIMIT_DEGREES[
                        next(name for name in _JOINT_LIMIT_DEGREES if link.trg_name.endswith(name))
                    ]
                )
            )
            for link in ordered
        },
        foot_profiles={
            name: FootContactProfile(
                bone_index=index_by_target[data["bone"]],
                anchors_local=np.asarray(data["points_local_m"], dtype=np.float64),
                forward_local=np.asarray(data["sole_forward_local"], dtype=np.float64),
                normal_local=np.asarray(data["sole_normal_local"], dtype=np.float64),
                lateral_local=np.asarray(data["sole_lateral_local"], dtype=np.float64),
                planar_residual=float(data.get("planar_residual", 0.0)),
            )
            for name, data in sole_anchors.items()
            if data["bone"] in index_by_target
        },
        contact_bone_indices={
            name: index_by_target[data["bone"]]
            for name, data in sole_anchors.items()
            if data["bone"] in index_by_target
        },
        contact_points_local={
            name: np.asarray(data["points_local_m"], dtype=np.float64)
            for name, data in sole_anchors.items()
            if data["bone"] in index_by_target
        },
    )
    result = RetargetContext(
        ordered_links=ordered,
        root_translation_scale=root_scale,
        solver_definition=definition,
    )
    for pose_bone in smplx_armature.pose.bones:
        if pose_bone.name in source_pose_bases:
            pose_bone.matrix_basis = source_pose_bases[pose_bone.name]
    for pose_bone in fbx_armature.pose.bones:
        if pose_bone.name in target_pose_bases:
            pose_bone.matrix_basis = target_pose_bases[pose_bone.name]
    bpy.context.view_layer.update()
    return result
