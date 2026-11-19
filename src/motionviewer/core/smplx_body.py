"""The SMPL-X body mesh, posed in NumPy from a locally provided model file.

The comparison videos need the *original* motion beside the character — the human the
T2M model generated, before any retarget. A stick skeleton shows the joints but not the
body, so contacts and silhouettes cannot be compared. The official route is the
SMPL-X for Blender addon, which is a separate licensed download; this module needs only
the model ``.npz`` that ships with SMPL-X (``SMPLX_NEUTRAL_2020.npz``), which the user
obtains under its own licence and which is never committed here.

This is the standard SMPL-X forward pass, body joints only:

    v_shaped = v_template + shapedirs[:, :, :n] @ betas
    J        = J_regressor @ v_shaped
    v_posed  = v_shaped + posedirs @ (R_j - I) for j >= 1        (pose correctives)
    v        = sum_j w_j * G_j @ v_posed                          (linear blend skinning)

Hands, jaw and eyes are held at the rest pose (the clips are body-22), and expression is
zero. The check that this is right is not visual: the first 22 regressed joints must
reproduce the clip's own ``joints22``, which came from the official fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .smplx_fk import rodrigues

#: Body joints driven by ``global_orient`` (0) and ``body_pose`` (1..21).
BODY_JOINTS = 22


@dataclass(frozen=True)
class SmplxModel:
    v_template: np.ndarray  # (V, 3)
    shapedirs: np.ndarray  # (V, 3, S)
    posedirs: np.ndarray  # (V, 3, 9 * (J - 1))
    j_regressor: np.ndarray  # (J, V)
    weights: np.ndarray  # (V, J)
    parents: np.ndarray  # (J,), root = -1
    faces: np.ndarray  # (F, 3)

    @property
    def joint_count(self) -> int:
        return len(self.parents)


def load_smplx_model(path: str | Path) -> SmplxModel:
    data = np.load(path, allow_pickle=True)
    parents = np.asarray(data["kintree_table"][0], dtype=np.int64).copy()
    parents[0] = -1
    return SmplxModel(
        v_template=np.asarray(data["v_template"], dtype=np.float64),
        shapedirs=np.asarray(data["shapedirs"], dtype=np.float64),
        posedirs=np.asarray(data["posedirs"], dtype=np.float64),
        j_regressor=np.asarray(data["J_regressor"], dtype=np.float64),
        weights=np.asarray(data["weights"], dtype=np.float64),
        parents=parents,
        faces=np.asarray(data["f"], dtype=np.int64),
    )


def shaped(model: SmplxModel, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rest vertices and joints for one body shape. ``betas`` may be shorter than the model's."""
    betas = np.asarray(betas, dtype=np.float64).ravel()
    count = min(len(betas), model.shapedirs.shape[2])
    vertices = model.v_template + model.shapedirs[:, :, :count] @ betas[:count]
    joints = model.j_regressor @ vertices
    return vertices, joints


def pose_frames(
    model: SmplxModel,
    betas: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posed vertices (T, V, 3) and joints (T, J, 3), in the clip's own space (Y-up, metres)."""
    frames = len(global_orient)
    rest_vertices, rest_joints = shaped(model, betas)
    joint_count = model.joint_count
    local = np.tile(np.eye(3), (frames, joint_count, 1, 1))
    local[:, 0] = rodrigues(np.asarray(global_orient, dtype=np.float64))
    local[:, 1:BODY_JOINTS] = rodrigues(np.asarray(body_pose, dtype=np.float64).reshape(frames, 21, 3))

    parents = model.parents
    offsets = rest_joints.copy()
    offsets[1:] -= rest_joints[parents[1:]]
    world = np.zeros((frames, joint_count, 4, 4))
    world[:, :, 3, 3] = 1.0
    for joint in range(joint_count):
        transform = np.zeros((frames, 4, 4))
        transform[:, :3, :3] = local[:, joint]
        transform[:, :3, 3] = offsets[joint]
        transform[:, 3, 3] = 1.0
        world[:, joint] = transform if parents[joint] < 0 else world[:, parents[joint]] @ transform
    posed_joints = world[:, :, :3, 3].copy()

    # Skinning matrices: remove each joint's rest position so they act on rest vertices.
    skin = world.copy()
    skin[:, :, :3, 3] -= np.einsum("tjab,jb->tja", world[:, :, :3, :3], rest_joints)

    correctives = (local[:, 1:] - np.eye(3)).reshape(frames, -1)
    vertices = rest_vertices[None] + np.einsum("vcp,tp->tvc", model.posedirs, correctives)

    blended = np.einsum("vj,tjab->tvab", model.weights, skin)
    vertices = np.einsum("tvab,tvb->tva", blended[:, :, :3, :3], vertices) + blended[:, :, :3, 3]
    shift = np.asarray(transl, dtype=np.float64)[:, None, :]
    return vertices + shift, posed_joints + shift
