"""SMPL-X forward pass on a tiny synthetic model (the licensed model file is not needed)."""
import numpy as np

from motionviewer.core.smplx_body import SmplxModel, pose_frames, shaped


def _chain_model():
    # 22 joints in a straight chain along +y, one vertex per joint, rigidly skinned.
    joints = 22
    template = np.zeros((joints, 3))
    template[:, 1] = np.arange(joints) * 0.1
    parents = np.arange(-1, joints - 1)
    return SmplxModel(
        v_template=template,
        shapedirs=np.zeros((joints, 3, 2)),
        posedirs=np.zeros((joints, 3, 9 * (joints - 1))),
        j_regressor=np.eye(joints),
        weights=np.eye(joints),
        parents=parents,
        faces=np.array([[0, 1, 2]]),
    )


def test_rest_pose_reproduces_the_template_plus_translation():
    model = _chain_model()
    frames = 3
    transl = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    vertices, joints = pose_frames(model, np.zeros(2), np.zeros((frames, 3)), np.zeros((frames, 63)), transl)
    assert np.allclose(vertices, model.v_template[None] + transl[:, None])
    assert np.allclose(joints, vertices)


def test_root_rotation_rotates_the_whole_body_about_the_root():
    model = _chain_model()
    quarter_turn_z = np.array([[0.0, 0.0, np.pi / 2]])
    vertices, _ = pose_frames(model, np.zeros(2), quarter_turn_z, np.zeros((1, 63)), np.zeros((1, 3)))
    # +y points to -x after a quarter turn about z.
    assert np.allclose(vertices[0, :, 0], -model.v_template[:, 1], atol=1e-9)
    assert np.allclose(vertices[0, :, 1], 0.0, atol=1e-9)


def test_child_rotation_moves_only_descendants():
    model = _chain_model()
    body_pose = np.zeros((1, 63))
    body_pose[0, 3 * 4 : 3 * 4 + 3] = [0.0, 0.0, np.pi / 2]  # joint 5 (body_pose index 4)
    vertices, _ = pose_frames(model, np.zeros(2), np.zeros((1, 3)), body_pose, np.zeros((1, 3)))
    assert np.allclose(vertices[0, :6], model.v_template[:6])
    # Joint 6 is 0.1 above joint 5 at rest; after the turn it sits 0.1 to the -x side.
    assert np.allclose(vertices[0, 6] - vertices[0, 5], [-0.1, 0.0, 0.0], atol=1e-9)
    lengths = np.linalg.norm(np.diff(vertices[0], axis=0), axis=1)
    assert np.allclose(lengths, 0.1)


def test_betas_offset_the_rest_shape():
    model = _chain_model()
    shapedirs = np.zeros_like(model.shapedirs)
    shapedirs[:, 0, 0] = 1.0
    model = SmplxModel(**{**model.__dict__, "shapedirs": shapedirs})
    vertices, _ = shaped(model, np.array([0.05, 0.0, 9.0]))  # extra betas beyond the model are ignored
    assert np.allclose(vertices[:, 0], 0.05)
