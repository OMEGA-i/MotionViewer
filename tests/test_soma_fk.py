"""SOMA-77 kinematics: the three facts the retarget stands on.

Each of these was established by measuring the probe set, and each is easy to break
silently. In particular the first one has a plausible wrong answer — treating the
stored rotations as world rotations still produces a skeleton, just one whose limbs are
11-19 degrees off, which is the kind of error that looks like "the retarget is a bit
loose" rather than like a bug.

The tests run on synthetic data so they need no fixture files; a companion check
against the real probe clips lives in ``scripts/_validate_soma_retarget.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from motionviewer.core.smplx_fk import rodrigues  # noqa: E402
from motionviewer.core.soma_fk import (  # noqa: E402
    SOMA77_NAMES,
    build_soma_motion,
    carry_rest_frames,
    check_layout,
    forward_kinematics,
    recover_rest_directions,
    rest_joints_from_directions,
    rest_lookat_frames,
    world_rotations,
)

# A short chain is enough to exercise every rule: root, two spine joints, a branch.
PARENTS = np.array([-1, 0, 1, 1], dtype=int)
DIRECTIONS = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
LENGTHS = np.array([0.0, 0.4, 0.3, 0.25], dtype=np.float64)


def _synthetic(frames: int = 12, seed: int = 0):
    """A rigid chain animated by known local rotations, plus the positions it implies."""
    rng = np.random.default_rng(seed)
    local = rodrigues(rng.normal(scale=0.4, size=(frames, len(PARENTS), 3)))
    world = world_rotations(local, PARENTS)
    rest = rest_joints_from_directions(DIRECTIONS, LENGTHS, PARENTS)
    root = np.cumsum(rng.normal(scale=0.01, size=(frames, 3)), axis=0)
    joints = forward_kinematics(rest, world, PARENTS, root)
    return local, world, rest, joints


def test_world_rotations_are_the_chained_product():
    local, world, _, _ = _synthetic()
    assert world[:, 0] == pytest.approx(local[:, 0])
    for child, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        expected = np.einsum("tij,tjk->tik", world[:, parent], local[:, child])
        assert world[:, child] == pytest.approx(expected, abs=1e-12)


def test_chaining_is_not_the_same_as_using_the_stored_rotation():
    """The wrong reading of this format; if it ever coincides, the test is vacuous."""
    local, world, _, _ = _synthetic()
    assert not np.allclose(world[:, 3], local[:, 3], atol=1e-6)


def test_rest_directions_are_recovered_from_positions_and_rotations():
    _, world, _, joints = _synthetic()
    recovered = recover_rest_directions(joints, world, PARENTS)
    for child, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        assert recovered[child] == pytest.approx(DIRECTIONS[child], abs=1e-9)


def test_recovered_directions_are_unit_length():
    _, world, _, joints = _synthetic()
    recovered = recover_rest_directions(joints, world, PARENTS)
    lengths = np.linalg.norm(recovered, axis=1)
    assert lengths[1:] == pytest.approx(np.ones(len(PARENTS) - 1), abs=1e-9)


def test_rigid_forward_kinematics_closes_on_the_positions():
    """The claim the README doubts: chained, the skeleton is rigid and FK is exact."""
    _, world, rest, joints = _synthetic()
    recovered = recover_rest_directions(joints, world, PARENTS)
    rebuilt = rest_joints_from_directions(recovered, LENGTHS, PARENTS)
    again = forward_kinematics(rebuilt, world, PARENTS, joints[:, 0])
    assert np.abs(again - joints).max() < 1e-9


def test_bone_lengths_come_from_the_lengths_argument():
    """Rest *directions* are solved for; rest *lengths* are given, never inferred."""
    rest = rest_joints_from_directions(DIRECTIONS, LENGTHS * 2.0, PARENTS)
    assert np.linalg.norm(rest[1] - rest[0]) == pytest.approx(LENGTHS[1] * 2.0)


def test_posed_frames_satisfy_the_solver_invariant():
    """``S(t, b) = R_g[b] @ S_rest(b)`` is what lets the MMD solver accept this source."""
    from motionviewer.core.smplx_fk import blender_rotation, source_to_blender

    _, world, rest, joints = _synthetic()
    rest_blender = source_to_blender(rest)
    frames = rest_lookat_frames(rest_blender, PARENTS, tuple(f"j{i}" for i in range(len(PARENTS))))
    posed = carry_rest_frames(frames, source_to_blender(joints), world)
    for frame in range(len(joints)):
        for joint in range(len(PARENTS)):
            expected = blender_rotation(world[frame, joint]) @ frames[joint, :3, :3]
            assert posed[frame, joint, :3, :3] == pytest.approx(expected, abs=1e-12)


def test_rest_frames_are_orthonormal_rotations():
    _, _, rest, joints = _synthetic()
    from motionviewer.core.smplx_fk import source_to_blender

    frames = rest_lookat_frames(source_to_blender(rest), PARENTS, tuple(f"j{i}" for i in range(len(PARENTS))))
    for joint in range(len(PARENTS)):
        block = frames[joint, :3, :3]
        assert block.T @ block == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(block) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(("side", "sign"), [("Left", 1.), ("Right", -1.)])
def test_wrist_axis_uses_the_middle_knuckle_not_the_offset_metacarpal_base(side, sign):
    """A palm's longitudinal axis is wrist→MCP, not wrist→metacarpal base.

    This geometry is independent of the retarget solver. The short, offset base
    would tilt the old wrist frame by about 18 degrees despite an otherwise flat
    hand. Finger motion must also not redefine the source wrist's own rotation.
    """
    from motionviewer.core.smplx_fk import blender_rotation, source_to_blender

    names = (side + "Hand", side + "HandMiddle1", side + "HandMiddle2", side + "HandMiddle3")
    parents = np.array([-1, 0, 1, 2])
    rest = np.array([[0., 0., 0.], [sign * .03, .002, .01], [sign * .09, 0., 0.], [sign * .13, 0., 0.]])
    rest_blender = source_to_blender(rest)
    frames = rest_lookat_frames(rest_blender, parents, names)
    palm_axis = rest_blender[2] - rest_blender[0]
    palm_axis /= np.linalg.norm(palm_axis)
    assert frames[0, :3, 1] == pytest.approx(palm_axis, abs=1e-12)
    base_axis = rest_blender[1] - rest_blender[0]
    base_axis /= np.linalg.norm(base_axis)
    assert np.degrees(np.arccos(np.clip(base_axis @ palm_axis, -1., 1.))) > 18.

    local = np.broadcast_to(np.eye(3), (3, 4, 3, 3)).copy()
    local[1, 1] = rodrigues(np.array([.2, -.3, .1]))  # only the metacarpal moves
    local[2, 0] = rodrigues(np.array([.4, .2, -.3]))  # source wrist now rotates
    world = world_rotations(local, parents)
    joints = source_to_blender(forward_kinematics(rest, world, parents, np.zeros((3, 3))))
    posed = carry_rest_frames(frames, joints, world)
    assert posed[1, 0, :3, :3] == pytest.approx(posed[0, 0, :3, :3], abs=1e-12)
    assert not np.allclose(joints[1, 2] - joints[1, 0], joints[0, 2] - joints[0, 0])
    for frame in range(3):
        expected = blender_rotation(world[frame, 0]) @ palm_axis
        assert posed[frame, 0, :3, 1] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# layout guard
# ---------------------------------------------------------------------------


def test_the_expected_layout_is_accepted():
    parents = np.full(77, -1, dtype=int)
    for child in range(1, 77):
        parents[child] = child - 1
    check_layout(SOMA77_NAMES, parents)


def test_a_renamed_joint_is_rejected():
    parents = np.arange(-1, 76, dtype=int)
    names = ("Pelvis",) + SOMA77_NAMES[1:]
    with pytest.raises(ValueError, match="joint_names"):
        check_layout(names, parents)


def test_a_forward_reference_in_the_tree_is_rejected():
    """Every parent must precede its child, or chaining would read a stale rotation."""
    parents = np.arange(-1, 76, dtype=int)
    parents[5] = 40
    with pytest.raises(ValueError, match="earlier joint"):
        check_layout(SOMA77_NAMES, parents)


def test_a_non_root_first_joint_is_rejected():
    parents = np.arange(-1, 76, dtype=int)
    parents[0] = 3
    with pytest.raises(ValueError, match="root"):
        check_layout(SOMA77_NAMES, parents)


def test_build_soma_motion_rejects_a_foreign_layout():
    payload = {
        "joints77": np.zeros((4, 3, 3)),
        "rotmat77": np.tile(np.eye(3), (4, 3, 1, 1)),
        "transl": np.zeros((4, 3)),
        "bone_lengths": np.zeros(3),
        "joint_names": np.array(["a", "b", "c"]),
        "joint_parents": np.array([-1, 0, 1]),
    }
    with pytest.raises(ValueError, match="77-joint layout"):
        build_soma_motion(payload)
