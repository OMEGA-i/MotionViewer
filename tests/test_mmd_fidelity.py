"""Source rotations and metre-valued root deltas are the default MMD contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from motionviewer.blender.retarget import _animate
from motionviewer.blender.retarget.mmd import MmdPolishOptions
from motionviewer.blender.retarget.mmd_solve import MmdChannel, MmdRetargetPlan, polish_source_frames
from motionviewer.core.smplx_fk import SMPLX_BODY22_NAMES, rodrigues, source_to_blender


@pytest.mark.parametrize("values", [None, {}, {"root_motion": "source"}])
def test_default_polish_keeps_every_source_frame_exactly(values):
    rng = np.random.default_rng(34)
    frames = np.broadcast_to(np.eye(4), (9, 22, 4, 4)).copy()
    frames[..., :3, :3] = rodrigues(rng.normal(scale=.35, size=(9, 22, 3)))
    rest = np.repeat(np.eye(4)[None], 22, axis=0)
    before = frames.copy()
    options = MmdPolishOptions.from_mapping(values)
    result = polish_source_frames(frames, SMPLX_BODY22_NAMES, rest, options)
    assert options.enabled is False
    assert options.root_motion == "source"
    assert np.array_equal(result, before)
    assert np.array_equal(frames, before)
    # This input would expose an accidentally re-enabled artistic pass.
    changed = polish_source_frames(frames, SMPLX_BODY22_NAMES, rest, MmdPolishOptions(enabled=True))
    assert not np.array_equal(changed, before)


@pytest.mark.parametrize(("mode", "expected_scale"), [(None, 1.), ("source", 1.), ("legacy-scaled", .73)])
def test_animation_transfers_root_deltas_in_metres_unless_legacy_is_explicit(monkeypatch, mode, expected_scale):
    # Non-zero starting position and offsets must not change path length or jump height.
    source_root = np.array([[4., 1.1, -2.], [4.8, 1.3, -1.7], [5.1, 1.7, -1.5], [4.9, 1.1, -1.2]])
    roots_blender = source_to_blender(source_root)
    source_frames = np.broadcast_to(np.eye(4), (4, 1, 4, 4)).copy()
    source_frames[:, 0, :3, :3] = rodrigues(np.array([[0., 0., 0.], [.2, .1, 0.], [.1, -.3, .4], [-.1, 0., .2]]))
    source_frames[:, 0, :3, 3] = roots_blender
    before = source_frames.copy()
    plan = MmdRetargetPlan(
        channels=(MmdChannel("腰", -1, np.eye(3), "absolute", source="pelvis", source_index=0),),
        source_names=("pelvis",), target_rest_global=np.eye(3)[None], source_rest_global=np.eye(3)[None],
        root_translation_scale=.73,
    )
    armature = {}
    motion = SimpleNamespace(
        names=("pelvis",), rest_frames=np.eye(4)[None], posed_frames=source_frames,
        root_locations=roots_blender, rest_by_name=lambda: {"pelvis": np.eye(4)},
    )
    boot = SimpleNamespace(
        lookat_motion=motion, num_frames=4, fbx_armature=armature, fbx_meshes=[], twist_pairs=(),
        offset=np.array([2., -3., .25]), label="fidelity",
    )
    mapping = SimpleNamespace(smplx_to_fbx={"pelvis": "腰"}, transfer_modes={"腰": "absolute"}, twist_pairs=())
    bpy = SimpleNamespace(context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None)))
    written = []
    monkeypatch.setattr(_animate, "build_mmd_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(_animate, "_write_mmd_channels", lambda bpy, boot, plan, result: written.append(result))
    monkeypatch.setattr(_animate, "relax_hands", lambda *args, **kwargs: pytest.fail("default must not invent hand poses"))
    options = None if mode is None else MmdPolishOptions.from_mapping({"root_motion": mode})
    _animate.animate_mmd_frames(bpy, boot, mapping, vertical_offset_z=-.12, polish=options)
    result = written[0]
    expected_delta = source_to_blender(source_root - source_root[0]) * expected_scale
    assert result.root_locations - result.root_locations[0] == pytest.approx(expected_delta, abs=1e-12)
    assert np.diff(result.root_locations, axis=0) == pytest.approx(np.diff(roots_blender, axis=0) * expected_scale)
    assert result.root_locations[0] == pytest.approx([2., -3., .13])
    assert result.world_rotations == pytest.approx(source_frames[..., :3, :3], abs=1e-12)
    assert np.array_equal(source_frames, before)
    assert np.array_equal(roots_blender, source_to_blender(source_root))
    report = json.loads(armature["motionviewer_mmd_transfer"])
    assert report["root_motion_mode"] == (mode or "source")
    assert report["root_translation_scale"] == expected_scale
    assert report["legacy_root_translation_scale"] == .73
    assert report["polish"]["hand_relax"] == 0.
    assert report["polish"]["arm_abduction_degrees"] == 0.


def test_invalid_root_mode_is_rejected_in_the_python_api():
    with pytest.raises(ValueError, match="root_motion"):
        MmdPolishOptions.from_mapping({"root_motion": "automatic"})
