"""A new prop must not enable any previously excluded pair of character bodies."""
from types import SimpleNamespace

from motionviewer.blender.mmd_prop_physics import add_animated_box_colliders


def test_box_reuses_dynamic_layers_without_changing_existing_masks():
    masks = [[i == layer for i in range(20)] for layer in (2, 5, 8)]
    bodies = [SimpleNamespace(rigid_body=SimpleNamespace(
        collision_collections=mask.copy(), kinematic=index == 2))
        for index, mask in enumerate(masks)]
    box = SimpleNamespace(name="box", type="MESH", dimensions=(.3192, .22, .15),
                          rigid_body=SimpleNamespace())
    report = add_animated_box_colliders(None, [box], SimpleNamespace(rigidBodies=lambda: bodies))
    assert box.rigid_body.kinematic
    assert box.rigid_body.collision_shape == "BOX"
    assert [i for i, on in enumerate(box.rigid_body.collision_collections) if on] == [2, 5]
    assert [body.rigid_body.collision_collections for body in bodies] == masks
    assert not any(a and b for a, b in zip(masks[0], masks[1], strict=True))
    assert not report["existing_collision_masks_modified"]
    assert report["enabled"] is True


def test_no_dynamic_collision_layers_leaves_box_unmodified():
    box = SimpleNamespace(rigid_body=None)
    report = add_animated_box_colliders(None, [box], SimpleNamespace(rigidBodies=lambda: []))
    assert box.rigid_body is None
    assert report["bodies"] == []
    assert report["enabled"] is False
