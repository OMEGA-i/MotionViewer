"""Kinematic collision boxes for animated props, added after the MMD rig builds.

The prop alone reuses the union of the model's existing dynamic collision layers.
No imported body's mask is changed, so its existing pair exclusions stay intact.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def add_animated_box_colliders(bpy: Any, objects: Iterable[Any], model: Any) -> dict:
    """Make the supplied visible boxes collide with the model's dynamic bodies.

    Call after ``Model.build`` and before point-cache baking. Animation and geometry
    are retained; Bullet treats these bodies as kinematic and cannot move the box or
    an already kinematic character body. The caller must pass rigid rectangular box
    objects, not arbitrary scene meshes. Model collision masks are never modified.
    """
    boxes = list(objects)
    dynamic = [obj for obj in model.rigidBodies()
               if obj.rigid_body is not None and not obj.rigid_body.kinematic]
    layers = [any(body.rigid_body.collision_collections[i] for body in dynamic)
              for i in range(20)]
    report: dict = {"enabled": False, "bodies": [],
                    "existing_collision_masks_modified": False,
                    "dynamic_bodies": len(dynamic)}
    if not boxes or not any(layers):
        report["reason"] = "no boxes or no dynamic collision layers"
        return report
    for obj in boxes:
        if obj.type != "MESH":
            raise ValueError("an animated box collider must be a mesh object")
        if obj.rigid_body is None:
            with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
                bpy.ops.rigidbody.object_add()
        body = obj.rigid_body
        body.type = "ACTIVE"
        body.kinematic = True
        body.collision_shape = "BOX"
        body.use_margin = True
        body.collision_margin = 0.001
        body.friction = 0.6
        body.restitution = 0.0
        body.collision_collections = layers
        report["bodies"].append({"name": obj.name, "dimensions_m": list(obj.dimensions),
                                 "collision_layers": [i for i, value in enumerate(layers) if value],
                                 "margin_m": body.collision_margin})
    report["enabled"] = bool(report["bodies"])
    return report
