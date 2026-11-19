"""Animated, segmented collision proxies for the selected visible weapon."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from motionviewer.core.collision_proxy import surface_cells


def add_animated_weapon_colliders(bpy: Any, objects: Iterable[Any], model: Any) -> dict:
    """Add kinematic hulls after MMD build, before bake, without changing its masks."""
    from mathutils import Matrix

    weapons = list(objects)
    dynamic = [obj for obj in model.rigidBodies()
               if obj.rigid_body is not None and not obj.rigid_body.kinematic]
    layers = [any(body.rigid_body.collision_collections[i] for body in dynamic)
              for i in range(20)]
    report: dict = {"enabled": False, "bodies": [], "cell_size_m": 0.12,
                    "margin_m": 0.001, "existing_collision_masks_modified": False,
                    "dynamic_bodies": len(dynamic)}
    if not weapons or not any(layers):
        report["reason"] = "no weapons or no dynamic collision layers"
        return report
    for weapon in weapons:
        if weapon.type != "MESH":
            raise ValueError("an animated weapon collider requires a mesh")
        if any(abs(float(value) - 1.0) > 1e-5 for value in weapon.scale):
            raise ValueError("weapon collider expects applied physical scale")
        weapon.data.calc_loop_triangles()
        vertices = np.asarray([tuple(v.co) for v in weapon.data.vertices])
        faces = np.asarray([tuple(t.vertices) for t in weapon.data.loop_triangles], dtype=int)
        cells = surface_cells(vertices, faces, cell_size=0.12)
        if not cells:
            raise ValueError("weapon has no collision surface")
        for index, points in enumerate(cells):
            if len(points) < 3:
                continue
            center = points.mean(axis=0)
            local = points - center
            _, singular, axes = np.linalg.svd(local, full_matrices=False)
            thickened = len(singular) < 3 or singular[-1] < 1e-7
            if thickened:
                # A planar blade patch needs non-zero volume for Bullet's hull.
                normal = axes[-1] * 0.0001
                local = np.concatenate((local + normal, local - normal))
            mesh = bpy.data.meshes.new(f"weapon_collision_{index:03d}")
            mesh.from_pydata(local.tolist(), [], [])
            mesh.update()
            obj = bpy.data.objects.new(mesh.name, mesh)
            bpy.context.scene.collection.objects.link(obj)
            obj.parent = weapon
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_basis = Matrix.Translation(center.tolist())
            obj.hide_render = True
            obj.display_type = "WIRE"
            with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
                bpy.ops.rigidbody.object_add()
            body = obj.rigid_body
            body.type = "ACTIVE"
            body.kinematic = True
            body.collision_shape = "CONVEX_HULL"
            body.use_margin = True
            body.collision_margin = 0.001
            body.friction = 0.3
            body.restitution = 0.0
            body.collision_collections = layers
            report["bodies"].append({"name": obj.name, "source": weapon.name,
                                     "vertices": len(local), "planar_thickness_m": 0.0002 if thickened else 0.0,
                                     "local_bounds_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()]})
    report["enabled"] = bool(report["bodies"])
    return report
