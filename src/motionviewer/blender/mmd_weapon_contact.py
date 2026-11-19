"""Asset-scoped grip calibration with evaluated-surface diagnostics.

This is deliberately not a universal hand IK solver. A calibrated profile preserves
the imported geometry and follows the wrist rigidly; unknown geometry is rejected.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


def _meshes(bpy: Any, armature: Any) -> list[Any]:
    return [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and not obj.name.endswith("_outline")
        and any(mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers)
    ]


def geometry_signature(obj: Any) -> str:
    """Hash imported local vertices and topology, independently of animation/labels."""
    digest = hashlib.sha256()
    digest.update(np.asarray([tuple(v.co) for v in obj.data.vertices], dtype="<f4").tobytes())
    for polygon in obj.data.polygons:
        digest.update(np.asarray([len(polygon.vertices), *polygon.vertices], dtype="<u4").tobytes())
    return digest.hexdigest()


def hand_signature(bpy: Any, armature: Any) -> str:
    """Include skin weights and rest bones, so changing the rig invalidates calibration."""
    digest = hashlib.sha256()
    meshes = _meshes(bpy, armature)
    for signature, obj in sorted(
        ((geometry_signature(obj), obj) for obj in meshes), key=lambda pair: pair[0]
    ):
        digest.update(signature.encode("ascii"))
        groups = {g.index: g.name for g in obj.vertex_groups}
        for vertex in obj.data.vertices:
            for group in sorted(vertex.groups, key=lambda item: groups[item.group]):
                name = groups[group.group]
                if name.startswith("右") and any(term in name for term in ("指", "手首")):
                    digest.update(name.encode("utf8"))
                    digest.update(np.asarray([vertex.index], dtype="<u4").tobytes())
                    digest.update(np.asarray([group.weight], dtype="<f4").tobytes())
    for bone in sorted(armature.data.bones, key=lambda bone: bone.name):
        if bone.name.startswith("右") and any(term in bone.name for term in ("指", "手首")):
            digest.update(bone.name.encode("utf8"))
            digest.update(np.asarray(bone.matrix_local, dtype="<f4").tobytes())
    return digest.hexdigest()


def _profile(bpy: Any, armature: Any, weapon: Any, hand_bone: str, grip: str) -> dict:
    from motionviewer.blender.weapon_grip_profiles import PROFILES

    hand_hash, weapon_hash = hand_signature(bpy, armature), geometry_signature(weapon)
    for profile in PROFILES:
        if (
            profile["hand_signature"] == hand_hash
            and weapon_hash in (profile["weapon_signature"], profile.get("source_weapon_signature"))
            and profile["hand_bone"] == hand_bone
        ):
            return profile
    raise ValueError(
        "surface grip has no calibration for this character/weapon/hand geometry; "
        "currently only the checked Feixiao right-hand weapon pair is supported. "
        "Use legacy/bounded explicitly for an uncalibrated asset."
    )


def select_calibrated_component(weapon: Any, profile: dict) -> dict:
    """Select an independently separated item from a known multi-weapon PMX pack.

    The source signature is checked before touching the imported working mesh.
    A polygon crossing the selection plane aborts: this must never slice a blade.
    """
    signature = geometry_signature(weapon)
    selection = profile.get("component_selection")
    if selection is None or signature == profile["weapon_signature"]:
        return {"selection": "already selected" if selection else "whole mesh", "dimensions_preserved": True}
    if signature != profile.get("source_weapon_signature"):
        raise ValueError("weapon component selection requires its exact source geometry signature")
    axis, limit = selection["local_axis"], selection["absolute_limit_m"]
    keep = np.asarray([abs(vertex.co[axis]) < limit for vertex in weapon.data.vertices])
    if not keep.any() or keep.all():
        raise ValueError("weapon component selection is empty or did not separate any component")
    if any(
        keep[list(face.vertices)].any() != keep[list(face.vertices)].all() for face in weapon.data.polygons
    ):
        raise ValueError("weapon component selection would cut through a polygon")
    before = len(weapon.data.vertices)
    selected = np.asarray([tuple(vertex.co) for vertex in weapon.data.vertices])[keep]
    dimensions = np.ptp(selected, axis=0)
    import bmesh

    mesh = bmesh.new()
    try:
        mesh.from_mesh(weapon.data)
        discard = [vertex for vertex in mesh.verts if abs(vertex.co[axis]) >= limit]
        bmesh.ops.delete(mesh, geom=discard, context="VERTS")
        mesh.to_mesh(weapon.data)
    finally:
        mesh.free()
    weapon.data.update()
    actual = geometry_signature(weapon)
    if actual != profile["weapon_signature"]:
        raise RuntimeError("selected weapon geometry did not match the calibrated component")
    return {
        "selection": selection["label"],
        "source_geometry_signature": signature,
        "selected_geometry_signature": actual,
        "source_vertices": before,
        "selected_vertices": len(weapon.data.vertices),
        "selected_local_dimensions_m": dimensions.tolist(),
        "dimensions_preserved": True,
        "source_asset_rewritten": False,
    }


def fit_surface_weapon(
    bpy: Any, weapon: Any, armature: Any, hand_bone: str, *, grip: str, frame: int | None
) -> dict:
    from mathutils import Matrix, Vector

    profile = _profile(bpy, armature, weapon, hand_bone, grip)
    component = select_calibrated_component(weapon, profile)
    if frame is not None:
        bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    wrist_world = armature.matrix_world @ armature.pose.bones[hand_bone].matrix
    placement = wrist_world @ Matrix(profile["weapon_in_hand"])
    weapon.parent = armature
    weapon.parent_type = "BONE"
    weapon.parent_bone = hand_bone
    weapon.matrix_parent_inverse = Matrix.Identity(4)
    # Blender must resolve the new bone parent before accepting a world matrix.
    bpy.context.view_layer.update()
    weapon.matrix_world = placement
    bpy.context.view_layer.update()
    error = float(np.max(np.abs(np.asarray(weapon.matrix_world) - np.asarray(placement))))
    if error > 1e-5:
        raise RuntimeError(f"surface grip parent transform did not settle ({error:g})")
    anchor = weapon.matrix_world @ Vector(profile["grip_center_local"])
    tip = weapon.matrix_world @ min(weapon.data.vertices, key=lambda vertex: vertex.co.y).co
    normal = (wrist_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    return {
        "fit_mode": "surface",
        "profile": profile["id"],
        "grip": profile["grip"],
        "requested_grip_hint": grip,
        "shaft_mm": profile["shaft_mm"],
        "length_m": profile["length_m"],
        "palm_normal": list(normal),
        "palm": list(anchor),
        "tip": list(tip),
        "placement_search": None,
        "geometry_verified": True,
        "component": component,
        "original_asset_scale_preserved": True,
        "grip_section": profile["grip_section"],
        "parent_transform_error": error,
        "scope": "Calibrated hand/weapon pair; body avoidance is audited separately, not solved here.",
    }


def _signed_distances(tree: Any, points: list[Any]) -> np.ndarray:
    """Nearest triangle distance, signed by a majority of three ray parity tests.

    Nearest-face normals alone falsely classified points outside the decorative
    shaft as inside. Ray parity also has limits on open/non-manifold meshes.
    """
    from mathutils import Vector

    rays = [
        Vector(direction).normalized()
        for direction in ((0.931, 0.211, 0.296), (-0.417, 0.866, 0.275), (0.113, -0.317, 0.942))
    ]
    result = []
    for point in points:
        _, _, _, distance = tree.find_nearest(point)
        votes = 0
        for ray in rays:
            position, crossings = point.copy(), 0
            for _ in range(32):
                hit, _, _, _ = tree.ray_cast(position, ray, 5.0)
                if hit is None:
                    break
                crossings += 1
                position = hit + ray * 1e-5
            votes += crossings % 2
        result.append(-distance if votes >= 2 else distance)
    return np.asarray(result)


def measure_hand_surface(bpy: Any, armature: Any, weapon: Any, hand_bone: str) -> dict:
    """Measure rendered skin vertices, triangle centres and edge midpoints."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    bpy.context.view_layer.update()
    tree = BVHTree.FromPolygons(
        [v.co.copy() for v in weapon.data.vertices],
        [tuple(face.vertices) for face in weapon.data.polygons],
        all_triangles=True,
    )
    inverse = weapon.matrix_world.inverted()
    side = "右" if hand_bone.startswith("右") else "左"
    depsgraph = bpy.context.evaluated_depsgraph_get()
    fingers = {}
    for finger, segments in (
        ("人指", "１２３"),
        ("中指", "１２３"),
        ("薬指", "１２３"),
        ("小指", "１２３"),
        ("親指", "０１２"),
    ):
        names = [side + finger + segment for segment in segments]
        points, pad_points = [], []
        for obj in _meshes(bpy, armature):
            if not all(name in obj.vertex_groups for name in names):
                continue
            evaluated = obj.evaluated_get(depsgraph)
            group_ids = {obj.vertex_groups[name].index for name in names}
            distal = obj.vertex_groups[names[-1]].index
            rest_inverse = (
                armature.data.bones[names[-1]].matrix_local.inverted() @ armature.matrix_world.inverted()
            )
            transform = inverse @ obj.matrix_world
            selected = {
                v.index
                for v in obj.data.vertices
                if sum(g.weight for g in v.groups if g.group in group_ids) > 0.05
            }
            coords = {index: transform @ evaluated.data.vertices[index].co for index in selected}
            points.extend(coords.values())
            for polygon in obj.data.polygons:
                if all(index in selected for index in polygon.vertices):
                    corners = [coords[index] for index in polygon.vertices]
                    points.append(sum(corners, Vector()) / len(corners))
                    points.extend(
                        (corners[j] + corners[(j + 1) % len(corners)]) / 2 for j in range(len(corners))
                    )
            for vertex in obj.data.vertices:
                if vertex.index not in selected:
                    continue
                rest_local = rest_inverse @ (obj.matrix_world @ vertex.co)
                if rest_local.y > armature.data.bones[names[-1]].length * 0.4 and any(
                    g.group == distal and g.weight > 0.55 for g in vertex.groups
                ):
                    pad_points.append(coords[vertex.index])
        if not points:
            continue
        distances = _signed_distances(tree, points)
        pad_distances = np.asarray([tree.find_nearest(point)[3] for point in pad_points])
        contact = np.sort(pad_distances)[: max(3, len(pad_distances) // 6)]
        fingers[finger] = {
            "skin_samples": len(points),
            "max_penetration_mm": max(0.0, float(-distances.min() * 1000)),
            "samples_inside_over_0_2mm": int(np.sum(distances < -0.0002)),
            "distal_pad_nearest_sixth_mean_gap_mm": float(contact.mean() * 1000) if len(contact) else None,
            "minimum_surface_gap_mm": max(0.0, float(distances.min() * 1000)),
        }
    return {
        "frame": bpy.context.scene.frame_current,
        "fingers": fingers,
        "max_penetration_mm": max((item["max_penetration_mm"] for item in fingers.values()), default=0.0),
        "method": "Evaluated skin vertices plus triangle centroids/edge midpoints against original weapon triangles; three-ray parity sign.",
        "limitations": "Sampled surfaces, not continuous collision certification. Open/non-manifold meshes can affect parity. Outline shells excluded.",
    }


def close_surface_hand(
    bpy: Any, armature: Any, hand_bone: str, *, weapon: Any, grip: str, frame_start: int, frames: int
) -> dict:
    from mathutils import Euler

    if weapon is None:
        raise ValueError("surface grip requires the matching imported weapon")
    profile = _profile(bpy, armature, weapon, hand_bone, grip)
    quaternions = {
        name: Euler(tuple(math.radians(v) for v in angles), "XYZ").to_quaternion()
        for name, angles in profile["finger_eulers_degrees"].items()
    }
    for frame in range(frame_start, frame_start + max(frames, 1)):
        bpy.context.scene.frame_set(frame)
        for name, quaternion in quaternions.items():
            bone = armature.pose.bones[name]
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = quaternion
            bone.keyframe_insert("rotation_quaternion", frame=frame)
    samples = sorted({frame_start, frame_start + max(0, frames - 1) // 2, frame_start + max(0, frames - 1)})
    diagnostics = []
    for frame in samples:
        bpy.context.scene.frame_set(frame)
        diagnostics.append(measure_hand_surface(bpy, armature, weapon, hand_bone))
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    return {
        "fit_mode": "surface",
        "mode": "calibrated",
        "profile": profile["id"],
        "bones": len(quaternions),
        "keys": len(quaternions) * max(1, frames),
        "solved": {name: angles[0] for name, angles in profile["finger_eulers_degrees"].items()},
        "total_local_euler_degrees": profile["finger_eulers_degrees"],
        "thumb": profile["thumb_contact"],
        "skin_surface_measured": True,
        "surface_samples": diagnostics,
        "source_motion_files_modified": False,
        "retargeted_finger_keys_replaced": True,
        "limits": "Asset-specific artistic grip; local joint angles are not general anatomical limits. See the separate left-arm and physics reports for body clearance.",
    }


def audit_weapon_body(
    bpy: Any, armature: Any, weapon: Any, *, frames: list[int], hand_bone: str = "右手首"
) -> dict:
    """Check actual non-hand character triangles against the entire weapon.

    BVH pairs are only candidates. Confirm both triangles' edges intersect the
    other triangle before reporting a crossing. Coplanar contact is not classified.
    """
    from collections import Counter

    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import intersect_ray_tri

    def crossing(first: list[Any], second: list[Any]) -> bool:
        for triangle, other in ((first, second), (second, first)):
            for index in range(3):
                start, end = triangle[index], triangle[(index + 1) % 3]
                delta = end - start
                if delta.length_squared < 1e-16:
                    continue
                hit = intersect_ray_tri(*other, delta, start, True)
                if hit is not None:
                    fraction = (hit - start).dot(delta) / delta.length_squared
                    if 1e-6 < fraction < 1 - 1e-6:
                        return True
        return False

    original_frame = bpy.context.scene.frame_current
    side = "右" if hand_bone.startswith("右") else "左"
    sources = []
    for obj in _meshes(bpy, armature):
        hand_groups = {
            g.index
            for g in obj.vertex_groups
            if g.name == hand_bone or (g.name.startswith(side) and "指" in g.name)
        }
        hand_vertices = {
            v.index
            for v in obj.data.vertices
            if sum(g.weight for g in v.groups if g.group in hand_groups) > 0.5
        }
        faces, materials = [], []
        for face in obj.data.polygons:
            if any(index in hand_vertices for index in face.vertices):
                continue
            # MMD imports triangles; fan triangulation also handles other callers.
            vertices = list(face.vertices)
            for i in range(1, len(vertices) - 1):
                faces.append((vertices[0], vertices[i], vertices[i + 1]))
                material = obj.data.materials[face.material_index]
                materials.append(material.name if material else "unnamed")
        sources.append((obj, faces, materials))
    weapon_faces = [tuple(face.vertices) for face in weapon.data.polygons]
    samples = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        weapon_points = [weapon.matrix_world @ vertex.co for vertex in weapon.data.vertices]
        weapon_tree = BVHTree.FromPolygons(weapon_points, weapon_faces, all_triangles=True)
        counts, candidates = Counter(), 0
        for obj, faces, materials in sources:
            evaluated = obj.evaluated_get(depsgraph)
            points = [obj.matrix_world @ vertex.co for vertex in evaluated.data.vertices]
            body_tree = BVHTree.FromPolygons(points, faces, all_triangles=True)
            overlaps = weapon_tree.overlap(body_tree)
            candidates += len(overlaps)
            for weapon_index, body_index in overlaps:
                first = [weapon_points[index] for index in weapon_faces[weapon_index]]
                second = [points[index] for index in faces[body_index]]
                if crossing(first, second):
                    counts[materials[body_index]] += 1
        samples.append(
            {
                "frame": frame,
                "triangle_crossings": sum(counts.values()),
                "materials": dict(counts),
                "bvh_candidates": candidates,
            }
        )
    bpy.context.scene.frame_set(original_frame)
    bpy.context.view_layer.update()
    return {
        "samples": samples,
        "frames_with_crossings": [row["frame"] for row in samples if row["triangle_crossings"]],
        "maximum_triangle_crossings": max((row["triangle_crossings"] for row in samples), default=0),
        "method": "Two-sided triangle-edge intersection on evaluated character skin and original weapon mesh; gripping hand and outline shells excluded.",
        "limitations": "Discrete frames; coplanar overlap and full containment without a boundary crossing are not classified.",
    }
