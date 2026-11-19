"""Optional, asset-profiled mesh cloth layered over the unmodified body animation.

Simulate a welded single surface, bind the original appearance to it, and remove
only the replaced garment's Bullet bodies. This is opt-in per verified PMX hash;
material and bone selections are character adaptations, not universal defaults.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MeshClothSettings:
    bone_prefixes: tuple[str, ...]
    materials: tuple[str, ...]
    outer_surface_materials: tuple[str, ...] = ()
    collider_material_tokens: tuple[str, ...] = ("肌", "skin", "裤")
    weld_distance_m: float = 0.0012
    pin_transition_weight: float = 0.7
    pin_full_weight: float = 0.05
    quality: int = 12
    mass: float = 0.25
    air_damping: float = 1.0
    tension_stiffness: float = 30.0
    compression_stiffness: float = 30.0
    shear_stiffness: float = 15.0
    bending_stiffness: float = 0.1
    structural_damping: float = 5.0
    bending_damping: float = 0.5
    collision_quality: int = 4
    collision_distance_m: float = 0.002
    collider_thickness_m: float = 0.0015
    self_distance_m: float = 0.002
    friction: float = 5.0
    #: Subdivide the cloth proxy this many times before simulating (0 = as authored).
    #: A PMX skirt face can be wider than a finger — Nilou's 裙 faces are ~27 mm
    #: across against a 15-18 mm finger — so a hand slices between the vertices of
    #: one triangle and no amount of collision on those vertices can stop it. One
    #: level halves the edges, which gives the cloth vertices to fold around a hand.
    subdivide: int = 0

    def __post_init__(self):
        for name in ("bone_prefixes", "materials", "outer_surface_materials", "collider_material_tokens"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"mesh cloth {name} must contain nonempty strings")
            if name != "outer_surface_materials" and not values:
                raise ValueError(f"mesh cloth {name} cannot be empty")
            object.__setattr__(self, name, tuple(values))
        if not set(self.outer_surface_materials) <= set(self.materials):
            raise ValueError("outer surface materials must also be selected materials")
        if isinstance(self.subdivide, bool) or not isinstance(self.subdivide, int) or not 0 <= self.subdivide <= 2:
            raise ValueError("mesh cloth subdivide must be an integer in 0..2")
        for name in ("quality", "collision_quality"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 80:
                raise ValueError(f"mesh cloth {name} must be an integer in 1..80")
        bounds = {
            "weld_distance_m": (0.00001, 0.005), "pin_transition_weight": (0, 1),
            "pin_full_weight": (0, 1), "mass": (0.001, 10), "air_damping": (0, 10),
            "tension_stiffness": (0, 10000), "compression_stiffness": (0, 10000),
            "shear_stiffness": (0, 10000), "bending_stiffness": (0, 10000),
            "structural_damping": (0, 50), "bending_damping": (0, 50),
            "collision_distance_m": (0.001, 0.01), "collider_thickness_m": (0.001, 0.01),
            "self_distance_m": (0.001, 0.01), "friction": (0, 80),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"mesh cloth {name} must be finite in {low}..{high}")
        if self.pin_full_weight >= self.pin_transition_weight:
            raise ValueError("pin full weight must be below transition weight")


@dataclass
class MeshClothRig:
    proxy: Any
    collider_collection: Any
    pins: str
    collision_exclusion: str
    settings: MeshClothSettings
    report: dict


def _components(neighbours: dict[int, set[int]]) -> list[set[int]]:
    pending = set(neighbours)
    components = []
    while pending:
        stack = [pending.pop()]
        component = set(stack)
        while stack:
            for neighbour in neighbours[stack.pop()] & pending:
                pending.remove(neighbour)
                component.add(neighbour)
                stack.append(neighbour)
        components.append(component)
    return components


def _outer_faces(mesh: Any, faces: set[int], materials: tuple[str, ...]) -> tuple[set[int], list[dict]]:
    """Select the outward sheet of an explicitly profiled double-sided material.

    Radial orientation is measured in the imported character's local XY plane.
    This rule is intended for verified torso/hip garments; it is not inferred for
    arbitrary sleeves, capes, or new characters.
    """
    kept = set(faces)
    report = []
    for material in materials:
        polygons = [p for p in mesh.data.polygons if p.index in faces and mesh.data.materials[p.material_index].name == material]
        neighbours: dict[int, set[int]] = {}
        for polygon in polygons:
            for vertex in polygon.vertices:
                neighbours.setdefault(vertex, set()).update(polygon.vertices)
        retained = discarded = 0
        for component in _components(neighbours):
            selected = [p for p in polygons if p.vertices[0] in component]
            area = sum(p.area for p in selected)
            radial = sum(p.area * (p.center.x * p.normal.x + p.center.y * p.normal.y) for p in selected) / max(area, 1e-12)
            outward = radial > 0
            if outward:
                retained += len(selected)
            else:
                discarded += len(selected)
                kept.difference_update(p.index for p in selected)
            report.append({"material": material, "faces": len(selected), "radial_orientation": radial, "retained": outward})
        if not retained or not discarded:
            raise ValueError(f"Expected distinct outward/inward sheets for {material!r}; verify the asset profile")
    return kept, report


def _copy_surface(bpy: Any, mesh: Any, name: str, faces: set[int], weld: float = 0, subdivide: int = 0) -> Any:
    import bmesh

    obj = mesh.copy()
    obj.data = mesh.data.copy()
    obj.name = name
    bpy.context.collection.objects.link(obj)
    if obj.data.shape_keys:
        obj.shape_key_clear()
    for modifier in list(obj.modifiers):
        if modifier.type != "ARMATURE":
            obj.modifiers.remove(modifier)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in faces], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    if weld:
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=weld)
        seen, duplicates = set(), []
        bm.verts.index_update()
        for face in bm.faces:
            key = tuple(sorted(v.index for v in face.verts))
            if key in seen:
                duplicates.append(face)
            seen.add(key)
        if duplicates:
            bmesh.ops.delete(bm, geom=duplicates, context="FACES_ONLY")
    if any(len(edge.link_faces) > 2 for edge in bm.edges):
        bm.free()
        raise ValueError(f"Non-manifold cloth surface: {name}")
    for _ in range(subdivide):
        # Midpoint subdivision; vertex-group weights interpolate onto the new
        # vertices, so the pins and the armature binding carry over.
        bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=1, use_grid_fill=True)
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    obj.hide_render = True
    obj.display_type = "WIRE"
    obj["motionviewer_helper"] = True
    return obj


def prepare_mesh_cloth(bpy: Any, armature: Any, settings: MeshClothSettings, *, frame_start: int) -> MeshClothRig:
    """Run after retarget/shading, before building any Bullet simulation."""
    from mmd_tools.core.model import FnModel, Model

    scene = bpy.context.scene
    model = Model(FnModel.find_root_object(armature))
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    meshes = [o for o in model.meshes() if not o.get("motionviewer_helper") and not any(m.name == "Outline" for m in o.modifiers)]
    if len(meshes) != 1:
        raise ValueError("Mesh cloth profile currently requires one original PMX mesh")
    mesh = meshes[0]
    names = {g.index: g.name for g in mesh.vertex_groups}
    garment = {n for n in names.values() if n.startswith(settings.bone_prefixes)}
    if not garment:
        raise ValueError("Mesh cloth profile selected no garment bones")
    missing = set(settings.materials) - {m.name for m in mesh.data.materials if m}
    if missing:
        raise ValueError(f"Mesh cloth materials not found: {sorted(missing)}")
    weights = {v.index: sum(g.weight for g in v.groups if names[g.group] in garment) for v in mesh.data.vertices}
    faces = {p.index for p in mesh.data.polygons if mesh.data.materials[p.material_index].name in settings.materials and max(weights[i] for i in p.vertices) > 1e-6}
    faces, layers = _outer_faces(mesh, faces, settings.outer_surface_materials)
    collider_faces = {p.index for p in mesh.data.polygons if max(weights[i] for i in p.vertices) < 1e-6 and any(t.casefold() in mesh.data.materials[p.material_index].name.casefold() for t in settings.collider_material_tokens)}
    if not faces or not collider_faces:
        raise ValueError("Mesh cloth requires nonempty garment and body collision surfaces")
    proxy = _copy_surface(bpy, mesh, "GarmentClothProxy", faces, settings.weld_distance_m, settings.subdivide)
    collider = _copy_surface(bpy, mesh, "GarmentBodyCollision", collider_faces)
    collection = bpy.data.collections.new("GarmentCollisions")
    scene.collection.children.link(collection)
    collection.objects.link(collider)
    collider.modifiers.new("ClothBodyCollision", "COLLISION")
    collider.collision.thickness_outer = settings.collider_thickness_m
    collider.collision.thickness_inner = settings.collider_thickness_m
    collider.collision.cloth_friction = settings.friction
    group_names = {g.index: g.name for g in proxy.vertex_groups}
    pin_group = proxy.vertex_groups.new(name="ClothPin")
    pins = set()
    for vertex in proxy.data.vertices:
        weight = sum(g.weight for g in vertex.groups if group_names.get(g.group) in garment)
        pin = min(1, max(0, (settings.pin_transition_weight - weight) / (settings.pin_transition_weight - settings.pin_full_weight)))
        if pin > 0:
            pin_group.add([vertex.index], pin, "REPLACE")
            pins.add(vertex.index)
    neighbours = {v.index: set() for v in proxy.data.vertices}
    for edge in proxy.data.edges:
        a, b = edge.vertices
        neighbours[a].add(b)
        neighbours[b].add(a)
    components = [{"vertices": len(c), "pins": len(c & pins)} for c in _components(neighbours)]
    if any(not c["pins"] for c in components):
        raise ValueError(f"Unattached garment island: {components}")
    exclusion = proxy.vertex_groups.new(name="ClothPinnedCollisionExclusion")
    exclusion.add(sorted(pins), 1, "REPLACE")
    visuals = [mesh] + [o for o in bpy.data.objects if o.type == "MESH" and any(m.name == "Outline" for m in o.modifiers) and any(m.type == "ARMATURE" and m.object == armature for m in o.modifiers)]
    bpy.context.view_layer.update()
    for obj in visuals:
        group_names = {g.index: g.name for g in obj.vertex_groups}
        influence = obj.vertex_groups.new(name="ClothSurfaceInfluence")
        for vertex in obj.data.vertices:
            weight = sum(g.weight for g in vertex.groups if group_names.get(g.group) in garment)
            if weight > 0:
                influence.add([vertex.index], min(1, weight), "REPLACE")
        deform = obj.modifiers.new("GarmentSurface", "SURFACE_DEFORM")
        deform.target, deform.vertex_group, deform.falloff = proxy, influence.name, 4
        with bpy.context.temp_override(object=obj, active_object=obj):
            if any(m.name == "Outline" for m in obj.modifiers):
                bpy.ops.object.modifier_move_up(modifier=deform.name)
            bpy.ops.object.surfacedeform_bind(modifier=deform.name)
        if not deform.is_bound:
            raise RuntimeError(f"Could not bind garment appearance: {obj.name}")
    removed = [o for o in model.rigidBodies() if o.mmd_rigid.bone in garment]
    removed_names = {o.name for o in removed}
    for joint in list(model.joints()):
        constraint = joint.rigid_body_constraint
        if constraint and any(o and o.name in removed_names for o in (constraint.object1, constraint.object2)):
            bpy.data.objects.remove(joint, do_unlink=True)
    for obj in removed:
        bpy.data.objects.remove(obj, do_unlink=True)
    report = {"settings": asdict(settings), "proxy_vertices": len(proxy.data.vertices), "proxy_faces": len(proxy.data.polygons), "pins": len(pins), "components": components, "surface_layers": layers, "collider_faces": len(collider.data.polygons), "replaced_bullet_bodies": len(removed), "appearance_objects": [o.name for o in visuals]}
    return MeshClothRig(proxy, collection, pin_group.name, exclusion.name, settings, report)


def bake_mesh_cloth(bpy: Any, rig: MeshClothRig, *, frame_start: int, frame_end: int, settle_frames: int) -> dict:
    """Bake after the remaining Bullet bones, before camera or render sampling."""
    scene = bpy.context.scene
    config = rig.settings
    cloth = rig.proxy.modifiers.new("ContinuousGarmentCloth", "CLOTH")
    settings = cloth.settings
    for name in ("quality", "mass", "air_damping", "tension_stiffness", "compression_stiffness", "shear_stiffness", "bending_stiffness", "bending_damping"):
        setattr(settings, name, getattr(config, name))
    settings.tension_damping = settings.compression_damping = settings.shear_damping = config.structural_damping
    settings.vertex_group_mass = rig.pins
    settings.pin_stiffness = 1
    settings.use_dynamic_mesh = False
    collision = cloth.collision_settings
    collision.use_collision = collision.use_self_collision = True
    collision.collection = rig.collider_collection
    collision.collision_quality = config.collision_quality
    collision.distance_min = config.collision_distance_m
    collision.self_distance_min = config.self_distance_m
    collision.self_friction = config.friction
    collision.vertex_group_object_collisions = rig.collision_exclusion
    cache = cloth.point_cache
    cache.frame_start, cache.frame_end = frame_start - settle_frames, frame_end
    original = scene.frame_start, scene.frame_end
    try:
        scene.frame_start, scene.frame_end = cache.frame_start, cache.frame_end
        scene.frame_set(cache.frame_start)
        with bpy.context.temp_override(scene=scene, object=rig.proxy, active_object=rig.proxy, point_cache=cache):
            result = bpy.ops.ptcache.bake(bake=True)
    finally:
        scene.frame_start, scene.frame_end = original
    if not cache.is_baked:
        raise RuntimeError("Mesh cloth cache did not bake")
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    cache_start, cache_end = cache.frame_start, cache.frame_end
    stored = _store_proxy_frames(bpy, rig.proxy, frame_start, frame_end)
    return {**rig.report, "baked": True, "bake_result": sorted(result), "cache_start": cache_start, "cache_end": cache_end, "fps": scene.render.fps / scene.render.fps_base, "stored_frames": stored, "cache_storage": "proxy_shape_keys"}


def _store_proxy_frames(bpy: Any, proxy: Any, first: int, last: int) -> int:
    """Store the simulated sheet explicitly so review/reopening cannot re-simulate.

    Only this small hidden proxy receives generated shape keys. Body bones and
    source animation remain untouched; interpolation is linear between samples.
    """
    import numpy as np

    from .retarget._animate import _linearize

    scene = bpy.context.scene
    positions = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        evaluated = proxy.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", coords)
            positions.append(coords)
        finally:
            evaluated.to_mesh_clear()
    for modifier in list(proxy.modifiers):
        proxy.modifiers.remove(modifier)
    proxy.shape_key_add(name="Basis")
    for frame, coords in enumerate(positions, first):
        key = proxy.shape_key_add(name=f"ClothFrame_{frame:06d}")
        key.data.foreach_set("co", coords)
        for offset, value in ((-1, 0), (0, 1), (1, 0)):
            key.value = value
            key.keyframe_insert("value", frame=frame + offset)
    _linearize(proxy.data.shape_keys)
    proxy["motionviewer_cloth_frame_start"] = first
    proxy["motionviewer_cloth_frame_end"] = last
    scene.frame_set(first)
    bpy.context.view_layer.update()
    return len(positions)
