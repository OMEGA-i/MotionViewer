"""Load 3D Genshin stages (.blend / .fbx / .pmx), place & orient animated actors on the
walkable floor without clipping, and frame natural perspective shots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _alpha_stats(bpy_img: Any) -> tuple[int, int, float]:
    import numpy as np

    try:
        width, height = bpy_img.size
        if width > 0 and height > 0 and bpy_img.channels == 4:
            pixels = np.empty(width * height * 4, dtype=np.float32)
            bpy_img.pixels.foreach_get(pixels)
            alpha = pixels[3::64] * 255.0
            return int(alpha.min()), int(alpha.max()), float(alpha.mean())
    except Exception:
        pass
    return 255, 255, 255.0


def _rebuild_genshin_fbx_materials(bpy: Any, stage_dir: Path, meshes: list[Any]) -> dict:
    """Wire Genshin's _Diffuse, _Normal, and _SMBE textures cleanly in EEVEE."""
    textures_by_stem = {p.stem.lower(): p for p in stage_dir.glob("*.png")}
    removed = 0
    rebuilt = 0

    for mesh in list(meshes):
        mname = mesh.name.lower()
        mats = [slot.material for slot in mesh.material_slots if slot.material]
        mat_names = [mt.name.lower() for mt in mats]
        has_tex = False
        for mt in mats:
            if mt.use_nodes and any(n.type == "TEX_IMAGE" and n.image for n in mt.node_tree.nodes):
                has_tex = True
            else:
                cand = mt.name.lower().removesuffix("_diffuse")
                if f"{cand}_diffuse" in textures_by_stem or cand in textures_by_stem:
                    has_tex = True
        if (
            not mats
            or not has_tex
            or mname.startswith("mesh1")
            or any(
                k in mname or any(k in mn for mn in mat_names)
                for k in (
                    "shadowmesh",
                    "collider",
                    "collision",
                    "window_effect",
                    "windoweffect",
                    "brightline",
                    "eff_aura",
                    "_ao",
                )
            )
        ):
            meshes.remove(mesh)
            bpy.data.objects.remove(mesh, do_unlink=True)
            removed += 1

    seen: set[str] = set()
    for mesh in meshes:
        for slot in mesh.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen:
                continue
            seen.add(mat.name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            diff_path: Path | None = None
            norm_path: Path | None = None
            for node in nodes:
                if node.type == "TEX_IMAGE" and node.image and node.image.filepath:
                    candidate = Path(bpy.path.abspath(node.image.filepath))
                    if not candidate.is_file():
                        candidate = stage_dir / candidate.name
                    if candidate.is_file():
                        low = candidate.stem.lower()
                        if "normal" in low:
                            norm_path = candidate
                        elif "diffuse" in low or diff_path is None:
                            diff_path = candidate
            if diff_path is None:
                cand = mat.name.lower().removesuffix("_diffuse")
                for kt in (f"{cand}_diffuse", cand):
                    if kt in textures_by_stem:
                        diff_path = textures_by_stem[kt]
                        break
            if diff_path is None:
                continue

            base_prefix = diff_path.stem
            for suffix in ("_Diffuse_Clamp", "_Diffuse", "_diffuse"):
                if base_prefix.endswith(suffix):
                    base_prefix = base_prefix[: -len(suffix)]
                    break
            if norm_path is None:
                for kt in (f"{base_prefix.lower()}_normal", f"{base_prefix.lower()}_normal_clamp"):
                    if kt in textures_by_stem:
                        norm_path = textures_by_stem[kt]
                        break
            smbe_path = None
            for kt in (
                f"{base_prefix.lower()}_smbe",
                f"{base_prefix.lower()}_smbe_clamp",
                f"{base_prefix.lower()}_smbe_nomip",
            ):
                if kt in textures_by_stem:
                    smbe_path = textures_by_stem[kt]
                    break

            nodes.clear()
            out = nodes.new("ShaderNodeOutputMaterial")
            out.location = (620, 0)
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (260, 0)
            bsdf.inputs["Roughness"].default_value = 0.55
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.38
            links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

            diff_tex = nodes.new("ShaderNodeTexImage")
            diff_tex.location = (-400, 220)
            diff_tex.image = bpy.data.images.load(str(diff_path), check_existing=True)
            links.new(diff_tex.outputs["Color"], bsdf.inputs["Base Color"])

            amin, _, amean = _alpha_stats(diff_tex.image)
            is_cutout = any(
                k in base_prefix.lower()
                for k in (
                    "grass",
                    "leaf",
                    "flower",
                    "tree",
                    "plant",
                    "pattern",
                    "clamp",
                    "curtain",
                    "carpet",
                    "blanket",
                    "decal",
                    "screen",
                )
            )
            if amin < 245 and (is_cutout or amean < 245.0):
                links.new(diff_tex.outputs["Alpha"], bsdf.inputs["Alpha"])
                if hasattr(mat, "surface_render_method"):
                    mat.surface_render_method = "DITHERED"

            if norm_path and norm_path.is_file():
                norm_tex = nodes.new("ShaderNodeTexImage")
                norm_tex.location = (-400, -320)
                norm_tex.image = bpy.data.images.load(str(norm_path), check_existing=True)
                norm_tex.image.colorspace_settings.name = "Non-Color"
                nmap = nodes.new("ShaderNodeNormalMap")
                nmap.location = (-80, -320)
                nmap.inputs["Strength"].default_value = 0.85
                links.new(norm_tex.outputs["Color"], nmap.inputs["Color"])
                links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

            if smbe_path and smbe_path.is_file():
                smbe_tex = nodes.new("ShaderNodeTexImage")
                smbe_tex.location = (-400, -40)
                smbe_tex.image = bpy.data.images.load(str(smbe_path), check_existing=True)
                smbe_tex.image.colorspace_settings.name = "Non-Color"
                sep = nodes.new("ShaderNodeSeparateColor")
                sep.location = (-120, -40)
                links.new(smbe_tex.outputs["Color"], sep.inputs["Color"])

                inv = nodes.new("ShaderNodeMath")
                inv.operation = "MULTIPLY_ADD"
                inv.location = (60, 40)
                inv.inputs[1].default_value = -0.68
                inv.inputs[2].default_value = 0.90
                inv.use_clamp = True
                links.new(sep.outputs["Red"], inv.inputs[0])
                links.new(inv.outputs["Value"], bsdf.inputs["Roughness"])
                links.new(sep.outputs["Green"], bsdf.inputs["Metallic"])

                s_amin, s_amax, _ = _alpha_stats(smbe_tex.image)
                if s_amin < 245 and s_amax > 20 and "Emission Color" in bsdf.inputs:
                    links.new(diff_tex.outputs["Color"], bsdf.inputs["Emission Color"])
                    em_mul = nodes.new("ShaderNodeMath")
                    em_mul.operation = "MULTIPLY"
                    em_mul.location = (60, -140)
                    em_mul.inputs[1].default_value = 2.8
                    links.new(smbe_tex.outputs["Alpha"], em_mul.inputs[0])
                    links.new(em_mul.outputs["Value"], bsdf.inputs["Emission Strength"])
            rebuilt += 1
    return {"materials_rebuilt": rebuilt, "removed_fx_meshes": removed}


def import_genshin_stage(bpy: Any, path: str | Path, *, scale: float | None = None) -> tuple[list[Any], dict]:
    """Import a static environment from `.blend`, `.fbx`, or `.pmx`."""
    stage_path = Path(path).resolve()
    suffix = stage_path.suffix.lower()
    scene = bpy.context.scene

    if suffix == ".blend":
        before = set(bpy.data.objects)
        with bpy.data.libraries.load(str(stage_path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
            data_to.worlds = list(data_from.worlds)
        meshes = []
        lights = 0
        for obj in data_to.objects:
            if obj is None or obj in before:
                continue
            if obj.type in {"MESH", "LIGHT"}:
                scene.collection.objects.link(obj)
                if obj.type == "MESH":
                    if scale is not None and scale != 1.0:
                        obj.scale = tuple(float(v) * float(scale) for v in obj.scale)
                    meshes.append(obj)
                else:
                    lights += 1
        if data_to.worlds:
            scene.world = data_to.worlds[0]
        bpy.context.view_layer.update()
        return meshes, {"format": "blend", "path": str(stage_path), "meshes": len(meshes), "lights": lights}

    if suffix == ".fbx":
        fbx_scale = float(scale) if scale is not None else 100.0
        before = set(bpy.data.objects)
        bpy.ops.import_scene.fbx(filepath=str(stage_path), global_scale=fbx_scale)
        added = [obj for obj in bpy.data.objects if obj not in before]
        meshes = [obj for obj in added if obj.type == "MESH"]
        for obj in added:
            if obj.type != "MESH":
                for child in list(obj.children):
                    mw = child.matrix_world.copy()
                    child.parent = None
                    child.matrix_world = mw
                bpy.data.objects.remove(obj, do_unlink=True)
        mat_info = _rebuild_genshin_fbx_materials(bpy, stage_path.parent, meshes)
        bpy.context.view_layer.update()
        return meshes, {"format": "fbx", "path": str(stage_path), "meshes": len(meshes), **mat_info}

    from .mmd_props import load_stage

    pmx_scale = float(scale) if scale is not None else 0.08
    meshes = load_stage(bpy, stage_path, scale=pmx_scale)
    bpy.context.view_layer.update()
    return meshes, {"format": "pmx", "path": str(stage_path), "meshes": len(meshes)}


def raycast_stage_floor(bpy: Any, stage_meshes: list[Any], x: float, y: float, *, probe_z: float = 1.6) -> float:
    """Find the walkable floor height at `(x, y)` using `scene.ray_cast` in world space!

    Note: `BVHTree.FromObject` works in object local space (where FBX objects may carry
    90-deg X rotations), whereas `scene.ray_cast` operates strictly in world space!
    Temporarily hides non-stage objects so only architectural stage surfaces are hit.
    """
    from mathutils import Vector  # type: ignore

    scene = bpy.context.scene
    stage_set = set(stage_meshes)
    restores: list[tuple[Any, bool]] = []
    for obj in scene.objects:
        if obj not in stage_set and not obj.hide_viewport:
            restores.append((obj, obj.hide_viewport))
            obj.hide_viewport = True
    try:
        deps = bpy.context.evaluated_depsgraph_get()
        for start_z in (probe_z, 0.8, 0.35):
            hit, loc, norm, _, hit_obj, _ = scene.ray_cast(
                deps, Vector((float(x), float(y), float(start_z))), Vector((0.0, 0.0, -1.0)), distance=6.0
            )
            if hit and norm.z > 0.82:
                # Skip tall furniture/pews if hit above z=0.25m unless on stairs/platform
                hname = hit_obj.name.lower() if hit_obj else ""
                if float(loc.z) > 0.25 and not any(k in hname for k in ("ground", "floor", "stairs", "tatami", "carpet")):
                    continue
                return float(loc.z)
    finally:
        for obj, prev in restores:
            obj.hide_viewport = prev
        bpy.context.view_layer.update()
    return 0.0


def place_and_ground_actor_on_stage(
    bpy: Any,
    actor: Any,
    stage_meshes: list[Any],
    *,
    frame_start: int,
    frames: int,
    target_xy: tuple[float, float] = (0.0, 0.0),
    yaw_degrees: float = 0.0,
    anchor_mode: str = "mean",
) -> dict:
    """Rotate and translate the MMD root object (`FnModel.find_root_object`) ONCE as a static
    rigid world transform BEFORE physics build/bake!

    Moving the MMD root object (`mmd_root`) carries the armature, character mesh, AND all
    imported rigid-body/joint children together without breaking `mmd_tools` rest alignment
    or triggering vertex explosions during Bullet/Cloth baking.
    """
    import numpy as np
    from mathutils import Euler, Matrix, Vector  # type: ignore
    from mmd_tools.core.model import FnModel  # type: ignore

    from .mmd_contact import measure_feet

    armature = actor.armature
    mmd_root = FnModel.find_root_object(armature) or armature
    scene = bpy.context.scene
    yaw_rad = math.radians(float(yaw_degrees))
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    rot_xy = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)

    # Read baked local armature locations across the clip (relative to identity mmd_root)
    local_locs = np.zeros((frames, 3), dtype=np.float64)
    for offset in range(frames):
        scene.frame_set(frame_start + offset)
        local_locs[offset] = tuple(armature.location)

    rotated_xy = local_locs[:, :2] @ rot_xy.T
    if anchor_mode == "start":
        anchor_xy = rotated_xy[0]
    elif anchor_mode == "end":
        anchor_xy = rotated_xy[-1]
    else:
        anchor_xy = rotated_xy.mean(axis=0)

    root_shift_xy = np.array(target_xy, dtype=np.float64) - anchor_xy
    world_xy = rotated_xy + root_shift_xy

    # Also parent mesh-cloth proxy (if any) under mmd_root so moving mmd_root carries the bound garment proxy!
    for helper in bpy.data.objects:
        if helper.get("motionviewer_helper") and helper.parent is None and helper.name == "GarmentClothProxy":
            helper.parent = mmd_root

    # Set static XY + Yaw on mmd_root (no keyframes on mmd_root, so pre-roll frames -29..0 inherit it!)
    mmd_root.rotation_mode = "XYZ"
    mmd_root.rotation_euler = (0.0, 0.0, yaw_rad)
    mmd_root.location = (float(root_shift_xy[0]), float(root_shift_xy[1]), 0.0)
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()

    # Raycast floor along the actual trajectory and measure animated shoe soles
    mid_x, mid_y = float(world_xy.mean(axis=0)[0]), float(world_xy.mean(axis=0)[1])
    floor_samples = [
        raycast_stage_floor(bpy, stage_meshes, float(world_xy[idx, 0]), float(world_xy[idx, 1]))
        for idx in np.linspace(0, frames - 1, min(7, frames), dtype=int)
    ]
    stage_floor_z = float(np.median(floor_samples))
    foot_samples = measure_feet(bpy, actor, frame_start=frame_start, frames=frames)
    per_frame_min_sole = foot_samples.sole_z.min(axis=1)
    ref_sole_z = float(np.percentile(per_frame_min_sole, 10))
    delta_z = stage_floor_z - ref_sole_z

    mmd_root.location = (float(root_shift_xy[0]), float(root_shift_xy[1]), float(delta_z))
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()

    span_xy = float(np.linalg.norm(world_xy.max(axis=0) - world_xy.min(axis=0)))
    _add_contact_shadow_catcher(
        bpy, center=(mid_x, mid_y, stage_floor_z + 0.0015), radius=max(span_xy * 0.75 + 1.2, 2.2)
    )
    _harmonize_character_tint(bpy, actor.mesh_objects)

    world_mat = Matrix.Translation(Vector((float(root_shift_xy[0]), float(root_shift_xy[1]), float(delta_z)))) @ Euler(
        (0.0, 0.0, yaw_rad), "XYZ"
    ).to_matrix().to_4x4()

    return {
        "target_xy": [round(float(target_xy[0]), 3), round(float(target_xy[1]), 3)],
        "yaw_degrees": round(float(yaw_degrees), 2),
        "stage_floor_z": round(stage_floor_z, 4),
        "ref_sole_z_before": round(ref_sole_z, 4),
        "applied_delta_z": round(delta_z, 4),
        "root_shift_xyz": [round(float(root_shift_xy[0]), 4), round(float(root_shift_xy[1]), 4), round(delta_z, 4)],
        "path_start_xy": np.round(world_xy[0], 3).tolist(),
        "path_end_xy": np.round(world_xy[-1], 3).tolist(),
        "path_span_m": round(span_xy, 3),
        "transform_matrix": np.asarray(world_mat, dtype=np.float64).tolist(),
    }


def _add_contact_shadow_catcher(bpy: Any, *, center: tuple[float, float, float], radius: float) -> Any:
    """Semi-transparent shadow-receiving decal plane in EEVEE via ShaderToRGB."""
    bpy.ops.mesh.primitive_circle_add(vertices=48, radius=radius, fill_type="NGON", location=center)
    disc = bpy.context.active_object
    disc.name = "Stage_ContactShadowDisc"
    mat = bpy.data.materials.new("Stage_ContactShadowMat")
    mat.use_nodes = True
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (680, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (420, 0)
    bsdf.inputs["Base Color"].default_value = (0.04, 0.03, 0.05, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0

    diff = nodes.new("ShaderNodeBsdfDiffuse")
    diff.location = (-360, 0)
    diff.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    s2rgb = nodes.new("ShaderNodeShaderToRGB")
    s2rgb.location = (-160, 0)
    bw = nodes.new("ShaderNodeRGBToBW")
    bw.location = (20, 0)
    ramp = nodes.new("ShaderNodeMapRange")
    ramp.location = (200, 0)
    # Lit floor (val >= 0.52) -> alpha 0.0 (invisible); shadowed floor (val <= 0.22) -> alpha 0.34
    ramp.inputs["From Min"].default_value = 0.18
    ramp.inputs["From Max"].default_value = 0.52
    ramp.inputs["To Min"].default_value = 0.34
    ramp.inputs["To Max"].default_value = 0.0
    ramp.clamp = True

    links.new(diff.outputs["BSDF"], s2rgb.inputs["Shader"])
    links.new(s2rgb.outputs["Color"], bw.inputs["Color"])
    links.new(bw.outputs["Val"], ramp.inputs["Value"])
    links.new(ramp.outputs["Result"], bsdf.inputs["Alpha"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    disc.data.materials.append(mat)
    return disc


def _harmonize_character_tint(bpy: Any, meshes: list[Any], strength: float = 0.14) -> None:
    """Blend a touch of the world's ambient tint into the character's cel emission so she belongs to the room."""
    world = bpy.context.scene.world
    if world is None or not world.use_nodes:
        return
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        return
    amb = tuple(float(c) for c in bg.inputs["Color"].default_value[:3])
    peak = max(amb) if max(amb) > 1e-4 else 1.0
    norm_tint = tuple(1.0 * (1.0 - strength) + (c / peak) * strength for c in amb)

    seen: set[str] = set()
    for mesh in meshes:
        for slot in mesh.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen or not mat.use_nodes:
                continue
            seen.add(mat.name)
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            for node in nodes:
                if node.type == "EMISSION" and node.inputs["Color"].is_linked:
                    src_socket = node.inputs["Color"].links[0].from_socket
                    mix = nodes.new("ShaderNodeMixRGB")
                    mix.blend_type = "MULTIPLY"
                    mix.inputs["Fac"].default_value = 1.0
                    mix.inputs["Color2"].default_value = (*norm_tint, 1.0)
                    links.new(src_socket, mix.inputs["Color1"])
                    links.new(mix.outputs["Color"], node.inputs["Color"])
                    break
