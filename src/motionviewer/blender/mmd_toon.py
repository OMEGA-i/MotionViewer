"""Cel shading, outlines and a lighting rig for imported MMD characters.

Genshin and Honkai models are authored for a non-photoreal shader: a small ramp
texture decides the shadow step, a sphere map fakes the specular, and an outline
sells the drawing.  ``mmd_tools`` loads all three and then wires the ramp into a
physically based mix as if it were a colour multiply, so the shading comes out
smooth and slightly waxy — the clay look — and the sphere map is switched off
entirely (``Sphere Tex Fac`` is 0).

This module rebuilds each material as the model was authored to be rendered:

    N·L  ->  sharpened step  ->  the model's own shadow tint  ->  x base texture

``Shader to RGB`` supplies ``N·L`` including cast shadows, which is why this is
EEVEE-only.  Because the result is emitted rather than lit, brightness is bounded
by the base texture and cannot blow out no matter how strong the key light is —
which is what fixed the washed-out faces.

Two decisions are taken from the model rather than guessed:

- a material with **no** toon ramp is left unlit.  Eyes, mouth interiors, teeth
  and brows are authored that way on every rig seen so far, and shading them is
  what makes anime eyes look dead.
- the shadow tint is sampled from the bottom of the model's own ramp, so a warm
  skin shadow stays warm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_TOON_MARK = "motionviewer_toon"
_OUTLINE_SUFFIX = "_outline"


@dataclass(frozen=True)
class ToonStyle:
    """Knobs for the cel look. Defaults aim at the source games' own grading."""

    # Terminator position and width in N·L. Lower puts more of the body in light.
    shadow_threshold: float = 0.42
    shadow_softness: float = 0.055
    # 0 keeps the model's ramp as-is, 1 doubles its contrast.
    shadow_depth: float = 0.55
    # Sphere-map specular, which the importer disables. 0 is off.
    sphere_strength: float = 0.55
    # Fresnel rim, the one addition not present in the source data. 0 is off.
    rim_strength: float = 0.18
    rim_width: float = 2.6
    outline_thickness_m: float = 0.0045
    # Fallback only. Each material's own ``edge_color`` is used where the PMX gives
    # one, which is what keeps the hair's edge a soft brown instead of hard black.
    outline_color: tuple[float, float, float] = (0.06, 0.045, 0.055)
    # Vertex rings around an open boundary that get zero shell thickness. Zero by
    # default: it sounds like the right cure for the shell showing through the eye and
    # mouth openings, but an MMD body is assembled from unwelded pieces — one surface
    # per material, hair built from separate cards — so 77% of Yoimiya's vertices sit
    # on an open boundary and 96% are within two rings. Tapering there does not trim
    # the artefact, it deletes the outline.
    outline_boundary_rings: int = 0
    # Multiplier on the authored edge colours. 1.0 is faithful — MMD draws the edge as
    # a flat unlit colour, which is what this does — but the values a PMX carries are
    # chosen against MMD's own grading, so lower it if the warm skin edge reads too
    # light against a pale background.
    outline_tint: float = 1.0
    # A face gets its own, much flatter terminator. Genshin drives face shadow
    # from an authored SDF map that a PMX rip does not carry, and a plain N·L
    # terminator cuts a hard line across the nose and eyes, which reads as a
    # blemish rather than as shading.
    face_shadow_threshold: float = 0.22
    face_shadow_softness: float = 0.30
    face_shadow_depth: float = 0.16


def _luminance_of_shading(tree: Any, nodes: Any, links: Any, x: float, y: float) -> Any:
    """``N·L`` with cast shadows, as a scalar."""
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.location = (x, y)
    diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    to_rgb = nodes.new("ShaderNodeShaderToRGB")
    to_rgb.location = (x + 180, y)
    links.new(diffuse.outputs["BSDF"], to_rgb.inputs["Shader"])
    # Rec.709 luminance keeps a coloured key light from biasing the terminator.
    luminance = nodes.new("ShaderNodeRGBToBW")
    luminance.location = (x + 360, y)
    links.new(to_rgb.outputs["Color"], luminance.inputs["Color"])
    return luminance.outputs["Val"]


def _sample_ramp_bottom(image: Any) -> tuple[float, float, float]:
    """Darkest colour of a toon ramp, which is its authored shadow tint."""
    if image is None or not image.has_data:
        return (0.62, 0.60, 0.66)
    width, height = image.size
    if width == 0 or height == 0:
        return (0.62, 0.60, 0.66)
    pixels = list(image.pixels[: width * height * 4])
    # Row 0 is the bottom of the image, and every ramp seen is a vertical
    # gradient with its shadow end there.
    row = pixels[: width * 4]
    count = max(width, 1)
    return (
        sum(row[0::4]) / count,
        sum(row[1::4]) / count,
        sum(row[2::4]) / count,
    )


def _srgb_to_linear(channel: float) -> float:
    """sRGB to linear, for colours read out of a PMX and fed to a shader node."""
    value = max(0.0, min(1.0, float(channel)))
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def face_base_images(materials: list[tuple[str, bool]]) -> set[str]:
    """Base textures that belong to the face, from ``(base_image, has_toon)``.

    A material authored without a toon ramp is one the artist wanted unlit, and
    on every rig seen so far those are the eyes, teeth and brows.  Whatever sheet
    they sample is the face texture, so anything else sampling it is face skin
    and needs the flatter terminator.  Reading it this way avoids matching
    material names, which differ per language and per rig.
    """
    return {image for image, has_toon in materials if image and not has_toon}


def _existing(nodes: Any, name: str) -> Any:
    node = nodes.get(name)
    return node if node is not None and getattr(node, "image", None) is not None else None


def apply_toon_shading(meshes: list[Any], *, style: ToonStyle | None = None) -> dict:
    """Rewire every MMD material on ``meshes`` into a cel-shaded graph."""
    import bpy  # type: ignore

    settings = style or ToonStyle()
    report: dict = {"shaded": [], "unlit": [], "skipped": [], "face": []}

    # Materials authored without a toon ramp are the unlit ones — eyes, teeth,
    # brows. Whatever texture they sample is the face sheet, so any material
    # sharing it belongs to the face and needs the flatter terminator. This
    # reads the intent off the model instead of matching bone or material names,
    # which differ per language and per rig.
    inventory: list[tuple[str, bool]] = []
    for mesh in meshes:
        for slot in mesh.material_slots:
            material = slot.material
            if material is None or not material.use_nodes:
                continue
            if getattr(material, "mmd_material", None) is None:
                continue
            base = _existing(material.node_tree.nodes, "mmd_base_tex")
            toon = _existing(material.node_tree.nodes, "mmd_toon_tex")
            if base is not None:
                inventory.append((base.image.name, toon is not None))
    face_images = face_base_images(inventory)

    seen: set[str] = set()
    for mesh in meshes:
        for slot in mesh.material_slots:
            material = slot.material
            if material is None or material.name in seen:
                continue
            seen.add(material.name)
            if getattr(material, "mmd_material", None) is None or not material.use_nodes:
                report["skipped"].append(material.name)
                continue

            tree = material.node_tree
            nodes, links = tree.nodes, tree.links
            base = _existing(nodes, "mmd_base_tex")
            toon = _existing(nodes, "mmd_toon_tex")
            sphere = _existing(nodes, "mmd_sphere_tex")
            uv = nodes.get("mmd_tex_uv")
            if base is None:
                report["skipped"].append(material.name)
                continue

            for node in [n for n in nodes if n.bl_idname == "ShaderNodeOutputMaterial"]:
                nodes.remove(node)
            for node in [n for n in nodes if n.name == "mmd_shader"]:
                nodes.remove(node)
            for node in [n for n in nodes if n.get(_TOON_MARK)]:
                nodes.remove(node)

            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (1200, 0)
            output[_TOON_MARK] = True

            colour = base.outputs["Color"]
            if toon is None:
                # Authored to be unlit: eyes, mouth, teeth, brows.
                report["unlit"].append(material.name)
            else:
                is_face = base.image is not None and base.image.name in face_images
                threshold = settings.face_shadow_threshold if is_face else settings.shadow_threshold
                softness = settings.face_shadow_softness if is_face else settings.shadow_softness
                depth = settings.face_shadow_depth if is_face else settings.shadow_depth
                if is_face:
                    report["face"].append(material.name)
                shading = _luminance_of_shading(tree, nodes, links, -600, -320)
                step = nodes.new("ShaderNodeMapRange")
                step.location = (-100, -320)
                step.inputs["From Min"].default_value = max(threshold - softness, 0.0)
                step.inputs["From Max"].default_value = threshold + softness
                step.inputs["To Min"].default_value = 0.0
                step.inputs["To Max"].default_value = 1.0
                step.clamp = True
                if hasattr(step, "interpolation_type"):
                    step.interpolation_type = "SMOOTHSTEP"
                links.new(shading, step.inputs["Value"])

                tint = _sample_ramp_bottom(toon.image)
                deepened = tuple(
                    max(0.0, channel - depth * (1.0 - channel) - 0.10 * depth) for channel in tint
                )
                shadow_mix = nodes.new("ShaderNodeMixRGB")
                shadow_mix.location = (150, -320)
                shadow_mix.blend_type = "MIX"
                shadow_mix.inputs["Color1"].default_value = (*deepened, 1.0)
                shadow_mix.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
                links.new(step.outputs["Result"], shadow_mix.inputs["Fac"])

                shaded = nodes.new("ShaderNodeMixRGB")
                shaded.location = (420, 0)
                shaded.blend_type = "MULTIPLY"
                shaded.inputs["Fac"].default_value = 1.0
                links.new(colour, shaded.inputs["Color1"])
                links.new(shadow_mix.outputs["Color"], shaded.inputs["Color2"])
                colour = shaded.outputs["Color"]
                report["shaded"].append(material.name)

            if sphere is not None and settings.sphere_strength > 0.0:
                # The importer leaves Sphere Tex Fac at 0; MMD type 2 is additive.
                gain = nodes.new("ShaderNodeMixRGB")
                gain.location = (420, -560)
                gain.blend_type = "MULTIPLY"
                gain.inputs["Fac"].default_value = 1.0
                gain.inputs["Color2"].default_value = (
                    settings.sphere_strength,
                    settings.sphere_strength,
                    settings.sphere_strength,
                    1.0,
                )
                links.new(sphere.outputs["Color"], gain.inputs["Color1"])
                add = nodes.new("ShaderNodeMixRGB")
                add.location = (640, -280)
                add.blend_type = "ADD"
                add.inputs["Fac"].default_value = 1.0
                links.new(colour, add.inputs["Color1"])
                links.new(gain.outputs["Color"], add.inputs["Color2"])
                colour = add.outputs["Color"]
                if uv is not None and "Sphere UV" in {s.name for s in uv.outputs}:
                    links.new(uv.outputs["Sphere UV"], sphere.inputs["Vector"])

            if settings.rim_strength > 0.0 and toon is not None:
                fresnel = nodes.new("ShaderNodeFresnel")
                fresnel.location = (420, -760)
                fresnel.inputs["IOR"].default_value = settings.rim_width
                rim = nodes.new("ShaderNodeMixRGB")
                rim.location = (860, -140)
                rim.blend_type = "ADD"
                rim.inputs["Color2"].default_value = (
                    settings.rim_strength,
                    settings.rim_strength,
                    settings.rim_strength,
                    1.0,
                )
                links.new(colour, rim.inputs["Color1"])
                links.new(fresnel.outputs["Fac"], rim.inputs["Fac"])
                colour = rim.outputs["Color"]

            emission = nodes.new("ShaderNodeEmission")
            emission.location = (1000, 0)
            emission.inputs["Strength"].default_value = 1.0
            links.new(colour, emission.inputs["Color"])

            surface = emission.outputs["Emission"]
            if str(getattr(material, "blend_method", "")) != "OPAQUE":
                transparent = nodes.new("ShaderNodeBsdfTransparent")
                transparent.location = (1000, -200)
                mix = nodes.new("ShaderNodeMixShader")
                mix.location = (1100, -100)
                links.new(base.outputs["Alpha"], mix.inputs["Fac"])
                links.new(transparent.outputs["BSDF"], mix.inputs[1])
                links.new(emission.outputs["Emission"], mix.inputs[2])
                surface = mix.outputs["Shader"]
            links.new(surface, output.inputs["Surface"])
            material[_TOON_MARK] = True

    _ = bpy  # imported for side effects on the current file only
    return report


def add_outline(
    meshes: list[Any],
    *,
    style: ToonStyle | None = None,
    respect_edge_flags: bool = True,
    exclude_materials: tuple[str, ...] | list[str] = (),
) -> list[Any]:
    """Inverted-hull outline: a slightly inflated, back-facing black shell.

    The shell is a copy of the character, so it keeps the armature modifier and
    deforms with the animation instead of drifting off the pose.

    **A PMX says per material whether it wants an edge, and that has to be obeyed.**
    On Yoimiya, 13 of 29 materials set ``enabled_toon_edge`` to false — the eyes,
    eye whites, eyelashes, brows, double eyelid, mouth, teeth, skirt. Outlining them
    anyway is visible and wrong: the shell shows through at the corners of the mouth
    as two near-black marks that read as fangs on a closed mouth, which is exactly
    the artefact the game does not have because the game respects the flag. Measured
    on a front-facing frame, dropping those faces takes the darkest pixel in the
    mouth region from 63/255 to 87/255 and halves the count of dark pixels.

    **The line is the colour the model asks for, not black.** A PMX carries an
    ``edge_color`` per material and they are not all the same: on Yoimiya the hair and
    nails ask for a warm brown ``(0.64, 0.37, 0.15)``, skin and face for
    ``(0.50, 0.25, 0.00)``, the head ornament for a reddish ``(0.50, 0.25, 0.25)``,
    and only the clothes for pure black. Painting one near-black line over all of them
    turns the hair's soft brown edge into a hard black one, which is the single
    biggest tell against the game's own look.

    **Thickness tapers to zero at open boundaries.** An inverted hull shows at every
    *opening* in a mesh, not only at the silhouette: the face mesh is open at the eyes
    and the mouth, and the inflated shell folds there and exposes its back faces —
    two dark marks at the corners of a closed mouth that read as fangs. Zeroing the
    shell weight on boundary vertices removes it exactly where it happens, which is
    what ``exclude_materials`` was doing far too bluntly: the obvious set to exclude,
    ``apply_toon_shading(...)["face"]``, also contains the *hair* on these rigs,
    because hair and face share one texture sheet, so it deleted the hair outline too.

    Faces are removed from the *shell*, not from the character, so the model's own
    shading is untouched. Set ``respect_edge_flags=False`` for the old behaviour.
    """
    import bmesh  # type: ignore
    import bpy  # type: ignore

    settings = style or ToonStyle()

    palette: dict[tuple[float, float, float], Any] = {}

    def outline_material(colour: tuple[float, float, float]) -> Any:
        existing = palette.get(colour)
        if existing is not None:
            return existing
        material = bpy.data.materials.new("MotionViewer_Outline")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*colour, 1.0)
        emission.inputs["Strength"].default_value = 1.0
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        material.use_backface_culling = True
        palette[colour] = material
        return material

    def authored_colour(slot_material: Any) -> tuple[float, float, float]:
        mmd = getattr(slot_material, "mmd_material", None)
        raw = getattr(mmd, "edge_color", None) if mmd is not None else None
        if raw is None or len(tuple(raw)) < 3:
            return settings.outline_color
        colour = tuple(float(channel) for channel in tuple(raw)[:3])
        if max(colour) <= 1e-6:
            # Pure black is what the clothes ask for; keep the style's near-black so
            # the line still reads as ink rather than as a hole in the image.
            return settings.outline_color
        # PMX stores the edge colour as sRGB; a node's default_value is linear. Fed in
        # raw, the skin's (0.50, 0.25, 0.00) emits as a bright gold line down the jaw
        # instead of the dark brown it is meant to be — sRGB 0.50 is linear 0.22.
        tint = max(0.0, settings.outline_tint)
        return tuple(_srgb_to_linear(channel) * tint for channel in colour)  # type: ignore[return-value]

    created: list[Any] = []
    dropped = 0
    tapered = 0
    for mesh in meshes:
        shell = mesh.copy()
        shell.data = mesh.data.copy()
        shell.name = f"{mesh.name}{_OUTLINE_SUFFIX}"
        for collection in mesh.users_collection:
            collection.objects.link(shell)

        if respect_edge_flags or exclude_materials:
            # Read the flags off the original slots, and delete the faces before the
            # material list is replaced: clearing it first would collapse every
            # material_index and lose the mapping.
            skip_names = set(exclude_materials)
            excluded = set()
            for index, slot_material in enumerate(shell.data.materials):
                if slot_material is None:
                    continue
                flagged_off = respect_edge_flags and not bool(
                    getattr(getattr(slot_material, "mmd_material", None), "enabled_toon_edge", True)
                )
                if flagged_off or slot_material.name in skip_names:
                    excluded.add(index)
            if excluded:
                bm = bmesh.new()
                bm.from_mesh(shell.data)
                faces = [face for face in bm.faces if face.material_index in excluded]
                dropped += len(faces)
                if faces:
                    bmesh.ops.delete(bm, geom=faces, context="FACES")
                bm.to_mesh(shell.data)
                bm.free()

        # One outline material per authored edge colour, swapped in slot for slot so
        # the existing material_index mapping keeps every face on the right colour.
        for index, slot_material in enumerate(shell.data.materials):
            if slot_material is not None:
                shell.data.materials[index] = outline_material(authored_colour(slot_material))

        solidify = shell.modifiers.new("Outline", "SOLIDIFY")
        solidify.thickness = settings.outline_thickness_m
        solidify.offset = 1.0
        solidify.use_flip_normals = True
        solidify.use_rim = False

        if settings.outline_boundary_rings > 0:
            # Optional and off by default; see the field's comment for why it is a trap
            # on this kind of mesh.
            group = shell.vertex_groups.new(name="MotionViewer_OutlineWeight")
            bm = bmesh.new()
            bm.from_mesh(shell.data)
            bm.verts.ensure_lookup_table()
            ring = {vertex.index for edge in bm.edges if edge.is_boundary for vertex in edge.verts}
            for _ in range(settings.outline_boundary_rings - 1):
                ring |= {
                    other.index
                    for index in list(ring)
                    for edge in bm.verts[index].link_edges
                    for other in edge.verts
                }
            bm.free()
            tapered += len(ring)
            interior = [v.index for v in shell.data.vertices if v.index not in ring]
            if interior:
                group.add(interior, 1.0, "REPLACE")
            if ring:
                group.add(sorted(ring), 0.0, "REPLACE")
            solidify.vertex_group = group.name
            # Zero weight means zero thickness, rather than the default full thickness.
            solidify.thickness_vertex_group = 0.0
        # Front faces are culled by the material, so only the inflated back
        # faces survive and they read as a line around the silhouette.
        created.append(shell)
    if dropped:
        print(f"outline: dropped {dropped} shell faces whose material asks for no edge")
    note = f", tapered at {tapered} boundary vertices" if tapered else ""
    print(f"outline: {len(palette)} authored edge colour(s){note}")
    return created


def add_toon_lighting(
    bounds_min: list[float],
    bounds_max: list[float],
    *,
    sun_strength: float = 3.2,
    ambient: float = 0.32,
    warm_key: bool = True,
) -> dict:
    """One directional key plus flat ambient, which is what cel shading wants.

    Three bright area lights give three overlapping terminators and wash the face
    out.  A single sun gives one clean shadow edge, and the ambient decides how
    dark the shadow side reads rather than the light rig doing it.
    """
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    scene = bpy.context.scene
    span = max(float(b) - float(a) for a, b in zip(bounds_min, bounds_max, strict=False))
    center = Vector(tuple((float(a) + float(b)) * 0.5 for a, b in zip(bounds_min, bounds_max, strict=False)))

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (
            ambient * 0.97,
            ambient * 0.98,
            ambient,
            1.0,
        )
        background.inputs["Strength"].default_value = 1.0

    light_data = bpy.data.lights.new("Toon_Key", type="SUN")
    light_data.energy = sun_strength
    # A wider sun softens both the body terminator and the cast shadow; a razor
    # edge on a stylised character reads as a rendering artifact.
    light_data.angle = 0.42
    if warm_key:
        light_data.color = (1.0, 0.97, 0.92)
    key = bpy.data.objects.new("Toon_Key", light_data)
    scene.collection.objects.link(key)
    key.location = center + Vector((-span * 0.8, -span * 1.1, span * 1.5))
    direction = center - key.location
    key.rotation_mode = "QUATERNION"
    key.rotation_quaternion = direction.to_track_quat("-Z", "Y")

    if hasattr(scene, "eevee"):
        scene.eevee.use_shadows = True
        if hasattr(scene.eevee, "shadow_ray_count"):
            scene.eevee.shadow_ray_count = 2
        if hasattr(scene.eevee, "shadow_step_count"):
            scene.eevee.shadow_step_count = 6
    scene.view_settings.view_transform = "Standard"
    return {"sun_strength": sun_strength, "ambient": ambient}


def add_ground(
    bounds_min: list[float],
    bounds_max: list[float],
    *,
    color: tuple[float, float, float] = (0.90, 0.89, 0.88),
    size_factor: float = 6.0,
    grid_metres: float = 0.0,
    grid_contrast: float = 0.045,
    plane_z: float | None = None,
) -> Any:
    """A floor that receives the character's shadow.

    Without it the character floats: a cast shadow is most of what tells a viewer
    where the feet are, and it is the cheapest cue that the motion has contact.

    ``plane_z`` is the floor height; it defaults to ``bounds_min[2]``, the lowest the
    character ever reaches. That default guarantees no foot ever intersects the
    floor, and also guarantees the character visibly hovers in every other frame:
    measured on a walk, the lowest mesh vertex ranges over 3.8–7.7 cm, so placing
    the floor at the minimum leaves a 2.1 cm median gap under the feet — about ten
    pixels at 800p, and enough with a soft shadow to read as floating. Passing a low
    percentile of the per-frame minimum instead lets planted frames make contact and
    trades the gap for a centimetre or two of intersection, which the contact shadow
    hides. Only the floor prop moves; no joint angle or root position is touched.

    ``grid_metres`` adds a world-fixed checker of that size. It is off by default
    and only wanted under a *follow* camera: on a featureless floor a tracking
    camera cancels the translation it is following, so a walk turns into a
    treadmill and the clip reads as marching in place. The contrast is deliberately
    tiny — a few percent of luminance — because the pattern only has to give the
    eye something stationary to measure against, and a strong checker would fight
    a cel-shaded character for attention.
    """
    import bpy  # type: ignore

    span = max(float(b) - float(a) for a, b in zip(bounds_min, bounds_max, strict=False))
    center_x = (float(bounds_min[0]) + float(bounds_max[0])) * 0.5
    center_y = (float(bounds_min[1]) + float(bounds_max[1])) * 0.5

    mesh = bpy.data.meshes.new("MotionViewer_Ground")
    half = max(span * size_factor, 4.0)
    mesh.from_pydata(
        [(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    ground = bpy.data.objects.new("MotionViewer_Ground", mesh)
    ground.location = (
        center_x,
        center_y,
        float(bounds_min[2]) if plane_z is None else float(plane_z),
    )
    bpy.context.scene.collection.objects.link(ground)

    material = bpy.data.materials.new("MotionViewer_Ground")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    if grid_metres > 0.0:
        # Generated coordinates span the mesh, so a checker keyed to them stays put
        # in world space while the camera moves over it.
        coords = tree.nodes.new("ShaderNodeTexCoord")
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (half * 2.0, half * 2.0, 1.0)
        checker = tree.nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = 1.0 / max(grid_metres, 1e-3)
        shade = max(0.0, min(1.0, 1.0 - grid_contrast))
        checker.inputs["Color1"].default_value = (*color, 1.0)
        checker.inputs["Color2"].default_value = (*(channel * shade for channel in color), 1.0)
        tree.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
        tree.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
        tree.links.new(checker.outputs["Color"], diffuse.inputs["Color"])
    else:
        diffuse.inputs["Color"].default_value = (*color, 1.0)
    tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    mesh.materials.append(material)
    ground.is_shadow_catcher = True if hasattr(ground, "is_shadow_catcher") else False
    return ground
