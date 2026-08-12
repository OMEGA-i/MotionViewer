from __future__ import annotations

from ..core.palette import Color, temporal_color


def make_material(name: str, color: Color, *, roughness: float = 0.55, flat: bool = False):
    import bpy  # type: ignore

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if flat:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = color.rgba()
        emission.inputs["Strength"].default_value = 1.0
        if color.a < 1.0:
            transparent = nodes.new("ShaderNodeBsdfTransparent")
            mix = nodes.new("ShaderNodeMixShader")
            mix.inputs[0].default_value = 1.0 - color.a
            links.new(emission.outputs[0], mix.inputs[1])
            links.new(transparent.outputs[0], mix.inputs[2])
            links.new(mix.outputs[0], output.inputs[0])
            mat.blend_method = "BLEND"
            if hasattr(mat, "shadow_method"):
                mat.shadow_method = "NONE"
        else:
            links.new(emission.outputs[0], output.inputs[0])
        return mat
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color.rgba()
        bsdf.inputs["Roughness"].default_value = roughness
        if color.a < 1.0:
            bsdf.inputs["Alpha"].default_value = color.a
            mat.blend_method = "BLEND"
            if hasattr(mat, "surface_render_method"):
                mat.surface_render_method = "BLENDED"
            if hasattr(mat, "shadow_method"):
                mat.shadow_method = "NONE"
    return mat


def apply_actor_material(actor, color: Color, *, roughness: float = 0.55) -> None:
    mat = make_material(f"{actor.label}_Material", color, roughness=roughness)
    for mesh in actor.mesh_objects:
        mesh.data.materials.clear()
        mesh.data.materials.append(mat)


def set_actor_visibility(
    actor, *, visible_from: int, visible_until: int, frame_start: int, frame_end: int
) -> None:
    """Keyframe actor visibility for an inclusive frame range."""

    objects = [actor.armature, *actor.mesh_objects]
    visible_from = max(frame_start, visible_from)
    visible_until = min(frame_end, visible_until)
    for obj in objects:
        if visible_from > frame_start:
            obj.hide_render = True
            obj.hide_viewport = True
            obj.keyframe_insert(data_path="hide_render", frame=frame_start)
            obj.keyframe_insert(data_path="hide_viewport", frame=frame_start)
            obj.keyframe_insert(data_path="hide_render", frame=visible_from - 1)
            obj.keyframe_insert(data_path="hide_viewport", frame=visible_from - 1)
        obj.hide_render = False
        obj.hide_viewport = False
        obj.keyframe_insert(data_path="hide_render", frame=visible_from)
        obj.keyframe_insert(data_path="hide_viewport", frame=visible_from)
        obj.keyframe_insert(data_path="hide_render", frame=visible_until)
        obj.keyframe_insert(data_path="hide_viewport", frame=visible_until)
        if visible_until < frame_end:
            obj.hide_render = True
            obj.hide_viewport = True
            obj.keyframe_insert(data_path="hide_render", frame=visible_until + 1)
            obj.keyframe_insert(data_path="hide_viewport", frame=visible_until + 1)


def add_freeze_fade(
    actor,
    *,
    freeze_frame: int,
    frame_end: int,
    fade_frames: int = 10,
    fade_alpha: float = 0.25,
) -> None:
    """Dim a held actor once its own motion data runs out before the shared timeline does.

    Blender F-curves hold the last keyframed pose flat by default (constant extrapolation),
    so an actor whose source sequence is shorter than the render's frame count simply freezes
    on its last real pose. This keyframes material alpha down so viewers can tell "this model's
    generation ended here" apart from "this model is still moving".
    """
    if freeze_frame >= frame_end or not actor.mesh_objects:
        return

    fade_end = min(frame_end, freeze_frame + max(1, fade_frames))
    materials = {mat for mesh in actor.mesh_objects for mat in mesh.data.materials if mat is not None}
    for mat in materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            continue
        mat.blend_method = "BLEND"
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = "NONE"
        alpha_input = bsdf.inputs["Alpha"]
        alpha_input.default_value = 1.0
        alpha_input.keyframe_insert(data_path="default_value", frame=freeze_frame)
        alpha_input.default_value = fade_alpha
        alpha_input.keyframe_insert(data_path="default_value", frame=fade_end)
        _set_fcurves_linear(mat.node_tree)


def _set_fcurves_linear(node_tree) -> None:
    """Best-effort linear interpolation; Blender's Action fcurve access has moved across
    versions (legacy `action.fcurves` vs. layered actions), so this tolerates either."""
    anim_data = getattr(node_tree, "animation_data", None)
    action = getattr(anim_data, "action", None) if anim_data else None
    if action is None:
        return
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves = getattr(channelbag, "fcurves", None)
                    if fcurves:
                        _linearize(fcurves)
        return
    _linearize(fcurves)


def _linearize(fcurves) -> None:
    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def add_ghost_snapshots(
    actor,
    *,
    frame_start: int,
    frame_end: int,
    count: int,
    base_color: Color,
    ramp: str = "light_to_dark",
    ghost_spec: dict | None = None,
) -> None:
    if count <= 0 or not actor.mesh_objects or frame_end <= frame_start:
        return
    import bpy  # type: ignore

    spec = ghost_spec or {}
    alpha = float(spec.get("alpha", 0.18))
    mode = str(spec.get("mode", "trail"))
    start_lightness = float(spec.get("start_lightness", 0.55))
    end_lightness = float(spec.get("end_lightness", 0.0))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frames = _sample_frames(frame_start, frame_end, count)
    for idx, frame in enumerate(frames):
        t = idx / max(1, len(frames) - 1)
        color = temporal_color(
            Color(base_color.r, base_color.g, base_color.b, alpha),
            t,
            ramp,
            start_lightness=start_lightness,
            end_lightness=end_lightness,
        )
        mat = make_material(f"{actor.label}_Ghost_{idx:03d}_Mat", color, roughness=0.92)
        bpy.context.scene.frame_set(frame)
        for mesh in actor.mesh_objects:
            evaluated = mesh.evaluated_get(depsgraph)
            snapshot_data = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
            snapshot = bpy.data.objects.new(f"{actor.label}_Ghost_{frame:04d}", snapshot_data)
            snapshot.matrix_world = evaluated.matrix_world.copy()
            bpy.context.collection.objects.link(snapshot)
            snapshot.data.materials.append(mat)
            if mode == "trail":
                snapshot.hide_render = True
                snapshot.keyframe_insert(data_path="hide_render", frame=max(1, frame - 1))
                snapshot.hide_render = False
                snapshot.keyframe_insert(data_path="hide_render", frame=frame)


def add_prefix_ghost_snapshots(
    actor,
    *,
    frame_start: int,
    prefix_end_frame: int,
    count: int,
    prefix_color: Color,
) -> None:
    if count <= 0 or prefix_end_frame < frame_start or not actor.mesh_objects:
        return
    sample_start = prefix_end_frame if count == 1 else frame_start
    add_ghost_snapshots(
        actor,
        frame_start=sample_start,
        frame_end=prefix_end_frame,
        count=count,
        base_color=Color(prefix_color.r, prefix_color.g, prefix_color.b, 0.20),
        ramp="light_to_dark",
        ghost_spec={"alpha": 0.20, "start_lightness": 0.72, "end_lightness": 0.45},
    )


def _sample_frames(frame_start: int, frame_end: int, count: int) -> list[int]:
    if count <= 1:
        return [frame_start]
    step = (frame_end - frame_start) / (count - 1)
    return [int(round(frame_start + idx * step)) for idx in range(count)]
