from __future__ import annotations

from typing import Any


def clear_scene() -> None:
    import bpy  # type: ignore

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_world(*, transparent: bool = False, background_rgb: tuple[int, int, int] | None = None) -> None:
    import bpy  # type: ignore

    scene = bpy.context.scene
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except (AttributeError, TypeError):
        pass
    bg = scene.world.node_tree.nodes.get("Background")
    if transparent:
        if bg:
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = 0.35
    else:
        rgb = background_rgb or (250, 248, 242)
        linear = tuple(_srgb_to_linear(value / 255.0) for value in rgb)
        scene.world.color = linear
        if bg:
            bg.inputs["Color"].default_value = (*linear, 1.0)
            bg.inputs["Strength"].default_value = 1.0
    scene.render.film_transparent = transparent


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def add_lighting(bounds_min: list[float] | None = None, bounds_max: list[float] | None = None) -> None:
    import bpy  # type: ignore

    if bounds_min is None or bounds_max is None:
        center = (0.0, 0.0, 0.0)
        span = 4.0
    else:
        center = tuple((float(a) + float(b)) * 0.5 for a, b in zip(bounds_min, bounds_max))
        span = max(float(b) - float(a) for a, b in zip(bounds_min, bounds_max))
    height = max(span * 0.85, 4.0)
    size = max(span * 0.75, 5.0)

    bpy.ops.object.light_add(type="AREA", location=(center[0], center[1] - span * 0.35, center[2] + height))
    key = bpy.context.object
    key.name = "Key_Light"
    key.data.energy = max(900, span * 120)
    key.data.size = size
    _aim_at(key, center)

    bpy.ops.object.light_add(
        type="AREA", location=(center[0] - span * 0.45, center[1] + span * 0.35, center[2] + height * 0.75)
    )
    fill = bpy.context.object
    fill.name = "Fill_Light"
    fill.data.energy = max(420, span * 60)
    fill.data.size = size * 0.8
    _aim_at(fill, center)

    bpy.ops.object.light_add(
        type="AREA", location=(center[0] + span * 0.35, center[1] + span * 0.15, center[2] + height * 0.6)
    )
    rim = bpy.context.object
    rim.name = "Rim_Light"
    rim.data.energy = max(280, span * 40)
    rim.data.size = size * 0.7
    _aim_at(rim, center)


def _aim_at(obj: Any, target: tuple[float, float, float]) -> None:
    """Point a Blender light's local -Z axis at a world-space target."""
    from mathutils import Vector  # type: ignore

    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_world_label(text: str, location: tuple[float, float, float], camera: Any | None = None) -> Any:
    import bpy  # type: ignore

    rotation = camera.rotation_euler if camera is not None else (1.2, 0.0, 0.0)
    bpy.ops.object.text_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = f"MotionViewer_Label_{text}"
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = 0.18
    return obj
