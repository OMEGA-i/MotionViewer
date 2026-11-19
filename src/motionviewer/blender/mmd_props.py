"""Hand props and scene sets, built from code rather than imported as assets.

A caption like "playing table tennis" is carried mostly by the paddle and the table.
Without them the retarget reads as someone swatting at nothing, which is a worse
failure than any joint error: the viewer does not think "the wrist is 3 cm off", they
think "this is not table tennis".

Everything here is procedural for three reasons. Licensed character assets already
constrain what may be redistributed, and generated geometry adds nothing to that
list. Dimensions come from the sport's own regulations, so a table is the right size
by construction instead of by eye. And the meshes go through the same
``apply_toon_shading``/``add_outline`` the character does, so a prop does not sit in
the frame looking like it came from a different renderer.

What this module cannot do is know where the equipment was. The motion carries the
body and nothing else: no table, no ball, no racket. Placement is therefore derived
from the character — heading, hand position, reach — and then verified by rendering.
Any scene here is a plausible reconstruction, not a measurement, and the ball
keyframes in particular are authored to match the swing rather than simulated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .mmd_box_support import build_supported_box_scene

# Regulation dimensions, in metres. Kept as named constants so a wrong prop is a
# wrong number here rather than a wrong-looking render nobody can account for.
TABLE_TENNIS = {
    "length": 2.74,
    "width": 1.525,
    "height": 0.76,
    "net_height": 0.1525,
    "net_overhang": 0.1525,
}
BADMINTON = {
    "width": 6.10,
    "length": 13.40,
    "net_height_centre": 1.524,
    "net_height_post": 1.55,
    "net_depth": 0.79,
}
BASKETBALL_DIAMETER = 0.24
SHUTTLE_LENGTH = 0.085
TT_BALL_DIAMETER = 0.04


@dataclass
class PropStyle:
    """Flat colours, because the toon shader quantises to a lit and an unlit band."""

    wood: tuple[float, float, float] = (0.16, 0.30, 0.52)
    table_line: tuple[float, float, float] = (0.92, 0.92, 0.94)
    net: tuple[float, float, float] = (0.10, 0.10, 0.12)
    metal: tuple[float, float, float] = (0.62, 0.63, 0.66)
    rubber: tuple[float, float, float] = (0.45, 0.06, 0.07)
    handle: tuple[float, float, float] = (0.35, 0.24, 0.16)
    cardboard: tuple[float, float, float] = (0.66, 0.50, 0.33)
    cardboard_tape: tuple[float, float, float] = (0.78, 0.70, 0.55)
    ball_orange: tuple[float, float, float] = (0.78, 0.33, 0.09)
    ball_white: tuple[float, float, float] = (0.94, 0.93, 0.88)
    shuttle: tuple[float, float, float] = (0.95, 0.95, 0.93)
    court: tuple[float, float, float] = (0.24, 0.42, 0.35)


@dataclass
class Prop:
    """One built object plus how it should follow the rig, if at all."""

    obj: Any
    bone: str | None = None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Slide toward this second bone by ``aim_fraction`` of the gap between them,
    #: measured off the rig rather than hardcoded. A two-handed carry wants 0.5, which
    #: puts the prop centre exactly between the wrists; the arms' spacing was measured
    #: at 357 +/- 48 mm over the carry clip, so one wrist plus half the gap tracks the
    #: hands far better than parenting to the torso, against which the same midpoint
    #: wanders over 0.42 x 0.47 m.
    aim_bone: str | None = None
    aim_fraction: float = 0.5
    #: Turn the prop so its own +Z runs from ``align_from`` to ``align_to``. A racket
    #: continues the forearm, and the Euler angles that express that depend on the
    #: rig's bone roll: hardcoding -95 degrees about X left the racket standing
    #: vertically while the measured forearm pointed [-0.81, -0.35, +0.48], mostly down
    #: and to the side. Reading the direction off the rig removes the guess.
    align_from: str | None = None
    align_to: str | None = None
    #: A final nudge in *world* axes, applied after ``aim_bone``. Needed for "hangs
    #: below the hands": a carried box is gripped near its top edge, so centring it on
    #: the wrists puts half the box above them — and this clip holds the hands at 80%
    #: of standing height, where that half lands over the face. Which bone axis points
    #: down depends on how the arm is posed, so this one has to be world-space.
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SceneBuild:
    """Everything a scene preset produced, for the caller to shade and report."""

    props: list[Prop] = field(default_factory=list)
    static: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    wants_static_camera: bool = False
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def objects(self) -> list[Any]:
        return [prop.obj for prop in self.props] + list(self.static)


def _material(bpy: Any, name: str, colour: tuple[float, float, float]) -> Any:
    """A plain diffuse material; ``apply_toon_shading`` rewrites it into bands."""
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            node.inputs["Base Color"].default_value = (*colour, 1.0)
            if "Roughness" in node.inputs:
                node.inputs["Roughness"].default_value = 0.6
            if "Specular IOR Level" in node.inputs:
                node.inputs["Specular IOR Level"].default_value = 0.2
    return material


def _cube(
    bpy: Any,
    name: str,
    size: tuple[float, float, float],
    centre: tuple[float, float, float],
    colour: tuple[float, float, float],
) -> Any:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=centre)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(_material(bpy, f"prop_{name}", colour))
    return obj


def _cylinder(
    bpy: Any,
    name: str,
    radius: float,
    depth: float,
    centre: tuple[float, float, float],
    colour: tuple[float, float, float],
    *,
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 24,
) -> Any:
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, location=centre, vertices=vertices, rotation=euler
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(_material(bpy, f"prop_{name}", colour))
    return obj


def _net_panel(
    bpy: Any,
    name: str,
    width: float,
    height: float,
    centre: tuple[float, float, float],
    colour: tuple[float, float, float],
    *,
    cell: float = 0.05,
    cord: float = 0.004,
) -> Any:
    """A see-through net: a grid of thin cords, not a slab.

    The first version of this was a solid box the size of the net, and that is why the
    badminton render came out with a black wall across the frame. A real net is mostly
    holes — at 5 cm cells and 4 mm cord it is about 85% open — and reading through it is
    what tells the eye it is a net rather than a barrier. Built as geometry rather than
    an alpha texture because the toon shader quantises lighting and does not carry
    alpha through the ShaderToRGB path.
    """
    bars = []
    columns = max(2, int(round(width / cell)))
    rows = max(2, int(round(height / cell)))
    for index in range(columns + 1):
        x = -width / 2.0 + index * (width / columns)
        bars.append(_cube(bpy, f"{name}_v{index}", (cord, cord, height), (0.0, x, 0.0), colour))
    for index in range(rows + 1):
        z = -height / 2.0 + index * (height / rows)
        bars.append(_cube(bpy, f"{name}_h{index}", (cord, width, cord), (0.0, 0.0, z), colour))
    panel = _join(bpy, bars, name)
    panel.location = centre
    return panel


def _sphere(
    bpy: Any,
    name: str,
    radius: float,
    centre: tuple[float, float, float],
    colour: tuple[float, float, float],
    *,
    segments: int = 20,
) -> Any:
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, location=centre, segments=segments, ring_count=segments // 2
    )
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(_material(bpy, f"prop_{name}", colour))
    return obj


# ---------------------------------------------------------------------------
# hand props
# ---------------------------------------------------------------------------


def build_cardboard_box(
    bpy: Any, style: PropStyle, *, side: float = 0.34, depth: float = 0.30, height: float = 0.30
) -> Any:
    """A closed crate, sized independently on each axis.

    Closed rather than open-topped: this is carried in front of the chest at a camera
    that looks slightly down, so an open box shows its empty interior and reads as a
    prop nobody is really holding. The lid seam and the corner battens are what make it
    legible as a container at this size, where a plain cube could be a block or a
    rendering error.
    """
    hw, hd, hh = side / 2.0, depth / 2.0, height / 2.0
    wall = 0.012
    parts = [
        _cube(bpy, "crate_top", (side, depth, wall), (0.0, 0.0, hh), style.cardboard),
        _cube(bpy, "crate_base", (side, depth, wall), (0.0, 0.0, -hh), style.cardboard),
        _cube(bpy, "crate_x+", (wall, depth, height), (hw, 0.0, 0.0), style.cardboard),
        _cube(bpy, "crate_x-", (wall, depth, height), (-hw, 0.0, 0.0), style.cardboard),
        _cube(bpy, "crate_y+", (side, wall, height), (0.0, hd, 0.0), style.cardboard),
        _cube(bpy, "crate_y-", (side, wall, height), (0.0, -hd, 0.0), style.cardboard),
        # A seam across the lid, and battens down the front corners.
        _cube(bpy, "crate_seam", (side * 1.004, 0.05, wall * 1.4), (0.0, 0.0, hh), style.cardboard_tape),
        _cube(
            bpy, "crate_batten_l", (0.035, wall * 1.6, height), (-hw + 0.03, -hd, 0.0), style.cardboard_tape
        ),
        _cube(
            bpy, "crate_batten_r", (0.035, wall * 1.6, height), (hw - 0.03, -hd, 0.0), style.cardboard_tape
        ),
    ]
    return _join(bpy, parts, "prop_box")


def build_table_tennis_paddle(bpy: Any, style: PropStyle) -> Any:
    """Blade, rubber face and handle. Dimensions are the common 15 x 15 cm blade."""
    blade = _cylinder(bpy, "paddle_blade", 0.075, 0.006, (0.0, 0.0, 0.085), style.handle, vertices=32)
    face = _cylinder(bpy, "paddle_face", 0.073, 0.0035, (0.0, 0.0045, 0.085), style.rubber, vertices=32)
    face.rotation_euler = (math.pi / 2, 0.0, 0.0)
    blade.rotation_euler = (math.pi / 2, 0.0, 0.0)
    handle = _cube(bpy, "paddle_grip", (0.026, 0.018, 0.10), (0.0, 0.0, 0.0), style.handle)
    return _join(bpy, [blade, face, handle], "prop_paddle")


def build_badminton_racket(bpy: Any, style: PropStyle) -> Any:
    """A 67 cm racket: oval head ring, throat, shaft, grip.

    The head is a ring rather than a disc — a badminton racket read as a solid paddle
    is the single most obvious way to get this prop wrong.
    """
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.115,
        minor_radius=0.006,
        location=(0.0, 0.0, 0.55),
        major_segments=32,
        minor_segments=8,
        rotation=(math.pi / 2, 0.0, 0.0),
    )
    head = bpy.context.active_object
    head.name = "racket_head"
    head.scale = (0.88, 1.0, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    head.data.materials.append(_material(bpy, "prop_racket_head", style.metal))

    strings = _cylinder(bpy, "racket_strings", 0.108, 0.001, (0.0, 0.0, 0.55), style.ball_white, vertices=24)
    strings.rotation_euler = (math.pi / 2, 0.0, 0.0)
    strings.scale = (0.88, 1.0, 1.0)
    shaft = _cylinder(bpy, "racket_shaft", 0.007, 0.30, (0.0, 0.0, 0.285), style.metal, vertices=12)
    grip = _cylinder(bpy, "racket_grip", 0.014, 0.13, (0.0, 0.0, 0.065), style.handle, vertices=12)
    return _join(bpy, [head, strings, shaft, grip], "prop_racket")


def build_basketball(bpy: Any, style: PropStyle) -> Any:
    """A sphere plus two seam bands; without seams it reads as a beach ball."""
    radius = BASKETBALL_DIAMETER / 2.0
    ball = _sphere(bpy, "ball_body", radius, (0.0, 0.0, 0.0), style.ball_orange, segments=28)
    seams = []
    for index, euler in enumerate(((0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0))):
        bpy.ops.mesh.primitive_torus_add(
            major_radius=radius,
            minor_radius=0.0035,
            location=(0.0, 0.0, 0.0),
            major_segments=28,
            minor_segments=6,
            rotation=euler,
        )
        seam = bpy.context.active_object
        seam.name = f"ball_seam{index}"
        seam.data.materials.append(_material(bpy, "prop_ball_seam", (0.09, 0.05, 0.03)))
        seams.append(seam)
    return _join(bpy, [ball, *seams], "prop_basketball")


def build_shuttlecock(bpy: Any, style: PropStyle) -> Any:
    cork = _sphere(bpy, "shuttle_cork", 0.013, (0.0, 0.0, 0.0), style.ball_white, segments=12)
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.013,
        radius2=0.033,
        depth=SHUTTLE_LENGTH * 0.75,
        location=(0.0, 0.0, SHUTTLE_LENGTH * 0.42),
        vertices=16,
    )
    skirt = bpy.context.active_object
    skirt.name = "shuttle_skirt"
    skirt.data.materials.append(_material(bpy, "prop_shuttle_skirt", style.shuttle))
    return _join(bpy, [cork, skirt], "prop_shuttlecock")


def build_table_tennis_ball(bpy: Any, style: PropStyle) -> Any:
    return _sphere(
        bpy, "prop_tt_ball", TT_BALL_DIAMETER / 2.0, (0.0, 0.0, 0.0), style.ball_white, segments=14
    )


def _join(bpy: Any, parts: list[Any], name: str) -> Any:
    """Merge parts into one object so a prop is a single thing to parent and shade."""
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    if len(parts) > 1:
        bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = name
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    return joined


# ---------------------------------------------------------------------------
# attaching to the rig
# ---------------------------------------------------------------------------


def attach_to_bone(bpy: Any, prop: Prop, armature: Any) -> bool:
    """Parent a prop to a bone so it follows the animation.

    Bone parenting is used rather than a Child Of constraint because the rig is baked
    to keyframes: a constraint would need the depsgraph evaluated in the right order
    on every frame, while a bone parent is resolved by the same armature evaluation
    that already runs.

    Returns False when the bone is absent, which happens on rigs whose finger or
    twist bones are named differently — the caller reports it rather than rendering a
    prop that silently sits at the origin.
    """
    if prop.bone is None:
        return True
    if prop.bone not in armature.data.bones:
        return False
    bone = armature.data.bones[prop.bone]
    obj = prop.obj
    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = prop.bone
    # Measured, because the obvious guess is wrong and fails quietly. With an identity
    # parent inverse, location (0,0,0) lands on the bone's *tail* and the local axes
    # are the bone's own: +Y runs head->tail along the bone, and +Z is world -Y at
    # rest. Setting the inverse to ``bone.matrix_local.inverted()`` instead — the usual
    # keep-it-in-place idiom for object parents — drops the bone's world position and
    # leaves the prop near the origin, which renders as a box floating in the corner
    # rather than as an error.
    from mathutils import Matrix, Vector  # Blender-only; never imported outside it.

    obj.matrix_parent_inverse = Matrix.Identity(4)
    # ``offset`` is given wrist-relative, so undo the head->tail run first: a hand prop
    # belongs where the hand is, not a bone length past it.
    obj.location = (
        prop.offset[0],
        prop.offset[1] - bone.length,
        prop.offset[2],
    )
    obj.rotation_euler = prop.euler
    if prop.align_from and prop.align_to:
        bones = armature.pose.bones
        if prop.align_from in bones and prop.align_to in bones:
            bpy.context.view_layer.update()
            start = armature.matrix_world @ bones[prop.align_from].head
            end = armature.matrix_world @ bones[prop.align_to].head
            direction = end - start
            if direction.length > 1e-6:
                # The prop is modelled with its handle at the origin and its length up
                # +Z, so pointing +Z along the limb is the whole alignment.
                world = obj.matrix_world.copy()
                where = world.translation.copy()
                world = direction.to_track_quat("Z", "Y").to_matrix().to_4x4()
                world.translation = where
                obj.matrix_world = world

    shift = None
    if prop.aim_bone and prop.aim_bone in armature.data.bones:
        # Slide toward the other bone in *world* space and let the parent relationship
        # convert it back. Working out which local axis points at the other hand would
        # mean encoding this rig's bone roll here, and MMD rigs do not agree on it.
        bpy.context.view_layer.update()
        own = armature.matrix_world @ armature.pose.bones[prop.bone].head
        other = armature.matrix_world @ armature.pose.bones[prop.aim_bone].head
        shift = (other - own) * prop.aim_fraction
    if any(prop.world_offset):
        extra = Vector(prop.world_offset)
        shift = extra if shift is None else shift + extra
    if shift is not None:
        bpy.context.view_layer.update()
        world = obj.matrix_world.copy()
        world.translation = world.translation + shift
        obj.matrix_world = world
    return True


# ---------------------------------------------------------------------------
# scene presets
# ---------------------------------------------------------------------------


def _place_flat(obj: Any, centre: tuple[float, float], z: float, heading: tuple[float, float]) -> None:
    """Put an object at a world spot, yawed to face along ``heading``."""
    obj.location = (centre[0], centre[1], z)
    obj.rotation_euler = (0.0, 0.0, math.atan2(heading[1], heading[0]))


def build_table_tennis_scene(
    bpy: Any,
    style: PropStyle,
    *,
    origin: tuple[float, float],
    heading: tuple[float, float],
    floor_z: float,
    hand_bone: str,
    stand_off: float = 0.55,
) -> SceneBuild:
    """A regulation table in front of the player, plus a paddle in hand.

    ``stand_off`` is the gap from the player to the near edge. It is a guess that has
    to be checked in a render: the motion says where the body is, never where the
    table was.
    """
    spec = TABLE_TENNIS
    forward = (heading[0], heading[1])
    # Table centre sits half a table length beyond the near edge.
    reach = stand_off + spec["length"] / 2.0
    centre = (origin[0] + forward[0] * reach, origin[1] + forward[1] * reach)

    top = _cube(
        bpy,
        "tt_top",
        (spec["length"], spec["width"], 0.022),
        (0.0, 0.0, spec["height"]),
        style.wood,
    )
    lines = [
        _cube(
            bpy,
            "tt_edge_line",
            (spec["length"], spec["width"], 0.001),
            (0.0, 0.0, spec["height"] + 0.012),
            style.table_line,
        )
    ]
    lines[0].scale = (1.0, 1.0, 1.0)
    centre_line = _cube(
        bpy,
        "tt_centre_line",
        (spec["length"], 0.003, 0.001),
        (0.0, 0.0, spec["height"] + 0.0135),
        style.table_line,
    )
    net = _net_panel(
        bpy,
        "tt_net",
        spec["width"] + 2 * spec["net_overhang"],
        spec["net_height"],
        (0.0, 0.0, spec["height"] + spec["net_height"] / 2.0),
        style.net,
        cell=0.025,
        cord=0.003,
    )
    legs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            legs.append(
                _cube(
                    bpy,
                    f"tt_leg_{sx}{sy}",
                    (0.05, 0.05, spec["height"]),
                    (
                        sx * (spec["length"] / 2.0 - 0.22),
                        sy * (spec["width"] / 2.0 - 0.12),
                        spec["height"] / 2.0,
                    ),
                    style.metal,
                )
            )
    table = _join(bpy, [top, *lines, centre_line, net, *legs], "set_table_tennis")
    _place_flat(table, centre, floor_z, forward)

    elbow = "右ひじ" if hand_bone.startswith("右") else "左ひじ"
    paddle = Prop(
        obj=build_table_tennis_paddle(bpy, style),
        bone=hand_bone,
        align_from=elbow,
        align_to=hand_bone,
    )
    return SceneBuild(
        props=[paddle],
        static=[table],
        wants_static_camera=True,
        notes=[
            f"table centre {centre[0]:+.2f},{centre[1]:+.2f} m, {stand_off:.2f} m off the player",
            "table position is inferred from heading, not present in the motion",
        ],
    )


def build_badminton_scene(
    bpy: Any,
    style: PropStyle,
    *,
    origin: tuple[float, float],
    heading: tuple[float, float],
    floor_z: float,
    hand_bone: str,
    stand_off: float = 2.2,
    visible_width: float = 3.6,
) -> SceneBuild:
    """Net, posts and the court lines nearest the player, plus a racket in hand.

    Only the near half of the court is built. The full 13.4 m would put the far
    tramlines outside any framing that keeps the character readable, and every metre
    of it costs the same shading and outline work as the part on screen.
    """
    spec = BADMINTON
    forward = (heading[0], heading[1])
    lateral = (-forward[1], forward[0])
    net_at = (origin[0] + forward[0] * stand_off, origin[1] + forward[1] * stand_off)

    posts = [
        _cylinder(
            bpy,
            f"bm_post_{side}",
            0.02,
            spec["net_height_post"],
            (0.0, side * min(spec["width"], visible_width) / 2.0, spec["net_height_post"] / 2.0),
            style.metal,
            vertices=12,
        )
        for side in (-1, 1)
    ]
    # Only the middle of the net is built. The regulation 6.10 m spans far wider than
    # any framing that keeps the character readable, and the part outside the frame
    # costs the same shading and outline work as the part inside it.
    net_width = min(spec["width"], visible_width)
    mesh = _net_panel(
        bpy,
        "bm_net",
        net_width,
        spec["net_depth"],
        (0.0, 0.0, spec["net_height_centre"] - spec["net_depth"] / 2.0),
        style.net,
    )
    tape = _cube(
        bpy,
        "bm_net_tape",
        (0.010, net_width, 0.038),
        (0.0, 0.0, spec["net_height_centre"]),
        style.ball_white,
    )
    net = _join(bpy, [*posts, mesh, tape], "set_badminton_net")
    _place_flat(net, net_at, floor_z, forward)

    # Court markings on the player's side: side lines, short service line, centre.
    marks = []
    half = spec["length"] / 4.0  # near half of the near court, enough to fill frame
    for side in (-1, 1):
        marks.append(
            _cube(
                bpy,
                f"bm_side_{side}",
                (half, 0.04, 0.001),
                (-half / 2.0, side * spec["width"] / 2.0, 0.0),
                style.table_line,
            )
        )
    marks.append(_cube(bpy, "bm_service", (0.04, spec["width"], 0.001), (-1.98, 0.0, 0.0), style.table_line))
    marks.append(_cube(bpy, "bm_centre", (half, 0.04, 0.001), (-half / 2.0, 0.0, 0.0), style.table_line))
    court = _join(bpy, marks, "set_badminton_court")
    _place_flat(court, net_at, floor_z + 0.002, forward)

    elbow = "右ひじ" if hand_bone.startswith("右") else "左ひじ"
    racket = Prop(
        obj=build_badminton_racket(bpy, style),
        bone=hand_bone,
        align_from=elbow,
        align_to=hand_bone,
    )
    del lateral
    return SceneBuild(
        props=[racket],
        static=[net, court],
        wants_static_camera=True,
        notes=[
            f"net {stand_off:.2f} m in front of the player, near half-court only",
            "court position is inferred from heading, not present in the motion",
        ],
    )


def build_carry_box_scene(
    bpy: Any,
    style: PropStyle,
    *,
    hand_bone: str,
    other_bone: str | None = "右手首",
    side: float = 0.34,
    depth: float = 0.30,
    height: float = 0.30,
    forward: tuple[float, float] = (0.0, -1.0),
    push: float = 0.10,
) -> SceneBuild:
    """A crate held between both hands, in front of them rather than under them.

    Sizing and placement are measured off the clip rather than guessed. Over the carry
    section of 016783 the two wrists sit 366 mm apart, 0.33 m in front of the chest and
    1.31 m off the floor, level to within 4 cm — and the palm normals point *forward*
    (dot 0.81 and 0.55 with the body's facing) while the forearms angle up (0.68). So the
    object is in front of the hands, not hanging below them.

    An earlier version hung it below, which is what a box carried at waist height would
    do, and the forearms — which come up from the elbows at that angle — went straight
    through it. Pushing it forward instead clears the torso by 17 cm and the chin by
    9 cm, with the hands just outside its sides.
    """
    # Lift the crate by half its height so its *base* rests on the hands: a load carried
    # in both arms sits on top of them. Two earlier versions had it centred on the wrists
    # (half the crate below them, so the forearms passed through it) and then hanging
    # below them (which reads as the box floating under the hands rather than being
    # held). Palm normals were measured pointing forward and slightly down, which is why
    # the second version looked plausible on paper; on screen a carried crate has to be
    # supported from underneath.
    box = Prop(
        obj=build_cardboard_box(bpy, style, side=side, depth=depth, height=height),
        bone=hand_bone,
        aim_bone=other_bone,
        aim_fraction=0.5,
        world_offset=(forward[0] * push, forward[1] * push, height / 2.0 + 0.015),
    )
    notes = [
        f"crate {side * 100:.0f}x{depth * 100:.0f}x{height * 100:.0f} cm, "
        f"resting on the wrists by its base, {push * 100:.0f} cm forward"
    ]
    if not other_bone:
        notes.append("no second hand given, so the crate hangs off one wrist and will drift")
    return SceneBuild(props=[box], notes=notes)


def build_basketball_scene(bpy: Any, style: PropStyle, *, hand_bone: str) -> SceneBuild:
    ball = Prop(
        obj=build_basketball(bpy, style),
        bone=hand_bone,
        offset=(0.0, 0.10, 0.0),
    )
    return SceneBuild(
        props=[ball],
        notes=["ball is pinned to the hand; a released shot would need authored keyframes"],
    )


def build_two_hand_carry_box_scene(
    bpy: Any,
    style: PropStyle,
    *,
    armature: Any,
    hand_bone: str = "左手首",
    other_bone: str | None = None,
    frame_start: int = 1,
    frames: int = 1,
    grip_mode: str = "side",
    fit_start: int | None = None,
    fit_end: int | None = None,
    side: float | None = None,
    depth: float | None = None,
    height: float | None = None,
    contact: bool = False,
    mesh_objects: list[Any] | None = None,
) -> SceneBuild:
    """Bake a fixed-size box to both hands; keep the old single-wrist preset intact.

    Fit bounds are inclusive Blender frame numbers. The box has its own world-space
    animation, so callers must not parent it to a wrist after this function returns.
    With ``contact=True``, only the derived actor's arm/hand rotations are fitted
    to the fixed box using evaluated skin. Original motion files remain unchanged.
    """
    import numpy as np
    from mathutils import Matrix, Vector

    from motionviewer.core.prop_fit import fit_two_hand_box

    if frames < 1:
        raise ValueError("frames must be positive")
    if contact and grip_mode != "side":
        raise ValueError("Box hand contact currently requires grip_mode='side'; bottom support is not validated")
    if other_bone is None:
        other_bone = "右手首" if hand_bone.startswith("左") else "左手首"
    bones = armature.pose.bones
    for name in (hand_bone, other_bone, "頭"):
        if name not in bones:
            raise KeyError(name)
    torso = next((name for name in ("上半身2", "上半身", "センター") if name in bones), None)
    if torso is None:
        raise KeyError("No torso bone for carry-box orientation")
    old_frame = bpy.context.scene.frame_current

    def point(name):
        return np.asarray(armature.matrix_world @ bones[name].head, dtype=float)

    def palm(name):
        prefix = "左" if name.startswith("左") else "右"
        knuckle = prefix + "中指１"
        return (point(name) + point(knuckle)) / 2 if knuckle in bones else point(name)

    left, right, heads, torsos = [], [], [], []
    try:
        for frame in range(frame_start, frame_start + frames):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            left.append(palm(hand_bone))
            right.append(palm(other_bone))
            heads.append(point("頭"))
            torsos.append(point(torso))
        fit_range = None
        if fit_start is not None or fit_end is not None:
            fit_range = ((fit_start if fit_start is not None else frame_start) - frame_start,
                         (fit_end if fit_end is not None else frame_start + frames - 1) - frame_start + 1)
        fit = fit_two_hand_box(np.asarray(left), np.asarray(right), np.asarray(heads), np.asarray(torsos),
                              grip_mode=grip_mode, fit_range=fit_range, side=side, depth=depth, height=height)
        obj = build_cardboard_box(bpy, style, side=fit.size[0], depth=fit.size[1], height=fit.size[2])
        # Joined cardboard panels inherit the top panel's origin. Re-centre their
        # mesh and normalize the actual outer bounds before applying the fitted pose.
        coordinates = np.asarray([tuple(vertex.co) for vertex in obj.data.vertices])
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
        normalized = (coordinates - (low + high) / 2) * (np.asarray(fit.size) / (high - low))
        for vertex, coordinate in zip(obj.data.vertices, normalized, strict=True):
            vertex.co = Vector(coordinate)
        obj.location = (0, 0, 0)
        obj.rotation_mode = "QUATERNION"
        obj.scale = (1, 1, 1)
        previous = None
        for index, frame in enumerate(range(frame_start, frame_start + frames)):
            quaternion = Matrix(fit.rotations[index]).to_quaternion()
            if previous is not None and quaternion.dot(previous) < 0:
                quaternion.negate()
            obj.location = Vector(fit.centres[index])
            obj.rotation_quaternion = quaternion
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_quaternion", frame=frame)
            previous = quaternion.copy()
        from motionviewer.blender.retarget._animate import _linearize

        _linearize(obj)
        report = dict(fit.report)
        report["fit_frames_inclusive"] = [frame_start + fit.report["fit_range_0based_half_open"][0],
                                          frame_start + fit.report["fit_range_0based_half_open"][1] - 1]
        report["frame_start"], report["frames"] = frame_start, frames
        if contact:
            from .mmd_box_contact import apply_box_hand_contact

            report["hand_contact"] = apply_box_hand_contact(
                bpy, armature, obj, frame_start=frame_start, frames=frames,
                size=fit.size, mesh_objects=mesh_objects,
            )
        return SceneBuild(
            props=[Prop(obj=obj)], report=report,
            notes=[f"fixed box {fit.size[0]:.3f} x {fit.size[1]:.3f} x {fit.size[2]:.3f} m; {grip_mode} grip",
                   ("Derived arm/hand rotations fitted to fixed box; original motion and root trajectory unchanged"
                    if contact else "World-space keys follow both palms; source motion and arm channels unchanged"),
                   f"contact error p95 {report['contact_error_per_hand_mm']['p95']:.1f} mm; inspect before acceptance"],
        )
    finally:
        bpy.context.scene.frame_set(old_frame)
        bpy.context.view_layer.update()


SCENE_BUILDERS = {
    "table_tennis": build_table_tennis_scene,
    "badminton": build_badminton_scene,
    "carry_box": build_carry_box_scene,
    "carry_box_two_hand": build_two_hand_carry_box_scene,
    "carry_box_supported": build_supported_box_scene,
    "basketball": build_basketball_scene,
}

#: Which hand each preset expects, and whether it needs a static camera. Kept beside
#: the builders so ``--scene`` can be validated before Blender does any work.
SCENE_DEFAULTS = {
    "table_tennis": {"hand": "右手首", "static_camera": True},
    "badminton": {"hand": "右手首", "static_camera": True},
    "carry_box": {"hand": "左手首", "static_camera": False},
    "carry_box_two_hand": {"hand": "左手首", "static_camera": False},
    "carry_box_supported": {"hand": "左手首", "static_camera": False},
    "basketball": {"hand": "右手首", "static_camera": False},
}


# ---------------------------------------------------------------------------
# weapons that ship as their own PMX
# ---------------------------------------------------------------------------

#: Character directory -> how to hold its weapon.
#:
#: ``grip`` says which end of the weapon's long axis the hand goes on: ``"low"``,
#: ``"high"`` or ``"mid"``. It is a table rather than something inferred, because it
#: cannot be inferred reliably. Three geometric rules were tried against the four
#: blades' measured girth profiles and each got three right and one wrong:
#:
#: * "the thin end is the grip" — a sword's blade tapers thinner than its pommel, so
#:   this puts the tip in the hand. This is the rule that shipped, and the render had
#:   the hilt pointing away and the tip through the back of the hand.
#: * "the grip is beside the widest slice" (the guard) — right for the two swords,
#:   wrong for a glaive whose blade fan is the widest thing on it.
#: * "the grip is the longest thin run" — right for the glaive's shaft, wrong for the
#:   swords, whose blades are longer than their grips.
#:
#: The profiles that decide each entry, in mm across 20 slices low to high:
#:   彦卿  40 38 31 32 [103] 86 51 39 ... 18   grip = the 4 thin slices below the guard
#:   流萤  20 38 26 31 [121 140] 68 ... 14     same shape
#:   青衣  64 61 61 101 31 ... 31 101 61 61 64  symmetric staff, held in the middle
#:   飞霄  32 [622 669 722 785] ... 30 50 41    blade fan is 70% of it; shaft is on top
WEAPON_BY_CHARACTER: dict[str, dict] = {
    "星穹铁道—彦卿": {"file": "武器.pmx", "hand": "右手首", "grip": "low"},
    "星穹铁道—流萤·春日手信": {"file": "剑.pmx", "hand": "右手首", "grip": "low"},
    "星穹铁道—飞霄无外套": {"file": "星穹铁道—飞霄武器.pmx", "hand": "右手首", "grip": "high"},
    "青衣": {"file": "武器.pmx", "hand": "右手首", "grip": "mid"},
    "星穹铁道—三月七": {"file": "弓1.0.pmx", "hand": "左手首", "grip": "mid"},
    "星穹铁道—火花·甜梦电波": {"file": "手杖（整.pmx", "hand": "右手首", "grip": "low"},
    "星穹铁道——银狼": {"file": "炮.pmx", "hand": "右手首", "grip": "mid"},
    "耀嘉音": {"file": "武器.pmx", "hand": "右手首", "grip": "mid"},
    "星穹铁道—知更鸟": {"file": "话筒.pmx", "hand": "右手首", "grip": "low"},
    "星穹铁道—花火": {"file": "剑玉.pmx", "hand": "右手首", "grip": "mid"},
    "奥黛塔": {"file": "剑.pmx", "hand": "右手首", "grip": "low"},
}


def load_weapon(bpy: Any, path: Any, *, label: str = "weapon") -> Any:
    """Import a weapon PMX and return one joined mesh, ready to be parented.

    The weapon arrives with its own tiny armature, which is discarded: the whole point
    is for the blade to follow the character's hand, and leaving a second armature in
    the scene would give the mesh a parent that never moves. Joining the meshes first
    means the prop is a single object, which is what ``attach_to_bone`` expects.
    """
    from motionviewer.blender.mmd_import import import_pmx_character

    before = set(bpy.data.objects)
    import_pmx_character(bpy, path, label=label)
    added = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in added if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh in {path}")
    for obj in added:
        if obj.type != "MESH":
            # Unparent the meshes first, or removing the armature takes them along.
            for child in list(obj.children):
                child.parent = None
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in meshes:
        obj.modifiers.clear()
        obj.parent = None
    return _join(bpy, meshes, f"weapon_{label}")


def build_weapon_scene(
    bpy: Any,
    style: PropStyle,
    *,
    hand_bone: str,
    weapon_path: Any,
    is_blade: bool = True,
    grip_offset: float = 0.0,
) -> SceneBuild:
    """A weapon held in one hand, aligned along the forearm.

    A blade is aligned to the forearm rather than given fixed Euler angles for the same
    reason the rackets are: the angles that express "pointing along the arm" depend on
    the rig's bone roll, and MMD rigs do not agree on it.

    The weapon keeps its own materials and textures, so it is *not* passed through
    ``apply_toon_shading`` by the caller unless asked — these models are already
    authored for MMD's toon look, the same as the character.
    """
    del style
    elbow = "右ひじ" if hand_bone.startswith("右") else "左ひじ"
    prop = Prop(
        obj=load_weapon(bpy, weapon_path),
        bone=hand_bone,
        align_from=elbow if is_blade else None,
        align_to=hand_bone if is_blade else None,
        offset=(0.0, grip_offset, 0.0),
    )
    return SceneBuild(
        props=[prop],
        notes=[
            f"weapon {getattr(weapon_path, 'name', weapon_path)} on {hand_bone}"
            + (", aligned to the forearm" if is_blade else ", at the rig's own orientation")
        ],
    )


def load_stage(bpy: Any, path: Any, *, label: str = "stage", scale: float = 1.0) -> list:
    """Import a community MMD stage PMX and leave it standing in the world.

    Stages are the format the MMD community actually distributes environments in, and
    the PMX importer already in this project reads them, so nothing new is needed to
    use one: point ``--stage`` at the file. A stage has no body rig, so its armature is
    dropped for the same reason a weapon's is — the geometry should sit still in world
    space, and keeping an unused armature as its parent only risks moving it.

    ``scale`` is exposed because MMD stages are authored in the same 8-units-per-metre
    convention as the characters but not always consistently, and a stage that is 12%
    off reads as the character being the wrong size.
    """
    from motionviewer.blender.mmd_import import import_pmx_character

    before = set(bpy.data.objects)
    import_pmx_character(bpy, path, label=label)
    added = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in added if obj.type == "MESH"]
    for obj in added:
        if obj.type != "MESH":
            for child in list(obj.children):
                child.parent = None
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in meshes:
        obj.modifiers.clear()
        obj.parent = None
        if scale != 1.0:
            obj.scale = (scale, scale, scale)
    return meshes


def _span(obj: Any, axis: int) -> float:
    values = [v.co[axis] for v in obj.data.vertices]
    return (max(values) - min(values)) if values else 0.0


def _girth_profile(obj: Any, axis: int, slices: int = 40) -> Any:
    """Mean cross-section width in each slice along ``axis``, low to high."""
    import numpy as np

    co = np.array([(v.co.x, v.co.y, v.co.z) for v in obj.data.vertices])
    low, high = co[:, axis].min(), co[:, axis].max()
    others = [index for index in range(3) if index != axis]
    edges = np.linspace(low, high, slices + 1)
    out = []
    for index in range(slices):
        mask = (co[:, axis] >= edges[index]) & (co[:, axis] < edges[index + 1])
        if mask.sum() < 3:
            out.append(0.0)
            continue
        chunk = co[mask]
        out.append(float(np.mean([chunk[:, o].max() - chunk[:, o].min() for o in others])))
    return np.array(out), float(low), float(high)


def _grip_point_and_axis(obj: Any, grip: str) -> tuple[Any, Any, float, float]:
    """Where the hand goes, which way the weapon points, its length and its shaft width.

    The grip *section* is measured, not assumed: it is the run of slices from the end the
    table names whose girth stays under 45% of the weapon's widest, and the hand goes at
    that run's midpoint. Putting the hand at the extreme end instead — which an earlier
    version did, at 5 cm in — leaves 20 cm of grip sticking out past the fist and reads
    as a palm pressed against the weapon's butt rather than a hand holding it. Measured:
    彦卿 21.8 cm of grip, 流萤 23.1, 飞霄 52.2, 青衣 120.6.
    """
    import numpy as np
    from mathutils import Vector

    co = np.array([(v.co.x, v.co.y, v.co.z) for v in obj.data.vertices])
    spans = co.max(axis=0) - co.min(axis=0)
    axis = int(np.argmax(spans))
    profile, low, high = _girth_profile(obj, axis)
    slices = len(profile)
    thin = profile.max() * 0.45

    direction = Vector((0.0, 0.0, 0.0))
    if grip == "mid":
        middle = slices // 2
        reach = 0
        while (
            middle - reach >= 0
            and middle + reach < slices
            and profile[middle - reach] < thin
            and profile[middle + reach] < thin
        ):  # empty slices count as thin; see the note below
            reach += 1
        first, last = middle - reach, middle + reach
        direction[axis] = 1.0
    elif grip == "low":
        # An empty slice means no vertex fell in it, not that the weapon got thicker, so
        # it must not end the run. Requiring ``profile[i] > 0`` did end it, and on Feixiao
        # that turned a 9-slice grip into a 1-slice one at the very tail of the shaft:
        # its profile has gaps at indices 30 and 38, so the hand was placed 1.10 m out on
        # the pole instead of 0.89 m, the weapon sat a metre from the fist, and the render
        # showed a closed hand holding nothing.
        last = 0
        while last < slices and profile[last] < thin:
            last += 1
        first = 0
        direction[axis] = 1.0
    else:
        first = slices - 1
        while first >= 0 and profile[first] < thin:
            first -= 1
        last = slices - 1
        direction[axis] = -1.0
    first, last = max(0, first), min(slices - 1, max(last, first + 1))

    step = (high - low) / slices
    band_from, band_to = low + first * step, low + (last + 1) * step
    centre = (band_from + band_to) / 2.0
    shaft = float(np.median([p for p in profile[first : last + 1] if p > 0] or [profile.max()]))

    inside = co[(co[:, axis] >= band_from) & (co[:, axis] <= band_to)]
    if not len(inside):
        inside = co
    grip_local = Vector((0.0, 0.0, 0.0))
    grip_local[axis] = centre
    for other in (index for index in range(3) if index != axis):
        grip_local[other] = float(inside[:, other].mean())
    return grip_local, direction, float(spans[axis]), shaft


def grip_axis_from_fingers(
    bpy: Any, armature: Any, hand_bone: str, *, frame: int | None = None, mode: str = "legacy"
) -> tuple[Any, Any, Any]:
    """The line a closed fist actually encircles, measured by closing it.

    Everything about this was guessed before and each guess was wrong in a way the
    numbers hid. The shaft was put along the knuckle *heads*, which are the bone line and
    run level with the back of the hand, so it rode on top of the knuckles. Sinking it by
    a radius needed a palm direction, and orienting that by the thumb's base gave the
    wrong sign — the thumb root sits beside the hand, not on the palm, and a test curl
    moves the fingertip the other way. Each fix moved the shaft somewhere new and still
    outside the fingers.

    So this stops guessing and asks the hand. It curls the four fingers to a typical fist,
    takes the midpoint between each knuckle and its own fingertip — the centre of the ring
    that finger has formed — and fits a line through the four. On Liuying those four
    midpoints are collinear to 1.4-2.1 mm, which is what makes them a grip axis rather
    than four unrelated points. The result is 5.2 degrees off the knuckle line, so the
    direction was nearly right all along; its centre is 9.4 mm away, on the palm side,
    and that offset is the whole difference between holding a sword and resting one on
    your fingers.

    Returns the axis point, its unit direction, and the palm normal, all in world space.
    The pose is restored before returning. ``bounded`` sets total local-X flexion;
    ``legacy`` preserves the old additive pose for reproducing existing shots.
    """
    from mathutils import Euler, Vector

    if mode not in {"legacy", "bounded"}:
        raise ValueError("mode must be 'legacy' or 'bounded'")

    # Measure on the frame the weapon will be placed for. Without this the function reads
    # whatever frame the scene happens to be on, and the retarget leaves it at the *last*
    # one: on Feixiao that put the axis 872 mm from the wrist instead of 69 mm, so the
    # weapon was never brought to the hand at all and the render showed an empty fist.
    # Liuying only escaped it because its final pose happened to be near its first.
    # Same class of mistake as building the physics rig on a stale frame.
    if frame is not None:
        bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()

    side = "右" if hand_bone.startswith("右") else "左"
    fingers = ("人指", "中指", "薬指", "小指")
    world = armature.matrix_world
    bones = armature.pose.bones

    saved = []
    for finger in fingers:
        for segment in ("１", "２", "３"):
            bone = bones.get(f"{side}{finger}{segment}")
            if bone is not None:
                bone.rotation_mode = "QUATERNION"
                saved.append((bone, bone.rotation_quaternion.copy()))
    if not saved:
        return None, None, None

    # A typical fist. The exact angles do not matter — they only have to be enough for
    # the fingers to form rings, since the solve refines each one afterwards.
    for bone, _ in saved:
        for segment, degrees in (("１", 75.0), ("２", 90.0), ("３", 60.0)):
            if bone.name.endswith(segment):
                if mode == "bounded":
                    original = bone.rotation_quaternion.to_euler("XYZ")
                    bone.rotation_quaternion = Euler(
                        (math.radians(degrees), original.y, original.z), "XYZ"
                    ).to_quaternion()
                else:
                    bone.rotation_quaternion = (
                        bone.rotation_quaternion @ Euler((math.radians(degrees), 0.0, 0.0), "XYZ").to_quaternion()
                    )
                break
    bpy.context.view_layer.update()

    centres = []
    for finger in fingers:
        knuckle = bones.get(f"{side}{finger}１")
        tip = bones.get(f"{side}{finger}３")
        if knuckle is None or tip is None:
            continue
        centres.append(((world @ knuckle.head) + (world @ tip.tail)) / 2.0)

    for bone, original in saved:
        bone.rotation_quaternion = original
    bpy.context.view_layer.update()

    if len(centres) < 2:
        return None, None, None

    origin = Vector((0.0, 0.0, 0.0))
    for point in centres:
        origin += point
    origin /= len(centres)
    # Direction from the spread of the centres: first to last is enough at four points,
    # and avoids pulling numpy into this module.
    direction = (centres[-1] - centres[0]).normalized()

    # Palm normal: from the knuckle line toward the axis, which is by construction the
    # side the fingers close on.
    knuckle_centre = Vector((0.0, 0.0, 0.0))
    count = 0
    for finger in fingers:
        knuckle = bones.get(f"{side}{finger}１")
        if knuckle is not None:
            knuckle_centre += world @ knuckle.head
            count += 1
    normal = origin - knuckle_centre / max(count, 1)
    normal = normal - direction * normal.dot(direction)
    if normal.length < 1e-6:
        normal = Vector((0.0, 0.0, 1.0))
    return origin, direction, normal.normalized()


def fit_weapon_to_hand(
    bpy: Any,
    obj: Any,
    armature: Any,
    hand_bone: str,
    grip: str = "low",
    *,
    frame: int | None = None,
    mode: str = "legacy",
) -> dict:
    """Put the grip in the palm and point the weapon away from the character.

    ``legacy`` preserves historical placement. ``bounded`` additionally searches a
    small palm-plane offset against four finger/cylinder proxies. ``surface`` uses
    an asset-checked calibration with evaluated-skin diagnostics and an opposed
    thumb; currently its support is limited to the calibrated Feixiao weapon pair.

    Two things went wrong before this and both showed up as the same picture. The grip
    end was chosen by a geometric guess that is inverted for a sword, so the tip sat in
    the hand and the hilt pointed outward. And the weapon was aimed along the *forearm*,
    which for a raised arm points the blade back over the character's own head.

    The grip end comes from the table. The *direction* is the one thing left that had to
    be measured, and the obvious answer was wrong too: pointing the shaft along the fist
    axis (wrist to middle knuckle) puts it exactly through the wrist and that one
    knuckle — both 0.0 mm from the axis — while the other three knuckles fan 18, 18 and
    33 mm off it. Fingers cannot wrap something lying along their own direction, and no
    amount of curl fixes it: solved per finger against the shaft, only the middle one
    reached 13 mm, the others stalled at 19-46 mm even at the anatomical limit.

    A real grip runs **across** the palm, perpendicular to the fingers, so the shaft is
    laid along the knuckle line — index knuckle to little knuckle — which is the axis all
    four fingers close around. The weapon then points outward along the fist axis, which
    is what ``local_dir`` already gives.
    """
    if mode == "surface":
        from motionviewer.blender.mmd_weapon_contact import fit_surface_weapon

        return fit_surface_weapon(bpy, obj, armature, hand_bone, grip=grip, frame=frame)

    from mathutils import Matrix, Vector

    if mode not in {"legacy", "bounded"}:
        raise ValueError("mode must be 'legacy', 'bounded' or 'surface'")

    # Every world-space measurement below must use the requested placement frame.
    # Retargeting leaves the scene on its last keyed frame; accepting ``frame``
    # without applying it makes this API depend on the caller's current frame.
    if frame is not None:
        bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()

    bones = armature.pose.bones
    if hand_bone not in bones:
        raise KeyError(hand_bone)
    side = "右" if hand_bone.startswith("右") else "左"
    knuckle = f"{side}中指１"
    world = armature.matrix_world
    wrist = world @ bones[hand_bone].head
    knuckles = world @ (bones[knuckle].head if knuckle in bones else bones[hand_bone].tail)
    forward = knuckles - wrist
    if forward.length < 1e-6:
        forward = Vector((0.0, 0.0, -1.0))
    forward.normalize()
    # The grip sits at the centre of the closed fist, which is the mean of the knuckles
    # — not a fraction along the wrist-to-middle-knuckle line. Using 0.6 of that line put
    # the shaft 29 mm off the palm, and since the fingers reach about 13 mm, that alone
    # made the grip unreachable no matter how they curled.
    knuckle_names = [f"{side}{finger}１" for finger in ("人指", "中指", "薬指", "小指")]
    present = [bones[name] for name in knuckle_names if name in bones]
    if present:
        palm = Vector((0.0, 0.0, 0.0))
        for bone in present:
            palm += world @ bone.head
        palm /= len(present)
    else:
        palm = wrist + forward * (knuckles - wrist).length * 0.6
    # Measured grip axis first: close the fist and see what line it encircles. The
    # knuckle-line construction below is the fallback for a rig with no finger bones.
    measured_origin, measured_dir, measured_normal = grip_axis_from_fingers(
        bpy, armature, hand_bone, mode=mode
    )

    # The knuckle line: the axis a closed fist encircles. Falls back to the hand bone's
    # own X when a rig has no finger bones to measure it from.
    index_knuckle = bones.get(f"{side}人指１")
    little_knuckle = bones.get(f"{side}小指１")
    if index_knuckle is not None and little_knuckle is not None:
        shaft_dir = (world @ little_knuckle.head) - (world @ index_knuckle.head)
    else:
        shaft_dir = (world @ bones[hand_bone].matrix).to_3x3() @ Vector((1.0, 0.0, 0.0))
    # Keep it square to the fingers, so the shaft crosses the palm instead of skewing
    # back along them.
    shaft_dir = shaft_dir - forward * shaft_dir.dot(forward)
    if shaft_dir.length < 1e-6:
        shaft_dir = Vector((1.0, 0.0, 0.0))
    shaft_dir.normalize()
    # Point the business end away from the body rather than into it: of the two ways the
    # knuckle line can run, take the one whose far end is further from the wrist.
    if (palm + shaft_dir).length < (palm - shaft_dir).length:
        shaft_dir = -shaft_dir

    # Sink the shaft into the palm. The knuckle *heads* are the bone line, which runs
    # level with the back of the hand — a shaft placed exactly there rides on top of the
    # knuckles, which is what "the hilt on the back of the hand" looks like. Measured on
    # Liuying: the grip sat at -0.0 mm on the palm normal while the curled fingertips
    # reached +43 to +45 mm, so the fingers closed through empty air underneath it.
    #
    # Distance to the shaft *axis* cannot catch this: an axis is an infinite line, and
    # 13.6 mm from it is satisfied on either side of the hand. It needs the signed
    # projection along the palm normal.
    palm_normal = forward.cross(shaft_dir)
    if palm_normal.length < 1e-6:
        palm_normal = Vector((0.0, 0.0, 1.0))
    palm_normal.normalize()
    # Orient it by where the fingers actually curl, not by where the thumb is. The
    # thumb's *base* sits to the side of the hand rather than on the palm, so testing
    # against it gave the wrong sign: it said +30.7 mm while a test curl moved the middle
    # fingertip -23.6 mm along the same vector. With the sign flipped the shaft was
    # placed on the back of the fingers, and the render showed it lying across their tops
    # with all four curled underneath.
    middle_tip = bones.get(f"{side}中指３")
    if middle_tip is not None:
        before = world @ middle_tip.tail
        saved = [
            (bones[f"{side}{finger}{segment}"], bones[f"{side}{finger}{segment}"].rotation_quaternion.copy())
            for finger in ("人指", "中指", "薬指", "小指")
            for segment in ("１", "２", "３")
            if f"{side}{finger}{segment}" in bones
        ]
        from mathutils import Euler as _Euler

        for bone, _ in saved:
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = (
                bone.rotation_quaternion @ _Euler((math.radians(60.0), 0.0, 0.0), "XYZ").to_quaternion()
            )
        bpy.context.view_layer.update()
        if palm_normal.dot((world @ middle_tip.tail) - before) < 0.0:
            palm_normal = -palm_normal
        for bone, original in saved:
            bone.rotation_quaternion = original
        bpy.context.view_layer.update()

    grip_local, local_dir, length, shaft = _grip_point_and_axis(obj, grip)

    if measured_origin is not None:
        # Use the measured ring centre and axis outright. Keep the sign of the direction
        # that points the weapon away from the wrist rather than back through the arm.
        palm = measured_origin
        if measured_dir.dot(shaft_dir) < 0.0:
            measured_dir = -measured_dir
        shaft_dir = measured_dir
        palm_normal = measured_normal
        # The measured axis is where the fist closes on *itself*, so it holds a shaft
        # only as thick as that ring. Measured at the flexion cap, the ring radius is
        # about 10 mm on these hands — a 20 mm shaft. Yanqing's is 31 mm and Feixiao's
        # 37 mm, and for the wider one the fingertips stopped 15-29 mm short of the
        # surface while Yanqing's reached 2-15 mm. Pushing the excess out of the palm
        # lets the fingers meet the shaft instead of asking them to bend past the limit.
        excess = shaft / 2.0 - _FIST_RING_RADIUS_M
        if excess > 0.0:
            palm = palm + palm_normal * excess
    else:
        # One shaft radius plus flesh below the knuckle line, so the fingers close over
        # the shaft rather than past it.
        palm = palm + palm_normal * (shaft / 2.0 + _SKIN_M)

    # Constrain the roll too: the blade's flat is turned across the palm rather than at
    # the camera. Leaving it to rotation_difference rendered a 2 m polearm as a rod.
    # The weapon's long axis goes along the knuckle line; its flat is turned to face
    # along the fingers, which is how a blade sits in a fist.
    up = shaft_dir.cross(forward)
    if up.length < 1e-6:
        up = Vector((0.0, 0.0, 1.0))
    up.normalize()
    target = Matrix((forward, up, shaft_dir)).transposed()

    flat_axis = max(
        (index for index in range(3) if abs(local_dir[index]) < 0.5),
        key=lambda index: _span(obj, index),
    )
    local_flat = Vector((0.0, 0.0, 0.0))
    local_flat[flat_axis] = 1.0
    local_up = local_dir.normalized().cross(local_flat)
    if local_up.length < 1e-6:
        local_up = Vector((0.0, 0.0, 1.0))
    local_up.normalize()
    source = Matrix((local_flat, local_up, local_dir.normalized())).transposed()
    rotation = (target @ source.inverted()).to_4x4()

    placement = Matrix.Translation(palm - (rotation @ grip_local)) @ rotation
    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = hand_bone
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_world = placement
    placement_search = None
    if mode == "bounded":
        import numpy as np

        from motionviewer.core.prop_fit import finger_chain_points, fit_finger_cylinder

        bpy.context.view_layer.update()
        axis_origin, axis_direction = _shaft_axis(obj, grip)
        normal = np.asarray(palm_normal, dtype=float)
        direction = np.asarray(axis_direction, dtype=float)
        radial = np.cross(direction, normal)
        radial /= max(np.linalg.norm(radial), 1e-9)
        chains = {}
        for finger in ("人指", "中指", "薬指", "小指"):
            chain = [bones.get(f"{side}{finger}{segment}") for segment in ("１", "２", "３")]
            if not all(bone is not None and bone.parent is not None for bone in chain):
                continue
            parent_world = np.asarray(world @ chain[0].parent.matrix)
            offsets = np.asarray([bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
                                  for bone in chain])
            lengths = np.asarray([bone.bone.length for bone in chain])
            reference = np.asarray([tuple(bone.rotation_quaternion.to_euler("XYZ")) for bone in chain])
            predicted = finger_chain_points(parent_world, offsets, lengths, reference)
            actual = np.asarray([tuple(world @ chain[0].head)] + [tuple(world @ bone.tail) for bone in chain])
            if float(np.max(np.linalg.norm(predicted - actual, axis=1))) < 0.0001:
                chains[finger] = (parent_world, offsets, lengths, reference)
        if len(chains) == 4:
            best = None
            for outward in (0.0, 0.01, 0.02):
                for along in (-0.02, -0.01, 0.0, 0.01, 0.02):
                    shift = normal * outward + radial * along
                    origin = np.asarray(axis_origin) + shift
                    fits = {finger: fit_finger_cylinder(*chain, origin, direction, shaft / 2 + _SKIN_M)
                            for finger, chain in chains.items()}
                    score = sum(item["score"] for item in fits.values()) + 0.01 * float(np.dot(shift, shift))
                    if best is None or score < best[0]:
                        best = (score, shift, fits)
            score, shift, fits = best
            translated = obj.matrix_world.copy()
            translated.translation += Vector(shift)
            obj.matrix_world = translated
            palm += Vector(shift)
            bpy.context.view_layer.update()
            placement_search = {
                "shift_world_mm": (shift * 1000).tolist(),
                "shift_length_mm": float(np.linalg.norm(shift) * 1000),
                "score": float(score),
                "candidate_count": 15,
                "range": "0..20 mm palmward, +/-20 mm along the orthogonal radial direction",
                "skin_surface_measured": False,
                "proxy": "four three-bone finger chains against shaft radius + 2 mm padding; thumb excluded",
                "finger_diagnostics": {finger: {key: value for key, value in item.items() if key != "points"}
                                       for finger, item in fits.items()},
            }
        else:
            placement_search = {"skipped": "four complete independently verified finger chains required"}
    return {
        "fit_mode": mode,
        "placement_search": placement_search,
        "grip": grip,
        "palm_normal": tuple(palm_normal),
        "shaft_mm": round(shaft * 1000, 1),
        "length_m": length,
        "palm": tuple(round(v, 3) for v in palm),
        "tip": tuple(round(v, 3) for v in (palm + forward * length)),
    }


#: Base curl per phalanx, for a shaft about 32 mm across. Scaled by shaft width in
#: ``close_hand``: a 26 mm hilt needs the fingers further round than a 37 mm pole, and a
#: single fixed set left visible daylight between the fingertips and the thinner grips.
CLOSED_FINGER_DEGREES = {
    "１": 46.0,
    "２": 58.0,
    "３": 40.0,
}
#: The shaft width the angles above are tuned for, in metres.
REFERENCE_SHAFT_M = 0.032


def _shaft_axis(weapon: Any, grip: str) -> tuple[Any, Any]:
    """A point on the weapon's grip and the unit direction of its shaft, in world space."""
    grip_local, local_dir, _, _ = _grip_point_and_axis(weapon, grip)
    matrix = weapon.matrix_world
    origin = matrix @ grip_local
    along = (matrix @ (grip_local + local_dir.normalized() * 0.1)) - origin
    return origin, along.normalized()


def _distance_to_axis(point: Any, origin: Any, direction: Any) -> float:
    offset = point - origin
    return (offset - direction * offset.dot(direction)).length


def _wrap_error(point: Any, origin: Any, direction: Any, normal: Any, radius: float) -> float:
    """How far a fingertip is from resting on the shaft, counting which side it is on.

    Distance to the axis alone is not enough, and this is the third indicator here that
    failed by measuring the wrong thing: an axis is an infinite line, so "13.6 mm from
    the axis" is satisfied by a fingertip curled 30 mm *past* the shaft just as well as
    by one touching it. Measured on Liuying, the tips sat at +43 mm on the palm normal
    while the shaft was at +13.6 — they had closed straight through it into the palm.

    So the target is the *far* surface of the shaft, on the side the fingers come from:
    a tip should reach the shaft and stop. Overshoot is penalised twice, since a finger
    through the grip looks worse than one a little short of it.
    """
    offset = point - origin
    radial = (offset - direction * offset.dot(direction)).length
    beyond = offset.dot(normal)
    error = abs(radial - radius)
    if beyond < -radius:
        error += 2.0 * abs(beyond + radius)
    return error


#: Ratio of curl between the three phalanges. Human fingers bend most at the middle
#: joint, and holding that shape while one number scales it keeps the search
#: one-dimensional per finger.
_PHALANX_RATIO = (1.0, 1.25, 0.85)
#: Configured total-flexion bounds, in degrees: knuckle, middle joint, fingertip joint.
#: The search used to stop at 140, a number with nothing behind it, and three of four
#: weapons solved past it — 100 to 136 at the knuckle, which drives the middle joint to
#: 125-170 and folds the finger through itself. That is the deformation, and it is why
#: Yanqing was the only acceptable hand: it solved at 88, just inside the limit.
_FLEXION_LIMITS = (90.0, 110.0, 90.0)
#: Radius of the ring a fist encloses at the flexion cap, measured on these rigs: the
#: fingertips land 7.6-13.1 mm from the grip axis, so about 10 mm.
_FIST_RING_RADIUS_M = 0.010
#: Legacy incremental cap. Since legacy adds this to a relaxed pose, it is NOT a
#: total-angle guarantee. The opt-in bounded mode sets total local-X angles instead.
_MAX_BASE_DEGREES = min(limit / ratio for limit, ratio in zip(_FLEXION_LIMITS, _PHALANX_RATIO))
#: Heuristic padding between bone centres and the shaft proxy, not measured skin.
_SKIN_M = 0.002


def close_hand(
    bpy: Any,
    armature: Any,
    hand_bone: str,
    *,
    frame_start: int = 1,
    frames: int = 1,
    shaft_m: float = 0.032,
    weapon: Any = None,
    grip: str = "low",
    palm_normal: Any = None,
    mode: str = "legacy",
) -> dict:
    """Fit a finger pose against the shaft, with explicit remaining fit errors.

    ``surface`` applies the matching asset calibration, including thumb opposition,
    and measures evaluated skin samples. The default ``legacy`` reproduces existing
    additive curls. ``bounded`` searches
    independent total joint flexions, keeps local Y/Z spread, checks its kinematics
    against Blender, and penalizes bone samples entering a padded cylinder. Neither
    mode is a skin/weapon collision solver, and reaching a proxy does not establish
    natural-looking contact. The thumb remains unchanged.

    With ``weapon`` given this solves rather than guesses: for each finger it sweeps the
    curl and keeps the angle whose fingertip lands one shaft radius plus 2 mm from the
    shaft axis. Solved on Liuying's 23 mm hilt the four fingers land at 13.5, 13.9, 14.2
    and 13.8 mm against a 13.6 mm target, and they need visibly different angles to do
    it — 78, 102, 80 and 102 degrees at the first joint. That spread is why the scaled
    single angle this replaces could not work: it moved all four together, so whatever
    closed the middle finger left the little finger 22 mm short.

    The solve is only possible because the shaft now crosses the palm. While it ran along
    the fingers instead, the knuckles sat 0, 18, 18 and 33 mm off the axis and no curl
    reached it at all — at the anatomical limit the fingertips still stopped 25 to 46 mm
    away. Two of those measurements also showed curling *past* the contact angle carries
    the tip back out again, so a search has to keep the best value rather than the last.

    Without ``weapon`` it falls back to scaling a fixed pose by shaft width, which is what
    the non-weapon props use.

    The thumb is left alone throughout: it wraps the opposite way, and the same curl
    drives it through the grip.
    """
    if mode == "surface":
        from motionviewer.blender.mmd_weapon_contact import close_surface_hand

        return close_surface_hand(bpy, armature, hand_bone, weapon=weapon, grip=grip,
                                  frame_start=frame_start, frames=frames)

    from mathutils import Euler

    if mode not in {"legacy", "bounded"}:
        raise ValueError("mode must be 'legacy', 'bounded' or 'surface'")
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()

    side = "右" if hand_bone.startswith("右") else "左"
    fingers = ("人指", "中指", "薬指", "小指")
    segments = ("１", "２", "３")

    def bone_of(finger: str, segment: str):
        return armature.pose.bones.get(f"{side}{finger}{segment}")

    # Whatever the retarget left on these bones is the pose the curl composes onto.
    rest = {}
    rest_eulers = {}
    for finger in fingers:
        for segment in segments:
            bone = bone_of(finger, segment)
            if bone is not None:
                bone.rotation_mode = "QUATERNION"
                rest[(finger, segment)] = bone.rotation_quaternion.copy()
                rest_eulers[(finger, segment)] = bone.rotation_quaternion.to_euler("XYZ")

    joint_degrees: dict[str, list[float]] = {}
    diagnostics: dict[str, dict] = {}

    def pose(finger: str, base: float) -> None:
        for segment, ratio in zip(segments, _PHALANX_RATIO):
            bone = bone_of(finger, segment)
            if bone is None:
                continue
            if mode == "bounded":
                original = rest_eulers[(finger, segment)]
                angle = min(base * ratio, _FLEXION_LIMITS[segments.index(segment)])
                bone.rotation_quaternion = Euler(
                    (math.radians(angle), original.y, original.z), "XYZ"
                ).to_quaternion()
            else:
                curl = Euler((math.radians(base * ratio), 0.0, 0.0), "XYZ").to_quaternion()
                bone.rotation_quaternion = rest[(finger, segment)] @ curl

    solved: dict[str, float] = {}
    scale = min(1.35, max(0.75, 0.032 / max(shaft_m, 1e-4)))
    fallback = {"人指": 46.0 * scale, "中指": 46.0 * scale, "薬指": 46.0 * scale, "小指": 46.0 * scale}

    if weapon is not None:
        bpy.context.scene.frame_set(frame_start)
        bpy.context.view_layer.update()
        origin, direction = _shaft_axis(weapon, grip)
        radius = shaft_m / 2.0 + _SKIN_M
        world = armature.matrix_world
        # Without a palm normal the solver can only match distance, which a fingertip
        # curled past the shaft satisfies too.
        normal = palm_normal
        for finger in fingers:
            tip = bone_of(finger, "３")
            if tip is None:
                solved[finger] = fallback[finger]
                continue
            if mode == "bounded":
                import numpy as np

                from motionviewer.core.prop_fit import finger_chain_points, fit_finger_cylinder

                chain = [bone_of(finger, segment) for segment in segments]
                if all(bone is not None and bone.parent is not None for bone in chain):
                    parent_world = np.asarray(world @ chain[0].parent.matrix)
                    offsets = np.asarray([bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
                                          for bone in chain])
                    lengths = np.asarray([bone.bone.length for bone in chain])
                    reference = np.asarray([tuple(rest_eulers[(finger, segment)]) for segment in segments])
                    predicted = finger_chain_points(parent_world, offsets, lengths, reference)
                    actual = np.asarray([tuple(world @ chain[0].head)] + [tuple(world @ bone.tail) for bone in chain])
                    fk_error = float(np.max(np.linalg.norm(predicted - actual, axis=1)))
                    if fk_error < 0.0001:
                        fitted = fit_finger_cylinder(parent_world, offsets, lengths, reference,
                                                     np.asarray(origin), np.asarray(direction), radius)
                        joint_degrees[finger] = fitted.pop("degrees")
                        fitted.pop("points")
                        fitted["fk_check_mm"] = fk_error * 1000
                        fitted["collision_proxy"] = "bone-centre samples against shaft radius + 2 mm padding"
                        fitted["skin_surface_measured"] = False
                        diagnostics[finger] = fitted
                        solved[finger] = joint_degrees[finger][0]
                        continue
                    diagnostics[finger] = {"fallback": "bounded ratio; independent FK check failed",
                                           "fk_check_mm": fk_error * 1000}
            best, best_error = fallback[finger], float("inf")
            for base in range(20, int(_MAX_BASE_DEGREES) + 1, 2):
                pose(finger, float(base))
                bpy.context.view_layer.update()
                point = world @ tip.tail
                if normal is None:
                    error = abs(_distance_to_axis(point, origin, direction) - radius)
                else:
                    error = _wrap_error(point, origin, direction, normal, radius)
                if error < best_error:
                    best, best_error = float(base), error
            solved[finger] = best
            pose(finger, best)
    else:
        solved = dict(fallback)

    keys = 0
    for offset in range(max(1, frames)):
        frame = frame_start + offset
        bpy.context.scene.frame_set(frame)
        for finger in fingers:
            for segment, ratio in zip(segments, _PHALANX_RATIO):
                bone = bone_of(finger, segment)
                if bone is None:
                    continue
                if mode == "bounded":
                    original = rest_eulers[(finger, segment)]
                    angle = (joint_degrees[finger][segments.index(segment)] if finger in joint_degrees
                             else min(solved[finger] * ratio, _FLEXION_LIMITS[segments.index(segment)]))
                    bone.rotation_quaternion = Euler(
                        (math.radians(angle), original.y, original.z), "XYZ"
                    ).to_quaternion()
                else:
                    curl = Euler((math.radians(solved[finger] * ratio), 0.0, 0.0), "XYZ").to_quaternion()
                    bone.rotation_quaternion = rest[(finger, segment)] @ curl
                bone.keyframe_insert("rotation_quaternion", frame=frame)
                keys += 1
    bpy.context.scene.frame_set(frame_start)
    return {
        "bones": len(fingers) * len(segments),
        "keys": keys,
        "solved": {k: round(v, 1) for k, v in solved.items()},
        "mode": "solved" if weapon is not None else "scaled",
        "fit_mode": mode,
        "total_joint_degrees": joint_degrees if mode == "bounded" else {},
        "diagnostics": diagnostics,
        "thumb": "unchanged; opposition not solved",
    }


def _long_axis_and_handle(obj: Any) -> tuple[int, bool, float, float]:
    """Which local axis the weapon runs along, and which end of it is the handle.

    Nothing about a weapon PMX can be assumed. Measured across four of them, the long
    axis is +Z on two and +Y on the other two, and the origin sits at the grip end on
    two and dead centre on the others — so code that hardcodes "length along Z, origin
    at the handle" points half of them the wrong way and buries the other half halfway
    into the wrist.

    The handle is found as the *thin* end: a blade is wide and a grip is not, so the
    cross-section perpendicular to the long axis separates them. Returns the axis index,
    whether the handle is at the low end of it, and the two extents along that axis.
    """
    mesh = obj.data
    coordinates = [(v.co.x, v.co.y, v.co.z) for v in mesh.vertices]
    lows = [min(c[axis] for c in coordinates) for axis in range(3)]
    highs = [max(c[axis] for c in coordinates) for axis in range(3)]
    spans = [highs[axis] - lows[axis] for axis in range(3)]
    axis = max(range(3), key=lambda index: spans[index])
    others = [index for index in range(3) if index != axis]

    def girth(select) -> float:
        """Mean cross-section extent over the selected slice, 0 if empty."""
        picked = [c for c in coordinates if select(c[axis])]
        if not picked:
            return 0.0
        return sum(max(c[other] for c in picked) - min(c[other] for c in picked) for other in others) / 2.0

    quarter = spans[axis] * 0.25
    low_girth = girth(lambda value: value <= lows[axis] + quarter)
    high_girth = girth(lambda value: value >= highs[axis] - quarter)
    handle_at_low = low_girth <= high_girth
    return axis, handle_at_low, lows[axis], highs[axis]
