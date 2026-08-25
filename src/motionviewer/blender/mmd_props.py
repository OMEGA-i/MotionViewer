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
    # Drop the crate by half its height so its *top* meets the hands rather than its
    # waistline. The palm normals point forward **and down** (measured: +0.81 forward,
    # -0.50 up), and a palm facing down is resting on something, not gripping its
    # middle — so the load hangs below and in front of the hands. Centring it on the
    # wrists put half the crate under them, which reads as the hands passing through it.
    box = Prop(
        obj=build_cardboard_box(bpy, style, side=side, depth=depth, height=height),
        bone=hand_bone,
        aim_bone=other_bone,
        aim_fraction=0.5,
        world_offset=(forward[0] * push, forward[1] * push, -height / 2.0 + 0.02),
    )
    notes = [
        f"crate {side * 100:.0f}x{depth * 100:.0f}x{height * 100:.0f} cm, "
        f"hung from the wrists by its top edge, {push * 100:.0f} cm forward"
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


SCENE_BUILDERS = {
    "table_tennis": build_table_tennis_scene,
    "badminton": build_badminton_scene,
    "carry_box": build_carry_box_scene,
    "basketball": build_basketball_scene,
}

#: Which hand each preset expects, and whether it needs a static camera. Kept beside
#: the builders so ``--scene`` can be validated before Blender does any work.
SCENE_DEFAULTS = {
    "table_tennis": {"hand": "右手首", "static_camera": True},
    "badminton": {"hand": "右手首", "static_camera": True},
    "carry_box": {"hand": "左手首", "static_camera": False},
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


def _grip_point_and_axis(obj: Any, grip: str) -> tuple[Any, Any, float]:
    """Where the hand goes on this weapon, and which way the business end points.

    ``grip`` is taken from the table rather than measured; see WEAPON_BY_CHARACTER for
    the four girth profiles that show why no single geometric rule fits all of them.
    """
    from mathutils import Vector

    coordinates = [(v.co.x, v.co.y, v.co.z) for v in obj.data.vertices]
    lows = [min(c[axis] for c in coordinates) for axis in range(3)]
    highs = [max(c[axis] for c in coordinates) for axis in range(3)]
    spans = [highs[axis] - lows[axis] for axis in range(3)]
    axis = max(range(3), key=lambda index: spans[index])

    grip_local = Vector((0.0, 0.0, 0.0))
    direction = Vector((0.0, 0.0, 0.0))
    if grip == "low":
        grip_local[axis] = lows[axis]
        direction[axis] = 1.0
    elif grip == "high":
        grip_local[axis] = highs[axis]
        direction[axis] = -1.0
    else:  # mid: a staff or a bow, held at its balance point
        grip_local[axis] = (lows[axis] + highs[axis]) / 2.0
        direction[axis] = 1.0
    # Centre the grip on the other two axes, so a thick pommel is not held by its corner.
    for other in (index for index in range(3) if index != axis):
        grip_local[other] = (lows[other] + highs[other]) / 2.0
    return grip_local, direction, spans[axis]


def fit_weapon_to_hand(bpy: Any, obj: Any, armature: Any, hand_bone: str, grip: str = "low") -> dict:
    """Put the grip in the palm and point the weapon away from the character.

    Two things went wrong before this and both showed up as the same picture. The grip
    end was chosen by a geometric guess that is inverted for a sword, so the tip sat in
    the hand and the hilt pointed outward. And the weapon was aimed along the *forearm*,
    which for a raised arm points the blade back over the character's own head.

    Now the grip end comes from the table, and the weapon points along the fist axis —
    wrist to knuckles, continued outward — which is the direction a held blade actually
    runs.
    """
    from mathutils import Matrix, Vector

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
    palm = wrist + forward * (knuckles - wrist).length * 0.6

    grip_local, local_dir, length = _grip_point_and_axis(obj, grip)

    # Constrain the roll too: the blade's flat is turned across the palm rather than at
    # the camera. Leaving it to rotation_difference rendered a 2 m polearm as a rod.
    across = (world @ bones[hand_bone].matrix).to_3x3() @ Vector((1.0, 0.0, 0.0))
    side_vec = across - forward * across.dot(forward)
    if side_vec.length < 1e-6:
        side_vec = Vector((0.0, 0.0, 1.0)) - forward * forward.z
    side_vec.normalize()
    up = forward.cross(side_vec)
    target = Matrix((side_vec, up, forward)).transposed()

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
    return {
        "grip": grip,
        "length_m": length,
        "palm": tuple(round(v, 3) for v in palm),
        "tip": tuple(round(v, 3) for v in (palm + forward * length)),
    }


CLOSED_FINGER_DEGREES = {
    "１": 42.0,
    "２": 52.0,
    "３": 34.0,
}


def close_hand(bpy: Any, armature: Any, hand_bone: str) -> int:
    """Curl the fingers of the weapon hand into a fist.

    The retarget leaves the hand in its relaxed rest curl because the body-22 source
    carries no finger data. Around a sword that reads as an open palm with a blade
    balanced on it, which is what the render showed. Bending the three phalanges of the
    four fingers — the thumb is left alone, since it wraps the other way and a uniform
    curl makes it cross the grip — is enough for a fist at the size a hand occupies.
    """
    del bpy
    side = "右" if hand_bone.startswith("右") else "左"
    turned = 0
    for finger in ("人指", "中指", "薬指", "小指"):
        for segment, degrees in CLOSED_FINGER_DEGREES.items():
            name = f"{side}{finger}{segment}"
            bone = armature.pose.bones.get(name)
            if bone is None:
                continue
            bone.rotation_mode = "XYZ"
            # Negative about local X curls the finger toward the palm on this rig family.
            bone.rotation_euler.x -= math.radians(degrees)
            bone.keyframe_insert("rotation_euler", index=-1)
            turned += 1
    return turned


def _span(obj: Any, axis: int) -> float:
    values = [v.co[axis] for v in obj.data.vertices]
    return (max(values) - min(values)) if values else 0.0


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
