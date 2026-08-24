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


def build_cardboard_box(bpy: Any, style: PropStyle, *, side: float = 0.38) -> Any:
    """A carried box: five panels plus a tape strip, so the open top reads as a box.

    A solid cube would be ambiguous at this size — it could be a crate, a block, or a
    rendering error. The rim and the tape are what make it legible.
    """
    half = side / 2.0
    wall = 0.012
    parts = [
        _cube(bpy, "box_base", (side, side, wall), (0.0, 0.0, -half), style.cardboard),
        _cube(bpy, "box_x+", (wall, side, side), (half, 0.0, 0.0), style.cardboard),
        _cube(bpy, "box_x-", (wall, side, side), (-half, 0.0, 0.0), style.cardboard),
        _cube(bpy, "box_y+", (side, wall, side), (0.0, half, 0.0), style.cardboard),
        _cube(bpy, "box_y-", (side, wall, side), (0.0, -half, 0.0), style.cardboard),
        _cube(
            bpy,
            "box_tape",
            (0.07, side * 1.002, wall * 1.6),
            (0.0, 0.0, -half + wall),
            style.cardboard_tape,
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
    side: float = 0.33,
) -> SceneBuild:
    """A box held between both hands.

    A prop cannot have two parents, so it hangs off one wrist and is then slid halfway
    to the other. That works because the two wrists are effectively rigid while the box
    is carried — measured over the carry clip they sit 357 mm apart with a 48 mm
    standard deviation — so half the gap lands on the midpoint the hands actually grip.
    Parenting to the torso instead would be stabler but wrong: the same midpoint ranges
    over 0.42 x 0.47 m relative to the chest as the arms move.

    ``side`` is sized just under that spacing so the palms sit on the outer faces
    rather than inside the cardboard.
    """
    box = Prop(
        obj=build_cardboard_box(bpy, style, side=side),
        bone=hand_bone,
        offset=(0.0, 0.0, 0.0),
        euler=(0.0, 0.0, 0.0),
        aim_bone=other_bone,
        aim_fraction=0.5,
        world_offset=(0.0, 0.0, -side / 2.0),
    )
    notes = [f"box {side * 100:.0f} cm, centred between {hand_bone} and {other_bone}"]
    if not other_bone:
        notes.append("no second hand given, so the box hangs off one wrist and will drift")
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

#: Character directory -> (weapon file, which hand, whether it is a blade). Genshin and
#: Star Rail rips ship the weapon as a separate model with a handful of bones and no
#: body, which is why the probe reports it as missing every required bone. That makes it
#: a prop, and the prop machinery already knows how to follow a wrist.
WEAPON_BY_CHARACTER: dict[str, tuple[str, str, bool]] = {
    "星穹铁道—彦卿": ("武器.pmx", "右手首", True),
    "星穹铁道—流萤·春日手信": ("剑.pmx", "右手首", True),
    "星穹铁道—飞霄无外套": ("星穹铁道—飞霄武器.pmx", "右手首", True),
    "青衣": ("武器.pmx", "右手首", True),
    "星穹铁道—三月七": ("弓1.0.pmx", "左手首", False),
    "星穹铁道—火花·甜梦电波": ("手杖（整.pmx", "右手首", False),
    "星穹铁道——银狼": ("炮.pmx", "右手首", False),
    "耀嘉音": ("武器.pmx", "右手首", False),
    "星穹铁道—知更鸟": ("ｻｰﾍｲ.pmx", "右手首", False),
    "星穹铁道—花火": ("剑玉.pmx", "右手首", False),
    "奥黛塔": ("ｽ｣.pmx", "右手首", False),
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
