"""Render the retargeted character beside the SMPL-X source skeleton.

Same scene, same camera, same light, so a pose that looks wrong can be traced to
either the source motion or the retarget rather than argued about.

  blender --background --python scripts/render_mmd_compare.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
    --motion <clip>/smplx_params.npz \
    --output outputs/compare/<name> --views three_quarter,front
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_BONE_RADIUS_M = 0.022
# Column indices in SMPLX_BODY22_NAMES, used for the heading.
_HIP_L, _HIP_R = 1, 2
# Target on-screen width of the inverted-hull outline, in pixels.
_OUTLINE_PIXELS = 1.6
_JOINT_RADIUS_M = 0.032


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", default="three_quarter")
    parser.add_argument("--frames", type=int, default=0, help="0 means the whole clip")
    parser.add_argument("--resolution", type=int, default=720, help="Frame height in pixels")
    parser.add_argument(
        "--aspect",
        type=float,
        default=1.0,
        help="Width / height. A standing figure wastes half a square frame; 0.75 suits one.",
    )
    parser.add_argument(
        "--panels",
        default="skeleton,character",
        help="Which passes to render. Both share one camera so frames can be stacked.",
    )
    parser.add_argument("--faithful", action="store_true", help="Disable the polish pass")
    parser.add_argument("--caption", default="")
    parser.add_argument("--toon", action="store_true", help="Cel shading, outline, floor shadow")
    parser.add_argument("--no-outline", action="store_true", help="With --toon, skip the outline shell")
    parser.add_argument("--no-ground", action="store_true", help="With --toon, skip the floor")
    parser.add_argument(
        "--spring",
        action="store_true",
        help="Bake secondary motion onto the bones the PMX marks as dynamic",
    )
    parser.add_argument("--spring-stiffness", type=float, default=None)
    parser.add_argument("--spring-damping", type=float, default=None)
    parser.add_argument("--spring-max-angle", type=float, default=None)
    parser.add_argument("--expression", default="smile", help="Facial expression preset, or 'none'")
    parser.add_argument(
        "--frame-motion",
        action="append",
        default=None,
        type=Path,
        help=(
            "Also cover this clip's extent when placing the camera. Repeatable. "
            "Two clips rendered with each other listed here get identical framing, "
            "which is what makes a side-by-side comparison honest: without it a "
            "run-away generation is framed wide and its ground truth framed tight, "
            "so the two look like different shots rather than the same motion."
        ),
    )
    parser.add_argument(
        "--frame-pad",
        type=float,
        default=0.12,
        help="Metres of slack around the joint hull, standing in for mesh past the bones",
    )
    parser.add_argument(
        "--camera",
        default="auto",
        choices=("auto", "static", "follow"),
        help=(
            "static frames the whole trajectory, which shrinks the figure to a speck "
            "on anything that walks: a 6 m clip at 800p leaves a 100 px character. "
            "follow tracks the root horizontally at an orthographic scale set by the "
            "poses rather than the path, so the figure's size no longer depends on how "
            "far it walks. auto picks follow once travel passes --follow-threshold."
        ),
    )
    parser.add_argument(
        "--follow-threshold",
        type=float,
        default=1.2,
        help="Metres of horizontal travel above which auto switches to a follow camera",
    )
    parser.add_argument(
        "--follow-smooth",
        type=int,
        default=11,
        help="Frames of smoothing on the camera path, so it does not bob with each step",
    )
    parser.add_argument("--samples", type=int, default=64, help="EEVEE render samples per frame")
    parser.add_argument(
        "--outline-thickness",
        type=float,
        default=None,
        help=(
            "Shell thickness in metres. The shell is world-space, so a fixed value "
            "gets thicker in pixels as the resolution rises. Default is auto-scaled "
            "to hold a constant width on screen"
        ),
    )
    parser.add_argument(
        "--outline-tint",
        type=float,
        default=1.0,
        help="Multiplier on the model's own edge colours; below 1 darkens the line",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=None,
        help=(
            "Override the preset's vertical component, as a ratio of horizontal reach. "
            "three_quarter uses 0.34, about 19 degrees above horizontal. Lowering it "
            "recovers almost no face — measured on a head-level clip, iris pixels went "
            "5810 -> 5245 from 0.34 down to 0.0, because what hides a face is the "
            "character's yaw, not the camera's pitch. Below about 0.10 the camera "
            "reaches the horizon and the floor stops filling the backdrop, leaving "
            "world grey behind the character"
        ),
    )
    parser.add_argument(
        "--filter-size",
        type=float,
        default=0.9,
        help=(
            "Pixel filter width. Blender defaults to 1.5 px, which softens every edge; "
            "cel shading is all hard edges and outlines, so it loses the most from it"
        ),
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np
    from mathutils import Vector  # type: ignore

    from motionviewer.blender.camera import add_camera_for_bounds
    from motionviewer.blender.mmd_expression import apply_expression
    from motionviewer.blender.mmd_spring import SpringStyle, apply_secondary_motion
    from motionviewer.blender.mmd_toon import (
        ToonStyle,
        add_ground,
        add_outline,
        add_toon_lighting,
        apply_toon_shading,
    )
    from motionviewer.blender.render import _normalize_engine
    from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz
    from motionviewer.blender.scene import add_lighting, clear_scene, setup_world
    from motionviewer.core.smplx_fk import SMPLX_BODY22_PARENTS, source_to_blender

    toon_dir = ROOT / ".local/blender_mmd_tools/mmd_tools/externals/MikuMikuDance"
    for name in ("toon01.bmp", "toon05.bmp"):
        source = toon_dir / name
        target = ROOT / name
        if source.is_file() and not target.exists():
            target.symlink_to(source)

    views = [value.strip() for value in args.views.split(",") if value.strip()]
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]
    width = max(int(round(args.resolution * args.aspect)), 16)

    clear_scene()
    setup_world(transparent=False)

    actor = create_fbx_actor_from_npz(
        args.motion,
        label="yoimiya",
        fbx_path=args.asset,
        bone_map="mmd",
        gender="female",
        fbx_scale=0.08,
        retarget_mode="direct",
        mmd_polish={"enabled": not args.faithful},
        mmd_physics=args.spring,
        mmd_morphs=args.expression != "none",
    )
    scene = bpy.context.scene
    frame_start = int(scene.frame_start)

    with np.load(args.motion, allow_pickle=False) as payload:
        joints = np.asarray(payload["joints22"], dtype=np.float64)
    total = len(joints) if args.frames <= 0 else min(args.frames, len(joints))
    frame_end = frame_start + total - 1

    expression_info: dict = {"preset": "none"}
    if args.expression != "none":
        expression_info = apply_expression(actor.mesh_objects, args.expression)
        print(f"expression: {json.dumps(expression_info, ensure_ascii=False)}")

    transfer = json.loads(actor.armature.get("motionviewer_mmd_transfer", "{}"))
    # The retarget's own height ratio. A pelvis-height ratio would mis-scale a
    # stylised character, whose legs and head are not human proportions.
    scale = float(transfer.get("root_translation_scale", 1.0))
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    character_root = np.asarray(actor.armature.matrix_world.translation, dtype=np.float64)

    def aligned_source_points(source_joints: np.ndarray) -> np.ndarray:
        """Source joints in Blender space, placed where the character stands.

        Height-scaled, grounded, and re-based on frame 0 the same way
        ``_mmd_root_locations`` re-bases the root path, so these points track the
        retargeted character rather than the raw SMPL-X world position.
        """
        points = source_to_blender(np.asarray(source_joints, dtype=np.float64)) * scale
        points[:, :, 2] -= float(points[:, :, 2].min())
        points[:, :, 0] += float(character_root[0]) - float(points[0, 0, 0])
        points[:, :, 1] += float(character_root[1]) - float(points[0, 0, 1])
        return points

    def make_keys_linear(obj) -> None:
        """Bezier keys on a per-frame sample overshoot; sampled data wants linear.

        Blender 4.4 moved fcurves onto slotted actions, so both layouts are walked.
        """
        action = obj.animation_data.action if obj.animation_data else None
        if action is None:
            return
        curves = list(getattr(action, "fcurves", None) or [])
        if not curves:
            for layer in getattr(action, "layers", ()):
                for strip in getattr(layer, "strips", ()):
                    for channelbag in getattr(strip, "channelbags", ()):
                        curves.extend(channelbag.fcurves)
        for curve in curves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"

    # ---- source skeleton -----------------------------------------------------
    # The skeleton is built at the character's own position, not beside her:
    # both passes then share one camera, so stacking the two image sequences
    # compares the same pose from the same angle instead of two viewpoints.
    skeleton_objects: list = []
    if "skeleton" in panels:
        points = aligned_source_points(joints[:total])

        material = bpy.data.materials.new("SourceSkeleton")
        material.use_nodes = True
        principled = material.node_tree.nodes["Principled BSDF"]
        principled.inputs["Base Color"].default_value = (0.16, 0.20, 0.30, 1.0)
        principled.inputs["Roughness"].default_value = 0.55

        def add_primitive(kind: str, name: str) -> object:
            if kind == "cylinder":
                bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, vertices=12)
            else:
                bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=12, ring_count=8)
            obj = bpy.context.active_object
            obj.name = name
            obj.data.materials.append(material)
            obj.rotation_mode = "QUATERNION"
            return obj

        bones = [(child, parent) for child, parent in enumerate(SMPLX_BODY22_PARENTS) if parent >= 0]
        segments = [add_primitive("cylinder", f"src_bone_{child}") for child, _ in bones]
        joint_balls = [add_primitive("sphere", f"src_joint_{index}") for index in range(22)]
        skeleton_objects = [*segments, *joint_balls]

        for offset in range(total):
            frame = frame_start + offset
            for segment, (child, parent) in zip(segments, bones, strict=True):
                head = Vector(points[offset, parent])
                tail = Vector(points[offset, child])
                direction = tail - head
                length = max(direction.length, 1e-5)
                segment.location = (head + tail) * 0.5
                segment.rotation_quaternion = direction.to_track_quat("Z", "Y")
                segment.scale = (_BONE_RADIUS_M, _BONE_RADIUS_M, length * 0.5)
                for channel in ("location", "rotation_quaternion", "scale"):
                    segment.keyframe_insert(data_path=channel, frame=frame)
            for index, ball in enumerate(joint_balls):
                ball.location = Vector(points[offset, index])
                ball.scale = (_JOINT_RADIUS_M,) * 3
                for channel in ("location", "scale"):
                    ball.keyframe_insert(data_path=channel, frame=frame)

        for obj in skeleton_objects:
            make_keys_linear(obj)

    # ---- framing over the whole clip ----------------------------------------
    def scene_bounds(targets) -> tuple[np.ndarray, np.ndarray]:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mins = np.full(3, 1e9)
        maxs = np.full(3, -1e9)
        for obj in targets:
            evaluated = obj.evaluated_get(depsgraph)
            matrix = evaluated.matrix_world
            for corner in evaluated.bound_box:
                world = matrix @ Vector(corner)
                mins = np.minimum(mins, (world.x, world.y, world.z))
                maxs = np.maximum(maxs, (world.x, world.y, world.z))
        return mins, maxs

    def root_at(frame: int) -> np.ndarray:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        return np.asarray(actor.armature.matrix_world.translation, dtype=np.float64)

    # The character's own extent, measured relative to its root so travel does not
    # inflate it. This is what a follow camera frames, and it is identical for both
    # sides of a comparison because it depends on the character, not the clip.
    everything = [*actor.mesh_objects, *skeleton_objects]
    # Twelve samples rather than three: the tight box has to hold the widest pose in
    # the clip, and a kick or a reach between the endpoints would otherwise be
    # cropped. Each sample is one depsgraph update, and the root is already sampled
    # every frame below, so the extra cost is noise.
    samples = max(8, min(24, total))
    sample_frames = sorted(
        {frame_start + round(offset * (total - 1) / (samples - 1)) for offset in range(samples)}
        | {frame_start, frame_end}
    )
    mins = np.full(3, 1e9)
    maxs = np.full(3, -1e9)
    tight_mins = np.full(3, 1e9)
    tight_maxs = np.full(3, -1e9)
    sole_heights: list[float] = []
    for frame in sample_frames:
        root = root_at(frame)
        frame_mins, frame_maxs = scene_bounds(everything)
        sole_heights.append(float(frame_mins[2]))
        mins = np.minimum(mins, frame_mins)
        maxs = np.maximum(maxs, frame_maxs)
        local_min = frame_mins - root
        local_max = frame_maxs - root
        local_min[2] += root[2]  # keep the floor at its true height, not root-relative
        local_max[2] += root[2]
        tight_mins = np.minimum(tight_mins, local_min)
        tight_maxs = np.maximum(tight_maxs, local_max)

    # Root path of this character, and the travel that decides static vs follow.
    root_path = np.stack([root_at(frame_start + offset) for offset in range(total)])

    # The follow box and the travel are both taken from the joint hull of every clip
    # listed for framing, root-relative. Two reasons it is not the mesh box measured
    # above: the mesh is only available for the clip currently loaded, and a
    # per-clip box makes the two sides of a comparison different sizes. Measured
    # over the 39 picks38 pairs the poses alone put gt and gen 1.7% apart at the
    # median but 19.5% apart at worst, which reads as two different shots. Unioning
    # over both clips makes it identical by construction.
    hull_mins = np.full(3, 1e9)
    hull_maxs = np.full(3, -1e9)
    travel_paths: list[np.ndarray] = []
    for source in [args.motion, *(args.frame_motion or [])]:
        with np.load(source, allow_pickle=False) as source_payload:
            source_joints = np.asarray(source_payload["joints22"], dtype=np.float64)
        limit = len(source_joints) if args.frames <= 0 else min(args.frames, len(source_joints))
        points = aligned_source_points(source_joints[:limit])
        pelvis = points[:, 0, :].copy()
        pelvis[:, 2] = 0.0  # height is kept absolute so the floor stays in shot
        relative = (points - pelvis[:, None, :]).reshape(-1, 3)
        hull_mins = np.minimum(hull_mins, relative.min(axis=0))
        hull_maxs = np.maximum(hull_maxs, relative.max(axis=0))
        travel_paths.append(points[:, 0, :2] - points[0, 0, :2])
    # Floor height: the lower quartile of the per-frame lowest vertex, not its
    # minimum. See add_ground's docstring — the minimum makes the character hover in
    # every frame but one, and a visible gap under the feet reads as floating where a
    # centimetre of intersection does not.
    floor_z = float(np.percentile(sole_heights, 25)) if sole_heights else 0.0

    follow_mins = hull_mins - args.frame_pad
    follow_maxs = hull_maxs + args.frame_pad
    follow_mins[2] = min(float(follow_mins[2]), floor_z)

    # Heading, so a camera can be put in front of the character rather than in front
    # of the world. cross(up, left_hip -> right_hip) is forward: validated against
    # travel on three walking clips (dot +1.00, +0.98) and correctly disagreeing on
    # the one captioned "jogging backward" (-0.98).
    headings = []
    for source in [args.motion, *(args.frame_motion or [])]:
        with np.load(source, allow_pickle=False) as source_payload:
            source_joints = np.asarray(source_payload["joints22"], dtype=np.float64)
        limit = len(source_joints) if args.frames <= 0 else min(args.frames, len(source_joints))
        points = source_to_blender(source_joints[:limit])
        lateral = points[:, _HIP_R] - points[:, _HIP_L]
        lateral[:, 2] = 0.0
        lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-9)
        forward = np.cross(np.array([0.0, 0.0, 1.0]), lateral)
        headings.append(forward.mean(axis=0))
    heading = np.asarray(headings).mean(axis=0)
    heading[2] = 0.0
    norm = float(np.linalg.norm(heading))
    heading = heading / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])

    stacked = np.concatenate(travel_paths, axis=0)
    travel = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    follow = args.camera == "follow" or (args.camera == "auto" and travel > args.follow_threshold)
    print(f"camera: {'follow' if follow else 'static'} (travel {travel:.2f} m)")
    print(
        f"follow box from joints {np.round(follow_maxs - follow_mins, 3)}"
        f" vs from mesh {np.round(tight_maxs - tight_mins, 3)}"
    )

    if args.frame_motion:
        # Frame from the joint hull of every listed clip *including this one*, and
        # ignore the mesh bounds above. Mixing the two would leave each side of a
        # comparison framed off its own mesh and only widened by the other's
        # joints, so the two cameras would differ by however far hair and skirt
        # reach — small, but enough to make a paired figure look mismatched.
        # Joints-only is identical on both sides by construction.
        hull_mins = np.full(3, 1e9)
        hull_maxs = np.full(3, -1e9)
        for extra in [args.motion, *args.frame_motion]:
            with np.load(extra, allow_pickle=False) as extra_payload:
                extra_joints = np.asarray(extra_payload["joints22"], dtype=np.float64)
            limit = len(extra_joints) if args.frames <= 0 else min(args.frames, len(extra_joints))
            flat = aligned_source_points(extra_joints[:limit]).reshape(-1, 3)
            hull_mins = np.minimum(hull_mins, flat.min(axis=0))
            hull_maxs = np.maximum(hull_maxs, flat.max(axis=0))
        mins = hull_mins - args.frame_pad
        maxs = hull_maxs + args.frame_pad
        # Hair and hat sit above the head joint, and the floor must stay in shot.
        mins[2] = min(float(mins[2]), 0.0)
        print(
            f"shared framing over {1 + len(args.frame_motion)} clips: {np.round(mins, 3)} .. {np.round(maxs, 3)}"
        )

    outline_shells: list = []
    if args.toon:
        report = apply_toon_shading(actor.mesh_objects)
        print(
            f"toon: {len(report['shaded'])} shaded, {len(report['unlit'])} unlit, {len(report['face'])} face"
        )
        add_toon_lighting(mins.tolist(), maxs.tolist())
        if not args.no_outline:
            # Hold the outline at a constant on-screen width. Thickness is in metres,
            # so leaving it fixed made the line 1.4 px at 800p and 3.3 px at 1920p —
            # the same setting reading as a fine line in one render and a heavy border
            # in another, and thick enough at 1920p to show through the mouth opening.
            character_height = float(follow_maxs[2] - follow_mins[2]) or 1.6
            pixels_per_metre = args.resolution / max(character_height * 1.35, 1e-6)
            thickness = (
                args.outline_thickness
                if args.outline_thickness is not None
                else _OUTLINE_PIXELS / pixels_per_metre
            )
            print(f"outline thickness {thickness * 1000:.2f} mm (~{_OUTLINE_PIXELS:.1f} px)")
            outline_shells = add_outline(
                actor.mesh_objects,
                style=ToonStyle(outline_thickness_m=thickness, outline_tint=args.outline_tint),
            )
        if not args.no_ground:
            # Only a follow camera needs the grid; a static shot already shows
            # travel as the figure crossing the frame.
            add_ground(
                mins.tolist(),
                maxs.tolist(),
                grid_metres=0.5 if follow else 0.0,
                plane_z=floor_z,
            )
    else:
        add_lighting(mins.tolist(), maxs.tolist())
    spring_info: dict = {"chains": 0}
    if args.spring:
        defaults = SpringStyle()
        style = SpringStyle(
            stiffness=defaults.stiffness if args.spring_stiffness is None else args.spring_stiffness,
            damping=defaults.damping if args.spring_damping is None else args.spring_damping,
            max_angle_degrees=(
                defaults.max_angle_degrees if args.spring_max_angle is None else args.spring_max_angle
            ),
        )
        spring_info = apply_secondary_motion(
            bpy, actor.armature, frame_start=frame_start, num_frames=total, style=style
        )
        print(f"spring: {json.dumps(spring_info, ensure_ascii=False)}")
    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = width
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filter_size = args.filter_size
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    args.output.mkdir(parents=True, exist_ok=True)

    for view in views:
        for camera in [obj for obj in bpy.data.objects if obj.type == "CAMERA"]:
            bpy.data.objects.remove(camera, do_unlink=True)
        # character_front / character_3q aim at the character's own heading; every
        # other view is a fixed world direction from the preset table.
        view_direction = None
        preset = view
        if view.startswith("character"):
            yaw = np.radians(0.0 if view == "character_front" else 32.0)
            cos, sin = np.cos(yaw), np.sin(yaw)
            rotated = np.array(
                [heading[0] * cos - heading[1] * sin, heading[0] * sin + heading[1] * cos, 0.0]
            )
            lift = args.camera_elevation if args.camera_elevation is not None else 0.18
            view_direction = (float(rotated[0]), float(rotated[1]), float(lift))
            preset = "three_quarter"
            print(f"{view}: heading {np.round(heading, 3)} -> camera direction {np.round(view_direction, 3)}")
        if follow:
            # Frame the character alone, then move the camera with it. The camera is
            # orthographic, so apparent size comes from ortho_scale and not from
            # distance: sizing the box from the poses rather than the path decouples
            # the figure's size from how far it walks, and unioning it over both
            # clips of a pair makes the two sides identical.
            camera = add_camera_for_bounds(
                (follow_mins + root_path[0] * (1, 1, 0)).tolist(),
                (follow_maxs + root_path[0] * (1, 1, 0)).tolist(),
                preset=preset,
                margin=1.15,
                resolution=(width, args.resolution),
                elevation=args.camera_elevation if view_direction is None else None,
                direction=view_direction,
            )
            offsets = root_path[:, :2] - root_path[0, :2]
            if args.follow_smooth > 1:
                # A root bobs sideways with every step; a camera that copies it makes
                # the whole shot wobble. Smoothing leaves the sway on the character.
                pad = args.follow_smooth // 2
                padded = np.concatenate(
                    [np.repeat(offsets[:1], pad, axis=0), offsets, np.repeat(offsets[-1:], pad, axis=0)]
                )
                kernel = np.ones(args.follow_smooth) / args.follow_smooth
                offsets = np.stack(
                    [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)], axis=1
                )
            base = np.asarray(camera.location, dtype=np.float64)
            for offset in range(total):
                # Vertical is deliberately not followed: a camera that tracks Z
                # cancels the jump it is meant to show.
                camera.location = (
                    base[0] + offsets[offset, 0],
                    base[1] + offsets[offset, 1],
                    base[2],
                )
                camera.keyframe_insert(data_path="location", frame=frame_start + offset)
            make_keys_linear(camera)
        else:
            add_camera_for_bounds(
                mins.tolist(),
                maxs.tolist(),
                preset=preset,
                margin=1.1,
                resolution=(width, args.resolution),
                elevation=args.camera_elevation if view_direction is None else None,
                direction=view_direction,
            )
        for panel in panels:
            # The outline shell is a separate object, so hiding only the
            # character meshes leaves a black silhouette in the skeleton panel.
            for obj in (*actor.mesh_objects, *outline_shells):
                obj.hide_render = panel != "character"
            for obj in skeleton_objects:
                obj.hide_render = panel != "skeleton"
            directory = args.output / panel / view
            directory.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(directory / "frame_")
            bpy.ops.render.render(animation=True)
            print(f"rendered {panel}/{view}: {total} frames")

    (args.output / "info.json").write_text(
        json.dumps(
            {
                "motion": str(args.motion),
                "asset": str(args.asset),
                "caption": args.caption,
                "frames": total,
                "views": views,
                "panels": panels,
                "faithful": bool(args.faithful),
                "toon": bool(args.toon),
                "spring": spring_info,
                "expression": expression_info,
                "transfer": transfer.get("polish", {}),
                "root_translation_scale": scale,
                "bounds": [mins.tolist(), maxs.tolist()],
                "shared_framing": [str(path) for path in (args.frame_motion or [])],
                "camera": "follow" if follow else "static",
                "floor_z": floor_z,
                "sole_heights_sampled": [round(v, 4) for v in sole_heights],
                "travel_m": travel,
                "follow_bounds": [follow_mins.tolist(), follow_maxs.tolist()],
                "mesh_bounds_root_relative": [tight_mins.tolist(), tight_maxs.tolist()],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
