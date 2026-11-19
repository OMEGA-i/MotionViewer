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
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_BONE_RADIUS_M = 0.022
# Column indices in SMPLX_BODY22_NAMES, used for the heading.
_HIP_L, _HIP_R = 1, 2
# Same two joints in the SOMA-77 tree, used for the heading and the framing hull.
_SOMA_HIP_L, _SOMA_HIP_R = 67, 72
# Target on-screen width of the inverted-hull outline, in pixels.
_OUTLINE_PIXELS = 1.6
_JOINT_RADIUS_M = 0.032


def prepare_upper_smoothed_motion(source: Path, output: Path, window: int) -> tuple[Path, dict]:
    """Write a complete derived NPZ; never modify the user's source motion."""
    import numpy as np

    from motionviewer.core.motion_smoothing import repose_joints, smooth_upper_body

    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {"enabled": False, "source": str(source.resolve()), "source_sha256": source_hash}
    if window == 0:
        return source, provenance
    destination = output / "processed-motion.npz"
    if destination.resolve() == source.resolve():
        raise ValueError("processed-motion.npz would overwrite the source; choose a different output directory")
    with np.load(source, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    original_pose = arrays["body_pose"]
    roots, pose, translation, report = smooth_upper_body(
        arrays["global_orient"], original_pose, arrays["transl"],
        fps=float(arrays.get("fps", 30.0)), window=window,
    )
    arrays["joints22"] = repose_joints(roots, original_pose, pose, arrays["joints22"])
    arrays.update(global_orient=roots, body_pose=pose, transl=translation)
    report.update(source=provenance["source"], source_sha256=source_hash,
                  processed_motion=str(destination.resolve()))
    arrays["motionviewer_source_path"] = np.str_(provenance["source"])
    arrays["motionviewer_source_sha256"] = np.str_(source_hash)
    arrays["motionviewer_upper_smoothing_report"] = np.str_(json.dumps(report, ensure_ascii=False))
    output.mkdir(parents=True, exist_ok=True)
    # A fixed staging filename could itself be the user's source. Reserve a
    # unique file so even an input called processed-motion.tmp.npz stays intact.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".processed-motion-", suffix=".npz", dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, default=None, help="SMPL-X body-22 clip")
    parser.add_argument(
        "--soma",
        type=Path,
        default=None,
        help=(
            "Drive from a SOMA-77 clip instead of a fitted SMPL-X one. Same shading, "
            "camera and expression; the body solve is identical and the fingers become "
            "driven. Skips the 26-28 mm RMSE of the SMPL-X fit"
        ),
    )
    parser.add_argument("--no-fingers", action="store_true", help="With --soma, leave the hands at bind")
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
    pose_style = parser.add_mutually_exclusive_group()
    pose_style.add_argument("--faithful", dest="faithful", action="store_true", default=True,
                            help="Default: retain mapped source rotations without pose polish")
    pose_style.add_argument("--polish", dest="faithful", action="store_false",
                            help="Explicit artistic changes to shoulders, arm swing and twist")
    parser.add_argument("--root-motion", choices=("source", "legacy-scaled"), default="source",
                        help="Default preserves source root deltas in metres; legacy-scaled replays earlier reviews")
    parser.add_argument("--caption", default="")
    parser.add_argument("--toon", action="store_true", help="Cel shading, outline, floor shadow")
    parser.add_argument("--no-outline", action="store_true", help="With --toon, skip the outline shell")
    parser.add_argument("--no-ground", action="store_true", help="With --toon, skip the floor")
    parser.add_argument("--foot-contact", choices=("on", "off"), default="off",
                        help="Optional bounded contact correction using the character's shoes")
    parser.add_argument("--floor-z", type=float, default=None, help="Explicit world-space floor height in metres")
    parser.add_argument("--upper-smooth", type=int, default=0,
                        help="0 disables; 3 enables upper-body SO(3) smoothing, capped at 2 deg per LOCAL joint")
    parser.add_argument("--arm-clearance-degrees", type=float, default=0.0,
                        help="Optional 0..4 degree upper-arm clearance; only validated for walking, not recommended for dance")
    parser.add_argument("--match-camera", type=Path, default=None,
                        help="Reuse per-frame cameras from a baseline info.json; same views/frames/resolution required")
    parser.add_argument("--physics-profile", type=Path, default=None,
                        help="Asset-hash-bound clothing settings; requires --bullet and overrides only listed physics fields")
    parser.add_argument(
        "--scene",
        default="",
        help=(
            "Add props and a set: table_tennis, badminton, carry_box, basketball, "
            "carry_box_supported (box-only placement above existing hands), "
            "carry_box_two_hand (optional hand-contact adaptation). "
            "A set is placed from the character's heading because the motion carries "
            "no equipment, so check the result before trusting it."
        ),
    )
    parser.add_argument(
        "--stage",
        default="",
        help=(
            "Import a community MMD stage PMX as the environment. Stages are the format "
            "the MMD community distributes sets in and this project's PMX importer "
            "already reads them, so any stage file drops in. Implies a static camera."
        ),
    )
    parser.add_argument("--stage-scale", type=float, default=1.0)
    parser.add_argument(
        "--weapon",
        default="",
        help=(
            "Hold this character's own weapon model. 'auto' looks the character up in "
            "WEAPON_BY_CHARACTER by its asset directory; a path uses that PMX. These "
            "rips ship the weapon as a separate model, so it is a prop on the wrist."
        ),
    )
    parser.add_argument(
        "--weapon-hand",
        default="",
        help="Override the hand the weapon hangs from, e.g. 左手首",
    )
    parser.add_argument("--weapon-fit-mode", choices=("legacy", "bounded", "surface"), default="legacy",
                        help="legacy keeps the old curl; bounded limits flexion; surface requires --toon and uses a verified asset-specific hand/weapon calibration")
    parser.add_argument("--weapon-arm-clearance", action="store_true",
                        help="Explicitly adapt the non-gripping arm for surface weapons; changes the generated pose")
    parser.add_argument("--override-source-fingers", action="store_true",
                        help="Explicitly replace native SOMA fingers with weapon grip poses; normally source fingers are kept")
    parser.add_argument("--weapon-grip-section", choices=("low", "high"), default=None,
                        help="Choose the legacy/bounded grip region (not a slide distance); surface uses its verified profile region. Default: auto catalog or low for explicit paths")
    parser.add_argument(
        "--weapon-grip",
        type=float,
        default=0.0,
        help="Deprecated unused offset; only 0 is accepted. Use --weapon-grip-section to select a grip region",
    )
    parser.add_argument(
        "--scene-hand",
        default="",
        help="Override which bone the prop hangs from (default per scene, e.g. 右手首)",
    )
    parser.add_argument("--scene-grip-mode", choices=("side", "bottom"), default="side",
                        help="For carry_box_two_hand, fit side or bottom hand contacts")
    parser.add_argument("--scene-fit-start", type=int, default=None,
                        help="Inclusive first Blender frame used to fit a two-hand box")
    parser.add_argument("--scene-fit-end", type=int, default=None,
                        help="Inclusive last Blender frame used to fit a two-hand box")
    parser.add_argument("--scene-contact", action="store_true",
                        help="Requires --toon; fit both arms, wrists and fingers to the two-hand box using evaluated hand skin")
    parser.add_argument("--scene-box-size", type=float, nargs=3, default=None,
                        metavar=("WIDTH", "DEPTH", "HEIGHT"),
                        help="Fixed dimensions in metres for carry_box_two_hand; never animated scale")
    parser.add_argument(
        "--scene-reach",
        type=float,
        default=3.0,
        help="Metres of set around the player that framing must cover",
    )
    parser.add_argument(
        "--scene-standoff",
        type=float,
        default=None,
        help="Metres from the player to the table's near edge / the net",
    )
    parser.add_argument(
        "--bullet",
        action="store_true",
        help=(
            "Simulate the PMX's authored rigid bodies and joints. Collision coverage "
            "depends on the model and still needs visual checking. Costs one physics "
            "bake per clip; mutually exclusive with --spring, which has no collisions."
        ),
    )
    parser.add_argument("--bullet-substeps", type=int, default=12)
    parser.add_argument("--bullet-iterations", type=int, default=24)
    parser.add_argument("--bullet-cloth-slack", type=float, default=0.,
                        help="Optional angular slack in degrees for originally locked skirt joint axes")
    parser.add_argument("--bullet-cloth-linear-slack", type=float, default=0.,
                        help="Optional 0..0.01 metre travel for originally locked skirt joint axes; validate per character")
    parser.add_argument("--bullet-hand-colliders", action="store_true",
                        help="Fit extra palm/finger collision hulls from skin; only interact with skirt bodies")
    parser.add_argument(
        "--smplx-model",
        type=Path,
        default=None,
        help=(
            "SMPL-X model .npz (e.g. SMPLX_NEUTRAL_2020.npz) for the 'body' panel: the original "
            "generated human as a real body mesh, posed from the clip's SMPL-X parameters"
        ),
    )
    parser.add_argument(
        "--source-smplx",
        type=Path,
        default=None,
        help=(
            "SMPL-X clip to pose the 'body' panel from. Defaults to --motion, or with --soma to "
            "the dataset's sibling smplx/<split>/<clip>.smplx.npz"
        ),
    )
    parser.add_argument("--bullet-neutral-settle", action="store_true",
                        help="Settle cloth with legs at rest, then ease them into the first pose before frame 1")
    parser.add_argument("--bullet-smooth", type=float, default=0.0,
                        help="After baking, Gaussian-smooth cloth tracks over this many frames (sigma) to remove flicker; ~1.5")
    parser.add_argument("--bullet-despike", action="store_true",
                        help="After baking, replace whole-panel solver snaps with smooth bridges; swing is kept")
    parser.add_argument(
        "--bullet-damping",
        choices=("pmx", "legacy-inverted"),
        default="legacy-inverted",
        help=(
            "pmx preserves the model's authored rigid-body damping. legacy-inverted "
            "reproduces the older showcase's 1-d tuning; it is an artistic override, "
            "not a conversion between PMX and Blender units."
        ),
    )
    parser.add_argument(
        "--bullet-settle",
        type=int,
        default=30,
        help="Frames simulated before the first rendered one, so cloth starts hanging rather than dropping on camera",
    )
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
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if (args.motion is None) == (args.soma is None):
        parser.error("give exactly one of --motion (SMPL-X) or --soma (SOMA-77)")
    if args.spring and args.bullet:
        parser.error("--spring approximates what --bullet simulates; pick one")
    if args.physics_profile is not None and not args.bullet:
        parser.error("--physics-profile requires --bullet")
    if args.upper_smooth < 0 or (args.upper_smooth > 0 and args.upper_smooth % 2 == 0):
        parser.error("--upper-smooth must be 0 or a positive odd frame count")
    if args.soma is not None and args.upper_smooth:
        parser.error("--upper-smooth currently supports --motion SMPL-X body-22 only, not SOMA-77")
    if args.floor_z is not None and not math.isfinite(args.floor_z):
        parser.error("--floor-z must be finite")
    if not math.isfinite(args.arm_clearance_degrees) or not 0.0 <= args.arm_clearance_degrees <= 20.0:
        parser.error("--arm-clearance-degrees must be finite and between 0 and 20 degrees")
    if args.scene_fit_start is not None and args.scene_fit_end is not None and args.scene_fit_start > args.scene_fit_end:
        parser.error("--scene-fit-start must not exceed --scene-fit-end")
    if args.scene_contact and args.scene != "carry_box_two_hand":
        parser.error("--scene-contact requires --scene carry_box_two_hand")
    if not args.toon and (args.scene_contact or (args.weapon and args.weapon_fit_mode == "surface")):
        parser.error("--scene-contact and surface weapons require --toon")
    if args.weapon_grip != 0.0:
        parser.error("--weapon-grip offsets were never applied; use --weapon-grip-section or a calibrated fit")
    if args.weapon_arm_clearance and (not args.weapon or args.weapon_fit_mode != "surface"):
        parser.error("--weapon-arm-clearance requires a surface weapon")
    if args.override_source_fingers and not args.weapon:
        parser.error("--override-source-fingers requires --weapon")
    if args.scene_contact and args.scene_grip_mode != "side":
        parser.error("--scene-contact currently requires --scene-grip-mode side")
    if not math.isfinite(args.bullet_cloth_slack) or not 0. <= args.bullet_cloth_slack <= 30.:
        parser.error("--bullet-cloth-slack must be between 0 and 30 degrees")
    if not math.isfinite(args.bullet_cloth_linear_slack) or not 0. <= args.bullet_cloth_linear_slack <= .01:
        parser.error("--bullet-cloth-linear-slack must be between 0 and 0.01 metres")
    if args.scene_box_size is not None:
        if args.scene not in {"carry_box_two_hand", "carry_box_supported"}:
            parser.error("--scene-box-size requires --scene carry_box_two_hand or carry_box_supported")
        if any(not math.isfinite(value) or value <= 0 for value in args.scene_box_size):
            parser.error("--scene-box-size dimensions must be finite and positive")
    return args


def main() -> None:
    args = parse_args()

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
    from motionviewer.blender.review_camera import (
        apply_camera_track,
        capture_camera_track,
        load_camera_tracks,
    )
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
    original_source_path = args.soma if args.soma is not None else args.motion
    physics_profile = None
    if args.physics_profile is not None:
        from motionviewer.assets.physics_profiles import resolve_physics_profile

        physics_profile = resolve_physics_profile(args.physics_profile, asset_path=args.asset, import_scale=0.08)
    original_framing_paths = list(args.frame_motion or [])
    source_path, smoothing_info = prepare_upper_smoothed_motion(original_source_path, args.output, args.upper_smooth)
    if args.upper_smooth:
        print(f"upper smoothing: {json.dumps(smoothing_info, ensure_ascii=False)}")
        # A comparison must frame the same pair on both sides. Mixing this
        # smoothed clip with the other raw clip gives two different joint hulls.
        args.frame_motion = [
            prepare_upper_smoothed_motion(path, args.output / "framing" / str(index), args.upper_smooth)[0]
            for index, path in enumerate(original_framing_paths)
        ]

    clear_scene()
    setup_world(transparent=False)

    if args.soma is not None:
        from motionviewer.blender.retarget.soma_pipeline import create_soma_actor_from_npz

        actor = create_soma_actor_from_npz(
            source_path,
            label="yoimiya",
            pmx_path=args.asset,
            pmx_scale=0.08,
            drive_fingers=not args.no_fingers,
            polish={"enabled": not args.faithful, "root_motion": args.root_motion},
            physics=args.spring or args.bullet,
            morphs=args.expression != "none",
        )
    else:
        actor = create_fbx_actor_from_npz(
            source_path,
            label="yoimiya",
            fbx_path=args.asset,
            bone_map="mmd",
            gender="female",
            fbx_scale=0.08,
            retarget_mode="direct",
            mmd_polish={"enabled": not args.faithful, "root_motion": args.root_motion},
            mmd_physics=args.spring or args.bullet,
            mmd_morphs=args.expression != "none",
        )
    scene = bpy.context.scene
    frame_start = int(scene.frame_start)

    # One reader for both source formats: the camera and framing only need joint
    # positions and the two hip joints, which both layouts provide.
    joint_key = "joints77" if args.soma is not None else "joints22"
    hip_left, hip_right = (_SOMA_HIP_L, _SOMA_HIP_R) if args.soma is not None else (_HIP_L, _HIP_R)

    def source_joints_of(path: Path) -> np.ndarray:
        with np.load(path, allow_pickle=False) as source_payload:
            return np.asarray(source_payload[joint_key], dtype=np.float64)

    joints = source_joints_of(source_path)
    total = len(joints) if args.frames <= 0 else min(args.frames, len(joints))
    if total < 1:
        raise ValueError("source motion has no frames to render")
    frame_end = frame_start + total - 1

    # The scene's frame rate had never been set, so it sat at Blender's default 24 while
    # the motion is baked one keyframe per source frame at 50 fps. That only mattered
    # once physics arrived: Bullet takes its timestep from the scene rate, so the bones
    # advanced a 50 fps step inside a 24 fps tick and flung the cloth apart — the skirt
    # came out torn rather than draped. Encoding never noticed because ffmpeg is told
    # the rate on its own command line.
    with np.load(source_path, allow_pickle=False) as fps_payload:
        source_fps = float(fps_payload["fps"]) if "fps" in fps_payload else 30.0
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"invalid source frame rate: {source_fps}")
    scene.render.fps = max(1, int(round(source_fps)))
    scene.render.fps_base = scene.render.fps / source_fps
    print(f"scene frame rate: {scene.render.fps / scene.render.fps_base:g} fps (from the clip)")

    matched_tracks: dict = {}
    matched_mode = ""
    if args.match_camera is not None:
        matched_tracks, matched_mode = load_camera_tracks(
            args.match_camera, views=views, frame_start=frame_start, frames=total,
            resolution=(width, args.resolution), fps=source_fps,
        )
    foot_contact_info: dict = {"enabled": False, "floor_z": args.floor_z}
    if args.foot_contact == "on":
        from motionviewer.blender.mmd_contact import apply_foot_contact

        foot_contact_info = apply_foot_contact(
            bpy, actor, source_path, frame_start=frame_start, frames=total, floor_z=args.floor_z,
        )
        print(f"foot contact: {json.dumps(foot_contact_info, ensure_ascii=False)}")

    arm_clearance_info: dict = {"enabled": False, "max_degrees": args.arm_clearance_degrees}
    if args.arm_clearance_degrees:
        from motionviewer.blender.mmd_arm_clearance import apply_arm_clearance

        arm_clearance_info = apply_arm_clearance(
            bpy, actor.armature, frame_start=frame_start, frame_end=frame_end,
            degrees=args.arm_clearance_degrees,
        )
        print(f"arm clearance: {json.dumps(arm_clearance_info, ensure_ascii=False)}")

    expression_info: dict = {"preset": "none"}
    if args.expression != "none":
        expression_info = apply_expression(actor.mesh_objects, args.expression)
        print(f"expression: {json.dumps(expression_info, ensure_ascii=False)}")

    transfer = json.loads(actor.armature.get("motionviewer_mmd_transfer", "{}"))
    # Source metres by default; a historical height ratio is explicit opt-in.
    scale = float(transfer.get("root_translation_scale", 1.0))
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    character_root = np.asarray(actor.armature.matrix_world.translation, dtype=np.float64)

    def aligned_source_points(source_joints: np.ndarray) -> np.ndarray:
        """Source joints in Blender space, placed where the character stands.

        Grounded and re-based on frame 0, with the selected root scale, the same way
        ``_mmd_root_locations`` re-bases the root path, so these points track the
        retargeted character rather than the raw SMPL-X world position.
        """
        points = source_to_blender(np.asarray(source_joints, dtype=np.float64)) * scale
        points[:, :, 2] -= float(points[:, :, 2].min())
        points[:, :, 0] += float(character_root[0]) - float(points[0, 0, 0])
        points[:, :, 1] += float(character_root[1]) - float(points[0, 0, 1])
        return points

    def build_source_body():
        from motionviewer.core.smplx_body import load_smplx_model, pose_frames

        if args.smplx_model is None:
            raise SystemExit("--panels body needs --smplx-model PATH (SMPLX_NEUTRAL_2020.npz)")
        clip_path = args.source_smplx
        if clip_path is None and args.soma is None:
            clip_path = args.motion
        if clip_path is None:
            # .../soma/<split>/<clip>.soma.npz  ->  .../smplx/<split>/<clip>.smplx.npz
            soma = Path(args.soma)
            clip_path = soma.parents[2] / "smplx" / soma.parent.name / soma.name.replace(".soma.npz", ".smplx.npz")
        if not Path(clip_path).is_file():
            raise SystemExit(f"--panels body: SMPL-X clip not found: {clip_path} (pass --source-smplx)")
        model = load_smplx_model(args.smplx_model)
        with np.load(clip_path, allow_pickle=False) as clip:
            betas = np.asarray(clip["betas"]) if "betas" in clip.files else np.zeros(10)
            vertices, body_joints = pose_frames(
                model, betas, clip["global_orient"][:total], clip["body_pose"][:total], clip["transl"][:total]
            )
        vertices = source_to_blender(vertices) * scale
        body_joints = source_to_blender(body_joints[:, :22]) * scale
        # Same placement as aligned_source_points, but grounded on the body's soles.
        shift = np.array([
            float(character_root[0]) - float(body_joints[0, 0, 0]),
            float(character_root[1]) - float(body_joints[0, 0, 1]),
            -float(vertices[:, :, 2].min()),
        ])
        vertices = vertices + shift

        mesh = bpy.data.meshes.new("SourceBody")
        mesh.from_pydata(vertices[0].tolist(), [], model.faces.tolist())
        mesh.update()
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        obj = bpy.data.objects.new("SourceBody", mesh)
        scene.collection.objects.link(obj)
        material = bpy.data.materials.new("SourceBodySkin")
        material.use_nodes = True
        principled = material.node_tree.nodes["Principled BSDF"]
        principled.inputs["Base Color"].default_value = (0.62, 0.66, 0.74, 1.0)
        principled.inputs["Roughness"].default_value = 0.6
        mesh.materials.append(material)

        obj.shape_key_add(name="Basis")
        flat = vertices.reshape(len(vertices), -1).astype(np.float32)
        for offset in range(len(vertices)):
            key = obj.shape_key_add(name=f"F{offset:05d}", from_mix=False)
            key.data.foreach_set("co", flat[offset])
            frame = frame_start + offset
            for delta, value in ((-1, 0.0), (0, 1.0), (1, 0.0)):
                key.value = value
                key.keyframe_insert("value", frame=frame + delta)
        from motionviewer.blender.retarget._animate import _linearize

        _linearize(mesh.shape_keys)
        info = {
            "enabled": True,
            "model": str(args.smplx_model),
            "clip": str(clip_path),
            "frames": len(vertices),
            "vertices": int(vertices.shape[1]),
            "betas_nonzero": int(np.count_nonzero(betas)),
            "note": "SMPL-X body joints only; hands, jaw, eyes at rest. The SMPL-X file is a fit to the "
                    "SOMA source (~27 mm RMSE), so this body can differ slightly from the SOMA skeleton.",
        }
        print(f"source body: {json.dumps(info, ensure_ascii=False)}", flush=True)
        return [obj], info

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
    # ---- source body (the original generated human, SMPL-X mesh) -------------
    # Placed with the same grounding and frame-0 re-basing as the skeleton below, but
    # computed from the body's own joints so the mesh and the stick figure coincide.
    body_objects: list = []
    body_info: dict = {"enabled": False}
    if "body" in panels:
        body_objects, body_info = build_source_body()

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

        if args.soma is not None:
            with np.load(source_path, allow_pickle=False) as soma_payload:
                source_parents = np.asarray(soma_payload["joint_parents"], dtype=int)
        else:
            source_parents = np.asarray(SMPLX_BODY22_PARENTS, dtype=int)
        bones = [(child, int(parent)) for child, parent in enumerate(source_parents) if parent >= 0]
        segments = [add_primitive("cylinder", f"src_bone_{child}") for child, _ in bones]
        joint_balls = [add_primitive("sphere", f"src_joint_{index}") for index in range(len(source_parents))]
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
    everything = [*actor.mesh_objects, *skeleton_objects, *body_objects]
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
    bounds_heights: list[float] = []
    for frame in sample_frames:
        root = root_at(frame)
        frame_mins, frame_maxs = scene_bounds(everything)
        bounds_heights.append(float(frame_mins[2]))
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
    for source in [source_path, *(args.frame_motion or [])]:
        source_joints = source_joints_of(source)
        limit = len(source_joints) if args.frames <= 0 else min(args.frames, len(source_joints))
        points = aligned_source_points(source_joints[:limit])
        pelvis = points[:, 0, :].copy()
        pelvis[:, 2] = 0.0  # height is kept absolute so the floor stays in shot
        relative = (points - pelvis[:, None, :]).reshape(-1, 3)
        hull_mins = np.minimum(hull_mins, relative.min(axis=0))
        hull_maxs = np.maximum(hull_maxs, relative.max(axis=0))
        travel_paths.append(points[:, 0, :2] - points[0, 0, :2])
    floor_override = foot_contact_info.get("floor_z")
    if floor_override is not None:
        floor_z = float(floor_override)
    else:
        from motionviewer.blender.mmd_contact import estimate_display_floor

        floor_estimation = estimate_display_floor(
            bpy, actor, frame_start=frame_start, frames=total,
            fallback_floor=float(np.percentile(bounds_heights, 25)) if bounds_heights else 0.0,
        )
        floor_z = floor_estimation["floor_z"]
        foot_contact_info["floor_estimation"] = floor_estimation
    foot_contact_info["floor_z"] = floor_z

    follow_mins = hull_mins - args.frame_pad
    follow_maxs = hull_maxs + args.frame_pad
    follow_mins[2] = min(float(follow_mins[2]), floor_z)

    # Heading, so a camera can be put in front of the character rather than in front
    # of the world. cross(up, left_hip -> right_hip) is forward: validated against
    # travel on three walking clips (dot +1.00, +0.98) and correctly disagreeing on
    # the one captioned "jogging backward" (-0.98).
    headings = []
    for source in [source_path, *(args.frame_motion or [])]:
        source_joints = source_joints_of(source)
        limit = len(source_joints) if args.frames <= 0 else min(args.frames, len(source_joints))
        points = source_to_blender(source_joints[:limit])
        lateral = points[:, hip_right] - points[:, hip_left]
        lateral[:, 2] = 0.0
        lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-9)
        forward = np.cross(np.array([0.0, 0.0, 1.0]), lateral)
        headings.append(forward.mean(axis=0))
    heading = np.asarray(headings).mean(axis=0)
    heading[2] = 0.0
    norm = float(np.linalg.norm(heading))
    heading = heading / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])

    # Where a set should be centred: the mean standing spot, not frame 0. Frame 0 is
    # the conditioning pose and can sit a step away from where the action happens, and
    # a table placed off that lands visibly to one side of the player.
    primary_points = aligned_source_points(
        source_joints_of(source_path)[: (None if args.frames <= 0 else args.frames)]
    )
    stand_origin = primary_points[:, 0, :2].mean(axis=0)

    stacked = np.concatenate(travel_paths, axis=0)
    travel = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    follow = args.camera == "follow" or (args.camera == "auto" and travel > args.follow_threshold)
    # A set is world-fixed, so a tracking camera slides it across the frame — the one
    # artefact that gives away a composited prop immediately. An explicit --camera
    # still wins, so this can be overridden when the drift is wanted.
    if (args.scene or args.stage) and args.camera == "auto":
        from motionviewer.blender.mmd_props import SCENE_DEFAULTS

        wants_static = bool(args.stage) or SCENE_DEFAULTS.get(args.scene, {}).get("static_camera")
        if wants_static and follow:
            print("camera: forcing static (a world-fixed set would drift under a tracking shot)")
            follow = False
    if matched_tracks:
        follow = matched_mode == "follow"
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
        for extra in [source_path, *args.frame_motion]:
            extra_joints = source_joints_of(extra)
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
    scene_box_colliders: tuple = ()
    scene_weapon_colliders: tuple = ()
    framing_weapon = None
    scene_info: dict = {"name": args.scene, "report": {}}
    weapon_info: dict = {"requested": args.weapon, "fit_mode": args.weapon_fit_mode}
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
        if args.scene:
            from motionviewer.blender.mmd_props import (
                SCENE_BUILDERS,
                SCENE_DEFAULTS,
                PropStyle,
                attach_to_bone,
            )

            if args.scene not in SCENE_BUILDERS:
                raise SystemExit(f"unknown --scene {args.scene!r}; have {sorted(SCENE_BUILDERS)}")
            hand = args.scene_hand or SCENE_DEFAULTS[args.scene]["hand"]
            builder = SCENE_BUILDERS[args.scene]
            kwargs: dict = {"hand_bone": hand}
            if args.scene == "carry_box":
                kwargs["forward"] = (float(heading[0]), float(heading[1]))
            if args.scene == "carry_box_two_hand":
                kwargs.update(
                    armature=actor.armature, frame_start=frame_start, frames=total,
                    grip_mode=args.scene_grip_mode, fit_start=args.scene_fit_start, fit_end=args.scene_fit_end,
                    contact=args.scene_contact, mesh_objects=actor.mesh_objects,
                )
                if args.scene_box_size is not None:
                    kwargs.update(zip(("side", "depth", "height"), args.scene_box_size, strict=True))
            if args.scene == "carry_box_supported":
                kwargs.update(armature=actor.armature, mesh_objects=actor.mesh_objects,
                              frame_start=frame_start, frames=total)
                if args.scene_box_size is not None:
                    kwargs.update(zip(("side", "depth", "height"), args.scene_box_size, strict=True))
            if args.scene in ("table_tennis", "badminton"):
                kwargs.update(
                    origin=(float(stand_origin[0]), float(stand_origin[1])),
                    heading=(float(heading[0]), float(heading[1])),
                    floor_z=floor_z,
                )
                if args.scene_standoff is not None:
                    kwargs["stand_off"] = args.scene_standoff
            build = builder(bpy, PropStyle(), **kwargs)
            if args.scene in {"carry_box_two_hand", "carry_box_supported"}:
                scene_box_colliders = tuple(prop.obj for prop in build.props)
            scene_info["report"] = build.report
            missing = [p.bone for p in build.props if not attach_to_bone(bpy, p, actor.armature)]
            if missing:
                raise SystemExit(f"--scene {args.scene}: rig has no bone {missing}")
            # Props go through the character's shader and outline so they do not read
            # as geometry from a different renderer sitting in the same frame.
            apply_toon_shading(build.objects)
            if not args.no_outline:
                add_outline(build.objects, style=ToonStyle(outline_thickness_m=thickness))
            print(
                f"scene {args.scene}: {len(build.props)} prop(s) on {hand}, {len(build.static)} set piece(s)"
            )
            # Anchor check, because a mis-parented prop renders as a shape floating in
            # the corner of the frame and reads as a framing bug rather than a bad
            # parent inverse. A hand prop should sit centimetres from the wrist.
            bpy.context.view_layer.update()
            for prop in build.props:
                if not prop.bone:
                    continue
                # The *posed* head, not ``data.bones[...].head_local``: a bone parent
                # follows the pose, so comparing against the rest position reports a
                # gap the size of the pose itself and looks like a parenting bug.
                posed = actor.armature.pose.bones[prop.bone]
                head = actor.armature.matrix_world @ posed.head
                got = prop.obj.matrix_world.translation
                print(
                    f"  anchor {prop.obj.name}: wrist {tuple(round(v, 3) for v in head)}"
                    f" -> prop {tuple(round(v, 3) for v in got)}"
                    f"  gap {(got - head).length * 100:.1f} cm"
                )
            for note in build.notes:
                print(f"  note: {note}")
            if build.static:
                # A set has to be in shot, and framing so far covers only the
                # character: measured on the badminton clip the net sits 2.2 m away
                # while the character's own hull is 1.43 x 1.12 m, so the net fell
                # entirely outside the frame and the sliver that survived read as a
                # wall. Expanding to the whole set is the other extreme — a 13.4 m
                # court would leave the figure a few dozen pixels — so the set is
                # clipped to a box around the player first.
                reach = args.scene_reach
                keep_min = np.array([stand_origin[0] - reach, stand_origin[1] - reach, floor_z])
                keep_max = np.array([stand_origin[0] + reach, stand_origin[1] + reach, floor_z + reach])
                for obj in build.static:
                    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                    box_min = np.min([[c.x, c.y, c.z] for c in corners], axis=0)
                    box_max = np.max([[c.x, c.y, c.z] for c in corners], axis=0)
                    box_min = np.maximum(box_min, keep_min)
                    box_max = np.minimum(box_max, keep_max)
                    if np.all(box_max > box_min):
                        mins = np.minimum(mins, box_min)
                        maxs = np.maximum(maxs, box_max)
                print(f"  framing widened for the set: {np.round(mins, 2)} .. {np.round(maxs, 2)}")
        if args.stage:
            from motionviewer.blender.mmd_props import load_stage

            stage_path = Path(args.stage)
            if not stage_path.is_file():
                raise SystemExit(f"--stage: {stage_path} not found")
            stage_meshes = load_stage(bpy, stage_path, scale=args.stage_scale)
            print(f"stage: {stage_path.name}, {len(stage_meshes)} mesh(es)")
            # The stage brings its own MMD materials, already authored for a toon look,
            # so it is left unshaded rather than pushed through the character's shader.
        if args.weapon:
            from motionviewer.blender.mmd_props import (
                WEAPON_BY_CHARACTER,
                close_hand,
                fit_weapon_to_hand,
                load_weapon,
            )

            directory = args.asset.parent
            if args.weapon == "auto":
                entry = WEAPON_BY_CHARACTER.get(directory.name)
                if entry is None:
                    # Walk up: some packs nest the model one level under the pack name.
                    entry = WEAPON_BY_CHARACTER.get(directory.parent.name)
                if entry is None:
                    raise SystemExit(
                        f"--weapon auto: no weapon known for {directory.name!r}; "
                        f"have {sorted(WEAPON_BY_CHARACTER)}"
                    )
                filename = entry["file"]
                hand = entry["hand"]
                grip = entry["grip"]
                weapon_path = directory / filename
                if not weapon_path.is_file():
                    weapon_path = directory.parent / filename
            else:
                weapon_path = Path(args.weapon)
                hand, grip = "右手首", "low"
            if args.weapon_hand:
                hand = args.weapon_hand
            if args.weapon_grip_section is not None:
                grip = args.weapon_grip_section
            if not weapon_path.is_file():
                raise SystemExit(f"--weapon: {weapon_path} not found")
            weapon = load_weapon(bpy, weapon_path)
            framing_weapon = weapon
            try:
                fit = fit_weapon_to_hand(
                    bpy, weapon, actor.armature, hand, grip=grip, frame=frame_start, mode=args.weapon_fit_mode,
                )
            except KeyError:
                raise SystemExit(f"--weapon: rig has no bone {hand!r}") from None
            if args.weapon_fit_mode == "surface":
                scene_weapon_colliders = (weapon,)
            if not args.no_outline:
                add_outline([weapon], style=ToonStyle(outline_thickness_m=thickness))
            if args.soma is not None and not args.override_source_fingers:
                driven_count = int(actor.armature.get("motionviewer_soma_fingers", 0))
                fingers = {"mode": "source_preserved" if driven_count else "bind_pose_preserved", "solved": {}, "keys": 0,
                           "reason": "Weapon attachment does not replace the existing finger channels",
                           "source_driven_bones": driven_count}
            else:
                fingers = close_hand(
                    bpy,
                    actor.armature,
                    hand,
                    frame_start=frame_start,
                    frames=total,
                    shaft_m=fit["shaft_mm"] / 1000.0,
                    weapon=weapon,
                    grip=grip,
                    palm_normal=Vector(fit["palm_normal"]),
                    mode=args.weapon_fit_mode,
                )
            weapon_info.update(fit=fit, fingers=fingers)
            weapon_info["left_arm_clearance"] = {"applied": False, "requested": args.weapon_arm_clearance}
            if args.weapon_arm_clearance:
                from motionviewer.blender.mmd_weapon_avoidance import apply_weapon_left_avoidance

                weapon_info["left_arm_clearance"] = apply_weapon_left_avoidance(
                    bpy, actor.armature, weapon, frame_start=frame_start, frames=total, fps=source_fps,
                )
            print(
                f"weapon: {weapon_path.name} on {hand}, grip={fit['grip']}, "
                f"{fit['length_m']:.2f} m, shaft {fit['shaft_mm']:.0f} mm, "
                f"{fingers['mode']} {fingers['solved']}, {fingers['keys']} keys"
            )
            print(f"  palm {fit['palm']}  tip {fit['tip']}")
            # A 1.2-2.3 m weapon raised overhead reaches far outside a hull built from
            # the body alone, so framing has to cover it or the blade leaves the frame
            # and the render shows a stub. Sampled over the clip because the weapon
            # follows the hand and its extent is not known from one frame.
            weapon_min = np.full(3, 1e9)
            weapon_max = np.full(3, -1e9)
            weapon_follow_min = np.full(3, 1e9)
            weapon_follow_max = np.full(3, -1e9)
            # A fast swing can reach its widest extent between sparse samples.
            # Follow cameras use a separate root-relative box; expanding only
            # the static world box used to leave the blade outside the image.
            for frame in range(frame_start, frame_start + total):
                bpy.context.scene.frame_set(frame)
                bpy.context.view_layer.update()
                for corner in weapon.bound_box:
                    point = weapon.matrix_world @ Vector(corner)
                    world_point = np.asarray(point, dtype=float)
                    weapon_min = np.minimum(weapon_min, world_point)
                    weapon_max = np.maximum(weapon_max, world_point)
                    relative = world_point.copy()
                    relative[:2] -= root_path[frame - frame_start, :2]
                    weapon_follow_min = np.minimum(weapon_follow_min, relative)
                    weapon_follow_max = np.maximum(weapon_follow_max, relative)
            bpy.context.scene.frame_set(frame_start)
            if np.all(weapon_max > weapon_min):
                mins = np.minimum(mins, weapon_min)
                maxs = np.maximum(maxs, weapon_max)
                follow_mins = np.minimum(follow_mins, weapon_follow_min - args.frame_pad)
                follow_maxs = np.maximum(follow_maxs, weapon_follow_max + args.frame_pad)
                follow_span = weapon_max - weapon_min
                print(f"  framing widened for the weapon: extent {np.round(follow_span, 2)}")
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
    physics_info: dict = {"built": False}
    if args.bullet:
        from motionviewer.blender.mmd_bullet import BulletSettings, build_and_bake

        bullet_overrides = dict(
            substeps=args.bullet_substeps,
            iterations=args.bullet_iterations,
            settle_frames=args.bullet_settle,
            fix_damping=args.bullet_damping == "legacy-inverted",
            cloth_angular_slack_degrees=args.bullet_cloth_slack,
            cloth_linear_slack_m=args.bullet_cloth_linear_slack,
            fit_hand_colliders=args.bullet_hand_colliders,
            despike=args.bullet_despike,
            neutral_settle=args.bullet_neutral_settle,
            smooth=args.bullet_smooth,
        )
        mesh_cloth = None
        if physics_profile is not None:
            bullet_overrides.update(physics_profile.bullet_overrides)
            if physics_profile.mesh_cloth_settings is not None:
                from motionviewer.blender.mmd_mesh_cloth import MeshClothSettings, prepare_mesh_cloth

                mesh_cloth = prepare_mesh_cloth(
                    bpy, actor.armature, MeshClothSettings(**physics_profile.mesh_cloth_settings),
                    frame_start=frame_start,
                )
        bullet_settings = BulletSettings(**bullet_overrides)
        # After the retarget and after any pass that jumps around the timeline: building
        # the rig disconnects the physics bones, and a rigid body cache fed frames out of
        # order bakes garbage.
        physics_info = build_and_bake(
            bpy,
            actor.armature,
            frame_start=frame_start,
            frame_end=frame_end,
            fps=source_fps,
            scene_box_colliders=scene_box_colliders,
            scene_weapon_colliders=scene_weapon_colliders,
            profile_joint_rules=physics_profile.profile.joint_rules if physics_profile is not None else (),
            profile_side_link_rules=physics_profile.profile.side_link_rules if physics_profile is not None else (),
            settings=bullet_settings,
        )
        if mesh_cloth is not None:
            from motionviewer.blender.mmd_mesh_cloth import bake_mesh_cloth

            physics_info["mesh_cloth"] = bake_mesh_cloth(
                bpy, mesh_cloth, frame_start=frame_start, frame_end=frame_end,
                settle_frames=bullet_settings.settle_frames,
            )
        if (bullet_settings.despike or bullet_settings.smooth > 0.0) and physics_info.get("baked"):
            # Last: it reads every finished bake and switches the Bullet world off.
            from motionviewer.blender.mmd_cloth_cleanup import cleanup

            physics_info["cleanup"] = cleanup(
                bpy, actor.armature, frame_start=frame_start, frame_end=frame_end, fps=source_fps,
                despike_enabled=bullet_settings.despike, smooth_sigma=bullet_settings.smooth,
            )
        if physics_profile is not None:
            from dataclasses import asdict

            physics_info["profile"] = physics_profile.report()
            physics_info["resolved_settings"] = asdict(bullet_settings)
        print(f"bullet: {json.dumps(physics_info, ensure_ascii=False)}")
        if not physics_info.get("built"):
            raise RuntimeError(f"--bullet requested but physics was not built: {physics_info.get('reason')}")
    if scene_box_colliders:
        from motionviewer.blender.mmd_box_contact import audit_animated_box_clearance

        box = scene_box_colliders[0]  # Both box modes create one fixed-size box.
        # Use the actual local mesh extent, not rounded fit-report dimensions.
        box_size = tuple(np.ptp(np.asarray([tuple(vertex.co) for vertex in box.data.vertices]), axis=0))
        evaluation_stage = ("post_baked_bullet" if physics_info.get("baked") else
                            "post_spring" if args.spring else "post_pose_no_physics")
        scene_info["post_physics_clearance"] = audit_animated_box_clearance(
            bpy, box, actor.mesh_objects, size=box_size, frame_start=frame_start,
            frames=total, evaluation_stage=evaluation_stage,
        )
        scene_info["post_physics_clearance"]["box_local_size_m"] = list(box_size)
        print(f"final box clearance: {json.dumps(scene_info['post_physics_clearance'], ensure_ascii=False)}")
    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = width
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filter_size = args.filter_size
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    args.output.mkdir(parents=True, exist_ok=True)

    camera_tracks: dict = {}
    camera_geometry_fit: dict = {}
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
        if view in matched_tracks:
            camera = apply_camera_track(bpy, matched_tracks[view], name=f"ReviewCamera_{view}")
        elif follow:
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
            camera = add_camera_for_bounds(
                mins.tolist(),
                maxs.tolist(),
                preset=preset,
                margin=1.1,
                resolution=(width, args.resolution),
                elevation=args.camera_elevation if view_direction is None else None,
                direction=view_direction,
            )
        framing_props = [framing_weapon] if framing_weapon is not None else list(scene_box_colliders)
        if framing_props and view not in matched_tracks and not args.frame_motion:
            from motionviewer.blender.review_camera import fit_orthographic_geometry

            camera_geometry_fit[view] = fit_orthographic_geometry(
                bpy, camera, [*actor.mesh_objects, *framing_props],
                frame_start=frame_start, frames=total,
            )
        camera_tracks[view] = capture_camera_track(bpy, camera, frame_start=frame_start, frames=total)
        for panel in panels:
            # The outline shell is a separate object, so hiding only the
            # character meshes leaves a black silhouette in the skeleton panel.
            for obj in (*actor.mesh_objects, *outline_shells):
                obj.hide_render = panel != "character"
            for obj in skeleton_objects:
                obj.hide_render = panel != "skeleton"
            for obj in body_objects:
                obj.hide_render = panel != "body"
            directory = args.output / panel / view
            directory.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(directory / "frame_")
            bpy.ops.render.render(animation=True)
            print(f"rendered {panel}/{view}: {total} frames")

    body_edits = []
    for enabled, name in (
        (not args.faithful, "pose_polish"),
        (args.upper_smooth, "upper_body_smoothing"),
        (args.foot_contact == "on", "foot_contact_ik"),
        (args.arm_clearance_degrees, "arm_clearance"),
        (args.scene_contact, "box_contact_ik"),
        (args.weapon_arm_clearance, "weapon_arm_clearance"),
        (args.root_motion != "source", "legacy_scaled_root"),
    ):
        if enabled:
            body_edits.append(name)
    native_finger_count = int(actor.armature.get("motionviewer_soma_fingers", 0))
    native_fingers = args.soma is not None and native_finger_count > 0
    finger_edits = []
    if weapon_info.get("fingers", {}).get("keys", 0):
        finger_edits.append("weapon_grip_override" if native_fingers else "synthetic_weapon_grip")
    if args.scene_contact:
        finger_edits.append("box_contact_grip")
    finger_override = native_fingers and bool(finger_edits)
    fidelity_info = {
        "source_representation": "native_soma77" if args.soma is not None else "smplx_body22",
        "pose_polish_enabled": not args.faithful,
        "body_edits_requested": body_edits,
        "root_motion_mode": args.root_motion,
        "source_root_deltas_in_metres": args.root_motion == "source",
        "source_fingers_driven": native_fingers,
        "source_finger_bones_driven": native_finger_count,
        "source_hand_axis_landmarks": json.loads(actor.armature.get("motionviewer_soma_hand_aim", "{}")),
        "source_fingers_overridden": finger_override,
        "finger_edits": finger_edits,
        "mapping_only": not body_edits and not finger_edits and (args.soma is None or native_fingers),
        "limits": (
            "Retargeted onto different body proportions; absolute joint positions and contacts "
            "need not match the source skeleton. Clothing and props are added visualization. "
            "The faithful flag disables pose polish only; all explicit adaptations are listed here."
        ),
    }
    (args.output / "info.json").write_text(
        json.dumps(
            {
                "motion": str(original_source_path),
                "motion_sha256": smoothing_info["source_sha256"],
                "render_motion": str(source_path),
                "upper_smoothing": smoothing_info,
                "foot_contact": foot_contact_info,
                "arm_clearance": arm_clearance_info,
                "scene": scene_info,
                "weapon": weapon_info,
                "source_format": "soma77" if args.soma is not None else "smplx22",
                "asset": str(args.asset),
                "caption": args.caption,
                "frames": total,
                "views": views,
                "panels": panels,
                "faithful": bool(args.faithful),
                "source_body": body_info,
                "motion_fidelity": fidelity_info,
                "toon": bool(args.toon),
                "spring": spring_info,
                "physics_requested": "bullet" if args.bullet else "spring" if args.spring else "none",
                "physics": physics_info,
                "source_fps": source_fps,
                "scene_fps": scene.render.fps / scene.render.fps_base,
                "expression": expression_info,
                "transfer": transfer.get("polish", {}),
                "root_translation_scale": scale,
                "bounds": [mins.tolist(), maxs.tolist()],
                "shared_framing": [str(path) for path in (args.frame_motion or [])],
                "shared_framing_sources": [str(path) for path in original_framing_paths],
                "camera": "follow" if follow else "static",
                "camera_match": str(args.match_camera) if args.match_camera else None,
                "camera_tracks": camera_tracks,
                "camera_geometry_fit": camera_geometry_fit,
                "floor_z": floor_z,
                "mesh_bounds_heights_sampled": [round(v, 4) for v in bounds_heights],
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
