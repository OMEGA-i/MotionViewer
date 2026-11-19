"""Render natural Genshin-Impact in-scene character animations AND stroboscopic teaser figures.

Supports:
1. Video mode (`--mode video`): Renders an animated Genshin PMX character inside a 3D Genshin stage
   (`.blend`, `.fbx`, or `.pmx`) with perspective cinema camera (static or smooth tracking),
   authentic cel shading + stage lighting harmonization, shoe-to-floor raycast grounding,
   contact shadow catcher, and full cloth physics (`--physics-profile`, `--bullet-neutral-settle`,
   `--arm-clearance-degrees`).
2. Stroboscopic Teaser mode (`--mode strobe`): Places K time-frozen poses along the motion trajectory
   inside the same 3D Genshin stage in a SINGLE render pass (true 3D occlusion, perspective, and
   floor shadows!). Supports two visual styles:
   - `hybrid` (matches Paper Image 1): K-1 alabaster/marble source-human SMPL-X meshes along the
     motion trajectory leading up to the full-color cel-shaded Genshin hero character at the climax/final pose!
   - `clones` (matches Paper Image 2): K time-frozen full-color Genshin character poses across the
     3D stage (with optional subtle atmospheric fade on earlier poses via compositor or full crisp 3D).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("video", "strobe"), default="video")
    parser.add_argument("--strobe-style", choices=("hybrid", "clones"), default="hybrid")
    parser.add_argument("--strobe-count", type=int, default=4, help="Total poses shown in strobe mode")
    parser.add_argument(
        "--strobe-frames",
        default="",
        help="Optional explicit comma-separated 1-based frame numbers; if empty, auto-spaced by motion/travel",
    )
    parser.add_argument("--asset", type=Path, required=True, help="Genshin character PMX")
    parser.add_argument("--soma", type=Path, required=True, help="Native SOMA-77 motion .npz")
    parser.add_argument("--smplx-model", type=Path, default=Path.home() / "Downloads/SMPLX_NEUTRAL_2020.npz")
    parser.add_argument("--stage", type=Path, required=True, help="3D Genshin stage (.blend, .fbx, or .pmx)")
    parser.add_argument("--stage-scale", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mp4", type=Path, default=None, help="Optional direct MP4 path for video mode")
    parser.add_argument("--frames", type=int, default=0, help="0 means full clip")
    parser.add_argument("--frame-start-clip", type=int, default=0, help="0-based start offset in source clip")
    parser.add_argument("--stand-xy", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--anchor-mode", choices=("mean", "start", "end"), default="mean")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Character world Z rotation in degrees")
    parser.add_argument("--auto-face-camera", action="store_true", help="Rotate motion heading toward/across camera")
    parser.add_argument(
        "--face-angle-offset",
        type=float,
        default=0.0,
        help="Degrees offset from facing straight at the camera (e.g. 28 for 3/4 diagonal walk, 70 for side-diagonal)",
    )
    parser.add_argument("--cam-loc", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--cam-look", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--lens", type=float, default=35.0, help="Perspective focal length in mm")
    parser.add_argument(
        "--cam-follow",
        type=float,
        default=0.0,
        help="0.0 = locked static camera; 0.5..1.0 = smooth horizontal cinema dolly tracking character XY",
    )
    parser.add_argument("--dof-fstop", type=float, default=0.0, help="0 disables DoF; e.g. 2.8 gives subtle cinema bokeh")
    parser.add_argument("--resolution", type=int, default=1080, help="Frame height in pixels")
    parser.add_argument("--aspect", type=float, default=16.0 / 9.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--arm-clearance-degrees", type=float, default=14.0)
    parser.add_argument("--expression", default="smile")
    parser.add_argument("--physics-profile", type=Path, default=None)
    parser.add_argument("--bullet-neutral-settle", action="store_true")
    parser.add_argument("--bullet-despike", action="store_true")
    parser.add_argument("--bullet-smooth", type=float, default=0.0)
    parser.add_argument("--mannequin-offset-xy", type=float, nargs=2, default=(0.0, 0.0),
                        help="Optional subtle XY offset (m) for source mannequins in hybrid strobe so trails stay clear")
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def _pick_strobe_frames(joints77, count: int, start_idx: int, total: int) -> list[int]:
    """Pick K 1-based Blender frames spaced cleanly along spatial travel + pose arc."""
    import numpy as np

    pts = np.asarray(joints77[start_idx : start_idx + total], dtype=np.float64)
    # Weight pelvis horizontal travel heavily so bodies separate cleanly in 3D space without piling up
    pelvis_step = np.linalg.norm(np.diff(pts[:, 0, [0, 2]], axis=0), axis=-1) * 3.5
    body_step = np.linalg.norm(np.diff(pts[:, :22, :], axis=0), axis=-1).mean(axis=1)
    step = pelvis_step + body_step * 0.35
    cum = np.concatenate([[0.0], np.cumsum(step)])
    if cum[-1] <= 1e-8:
        indices = np.linspace(0, total - 1, count, dtype=int)
    else:
        # Leave 4% trim at start/end if clip starts/ends static, unless very short
        lo = cum[-1] * 0.04 if total > 40 else 0.0
        hi = cum[-1] * 0.96 if total > 40 else cum[-1]
        targets = np.linspace(lo, hi, count)
        indices = [int(np.searchsorted(cum, t)) for t in targets]
    # Deduplicate strictly increasing
    out: list[int] = []
    for idx in indices:
        frame = 1 + int(min(max(idx, 0), total - 1))
        if out and frame <= out[-1]:
            frame = min(out[-1] + 1, total)
        out.append(frame)
    return out


def _compute_auto_yaw(soma_path: Path, start_idx: int, total: int, cam_loc, stand_xy, angle_offset_deg: float) -> float:
    """Orient the character's travel/facing direction relative to the camera."""
    import numpy as np
    from motionviewer.core.smplx_fk import source_to_blender

    with np.load(soma_path, allow_pickle=False) as d:
        j = source_to_blender(np.asarray(d["joints77"][start_idx : start_idx + total], dtype=np.float64))
    pelvis_xy = j[:, 0, :2]
    net_vec = pelvis_xy[-1] - pelvis_xy[0]
    if float(np.linalg.norm(net_vec)) > 0.45:
        src_dir = net_vec / float(np.linalg.norm(net_vec))
    else:
        # Use average body forward cross(up, left_hip -> right_hip); SOMA hips 67 (L), 72 (R)
        lat = j[:, 72, :2] - j[:, 67, :2]
        fwd = np.stack([-lat[:, 1], lat[:, 0]], axis=1).mean(axis=0)
        nrm = float(np.linalg.norm(fwd))
        src_dir = fwd / nrm if nrm > 1e-6 else np.array([0.0, -1.0])

    to_cam = np.array([cam_loc[0] - stand_xy[0], cam_loc[1] - stand_xy[1]], dtype=np.float64)
    to_cam /= max(float(np.linalg.norm(to_cam)), 1e-6)
    desired_angle = math.atan2(to_cam[1], to_cam[0]) + math.radians(angle_offset_deg)
    src_angle = math.atan2(float(src_dir[1]), float(src_dir[0]))
    return math.degrees(desired_angle - src_angle)


def _snapshot_evaluated_character(bpy, actor, outline_shells: list, tag: str) -> list:
    """Freeze the character's exact posed & cloth-simulated meshes at the current frame into static 3D objects."""
    import bmesh  # type: ignore

    deps = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    extracted = []
    for src_obj in (*actor.mesh_objects, *outline_shells):
        eval_obj = src_obj.evaluated_get(deps)
        matrix_world = eval_obj.matrix_world.copy()
        # Persist original material IDs, never temporary evaluated copies.
        materials = list(src_obj.data.materials)
        bm = bmesh.new()
        bm.from_object(src_obj, deps)
        new_mesh = bpy.data.meshes.new(f"{src_obj.name}_{tag}_mesh")
        bm.to_mesh(new_mesh)
        bm.free()
        for mat in materials:
            new_mesh.materials.append(mat)
        new_mesh.update()
        extracted.append((f"{src_obj.name}_{tag}", new_mesh, matrix_world))
    # Linking a clone invalidates the depsgraph; finish all reads first.
    frozen = []
    for name, new_mesh, matrix_world in extracted:
        clone = bpy.data.objects.new(name, new_mesh)
        clone.matrix_world = matrix_world
        scene.collection.objects.link(clone)
        frozen.append(clone)
    return frozen


def _build_alabaster_mannequins(
    bpy,
    *,
    soma_path: Path,
    smplx_model_path: Path,
    start_idx: int,
    total: int,
    pick_frames_1based: list[int],
    actor_root_frame1,
    placement_report: dict,
    offset_xy: tuple[float, float],
) -> list:
    """Build smooth matte-sculpted alabaster SMPL-X source humans at each earlier strobe timestamp
    (matching the white sculptural source mannequins in Paper Reference Image 1)."""
    import numpy as np
    from motionviewer.core.smplx_body import load_smplx_model, pose_frames
    from motionviewer.core.smplx_fk import source_to_blender

    smplx_clip = soma_path.parents[2] / "smplx" / soma_path.parent.name / soma_path.name.replace(".soma.npz", ".smplx.npz")
    if not smplx_clip.is_file():
        raise FileNotFoundError(f"Sibling SMPL-X clip not found for hybrid strobe: {smplx_clip}")

    model = load_smplx_model(smplx_model_path)
    with np.load(smplx_clip, allow_pickle=False) as clip:
        betas = np.asarray(clip["betas"]) if "betas" in clip.files else np.zeros(10)
        verts, joints = pose_frames(
            model,
            betas,
            clip["global_orient"][start_idx : start_idx + total],
            clip["body_pose"][start_idx : start_idx + total],
            clip["transl"][start_idx : start_idx + total],
        )
    verts = source_to_blender(verts)
    joints = source_to_blender(joints[:, :22])

    # First place in pre-stage character frame (same as render_mmd_compare.py), then apply exact stage yaw & XY & floor Z!
    # Reconstruct pre-stage root0 from placement_report's transform_matrix
    yaw_rad = math.radians(float(placement_report["yaw_degrees"]))
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    rot_xy = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)

    # Centre relative to frame 0 pelvis XY, rotate by yaw, and translate to path_start_xy
    p0_xy = joints[0, 0, :2].copy()
    floor_z = float(placement_report["stage_floor_z"])
    min_vert_z = float(np.percentile(verts[:, :, 2].min(axis=1), 10))
    start_xy = np.asarray(placement_report["path_start_xy"], dtype=np.float64)

    mat = bpy.data.materials.new("AlabasterSourceHuman")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.86, 0.85, 0.83, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.52
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.08

    scene = bpy.context.scene
    created = []
    for k_idx, f_1based in enumerate(pick_frames_1based):
        v = verts[f_1based - 1].copy()
        rel_xy = v[:, :2] - p0_xy
        v[:, :2] = rel_xy @ rot_xy.T + start_xy + np.asarray(offset_xy, dtype=np.float64)
        v[:, 2] = v[:, 2] - min_vert_z + floor_z

        mesh = bpy.data.meshes.new(f"SourceHuman_f{f_1based:04d}_mesh")
        mesh.from_pydata(v.tolist(), [], model.faces.tolist())
        mesh.update()
        for poly in mesh.polygons:
            poly.use_smooth = True
        mesh.materials.append(mat)
        obj = bpy.data.objects.new(f"SourceHuman_{k_idx:02d}_f{f_1based:04d}", mesh)
        scene.collection.objects.link(obj)
        created.append(obj)
    return created


def _AuditStageClearance(bpy, stage_meshes: list, char_meshes: list, sample_frames: list[int], floor_z: float) -> dict:
    """Verify the character does not clip into walls/pillars/furniture above ankle height (`floor_z + 0.25m`) in world space."""
    import numpy as np

    # Precollect world-space obstacle vertices from stage meshes in height band [floor_z + 0.30, floor_z + 2.00]
    obs_chunks = []
    for sm in stage_meshes:
        if sm.hide_render or len(sm.data.vertices) == 0:
            continue
        mw = np.asarray(sm.matrix_world, dtype=np.float64)
        co = np.empty((len(sm.data.vertices), 3), dtype=np.float64)
        sm.data.vertices.foreach_get("co", co.ravel())
        wco = co @ mw[:3, :3].T + mw[:3, 3]
        band = wco[(wco[:, 2] > floor_z + 0.32) & (wco[:, 2] < floor_z + 2.00)]
        if len(band):
            obs_chunks.append(band)
    obs_xyz = np.concatenate(obs_chunks, axis=0) if obs_chunks else np.zeros((0, 3))

    min_wall_dist = 1e9
    worst_frame = -1
    sunk_soles_cm = 0.0

    for frame in sample_frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        pts = []
        for cm in char_meshes:
            ev = cm.evaluated_get(deps)
            mw = np.asarray(ev.matrix_world, dtype=np.float64)
            tmp = ev.to_mesh()
            try:
                co = np.empty((len(tmp.vertices), 3), dtype=np.float64)
                tmp.vertices.foreach_get("co", co.ravel())
                wco = co @ mw[:3, :3].T + mw[:3, 3]
                pts.append(wco)
            finally:
                ev.to_mesh_clear()
        if not pts:
            continue
        all_pts = np.concatenate(pts, axis=0)
        sunk_soles_cm = max(sunk_soles_cm, float(max(0.0, (floor_z - all_pts[:, 2].min()) * 100.0)))
        upper = all_pts[all_pts[:, 2] > floor_z + 0.32][::20]
        if len(obs_xyz) and len(upper):
            # 2D XY distance to nearest architectural obstacle vertex in body height band
            diffs = upper[:, None, :2] - obs_xyz[None, :, :2]
            d = float(np.sqrt((diffs * diffs).sum(axis=-1)).min())
            if d < min_wall_dist:
                min_wall_dist = d
                worst_frame = int(frame)

    return {
        "min_obstacle_clearance_m": round(float(min_wall_dist if min_wall_dist < 1e8 else 9.99), 3),
        "worst_clearance_frame": worst_frame,
        "max_below_floor_cm": round(sunk_soles_cm, 2),
        "obstacle_clip_free": bool(min_wall_dist > 0.12),
    }


def main() -> None:
    args = parse_args()

    import bpy  # type: ignore
    import numpy as np
    from mathutils import Vector  # type: ignore

    from motionviewer.assets.physics_profiles import resolve_physics_profile
    from motionviewer.blender.genshin_stage import (
        import_genshin_stage,
        place_and_ground_actor_on_stage,
    )
    from motionviewer.blender.mmd_arm_clearance import apply_arm_clearance
    from motionviewer.blender.mmd_bullet import BulletSettings, build_and_bake
    from motionviewer.blender.mmd_cloth_cleanup import cleanup
    from motionviewer.blender.mmd_expression import apply_expression
    from motionviewer.blender.mmd_toon import ToonStyle, add_outline, apply_toon_shading
    from motionviewer.blender.render import _normalize_engine
    from motionviewer.blender.retarget._animate import _linearize
    from motionviewer.blender.retarget.soma_pipeline import create_soma_actor_from_npz
    from motionviewer.blender.scene import clear_scene, setup_world

    toon_dir = ROOT / ".local/blender_mmd_tools/mmd_tools/externals/MikuMikuDance"
    for name in ("toon01.bmp", "toon05.bmp"):
        source = toon_dir / name
        target = ROOT / name
        if source.is_file() and not target.exists():
            target.symlink_to(source)

    # Optionally slice source clip if --frame-start-clip > 0
    soma_input = args.soma
    with np.load(args.soma, allow_pickle=False) as raw_soma:
        full_j77 = np.asarray(raw_soma["joints77"], dtype=np.float64)
        source_fps = float(raw_soma["fps"]) if "fps" in raw_soma.files else 50.0
        caption = str(raw_soma["caption"]) if "caption" in raw_soma.files else ""
        start_idx = max(0, int(args.frame_start_clip))
        avail = len(full_j77) - start_idx
        total = avail if args.frames <= 0 else min(int(args.frames), avail)
        if start_idx > 0:
            args.output.mkdir(parents=True, exist_ok=True)
            sliced_path = args.output / "_sliced.soma.npz"
            sliced_dict = {}
            for k in raw_soma.files:
                arr = raw_soma[k]
                if arr.ndim >= 1 and len(arr) == len(full_j77):
                    sliced_dict[k] = arr[start_idx : start_idx + total]
                else:
                    sliced_dict[k] = arr
            np.savez_compressed(sliced_path, **sliced_dict)
            soma_input = sliced_path

    physics_profile = None
    if args.physics_profile is not None:
        physics_profile = resolve_physics_profile(args.physics_profile, asset_path=args.asset, import_scale=0.08)

    clear_scene()
    setup_world(transparent=False)

    # 1. Import 3D Genshin Stage (.blend / .fbx / .pmx)
    stage_meshes, stage_info = import_genshin_stage(bpy, args.stage, scale=args.stage_scale)

    # 2. Import & retarget Genshin character on native SOMA-77 (keeps all 30 finger bones)
    actor = create_soma_actor_from_npz(
        soma_input,
        label="hero",
        pmx_path=args.asset,
        pmx_scale=0.08,
        drive_fingers=True,
        polish={"enabled": False, "root_motion": "source"},
        physics=True,
        morphs=args.expression != "none",
    )

    scene = bpy.context.scene
    frame_start = int(scene.frame_start)
    frame_end = frame_start + total - 1
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.render.fps = max(1, int(round(source_fps)))
    scene.render.fps_base = scene.render.fps / source_fps

    # 3. Apply arm clearance (14 deg fixes narrow-hip skirt/thigh hand burial while keeping hand orientation)
    if args.arm_clearance_degrees > 0:
        apply_arm_clearance(
            bpy, actor.armature, frame_start=frame_start, frame_end=frame_end, degrees=args.arm_clearance_degrees
        )
    if args.expression != "none":
        apply_expression(actor.mesh_objects, args.expression)

    # 4. Apply cel shading & crisp constant-pixel outline shell BEFORE stage tint harmonization
    apply_toon_shading(actor.mesh_objects)
    width = max(int(round(args.resolution * args.aspect)), 16)
    cam_dist = float(np.linalg.norm(np.asarray(args.cam_loc) - np.asarray([args.stand_xy[0], args.stand_xy[1], 1.1])))
    fov_h = 2.0 * math.atan(18.0 / max(args.lens, 1.0))
    visible_h_m = max(2.0 * cam_dist * math.tan(fov_h * 0.5) / max(args.aspect, 0.5), 1.4)
    px_per_m = args.resolution / visible_h_m
    thickness = min(max(1.5 / max(px_per_m, 100.0), 0.0012), 0.0038)
    outline_shells = add_outline(actor.mesh_objects, style=ToonStyle(outline_thickness_m=thickness))

    # 5. Build and bake Bullet + Mesh Cloth physics at the origin FIRST (where local XY radial rules
    # and rest-pose alignments are exact!), THEN rigidly place `mmd_root` + baked caches onto the stage!
    bullet_overrides = dict(
        substeps=12,
        iterations=24,
        settle_frames=30,
        fix_damping=False,
        cloth_angular_slack_degrees=0.0,
        cloth_linear_slack_m=0.0,
        fit_hand_colliders=False,
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
                bpy, actor.armature, MeshClothSettings(**physics_profile.mesh_cloth_settings), frame_start=frame_start
            )
    bullet_settings = BulletSettings(**bullet_overrides)
    physics_info = build_and_bake(
        bpy,
        actor.armature,
        frame_start=frame_start,
        frame_end=frame_end,
        fps=source_fps,
        profile_joint_rules=physics_profile.profile.joint_rules if physics_profile is not None else (),
        profile_side_link_rules=physics_profile.profile.side_link_rules if physics_profile is not None else (),
        settings=bullet_settings,
    )
    if mesh_cloth is not None:
        from motionviewer.blender.mmd_mesh_cloth import bake_mesh_cloth

        physics_info["mesh_cloth"] = bake_mesh_cloth(
            bpy, mesh_cloth, frame_start=frame_start, frame_end=frame_end, settle_frames=bullet_settings.settle_frames
        )
    # Always bake rigid-body tracks to keyframes in `mmd_root` parent space & disable live Bullet world
    # so moving `mmd_root` onto the Genshin stage carries every hair/skirt bone deterministically!
    physics_info["cleanup"] = cleanup(
        bpy,
        actor.armature,
        frame_start=frame_start,
        frame_end=frame_end,
        fps=source_fps,
        despike_enabled=bullet_settings.despike,
        smooth_sigma=bullet_settings.smooth,
    )

    # 6. Place & orient character (`mmd_root`) on the exact walkable Genshin floor Z + contact shadow catcher + ambient tint
    yaw_deg = (
        _compute_auto_yaw(
            args.soma, start_idx, total, args.cam_loc, args.stand_xy, args.face_angle_offset
        )
        if args.auto_face_camera
        else float(args.yaw_deg)
    )
    placement = place_and_ground_actor_on_stage(
        bpy,
        actor,
        stage_meshes,
        frame_start=frame_start,
        frames=total,
        target_xy=(float(args.stand_xy[0]), float(args.stand_xy[1])),
        yaw_degrees=yaw_deg,
        anchor_mode=args.anchor_mode,
    )

    # 7. Audit 3D character-to-stage obstacle & floor clearance
    sample_check = sorted({frame_start + int(round(i * (total - 1) / 9)) for i in range(10)})
    clearance = _AuditStageClearance(bpy, stage_meshes, actor.mesh_objects, sample_check, placement["stage_floor_z"])

    # 8. Configure Perspective Cinema Camera & EEVEE render quality
    cam_data = bpy.data.cameras.new("GenshinSceneCam")
    cam_data.type = "PERSP"
    cam_data.lens = float(args.lens)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.12
    cam_data.clip_end = 400.0
    cam = bpy.data.objects.new("GenshinSceneCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    base_cam = np.asarray(args.cam_loc, dtype=np.float64)
    base_look = np.asarray(args.cam_look, dtype=np.float64)

    if args.dof_fstop > 0:
        cam_data.dof.use_dof = True
        cam_data.dof.aperture_fstop = float(args.dof_fstop)
        cam_data.dof.focus_distance = float(np.linalg.norm(base_look - base_cam))

    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = width
    scene.render.resolution_y = int(args.resolution)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.filter_size = 0.9
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(args.samples)
        scene.eevee.use_shadows = True

    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "video":
        if args.cam_follow > 0.0 and total > 1:
            roots_xy = np.zeros((total, 2), dtype=np.float64)
            for idx in range(total):
                scene.frame_set(frame_start + idx)
                roots_xy[idx] = (actor.armature.location.x, actor.armature.location.y)
            delta_xy = (roots_xy - roots_xy.mean(axis=0)) * float(args.cam_follow)
            # Smooth dolly over 15 frames
            win = min(15, total if total % 2 == 1 else max(1, total - 1))
            if win > 1:
                pad = win // 2
                padded = np.pad(delta_xy, ((pad, pad), (0, 0)), mode="edge")
                kern = np.ones(win) / win
                delta_xy = np.stack([np.convolve(padded[:, a], kern, mode="valid") for a in range(2)], axis=1)
            for idx in range(total):
                f = frame_start + idx
                loc_v = Vector((base_cam[0] + delta_xy[idx, 0], base_cam[1] + delta_xy[idx, 1], base_cam[2]))
                look_v = Vector((base_look[0] + delta_xy[idx, 0], base_look[1] + delta_xy[idx, 1], base_look[2]))
                cam.location = loc_v
                cam.rotation_euler = (look_v - loc_v).to_track_quat("-Z", "Y").to_euler()
                cam.keyframe_insert(data_path="location", frame=f)
                cam.keyframe_insert(data_path="rotation_euler", frame=f)
            _linearize(cam)
        else:
            cam.location = Vector(tuple(base_cam))
            cam.rotation_euler = (Vector(tuple(base_look)) - cam.location).to_track_quat("-Z", "Y").to_euler()

        frames_dir = args.output / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(frames_dir / "frame_")
        bpy.ops.render.render(animation=True)

        mp4_path = args.mp4 or (args.output / "video.mp4")
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                f"{source_fps:g}",
                "-start_number",
                str(frame_start),
                "-i",
                str(frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "16",
                str(mp4_path),
            ],
            check=True,
        )
        mode_report = {"mode": "video", "mp4": str(mp4_path), "frames": total}

    else:
        # Stroboscopic Teaser Mode: True 3D multi-timestamp freeze inside the static Genshin stage!
        cam.location = Vector(tuple(base_cam))
        cam.rotation_euler = (Vector(tuple(base_look)) - cam.location).to_track_quat("-Z", "Y").to_euler()

        if args.strobe_frames.strip():
            picks = [int(x.strip()) for x in args.strobe_frames.split(",") if x.strip()]
        else:
            picks = _pick_strobe_frames(full_j77, args.strobe_count, start_idx, total)

        if args.strobe_style == "hybrid":
            # Earlier K-1 timestamps: Alabaster sculpted SMPL-X source humans; Final timestamp: Genshin hero!
            mannequin_frames = picks[:-1]
            hero_frame = picks[-1]
            scene.frame_set(frame_start)
            bpy.context.view_layer.update()
            root1 = tuple(actor.armature.location)
            _build_alabaster_mannequins(
                bpy,
                soma_path=args.soma,
                smplx_model_path=args.smplx_model,
                start_idx=start_idx,
                total=total,
                pick_frames_1based=mannequin_frames,
                actor_root_frame1=root1,
                placement_report=placement,
                offset_xy=(float(args.mannequin_offset_xy[0]), float(args.mannequin_offset_xy[1])),
            )
            scene.frame_set(hero_frame)
            bpy.context.view_layer.update()
        else:
            # All K timestamps: Freeze K full-color 3D Genshin character snapshots along the trajectory!
            for k_idx, f_pick in enumerate(picks[:-1]):
                scene.frame_set(f_pick)
                bpy.context.view_layer.update()
                _snapshot_evaluated_character(bpy, actor, outline_shells, tag=f"strobe_{k_idx:02d}_f{f_pick:04d}")
            hero_frame = picks[-1]
            scene.frame_set(hero_frame)
            bpy.context.view_layer.update()

        still_path = args.output / f"strobe_{args.strobe_style}.png"
        scene.render.filepath = str(still_path)
        bpy.ops.render.render(write_still=True)
        mode_report = {
            "mode": "strobe",
            "style": args.strobe_style,
            "picked_frames_1based": picks,
            "still": str(still_path),
        }

    report = {
        "asset": str(args.asset),
        "soma": str(args.soma),
        "caption": caption,
        "stage": stage_info,
        "placement": placement,
        "clearance_audit": clearance,
        "physics_baked": bool(physics_info.get("baked")),
        "camera": {
            "loc": list(args.cam_loc),
            "look": list(args.cam_look),
            "lens_mm": args.lens,
            "follow": args.cam_follow,
            "resolution": [width, int(args.resolution)],
        },
        **mode_report,
    }
    (args.output / "scene_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SCENE_RENDER_DONE:", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
