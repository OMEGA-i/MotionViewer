"""Matched Bullet audit with torso-relative garment motion and skin intersection proxies.

Arguments not consumed here pass to render_mmd_compare. --audit-only skips images.
This never modifies source PMX/NPZ files. Metrics are diagnostics, not visual acceptance.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def audit(bpy, armature, model, frame_start, frame_end, output, report):
    import numpy as np
    from mathutils.bvhtree import BVHTree

    dynamic = [o for o in model.rigidBodies() if o.rigid_body and not o.rigid_body.kinematic]
    skirt = {
        o.mmd_rigid.bone
        for o in dynamic
        if any(t in o.mmd_rigid.bone for t in ("裙", "裾", "スカート", "skirt"))
    }
    names = [o.mmd_rigid.bone for o in dynamic if o.mmd_rigid.bone in armature.pose.bones]
    selection = []
    inventory = []
    for obj in model.rigidBodies():
        rb = obj.rigid_body
        inventory.append(
            dict(
                name=obj.name,
                bone=obj.mmd_rigid.bone,
                mode=obj.mmd_rigid.type,
                group=obj.mmd_rigid.collision_group_number,
                ignore=list(obj.mmd_rigid.collision_group_mask),
                damping=[rb.linear_damping, rb.angular_damping],
                mass=rb.mass,
                shape=rb.collision_shape,
                size=list(obj.dimensions),
            )
        )
    for mesh in model.meshes():
        if any(m.type == "SOLIDIFY" and m.name == "Outline" for m in mesh.modifiers):
            continue
        group_names = {g.index: g.name for g in mesh.vertex_groups}
        groups = {i for i, n in group_names.items() if n in skirt}
        ids = {
            v.index for v in mesh.data.vertices if sum(g.weight for g in v.groups if g.group in groups) > 0.5
        }
        if not ids:
            continue
        skin_faces = [
            tuple(p.vertices)
            for p in mesh.data.polygons
            if any(t in mesh.data.materials[p.material_index].name.casefold() for t in ("肌", "skin"))
        ]
        skin_face_set = set(skin_faces)
        cloth_faces = [
            tuple(p.vertices)
            for p in mesh.data.polygons
            if all(v in ids for v in p.vertices) and tuple(p.vertices) not in skin_face_set
        ]
        selection.append((mesh, sorted(ids), cloth_faces, skin_faces))
    tracks, bones, crossings, depths, settle = [], [], [], [], []
    penetration_details = []
    crossing_details = []
    shared_touch_counts = []
    scene = bpy.context.scene
    # Negative cache-frame playback is not reliable in the current Blender build;
    # do not use those reads as evidence that pre-roll has settled.
    for frame in range(max(0, frame_start - 10), frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        pelvis = armature.matrix_world @ armature.pose.bones["下半身"].matrix
        inverse = np.asarray(pelvis.inverted(), dtype=float)
        points = []
        intersections, shared_touches, negative = 0, 0, []
        for mesh, ids, cloth_faces, skin_faces in selection:
            evaluated = mesh.evaluated_get(depsgraph)
            data = evaluated.to_mesh()
            try:
                coords = np.array([tuple(evaluated.matrix_world @ v.co) for v in data.vertices])
                points.append(coords[ids] @ inverse[:3, :3].T + inverse[:3, 3])
                if frame >= frame_start and skin_faces and cloth_faces:
                    skin = BVHTree.FromPolygons(coords.tolist(), skin_faces)
                    cloth = BVHTree.FromPolygons(coords.tolist(), cloth_faces)
                    for cloth_index, skin_index in cloth.overlap(skin):
                        a, b = cloth_faces[cloth_index], skin_faces[skin_index]
                        # Different BVH trees do not omit shared mesh seams. A
                        # shared vertex/edge alone is not evidence of penetration.
                        if set(a).intersection(b):
                            shared_touches += 1
                            continue
                        intersections += 1
                        crossing_details.append(
                            dict(
                                frame=frame,
                                mesh=mesh.name,
                                cloth_vertices=a,
                                skin_vertices=b,
                                cloth_coordinates=coords[list(a)].tolist(),
                                skin_coordinates=coords[list(b)].tolist(),
                                skin_bones=sorted(
                                    {
                                        mesh.vertex_groups[g.group].name
                                        for vi in b
                                        for g in mesh.data.vertices[vi].groups
                                        if g.weight > 0.5
                                        and mesh.vertex_groups[g.group].name in armature.data.bones
                                    }
                                ),
                            )
                        )
                    for idx in ids:
                        loc, normal, face, distance = skin.find_nearest(coords[idx])
                        if loc is not None and distance < 0.03:
                            signed = float((coords[idx] - np.asarray(loc)) @ np.asarray(normal))
                            negative.append(max(0.0, -signed))
                            if signed < -0.001:
                                collider_distance = []
                                for obj in bpy.data.objects:
                                    rb = obj.rigid_body
                                    if not rb or not rb.kinematic:
                                        continue
                                    mat = obj.evaluated_get(depsgraph).matrix_world
                                    rot = np.asarray(mat.to_quaternion().to_matrix())
                                    q = (np.asarray(loc) - np.asarray(mat.translation)) @ rot
                                    size = np.asarray(obj.dimensions)
                                    if rb.collision_shape == "SPHERE":
                                        d = np.linalg.norm(q) - size[0] / 2
                                    elif rb.collision_shape == "CAPSULE":
                                        q[2] -= np.clip(
                                            q[2],
                                            -max(0.0, size[2] / 2 - size[0] / 2),
                                            max(0.0, size[2] / 2 - size[0] / 2),
                                        )
                                        d = np.linalg.norm(q) - size[0] / 2
                                    elif rb.collision_shape == "BOX":
                                        v = np.abs(q) - size / 2
                                        d = np.linalg.norm(np.maximum(v, 0)) + min(float(v.max()), 0.0)
                                    else:
                                        continue
                                    collider_distance.append((float(d), obj.name))
                                closest = min(collider_distance, default=(float("nan"), ""))
                                penetration_details.append(
                                    dict(
                                        frame=frame,
                                        vertex=idx,
                                        depth_mm=-signed * 1000,
                                        skin_point=list(loc),
                                        skin_bones=list(
                                            {
                                                mesh.vertex_groups[g.group].name
                                                for vi in skin_faces[face]
                                                for g in mesh.data.vertices[vi].groups
                                                if g.weight > 0.5
                                                and mesh.vertex_groups[g.group].name in armature.data.bones
                                            }
                                        ),
                                        collider_signed_mm=closest[0] * 1000,
                                        collider=closest[1],
                                    )
                                )
            finally:
                evaluated.to_mesh_clear()
        points = np.concatenate(points) if points else np.empty((0, 3))
        if frame < frame_start:
            settle.append(points)
            continue
        tracks.append(points)
        b = np.array([tuple(armature.matrix_world @ armature.pose.bones[n].tail) for n in names])
        bones.append(b @ inverse[:3, :3].T + inverse[:3, 3])
        crossings.append(intersections)
        shared_touch_counts.append(shared_touches)
        depths.append(max(negative, default=0.0))
    points = np.asarray(tracks)
    bone_points = np.asarray(bones)
    fps = scene.render.fps / scene.render.fps_base

    def movement(x):
        speed = np.linalg.norm(np.diff(x, axis=0), axis=-1) * fps
        acc = np.linalg.norm(np.diff(x, n=2, axis=0), axis=-1) * fps**2
        # One-frame deviation from adjacent-frame midpoint; reported in millimetres.
        jitter = 0.5 * np.linalg.norm(np.diff(x, n=2, axis=0), axis=-1) * 1000
        return dict(
            speed_m_s_p95=float(np.percentile(speed, 95)),
            acceleration_m_s2_p95=float(np.percentile(acc, 95)),
            midpoint_jitter_mm_p95=float(np.percentile(jitter, 95)),
            midpoint_jitter_mm_max=float(jitter.max()),
            worst_jitter_frame_1based=int(
                np.unravel_index(jitter.argmax(), jitter.shape)[0] + frame_start + 1
            ),
            first_10_jitter_mm_p95=float(np.percentile(jitter[:10], 95)),
        )

    metrics = dict(
        physics=report,
        frame_start=frame_start,
        frame_end=frame_end,
        fps=fps,
        skirt_bones=sorted(skirt),
        skirt_vertices=int(points.shape[1]),
        selected_meshes=[dict(name=m.name, vertices=len(ids)) for m, ids, _, _ in selection],
        skirt=movement(points),
        all_dynamic_bones=movement(bone_points),
        skin_cloth_triangle_intersection_pairs_mean=float(np.mean(crossings)),
        skin_cloth_triangle_intersection_pairs_max=int(max(crossings)),
        skin_cloth_triangle_intersection_frames=int(np.count_nonzero(crossings)),
        shared_vertex_touch_pairs_excluded=int(sum(shared_touch_counts)),
        intersection_method="Base-mesh skin vs selected garment triangle intersections; shared-vertex pairs excluded.",
        nearest_skin_inward_distance_max_mm=float(max(depths) * 1000),
        proxy_caveat="Skin is an open surface; triangle crossings and signed nearest-surface distances require visual review.",
        rigid_bodies=inventory,
        joints=[
            dict(
                name=j.name,
                type=j.rigid_body_constraint.spring_type,
                disable_collisions=j.rigid_body_constraint.disable_collisions,
                stiffness=[
                    getattr(j.rigid_body_constraint, "spring_stiffness" + axis)
                    for axis in ("_x", "_y", "_z", "_ang_x", "_ang_y", "_ang_z")
                ],
                damping=[
                    getattr(j.rigid_body_constraint, "spring_damping" + axis)
                    for axis in ("_x", "_y", "_z", "_ang_x", "_ang_y", "_ang_z")
                ],
                linear_limits=[
                    [
                        getattr(j.rigid_body_constraint, "limit_lin_" + axis + "_" + bound)
                        for bound in ("lower", "upper")
                    ]
                    for axis in ("x", "y", "z")
                ],
                angular_limits=[
                    [
                        getattr(j.rigid_body_constraint, "limit_ang_" + axis + "_" + bound)
                        for bound in ("lower", "upper")
                    ]
                    for axis in ("x", "y", "z")
                ],
            )
            for j in model.joints()
        ],
        non_collision_constraints=sum(o.mmd_type == "NON_COLLISION_CONSTRAINT" for o in bpy.data.objects),
    )
    np.savez_compressed(
        output / "cloth.npz",
        skirt_torso=points,
        bones_torso=bone_points,
        bone_names=np.asarray(names),
        skin_crossing_pairs=crossings,
        shared_vertex_touch_pairs=shared_touch_counts,
        skin_inward_max_m=depths,
        settle_skirt_torso=np.asarray(settle),
        fps=fps,
    )
    (output / "penetrations.json").write_text(
        json.dumps(sorted(penetration_details, key=lambda v: -v["depth_mm"]), ensure_ascii=False, indent=2)
    )
    (output / "crossings.json").write_text(json.dumps(crossing_details, ensure_ascii=False))
    (output / "cloth.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(
        "CLOTH "
        + json.dumps(
            {k: v for k, v in metrics.items() if k not in {"rigid_bodies", "skirt_bones", "joints"}},
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--linear-floor", type=float)
    parser.add_argument("--angular-floor", type=float)
    parser.add_argument("--all-mask-exclusions", action="store_true")
    parser.add_argument("--collision-margin", type=float)
    parser.add_argument("--skirt-angle-slack", type=float)
    parser.add_argument("--skirt-linear-slack", type=float)
    parser.add_argument("--render-step", type=int, default=1)
    parser.add_argument("--fit-hand-colliders", action="store_true")
    args, remaining = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
    import render_mmd_compare

    from motionviewer.blender import mmd_bullet

    original = mmd_bullet.build_and_bake

    def wrapped(bpy, armature, **kwargs):
        from mmd_tools.core.model import FnModel, Model

        settings = kwargs["settings"]
        if args.fit_hand_colliders:
            settings = replace(settings, fit_hand_colliders=True)
        if args.all_mask_exclusions:
            settings = replace(settings, non_collision_distance_scale=1e9)
        if args.collision_margin is not None:
            settings = replace(settings, collision_margin=args.collision_margin)
        if args.skirt_angle_slack is not None:
            settings = replace(settings, cloth_angular_slack_degrees=args.skirt_angle_slack)
        if args.skirt_linear_slack is not None:
            settings = replace(settings, cloth_linear_slack_m=args.skirt_linear_slack)
        kwargs["settings"] = settings
        old_build = Model.build

        def custom_build(model, *pos, **kw):
            result = old_build(model, *pos, **kw)
            for obj in model.rigidBodies():
                rb = obj.rigid_body
                if rb is None or rb.kinematic:
                    continue
                # Legacy inversion runs after build; pre-invert to achieve the requested minimum.
                for attr, floor in [
                    ("linear_damping", args.linear_floor),
                    ("angular_damping", args.angular_floor),
                ]:
                    if floor is not None:
                        value = 1 - getattr(rb, attr) if settings.fix_damping else getattr(rb, attr)
                        value = max(value, floor)
                        setattr(rb, attr, 1 - value if settings.fix_damping else value)
            return result

        Model.build = custom_build
        try:
            result = original(bpy, armature, **kwargs)
        finally:
            Model.build = old_build
        out = Path(remaining[remaining.index("--output") + 1])
        out.mkdir(parents=True, exist_ok=True)
        result["audit_overrides"] = vars(args)
        result["actual_cache_start"] = bpy.context.scene.rigidbody_world.point_cache.frame_start
        result["actual_cache_end"] = bpy.context.scene.rigidbody_world.point_cache.frame_end
        audit(
            bpy,
            armature,
            Model(FnModel.find_root_object(armature)),
            kwargs["frame_start"],
            kwargs["frame_end"],
            out,
            result,
        )
        bpy.context.scene.frame_step = args.render_step
        return result

    mmd_bullet.build_and_bake = wrapped
    if args.audit_only:
        remaining.extend(["--panels", ""])
    sys.argv = [sys.argv[0], "--", *remaining]
    render_mmd_compare.main()


if __name__ == "__main__":
    main()
