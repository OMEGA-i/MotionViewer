"""An upward box placement which never writes character pose channels."""
from __future__ import annotations

from typing import Any

import numpy as np

from motionviewer.core.box_support import fit_supported_box


def build_supported_box_scene(bpy: Any, style: Any, *, armature: Any,
                              mesh_objects: list[Any], frame_start: int = 1, frames: int = 1,
                              side: float = .40, depth: float = .22, height: float = .15,
                              hand_bone: str = "左手首") -> Any:
    """Place a fixed box above existing hand skin, reporting support limitations."""
    from mathutils import Matrix, Vector

    from .mmd_box_contact import audit_animated_box_clearance
    from .mmd_props import Prop, SceneBuild, build_cardboard_box
    from .retarget._animate import _linearize
    from .retarget.mmd_hands import _flexion_axis

    del hand_bone
    if frames < 1:
        raise ValueError("Box support needs a positive frame count")
    scene = bpy.context.scene
    original_frame = scene.frame_current
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    bones = armature.pose.bones
    local_normals = []
    for prefix in ("左", "右"):
        axis = _flexion_axis(armature.matrix_world, armature.data.bones, prefix)
        if axis is None:
            raise ValueError("Cannot identify the model palm-facing direction")
        finger_rest = np.asarray((armature.matrix_world @ armature.data.bones[prefix + "中指１"].matrix_local).to_quaternion().to_matrix())[:, 1]
        normal = np.cross(axis, finger_rest)
        normal /= np.linalg.norm(normal)
        rest = np.asarray((armature.matrix_world @ armature.data.bones[prefix + "手首"].matrix_local).to_quaternion().to_matrix())
        local_normals.append(rest.T @ normal)
    selections = []
    labels = [[], []]
    for mesh in mesh_objects:
        groups = {g.index: g.name for g in mesh.vertex_groups}
        indices = []
        for hand, prefix in enumerate(("左", "右")):
            names = {name for name in groups.values() if name.startswith(prefix)
                     and any(part in name for part in ("手首", "指"))}
            ids = [v.index for v in mesh.data.vertices
                   if sum(g.weight for g in v.groups if groups[g.group] in names) >= .7]
            indices.append(np.asarray(ids, dtype=int))
            for index in ids:
                vertex = mesh.data.vertices[index]
                relevant = [(g.weight, groups[g.group]) for g in vertex.groups if groups[g.group] in names]
                labels[hand].append(max(relevant)[1])
        selections.append((mesh, indices))
    if not all(labels):
        raise ValueError("Cannot identify both hands' evaluated skin")
    palms, normals, torsos, hand_clouds = [], [], [], [[], []]
    source_basis, source_root = [], []
    try:
        for frame in range(frame_start, frame_start + frames):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            world = armature.matrix_world
            frame_palms, frame_normals = [], []
            for hand, prefix in enumerate(("左", "右")):
                wrist = bones[prefix + "手首"]
                frame_palms.append((np.asarray(world @ wrist.head) + np.asarray(world @ bones[prefix + "中指１"].head)) / 2)
                frame_normals.append(np.asarray((world @ wrist.matrix).to_quaternion().to_matrix()) @ local_normals[hand])
            palms.append(frame_palms)
            normals.append(frame_normals)
            torsos.append(np.asarray(world @ bones["上半身2"].head).copy())
            chunks = [[], []]
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for mesh, indices in selections:
                obj = mesh.evaluated_get(depsgraph)
                data = obj.to_mesh()
                try:
                    for hand in range(2):
                        chunks[hand].extend(tuple(obj.matrix_world @ data.vertices[int(i)].co) for i in indices[hand])
                finally:
                    obj.to_mesh_clear()
            for hand in range(2):
                hand_clouds[hand].append(chunks[hand])
            source_basis.append(np.asarray([np.asarray(b.matrix_basis) for b in bones]).copy())
            source_root.append(np.asarray(world).copy())
        fit = fit_supported_box(np.asarray(palms), np.asarray(normals),
                                tuple(np.asarray(v) for v in hand_clouds), np.asarray(torsos),
                                size=(side, depth, height))
        obj = build_cardboard_box(bpy, style, side=side, depth=depth, height=height)
        coords = np.asarray([tuple(v.co) for v in obj.data.vertices])
        low, high = coords.min(axis=0), coords.max(axis=0)
        coords = (coords - (low + high) / 2) * (np.asarray(fit.size) / (high - low))
        for vertex, coordinate in zip(obj.data.vertices, coords, strict=True):
            vertex.co = Vector(coordinate)
        obj.scale = (1., 1., 1.)
        obj.rotation_mode = "QUATERNION"
        previous = None
        for i, frame in enumerate(range(frame_start, frame_start + frames)):
            quaternion = Matrix(fit.rotations[i]).to_quaternion()
            if previous is not None and quaternion.dot(previous) < 0:
                quaternion.negate()
            obj.location = Vector(fit.centres[i])
            obj.rotation_quaternion = quaternion
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_quaternion", frame=frame)
            previous = quaternion.copy()
        _linearize(obj)
        report = dict(fit.report)
        for key in ("support_gap_over_10mm", "palm_misalignment_over_45deg", "palm_back_facing_box"):
            report["frames_" + key] = [frame_start + i for i in report["frame_offsets_" + key]]
        maximum_pose_change, maximum_root_change = 0., 0.
        for i, frame in enumerate(range(frame_start, frame_start + frames)):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            maximum_pose_change = max(maximum_pose_change, float(np.max(np.abs(
                np.asarray([np.asarray(b.matrix_basis) for b in bones]) - source_basis[i]))))
            maximum_root_change = max(maximum_root_change, float(np.max(np.abs(np.asarray(armature.matrix_world) - source_root[i]))))
            report["frames"][i]["frame"] = frame
            report["frames"][i]["support_bones"] = [labels[h][j] for h, j in enumerate(report["frames"][i]["support_vertex_indices"])]
        report.update(character_pose_matrix_max_change=maximum_pose_change,
                      root_matrix_max_change=maximum_root_change,
                      source_finger_bones_driven=int(armature.get("motionviewer_soma_fingers", 0)),
                      artificial_finger_pose=False, evaluation_stage="pre_physics")
        report["actor_box_clearance"] = audit_animated_box_clearance(
            bpy, obj, mesh_objects, size=fit.size, frame_start=frame_start, frames=frames,
            evaluation_stage="box_placement_on_existing_pose")
        return SceneBuild(props=[Prop(obj=obj)], report=report,
                          notes=["Fixed box above unchanged source hands; inspect support gaps and palm mismatch"])
    finally:
        scene.frame_set(original_frame)
        bpy.context.view_layer.update()
